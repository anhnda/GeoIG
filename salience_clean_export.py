"""
Saliency Map Export Script with Expected Gradients Support

This script generates Integrated Gradients (IG) saliency maps for ResNet50 on ImageNet-1k
with support for Expected Gradients (averaging over multiple baselines).

Expected Gradients baselines:
1. Black (dark) image
2. White image
3. Gaussian noise (from zero)
4. Blur image
5. Background (mean of four corners, each corner 5%x5%)

Default baseline (--use_mean=False): Background reference

Usage:
    # Default: Background baseline only
    python salience_clean_export.py --images_per_class 100

    # Expected Gradients: Average over all baselines
    python salience_clean_export.py --images_per_class 100 --use_mean

    # With custom settings
    python salience_clean_export.py --use_mean --batch_size 16 --ig_steps 50
"""

import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
import numpy as np
from pathlib import Path
from tqdm import tqdm
import joblib
import argparse
from PIL import Image, ImageFilter
import io
import pandas as pd
from collections import defaultdict
import random
import sys

sys.path.append('.')
from full_classes import IMAGENET2012_CLASSES

# ==========================================
# Configuration
# ==========================================

IMAGENET_RAW_DIR = Path("/data/imagenet_raw/data")
IMAGENET_SAMPLED_DIR = Path("/data/imagenet1k_sampled")
SALIENCY_OUTPUT_DIR = Path("./data/saliency_imagenet1k_resnet50_100")

IMAGES_PER_CLASS = 100
NUM_CLASSES = 1000

# ==========================================
# Enhanced Integrated Gradients with Expected Gradients
# ==========================================

class IntegratedGradientsExpected:
    """
    Integrated Gradients with Expected Gradients support.

    Expected Gradients: Average IG attributions over multiple baselines to
    reduce sensitivity to baseline choice.

    Reference:
    - Sundararajan et al. "Axiomatic Attribution for Deep Networks" (ICML 2017)
    - Erion et al. "Improving Performance of Deep Learning Models with Expected Gradients" (2020)
    """

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()

    def generate_baseline(self, image, baseline_type='background'):
        """
        Generate baseline image for IG computation.

        Args:
            image: Input image tensor [1, 3, H, W]
            baseline_type: Type of baseline
                - 'black': Black image (zeros)
                - 'white': White image (ones in normalized space)
                - 'gaussian': Gaussian noise from zero mean
                - 'blur': Blurred version of the input image
                - 'background': Mean of 4 corners (5%x5% each)

        Returns:
            Baseline tensor [1, 3, H, W]
        """
        if baseline_type == 'black':
            # Black image (zeros)
            return torch.zeros_like(image)

        elif baseline_type == 'white':
            # White image in normalized ImageNet space
            # ImageNet normalization: (x - mean) / std
            # For white pixel (1.0, 1.0, 1.0), normalized value is:
            # (1.0 - mean) / std
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(image.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(image.device)
            white_normalized = (1.0 - mean) / std
            return white_normalized.expand_as(image)

        elif baseline_type == 'gaussian':
            # Gaussian noise with zero mean and small std
            return torch.randn_like(image) * 0.1

        elif baseline_type == 'blur':
            # Blurred version of the input (requires PIL conversion)
            # Denormalize -> blur -> normalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(image.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(image.device)

            # Denormalize to [0, 1]
            image_denorm = image * std + mean
            image_denorm = torch.clamp(image_denorm, 0, 1)

            # Apply strong Gaussian blur
            blurred = F.avg_pool2d(image_denorm, kernel_size=31, stride=1, padding=15)

            # Re-normalize
            blurred_normalized = (blurred - mean) / std
            return blurred_normalized

        elif baseline_type == 'background':
            # Background: Mean of 4 corners (5% x 5% each)
            _, _, H, W = image.shape
            corner_h = max(1, int(H * 0.05))
            corner_w = max(1, int(W * 0.05))

            # Extract 4 corners
            top_left = image[:, :, :corner_h, :corner_w]
            top_right = image[:, :, :corner_h, -corner_w:]
            bottom_left = image[:, :, -corner_h:, :corner_w]
            bottom_right = image[:, :, -corner_h:, -corner_w:]

            # Compute mean across all corners
            corners = torch.cat([
                top_left.flatten(2),
                top_right.flatten(2),
                bottom_left.flatten(2),
                bottom_right.flatten(2)
            ], dim=2)

            background_color = corners.mean(dim=2, keepdim=True).unsqueeze(-1)  # [1, 3, 1, 1]
            baseline = background_color.expand_as(image)

            return baseline

        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")

    def compute_gradients(self, image, target_class):
        """Compute gradients of target class score w.r.t. input image."""
        image.requires_grad = True

        # Forward pass
        output = self.model(image)

        # Zero out all gradients
        self.model.zero_grad()

        # Get score for target class
        target_score = output[0, target_class]

        # Backward pass
        target_score.backward()

        # Return gradients
        gradients = image.grad.detach()
        image.requires_grad = False

        return gradients

    def compute_integrated_gradients(self, image, target_class, baseline_type='background', steps=50):
        """
        Compute Integrated Gradients attribution map for a single baseline.

        Args:
            image: Input image tensor [1, 3, H, W]
            target_class: Target class index
            baseline_type: Type of baseline
            steps: Number of integration steps

        Returns:
            Attribution map [1, H, W] - aggregated across RGB channels
        """
        # Generate baseline
        baseline = self.generate_baseline(image, baseline_type)

        # Compute path interpolation
        alphas = torch.linspace(0, 1, steps + 1).to(self.device)

        # Accumulate gradients
        integrated_grads = torch.zeros_like(image)

        for i in range(steps):
            # Interpolate between baseline and image
            alpha = alphas[i]
            interpolated_image = baseline + alpha * (image - baseline)

            # Compute gradients at this point
            gradients = self.compute_gradients(interpolated_image, target_class)

            # Accumulate
            integrated_grads += gradients

        # Average the gradients
        integrated_grads = integrated_grads / steps

        # Multiply by (image - baseline) to get final attribution
        attribution = (image - baseline) * integrated_grads

        # Aggregate across RGB channels (sum of absolute values)
        attribution_map = torch.abs(attribution).sum(dim=1, keepdim=False)  # [1, H, W]

        # Normalize to [0, 1]
        attr_min = attribution_map.min()
        attr_max = attribution_map.max()
        if attr_max - attr_min > 1e-8:
            attribution_map = (attribution_map - attr_min) / (attr_max - attr_min)

        return attribution_map

    def compute_expected_gradients(self, image, target_class, steps=50):
        """
        Compute Expected Gradients: Average IG over multiple baselines.

        Baselines used:
        1. Black (dark) image
        2. White image
        3. Gaussian noise
        4. Blur image
        5. Background (corners)

        Args:
            image: Input image tensor [1, 3, H, W]
            target_class: Target class index
            steps: Number of integration steps per baseline

        Returns:
            Attribution map [1, H, W] - averaged across all baselines
        """
        baseline_types = ['black', 'white', 'gaussian', 'blur', 'background']

        attribution_maps = []

        for baseline_type in baseline_types:
            attr_map = self.compute_integrated_gradients(
                image, target_class, baseline_type=baseline_type, steps=steps
            )
            attribution_maps.append(attr_map)

        # Average across all baselines
        expected_attribution = torch.stack(attribution_maps).mean(dim=0)

        # Re-normalize to [0, 1]
        attr_min = expected_attribution.min()
        attr_max = expected_attribution.max()
        if attr_max - attr_min > 1e-8:
            expected_attribution = (expected_attribution - attr_min) / (attr_max - attr_min)

        return expected_attribution

    @torch.no_grad()
    def predict(self, image):
        """Get model prediction for an image."""
        output = self.model(image)
        probabilities = F.softmax(output, dim=1)
        pred_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, pred_class].item()
        return pred_class, confidence


# ==========================================
# Dataset Sampler (Same as saliency_map_export.py)
# ==========================================

class ImageNet1kSaliencyDataset:
    """Dataset that samples ImageNet-1k images and generates IG saliency maps."""

    def __init__(self, raw_dir: Path, sampled_dir: Path, output_dir: Path,
                 images_per_class: int = 100, force_resample: bool = False):
        self.raw_dir = raw_dir
        self.sampled_dir = sampled_dir
        self.output_dir = output_dir
        self.images_per_class = images_per_class

        # Create directories
        self.sampled_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_path = self.sampled_dir / f"metadata_{images_per_class}.pkl"

        # Load or create sampled dataset
        if self.metadata_path.exists() and not force_resample:
            print(f"\n{'='*80}")
            print(f"Loading cached sampled dataset from {self.sampled_dir}")
            print(f"{'='*80}")
            self.load_cached_dataset()
        else:
            print(f"\n{'='*80}")
            print(f"Creating new sampled dataset...")
            print(f"{'='*80}")
            self.create_sampled_dataset()

    def create_sampled_dataset(self):
        """Sample images from parquet files."""
        print(f"Sampling {self.images_per_class} images per class from {NUM_CLASSES} classes...")
        print(f"Total target images: {self.images_per_class * NUM_CLASSES}")

        self.wnid_to_idx = {wnid: idx for idx, wnid in enumerate(IMAGENET2012_CLASSES.keys())}
        self.idx_to_wnid = {idx: wnid for wnid, idx in self.wnid_to_idx.items()}

        class_samples = defaultdict(list)

        train_parquet_files = sorted(self.raw_dir.glob("train-*.parquet"))

        if len(train_parquet_files) == 0:
            raise FileNotFoundError(f"No train parquet files found in {self.raw_dir}")

        print(f"Found {len(train_parquet_files)} train parquet files")

        for parquet_file in tqdm(train_parquet_files, desc="Reading parquet files"):
            df = pd.read_parquet(parquet_file)

            for _, row in df.iterrows():
                label = row['label']

                if len(class_samples[label]) < self.images_per_class:
                    image_bytes = row['image']['bytes']
                    class_samples[label].append((image_bytes, label))

            # Check if we have enough samples
            min_samples = min(len(samples) for samples in class_samples.values()) if class_samples else 0
            if min_samples >= self.images_per_class and len(class_samples) == NUM_CLASSES:
                print(f"\nCollected {self.images_per_class} samples for all {NUM_CLASSES} classes!")
                break

        print(f"\nSampling complete. Samples per class:")
        for class_idx in range(min(10, NUM_CLASSES)):
            print(f"  Class {class_idx}: {len(class_samples[class_idx])} images")
        print(f"  ...")

        # Select samples
        self.samples = []
        for class_idx in range(NUM_CLASSES):
            if len(class_samples[class_idx]) >= self.images_per_class:
                sampled = random.sample(class_samples[class_idx], self.images_per_class)
                self.samples.extend(sampled)
            else:
                print(f"WARNING: Class {class_idx} has only {len(class_samples[class_idx])} samples")
                self.samples.extend(class_samples[class_idx])

        print(f"\nTotal sampled images: {len(self.samples)}")

        # Save metadata
        print(f"Saving sampled dataset to {self.sampled_dir}...")
        joblib.dump({
            'samples': self.samples,
            'images_per_class': self.images_per_class,
            'num_classes': NUM_CLASSES,
            'wnid_to_idx': self.wnid_to_idx,
            'idx_to_wnid': self.idx_to_wnid
        }, self.metadata_path)
        print(f"✓ Sampled dataset cached!")

    def load_cached_dataset(self):
        """Load cached sampled dataset."""
        metadata = joblib.load(self.metadata_path)
        self.samples = metadata['samples']
        self.wnid_to_idx = metadata['wnid_to_idx']
        self.idx_to_wnid = metadata['idx_to_wnid']

        print(f"Loaded {len(self.samples)} images from cache")
        print(f"  Images per class: {metadata['images_per_class']}")
        print(f"  Number of classes: {metadata['num_classes']}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get image and label."""
        image_bytes, label = self.samples[idx]
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return image, label

    def get_class_samples_info(self):
        """Get information about samples per class."""
        class_counts = defaultdict(int)
        for _, label in self.samples:
            class_counts[label] += 1
        return class_counts


# ==========================================
# Enhanced Saliency Map Generator
# ==========================================

class SaliencyMapGeneratorClean:
    """Generate and save IG/Expected Gradients saliency maps."""

    def __init__(self, dataset, output_dir, device='cuda', batch_size=16,
                 ig_steps=50, use_mean=False):
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.device = device
        self.batch_size = batch_size
        self.ig_steps = ig_steps
        self.use_mean = use_mean

        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.saliency_dir = self.output_dir / "saliency_maps"
        self.saliency_dir.mkdir(exist_ok=True)

        # Data transform
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load ResNet50
        print(f"\nLoading ResNet50 (pretrained on ImageNet-1k)...")
        self.model = models.resnet50(pretrained=True).to(device)
        self.model.eval()

        # Initialize IG with Expected Gradients support
        self.ig = IntegratedGradientsExpected(self.model, device=device)

        print(f"✓ Model loaded and ready!")
        if use_mean:
            print(f"✓ Expected Gradients enabled (averaging over 5 baselines)")
        else:
            print(f"✓ Single baseline: Background (corners)")

    def generate_all_saliency_maps(self, save_frequency=100):
        """Generate IG/Expected Gradients saliency maps for all images (only correct predictions)."""
        print(f"\n{'='*80}")
        print(f"Generating Saliency Maps")
        print(f"{'='*80}")
        print(f"Method: {'Expected Gradients (5 baselines)' if self.use_mean else 'Integrated Gradients (background baseline)'}")
        print(f"Total images: {len(self.dataset)}")
        print(f"IG steps: {self.ig_steps}")
        print(f"Batch size: {self.batch_size}")
        print(f"Output directory: {self.output_dir}")
        print(f"Filter: ONLY CORRECT PREDICTIONS")
        print(f"Includes: RGB images for edge-aware gating")

        # Storage for saliency maps
        saliency_data = []

        # Tracking statistics
        total_processed = 0
        correct_count = 0
        incorrect_count = 0

        # Process images
        for idx in tqdm(range(len(self.dataset)), desc="Generating saliency maps"):
            image_pil, true_label = self.dataset[idx]

            # Transform image
            image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)

            # Get model prediction
            pred_class, confidence = self.ig.predict(image_tensor)

            total_processed += 1
            is_correct = (true_label == pred_class)

            # ONLY process correct predictions
            if not is_correct:
                incorrect_count += 1
                continue

            correct_count += 1

            # Compute saliency map
            with torch.enable_grad():
                if self.use_mean:
                    # Expected Gradients: Average over multiple baselines
                    saliency_map = self.ig.compute_expected_gradients(
                        image_tensor,
                        target_class=pred_class,
                        steps=self.ig_steps
                    )
                else:
                    # Single baseline: Background (corners)
                    saliency_map = self.ig.compute_integrated_gradients(
                        image_tensor,
                        target_class=pred_class,
                        baseline_type='background',
                        steps=self.ig_steps
                    )

            # Convert to numpy
            saliency_np = saliency_map.squeeze(0).cpu().numpy()  # [224, 224]

            # Convert RGB image tensor to numpy for storage
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
            image_denorm = image_tensor.squeeze(0) * std + mean
            image_denorm = torch.clamp(image_denorm, 0, 1)
            rgb_image_np = image_denorm.cpu().numpy()  # [3, 224, 224]

            # Store metadata
            saliency_data.append({
                'index': idx,
                'sample_index': idx,
                'true_label': true_label,
                'pred_label': pred_class,
                'confidence': confidence,
                'saliency_map': saliency_np,
                'rgb_image': rgb_image_np,
                'correct': True,
                'method': 'expected_gradients' if self.use_mean else 'integrated_gradients',
                'baseline': 'multi' if self.use_mean else 'background'
            })

            # Periodically save to disk
            if len(saliency_data) >= save_frequency or (idx + 1) == len(self.dataset):
                if len(saliency_data) > 0:
                    self._save_batch(saliency_data, start_idx=correct_count - len(saliency_data))
                    saliency_data = []

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        print(f"\n✓ All saliency maps generated and saved!")
        print(f"  Total processed: {total_processed}")
        print(f"  Correct predictions saved: {correct_count}")
        print(f"  Incorrect predictions skipped: {incorrect_count}")
        print(f"  Accuracy: {correct_count / total_processed * 100:.2f}%")

        # Create final metadata file
        self._create_metadata()

    def _save_batch(self, batch_data, start_idx):
        """Save a batch of saliency maps to disk."""
        batch_id = start_idx // 100
        batch_path = self.saliency_dir / f"batch_{batch_id:04d}.pkl"
        joblib.dump(batch_data, batch_path, compress=3)

    def _create_metadata(self):
        """Create comprehensive metadata file."""
        batch_files = sorted(self.saliency_dir.glob("batch_*.pkl"))

        total_samples = 0
        correct_predictions = 0
        class_counts = defaultdict(int)
        class_correct = defaultdict(int)

        print(f"\nComputing statistics from {len(batch_files)} batches...")

        for batch_file in tqdm(batch_files, desc="Processing batches"):
            batch_data = joblib.load(batch_file)

            for item in batch_data:
                total_samples += 1
                true_label = item['true_label']
                class_counts[true_label] += 1

                if item['correct']:
                    correct_predictions += 1
                    class_correct[true_label] += 1

        accuracy = correct_predictions / total_samples if total_samples > 0 else 0

        metadata = {
            'total_samples': total_samples,
            'num_classes': NUM_CLASSES,
            'images_per_class': self.dataset.images_per_class,
            'ig_steps': self.ig_steps,
            'model': 'resnet50',
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'class_counts': dict(class_counts),
            'class_correct': dict(class_correct),
            'saliency_map_shape': (224, 224),
            'rgb_image_shape': (3, 224, 224),
            'rgb_image_format': 'float32',
            'num_batches': len(batch_files),
            'filter_correct_only': True,
            'includes_rgb_images': True,
            'method': 'expected_gradients' if self.use_mean else 'integrated_gradients',
            'baseline': 'multi (black, white, gaussian, blur, background)' if self.use_mean else 'background (corners)',
            'note': f"Saliency maps generated using {'Expected Gradients (average over 5 baselines)' if self.use_mean else 'Integrated Gradients (background baseline)'}. Only correct predictions saved."
        }

        metadata_path = self.output_dir / "metadata.pkl"
        joblib.dump(metadata, metadata_path)

        print(f"\n{'='*80}")
        print(f"Saliency Map Generation Summary")
        print(f"{'='*80}")
        print(f"Method: {metadata['method']}")
        print(f"Baseline: {metadata['baseline']}")
        print(f"Total samples saved: {total_samples} (CORRECT predictions only)")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Classes with samples: {len(class_counts)}")
        print(f"Output directory: {self.output_dir}")
        print(f"Number of batches: {len(batch_files)}")

        total_size = sum(f.stat().st_size for f in self.output_dir.rglob("*.pkl"))
        size_mb = total_size / (1024 * 1024)
        print(f"Total size: {size_mb:.1f} MB")
        print(f"{'='*80}")


# ==========================================
# Main Script
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate IG/Expected Gradients saliency maps for ImageNet-1k'
    )
    parser.add_argument('--images_per_class', type=int, default=100,
                       help='Number of images to sample per class (default: 100)')
    parser.add_argument('--force_resample', action='store_true',
                       help='Force resampling of dataset')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for processing (default: 16)')
    parser.add_argument('--ig_steps', type=int, default=50,
                       help='Number of integration steps for IG (default: 50)')
    parser.add_argument('--output_dir', type=str, default='./data/saliency_imagenet1k_resnet50_100',
                       help='Output directory for saliency maps')
    parser.add_argument('--use_mean', action='store_true',
                       help='Use Expected Gradients (average over multiple baselines: black, white, gaussian, blur, background)')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"Saliency Map Export with Expected Gradients Support")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Images per class: {args.images_per_class}")
    print(f"IG steps: {args.ig_steps}")
    print(f"Method: {'Expected Gradients' if args.use_mean else 'Integrated Gradients'}")
    print(f"Baseline: {'Multi (5 baselines)' if args.use_mean else 'Background (corners)'}")

    # Create dataset
    dataset = ImageNet1kSaliencyDataset(
        raw_dir=IMAGENET_RAW_DIR,
        sampled_dir=IMAGENET_SAMPLED_DIR,
        output_dir=Path(args.output_dir),
        images_per_class=args.images_per_class,
        force_resample=args.force_resample
    )

    # Generate saliency maps
    generator = SaliencyMapGeneratorClean(
        dataset=dataset,
        output_dir=args.output_dir,
        device=device,
        batch_size=args.batch_size,
        ig_steps=args.ig_steps,
        use_mean=args.use_mean
    )

    generator.generate_all_saliency_maps()

    print(f"\n{'='*80}")
    print(f"✓ Saliency map generation complete!")
    print(f"  Method: {'Expected Gradients' if args.use_mean else 'Integrated Gradients'}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
