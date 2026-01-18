"""
Saliency Map Export Script for ImageNet-1k using Integrated Gradients

This script generates Integrated Gradients (IG) saliency maps for ResNet50 on ImageNet-1k.
It samples ~100 images per class (1000 classes total) and stores the results in a structured format.

Usage:
    python saliency_map_export.py --images_per_class 100
    python saliency_map_export.py --force_resample
    python saliency_map_export.py --batch_size 16
"""

import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import joblib
import argparse
from PIL import Image
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
# Integrated Gradients Implementation
# ==========================================

class IntegratedGradients:
    """
    Integrated Gradients attribution method for neural networks.

    Reference: Sundararajan et al. "Axiomatic Attribution for Deep Networks" (ICML 2017)
    """

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()

    def generate_baseline(self, image, baseline_type='zeros'):
        """Generate baseline image for IG computation."""
        if baseline_type == 'zeros':
            return torch.zeros_like(image)
        elif baseline_type == 'gaussian':
            return torch.randn_like(image) * 0.1
        elif baseline_type == 'mean':
            # ImageNet mean in normalized space
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(image.device)
            return mean.expand_as(image)
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

    def compute_integrated_gradients(self, image, target_class, baseline_type='zeros', steps=50):
        """
        Compute Integrated Gradients attribution map.

        Args:
            image: Input image tensor [1, 3, H, W]
            target_class: Target class index
            baseline_type: Type of baseline ('zeros', 'gaussian', 'mean')
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

    @torch.no_grad()
    def predict(self, image):
        """Get model prediction for an image."""
        output = self.model(image)
        probabilities = F.softmax(output, dim=1)
        pred_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, pred_class].item()
        return pred_class, confidence


# ==========================================
# Dataset Sampler (Adapted from ref_img1k.py)
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

            for idx, row in df.iterrows():
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
# Saliency Map Generator
# ==========================================

class SaliencyMapGenerator:
    """Generate and save Integrated Gradients saliency maps."""

    def __init__(self, dataset, output_dir, device='cuda', batch_size=16, ig_steps=50):
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.device = device
        self.batch_size = batch_size
        self.ig_steps = ig_steps

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

        # Initialize IG
        self.ig = IntegratedGradients(self.model, device=device)

        print(f"✓ Model loaded and ready!")

    def generate_all_saliency_maps(self, save_frequency=100):
        """Generate IG saliency maps for all images in the dataset."""
        print(f"\n{'='*80}")
        print(f"Generating Integrated Gradients Saliency Maps")
        print(f"{'='*80}")
        print(f"Total images: {len(self.dataset)}")
        print(f"IG steps: {self.ig_steps}")
        print(f"Batch size: {self.batch_size}")
        print(f"Output directory: {self.output_dir}")

        # Storage for saliency maps
        saliency_data = []

        # Process images
        for idx in tqdm(range(len(self.dataset)), desc="Generating saliency maps"):
            image_pil, true_label = self.dataset[idx]

            # Transform image
            image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)

            # Get model prediction
            pred_class, confidence = self.ig.predict(image_tensor)

            # Compute IG saliency map (using predicted class)
            with torch.enable_grad():
                saliency_map = self.ig.compute_integrated_gradients(
                    image_tensor,
                    target_class=pred_class,
                    baseline_type='zeros',
                    steps=self.ig_steps
                )

            # Convert to numpy
            saliency_np = saliency_map.squeeze(0).cpu().numpy()  # [224, 224]

            # Store metadata
            saliency_data.append({
                'index': idx,
                'true_label': true_label,
                'pred_label': pred_class,
                'confidence': confidence,
                'saliency_map': saliency_np,
                'correct': (true_label == pred_class)
            })

            # Periodically save to disk to avoid memory overflow
            if (idx + 1) % save_frequency == 0 or (idx + 1) == len(self.dataset):
                self._save_batch(saliency_data, start_idx=idx - len(saliency_data) + 1)
                saliency_data = []  # Clear memory

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print(f"\n✓ All saliency maps generated and saved!")

        # Create final metadata file
        self._create_metadata()

    def _save_batch(self, batch_data, start_idx):
        """Save a batch of saliency maps to disk."""
        batch_id = start_idx // 100
        batch_path = self.saliency_dir / f"batch_{batch_id:04d}.pkl"

        joblib.dump(batch_data, batch_path, compress=3)

    def _create_metadata(self):
        """Create comprehensive metadata file."""
        # Load all batches to compute statistics
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

        # Compute overall accuracy
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0

        # Create metadata
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
            'num_batches': len(batch_files)
        }

        # Save metadata
        metadata_path = self.output_dir / "metadata.pkl"
        joblib.dump(metadata, metadata_path)

        # Print summary
        print(f"\n{'='*80}")
        print(f"Saliency Map Generation Summary")
        print(f"{'='*80}")
        print(f"Total samples: {total_samples}")
        print(f"Model accuracy: {accuracy * 100:.2f}%")
        print(f"Correct predictions: {correct_predictions}/{total_samples}")
        print(f"Output directory: {self.output_dir}")
        print(f"Number of batches: {len(batch_files)}")

        # Calculate total size
        total_size = sum(f.stat().st_size for f in self.output_dir.rglob("*.pkl"))
        size_mb = total_size / (1024 * 1024)
        print(f"Total size: {size_mb:.1f} MB")
        print(f"{'='*80}")


# ==========================================
# Utility: Load Saliency Maps
# ==========================================

class SaliencyMapLoader:
    """Utility class to load saved saliency maps efficiently."""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.saliency_dir = self.data_dir / "saliency_maps"
        self.metadata_path = self.data_dir / "metadata.pkl"

        # Load metadata
        if self.metadata_path.exists():
            self.metadata = joblib.load(self.metadata_path)
            print(f"Loaded metadata: {self.metadata['total_samples']} samples")
        else:
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")

        # Index batch files
        self.batch_files = sorted(self.saliency_dir.glob("batch_*.pkl"))
        print(f"Found {len(self.batch_files)} batch files")

    def get_saliency_by_class(self, class_id, max_samples=None):
        """Load all saliency maps for a specific class."""
        saliency_maps = []
        sample_count = 0

        for batch_file in self.batch_files:
            batch_data = joblib.load(batch_file)

            for item in batch_data:
                if item['true_label'] == class_id:
                    saliency_maps.append(item['saliency_map'])
                    sample_count += 1

                    if max_samples and sample_count >= max_samples:
                        return np.array(saliency_maps)

        return np.array(saliency_maps) if saliency_maps else None

    def get_all_saliency_maps(self):
        """Load all saliency maps (memory intensive!)."""
        all_maps = []
        all_labels = []

        for batch_file in tqdm(self.batch_files, desc="Loading all batches"):
            batch_data = joblib.load(batch_file)

            for item in batch_data:
                all_maps.append(item['saliency_map'])
                all_labels.append(item['true_label'])

        return np.array(all_maps), np.array(all_labels)


# ==========================================
# Main Script
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate Integrated Gradients saliency maps for ImageNet-1k'
    )
    parser.add_argument('--images_per_class', type=int, default=100,
                       help='Number of images to sample per class (default: 100)')
    parser.add_argument('--force_resample', action='store_true',
                       help='Force resampling of dataset')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for processing (default: 16)')
    parser.add_argument('--ig_steps', type=int, default=10,
                       help='Number of integration steps for IG (default: 50)')
    parser.add_argument('--output_dir', type=str, default='./data/saliency_imagenet1k_resnet50_100',
                       help='Output directory for saliency maps')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"Integrated Gradients Saliency Map Export")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Images per class: {args.images_per_class}")
    print(f"IG steps: {args.ig_steps}")

    # Create dataset
    dataset = ImageNet1kSaliencyDataset(
        raw_dir=IMAGENET_RAW_DIR,
        sampled_dir=IMAGENET_SAMPLED_DIR,
        output_dir=Path(args.output_dir),
        images_per_class=args.images_per_class,
        force_resample=args.force_resample
    )

    # Generate saliency maps
    generator = SaliencyMapGenerator(
        dataset=dataset,
        output_dir=args.output_dir,
        device=device,
        batch_size=args.batch_size,
        ig_steps=args.ig_steps
    )

    generator.generate_all_saliency_maps()

    print(f"\n{'='*80}")
    print(f"✓ Saliency map generation complete!")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
