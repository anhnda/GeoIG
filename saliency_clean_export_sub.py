"""
Saliency Map Export Script - SUBSET VERSION

Export saliency maps for a SUBSET of ImageNet classes with random sampling.

Key differences from salience_clean_export.py:
- Only export first N classes (default: 20)
- Random sampling S samples per class (default: 100)
- Faster for prototyping and testing

Usage:
    # Default: First 20 classes, 100 samples each
    python saliency_clean_export_sub.py

    # Custom: First 50 classes, 50 samples each
    python saliency_clean_export_sub.py --num_classes 50 --samples_per_class 50

    # With Expected Gradients
    python saliency_clean_export_sub.py --num_classes 20 --use_mean
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
from PIL import Image
import io
import pandas as pd
from collections import defaultdict
import random
import sys
import time

sys.path.append('.')
from full_classes import IMAGENET2012_CLASSES

# Import the optimized IG implementation from the full version
from salience_clean_export import IntegratedGradientsExpected

# ==========================================
# Configuration
# ==========================================

IMAGENET_RAW_DIR = Path("/data/imagenet_raw/data")
IMAGENET_SAMPLED_DIR = Path("/data/imagenet1k_sampled_sub")

# ==========================================
# Subset Dataset Sampler
# ==========================================

class ImageNet1kSubsetDataset:
    """Dataset that samples a SUBSET of ImageNet classes with random sampling."""

    def __init__(self, raw_dir: Path, sampled_dir: Path, output_dir: Path,
                 num_classes: int = 20, samples_per_class: int = 100,
                 force_resample: bool = False):
        self.raw_dir = raw_dir
        self.sampled_dir = sampled_dir
        self.output_dir = output_dir
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class

        # Create directories
        self.sampled_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_path = self.sampled_dir / f"metadata_c{num_classes}_s{samples_per_class}.pkl"

        # Load or create sampled dataset
        if self.metadata_path.exists() and not force_resample:
            print(f"\n{'='*80}")
            print(f"Loading cached subset dataset from {self.sampled_dir}")
            print(f"{'='*80}")
            self.load_cached_dataset()
        else:
            print(f"\n{'='*80}")
            print(f"Creating new subset dataset...")
            print(f"{'='*80}")
            self.create_sampled_dataset()

    def create_sampled_dataset(self):
        """Sample images from the first N classes."""
        print(f"Subset sampling configuration:")
        print(f"  Number of classes: {self.num_classes} (first {self.num_classes} classes)")
        print(f"  Samples per class: {self.samples_per_class}")
        print(f"  Total target images: {self.num_classes * self.samples_per_class}")

        self.wnid_to_idx = {wnid: idx for idx, wnid in enumerate(IMAGENET2012_CLASSES.keys())}
        self.idx_to_wnid = {idx: wnid for wnid, idx in self.wnid_to_idx.items()}

        # Only consider first N classes
        target_classes = list(range(self.num_classes))
        class_samples = {cls_idx: [] for cls_idx in target_classes}

        train_parquet_files = sorted(self.raw_dir.glob("train-*.parquet"))

        if len(train_parquet_files) == 0:
            raise FileNotFoundError(f"No train parquet files found in {self.raw_dir}")

        print(f"Found {len(train_parquet_files)} train parquet files")
        print(f"Target classes: 0-{self.num_classes-1}")

        for parquet_file in tqdm(train_parquet_files, desc="Reading parquet files"):
            df = pd.read_parquet(parquet_file)

            for _, row in df.iterrows():
                label = row['label']

                # Only process first N classes
                if label not in target_classes:
                    continue

                if len(class_samples[label]) < self.samples_per_class:
                    image_bytes = row['image']['bytes']
                    class_samples[label].append((image_bytes, label))

            # Check if we have enough samples for all target classes
            min_samples = min(len(samples) for samples in class_samples.values())
            if min_samples >= self.samples_per_class:
                print(f"\nCollected {self.samples_per_class} samples for all {self.num_classes} classes!")
                break

        # Display sampling statistics
        print(f"\nSampling statistics:")
        for class_idx in target_classes[:10]:
            print(f"  Class {class_idx}: {len(class_samples[class_idx])} images")
        if self.num_classes > 10:
            print(f"  ...")

        # Randomly select samples from collected data
        self.samples = []
        for class_idx in target_classes:
            available = class_samples[class_idx]
            if len(available) >= self.samples_per_class:
                # Random sampling
                sampled = random.sample(available, self.samples_per_class)
                self.samples.extend(sampled)
            else:
                print(f"WARNING: Class {class_idx} has only {len(available)} samples")
                self.samples.extend(available)

        print(f"\nTotal sampled images: {len(self.samples)}")

        # Save metadata
        print(f"Saving subset dataset to {self.sampled_dir}...")
        joblib.dump({
            'samples': self.samples,
            'num_classes': self.num_classes,
            'samples_per_class': self.samples_per_class,
            'wnid_to_idx': self.wnid_to_idx,
            'idx_to_wnid': self.idx_to_wnid,
            'target_classes': target_classes
        }, self.metadata_path)
        print(f"✓ Subset dataset cached!")

    def load_cached_dataset(self):
        """Load cached subset dataset."""
        metadata = joblib.load(self.metadata_path)
        self.samples = metadata['samples']
        self.wnid_to_idx = metadata['wnid_to_idx']
        self.idx_to_wnid = metadata['idx_to_wnid']

        print(f"Loaded {len(self.samples)} images from cache")
        print(f"  Number of classes: {metadata['num_classes']}")
        print(f"  Samples per class: {metadata['samples_per_class']}")

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
# Saliency Map Generator (reuse from full version)
# ==========================================

class SaliencyMapGeneratorSubset:
    """Generate saliency maps for subset dataset."""

    def __init__(self, dataset, output_dir, device='cuda', batch_size=16,
                 ig_steps=50, use_mean=False, use_compile=False, use_amp=True,
                 parallel_baselines=True):
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.device = device
        self.batch_size = batch_size
        self.ig_steps = ig_steps
        self.use_mean = use_mean
        self.use_amp = use_amp and torch.cuda.is_available()
        self.parallel_baselines = parallel_baselines

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
        self.ig = IntegratedGradientsExpected(self.model, device=device, use_compile=use_compile)

        print(f"✓ Model loaded and ready!")
        print(f"  Method: {'Expected Gradients (5 baselines)' if use_mean else 'Integrated Gradients (background)'}")

    def generate_all_saliency_maps(self, save_frequency=100):
        """Generate saliency maps for subset (only correct predictions)."""
        print(f"\n{'='*80}")
        print(f"Generating Saliency Maps for Subset")
        print(f"{'='*80}")
        print(f"Total images: {len(self.dataset)}")
        print(f"Filter: ONLY CORRECT PREDICTIONS")

        saliency_data = []
        total_processed = 0
        correct_count = 0
        incorrect_count = 0
        total_time = 0.0
        num_timed_samples = 0

        for idx in tqdm(range(len(self.dataset)), desc="Generating saliency maps"):
            image_pil, true_label = self.dataset[idx]

            # Transform
            image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)

            # Predict
            pred_class, confidence = self.ig.predict(image_tensor)

            total_processed += 1
            is_correct = (true_label == pred_class)

            if not is_correct:
                incorrect_count += 1
                continue

            correct_count += 1

            # Compute saliency
            start_time = time.time()

            with torch.enable_grad():
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    if self.use_mean:
                        saliency_map = self.ig.compute_expected_gradients(
                            image_tensor, target_class=pred_class,
                            steps=self.ig_steps, parallel_baselines=self.parallel_baselines
                        )
                    else:
                        saliency_map = self.ig.compute_integrated_gradients(
                            image_tensor, target_class=pred_class,
                            baseline_type='background', steps=self.ig_steps
                        )

            if correct_count > 3:
                elapsed = time.time() - start_time
                total_time += elapsed
                num_timed_samples += 1

            # Convert to numpy
            saliency_np = saliency_map.squeeze(0).cpu().numpy()

            # RGB image
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
            image_denorm = image_tensor.squeeze(0) * std + mean
            image_denorm = torch.clamp(image_denorm, 0, 1)
            rgb_image_np = image_denorm.cpu().numpy()

            # Store
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

            # Save periodically
            if len(saliency_data) >= save_frequency or (idx + 1) == len(self.dataset):
                if len(saliency_data) > 0:
                    self._save_batch(saliency_data, start_idx=correct_count - len(saliency_data))
                    saliency_data = []

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        print(f"\n✓ Subset saliency maps generated!")
        print(f"  Total processed: {total_processed}")
        print(f"  Correct predictions saved: {correct_count}")
        print(f"  Incorrect predictions skipped: {incorrect_count}")
        print(f"  Accuracy: {correct_count / total_processed * 100:.2f}%")

        if num_timed_samples > 0:
            avg_time = total_time / num_timed_samples
            print(f"\n  Average time per image: {avg_time:.3f}s")
            print(f"  Throughput: {1.0/avg_time:.2f} images/sec")

        self._create_metadata()

    def _save_batch(self, batch_data, start_idx):
        """Save batch to disk."""
        batch_id = start_idx // 100
        batch_path = self.saliency_dir / f"batch_{batch_id:04d}.pkl"
        joblib.dump(batch_data, batch_path, compress=3)

    def _create_metadata(self):
        """Create metadata file."""
        batch_files = sorted(self.saliency_dir.glob("batch_*.pkl"))

        total_samples = 0
        correct_predictions = 0
        class_counts = defaultdict(int)
        class_correct = defaultdict(int)

        for batch_file in tqdm(batch_files, desc="Computing stats"):
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
            'num_classes': self.dataset.num_classes,
            'samples_per_class': self.dataset.samples_per_class,
            'ig_steps': self.ig_steps,
            'model': 'resnet50',
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'class_counts': dict(class_counts),
            'class_correct': dict(class_correct),
            'saliency_map_shape': (224, 224),
            'rgb_image_shape': (3, 224, 224),
            'num_batches': len(batch_files),
            'filter_correct_only': True,
            'includes_rgb_images': True,
            'method': 'expected_gradients' if self.use_mean else 'integrated_gradients',
            'baseline': 'multi' if self.use_mean else 'background',
            'note': f"SUBSET: First {self.dataset.num_classes} classes, {self.dataset.samples_per_class} samples each"
        }

        joblib.dump(metadata, self.output_dir / "metadata.pkl")

        print(f"\n{'='*80}")
        print(f"Subset Export Summary")
        print(f"{'='*80}")
        print(f"Classes: {self.dataset.num_classes}")
        print(f"Samples per class: {self.dataset.samples_per_class}")
        print(f"Total samples saved: {total_samples}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Output: {self.output_dir}")
        print(f"{'='*80}")


# ==========================================
# Main Script
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate saliency maps for ImageNet SUBSET (fast prototyping)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
SUBSET VERSION - For Fast Prototyping:
  --num_classes N        Export first N classes only (default: 20)
  --samples_per_class S  Random sample S images per class (default: 100)

EXAMPLES:
  # Default: First 20 classes, 100 samples each
  python saliency_clean_export_sub.py

  # Custom subset
  python saliency_clean_export_sub.py --num_classes 50 --samples_per_class 50

  # With Expected Gradients
  python saliency_clean_export_sub.py --num_classes 20 --use_mean
        """
    )
    parser.add_argument('--num_classes', type=int, default=20,
                       help='Number of classes to export (first N classes, default: 20)')
    parser.add_argument('--samples_per_class', type=int, default=100,
                       help='Number of samples per class (random sampling, default: 100)')
    parser.add_argument('--force_resample', action='store_true',
                       help='Force resampling of dataset')
    parser.add_argument('--ig_steps', type=int, default=50,
                       help='Number of IG steps (default: 50)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (auto-generated if not specified)')
    parser.add_argument('--use_mean', action='store_true',
                       help='Use Expected Gradients (5 baselines)')
    parser.add_argument('--use_compile', action='store_true',
                       help='Enable torch.compile')
    parser.add_argument('--use_amp', dest='use_amp', action='store_true', default=True,
                       help='Enable mixed precision (default: ON)')
    parser.add_argument('--no_amp', dest='use_amp', action='store_false',
                       help='Disable mixed precision')
    parser.add_argument('--parallel_baselines', dest='parallel_baselines', action='store_true', default=True,
                       help='Parallel baselines (default: ON)')
    parser.add_argument('--no_parallel_baselines', dest='parallel_baselines', action='store_false',
                       help='Sequential baselines')

    args = parser.parse_args()

    # Auto-generate output directory name
    if args.output_dir is None:
        args.output_dir = f"./data/saliency_imagenet_sub_c{args.num_classes}_s{args.samples_per_class}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"SUBSET Saliency Map Export")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Classes: First {args.num_classes} classes")
    print(f"Samples per class: {args.samples_per_class}")
    print(f"Method: {'Expected Gradients' if args.use_mean else 'Integrated Gradients'}")

    # Create subset dataset
    dataset = ImageNet1kSubsetDataset(
        raw_dir=IMAGENET_RAW_DIR,
        sampled_dir=IMAGENET_SAMPLED_DIR,
        output_dir=Path(args.output_dir),
        num_classes=args.num_classes,
        samples_per_class=args.samples_per_class,
        force_resample=args.force_resample
    )

    # Generate saliency maps
    generator = SaliencyMapGeneratorSubset(
        dataset=dataset,
        output_dir=args.output_dir,
        device=device,
        ig_steps=args.ig_steps,
        use_mean=args.use_mean,
        use_compile=args.use_compile,
        use_amp=args.use_amp,
        parallel_baselines=args.parallel_baselines
    )

    generator.generate_all_saliency_maps()

    print(f"\n{'='*80}")
    print(f"✓ Subset export complete!")
    print(f"  Classes: {args.num_classes}")
    print(f"  Samples per class: {args.samples_per_class}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
