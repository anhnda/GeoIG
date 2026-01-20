#!/usr/bin/env python
"""
Quick script to check a single sample from the dataset.
Run this to debug shape issues.

Usage:
    python check_single_sample.py --data_dir ./data/saliency_imagenet1k_resnet50_100 --idx 0
"""

import sys
import argparse
from pathlib import Path

# Import the dataset class
from geo_v2 import SaliencyMapDataset

def check_sample(data_dir, idx=0, use_edge_gating=True):
    """Check a single sample."""
    print(f"Checking sample {idx} from {data_dir}\n")

    # Create dataset
    print("Creating dataset...")
    dataset = SaliencyMapDataset(
        data_dir=data_dir,
        max_samples_per_class=None,
        load_images=use_edge_gating,
        cache_size=2
    )

    print(f"Dataset size: {len(dataset)} samples\n")

    if idx >= len(dataset):
        print(f"ERROR: Index {idx} out of range (dataset has {len(dataset)} samples)")
        return

    # Load sample
    print(f"Loading sample {idx}...")
    try:
        saliency, label, image = dataset[idx]

        print("✅ Sample loaded successfully!\n")
        print(f"Saliency shape: {saliency.shape}")
        print(f"Label: {label}")
        if image is not None:
            print(f"Image shape: {image.shape}")
        else:
            print("Image: None (edge gating disabled)")

        # Validate shapes
        print("\nValidating shapes...")
        if saliency.shape[0] != 1:
            print(f"❌ ERROR: Saliency has {saliency.shape[0]} channels, expected 1")
        elif saliency.shape[1:] != (224, 224):
            print(f"❌ ERROR: Saliency has spatial size {saliency.shape[1:]}, expected (224, 224)")
        else:
            print("✅ Saliency shape is correct: [1, 224, 224]")

        if image is not None:
            if image.shape[0] != 3:
                print(f"❌ ERROR: Image has {image.shape[0]} channels, expected 3")
            elif image.shape[1:] != (224, 224):
                print(f"❌ ERROR: Image has spatial size {image.shape[1:]}, expected (224, 224)")
            else:
                print("✅ Image shape is correct: [3, 224, 224]")

    except Exception as e:
        print(f"❌ ERROR loading sample: {e}")
        import traceback
        traceback.print_exc()
        return

    # Try loading a batch
    print(f"\n{'='*60}")
    print("Testing batch loading with 4 samples...")
    from torch.utils.data import DataLoader
    from geo_v2 import safe_collate_fn

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=safe_collate_fn
    )

    try:
        batch = next(iter(loader))
        saliency_batch, label_batch, image_batch = batch

        print("✅ Batch loaded successfully!\n")
        print(f"Saliency batch shape: {saliency_batch.shape}")
        print(f"Label batch shape: {label_batch.shape}")
        if image_batch is not None:
            print(f"Image batch shape: {image_batch.shape}")

        if saliency_batch.shape[1] == 0:
            print("\n❌ ERROR: Saliency batch has 0 channels!")
            print("This is the bug causing [B, 0, 224, 224] error")
        else:
            print(f"\n✅ Batch shapes look good!")

    except Exception as e:
        print(f"❌ ERROR creating batch: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check a single dataset sample")
    parser.add_argument('--data_dir', type=str,
                       default='./data/saliency_imagenet1k_resnet50_100',
                       help='Data directory')
    parser.add_argument('--idx', type=int, default=0,
                       help='Sample index to check')
    parser.add_argument('--no_edge_gating', action='store_true',
                       help='Disable edge gating (don\'t load RGB images)')

    args = parser.parse_args()

    if not Path(args.data_dir).exists():
        print(f"ERROR: Data directory not found: {args.data_dir}")
        sys.exit(1)

    check_sample(args.data_dir, args.idx, use_edge_gating=not args.no_edge_gating)
