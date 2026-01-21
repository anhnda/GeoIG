#!/usr/bin/env python3
"""
Diagnostic script to check data integrity in batch files.
"""

import joblib
import numpy as np
import torch
from pathlib import Path
import argparse
from collections import defaultdict


def diagnose_batch_file(batch_file):
    """Diagnose a single batch file."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {batch_file.name}")
    print(f"{'='*80}")

    try:
        batch_data = joblib.load(batch_file)
        print(f"✓ Loaded successfully, contains {len(batch_data)} items")
    except Exception as e:
        print(f"❌ FAILED to load batch file: {e}")
        return False

    issues = []

    for idx, item in enumerate(batch_data):
        # Check keys
        if 'saliency_map' not in item:
            issues.append(f"Item {idx}: Missing 'saliency_map' key")
            continue

        # Check saliency map
        sal = item['saliency_map']
        if not isinstance(sal, np.ndarray):
            issues.append(f"Item {idx}: saliency_map is {type(sal)}, not ndarray")
            continue

        if sal.shape != (224, 224) and sal.shape != (1, 224, 224):
            issues.append(f"Item {idx}: Invalid saliency shape {sal.shape}")

        if np.any(np.isnan(sal)):
            issues.append(f"Item {idx}: saliency_map contains NaN")

        if np.any(np.isinf(sal)):
            issues.append(f"Item {idx}: saliency_map contains Inf")

        # Check RGB image if present
        if 'rgb_image' in item:
            rgb = item['rgb_image']
            if not isinstance(rgb, np.ndarray):
                issues.append(f"Item {idx}: rgb_image is {type(rgb)}, not ndarray")
            elif rgb.shape not in [(3, 224, 224), (224, 224, 3)]:
                issues.append(f"Item {idx}: Invalid RGB shape {rgb.shape}")
            elif np.any(np.isnan(rgb)):
                issues.append(f"Item {idx}: rgb_image contains NaN")
            elif np.any(np.isinf(rgb)):
                issues.append(f"Item {idx}: rgb_image contains Inf")

        # Check label
        if 'true_label' not in item:
            issues.append(f"Item {idx}: Missing 'true_label' key")

    if issues:
        print(f"\n❌ Found {len(issues)} issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"   {issue}")
        if len(issues) > 10:
            print(f"   ... and {len(issues) - 10} more issues")
        return False
    else:
        print(f"✓ All {len(batch_data)} items are valid")
        return True


def test_dataset_loading(data_dir, load_images=True):
    """Test the actual Dataset class loading."""
    print(f"\n{'='*80}")
    print(f"Testing Dataset Loading (load_images={load_images})")
    print(f"{'='*80}")

    # Import the dataset class
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from geo_v2 import SaliencyMapDataset

    try:
        dataset = SaliencyMapDataset(data_dir, load_images=load_images, cache_size=2)
        print(f"✓ Dataset created successfully with {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Failed to create dataset: {e}")
        return False

    # Test loading first few samples
    print(f"\nTesting sample loading...")
    issues = []

    for idx in range(min(10, len(dataset))):
        try:
            result = dataset[idx]

            if load_images:
                if len(result) != 3:
                    issues.append(f"Sample {idx}: Expected 3 items, got {len(result)}")
                    continue
                saliency, label, image = result

                # Check saliency
                if not isinstance(saliency, torch.Tensor):
                    issues.append(f"Sample {idx}: saliency is {type(saliency)}, not Tensor")
                elif saliency.shape != (1, 224, 224):
                    issues.append(f"Sample {idx}: saliency shape is {saliency.shape}, expected (1, 224, 224)")
                elif saliency.shape[0] == 0:
                    issues.append(f"Sample {idx}: ❌ SALIENCY HAS 0 CHANNELS! Shape: {saliency.shape}")

                # Check image
                if image is not None:
                    if not isinstance(image, torch.Tensor):
                        issues.append(f"Sample {idx}: image is {type(image)}, not Tensor")
                    elif image.shape != (3, 224, 224):
                        issues.append(f"Sample {idx}: image shape is {image.shape}, expected (3, 224, 224)")
                    elif image.shape[0] == 0:
                        issues.append(f"Sample {idx}: ❌ IMAGE HAS 0 CHANNELS! Shape: {image.shape}")
            else:
                if len(result) != 2:
                    issues.append(f"Sample {idx}: Expected 2 items, got {len(result)}")
                    continue
                saliency, label = result

                if not isinstance(saliency, torch.Tensor):
                    issues.append(f"Sample {idx}: saliency is {type(saliency)}, not Tensor")
                elif saliency.shape != (1, 224, 224):
                    issues.append(f"Sample {idx}: saliency shape is {saliency.shape}, expected (1, 224, 224)")
                elif saliency.shape[0] == 0:
                    issues.append(f"Sample {idx}: ❌ SALIENCY HAS 0 CHANNELS! Shape: {saliency.shape}")

        except Exception as e:
            issues.append(f"Sample {idx}: Exception during loading: {e}")

    if issues:
        print(f"\n❌ Found {len(issues)} issues during loading:")
        for issue in issues:
            print(f"   {issue}")
        return False
    else:
        print(f"✓ All tested samples loaded correctly")
        return True


def test_dataloader_collation(data_dir, load_images=True):
    """Test DataLoader with collate function."""
    print(f"\n{'='*80}")
    print(f"Testing DataLoader Collation (load_images={load_images})")
    print(f"{'='*80}")

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from geo_v2 import SaliencyMapDataset, safe_collate_fn
    from torch.utils.data import DataLoader

    try:
        dataset = SaliencyMapDataset(data_dir, load_images=load_images, cache_size=2)
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,  # Use 0 for debugging
            collate_fn=safe_collate_fn
        )
        print(f"✓ DataLoader created successfully")
    except Exception as e:
        print(f"❌ Failed to create DataLoader: {e}")
        return False

    # Test first batch
    print(f"\nTesting first batch...")
    try:
        batch = next(iter(dataloader))

        if load_images:
            saliency_batch, label_batch, image_batch = batch
            print(f"✓ Batch loaded successfully:")
            print(f"   saliency_batch.shape = {saliency_batch.shape}")
            print(f"   label_batch.shape = {label_batch.shape}")
            print(f"   image_batch = {image_batch.shape if image_batch is not None else None}")

            # Critical check
            if saliency_batch.shape[1] == 0:
                print(f"\n❌❌❌ CRITICAL ERROR: saliency_batch has 0 channels!")
                print(f"   This is the bug causing the Gaussian blur error!")
                return False

            if image_batch is not None and image_batch.shape[1] == 0:
                print(f"\n❌❌❌ CRITICAL ERROR: image_batch has 0 channels!")
                return False
        else:
            saliency_batch, label_batch = batch
            print(f"✓ Batch loaded successfully:")
            print(f"   saliency_batch.shape = {saliency_batch.shape}")
            print(f"   label_batch.shape = {label_batch.shape}")

            if saliency_batch.shape[1] == 0:
                print(f"\n❌❌❌ CRITICAL ERROR: saliency_batch has 0 channels!")
                return False

        return True

    except Exception as e:
        print(f"❌ Failed to load batch: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Diagnose data integrity")
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to data directory')
    parser.add_argument('--check_files', action='store_true',
                       help='Check individual batch files')
    parser.add_argument('--num_files', type=int, default=5,
                       help='Number of batch files to check')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    saliency_dir = data_dir / "saliency_maps"

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return

    if not saliency_dir.exists():
        print(f"❌ Saliency maps directory not found: {saliency_dir}")
        return

    print(f"Data directory: {data_dir}")
    print(f"Saliency directory: {saliency_dir}")

    # Check batch files
    batch_files = sorted(saliency_dir.glob("batch_*.pkl"))
    print(f"\nFound {len(batch_files)} batch files")

    if args.check_files:
        print(f"\nChecking first {args.num_files} batch files...")
        all_valid = True
        for batch_file in batch_files[:args.num_files]:
            if not diagnose_batch_file(batch_file):
                all_valid = False

        if not all_valid:
            print(f"\n❌ DATA IS CORRUPTED - FIX YOUR DATA FIRST!")
            return

    # Test dataset loading
    if not test_dataset_loading(data_dir, load_images=True):
        print(f"\n❌ DATASET LOADING FAILED - CHECK THE Dataset.__getitem__ METHOD!")
        return

    # Test dataloader
    if not test_dataloader_collation(data_dir, load_images=True):
        print(f"\n❌ DATALOADER COLLATION FAILED - CHECK THE COLLATE FUNCTION!")
        return

    print(f"\n{'='*80}")
    print(f"✅ ALL CHECKS PASSED - DATA IS VALID")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
