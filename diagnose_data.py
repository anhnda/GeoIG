"""
Diagnostic script to inspect saliency data batch files.

This helps identify shape mismatches or corrupted data that could cause
the [32, 0, 224, 224] tensor error.

Usage:
    python diagnose_data.py --data_dir ./data/saliency_imagenet1k_resnet50_100
"""

import argparse
import joblib
from pathlib import Path
import numpy as np


def diagnose_batch_files(data_dir):
    """Inspect batch files for shape issues."""
    data_dir = Path(data_dir)
    saliency_dir = data_dir / "saliency_maps"

    if not saliency_dir.exists():
        print(f"ERROR: Saliency directory not found: {saliency_dir}")
        return

    batch_files = sorted(saliency_dir.glob("batch_*.pkl"))
    print(f"Found {len(batch_files)} batch files\n")

    if len(batch_files) == 0:
        print("ERROR: No batch files found!")
        return

    # Check metadata
    metadata_path = data_dir / "metadata.pkl"
    if metadata_path.exists():
        metadata = joblib.load(metadata_path)
        print("="*80)
        print("METADATA")
        print("="*80)
        for key, value in metadata.items():
            if key not in ['class_counts', 'class_correct']:
                print(f"  {key}: {value}")
        print()

    # Inspect first few batch files
    issues_found = []

    for i, batch_file in enumerate(batch_files[:5]):  # Check first 5 batches
        print(f"{"="*80}")
        print(f"BATCH FILE {i}: {batch_file.name}")
        print(f"{"="*80}")

        try:
            batch_data = joblib.load(batch_file)
            print(f"  Items in batch: {len(batch_data)}")

            if len(batch_data) == 0:
                print(f"  ❌ WARNING: Empty batch file!")
                issues_found.append(f"{batch_file.name}: empty batch")
                continue

            # Check first item
            item = batch_data[0]
            print(f"\n  First item keys: {list(item.keys())}")

            # Check saliency map
            if 'saliency_map' in item:
                saliency_shape = item['saliency_map'].shape
                saliency_dtype = item['saliency_map'].dtype
                saliency_min = item['saliency_map'].min()
                saliency_max = item['saliency_map'].max()

                print(f"\n  Saliency Map:")
                print(f"    Shape: {saliency_shape}")
                print(f"    Dtype: {saliency_dtype}")
                print(f"    Range: [{saliency_min:.4f}, {saliency_max:.4f}]")

                if saliency_shape != (224, 224):
                    print(f"    ❌ WARNING: Expected shape (224, 224), got {saliency_shape}")
                    issues_found.append(f"{batch_file.name}: invalid saliency shape {saliency_shape}")
                else:
                    print(f"    ✅ Shape is correct")

            else:
                print(f"  ❌ ERROR: No 'saliency_map' key found!")
                issues_found.append(f"{batch_file.name}: missing saliency_map key")

            # Check RGB image if present
            if 'rgb_image' in item:
                rgb_shape = item['rgb_image'].shape
                rgb_dtype = item['rgb_image'].dtype
                rgb_min = item['rgb_image'].min()
                rgb_max = item['rgb_image'].max()

                print(f"\n  RGB Image:")
                print(f"    Shape: {rgb_shape}")
                print(f"    Dtype: {rgb_dtype}")
                print(f"    Range: [{rgb_min:.4f}, {rgb_max:.4f}]")

                if rgb_shape != (3, 224, 224):
                    print(f"    ⚠️  WARNING: Expected shape (3, 224, 224), got {rgb_shape}")
                    if len(rgb_shape) == 3 and 3 in rgb_shape:
                        print(f"    ℹ️  Shape can be handled (will transpose if needed)")
                    else:
                        issues_found.append(f"{batch_file.name}: invalid RGB shape {rgb_shape}")
                else:
                    print(f"    ✅ Shape is correct")
            else:
                print(f"\n  ℹ️  No 'rgb_image' key (this is OK if not using edge gating)")

            # Check all items for consistency
            print(f"\n  Checking all {len(batch_data)} items...")
            shape_issues = 0
            missing_keys = 0

            for idx, item in enumerate(batch_data):
                if 'saliency_map' not in item:
                    missing_keys += 1
                    print(f"    ❌ Item {idx}: missing saliency_map")
                else:
                    if item['saliency_map'].shape != (224, 224):
                        shape_issues += 1
                        print(f"    ❌ Item {idx}: shape {item['saliency_map'].shape}")

            if shape_issues > 0:
                print(f"  ❌ Found {shape_issues} items with wrong saliency shape")
                issues_found.append(f"{batch_file.name}: {shape_issues} items with wrong shape")
            elif missing_keys > 0:
                print(f"  ❌ Found {missing_keys} items with missing saliency_map")
                issues_found.append(f"{batch_file.name}: {missing_keys} items missing key")
            else:
                print(f"  ✅ All items have correct shape")

        except Exception as e:
            print(f"  ❌ ERROR loading batch file: {e}")
            issues_found.append(f"{batch_file.name}: failed to load - {e}")

        print()

    # Summary
    print("="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80)

    if len(issues_found) == 0:
        print("✅ No issues found! Data appears to be in correct format.")
        print("\nIf you're still getting [32, 0, 224, 224] error, the issue might be:")
        print("  1. During DataLoader collation")
        print("  2. In a later batch file (only first 5 were checked)")
        print("  3. In the model's data preprocessing")
    else:
        print(f"❌ Found {len(issues_found)} issues:")
        for issue in issues_found:
            print(f"  - {issue}")

        print("\nRECOMMENDED ACTIONS:")
        print("  1. Re-export the saliency data using saliency_map_export.py")
        print("  2. Check if the export process completed successfully")
        print("  3. Verify disk space wasn't an issue during export")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose saliency data batch files")
    parser.add_argument('--data_dir', type=str,
                       default='./data/saliency_imagenet1k_resnet50_100',
                       help='Path to data directory')

    args = parser.parse_args()

    diagnose_batch_files(args.data_dir)
