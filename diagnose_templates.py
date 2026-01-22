"""
Quick diagnostic script to check template health in a checkpoint.
"""

import torch
import numpy as np
import argparse
from pathlib import Path

def diagnose_checkpoint(checkpoint_path, class_id=9):
    """Diagnose template quality in checkpoint."""

    print(f"\n{'='*80}")
    print(f"Template Health Diagnostic")
    print(f"{'='*80}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    templates = checkpoint['templates']  # (num_classes, K, 1, H, W)
    template_counts = checkpoint['template_counts']  # (num_classes, K)

    num_classes, K, _, H, W = templates.shape
    print(f"\nCheckpoint Info:")
    print(f"  Classes: {num_classes}")
    print(f"  Sub-patterns per class: {K}")
    print(f"  Template resolution: {H}x{W}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")

    # Global statistics
    print(f"\n{'='*80}")
    print(f"GLOBAL TEMPLATE STATISTICS")
    print(f"{'='*80}")

    templates_np = templates.numpy()
    print(f"Overall min: {templates_np.min():.6f}")
    print(f"Overall max: {templates_np.max():.6f}")
    print(f"Overall mean: {templates_np.mean():.6f}")
    print(f"Overall std: {templates_np.std():.6f}")
    print(f"Percentage of zeros: {(templates_np == 0).sum() / templates_np.size * 100:.2f}%")
    print(f"Percentage < 0.01: {(templates_np < 0.01).sum() / templates_np.size * 100:.2f}%")

    # Per-class analysis for the specified class
    print(f"\n{'='*80}")
    print(f"CLASS {class_id} ANALYSIS (Ostrich)")
    print(f"{'='*80}")

    class_templates = templates[class_id].numpy()  # (K, 1, H, W)
    class_counts = template_counts[class_id].numpy()  # (K,)

    for k in range(K):
        template = class_templates[k, 0]  # (H, W)
        count = class_counts[k]

        print(f"\nPattern {k}:")
        print(f"  Usage count: {int(count)}")
        print(f"  Min: {template.min():.6f}")
        print(f"  Max: {template.max():.6f}")
        print(f"  Mean: {template.mean():.6f}")
        print(f"  Std: {template.std():.6f}")
        print(f"  Zeros: {(template == 0).sum() / template.size * 100:.2f}%")
        print(f"  Values > 0.01: {(template > 0.01).sum() / template.size * 100:.2f}%")
        print(f"  Values > 0.1: {(template > 0.1).sum() / template.size * 100:.2f}%")

        # Check if template has collapsed
        if template.max() < 0.01:
            print(f"  ⚠️  WARNING: Template has COLLAPSED (max < 0.01)!")
        elif (template > 0.01).sum() / template.size < 0.01:
            print(f"  ⚠️  WARNING: Template is >99% near-zero!")

    # Find problematic templates across all classes
    print(f"\n{'='*80}")
    print(f"COLLAPSED TEMPLATES SUMMARY")
    print(f"{'='*80}")

    collapsed_count = 0
    sparse_count = 0

    for c in range(num_classes):
        for k in range(K):
            template = templates[c, k, 0].numpy()
            if template.max() < 0.01:
                collapsed_count += 1
            elif (template > 0.01).sum() / template.size < 0.01:
                sparse_count += 1

    total_templates = num_classes * K
    print(f"Total templates: {total_templates}")
    print(f"Collapsed (max < 0.01): {collapsed_count} ({collapsed_count/total_templates*100:.2f}%)")
    print(f"Nearly empty (>99% zeros): {sparse_count} ({sparse_count/total_templates*100:.2f}%)")
    print(f"Healthy: {total_templates - collapsed_count - sparse_count} ({(total_templates-collapsed_count-sparse_count)/total_templates*100:.2f}%)")

    # Usage statistics
    print(f"\n{'='*80}")
    print(f"TEMPLATE USAGE STATISTICS")
    print(f"{'='*80}")

    counts_np = template_counts.numpy()
    print(f"Total usage count: {counts_np.sum():.0f}")
    print(f"Average usage per template: {counts_np.mean():.2f}")
    print(f"Unused templates (count=0): {(counts_np == 0).sum()}")
    print(f"Min usage: {counts_np.min():.0f}")
    print(f"Max usage: {counts_np.max():.0f}")

    print(f"\n{'='*80}")
    print(f"DIAGNOSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Diagnose checkpoint template health')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--class_id', type=int, default=9,
                       help='Class ID to analyze in detail (default: 9 for Ostrich)')

    args = parser.parse_args()

    diagnose_checkpoint(args.checkpoint, args.class_id)
