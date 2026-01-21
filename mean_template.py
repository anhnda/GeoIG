"""
Mean Template - Direct mean calculation (no training needed!)

Simply computes the mean saliency map for each class.
No epochs, no optimization, just pure averaging.

Usage:
    python mean_template.py --data_dir ./data/saliency_imagenet1k_resnet50_100 \
                            --output mean_templates.pth
"""

import torch
from pathlib import Path
from tqdm import tqdm
import argparse

from geo_v2 import SaliencyMapDataset


def compute_mean_templates(dataset, num_classes=1000, sparsity_percentile=98, top_k_percent=3):
    """
    Compute mean template for each class by averaging all samples.

    Args:
        dataset: SaliencyMapDataset
        num_classes: Number of classes
        sparsity_percentile: Threshold for sparsity
        top_k_percent: Keep only top K% of pixels

    Returns:
        templates: (num_classes, 1, 224, 224) mean templates
        counts: (num_classes,) number of samples per class
    """
    print(f"\nComputing mean templates for {num_classes} classes...")

    # Accumulate sums and counts
    sums = torch.zeros((num_classes, 1, 224, 224))
    counts = torch.zeros(num_classes)

    # Single pass through dataset
    for idx in tqdm(range(len(dataset)), desc="Computing means"):
        saliency, label, _ = dataset[idx]
        sums[label] += saliency.cpu()
        counts[label] += 1

    # Compute means
    templates = torch.zeros((num_classes, 1, 224, 224))

    for c in tqdm(range(num_classes), desc="Finalizing templates"):
        if counts[c] == 0:
            continue

        # Simple mean
        mean_template = sums[c] / counts[c]

        # === TOP-K SHARPENING ===
        num_pixels = mean_template.numel()
        top_k = max(int(num_pixels * top_k_percent / 100.0), 1)

        # Get top-K values
        mean_flat = mean_template.flatten()
        top_k_values, top_k_indices = torch.topk(mean_flat, k=top_k)

        # Zero out all but top-K
        mean_sharp = torch.zeros_like(mean_flat)
        mean_sharp[top_k_indices] = top_k_values
        mean_template = mean_sharp.reshape(mean_template.shape)

        # Additional sparsity thresholding
        threshold = torch.quantile(mean_template.flatten(), sparsity_percentile / 100.0)
        mean_template = torch.where(
            mean_template >= threshold,
            mean_template,
            torch.zeros_like(mean_template)
        )

        # Normalize
        mean_max = mean_template.max()
        if mean_max > 0:
            mean_template = mean_template / mean_max

        templates[c] = mean_template

    print(f"\n✓ Mean templates computed!")
    print(f"  Classes with samples: {(counts > 0).sum().item()}/{num_classes}")
    print(f"  Min samples/class: {counts[counts > 0].min().item():.0f}")
    print(f"  Max samples/class: {counts.max().item():.0f}")
    print(f"  Avg samples/class: {counts[counts > 0].mean().item():.1f}")

    return templates, counts


def main():
    parser = argparse.ArgumentParser(
        description='Compute Mean Templates (no training needed!)',
        epilog="""
Example:
  python mean_template.py --data_dir ./data/saliency_imagenet1k_resnet50_100 \\
                          --output mean_templates.pth
        """
    )
    parser.add_argument('--data_dir', type=str,
                       default='./data/saliency_imagenet1k_resnet50_100',
                       help='Directory with saliency maps')
    parser.add_argument('--num_classes', type=int, default=1000,
                       help='Number of classes (default: 1000)')
    parser.add_argument('--output', type=str, default='checkpoints_mean/mean_templates.pth',
                       help='Output file for mean templates (default: mean_templates.pth)')
    parser.add_argument('--sparsity_percentile', type=float, default=20,
                       help='Sparsity percentile threshold (default: 98)')
    parser.add_argument('--top_k_percent', type=float, default=85,
                       help='Keep only top K%% of pixels (default: 3)')
    parser.add_argument('--max_samples_per_class', type=int, default=None,
                       help='Max samples per class (default: None = all)')
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"Mean Template Computation (Direct - No Training!)")
    print(f"{'='*80}")

    # Load dataset
    print(f"\nLoading dataset from {args.data_dir}...")
    dataset = SaliencyMapDataset(
        data_dir=args.data_dir,
        max_samples_per_class=args.max_samples_per_class,
        load_images=False,  # Don't need images, just saliency maps
        max_samples=100000
    )

    # Compute mean templates
    templates, counts = compute_mean_templates(
        dataset=dataset,
        num_classes=args.num_classes,
        sparsity_percentile=args.sparsity_percentile,
        top_k_percent=args.top_k_percent
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'templates': templates,  # (num_classes, 1, 224, 224)
        'counts': counts,  # (num_classes,)
        'config': {
            'num_classes': args.num_classes,
            'sparsity_percentile': args.sparsity_percentile,
            'top_k_percent': args.top_k_percent,
            'data_dir': args.data_dir
        }
    }, output_path)

    print(f"\n✓ Mean templates saved to {output_path}")
    print(f"\nTo visualize:")
    print(f"  python visualize_mean.py --checkpoint {output_path} --class_id 0")


if __name__ == "__main__":
    main()
