"""
Visualize Mean Templates

This script visualizes:
1. A random sample image from a chosen class
2. The corresponding IG/saliency map
3. All K mean templates for that class

Usage:
    python visualize_mean.py --checkpoint ./checkpoints_mean/mean_template_model_final.pth \\
                             --data_dir ./data/saliency_imagenet1k_resnet50_100 \\
                             --class_id 0 \\
                             --num_samples 5
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import joblib
from collections import defaultdict
from tqdm import tqdm

# Import model architecture
from mean_template import MeanTemplatePipeline
from geo_v2 import SaliencyMapDataset


def load_checkpoint(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model configuration from checkpoint
    templates = checkpoint['templates']
    num_classes = templates.shape[0]
    k_subpatterns = templates.shape[1]

    print(f"  Detected {num_classes} classes with {k_subpatterns} sub-patterns each")

    # Create model with same configuration
    model = MeanTemplatePipeline(
        num_classes=num_classes,
        k_subpatterns=k_subpatterns,
        img_res=(224, 224),
        device=device,
        temperature=0.1,
        use_edge_gating=True
    )

    # Load state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Checkpoint loaded successfully!")
    print(f"  Training epoch: {checkpoint['epoch']}")
    print(f"  Template counts: min={checkpoint['template_counts'].min().item():.0f}, "
          f"max={checkpoint['template_counts'].max().item():.0f}, "
          f"mean={checkpoint['template_counts'].mean().item():.1f}")

    return model, checkpoint


def get_class_samples(dataset, class_id, num_samples=5):
    """Get random samples from a specific class."""
    # Find all samples for this class
    class_indices = []
    for idx in range(len(dataset)):
        if dataset.labels[idx].item() == class_id:
            class_indices.append(idx)

    if len(class_indices) == 0:
        raise ValueError(f"No samples found for class {class_id}")

    print(f"Found {len(class_indices)} samples for class {class_id}")

    # Randomly sample
    num_samples = min(num_samples, len(class_indices))
    selected_indices = np.random.choice(class_indices, size=num_samples, replace=False)

    samples = []
    for idx in selected_indices:
        saliency, label, image = dataset[idx]
        samples.append({
            'saliency': saliency,  # (1, 224, 224)
            'label': label,
            'image': image,  # (3, 224, 224) or None
            'index': idx
        })

    return samples


def visualize_class_templates(model, dataset, class_id, num_samples=5, save_path=None):
    """
    Visualize mean templates for a class along with sample images.

    Layout:
    - Row 1: Sample images
    - Row 2: Corresponding IG/saliency maps
    - Row 3+: Mean templates (K subpatterns)
    """
    device = next(model.parameters()).device

    # Get random samples from this class
    samples = get_class_samples(dataset, class_id, num_samples=num_samples)
    num_samples = len(samples)

    # Get mean templates for this class
    templates = model.templates[class_id]  # (K, 1, 224, 224)
    template_counts = model.template_counts[class_id]  # (K,)
    K = templates.shape[0]

    print(f"\nVisualizing class {class_id}:")
    print(f"  Number of samples to show: {num_samples}")
    print(f"  Number of subpatterns: {K}")
    print(f"  Template usage counts: {template_counts.cpu().numpy()}")

    # Create figure
    # Rows: 1 (images) + 1 (saliency) + K (templates)
    num_rows = 2 + K
    fig, axes = plt.subplots(num_rows, num_samples, figsize=(3*num_samples, 3*num_rows))

    # Handle case where num_samples=1 (axes won't be 2D)
    if num_samples == 1:
        axes = axes.reshape(-1, 1)

    # Row 1: Original images
    for col_idx, sample in enumerate(samples):
        ax = axes[0, col_idx]
        image = sample['image']  # (3, 224, 224)

        if image is not None:
            # Convert from (C, H, W) to (H, W, C) for visualization
            img_np = image.cpu().numpy().transpose(1, 2, 0)
            # Clip to [0, 1] range
            img_np = np.clip(img_np, 0, 1)
            ax.imshow(img_np)
        else:
            ax.text(0.5, 0.5, 'No Image', ha='center', va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        ax.set_title(f'Sample {col_idx+1}\n(idx={sample["index"]})', fontsize=10)
        ax.axis('off')

    # Row 2: IG/Saliency maps
    for col_idx, sample in enumerate(samples):
        ax = axes[1, col_idx]
        saliency = sample['saliency'][0].cpu().numpy()  # (224, 224)

        im = ax.imshow(saliency, cmap='hot', interpolation='nearest')
        ax.set_title(f'IG/Saliency', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Rows 3+: Mean templates (K subpatterns)
    for k in range(K):
        for col_idx in range(num_samples):
            ax = axes[2 + k, col_idx]

            if col_idx == 0:
                # Show template for this subpattern
                template = templates[k, 0].cpu().numpy()  # (224, 224)
                count = template_counts[k].item()

                im = ax.imshow(template, cmap='hot', interpolation='nearest')
                ax.set_title(f'Template {k}\n(count={count:.0f})', fontsize=10)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                # Hide unused axes
                ax.axis('off')

    plt.suptitle(f'Class {class_id}: Mean Templates Visualization', fontsize=16, y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to {save_path}")

    plt.show()


def visualize_single_sample(model, dataset, class_id, save_path=None):
    """
    Visualize a single sample with all templates side-by-side.

    Layout: [Image] [IG] [Template 0] [Template 1] ... [Template K-1]
    """
    device = next(model.parameters()).device

    # Get one sample
    samples = get_class_samples(dataset, class_id, num_samples=1)
    sample = samples[0]

    # Get mean templates
    templates = model.templates[class_id]  # (K, 1, 224, 224)
    template_counts = model.template_counts[class_id]  # (K,)
    K = templates.shape[0]

    print(f"\nVisualizing single sample from class {class_id}:")
    print(f"  Sample index: {sample['index']}")
    print(f"  Number of subpatterns: {K}")

    # Create figure: 1 row, 2+K columns (Image, IG, K templates)
    fig, axes = plt.subplots(1, 2 + K, figsize=(3*(2+K), 3))

    # Column 0: Original image
    ax = axes[0]
    image = sample['image']
    if image is not None:
        img_np = image.cpu().numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np, 0, 1)
        ax.imshow(img_np)
    else:
        ax.text(0.5, 0.5, 'No Image', ha='center', va='center')
    ax.set_title('Original Image', fontsize=12)
    ax.axis('off')

    # Column 1: IG/Saliency
    ax = axes[1]
    saliency = sample['saliency'][0].cpu().numpy()
    im = ax.imshow(saliency, cmap='hot', interpolation='nearest')
    ax.set_title('IG/Saliency', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Columns 2+: Mean templates
    for k in range(K):
        ax = axes[2 + k]
        template = templates[k, 0].cpu().numpy()
        count = template_counts[k].item()

        im = ax.imshow(template, cmap='hot', interpolation='nearest')
        ax.set_title(f'Template {k}\n(n={count:.0f})', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f'Class {class_id}: Sample Image + IG + Mean Templates', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Mean Templates',
        epilog="""
Examples:
  # Show single sample with all templates
  python visualize_mean.py --checkpoint ./checkpoints_mean/mean_template_model_final.pth \\
                           --class_id 0 --layout single

  # Show multiple samples in grid
  python visualize_mean.py --checkpoint ./checkpoints_mean/mean_template_model_final.pth \\
                           --class_id 0 --layout grid --num_samples 5
        """
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str,
                       default='./data/saliency_imagenet1k_resnet50_100',
                       help='Directory with saliency maps')
    parser.add_argument('--class_id', type=int, default=0,
                       help='Class ID to visualize (default: 0)')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to show (default: 5, only for grid layout)')
    parser.add_argument('--layout', type=str, default='single', choices=['single', 'grid'],
                       help='Visualization layout: single (1 sample + all templates) or grid (multiple samples)')
    parser.add_argument('--save_path', type=str, default=None,
                       help='Path to save visualization (default: auto-generated)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sample selection (default: 42)')
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model, checkpoint = load_checkpoint(checkpoint_path, device=device)

    # Load dataset
    print(f"\nLoading dataset from {args.data_dir}...")
    dataset = SaliencyMapDataset(
        data_dir=args.data_dir,
        max_samples_per_class=None,
        load_images=True,  # Need images for visualization
        max_samples=100000
    )

    # Auto-generate save path if not provided
    if args.save_path is None:
        checkpoint_dir = checkpoint_path.parent
        args.save_path = checkpoint_dir / f'visualization_class{args.class_id}_{args.layout}.png'

    # Visualize
    if args.layout == 'single':
        visualize_single_sample(
            model=model,
            dataset=dataset,
            class_id=args.class_id,
            save_path=args.save_path
        )
    else:  # grid
        visualize_class_templates(
            model=model,
            dataset=dataset,
            class_id=args.class_id,
            num_samples=args.num_samples,
            save_path=args.save_path
        )

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
