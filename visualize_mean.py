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
from PIL import Image
from torchvision import transforms

from saliency_map_export import ImageNet1kSaliencyDataset, IMAGENET_RAW_DIR, IMAGENET_SAMPLED_DIR


def load_checkpoint(checkpoint_path, device='cuda'):
    """Load mean templates from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract templates
    templates = checkpoint['templates']  # (num_classes, 1, 224, 224)
    counts = checkpoint['counts']  # (num_classes,)

    print(f"✓ Checkpoint loaded successfully!")
    print(f"  Number of classes: {templates.shape[0]}")
    print(f"  Template counts: min={counts[counts > 0].min().item():.0f}, "
          f"max={counts.max().item():.0f}, "
          f"mean={counts[counts > 0].mean().item():.1f}")

    return templates, counts


def _is_cache_valid(cache_file, saliency_dir):
    """
    Check if the cached index file exists and is up-to-date.

    Returns:
        bool: True if cache is valid, False otherwise
    """
    if not cache_file.exists():
        return False

    # Check if cache is newer than all batch files
    cache_mtime = cache_file.stat().st_mtime
    batch_files = sorted(saliency_dir.glob("batch_*.pkl"))

    if not batch_files:
        return False

    # If any batch file is newer than cache, cache is invalid
    for batch_file in batch_files:
        if batch_file.stat().st_mtime > cache_mtime:
            return False

    return True


def _build_class_index(saliency_dir):
    """
    Build an index mapping class_id -> list of (batch_file, item_index).

    This allows fast lookup without loading all batch files.

    Returns:
        dict: {class_id: [(batch_file, item_idx), ...]}
    """
    import time

    print("  Building class index from batch files...")
    start_time = time.time()

    class_index = defaultdict(list)
    batch_files = sorted(saliency_dir.glob("batch_*.pkl"))

    if not batch_files:
        print(f"  Warning: No batch files found in {saliency_dir}")
        return {}

    for batch_idx, batch_file in enumerate(batch_files):
        batch_data = joblib.load(batch_file)

        for item_idx, item in enumerate(batch_data):
            class_id = item['true_label']
            class_index[class_id].append((str(batch_file), item_idx))

        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {batch_idx + 1}/{len(batch_files)} batch files...")

    elapsed = time.time() - start_time
    total_samples = sum(len(samples) for samples in class_index.values())
    print(f"  ✓ Index built in {elapsed:.2f}s: {len(class_index)} classes, {total_samples} samples")

    return dict(class_index)


def _save_index_cache(cache_file, class_index):
    """
    Save the class index to cache file.

    Args:
        cache_file: Path to cache file
        class_index: The index to save
    """
    print(f"  Saving index cache to {cache_file}...")
    joblib.dump(class_index, cache_file, compress=3)
    print(f"  ✓ Cache saved")


def _load_index_cache(cache_file):
    """
    Load the class index from cache file.

    Returns:
        dict: The cached index
    """
    import time

    print(f"  Loading index from cache...")
    start_time = time.time()
    class_index = joblib.load(cache_file)
    elapsed = time.time() - start_time
    total_samples = sum(len(samples) for samples in class_index.values())
    print(f"  ✓ Cache loaded in {elapsed:.2f}s: {len(class_index)} classes, {total_samples} samples")
    return class_index


def _load_or_build_index(cache_file, saliency_dir):
    """
    Load index from cache if valid, otherwise build and cache it.

    Returns:
        dict: {class_id: [(batch_file, item_idx), ...]}
    """
    if _is_cache_valid(cache_file, saliency_dir):
        print("  Cache is up-to-date, loading from cache...")
        return _load_index_cache(cache_file)
    else:
        if cache_file.exists():
            print("  Cache is outdated, rebuilding...")
        else:
            print("  No cache found, building for the first time...")

        class_index = _build_class_index(saliency_dir)
        _save_index_cache(cache_file, class_index)
        return class_index


def denormalize_image(image_tensor):
    """
    Denormalize ImageNet-normalized image for visualization.

    Args:
        image_tensor: (3, H, W) normalized image

    Returns:
        (H, W, 3) RGB image in [0, 1] range
    """
    # ImageNet normalization parameters
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = image_tensor.copy()
    # Denormalize: x_orig = x_norm * std + mean
    for i in range(3):
        image[i] = image[i] * std[i] + mean[i]
    # Clip to valid range
    image = np.clip(image, 0, 1)
    # Convert to HWC format
    image = image.transpose(1, 2, 0)
    return image


def get_original_image(sample, imagenet_dataset, image_transform):
    """
    Get original image from sample - either from stored data or load from ImageNet dataset.

    Args:
        sample: Sample dictionary
        imagenet_dataset: ImageNet dataset instance (or None)
        image_transform: Transform to apply to images

    Returns:
        (H, W, 3) RGB image in [0, 1] range, or None if not available
    """
    # Method 1: Image already stored in normalized format (backward compatibility)
    if 'image' in sample and sample['image'] is not None:
        # Already a tensor in (3, 224, 224) format
        img_np = sample['image'].cpu().numpy()
        return denormalize_image(img_np)

    if 'original_image' in sample and sample['original_image'] is not None:
        return denormalize_image(sample['original_image'])

    # Method 2: Load from ImageNet dataset using sample_index
    if 'sample_index' in sample and imagenet_dataset is not None:
        try:
            sample_idx = sample['sample_index']
            image_pil, _ = imagenet_dataset[sample_idx]
            # Transform to tensor and convert to numpy array
            img_tensor = image_transform(image_pil).numpy()
            # Convert from CHW to HWC and ensure [0, 1] range
            img_array = img_tensor.transpose(1, 2, 0)
            return img_array
        except Exception as e:
            print(f"Warning: Could not load image from dataset index {sample.get('sample_index')}: {e}")

    return None


def get_class_samples(class_index, class_id, num_samples=5):
    """
    Get random samples from a specific class using pre-built index for fast lookup.

    Args:
        class_index: Pre-built class index {class_id: [(batch_file, item_idx), ...]}
        class_id: Class index to sample from
        num_samples: Number of samples to retrieve

    Returns:
        List of sample dictionaries
    """
    # Check if class exists in index
    if class_id not in class_index:
        print(f"Warning: Class {class_id} not found in index")
        return []

    # Get locations for this class from index
    class_locations = class_index[class_id]
    print(f"Found {len(class_locations)} samples for class {class_id}")

    # Randomly sample locations
    num_samples = min(num_samples, len(class_locations))
    selected_indices = np.random.choice(len(class_locations), size=num_samples, replace=False)
    selected_locations = [class_locations[i] for i in selected_indices]

    # Load only the required batch files (group by batch file)
    batch_to_indices = defaultdict(list)
    for batch_file, item_idx in selected_locations:
        batch_to_indices[batch_file].append(item_idx)

    # Load samples from each batch file
    samples = []
    loaded_batches = {}  # Cache loaded batches

    for batch_file, item_indices in batch_to_indices.items():
        # Load batch file once
        if batch_file not in loaded_batches:
            batch_data = joblib.load(batch_file)
            loaded_batches[batch_file] = batch_data

        # Extract requested items
        for item_idx in item_indices:
            item = loaded_batches[batch_file][item_idx]

            # Convert to tensor format expected by visualization functions
            saliency = torch.from_numpy(item['saliency_map']).float().unsqueeze(0)  # (1, 224, 224)

            # Store sample with necessary metadata
            samples.append({
                'saliency': saliency,
                'label': class_id,
                'image': None,  # Will be loaded on-demand
                'sample_index': item.get('sample_index', item.get('index')),
                'original_image': item.get('original_image'),  # If stored in batch
                'index': item.get('sample_index', item_idx)
            })

    return samples


def visualize_class_templates(templates, counts, class_index, class_id, num_samples=5,
                            imagenet_dataset=None, image_transform=None, save_path=None):
    """
    Visualize mean template for a class along with sample images.

    Args:
        templates: (num_classes, 1, 224, 224) mean templates
        counts: (num_classes,) sample counts
        class_index: Pre-built class index
        class_id: Class to visualize
        num_samples: Number of samples to show
        imagenet_dataset: Optional ImageNet dataset for loading images
        image_transform: Transform for images
        save_path: Optional path to save figure

    Layout:
    - Row 1: Sample images
    - Row 2: Corresponding IG/saliency maps
    - Row 3: Mean template
    """
    # Get random samples from this class
    samples = get_class_samples(class_index, class_id, num_samples=num_samples)
    num_samples = len(samples)

    # Get mean template for this class
    template = templates[class_id]  # (1, 224, 224)
    count = counts[class_id].item()

    print(f"\nVisualizing class {class_id}:")
    print(f"  Number of samples to show: {num_samples}")
    print(f"  Template sample count: {count:.0f}")

    # Create figure: 3 rows (images, saliency, template)
    fig, axes = plt.subplots(3, num_samples, figsize=(3*num_samples, 9))

    # Handle case where num_samples=1 (axes won't be 2D)
    if num_samples == 1:
        axes = axes.reshape(-1, 1)

    # Row 1: Original images
    for col_idx, sample in enumerate(samples):
        ax = axes[0, col_idx]

        # Load original image on-demand
        original_image = get_original_image(sample, imagenet_dataset, image_transform)

        if original_image is not None:
            ax.imshow(original_image)
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

    # Row 3: Mean template (same for all columns, but shown once)
    for col_idx in range(num_samples):
        ax = axes[2, col_idx]

        if col_idx == 0:
            # Show mean template
            template_np = template[0].cpu().numpy()  # (224, 224)

            im = ax.imshow(template_np, cmap='hot', interpolation='nearest')
            ax.set_title(f'Mean Template\n(n={count:.0f})', fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            # Hide unused axes
            ax.axis('off')

    plt.suptitle(f'Class {class_id}: Mean Template Visualization', fontsize=16, y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to {save_path}")

    plt.show()


def visualize_single_sample(templates, counts, class_index, class_id,
                          imagenet_dataset=None, image_transform=None, save_path=None):
    """
    Visualize a single sample with mean template side-by-side.

    Args:
        templates: (num_classes, 1, 224, 224) mean templates
        counts: (num_classes,) sample counts
        class_index: Pre-built class index
        class_id: Class to visualize
        imagenet_dataset: Optional ImageNet dataset for loading images
        image_transform: Transform for images
        save_path: Optional path to save figure

    Layout: [Image] [IG] [Mean Template]
    """
    # Get one sample
    samples = get_class_samples(class_index, class_id, num_samples=1)
    sample = samples[0]

    # Get mean template
    template = templates[class_id]  # (1, 224, 224)
    count = counts[class_id].item()

    print(f"\nVisualizing single sample from class {class_id}:")
    print(f"  Sample index: {sample['index']}")
    print(f"  Template sample count: {count:.0f}")

    # Create figure: 1 row, 3 columns (Image, IG, Template)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Column 0: Original image
    ax = axes[0]

    # Load original image on-demand
    original_image = get_original_image(sample, imagenet_dataset, image_transform)

    if original_image is not None:
        ax.imshow(original_image)
    else:
        ax.text(0.5, 0.5, 'No Image', ha='center', va='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    ax.set_title('Original Image', fontsize=12)
    ax.axis('off')

    # Column 1: IG/Saliency
    ax = axes[1]
    saliency = sample['saliency'][0].cpu().numpy()
    im = ax.imshow(saliency, cmap='hot', interpolation='nearest')
    ax.set_title('IG/Saliency', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Column 2: Mean template
    ax = axes[2]
    template_np = template[0].cpu().numpy()
    im = ax.imshow(template_np, cmap='hot', interpolation='nearest')
    ax.set_title(f'Mean Template\n(n={count:.0f})', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f'Class {class_id}: Sample Image + IG + Mean Template', fontsize=14)
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
    parser.add_argument('--images_per_class', type=int, default=100,
                       help='Number of images per class (must match saliency generation, default: 100)')
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

    templates, counts = load_checkpoint(checkpoint_path, device=device)

    # Load ImageNet dataset for accessing original images
    print(f"\nLoading ImageNet dataset for original images...")
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    try:
        imagenet_dataset = ImageNet1kSaliencyDataset(
            raw_dir=IMAGENET_RAW_DIR,
            sampled_dir=IMAGENET_SAMPLED_DIR,
            output_dir=Path(args.data_dir),
            images_per_class=args.images_per_class,
            force_resample=False
        )
        print(f"✓ ImageNet dataset loaded with {len(imagenet_dataset)} images")
    except Exception as e:
        print(f"Warning: Could not load ImageNet dataset: {e}")
        print(f"Original images will not be available for visualization")
        imagenet_dataset = None

    # Build or load class index for fast sample lookup
    data_dir = Path(args.data_dir)
    saliency_dir = data_dir / "saliency_maps"

    if not saliency_dir.exists():
        raise FileNotFoundError(f"Saliency directory not found: {saliency_dir}")

    cache_file = data_dir / "class_index_cache.pkl"
    print(f"\nBuilding/loading class index for fast lookup...")
    class_index = _load_or_build_index(cache_file, saliency_dir)
    print(f"✓ Class index ready!")

    # Auto-generate save path if not provided
    if args.save_path is None:
        checkpoint_dir = checkpoint_path.parent
        if checkpoint_dir.name == '.':
            checkpoint_dir = Path.cwd()
        args.save_path = checkpoint_dir / f'visualization_class{args.class_id}_{args.layout}.png'

    # Visualize
    if args.layout == 'single':
        visualize_single_sample(
            templates=templates,
            counts=counts,
            class_index=class_index,
            class_id=args.class_id,
            imagenet_dataset=imagenet_dataset,
            image_transform=image_transform,
            save_path=args.save_path
        )
    else:  # grid
        visualize_class_templates(
            templates=templates,
            counts=counts,
            class_index=class_index,
            class_id=args.class_id,
            num_samples=args.num_samples,
            imagenet_dataset=imagenet_dataset,
            image_transform=image_transform,
            save_path=args.save_path
        )

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
