"""
Pattern Visualization for Geodesic LDDMM Model

This script visualizes:
1. The K sub-patterns (templates) learned for a specific class
2. A sample image's IG saliency map
3. The decomposition of the IG into class-patterns (alignment + cluster assignment)

Usage:
    # Visualize class 0 patterns
    python visualize_patterns.py --checkpoint checkpoints/lddmm_model_final.pth --class_id 0

    # Visualize specific class with custom data
    python visualize_patterns.py --checkpoint checkpoints/lddmm_model_epoch_10.pth \
                                  --class_id 281 \
                                  --data_dir ./data/saliency_imagenet1k_resnet50_100
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import joblib
from PIL import Image
from torchvision import transforms

# Import model components from geo_patterns and geo_x
import sys
sys.path.append('.')
from geo_patterns import LDDMM_GlobalPatternPipeline
from misc.geo_x import AdvancedLDDMM_Pipeline
from full_classes import IMAGENET2012_CLASSES
from misc.saliency_map_export import ImageNet1kSaliencyDataset, IMAGENET_RAW_DIR, IMAGENET_SAMPLED_DIR


class PatternVisualizer:
    """Visualize learned patterns and decompositions."""

    def __init__(self, checkpoint_path, data_dir, device='cuda', imagenet_dir=None, images_per_class=100):
        self.device = device
        self.data_dir = Path(data_dir)
        self.imagenet_dir = Path(imagenet_dir) if imagenet_dir else None

        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        # Transform for loading ImageNet images (no normalization - just resize)
        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        # Load ImageNet dataset for accessing original images
        print(f"\nLoading ImageNet dataset for original images...")
        try:
            self.imagenet_dataset = ImageNet1kSaliencyDataset(
                raw_dir=IMAGENET_RAW_DIR,
                sampled_dir=IMAGENET_SAMPLED_DIR,
                output_dir=self.data_dir,
                images_per_class=images_per_class,
                force_resample=False
            )
            print(f"✓ ImageNet dataset loaded with {len(self.imagenet_dataset)} images")
        except Exception as e:
            print(f"Warning: Could not load ImageNet dataset: {e}")
            print(f"Original images will not be available for visualization")
            self.imagenet_dataset = None

        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract configuration
        self.num_classes = checkpoint['templates'].shape[0]
        self.k_subpatterns = checkpoint['templates'].shape[1]

        # Auto-detect velocity field resolution from checkpoint
        # Try to detect if this is an advanced (multi-scale) or simple (single-scale) checkpoint
        # Advanced model has: predictor.coarse_v_head and predictor.fine_v_head
        # Simple model has: predictor.v_head

        # First, try to find fine_v_head (advanced model)
        v_head_keys = [k for k in checkpoint['model_state_dict'].keys()
                       if 'predictor.fine_v_head' in k and 'weight' in k and 'Linear' not in k]

        if not v_head_keys:
            # Try simple model (single v_head)
            v_head_keys = [k for k in checkpoint['model_state_dict'].keys()
                           if 'predictor.v_head' in k and 'weight' in k and 'Linear' not in k]

        if not v_head_keys:
            # Last resort: try coarse_v_head
            v_head_keys = [k for k in checkpoint['model_state_dict'].keys()
                           if 'predictor.coarse_v_head' in k and 'weight' in k and 'Linear' not in k]

        if not v_head_keys:
            # If still not found, print available keys to help debug
            print("\nAvailable model keys:")
            for k in sorted(checkpoint['model_state_dict'].keys()):
                if 'predictor' in k and 'weight' in k:
                    print(f"  {k}: {checkpoint['model_state_dict'][k].shape}")
            raise ValueError("Could not find velocity head in checkpoint. See available keys above.")

        # Sort to get the last layer (highest index)
        v_head_keys.sort()
        v_head_key = v_head_keys[-1]

        v_head_weight_shape = checkpoint['model_state_dict'][v_head_key].shape
        v_dim = v_head_weight_shape[0]  # Output dimension: 2 * v_H * v_W

        # v_dim = 2 * v_res * v_res, so v_res = sqrt(v_dim / 2)
        v_res_size = int(np.sqrt(v_dim / 2))
        v_res = (v_res_size, v_res_size)

        # Detect model type based on checkpoint keys
        has_fine_v_head = any('fine_v_head' in k for k in checkpoint['model_state_dict'].keys())
        has_coarse_v_head = any('coarse_v_head' in k for k in checkpoint['model_state_dict'].keys())
        is_advanced_model = has_fine_v_head or has_coarse_v_head

        print(f"Loaded model:")
        print(f"  Type: {'Advanced (Multi-Scale)' if is_advanced_model else 'Simple (Single-Scale)'}")
        print(f"  Classes: {self.num_classes}")
        print(f"  Sub-patterns per class: {self.k_subpatterns}")
        print(f"  Velocity field resolution: {v_res} (auto-detected)")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")

        # Create model with auto-detected architecture
        self.is_advanced_model = is_advanced_model
        if is_advanced_model:
            self.model = AdvancedLDDMM_Pipeline(
                num_classes=self.num_classes,
                k_subpatterns=self.k_subpatterns,
                img_res=(224, 224),
                device=device,
                use_ot=True,  # Default to True for advanced model
                use_edge_gating=False  # Default to False (no edge gating during visualization)
            ).to(device)
        else:
            self.model = LDDMM_GlobalPatternPipeline(
                num_classes=self.num_classes,
                k_subpatterns=self.k_subpatterns,
                img_res=(224, 224),
                device=device,
                v_res=v_res
            ).to(device)

        # Load weights (handle torch.compile() prefix and incompatible keys)
        state_dict = checkpoint['model_state_dict']

        # Strip _orig_mod. prefix if present (from torch.compile())
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            print("  Detected torch.compile() checkpoint - stripping _orig_mod. prefix...")
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        # Filter out keys that don't exist in the current model (e.g., signal_enhancer from older versions)
        model_keys = set(self.model.state_dict().keys())
        filtered_state_dict = {}
        incompatible_keys = []

        for k, v in state_dict.items():
            if k in model_keys:
                filtered_state_dict[k] = v
            else:
                incompatible_keys.append(k)

        if incompatible_keys:
            print(f"  Skipping {len(incompatible_keys)} incompatible keys from checkpoint:")
            for k in incompatible_keys[:5]:  # Show first 5
                print(f"    - {k}")
            if len(incompatible_keys) > 5:
                print(f"    ... and {len(incompatible_keys) - 5} more")

        self.model.load_state_dict(filtered_state_dict, strict=False)
        self.model.eval()

        # Templates are already in the model's buffers from state_dict
        print(f"✓ Model loaded successfully!")

        # Store path to saliency data (don't load into memory)
        self.saliency_dir = self.data_dir / "saliency_maps"
        if not self.saliency_dir.exists():
            print(f"Warning: Saliency directory not found: {self.saliency_dir}")

        # Get class names
        self.class_names = {idx: name for idx, (wnid, name) in enumerate(IMAGENET2012_CLASSES.items())}

        # Build or load class index for fast sample lookup
        self.cache_file = self.data_dir / "class_index_cache.pkl"
        print(f"\nBuilding/loading class index for fast lookup...")
        self.class_index = self._load_or_build_index()
        print(f"✓ Class index ready!")

    def denormalize_image(self, image_tensor):
        """
        Denormalize ImageNet-normalized image for visualization.

        Args:
            image_tensor: (3, H, W) normalized image

        Returns:
            (H, W, 3) RGB image in [0, 1] range
        """
        image = image_tensor.copy()
        # Denormalize: x_orig = x_norm * std + mean
        for i in range(3):
            image[i] = image[i] * self.std[i] + self.mean[i]
        # Clip to valid range
        image = np.clip(image, 0, 1)
        # Convert to HWC format
        image = image.transpose(1, 2, 0)
        return image

    def _is_cache_valid(self):
        """
        Check if the cached index file exists and is up-to-date.

        Returns:
            bool: True if cache is valid, False otherwise
        """
        if not self.cache_file.exists():
            return False

        # Check if cache is newer than all batch files
        cache_mtime = self.cache_file.stat().st_mtime
        batch_files = sorted(self.saliency_dir.glob("batch_*.pkl"))

        if not batch_files:
            return False

        # If any batch file is newer than cache, cache is invalid
        for batch_file in batch_files:
            if batch_file.stat().st_mtime > cache_mtime:
                return False

        return True

    def _build_class_index(self):
        """
        Build an index mapping class_id -> list of (batch_file, item_index).

        This allows fast lookup without loading all batch files.

        Returns:
            dict: {class_id: [(batch_file, item_idx), ...]}
        """
        from collections import defaultdict
        import time

        print("  Building class index from batch files...")
        start_time = time.time()

        class_index = defaultdict(list)
        batch_files = sorted(self.saliency_dir.glob("batch_*.pkl"))

        if not batch_files:
            print(f"  Warning: No batch files found in {self.saliency_dir}")
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

    def _save_index_cache(self, class_index):
        """
        Save the class index to cache file.

        Args:
            class_index: The index to save
        """
        print(f"  Saving index cache to {self.cache_file}...")
        joblib.dump(class_index, self.cache_file, compress=3)
        print(f"  ✓ Cache saved")

    def _load_index_cache(self):
        """
        Load the class index from cache file.

        Returns:
            dict: The cached index
        """
        print(f"  Loading index from cache...")
        import time
        start_time = time.time()
        class_index = joblib.load(self.cache_file)
        elapsed = time.time() - start_time
        total_samples = sum(len(samples) for samples in class_index.values())
        print(f"  ✓ Cache loaded in {elapsed:.2f}s: {len(class_index)} classes, {total_samples} samples")
        return class_index

    def _load_or_build_index(self):
        """
        Load index from cache if valid, otherwise build and cache it.

        Returns:
            dict: {class_id: [(batch_file, item_idx), ...]}
        """
        if self._is_cache_valid():
            print("  Cache is up-to-date, loading from cache...")
            return self._load_index_cache()
        else:
            if self.cache_file.exists():
                print("  Cache is outdated, rebuilding...")
            else:
                print("  No cache found, building for the first time...")

            class_index = self._build_class_index()
            self._save_index_cache(class_index)
            return class_index

    def sample_class_images(self, class_id, num_samples=None):
        """
        Sample images from a specific class using pre-built index for fast lookup.

        Args:
            class_id: Class index to sample from
            num_samples: Number of samples to retrieve (None = all available)

        Returns:
            List of sample dictionaries for the class
        """
        # Check if class exists in index
        if class_id not in self.class_index:
            print(f"Warning: Class {class_id} not found in index")
            return []

        # Get locations for this class from index
        class_locations = self.class_index[class_id]

        # Randomly sample if num_samples is specified
        if num_samples is not None and num_samples < len(class_locations):
            selected_indices = np.random.choice(len(class_locations), size=num_samples, replace=False)
            class_locations = [class_locations[i] for i in selected_indices]
        elif num_samples is not None:
            # If requested more samples than available, just use all
            class_locations = class_locations[:num_samples]

        # Load only the required batch files (group by batch file)
        from collections import defaultdict
        batch_to_indices = defaultdict(list)

        for batch_file, item_idx in class_locations:
            batch_to_indices[batch_file].append(item_idx)

        # Load samples from each batch file
        class_samples = []
        loaded_items = {}  # Cache loaded batches

        for batch_file, item_indices in batch_to_indices.items():
            # Load batch file once
            if batch_file not in loaded_items:
                batch_data = joblib.load(batch_file)
                loaded_items[batch_file] = batch_data

            # Extract requested items
            for item_idx in item_indices:
                item = loaded_items[batch_file][item_idx]

                # Only store necessary data
                sample_dict = {
                    'sample_index': item.get('sample_index', item.get('index')),
                    'confidence': item['confidence'],
                    'saliency_map': item['saliency_map']
                }

                # Add optional fields if available
                if 'original_image' in item:
                    sample_dict['original_image'] = item['original_image']
                if 'image_path' in item:
                    sample_dict['image_path'] = item['image_path']

                class_samples.append(sample_dict)

        return class_samples

    def get_original_image(self, sample):
        """
        Get original image from sample - either from stored data or load from ImageNet dataset.

        Args:
            sample: Sample dictionary

        Returns:
            (H, W, 3) RGB image in [0, 1] range, or None if not available
        """
        # Method 1: Image already stored in normalized format (backward compatibility)
        if 'original_image' in sample:
            return self.denormalize_image(sample['original_image'])

        # Method 2: Load from ImageNet dataset using sample_index
        if 'sample_index' in sample and self.imagenet_dataset is not None:
            try:
                sample_idx = sample['sample_index']
                image_pil, _ = self.imagenet_dataset[sample_idx]
                # Transform to tensor and convert to numpy array
                img_tensor = self.image_transform(image_pil).numpy()
                # Convert from CHW to HWC and ensure [0, 1] range
                img_array = img_tensor.transpose(1, 2, 0)
                return img_array
            except Exception as e:
                print(f"Warning: Could not load image from dataset index {sample.get('sample_index')}: {e}")

        # Method 3: Load from ImageNet directory using image_path (fallback)
        if 'image_path' in sample and self.imagenet_dir:
            try:
                img_path = self.imagenet_dir / sample['image_path']
                if img_path.exists():
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = self.image_transform(img).numpy()
                    return img_tensor.transpose(1, 2, 0)
            except Exception as e:
                print(f"Warning: Could not load image from {sample.get('image_path')}: {e}")

        return None

    def visualize_class_patterns(self, class_id, save_path=None):
        """
        Visualize all K sub-patterns for a specific class.

        Args:
            class_id: Class index
            save_path: Optional path to save figure
        """
        class_name = self.class_names.get(class_id, f"Class {class_id}")

        # Get templates for this class
        templates = self.model.templates[class_id].cpu().numpy()  # (K, 1, H, W)
        counts = self.model.template_counts[class_id].cpu().numpy()  # (K,)

        # Create visualization
        fig, axes = plt.subplots(2, self.k_subpatterns // 2, figsize=(20, 8))
        axes = axes.flatten()

        fig.suptitle(f'Learned Sub-Patterns for {class_name} (Class {class_id})',
                     fontsize=16, fontweight='bold')

        for k in range(self.k_subpatterns):
            ax = axes[k]

            # Plot template with normalization
            template = templates[k, 0]  # (H, W)
            vmax = template.max() if template.max() > 0 else 1.0
            im = ax.imshow(template, cmap='hot', interpolation='bilinear', vmin=0, vmax=vmax)

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046)

            # Title with usage count and max value
            usage_pct = counts[k] / counts.sum() * 100 if counts.sum() > 0 else 0
            ax.set_title(f'Pattern {k} (max={vmax:.3f})\nUsage: {int(counts[k])} ({usage_pct:.1f}%)',
                        fontsize=9)
            ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved class patterns to {save_path}")

        plt.show()
        return fig

    def visualize_decomposition(self, class_id, sample_idx=0, save_path=None):
        """
        Visualize the decomposition of a sample IG map into patterns.

        Shows:
        1. Original IG saliency map
        2. Aligned IG map
        3. Cluster assignment probabilities
        4. Top-3 matching patterns
        5. Deformation field visualization

        Args:
            class_id: Class index
            sample_idx: Which sample from the class to visualize
            save_path: Optional path to save figure
        """
        class_name = self.class_names.get(class_id, f"Class {class_id}")

        # Sample images on-demand (load only what we need)
        samples = self.sample_class_images(class_id, num_samples=sample_idx + 1)

        if not samples:
            print(f"Error: No samples found for class {class_id}")
            return None

        if sample_idx >= len(samples):
            print(f"Error: Sample {sample_idx} not found (only {len(samples)} samples available)")
            print(f"Loading more samples...")
            samples = self.sample_class_images(class_id)
            if sample_idx >= len(samples):
                print(f"Error: Sample {sample_idx} not found (only {len(samples)} total samples)")
                return None

        sample = samples[sample_idx]
        saliency_map = sample['saliency_map']
        confidence = sample['confidence']

        # Try to get original image
        original_image = self.get_original_image(sample)
        has_original_image = original_image is not None

        # Convert to tensor
        h_i = torch.from_numpy(saliency_map).float().unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, H, W)

        # Forward pass
        with torch.no_grad():
            if self.is_advanced_model:
                # Advanced model returns: h_aligned, cluster_probs, phi_coarse, phi_fine, v_coarse, v_fine
                h_aligned, cluster_probs, phi_coarse, phi_fine, v_coarse, v_fine = self.model(
                    h_i, update_templates=False
                )
                # Use fine deformation for visualization (final result)
                phi = phi_fine
            else:
                # Simple model returns: h_aligned, cluster_probs, phi, v
                h_aligned, cluster_probs, phi, _ = self.model(h_i, update_templates=False)

            cluster_assigned = torch.argmax(cluster_probs, dim=-1).item()

        # Move to CPU for visualization
        h_i_np = h_i.squeeze().cpu().numpy()
        h_aligned_np = h_aligned.squeeze().cpu().numpy()
        cluster_probs_np = cluster_probs.squeeze().cpu().numpy()
        phi_np = phi.squeeze().cpu().numpy()  # (2, H, W)

        # Get templates
        templates = self.model.templates[class_id].cpu().numpy()  # (K, 1, H, W)

        # Create comprehensive visualization with extra column for original image
        num_cols = 6 if has_original_image else 5
        fig = plt.figure(figsize=(4*num_cols, 12))
        gs = fig.add_gridspec(3, num_cols, hspace=0.3, wspace=0.3)

        # Title
        fig.suptitle(f'Pattern Decomposition: {class_name} (Class {class_id})\n'
                    f'Sample {sample_idx} | Confidence: {confidence:.3f}',
                    fontsize=16, fontweight='bold')

        col_offset = 0

        # Row 1 Col 0: Original Image (if available)
        if has_original_image:
            ax0 = fig.add_subplot(gs[0, 0])
            ax0.imshow(original_image)
            ax0.set_title('Original Image', fontsize=12, fontweight='bold')
            ax0.axis('off')
            col_offset = 1

        # Row 1: Original IG, Aligned, Deformation
        ax1 = fig.add_subplot(gs[0, col_offset])
        im1 = ax1.imshow(h_i_np, cmap='hot', interpolation='bilinear')
        ax1.set_title('Original IG Map', fontsize=12, fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046)

        ax2 = fig.add_subplot(gs[0, col_offset+1])
        im2 = ax2.imshow(h_aligned_np, cmap='hot', interpolation='bilinear')
        ax2.set_title('Aligned IG Map', fontsize=12, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046)

        ax3 = fig.add_subplot(gs[0, col_offset+2])
        # Visualize deformation field magnitude
        deform_mag = np.sqrt(phi_np[0]**2 + phi_np[1]**2)
        im3 = ax3.imshow(deform_mag, cmap='viridis', interpolation='bilinear')
        ax3.set_title('Deformation Magnitude', fontsize=12, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046)

        # Deformation field arrows (downsampled)
        ax4 = fig.add_subplot(gs[0, col_offset+3:col_offset+5])
        ax4.imshow(h_i_np, cmap='gray', alpha=0.3, interpolation='bilinear')

        # Downsample for arrow visualization
        step = 16
        y_coords = np.arange(0, 224, step)
        x_coords = np.arange(0, 224, step)
        Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')

        U = phi_np[0, ::step, ::step] * 224 * 0.5  # Scale for visualization
        V = phi_np[1, ::step, ::step] * 224 * 0.5

        ax4.quiver(X, Y, U, V, color='red', alpha=0.7, scale=50)
        ax4.set_title('Deformation Field (arrows)', fontsize=12, fontweight='bold')
        ax4.axis('off')

        # Row 2: Cluster assignment probabilities
        ax5 = fig.add_subplot(gs[1, :3])
        bars = ax5.bar(range(self.k_subpatterns), cluster_probs_np,
                      color=['red' if i == cluster_assigned else 'blue'
                            for i in range(self.k_subpatterns)])
        ax5.set_xlabel('Pattern Index', fontsize=11)
        ax5.set_ylabel('Assignment Probability', fontsize=11)
        ax5.set_title(f'Cluster Assignment (Assigned to Pattern {cluster_assigned})',
                     fontsize=12, fontweight='bold')
        ax5.set_xticks(range(self.k_subpatterns))
        ax5.grid(axis='y', alpha=0.3)

        # Add percentage labels
        for i, (bar, prob) in enumerate(zip(bars, cluster_probs_np)):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{prob*100:.1f}%',
                    ha='center', va='bottom', fontsize=9)

        # Row 2-3: Top-3 matching patterns
        top_k_indices = np.argsort(cluster_probs_np)[::-1][:3]

        for i, k in enumerate(top_k_indices):
            # Pattern template
            ax = fig.add_subplot(gs[1, 3+i]) if i < 2 else fig.add_subplot(gs[2, 0])
            template = templates[k, 0]
            # Normalize template for visualization
            vmax = template.max() if template.max() > 0 else 1.0
            im = ax.imshow(template, cmap='hot', interpolation='bilinear', vmin=0, vmax=vmax)
            ax.set_title(f'Pattern {k}\n({cluster_probs_np[k]*100:.1f}%) max={vmax:.3f}',
                        fontsize=9, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

        # Row 3: Difference maps
        for i, k in enumerate(top_k_indices[:3]):
            ax = fig.add_subplot(gs[2, 1+i])
            diff = np.abs(h_aligned_np - templates[k, 0])
            im = ax.imshow(diff, cmap='RdYlGn_r', interpolation='bilinear')
            ax.set_title(f'|Aligned - Pattern {k}|', fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

        # Reconstruction error
        ax_error = fig.add_subplot(gs[2, 4])
        weighted_template = sum(cluster_probs_np[k] * templates[k, 0] for k in range(self.k_subpatterns))
        error = np.abs(h_aligned_np - weighted_template)
        im_error = ax_error.imshow(error, cmap='RdYlGn_r', interpolation='bilinear')
        ax_error.set_title(f'Reconstruction Error\nMSE: {np.mean(error**2):.6f}', fontsize=10)
        ax_error.axis('off')
        plt.colorbar(im_error, ax=ax_error, fraction=0.046)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved decomposition to {save_path}")

        plt.show()
        return fig

    def compare_samples(self, class_id, num_samples=6, save_path=None):
        """
        Compare multiple samples from the same class.

        Shows how different samples are assigned to different patterns.

        Args:
            class_id: Class index
            num_samples: Number of samples to compare
            save_path: Optional path to save figure
        """
        class_name = self.class_names.get(class_id, f"Class {class_id}")

        # Sample images on-demand (load only what we need)
        samples = self.sample_class_images(class_id, num_samples=num_samples)

        if not samples:
            print(f"Error: No samples found for class {class_id}")
            return None

        num_samples = min(num_samples, len(samples))

        # Check if original images are available
        first_sample_image = self.get_original_image(samples[0])
        has_original_images = first_sample_image is not None
        num_cols = 5 if has_original_images else 4

        fig, axes = plt.subplots(num_samples, num_cols, figsize=(4*num_cols, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(f'Sample Comparison: {class_name} (Class {class_id})',
                    fontsize=16, fontweight='bold')

        for i in range(num_samples):
            sample = samples[i]
            saliency_map = sample['saliency_map']

            # Convert to tensor
            h_i = torch.from_numpy(saliency_map).float().unsqueeze(0).unsqueeze(0).to(self.device)

            # Forward pass
            with torch.no_grad():
                if self.is_advanced_model:
                    # Advanced model returns: h_aligned, cluster_probs, phi_coarse, phi_fine, v_coarse, v_fine
                    h_aligned, cluster_probs, _, _, _, _ = self.model(h_i, update_templates=False)
                else:
                    # Simple model returns: h_aligned, cluster_probs, phi, v
                    h_aligned, cluster_probs, _, _ = self.model(h_i, update_templates=False)
                cluster_assigned = torch.argmax(cluster_probs, dim=-1).item()

            col = 0

            # Original Image (if available)
            if has_original_images:
                original_img = self.get_original_image(sample)
                if original_img is not None:
                    axes[i, col].imshow(original_img)
                    axes[i, col].set_title(f'Sample {i}: Original', fontsize=10)
                    axes[i, col].axis('off')
                col += 1

            # Original IG
            axes[i, col].imshow(saliency_map, cmap='hot', interpolation='bilinear')
            axes[i, col].set_title(f'Original IG', fontsize=10)
            axes[i, col].axis('off')
            col += 1

            # Aligned IG
            axes[i, col].imshow(h_aligned.squeeze().cpu().numpy(), cmap='hot', interpolation='bilinear')
            axes[i, col].set_title(f'Aligned (→ Pattern {cluster_assigned})', fontsize=10)
            axes[i, col].axis('off')
            col += 1

            # Assigned pattern
            template = self.model.templates[class_id, cluster_assigned].squeeze().cpu().numpy()
            # Normalize template for better visualization (vmin=0, vmax=template's max)
            vmax = template.max() if template.max() > 0 else 1.0
            axes[i, col].imshow(template, cmap='hot', interpolation='bilinear', vmin=0, vmax=vmax)
            axes[i, col].set_title(f'Pattern {cluster_assigned} Template\n(max={vmax:.3f})', fontsize=9)
            axes[i, col].axis('off')
            col += 1

            # Cluster probabilities
            cluster_probs_np = cluster_probs.squeeze().cpu().numpy()
            axes[i, col].bar(range(self.k_subpatterns), cluster_probs_np,
                          color=['red' if k == cluster_assigned else 'blue'
                                for k in range(self.k_subpatterns)])
            axes[i, col].set_title(f'Assignment Probs', fontsize=10)
            axes[i, col].set_ylim(0, 1)
            axes[i, col].grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved comparison to {save_path}")

        plt.show()
        return fig


def main():
    parser = argparse.ArgumentParser(
        description='Visualize learned patterns from LDDMM model'
    )
    parser.add_argument('--checkpoint', type=str, default='checkpoints/lddmm_model_final.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str,
                       default='./data/saliency_imagenet1k_resnet50_100',
                       help='Directory with saliency maps')
    parser.add_argument('--class_id', type=int, default=0,
                       help='Class ID to visualize')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Sample index to visualize for decomposition')
    parser.add_argument('--num_samples', type=int, default=6,
                       help='Number of samples for comparison')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                       help='Output directory for saved figures')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--imagenet_dir', type=str, default=None,
                       help='Path to ImageNet directory (to load original images if not in saliency data)')
    parser.add_argument('--images_per_class', type=int, default=100,
                       help='Number of images per class (must match saliency generation)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sample selection (default: 42)')

    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create visualizer
    print(f"\n{'='*80}")
    print(f"Pattern Visualization")
    print(f"{'='*80}")

    visualizer = PatternVisualizer(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        device=args.device,
        imagenet_dir=args.imagenet_dir,
        images_per_class=args.images_per_class
    )

    # Visualize class patterns
    print(f"\n{'='*80}")
    print(f"Visualizing Class {args.class_id} Patterns")
    print(f"{'='*80}")

    patterns_path = output_dir / f"class_{args.class_id}_patterns.png"
    visualizer.visualize_class_patterns(
        class_id=args.class_id,
        save_path=patterns_path
    )

    # Visualize decomposition
    print(f"\n{'='*80}")
    print(f"Visualizing Sample Decomposition")
    print(f"{'='*80}")

    decomp_path = output_dir / f"class_{args.class_id}_decomposition_sample_{args.sample_idx}.png"
    visualizer.visualize_decomposition(
        class_id=args.class_id,
        sample_idx=args.sample_idx,
        save_path=decomp_path
    )

    # Compare multiple samples
    print(f"\n{'='*80}")
    print(f"Comparing Multiple Samples")
    print(f"{'='*80}")

    comparison_path = output_dir / f"class_{args.class_id}_comparison.png"
    visualizer.compare_samples(
        class_id=args.class_id,
        num_samples=args.num_samples,
        save_path=comparison_path
    )

    print(f"\n{'='*80}")
    print(f"✓ All visualizations saved to {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
