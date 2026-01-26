"""
Template Matching Visualization for Geodesic LDDMM Model

This script visualizes the mechanical template matching process:
1. Original IG saliency map
2. Grid deformation showing diffeomorphic warping
3. Aligned IG map after LDDMM registration
4. Best matching template
5. Cluster assignment probabilities

This provides visual proof that the model performs valid topological alignment.

Usage:
    # Visualize template matching for class 0, sample 0
    python visualize_template_matching.py --checkpoint checkpoints/lddmm_model_final.pth --class_id 0 --sample_idx 0

    # Visualize with custom grid spacing
    python visualize_template_matching.py --checkpoint checkpoints/model.pth \
                                          --class_id 281 --sample_idx 5 \
                                          --grid_spacing 20
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import argparse
import joblib
from torchvision import transforms

# Import model components
import sys
sys.path.append('.')
from geo_patterns import LDDMM_GlobalPatternPipeline
from misc.geo_x import AdvancedLDDMM_Pipeline
from full_classes import IMAGENET2012_CLASSES
from misc.saliency_map_export import ImageNet1kSaliencyDataset, IMAGENET_RAW_DIR, IMAGENET_SAMPLED_DIR


class TemplateMatchingVisualizer:
    """Visualize the template matching and deformation process."""

    def __init__(self, checkpoint_path, data_dir, device='cuda', images_per_class=100):
        self.device = device
        self.data_dir = Path(data_dir)

        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract configuration
        self.num_classes = checkpoint['templates'].shape[0]
        self.k_subpatterns = checkpoint['templates'].shape[1]

        # Auto-detect velocity field resolution and model type
        v_head_keys = [k for k in checkpoint['model_state_dict'].keys()
                       if 'predictor.fine_v_head' in k and 'weight' in k and 'Linear' not in k]

        if not v_head_keys:
            v_head_keys = [k for k in checkpoint['model_state_dict'].keys()
                           if 'predictor.v_head' in k and 'weight' in k and 'Linear' not in k]

        if not v_head_keys:
            v_head_keys = [k for k in checkpoint['model_state_dict'].keys()
                           if 'predictor.coarse_v_head' in k and 'weight' in k and 'Linear' not in k]

        if not v_head_keys:
            raise ValueError("Could not find velocity head in checkpoint.")

        v_head_keys.sort()
        v_head_key = v_head_keys[-1]

        v_head_weight_shape = checkpoint['model_state_dict'][v_head_key].shape
        v_dim = v_head_weight_shape[0]
        v_res_size = int(np.sqrt(v_dim / 2))
        v_res = (v_res_size, v_res_size)

        # Detect model type
        has_fine_v_head = any('fine_v_head' in k for k in checkpoint['model_state_dict'].keys())
        has_coarse_v_head = any('coarse_v_head' in k for k in checkpoint['model_state_dict'].keys())
        is_advanced_model = has_fine_v_head or has_coarse_v_head

        print(f"Loaded model:")
        print(f"  Type: {'Advanced (Multi-Scale)' if is_advanced_model else 'Simple (Single-Scale)'}")
        print(f"  Classes: {self.num_classes}")
        print(f"  Sub-patterns per class: {self.k_subpatterns}")
        print(f"  Velocity field resolution: {v_res}")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")

        # Create model
        self.is_advanced_model = is_advanced_model
        if is_advanced_model:
            self.model = AdvancedLDDMM_Pipeline(
                num_classes=self.num_classes,
                k_subpatterns=self.k_subpatterns,
                img_res=(224, 224),
                device=device,
                use_ot=True,
                use_edge_gating=False
            ).to(device)
        else:
            self.model = LDDMM_GlobalPatternPipeline(
                num_classes=self.num_classes,
                k_subpatterns=self.k_subpatterns,
                img_res=(224, 224),
                device=device,
                v_res=v_res
            ).to(device)

        # Load weights
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            print("  Detected torch.compile() checkpoint - stripping _orig_mod. prefix...")
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict)
        self.model.eval()

        print(f"✓ Model loaded successfully!")

        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        # Transform for loading ImageNet images
        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        # Load ImageNet dataset
        print(f"\nLoading ImageNet dataset...")
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
            self.imagenet_dataset = None

        # Load class index
        self.saliency_dir = self.data_dir / "saliency_maps"
        self.cache_file = self.data_dir / "class_index_cache.pkl"
        print(f"\nLoading class index...")
        self.class_index = self._load_index_cache()
        print(f"✓ Class index ready!")

        # Get class names
        self.class_names = {idx: name for idx, name in enumerate(IMAGENET2012_CLASSES.values())}

    def _load_index_cache(self):
        """Load the class index from cache file."""
        if not self.cache_file.exists():
            raise RuntimeError(f"Class index cache not found at {self.cache_file}. "
                             f"Please run visualize_patterns.py first to build the cache.")
        class_index = joblib.load(self.cache_file)
        return class_index

    def sample_class_images(self, class_id, num_samples=None):
        """Sample images from a specific class."""
        if class_id not in self.class_index:
            print(f"Warning: Class {class_id} not found in index")
            return []

        class_locations = self.class_index[class_id]
        if num_samples is not None:
            class_locations = class_locations[:num_samples]

        from collections import defaultdict
        batch_to_indices = defaultdict(list)

        for batch_file, item_idx in class_locations:
            batch_to_indices[batch_file].append(item_idx)

        class_samples = []
        loaded_items = {}

        for batch_file, item_indices in batch_to_indices.items():
            if batch_file not in loaded_items:
                batch_data = joblib.load(batch_file)
                loaded_items[batch_file] = batch_data

            for item_idx in item_indices:
                item = loaded_items[batch_file][item_idx]
                sample_dict = {
                    'sample_index': item.get('sample_index', item.get('index')),
                    'confidence': item['confidence'],
                    'saliency_map': item['saliency_map']
                }
                class_samples.append(sample_dict)

        return class_samples

    def get_original_image(self, sample):
        """
        Get original image from sample.

        Args:
            sample: Sample dictionary

        Returns:
            (H, W, 3) RGB image in [0, 1] range, or None if not available
        """
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

        return None

    def plot_grid_deformation(self, phi, img_res=(224, 224), grid_spacing=16, title="Grid Deformation"):
        """
        Visualizes the mechanical deformation by warping a uniform grid.

        Args:
            phi: (1, 2, H, W) deformation field tensor
            img_res: Resolution of the grid
            grid_spacing: Number of pixels between grid lines
            title: Plot title

        Returns:
            Figure with warped grid
        """
        device = phi.device

        # Create a canonical identity grid [-1, 1]
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, img_res[0], device=device),
            torch.linspace(-1, 1, img_res[1], device=device),
            indexing='ij'
        )
        identity = torch.stack((x, y), dim=-1).unsqueeze(0)  # (1, H, W, 2)

        # Create a white image with a black grid
        grid_img = torch.ones((1, 1, *img_res), device=device)
        for i in range(0, img_res[0], grid_spacing):
            grid_img[:, :, i, :] = 0
        for j in range(0, img_res[1], grid_spacing):
            grid_img[:, :, :, j] = 0

        # Combine displacement fields (Phi = Identity + displacement)
        # Convert phi from (B, 2, H, W) to (B, H, W, 2)
        phi_hwc = phi.permute(0, 2, 3, 1)
        warped_grid_coords = identity + phi_hwc

        # Warp the grid image
        warped_grid = F.grid_sample(
            grid_img, warped_grid_coords,
            mode='bilinear', padding_mode='border', align_corners=True
        )

        # Plot
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(warped_grid.squeeze().cpu().numpy(), cmap='gray')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

        return fig

    def visualize_template_matching(self, class_id, sample_idx=0, grid_spacing=16, save_path=None):
        """
        Visualize the complete template matching process for a sample.

        Shows:
        1. Original IG map
        2. Grid deformation (coarse, fine, combined)
        3. Aligned IG map
        4. Best matching template
        5. Cluster assignment probabilities

        Args:
            class_id: Class index
            sample_idx: Which sample from the class to visualize
            grid_spacing: Spacing between grid lines for deformation visualization
            save_path: Optional path to save figure
        """
        class_name = self.class_names.get(class_id, f"Class {class_id}")

        # Sample images
        samples = self.sample_class_images(class_id, num_samples=sample_idx + 1)

        if not samples or sample_idx >= len(samples):
            print(f"Error: Sample {sample_idx} not found for class {class_id}")
            return None

        sample = samples[sample_idx]
        saliency_map = sample['saliency_map']
        confidence = sample['confidence']

        # Get original image
        original_image = self.get_original_image(sample)
        has_original_image = original_image is not None

        # Convert to tensor
        h_i = torch.from_numpy(saliency_map).float().unsqueeze(0).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            if self.is_advanced_model:
                h_aligned, cluster_probs, phi_coarse, phi_fine, _, _ = self.model(
                    h_i, update_templates=False
                )
                cluster_assigned = torch.argmax(cluster_probs, dim=-1).item()
            else:
                h_aligned, cluster_probs, phi, _ = self.model(h_i, update_templates=False)
                cluster_assigned = torch.argmax(cluster_probs, dim=-1).item()
                # For simple model, treat as single-scale
                phi_coarse = phi * 0  # No coarse component
                phi_fine = phi

        # Get data for visualization
        h_i_np = h_i.squeeze().cpu().numpy()
        h_aligned_np = h_aligned.squeeze().cpu().numpy()
        cluster_probs_np = cluster_probs.squeeze().cpu().numpy()
        templates = self.model.templates[class_id].cpu().numpy()
        matched_template = templates[cluster_assigned, 0]

        # Create comprehensive figure
        # Adjust layout based on whether we have original image
        num_cols = 6 if has_original_image else 5
        fig = plt.figure(figsize=(4*num_cols, 12))
        gs = GridSpec(3, num_cols, figure=fig, hspace=0.3, wspace=0.3)

        # Title
        fig.suptitle(f'Template Matching Visualization: {class_name} (Class {class_id})\n'
                    f'Sample {sample_idx} | Matched to Pattern {cluster_assigned} | Confidence: {confidence:.3f}',
                    fontsize=16, fontweight='bold')

        col_offset = 0

        # Original Image (if available)
        if has_original_image:
            ax0 = fig.add_subplot(gs[0, 0])
            ax0.imshow(original_image)
            ax0.set_title('Original Image', fontsize=11, fontweight='bold')
            ax0.axis('off')
            col_offset = 1

        # Row 1: Original IG, Deformations, Aligned
        ax1 = fig.add_subplot(gs[0, col_offset])
        im1 = ax1.imshow(h_i_np, cmap='hot', interpolation='bilinear')
        ax1.set_title('Original IG Map', fontsize=11, fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046)

        # Grid deformations
        if self.is_advanced_model:
            # Coarse deformation
            ax2 = fig.add_subplot(gs[0, col_offset+1])
            grid_coarse = self._create_grid_deformation_image(phi_coarse, grid_spacing)
            ax2.imshow(grid_coarse, cmap='gray')
            ax2.set_title('Coarse Deformation\n(Global Alignment)', fontsize=10, fontweight='bold')
            ax2.axis('off')

            # Fine deformation
            ax3 = fig.add_subplot(gs[0, col_offset+2])
            grid_fine = self._create_grid_deformation_image(phi_fine, grid_spacing)
            ax3.imshow(grid_fine, cmap='gray')
            ax3.set_title('Fine Deformation\n(Local Refinement)', fontsize=10, fontweight='bold')
            ax3.axis('off')

            # Combined deformation
            ax4 = fig.add_subplot(gs[0, col_offset+3])
            phi_combined = phi_coarse + phi_fine
            grid_combined = self._create_grid_deformation_image(phi_combined, grid_spacing)
            ax4.imshow(grid_combined, cmap='gray')
            ax4.set_title('Combined Deformation\n(Total Warping)', fontsize=10, fontweight='bold')
            ax4.axis('off')
        else:
            # Single-scale deformation
            ax_single = fig.add_subplot(gs[0, col_offset+1:col_offset+4])
            grid_single = self._create_grid_deformation_image(phi_fine, grid_spacing)
            ax_single.imshow(grid_single, cmap='gray')
            ax_single.set_title('Deformation Field\n(Single-Scale)', fontsize=11, fontweight='bold')
            ax_single.axis('off')

        # Aligned result
        ax5 = fig.add_subplot(gs[0, col_offset+4])
        im5 = ax5.imshow(h_aligned_np, cmap='hot', interpolation='bilinear')
        ax5.set_title('Aligned IG Map', fontsize=11, fontweight='bold')
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5, fraction=0.046)

        # Row 2: Deformation field details
        # Deformation magnitude
        phi_total = phi_coarse + phi_fine
        deform_mag = torch.sqrt(phi_total[:, 0]**2 + phi_total[:, 1]**2).squeeze().cpu().numpy()

        # Show original image with saliency overlay if available
        row2_start = 0
        if has_original_image:
            ax_overlay = fig.add_subplot(gs[1, 0])
            # Create overlay: original image with saliency heatmap
            ax_overlay.imshow(original_image)
            ax_overlay.imshow(h_i_np, cmap='hot', alpha=0.5, interpolation='bilinear')
            ax_overlay.set_title('Image + Saliency\nOverlay', fontsize=11, fontweight='bold')
            ax_overlay.axis('off')
            row2_start = 1

        ax6 = fig.add_subplot(gs[1, row2_start])
        im6 = ax6.imshow(deform_mag, cmap='viridis', interpolation='bilinear')
        ax6.set_title('Deformation Magnitude', fontsize=11, fontweight='bold')
        ax6.axis('off')
        plt.colorbar(im6, ax=ax6, fraction=0.046)

        # Deformation direction (quiver plot)
        ax7 = fig.add_subplot(gs[1, row2_start+1:row2_start+3])
        ax7.imshow(h_i_np, cmap='gray', alpha=0.3, interpolation='bilinear')

        # Downsample for arrow visualization
        step = 16
        y_coords = np.arange(0, 224, step)
        x_coords = np.arange(0, 224, step)
        Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')

        phi_np = phi_total.squeeze().cpu().numpy()
        U = phi_np[0, ::step, ::step] * 224 * 0.5
        V = phi_np[1, ::step, ::step] * 224 * 0.5

        ax7.quiver(X, Y, U, V, color='red', alpha=0.7, scale=50)
        ax7.set_title('Deformation Field (Vector Field)', fontsize=11, fontweight='bold')
        ax7.axis('off')

        # Matched template
        ax8 = fig.add_subplot(gs[1, row2_start+3])
        im8 = ax8.imshow(matched_template, cmap='hot', interpolation='bilinear')
        ax8.set_title(f'Matched Template\n(Pattern {cluster_assigned})', fontsize=11, fontweight='bold')
        ax8.axis('off')
        plt.colorbar(im8, ax=ax8, fraction=0.046)

        # Difference map
        ax9 = fig.add_subplot(gs[1, row2_start+4])
        diff = np.abs(h_aligned_np - matched_template)
        im9 = ax9.imshow(diff, cmap='RdYlGn_r', interpolation='bilinear')
        ax9.set_title(f'Alignment Error\nMSE: {np.mean(diff**2):.6f}', fontsize=11, fontweight='bold')
        ax9.axis('off')
        plt.colorbar(im9, ax=ax9, fraction=0.046)

        # Row 3: All templates and cluster assignment
        # Cluster probabilities
        ax10 = fig.add_subplot(gs[2, 0:3])
        bars = ax10.bar(range(self.k_subpatterns), cluster_probs_np,
                       color=['red' if i == cluster_assigned else 'blue'
                             for i in range(self.k_subpatterns)])
        ax10.set_xlabel('Pattern Index', fontsize=11)
        ax10.set_ylabel('Assignment Probability', fontsize=11)
        ax10.set_title(f'Cluster Assignment (Matched to Pattern {cluster_assigned})',
                      fontsize=11, fontweight='bold')
        ax10.set_xticks(range(self.k_subpatterns))
        ax10.grid(axis='y', alpha=0.3)

        for i, (bar, prob) in enumerate(zip(bars, cluster_probs_np)):
            height = bar.get_height()
            ax10.text(bar.get_x() + bar.get_width()/2., height,
                     f'{prob*100:.1f}%',
                     ha='center', va='bottom', fontsize=9)

        # Top-3 templates
        top_k_indices = np.argsort(cluster_probs_np)[::-1][:2]

        for i, k in enumerate(top_k_indices):
            ax = fig.add_subplot(gs[2, 3+i])
            template = templates[k, 0]
            im = ax.imshow(template, cmap='hot', interpolation='bilinear')
            ax.set_title(f'Pattern {k}\n({cluster_probs_np[k]*100:.1f}%)',
                        fontsize=10, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved visualization to {save_path}")

        plt.show()
        return fig

    def _create_grid_deformation_image(self, phi, grid_spacing):
        """Helper to create grid deformation image."""
        device = phi.device
        img_res = (224, 224)

        # Create identity grid
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, img_res[0], device=device),
            torch.linspace(-1, 1, img_res[1], device=device),
            indexing='ij'
        )
        identity = torch.stack((x, y), dim=-1).unsqueeze(0)

        # Create grid image
        grid_img = torch.ones((1, 1, *img_res), device=device)
        for i in range(0, img_res[0], grid_spacing):
            grid_img[:, :, i, :] = 0
        for j in range(0, img_res[1], grid_spacing):
            grid_img[:, :, :, j] = 0

        # Warp grid
        phi_hwc = phi.permute(0, 2, 3, 1)
        warped_coords = identity + phi_hwc
        warped_grid = F.grid_sample(
            grid_img, warped_coords,
            mode='bilinear', padding_mode='border', align_corners=True
        )

        return warped_grid.squeeze().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize template matching and deformation for LDDMM model'
    )
    parser.add_argument('--checkpoint', type=str, default='checkpoints/lddmm_model_final.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str,
                       default='./data/saliency_imagenet1k_resnet50_100',
                       help='Directory with saliency maps')
    parser.add_argument('--class_id', type=int, default=0,
                       help='Class ID to visualize')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Sample index to visualize')
    parser.add_argument('--grid_spacing', type=int, default=16,
                       help='Spacing between grid lines (default: 16)')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                       help='Output directory for saved figures')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--images_per_class', type=int, default=100,
                       help='Number of images per class (must match saliency generation)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create visualizer
    print(f"\n{'='*80}")
    print(f"Template Matching Visualization")
    print(f"{'='*80}")

    visualizer = TemplateMatchingVisualizer(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        device=args.device,
        images_per_class=args.images_per_class
    )

    # Visualize template matching
    print(f"\n{'='*80}")
    print(f"Visualizing Template Matching for Class {args.class_id}, Sample {args.sample_idx}")
    print(f"{'='*80}")

    save_path = output_dir / f"template_matching_class_{args.class_id}_sample_{args.sample_idx}.png"
    visualizer.visualize_template_matching(
        class_id=args.class_id,
        sample_idx=args.sample_idx,
        grid_spacing=args.grid_spacing,
        save_path=save_path
    )

    print(f"\n{'='*80}")
    print(f"✓ Visualization saved to {save_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
