"""
Multi-Sample Geodesic Visualization for Research Papers

This script creates publication-ready visualizations showing:
1. Anatomical Landmark Overlay - Pattern templates overlaid on canonical images
2. Multi-Sample Feature Flow - How different poses align to the same template
3. Geodesic Path Demonstration - The warping paths taken by the model

These visualizations bridge the gap between abstract patterns and biological structures,
making it clear that patterns represent anatomical landmarks rather than random noise.

Usage:
    # Create anatomical landmark overlay for a class
    python visualize_multi_geos.py --checkpoint checkpoints/model.pth --class_id 0 --mode anatomical

    # Create multi-sample flow visualization
    python visualize_multi_geos.py --checkpoint checkpoints/model.pth --class_id 9 --mode flow --num_samples 6

    # Create both visualizations
    python visualize_multi_geos.py --checkpoint checkpoints/model.pth --class_id 0 --mode all
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


class MultiGeoVisualizer:
    """Create publication-quality visualizations of geodesic pattern alignment."""

    def __init__(self, checkpoint_path, data_dir, device='cuda', images_per_class=100):
        self.device = device
        self.data_dir = Path(data_dir)

        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract configuration
        self.num_classes = checkpoint['templates'].shape[0]
        self.k_subpatterns = checkpoint['templates'].shape[1]

        # Auto-detect model architecture
        v_head_keys = [k for k in checkpoint['model_state_dict'].keys()
                       if 'predictor.fine_v_head' in k and 'weight' in k and 'Linear' not in k]
        if not v_head_keys:
            v_head_keys = [k for k in checkpoint['model_state_dict'].keys()
                           if 'predictor.v_head' in k and 'weight' in k and 'Linear' not in k]
        if not v_head_keys:
            v_head_keys = [k for k in checkpoint['model_state_dict'].keys()
                           if 'predictor.coarse_v_head' in k and 'weight' in k and 'Linear' not in k]

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

        print(f"Model: {'Advanced (Multi-Scale)' if is_advanced_model else 'Simple (Single-Scale)'}")
        print(f"Classes: {self.num_classes}, Patterns per class: {self.k_subpatterns}")

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
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        # Load dataset
        print(f"Loading ImageNet dataset...")
        self.imagenet_dataset = ImageNet1kSaliencyDataset(
            raw_dir=IMAGENET_RAW_DIR,
            sampled_dir=IMAGENET_SAMPLED_DIR,
            output_dir=self.data_dir,
            images_per_class=images_per_class,
            force_resample=False
        )

        # Load class index
        self.cache_file = self.data_dir / "class_index_cache.pkl"
        self.class_index = joblib.load(self.cache_file)
        self.class_names = {idx: name for idx, name in enumerate(IMAGENET2012_CLASSES.values())}

        print(f"✓ Loaded successfully!")

    def sample_class_images(self, class_id, num_samples=None):
        """Sample images from a specific class."""
        if class_id not in self.class_index:
            return []

        class_locations = self.class_index[class_id]
        if num_samples is not None:
            class_locations = class_locations[:num_samples]

        from collections import defaultdict
        batch_to_indices = defaultdict(list)

        for batch_file, item_idx in class_locations:
            batch_to_indices[batch_file].append(item_idx)

        class_samples = []
        for batch_file, item_indices in batch_to_indices.items():
            batch_data = joblib.load(batch_file)
            for item_idx in item_indices:
                item = batch_data[item_idx]
                sample_dict = {
                    'sample_index': item.get('sample_index', item.get('index')),
                    'confidence': item['confidence'],
                    'saliency_map': item['saliency_map']
                }
                class_samples.append(sample_dict)

        return class_samples

    def get_original_image(self, sample_idx):
        """Load original image from dataset."""
        try:
            image_pil, _ = self.imagenet_dataset[sample_idx]
            img_tensor = self.image_transform(image_pil).numpy()
            return img_tensor.transpose(1, 2, 0)
        except:
            return None

    def visualize_anatomical_landmarks(self, class_id, pattern_id=None, save_path=None):
        """
        Strategy 1: Anatomical Landmark Overlay

        Shows pattern templates overlaid on canonical (high-confidence) images
        to demonstrate that patterns align with biological structures.
        """
        class_name = self.class_names.get(class_id, f"Class {class_id}")

        # Get templates and counts
        templates = self.model.templates[class_id].cpu().numpy()
        counts = self.model.template_counts[class_id].cpu().numpy()
        usage_percentages = (counts / counts.sum() * 100) if counts.sum() > 0 else np.zeros_like(counts)

        # Determine which patterns to visualize
        if pattern_id is not None:
            pattern_ids = [pattern_id]
        else:
            # Use top 4 most-used patterns
            pattern_ids = np.argsort(counts)[::-1][:4]
            pattern_ids = [p for p in pattern_ids if usage_percentages[p] > 0]

        if not pattern_ids:
            print("No valid patterns to visualize")
            return None

        # For each pattern, find the best matching sample
        samples = self.sample_class_images(class_id, num_samples=50)

        pattern_best_matches = {}
        for pid in pattern_ids:
            best_match = None
            best_prob = 0

            for sample in samples:
                h_i = torch.from_numpy(sample['saliency_map']).float().unsqueeze(0).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    if self.is_advanced_model:
                        _, cluster_probs, _, _, _, _ = self.model(h_i, update_templates=False)
                    else:
                        _, cluster_probs, _, _ = self.model(h_i, update_templates=False)

                    prob = cluster_probs[0, pid].item()
                    if prob > best_prob:
                        best_prob = prob
                        best_match = sample

            if best_match:
                pattern_best_matches[pid] = (best_match, best_prob)

        # Create visualization
        num_patterns = len(pattern_best_matches)
        fig, axes = plt.subplots(num_patterns, 4, figsize=(16, 4*num_patterns))
        if num_patterns == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(f'Anatomical Landmark Overlay: {class_name} (Class {class_id})\n'
                    f'Pattern Templates Aligned to Biological Structures',
                    fontsize=16, fontweight='bold')

        for idx, (pid, (sample, prob)) in enumerate(pattern_best_matches.items()):
            # Get original image
            orig_img = self.get_original_image(sample['sample_index'])
            saliency = sample['saliency_map']
            template = templates[pid, 0]

            # Column 1: Original Image
            axes[idx, 0].imshow(orig_img)
            axes[idx, 0].set_title(f'Pattern {pid}\nCanonical Image\n(Match: {prob*100:.1f}%)',
                                  fontsize=10, fontweight='bold')
            axes[idx, 0].axis('off')

            # Column 2: Saliency Map
            axes[idx, 1].imshow(saliency, cmap='hot', interpolation='bilinear')
            axes[idx, 1].set_title('Original IG\nSaliency Map', fontsize=10, fontweight='bold')
            axes[idx, 1].axis('off')

            # Column 3: Pattern Template Overlay on Image
            axes[idx, 2].imshow(orig_img)
            # Normalize template for visualization
            template_norm = (template - template.min()) / (template.max() - template.min() + 1e-8)
            axes[idx, 2].imshow(template_norm, cmap='cool', alpha=0.6, interpolation='bilinear')
            axes[idx, 2].set_title('Anatomical Overlay\n(Template on Image)',
                                  fontsize=10, fontweight='bold')
            axes[idx, 2].axis('off')

            # Column 4: Pattern Template
            im = axes[idx, 3].imshow(template, cmap='hot', interpolation='bilinear')
            axes[idx, 3].set_title(f'Pattern Template\n(Usage: {usage_percentages[pid]:.1f}%)',
                                  fontsize=10, fontweight='bold')
            axes[idx, 3].axis('off')
            plt.colorbar(im, ax=axes[idx, 3], fraction=0.046)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved anatomical landmarks to {save_path}")

        plt.show()
        return fig

    def visualize_multi_sample_flow(self, class_id, pattern_id=None, num_samples=6, save_path=None):
        """
        Strategy 2: Multi-Sample Feature Flow

        Shows how different poses/variations get warped into the same template,
        demonstrating the geodesic alignment process.
        """
        class_name = self.class_names.get(class_id, f"Class {class_id}")

        # Get templates
        templates = self.model.templates[class_id].cpu().numpy()
        counts = self.model.template_counts[class_id].cpu().numpy()

        # Select pattern
        if pattern_id is None:
            pattern_id = np.argmax(counts)  # Most used pattern

        template = templates[pattern_id, 0]
        usage_pct = counts[pattern_id] / counts.sum() * 100 if counts.sum() > 0 else 0

        # Sample diverse images that use this pattern
        all_samples = self.sample_class_images(class_id, num_samples=100)

        pattern_samples = []
        for sample in all_samples:
            h_i = torch.from_numpy(sample['saliency_map']).float().unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                if self.is_advanced_model:
                    h_aligned, cluster_probs, phi_coarse, phi_fine, _, _ = self.model(h_i, update_templates=False)
                    phi_total = phi_coarse + phi_fine
                else:
                    h_aligned, cluster_probs, phi, _ = self.model(h_i, update_templates=False)
                    phi_total = phi

                assigned = torch.argmax(cluster_probs, dim=-1).item()
                if assigned == pattern_id:
                    pattern_samples.append({
                        'sample': sample,
                        'h_aligned': h_aligned.cpu(),
                        'phi': phi_total.cpu(),
                        'prob': cluster_probs[0, pattern_id].item()
                    })

        # Select diverse samples (by deformation magnitude)
        if len(pattern_samples) > num_samples:
            # Sort by deformation magnitude for diversity
            for ps in pattern_samples:
                deform_mag = torch.sqrt(ps['phi'][:, 0]**2 + ps['phi'][:, 1]**2).mean().item()
                ps['deform_mag'] = deform_mag
            pattern_samples.sort(key=lambda x: x['deform_mag'])
            # Take samples spread across the range
            indices = np.linspace(0, len(pattern_samples)-1, num_samples, dtype=int)
            pattern_samples = [pattern_samples[i] for i in indices]

        num_samples = len(pattern_samples)
        if num_samples == 0:
            print(f"No samples found using pattern {pattern_id}")
            return None

        # Create visualization
        fig = plt.figure(figsize=(20, 4*num_samples))
        gs = GridSpec(num_samples, 5, figure=fig, hspace=0.4, wspace=0.3)

        fig.suptitle(f'Multi-Sample Geodesic Flow: {class_name} (Class {class_id})\n'
                    f'Pattern {pattern_id} (Usage: {usage_pct:.1f}%) - Demonstrating Pose Invariance',
                    fontsize=16, fontweight='bold')

        for i, ps in enumerate(pattern_samples):
            sample = ps['sample']
            orig_img = self.get_original_image(sample['sample_index'])
            saliency = sample['saliency_map']
            h_aligned = ps['h_aligned'].squeeze().numpy()
            phi = ps['phi']

            # Column 1: Original Image
            ax1 = fig.add_subplot(gs[i, 0])
            if orig_img is not None:
                ax1.imshow(orig_img)
            ax1.set_title(f'Sample {i+1}\nOriginal Image', fontsize=9, fontweight='bold')
            ax1.axis('off')

            # Column 2: Original IG
            ax2 = fig.add_subplot(gs[i, 1])
            ax2.imshow(saliency, cmap='hot', interpolation='bilinear')
            ax2.set_title('Original IG\n(Before Alignment)', fontsize=9, fontweight='bold')
            ax2.axis('off')

            # Column 3: Vector Field showing geodesic path
            ax3 = fig.add_subplot(gs[i, 2])
            ax3.imshow(saliency, cmap='gray', alpha=0.3, interpolation='bilinear')

            # Plot deformation vectors
            step = 16
            y_coords = np.arange(0, 224, step)
            x_coords = np.arange(0, 224, step)
            Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')

            phi_np = phi.squeeze().numpy()
            U = phi_np[0, ::step, ::step] * 224 * 0.5
            V = phi_np[1, ::step, ::step] * 224 * 0.5

            ax3.quiver(X, Y, U, V, color='cyan', alpha=0.8, scale=40, width=0.003)
            ax3.set_title('Geodesic Path\n(Deformation Field)', fontsize=9, fontweight='bold')
            ax3.axis('off')

            # Column 4: Aligned IG
            ax4 = fig.add_subplot(gs[i, 3])
            ax4.imshow(h_aligned, cmap='hot', interpolation='bilinear')
            ax4.set_title('Aligned IG\n(After Warping)', fontsize=9, fontweight='bold')
            ax4.axis('off')

            # Column 5: Target Template (same for all rows)
            ax5 = fig.add_subplot(gs[i, 4])
            im = ax5.imshow(template, cmap='hot', interpolation='bilinear')
            ax5.set_title(f'Target Template\n(Pattern {pattern_id})', fontsize=9, fontweight='bold')
            ax5.axis('off')
            if i == 0:
                plt.colorbar(im, ax=ax5, fraction=0.046)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved multi-sample flow to {save_path}")

        plt.show()
        return fig


def main():
    parser = argparse.ArgumentParser(
        description='Create publication-quality geodesic visualizations'
    )
    parser.add_argument('--checkpoint', type=str, default='checkpoints/lddmm_model_final.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str,
                       default='./data/saliency_imagenet1k_resnet50_100',
                       help='Directory with saliency maps')
    parser.add_argument('--class_id', type=int, required=True,
                       help='Class ID to visualize')
    parser.add_argument('--pattern_id', type=int, default=None,
                       help='Specific pattern ID (default: auto-select top patterns)')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['anatomical', 'flow', 'all'],
                       help='Visualization mode')
    parser.add_argument('--num_samples', type=int, default=6,
                       help='Number of samples for flow visualization')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--images_per_class', type=int, default=100,
                       help='Images per class')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n{'='*80}")
    print(f"Multi-Sample Geodesic Visualization")
    print(f"{'='*80}")

    visualizer = MultiGeoVisualizer(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        device=args.device,
        images_per_class=args.images_per_class
    )

    if args.mode in ['anatomical', 'all']:
        print(f"\n{'='*80}")
        print(f"Creating Anatomical Landmark Overlay...")
        print(f"{'='*80}")
        save_path = output_dir / f"anatomical_landmarks_class_{args.class_id}.png"
        visualizer.visualize_anatomical_landmarks(
            class_id=args.class_id,
            pattern_id=args.pattern_id,
            save_path=save_path
        )

    if args.mode in ['flow', 'all']:
        print(f"\n{'='*80}")
        print(f"Creating Multi-Sample Flow Visualization...")
        print(f"{'='*80}")
        save_path = output_dir / f"multi_sample_flow_class_{args.class_id}_pattern_{args.pattern_id if args.pattern_id else 'auto'}.png"
        visualizer.visualize_multi_sample_flow(
            class_id=args.class_id,
            pattern_id=args.pattern_id,
            num_samples=args.num_samples,
            save_path=save_path
        )

    print(f"\n{'='*80}")
    print(f"✓ Visualizations saved to {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
