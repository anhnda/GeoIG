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
from tqdm import tqdm

# Import model components from geo_patterns
import sys
sys.path.append('.')
from geo_patterns import LDDMM_GlobalPatternPipeline
from full_classes import IMAGENET2012_CLASSES


class PatternVisualizer:
    """Visualize learned patterns and decompositions."""

    def __init__(self, checkpoint_path, data_dir, device='cuda'):
        self.device = device
        self.data_dir = Path(data_dir)

        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract configuration
        self.num_classes = checkpoint['templates'].shape[0]
        self.k_subpatterns = checkpoint['templates'].shape[1]

        print(f"Loaded model:")
        print(f"  Classes: {self.num_classes}")
        print(f"  Sub-patterns per class: {self.k_subpatterns}")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")

        # Create model
        self.model = LDDMM_GlobalPatternPipeline(
            num_classes=self.num_classes,
            k_subpatterns=self.k_subpatterns,
            img_res=(224, 224),
            device=device
        ).to(device)

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Templates are already in the model's buffers from state_dict
        print(f"✓ Model loaded successfully!")

        # Load saliency data
        self._load_saliency_data()

        # Get class names
        self.class_names = {idx: name for idx, (wnid, name) in enumerate(IMAGENET2012_CLASSES.items())}

    def _load_saliency_data(self):
        """Load saliency map data."""
        saliency_dir = self.data_dir / "saliency_maps"
        batch_files = sorted(saliency_dir.glob("batch_*.pkl"))

        print(f"\nLoading saliency data from {len(batch_files)} batches...")

        self.saliency_by_class = {}

        for batch_file in tqdm(batch_files, desc="Loading saliency maps"):
            batch_data = joblib.load(batch_file)

            for item in batch_data:
                label = item['true_label']

                if label not in self.saliency_by_class:
                    self.saliency_by_class[label] = []

                self.saliency_by_class[label].append({
                    'saliency_map': item['saliency_map'],
                    'confidence': item['confidence']
                })

        print(f"✓ Loaded saliency maps for {len(self.saliency_by_class)} classes")

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

            # Plot template
            template = templates[k, 0]  # (H, W)
            im = ax.imshow(template, cmap='hot', interpolation='bilinear')

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046)

            # Title with usage count
            usage_pct = counts[k] / counts.sum() * 100 if counts.sum() > 0 else 0
            ax.set_title(f'Pattern {k}\nUsage: {int(counts[k])} ({usage_pct:.1f}%)',
                        fontsize=10)
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

        # Get sample
        if class_id not in self.saliency_by_class:
            print(f"Error: No samples found for class {class_id}")
            return None

        samples = self.saliency_by_class[class_id]
        if sample_idx >= len(samples):
            print(f"Error: Sample {sample_idx} not found (only {len(samples)} samples)")
            return None

        sample = samples[sample_idx]
        saliency_map = sample['saliency_map']
        confidence = sample['confidence']

        # Convert to tensor
        h_i = torch.from_numpy(saliency_map).float().unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, H, W)

        # Forward pass
        with torch.no_grad():
            h_aligned, cluster_probs, phi, v_low = self.model(h_i, update_templates=False)
            cluster_assigned = torch.argmax(cluster_probs, dim=-1).item()

        # Move to CPU for visualization
        h_i_np = h_i.squeeze().cpu().numpy()
        h_aligned_np = h_aligned.squeeze().cpu().numpy()
        cluster_probs_np = cluster_probs.squeeze().cpu().numpy()
        phi_np = phi.squeeze().cpu().numpy()  # (2, H, W)

        # Get templates
        templates = self.model.templates[class_id].cpu().numpy()  # (K, 1, H, W)

        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 5, hspace=0.3, wspace=0.3)

        # Title
        fig.suptitle(f'Pattern Decomposition: {class_name} (Class {class_id})\n'
                    f'Sample {sample_idx} | Confidence: {confidence:.3f}',
                    fontsize=16, fontweight='bold')

        # Row 1: Original, Aligned, Deformation
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(h_i_np, cmap='hot', interpolation='bilinear')
        ax1.set_title('Original IG Map', fontsize=12, fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046)

        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(h_aligned_np, cmap='hot', interpolation='bilinear')
        ax2.set_title('Aligned IG Map', fontsize=12, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046)

        ax3 = fig.add_subplot(gs[0, 2])
        # Visualize deformation field magnitude
        deform_mag = np.sqrt(phi_np[0]**2 + phi_np[1]**2)
        im3 = ax3.imshow(deform_mag, cmap='viridis', interpolation='bilinear')
        ax3.set_title('Deformation Magnitude', fontsize=12, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046)

        # Deformation field arrows (downsampled)
        ax4 = fig.add_subplot(gs[0, 3:5])
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
            im = ax.imshow(template, cmap='hot', interpolation='bilinear')
            ax.set_title(f'Pattern {k}\n({cluster_probs_np[k]*100:.1f}%)',
                        fontsize=10, fontweight='bold')
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

        if class_id not in self.saliency_by_class:
            print(f"Error: No samples found for class {class_id}")
            return None

        samples = self.saliency_by_class[class_id]
        num_samples = min(num_samples, len(samples))

        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
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
                h_aligned, cluster_probs, _, _ = self.model(h_i, update_templates=False)
                cluster_assigned = torch.argmax(cluster_probs, dim=-1).item()

            # Original IG
            axes[i, 0].imshow(saliency_map, cmap='hot', interpolation='bilinear')
            axes[i, 0].set_title(f'Sample {i}: Original IG', fontsize=10)
            axes[i, 0].axis('off')

            # Aligned IG
            axes[i, 1].imshow(h_aligned.squeeze().cpu().numpy(), cmap='hot', interpolation='bilinear')
            axes[i, 1].set_title(f'Aligned (→ Pattern {cluster_assigned})', fontsize=10)
            axes[i, 1].axis('off')

            # Assigned pattern
            template = self.model.templates[class_id, cluster_assigned].squeeze().cpu().numpy()
            axes[i, 2].imshow(template, cmap='hot', interpolation='bilinear')
            axes[i, 2].set_title(f'Pattern {cluster_assigned} Template', fontsize=10)
            axes[i, 2].axis('off')

            # Cluster probabilities
            cluster_probs_np = cluster_probs.squeeze().cpu().numpy()
            axes[i, 3].bar(range(self.k_subpatterns), cluster_probs_np,
                          color=['red' if k == cluster_assigned else 'blue'
                                for k in range(self.k_subpatterns)])
            axes[i, 3].set_title(f'Assignment Probs', fontsize=10)
            axes[i, 3].set_ylim(0, 1)
            axes[i, 3].grid(axis='y', alpha=0.3)

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
    parser.add_argument('--checkpoint', type=str, required=True,
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

    args = parser.parse_args()

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
        device=args.device
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
