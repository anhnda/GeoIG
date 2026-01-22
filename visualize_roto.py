"""
Roto-LDDMM Visualization: Part-Based Pattern Decomposition

This script visualizes:
1. The K learnable atoms (LEGO pieces) for a specific class
2. Sample image decomposition showing:
   - Original IG saliency map
   - Composed/reconstructed map from atoms
   - Individual transformed atoms with their SE(2) poses
   - Attention weights (contribution of each atom)

Usage:
    # Visualize class 0 atoms and decomposition
    python visualize_roto.py --checkpoint checkpoints_roto/roto_lddmm_model_final.pth --class_id 0

    # Visualize specific class with custom data
    python visualize_roto.py --checkpoint checkpoints_roto/roto_lddmm_model_epoch_10.pth \
                             --class_id 281 \
                             --data_dir ./data/saliency_imagenet1k_resnet50_100
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Circle
from pathlib import Path
import argparse
import joblib
from collections import defaultdict

# Import Roto-LDDMM components
import sys
sys.path.append('.')
from geo_roto import RotoLDDMM_Pipeline
from full_classes import IMAGENET2012_CLASSES


class RotoVisualizer:
    """Visualize learned atoms and part-based decompositions."""

    def __init__(self, checkpoint_path, data_dir, device='cuda'):
        self.device = device
        self.data_dir = Path(data_dir)

        # Load checkpoint
        print(f"Loading Roto-LDDMM checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract configuration from atoms shape
        if 'atoms' in checkpoint:
            atoms = checkpoint['atoms']
            if atoms.dim() == 4:
                # Shared atoms: (K, 1, H, W)
                self.shared_atoms = True
                self.num_classes = 1000  # Default
                self.k_atoms = atoms.shape[0]
                self.atom_res = (atoms.shape[2], atoms.shape[3])
            else:
                # Class-specific atoms: (num_classes, K, 1, H, W)
                self.shared_atoms = False
                self.num_classes = atoms.shape[0]
                self.k_atoms = atoms.shape[1]
                self.atom_res = (atoms.shape[3], atoms.shape[4])
        else:
            raise ValueError("Checkpoint does not contain 'atoms'. Is this a Roto-LDDMM checkpoint?")

        print(f"Loaded model:")
        print(f"  Atom mode: {'SHARED' if self.shared_atoms else 'CLASS-SPECIFIC'}")
        print(f"  Classes: {self.num_classes}")
        print(f"  Atoms per class: {self.k_atoms}")
        print(f"  Atom resolution: {self.atom_res}")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")

        # Create model
        self.model = RotoLDDMM_Pipeline(
            num_classes=self.num_classes,
            k_atoms=self.k_atoms,
            img_res=(224, 224),
            atom_res=self.atom_res,
            device=device,
            use_local_warp=True,
            composition_mode='softmax',
            shared_atoms=self.shared_atoms
        ).to(device)

        # Load weights
        state_dict = checkpoint['model_state_dict']

        # Strip _orig_mod. prefix if present (from torch.compile())
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            print("  Detected torch.compile() checkpoint - stripping _orig_mod. prefix...")
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        print(f"✓ Model loaded successfully!")

        # Store path to saliency data
        self.saliency_dir = self.data_dir / "saliency_maps"
        if not self.saliency_dir.exists():
            print(f"Warning: Saliency directory not found: {self.saliency_dir}")

        # Get class names
        self.class_names = {idx: name for idx, (_, name) in enumerate(IMAGENET2012_CLASSES.items())}

        # Build class index for fast lookup
        self.cache_file = self.data_dir / "class_index_cache_roto.pkl"
        print(f"\nBuilding/loading class index for fast lookup...")
        self.class_index = self._load_or_build_index()
        print(f"✓ Class index ready!")

    def _is_cache_valid(self):
        """Check if cached index exists and is up-to-date."""
        if not self.cache_file.exists():
            return False

        cache_mtime = self.cache_file.stat().st_mtime
        batch_files = sorted(self.saliency_dir.glob("batch_*.pkl"))

        if not batch_files:
            return False

        for batch_file in batch_files:
            if batch_file.stat().st_mtime > cache_mtime:
                return False

        return True

    def _build_class_index(self):
        """Build index mapping class_id -> list of (batch_file, item_index)."""
        import time

        print("  Building class index from batch files...")
        start_time = time.time()

        class_index = defaultdict(list)
        batch_files = sorted(self.saliency_dir.glob("batch_*.pkl"))

        if not batch_files:
            print(f"  Warning: No batch files found in {self.saliency_dir}")
            return {}

        for batch_file in batch_files:
            batch_data = joblib.load(batch_file)

            for item_idx, item in enumerate(batch_data):
                class_id = item['true_label']
                class_index[class_id].append((str(batch_file), item_idx))

            
        elapsed = time.time() - start_time
        total_samples = sum(len(samples) for samples in class_index.values())
        print(f"  ✓ Index built in {elapsed:.2f}s: {len(class_index)} classes, {total_samples} samples")

        return dict(class_index)

    def _save_index_cache(self, class_index):
        """Save class index to cache file."""
        print(f"  Saving index cache to {self.cache_file}...")
        joblib.dump(class_index, self.cache_file, compress=3)
        print(f"  ✓ Cache saved")

    def _load_index_cache(self):
        """Load class index from cache file."""
        import time
        print(f"  Loading index from cache...")
        start_time = time.time()
        class_index = joblib.load(self.cache_file)
        elapsed = time.time() - start_time
        total_samples = sum(len(samples) for samples in class_index.values())
        print(f"  ✓ Cache loaded in {elapsed:.2f}s: {len(class_index)} classes, {total_samples} samples")
        return class_index

    def _load_or_build_index(self):
        """Load index from cache if valid, otherwise build and cache it."""
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
        """Sample images from a specific class using pre-built index."""
        if class_id not in self.class_index:
            print(f"Warning: Class {class_id} not found in index")
            return []

        class_locations = self.class_index[class_id]

        if num_samples is not None and num_samples < len(class_locations):
            selected_indices = np.random.choice(len(class_locations), size=num_samples, replace=False)
            class_locations = [class_locations[i] for i in selected_indices]
        elif num_samples is not None:
            class_locations = class_locations[:num_samples]

        # Load samples
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

    def visualize_class_atoms(self, class_id, save_path=None):
        """
        Visualize all K atoms (LEGO pieces) for a specific class.

        Args:
            class_id: Class index
            save_path: Optional path to save figure
        """
        class_name = self.class_names.get(class_id, f"Class {class_id}")

        # Get atoms for this class
        with torch.no_grad():
            atoms_tensor = self.model.atom_bank.get_atoms_for_class(class_id)
            atoms = atoms_tensor.detach().cpu().numpy()  # (K, 1, H, W)
            if atoms.ndim == 3:
                atoms = np.expand_dims(atoms, 1)  # (K, 1, H, W)

        # Get usage counts
        usage_counts = self.model.atom_bank.usage_counts
        if self.shared_atoms:
            counts = usage_counts.detach().cpu().numpy()  # (K,)
        else:
            counts = usage_counts[class_id].detach().cpu().numpy()  # (K,)

        # Create visualization
        ncols = min(5, self.k_atoms)
        nrows = (self.k_atoms + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
        if nrows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        fig.suptitle(f'Learned Atoms (LEGO Pieces) for {class_name} (Class {class_id})',
                     fontsize=16, fontweight='bold')

        for k in range(self.k_atoms):
            ax = axes[k]

            atom = atoms[k, 0] if atoms.ndim == 4 else atoms[k]  # (H, W)

            # Aggressive normalization for visualization
            nonzero_vals = atom[atom > 0.01]
            if len(nonzero_vals) > 0:
                vmax = np.percentile(nonzero_vals, 90)
                vmax = max(vmax, 0.1)
            else:
                vmax = atom.max() if atom.max() > 0 else 1.0

            im = ax.imshow(atom, cmap='hot', interpolation='bilinear', vmin=0, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046)

            sparsity = (atom > 0.01).sum() / atom.size * 100
            usage_pct = counts[k] / counts.sum() * 100 if counts.sum() > 0 else 0
            ax.set_title(f'Atom {k}\nmax={atom.max():.3f}, vmax={vmax:.3f}\n'
                        f'Active: {sparsity:.1f}% | Usage: {usage_pct:.1f}%',
                        fontsize=9)
            ax.axis('off')

        # Hide unused subplots
        for k in range(self.k_atoms, len(axes)):
            axes[k].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved class atoms to {save_path}")

        plt.show()
        return fig

    def visualize_decomposition(self, class_id, sample_idx=0, save_path=None):
        """
        Visualize part-based decomposition of a sample into atoms.

        Shows:
        1. Original IG saliency map
        2. Composed/reconstructed map
        3. Individual transformed atoms (top-5 by attention)
        4. Attention weights bar chart
        5. Atom poses (translation, rotation, scale)

        Args:
            class_id: Class index
            sample_idx: Which sample from the class to visualize
            save_path: Optional path to save figure
        """
        class_name = self.class_names.get(class_id, f"Class {class_id}")

        # Sample image
        samples = self.sample_class_images(class_id, num_samples=sample_idx + 1)

        if not samples or sample_idx >= len(samples):
            print(f"Error: Sample {sample_idx} not found for class {class_id}")
            return None

        sample = samples[sample_idx]
        saliency_map = sample['saliency_map']
        confidence = sample['confidence']

        # Convert to tensor
        h_i = torch.from_numpy(saliency_map).float().unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, H, W)

        # Forward pass
        with torch.no_grad():
            h_composed, poses, attention, atoms_transformed = self.model(
                h_i,
                class_ids=torch.tensor([class_id], device=self.device),
                update_atoms=False
            )

        # Move to CPU
        h_i_np = h_i[0, 0].cpu().numpy()  # (H, W)
        h_composed_np = h_composed[0, 0].cpu().numpy()  # (H, W)
        attention_np = attention[0].cpu().numpy()  # (K,)
        poses_np = poses[0].cpu().numpy()  # (K, 4)
        atoms_transformed_np = atoms_transformed[0].cpu().numpy()  # (K, 1, H, W)

        # Get top-5 atoms by attention
        top_k_indices = np.argsort(attention_np)[::-1][:5]

        # Create comprehensive visualization
        fig = plt.figure(figsize=(24, 14))
        gs = fig.add_gridspec(4, 6, hspace=0.35, wspace=0.3)

        # Title
        fig.suptitle(f'Roto-LDDMM Decomposition: {class_name} (Class {class_id})\n'
                    f'Sample {sample_idx} | Confidence: {confidence:.3f}',
                    fontsize=16, fontweight='bold')

        # Row 1: Original IG and Composed Map
        ax_orig = fig.add_subplot(gs[0, 0:2])
        im_orig = ax_orig.imshow(h_i_np, cmap='hot', interpolation='bilinear')
        ax_orig.set_title('Original IG Map', fontsize=14, fontweight='bold')
        ax_orig.axis('off')
        plt.colorbar(im_orig, ax=ax_orig, fraction=0.046)

        ax_composed = fig.add_subplot(gs[0, 2:4])
        im_composed = ax_composed.imshow(h_composed_np, cmap='hot', interpolation='bilinear')
        ax_composed.set_title('Composed Map (from atoms)', fontsize=14, fontweight='bold')
        ax_composed.axis('off')
        plt.colorbar(im_composed, ax=ax_composed, fraction=0.046)

        # Reconstruction error
        ax_error = fig.add_subplot(gs[0, 4:6])
        error = np.abs(h_i_np - h_composed_np)
        im_error = ax_error.imshow(error, cmap='RdYlGn_r', interpolation='bilinear')
        mse = np.mean(error**2)
        ax_error.set_title(f'Reconstruction Error\nMSE: {mse:.6f}', fontsize=14, fontweight='bold')
        ax_error.axis('off')
        plt.colorbar(im_error, ax=ax_error, fraction=0.046)

        # Row 2: Attention weights bar chart
        ax_attention = fig.add_subplot(gs[1, :3])
        bars = ax_attention.bar(range(self.k_atoms), attention_np, color='steelblue')
        # Highlight top-5
        for i in top_k_indices:
            bars[i].set_color('red')
        ax_attention.set_xlabel('Atom Index', fontsize=12)
        ax_attention.set_ylabel('Attention Weight', fontsize=12)
        ax_attention.set_title('Atom Contributions (Top-5 in Red)', fontsize=14, fontweight='bold')
        ax_attention.set_xticks(range(self.k_atoms))
        ax_attention.grid(axis='y', alpha=0.3)

        # Add percentage labels for top-5
        for i in top_k_indices:
            height = bars[i].get_height()
            ax_attention.text(i, height, f'{attention_np[i]*100:.1f}%',
                            ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Row 2: Pose parameters table
        ax_poses = fig.add_subplot(gs[1, 3:])
        ax_poses.axis('off')
        pose_text = "Top-5 Atom Poses (tx, ty, θ, scale):\n\n"
        for i, k in enumerate(top_k_indices):
            tx, ty, theta, scale = poses_np[k]
            theta_deg = theta * 180 / np.pi
            pose_text += f"Atom {k}: tx={tx:.2f}, ty={ty:.2f}, θ={theta_deg:.0f}°, s={scale:.2f}\n"
            pose_text += f"         Attention: {attention_np[k]*100:.1f}%\n\n"
        ax_poses.text(0.1, 0.9, pose_text, fontsize=11, family='monospace',
                     verticalalignment='top', transform=ax_poses.transAxes)

        # Rows 3-4: Top-5 transformed atoms
        for i, k in enumerate(top_k_indices):
            row = 2 + i // 3
            col = (i % 3) * 2

            # Transformed atom
            ax_atom = fig.add_subplot(gs[row, col:col+2])
            atom = atoms_transformed_np[k, 0]  # (H, W)

            nonzero_vals = atom[atom > 0.01]
            if len(nonzero_vals) > 0:
                vmax = np.percentile(nonzero_vals, 90)
                vmax = max(vmax, 0.05)
            else:
                vmax = atom.max() if atom.max() > 0 else 1.0

            im_atom = ax_atom.imshow(atom, cmap='hot', interpolation='bilinear', vmin=0, vmax=vmax)
            plt.colorbar(im_atom, ax=ax_atom, fraction=0.046)

            tx, ty, theta, scale = poses_np[k]
            theta_deg = theta * 180 / np.pi
            ax_atom.set_title(f'Atom {k} (Attention: {attention_np[k]*100:.1f}%)\n'
                            f'Pose: tx={tx:.2f}, ty={ty:.2f}, θ={theta_deg:.0f}°, s={scale:.2f}\n'
                            f'max={atom.max():.3f}, vmax={vmax:.3f}',
                            fontsize=10)
            ax_atom.axis('off')

            # Visualize pose as arrow on composed map
            if i < 3:  # Only show for top-3
                # Convert normalized coords to pixel coords
                cx = (tx + 1) * 112  # Map [-1, 1] to [0, 224]
                cy = (ty + 1) * 112

                # Draw circle at center
                circle = Circle((cx, cy), radius=5, color='cyan', fill=True, alpha=0.8)
                ax_composed.add_patch(circle)

                # Draw arrow showing rotation
                arrow_len = 20 * scale
                dx = arrow_len * np.cos(theta)
                dy = arrow_len * np.sin(theta)
                arrow = FancyArrow(cx, cy, dx, dy, width=3, color='cyan', alpha=0.8)
                ax_composed.add_patch(arrow)

                # Label
                ax_composed.text(cx + dx + 5, cy + dy + 5, f'{k}', color='cyan',
                               fontsize=12, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved decomposition to {save_path}")

        plt.show()
        return fig

    def compare_samples(self, class_id, num_samples=4, save_path=None):
        """
        Compare decompositions of multiple samples from the same class.

        Shows how different images use different atoms with different poses.

        Args:
            class_id: Class index
            num_samples: Number of samples to compare
            save_path: Optional path to save figure
        """
        class_name = self.class_names.get(class_id, f"Class {class_id}")

        # Sample images
        samples = self.sample_class_images(class_id, num_samples=num_samples)

        if not samples:
            print(f"Error: No samples found for class {class_id}")
            return None

        num_samples = min(num_samples, len(samples))

        # Create visualization
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
                h_composed, poses, attention, _ = self.model(
                    h_i,
                    class_ids=torch.tensor([class_id], device=self.device),
                    update_atoms=False
                )

            # Original IG
            axes[i, 0].imshow(saliency_map, cmap='hot', interpolation='bilinear')
            axes[i, 0].set_title(f'Sample {i}: Original IG', fontsize=11)
            axes[i, 0].axis('off')

            # Composed map
            axes[i, 1].imshow(h_composed[0, 0].cpu().numpy(), cmap='hot', interpolation='bilinear')
            axes[i, 1].set_title(f'Composed Map', fontsize=11)
            axes[i, 1].axis('off')

            # Attention weights
            attention_np = attention[0].cpu().numpy()  # (K,)
            top_3 = np.argsort(attention_np)[::-1][:3]
            bars = axes[i, 2].bar(range(self.k_atoms), attention_np, color='steelblue')
            for k in top_3:
                bars[k].set_color('red')
            axes[i, 2].set_title(f'Atom Attention (Top-3: {top_3})', fontsize=11)
            axes[i, 2].set_ylim(0, max(1.0, attention_np.max() * 1.1))
            axes[i, 2].grid(axis='y', alpha=0.3)

            # Pose info text
            axes[i, 3].axis('off')
            poses_np = poses[0].cpu().numpy()  # (K, 4)
            pose_text = "Top-3 Atoms:\n\n"
            for k in top_3:
                _, _, theta, scale = poses_np[k]  # tx, ty not used in summary
                theta_deg = theta * 180 / np.pi
                pose_text += f"Atom {k}: θ={theta_deg:.0f}°, s={scale:.2f}\n"
                pose_text += f"  Attn: {attention_np[k]*100:.1f}%\n\n"
            axes[i, 3].text(0.1, 0.9, pose_text, fontsize=10, family='monospace',
                          verticalalignment='top', transform=axes[i, 3].transAxes)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved comparison to {save_path}")

        plt.show()
        return fig


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Roto-LDDMM part-based decomposition'
    )
    parser.add_argument('--checkpoint', type=str, default='checkpoints_roto/roto_lddmm_model_final.pth',
                       help='Path to Roto-LDDMM checkpoint')
    parser.add_argument('--data_dir', type=str,
                       default='./data/saliency_imagenet1k_resnet50_100',
                       help='Directory with saliency maps')
    parser.add_argument('--class_id', type=int, default=0,
                       help='Class ID to visualize')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Sample index for decomposition')
    parser.add_argument('--num_samples', type=int, default=4,
                       help='Number of samples for comparison')
    parser.add_argument('--output_dir', type=str, default='./visualizations_roto',
                       help='Output directory for saved figures')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sample selection')

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create visualizer
    print(f"\n{'='*80}")
    print(f"Roto-LDDMM Visualization")
    print(f"{'='*80}")

    visualizer = RotoVisualizer(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        device=args.device
    )

    # Visualize class atoms
    print(f"\n{'='*80}")
    print(f"Visualizing Class {args.class_id} Atoms (LEGO Pieces)")
    print(f"{'='*80}")

    atoms_path = output_dir / f"class_{args.class_id}_atoms.png"
    visualizer.visualize_class_atoms(
        class_id=args.class_id,
        save_path=atoms_path
    )

    # Visualize decomposition
    print(f"\n{'='*80}")
    print(f"Visualizing Part-Based Decomposition")
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
