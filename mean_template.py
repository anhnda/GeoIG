"""
Mean Template - Simplified version of geo_v2.py

CHANGES FROM geo_v2.py:
1. Direct mean calculation for template updates (no Sinkhorn-Knopp)
2. Simplified template update function
3. Faster convergence with straightforward averaging

Usage:
    python mean_template.py --data_dir ./data/saliency_imagenet1k_resnet50_100 \
                            --num_classes 1000 --k_subpatterns 10 --epochs 50 --use_edge_gating
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import joblib
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import kornia
import gc

# Import all necessary components from geo_v2
from geo_v2 import (
    MultiScalePredictor,
    DiffeomorphicWarper,
    EdgeAwareGating,
    AdvancedLDDMMLoss,
    SaliencyMapDataset,
    safe_collate_fn
)


# ==========================================
# Simplified Pipeline with Mean Templates
# ==========================================

class MeanTemplatePipeline(nn.Module):
    """
    Simplified LDDMM pipeline using direct mean for template updates.

    KEY DIFFERENCE FROM geo_v2.py:
    - Template updates use simple mean (no Sinkhorn-Knopp optimal transport)
    - Faster and more memory-efficient
    - Easier to understand and debug
    """

    def __init__(self, num_classes=1000, k_subpatterns=10, img_res=(224, 224),
                 device='cuda', temperature=0.1, use_edge_gating=True):
        super().__init__()
        self.num_classes = num_classes
        self.K = k_subpatterns
        self.res = img_res
        self.device = device
        self.temperature = temperature
        self.use_edge_gating = use_edge_gating

        # 1. Multi-scale Predictor
        self.predictor = MultiScalePredictor(k_subpatterns=k_subpatterns, img_res=img_res)

        # 2. Warper (for both coarse and fine stages)
        self.warper = DiffeomorphicWarper(img_res=img_res, num_steps=7)

        # 3. Edge-aware gating (optional)
        if use_edge_gating:
            self.edge_gating = EdgeAwareGating()

        # 4. Template Bank
        templates_init = torch.rand((num_classes, k_subpatterns, 1, *img_res)) * 0.1
        templates_init = torch.where(templates_init > 0.07, templates_init, torch.zeros_like(templates_init))
        self.register_buffer('templates', templates_init)
        self.register_buffer('template_counts', torch.zeros(num_classes, k_subpatterns))

        print(f"\nMean Template Pipeline Initialized:")
        print(f"  Multi-scale: Coarse (14×14) → Fine (112×112)")
        print(f"  Template update: Direct Mean (simplified)")
        print(f"    - Sparsity: 98th percentile")
        print(f"    - Top-K sharpening: 3%")
        print(f"  Edge gating: {'Enabled' if use_edge_gating else 'Disabled'}")
        print(f"  Classes: {num_classes}, Sub-patterns: {k_subpatterns}")

    def forward(self, h_i, class_ids=None, update_templates=True, original_images=None):
        """
        Forward pass with multi-scale alignment.

        Args:
            h_i: (B, 1, H, W) - Input saliency maps
            class_ids: (B,) - Class labels
            update_templates: bool - Whether to update template bank
            original_images: (B, 3, H, W) - Original images (for edge gating)

        Returns:
            h_aligned: (B, 1, H, W) - Aligned saliency maps
            cluster_probs: (B, K) - Soft cluster assignments
            phi_coarse: (B, 2, H, W) - Coarse diffeomorphism
            phi_fine: (B, 2, H, W) - Fine diffeomorphism
            v_coarse: (B, 2, 14, 14) - Coarse velocity field
            v_fine: (B, 2, 112, 112) - Fine velocity field
        """
        if h_i.shape[1] == 0:
            raise RuntimeError(f"Input tensor has invalid shape: {h_i.shape}, expected [B, 1, 224, 224]")

        B = h_i.shape[0]

        # Optional: Apply edge gating
        if self.use_edge_gating and original_images is not None:
            h_i_gated, edge_mask = self.edge_gating(original_images, h_i)
        else:
            h_i_gated = h_i
            edge_mask = None

        # Ensure h_i_gated is always (B, 1, H, W)
        if h_i_gated.dim() == 3:
            h_i_gated = h_i_gated.unsqueeze(1)
        elif h_i_gated.shape[1] == 0:
            h_i_gated = h_i.clone()

        # Step 1: Predict cluster assignment and multi-scale velocity fields
        cluster_logits, v_coarse, v_fine, h_i_blurred = self.predictor(h_i_gated)
        cluster_probs = F.softmax(cluster_logits / self.temperature, dim=-1)
        cluster_assigned = torch.argmax(cluster_probs, dim=-1)

        # Step 2: COARSE alignment (warp blurred IG with coarse field)
        phi_coarse = self.warper(v_coarse)
        h_coarse_aligned = self.warper.warp_image(h_i_blurred, phi_coarse)

        # Step 3: FINE alignment (warp coarse-aligned IG with fine residual field)
        phi_fine = self.warper(v_fine)
        h_aligned = self.warper.warp_image(h_coarse_aligned, phi_fine)

        # Step 4: Update template bank using MEAN
        if update_templates and class_ids is not None:
            self._update_template_bank_mean(h_aligned, class_ids, cluster_assigned)

        return h_aligned, cluster_probs, phi_coarse, phi_fine, v_coarse, v_fine

    @torch.no_grad()
    def _update_template_bank_mean(self, h_aligned, class_ids, cluster_assigned,
                                    sparsity_percentile=98, top_k_percent=3):
        """
        Update template bank using DIRECT MEAN with top-K sharpening.

        This is simpler and faster than Sinkhorn-Knopp optimal transport.
        For each (class, cluster), we simply average all aligned samples.

        Args:
            h_aligned: (B, 1, H, W) - Aligned saliency maps
            class_ids: (B,) - Class indices
            cluster_assigned: (B,) - Cluster indices
            sparsity_percentile: float - Percentile threshold for sparsity
            top_k_percent: float - Keep only top K% of pixels
        """
        B = h_aligned.shape[0]

        # Group samples by (class, cluster)
        unique_pairs = torch.unique(torch.stack([class_ids, cluster_assigned], dim=1), dim=0)

        for pair in unique_pairs:
            c, k = pair[0], pair[1]

            # Find all samples for this (class, cluster)
            mask = (class_ids == c) & (cluster_assigned == k)
            if not mask.any():
                continue

            samples = h_aligned[mask]  # (N, 1, H, W)

            # Compute MEAN (simple average)
            mean_template = samples.mean(dim=0)  # (1, H, W)

            # === TOP-K SHARPENING ===
            num_pixels = mean_template.numel()
            top_k = max(int(num_pixels * top_k_percent / 100.0), 1)

            # Get indices of top-K values
            mean_flat = mean_template.flatten()
            top_k_values, top_k_indices = torch.topk(mean_flat, k=top_k)

            # Create sharpened version: zero out all but top-K
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

            # Blend with existing template (slow EMA for stability)
            count = self.template_counts[c, k]
            if count == 0:
                self.templates[c, k] = mean_template
            else:
                eta = torch.clamp(torch.tensor(0.1, device=count.device),
                                  max=1.0 / (count + 1))
                self.templates[c, k] = (1 - eta) * self.templates[c, k] + eta * mean_template

            self.template_counts[c, k] += mask.sum()

            # Free memory
            del samples, mean_template, mean_flat, top_k_values, top_k_indices, mean_sharp

    def get_batch_templates(self, class_ids):
        """Get templates for a batch of classes."""
        return self.templates[class_ids]  # (B, K, 1, H, W)


# ==========================================
# Training
# ==========================================

class MeanTemplateTrainer:
    """Trainer for Mean Template pipeline."""

    def __init__(self, model, train_loader, val_loader=None,
                 lr=1e-3, device='cuda', checkpoint_dir='./checkpoints_mean', use_amp=True):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.use_amp = use_amp and torch.cuda.is_available()

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5)

        # Loss
        self.criterion = AdvancedLDDMMLoss(
            lambda_smooth=0.01,
            lambda_entropy=1.0,
            lambda_magnitude=0.00005,
            lambda_diversity=2.0,
            lambda_template_diversity=2.0,
            lambda_template_sparsity=7.0,
            lambda_spatial_diversity=1,
            lambda_compactness=10.0,
            lambda_mass_conservation=1.0,
            lambda_sparsity_match=10.0,
            lambda_tv=0.5,
            lambda_jacobian=75.0,
            lambda_coarse_smooth=0.03
        ).to(device)

        # Automatic Mixed Precision
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print(f"  Using Automatic Mixed Precision (AMP) for faster training")
        else:
            self.scaler = None

        # History
        self.history = defaultdict(list)

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()

        epoch_losses = defaultdict(float)
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch_data in enumerate(pbar):
            if len(batch_data) == 3:
                saliency_maps, labels, rgb_images = batch_data
            else:
                saliency_maps, labels = batch_data
                rgb_images = None

            saliency_maps = saliency_maps.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            if rgb_images is not None:
                rgb_images = rgb_images.to(self.device, non_blocking=True)

            # Forward pass with AMP
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                h_aligned, cluster_probs, phi_coarse, phi_fine, v_coarse, v_fine = self.model(
                    saliency_maps,
                    class_ids=labels,
                    update_templates=True,
                    original_images=rgb_images
                )

                # Get templates for this batch
                batch_templates = self.model.get_batch_templates(labels)

                # Compute loss
                losses = self.criterion(saliency_maps, h_aligned, batch_templates,
                                      cluster_probs, v_coarse, v_fine)

            # Backward with AMP
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(losses['total']).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            # Accumulate losses
            for key, value in losses.items():
                epoch_losses[key] += value.item()
            num_batches += 1

            # Update progress bar
            postfix_dict = {
                'loss': losses['total'].item(),
                'align': losses['alignment'].item(),
            }
            if torch.cuda.is_available():
                gpu_mem_used = torch.cuda.memory_allocated() / (1024**3)
                postfix_dict['GPU_GB'] = f'{gpu_mem_used:.1f}'

            pbar.set_postfix(postfix_dict)

            # Periodic memory cleanup
            if batch_idx % 50 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        # Average losses
        avg_losses = {key: value / num_batches for key, value in epoch_losses.items()}

        # Update history
        for key, value in avg_losses.items():
            self.history[f'train_{key}'].append(value)

        # Run garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return avg_losses

    def train(self, num_epochs, save_frequency=5):
        """Train for multiple epochs."""
        print(f"\n{'='*80}")
        print(f"Starting Mean Template Training")
        print(f"{'='*80}")
        print(f"Epochs: {num_epochs}")
        print(f"Device: {self.device}")

        for epoch in range(1, num_epochs + 1):
            avg_losses = self.train_epoch(epoch)

            print(f"\nEpoch {epoch}/{num_epochs} Summary:")
            print(f"  Total Loss: {avg_losses['total']:.4f}")
            print(f"  Alignment: {avg_losses['alignment']:.4f}")
            print(f"  Mass Conservation: {avg_losses['mass_conservation']:.4f}")
            print(f"  Sparsity Match: {avg_losses['sparsity_match']:.4f}")

            # Update learning rate scheduler
            self.scheduler.step(avg_losses['total'])

            # Save checkpoint
            if epoch % save_frequency == 0:
                self.save_checkpoint(epoch)

        # Final save
        self.save_checkpoint(num_epochs, final=True)

        print(f"\n{'='*80}")
        print(f"✓ Training Complete!")
        print(f"{'='*80}")

    def save_checkpoint(self, epoch, final=False):
        """Save model checkpoint."""
        if final:
            checkpoint_path = self.checkpoint_dir / "mean_template_model_final.pth"
        else:
            checkpoint_path = self.checkpoint_dir / f"mean_template_model_epoch_{epoch}.pth"

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': dict(self.history),
            'templates': self.model.templates,
            'template_counts': self.model.template_counts
        }, checkpoint_path)

        print(f"  Checkpoint saved: {checkpoint_path}")

    def plot_training_curves(self, save_path='training_curves_mean.png'):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        metrics = [
            ('train_total', 'Total Loss'),
            ('train_alignment', 'Alignment Loss'),
            ('train_mass_conservation', 'Mass Conservation'),
            ('train_sparsity_match', 'Sparsity Match')
        ]

        for idx, (key, title) in enumerate(metrics):
            if key in self.history:
                axes[idx].plot(self.history[key])
                axes[idx].set_title(title)
                axes[idx].set_xlabel('Epoch')
                axes[idx].grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Training curves saved to {save_path}")


# ==========================================
# Main Script
# ==========================================

def main():
    torch._dynamo.config.capture_scalar_outputs = True

    parser = argparse.ArgumentParser(
        description='Mean Template - Simplified Geodesic Pattern Learning',
        epilog="""
SIMPLIFIED APPROACH:
  - Direct mean for template updates (no Sinkhorn-Knopp)
  - Faster and more memory-efficient
  - Same aggressive sparsity enforcement as v2

USAGE:
  python mean_template.py --data_dir ./data/saliency_imagenet1k_resnet50_100 \\
                          --use_edge_gating --epochs 50
        """
    )
    parser.add_argument('--data_dir', type=str,
                       default='./data/saliency_imagenet1k_resnet50_100',
                       help='Directory with saliency maps')
    parser.add_argument('--num_classes', type=int, default=1000,
                       help='Number of classes (default: 1000)')
    parser.add_argument('--k_subpatterns', type=int, default=10,
                       help='Number of sub-patterns per class (default: 10)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--use_edge_gating', dest='use_edge_gating', action='store_true', default=True,
                       help='Use edge-aware attention gating (DEFAULT: ON)')
    parser.add_argument('--no_edge_gating', dest='use_edge_gating', action='store_false',
                       help='Disable edge-aware attention gating')
    parser.add_argument('--max_samples_per_class', type=int, default=None,
                       help='Max samples per class (default: None = all)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_mean',
                       help='Checkpoint directory (default: ./checkpoints_mean)')
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"Mean Template - Simplified Geodesic Pattern Learning")
    print(f"{'='*80}")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total GPU Memory: {total_mem:.1f} GB")
        torch.cuda.empty_cache()

    # Load dataset
    print(f"\nLoading saliency maps from {args.data_dir}...")
    dataset = SaliencyMapDataset(
        data_dir=args.data_dir,
        max_samples_per_class=args.max_samples_per_class,
        load_images=args.use_edge_gating,
        max_samples=100000
    )

    # DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        collate_fn=safe_collate_fn,
        persistent_workers=True
    )

    # Create model
    print(f"\nInitializing Mean Template model...")
    model = MeanTemplatePipeline(
        num_classes=args.num_classes,
        k_subpatterns=args.k_subpatterns,
        img_res=(224, 224),
        device=device,
        temperature=0.1,
        use_edge_gating=args.use_edge_gating
    )
    model = torch.compile(model)

    # Create trainer
    trainer = MeanTemplateTrainer(
        model=model,
        train_loader=train_loader,
        lr=args.lr,
        device=device,
        checkpoint_dir=args.checkpoint_dir
    )

    # Train
    trainer.train(num_epochs=args.epochs, save_frequency=5)

    # Plot training curves
    trainer.plot_training_curves(
        save_path=Path(args.checkpoint_dir) / 'training_curves_mean.png'
    )

    print(f"\n✓ All done!")
    print(f"\nTo visualize mean templates:")
    print(f"  python visualize_mean.py --checkpoint {args.checkpoint_dir}/mean_template_model_final.pth --class_id 0")


if __name__ == "__main__":
    main()
