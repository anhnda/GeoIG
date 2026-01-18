"""
Geodesic Pattern Learning via Amortized LDDMM (Large Deformation Diffeomorphic Metric Mapping)

This module implements an amortized diffeomorphic alignment pipeline for learning
global patterns from local saliency maps. The key innovation is using Riemannian
geometry (diffeomorphisms) instead of simple Euclidean averaging to preserve sharp
features across aligned samples.

Architecture:
1. Predictor Network: Amortized CNN that predicts Stationary Velocity Fields (SVF)
2. Warping Engine: Scaling-and-squaring exponential map from Lie algebra to diffeomorphisms
3. Template Bank: Multi-modal storage for K sub-patterns per class

Reference:
- LDDMM: Beg et al. "Computing Large Deformation Metric Mappings via Geodesic Flows" (IJCV 2005)
- Stationary Velocity: Ashburner (2007)

Usage:
    # Training
    python geo_patterns.py --data_dir ./data/saliency_imagenet1k_resnet50_100 \
                          --num_classes 1000 --k_subpatterns 10 --epochs 50

    # Inference
    python geo_patterns.py --mode inference --checkpoint checkpoints/lddmm_model.pth
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


# ==========================================
# LDDMM Core Components
# ==========================================

class AmortizedPredictor(nn.Module):
    """
    Shared CNN backbone that predicts:
    1. Cluster assignment (soft assignment to K sub-patterns)
    2. Stationary Velocity Field (SVF) in Lie algebra

    Input: Saliency map h_i (1, H, W)
    Output: (cluster_logits, velocity_field)
    """

    def __init__(self, k_subpatterns=10, img_res=(224, 224)):
        super().__init__()
        self.K = k_subpatterns
        self.res = img_res

        # Shared backbone (input: 1-channel saliency map)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),  # 224 -> 112
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 112 -> 56
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 56 -> 28
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # 28 -> 14
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten()
        )

        feature_dim = 128 * 7 * 7

        # Head 1: Cluster Assignment
        self.assignment_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.K)
        )

        # Head 2: Velocity Field (predict at lower resolution for smoothness)
        self.v_res = (28, 28)  # Low-res velocity field
        self.v_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2 * self.v_res[0] * self.v_res[1])  # 2 channels (x, y flow)
        )

        # Initialize velocity field head to predict near-zero (small deformations initially)
        nn.init.normal_(self.v_head[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.v_head[-1].bias)

    def forward(self, h_i):
        """
        Args:
            h_i: (B, 1, H, W) - Saliency maps

        Returns:
            cluster_logits: (B, K) - Assignment scores
            v_low: (B, 2, v_H, v_W) - Low-resolution velocity field
        """
        B = h_i.shape[0]

        # Extract features
        features = self.backbone(h_i)

        # Predict cluster assignment
        cluster_logits = self.assignment_head(features)

        # Predict velocity field
        v_flat = self.v_head(features)
        v_low = v_flat.view(B, 2, *self.v_res)

        return cluster_logits, v_low


class DiffeomorphicWarper(nn.Module):
    """
    Scaling-and-squaring exponential map for diffeomorphic warping.

    Takes a Stationary Velocity Field (SVF) from Lie algebra and computes
    the exponential map to get a diffeomorphism (smooth, invertible transformation).

    Algorithm: Arsigny et al. "A Log-Euclidean Framework for Statistics on Diffeomorphisms" (MICCAI 2006)
    """

    def __init__(self, img_res=(224, 224), num_steps=7):
        super().__init__()
        self.res = img_res
        self.num_steps = num_steps  # Number of scaling-and-squaring steps

        # Create identity grid
        self.register_buffer('identity_grid', self._make_identity_grid())

    def _make_identity_grid(self):
        """Create normalized identity grid in [-1, 1]."""
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, self.res[0]),
            torch.linspace(-1, 1, self.res[1]),
            indexing='ij'
        )
        grid = torch.stack((x, y), dim=-1).unsqueeze(0)  # (1, H, W, 2)
        return grid

    def forward(self, v_low):
        """
        Compute diffeomorphism via scaling-and-squaring.

        Args:
            v_low: (B, 2, v_H, v_W) - Low-resolution velocity field

        Returns:
            phi: (B, 2, H, W) - Diffeomorphic displacement field
        """
        B = v_low.shape[0]

        # Upsample velocity field to full resolution (with smoothing)
        v = F.interpolate(v_low, size=self.res, mode='bilinear', align_corners=True)

        # Scale by 2^(-num_steps) for numerical stability
        v = v / (2 ** self.num_steps)

        # Scaling-and-squaring: iteratively compose flow with itself
        for _ in range(self.num_steps):
            # Current flow in grid format
            flow = v.permute(0, 2, 3, 1)  # (B, H, W, 2)

            # Create sampling grid
            identity = self.identity_grid.expand(B, -1, -1, -1)
            grid = identity + flow

            # Compose: v <- v + v(v)  (warp v by itself)
            v_warped = F.grid_sample(
                v, grid, mode='bilinear',
                padding_mode='border', align_corners=True
            )

            v = v + v_warped

        return v  # (B, 2, H, W)

    def warp_image(self, image, phi):
        """
        Warp image using diffeomorphism phi.

        Args:
            image: (B, C, H, W) - Input image
            phi: (B, 2, H, W) - Displacement field

        Returns:
            warped: (B, C, H, W) - Warped image
        """
        B = image.shape[0]

        # Convert displacement to sampling grid
        flow = phi.permute(0, 2, 3, 1)  # (B, H, W, 2)
        identity = self.identity_grid.expand(B, -1, -1, -1)
        grid = identity + flow

        # Sample
        warped = F.grid_sample(
            image, grid, mode='bilinear',
            padding_mode='border', align_corners=True
        )

        return warped


# ==========================================
# Complete LDDMM Pipeline
# ==========================================

class LDDMM_GlobalPatternPipeline(nn.Module):
    """
    Complete amortized LDDMM pipeline for global pattern discovery.

    Components:
    1. Predictor: Predicts velocity field and cluster assignment
    2. Warper: Applies diffeomorphic transformation
    3. Template Bank: Stores K sub-patterns per class
    """

    def __init__(self, num_classes=1000, k_subpatterns=10, img_res=(224, 224), device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.K = k_subpatterns
        self.res = img_res
        self.device = device

        # 1. Predictor
        self.predictor = AmortizedPredictor(k_subpatterns=k_subpatterns, img_res=img_res)

        # 2. Warper
        self.warper = DiffeomorphicWarper(img_res=img_res, num_steps=7)

        # 3. Template Bank (GPU-resident for fast access during training)
        # Shape: (num_classes, K, 1, H, W)
        # Register as buffer so it moves with the model and persists in checkpoints
        template_size = num_classes * k_subpatterns * 1 * img_res[0] * img_res[1] * 4 / (1024**2)
        print(f"\nLDDMM Pipeline Initialized:")
        print(f"  Classes: {num_classes}")
        print(f"  Sub-patterns per class: {k_subpatterns}")
        print(f"  Resolution: {img_res}")
        print(f"  Total templates: {num_classes * k_subpatterns}")
        print(f"  Template bank size: {template_size:.1f} MB")

        self.register_buffer('templates', torch.zeros((num_classes, k_subpatterns, 1, *img_res)))
        self.register_buffer('template_counts', torch.zeros(num_classes, k_subpatterns))

    def forward(self, h_i, class_ids=None, update_templates=True):
        """
        Forward pass: align saliency maps to canonical space.

        Args:
            h_i: (B, 1, H, W) - Input saliency maps
            class_ids: (B,) - Class labels (optional, for template update)
            update_templates: bool - Whether to update template bank

        Returns:
            h_aligned: (B, 1, H, W) - Aligned saliency maps
            cluster_probs: (B, K) - Soft cluster assignments
            phi: (B, 2, H, W) - Diffeomorphisms
        """
        B = h_i.shape[0]

        # Step 1: Predict velocity field and cluster assignment
        cluster_logits, v_low = self.predictor(h_i)
        cluster_probs = F.softmax(cluster_logits, dim=-1)
        cluster_assigned = torch.argmax(cluster_probs, dim=-1)  # Hard assignment for updates

        # Step 2: Compute diffeomorphism (Exp map)
        phi = self.warper(v_low)

        # Step 3: Warp saliency map to canonical space
        h_aligned = self.warper.warp_image(h_i, phi)

        # Step 4: Update template bank (if training)
        if update_templates and class_ids is not None:
            self._update_template_bank(h_aligned, class_ids, cluster_assigned)

        return h_aligned, cluster_probs, phi

    @torch.no_grad()
    def _update_template_bank(self, h_aligned, class_ids, cluster_assigned):
        """
        Update template bank using running average (in-place on GPU).

        Args:
            h_aligned: (B, 1, H, W) - Aligned saliency maps on GPU
            class_ids: (B,) - Class indices
            cluster_assigned: (B,) - Assigned cluster indices
        """
        for i in range(h_aligned.shape[0]):
            c = class_ids[i].item()
            k = cluster_assigned[i].item()

            # Increment count
            self.template_counts[c, k] += 1
            count = self.template_counts[c, k].item()

            # Running average: T = (1 - eta) * T + eta * h_aligned
            # Use 1/count for true mean
            eta = 1.0 / count

            # Update in-place on GPU (templates buffer is already on GPU)
            self.templates[c, k] = (1 - eta) * self.templates[c, k] + eta * h_aligned[i]

    def get_template(self, class_id, cluster_id=None):
        """
        Get template(s) for a class.

        Args:
            class_id: int - Class index
            cluster_id: int or None - Specific cluster (None = all clusters)

        Returns:
            Template tensor
        """
        if cluster_id is not None:
            return self.templates[class_id, cluster_id]
        else:
            return self.templates[class_id]  # All K sub-patterns

    def get_batch_templates(self, class_ids):
        """
        Get templates for a batch of classes (optimized for GPU).

        Args:
            class_ids: (B,) - Tensor of class indices

        Returns:
            (B, K, 1, H, W) - Templates for each class in the batch
        """
        # Use advanced indexing to gather templates efficiently
        return self.templates[class_ids]  # (B, K, 1, H, W)

    def get_dominant_template(self, class_id):
        """Get the most frequently used template for a class."""
        counts = self.template_counts[class_id]
        dominant_k = torch.argmax(counts).item()
        return self.templates[class_id, dominant_k], dominant_k


# ==========================================
# Loss Functions
# ==========================================

class LDDMMLoss(nn.Module):
    """
    Combined loss for LDDMM training.

    Components:
    1. Alignment Loss: L2 distance between aligned map and template
    2. Smoothness Loss: Regularization on velocity field gradients
    3. Entropy Loss: Encourage confident cluster assignments
    """

    def __init__(self, lambda_smooth=0.01, lambda_entropy=0.001):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_entropy = lambda_entropy

    def alignment_loss(self, h_aligned, templates, cluster_probs):
        """
        Weighted alignment loss to all templates.

        Args:
            h_aligned: (B, 1, H, W)
            templates: (B, K, 1, H, W) - Templates for this batch
            cluster_probs: (B, K) - Soft assignments

        Returns:
            loss: scalar
        """
        B, K = cluster_probs.shape

        # Expand h_aligned to match templates
        h_expanded = h_aligned.unsqueeze(1).expand(-1, K, -1, -1, -1)  # (B, K, 1, H, W)

        # Compute L2 distance to each template
        distances = ((h_expanded - templates) ** 2).mean(dim=(2, 3, 4))  # (B, K)

        # Weight by cluster probabilities
        weighted_dist = (distances * cluster_probs).sum(dim=1)  # (B,)

        return weighted_dist.mean()

    def smoothness_loss(self, v_field):
        """
        Penalize spatial gradients of velocity field (encourage smooth deformations).

        Args:
            v_field: (B, 2, H, W)

        Returns:
            loss: scalar
        """
        # Compute spatial gradients
        diff_h = torch.abs(v_field[:, :, 1:, :] - v_field[:, :, :-1, :])
        diff_w = torch.abs(v_field[:, :, :, 1:] - v_field[:, :, :, :-1])

        smooth_loss = diff_h.mean() + diff_w.mean()
        return smooth_loss

    def entropy_loss(self, cluster_probs):
        """
        Entropy regularization to encourage confident assignments.

        Args:
            cluster_probs: (B, K)

        Returns:
            loss: scalar
        """
        # H(p) = -sum(p * log(p))
        entropy = -(cluster_probs * torch.log(cluster_probs + 1e-10)).sum(dim=1)
        return entropy.mean()

    def forward(self, h_aligned, templates, cluster_probs, v_field):
        """
        Compute total loss.

        Args:
            h_aligned: (B, 1, H, W)
            templates: (B, K, 1, H, W)
            cluster_probs: (B, K)
            v_field: (B, 2, H, W)

        Returns:
            loss_dict: Dictionary of loss components
        """
        loss_align = self.alignment_loss(h_aligned, templates, cluster_probs)
        loss_smooth = self.smoothness_loss(v_field)
        loss_entropy = self.entropy_loss(cluster_probs)

        total_loss = (
            loss_align +
            self.lambda_smooth * loss_smooth +
            self.lambda_entropy * loss_entropy
        )

        return {
            'total': total_loss,
            'alignment': loss_align,
            'smoothness': loss_smooth,
            'entropy': loss_entropy
        }


# ==========================================
# Dataset
# ==========================================

class SaliencyMapDataset(Dataset):
    """Dataset for loading pre-computed saliency maps with lazy loading.

    This version uses lazy loading to avoid loading all data into CPU memory at once.
    Instead, it builds an index and loads batches on-demand.
    """

    def __init__(self, data_dir, max_samples_per_class=None):
        self.data_dir = Path(data_dir)
        self.saliency_dir = self.data_dir / "saliency_maps"

        # Load metadata
        metadata_path = self.data_dir / "metadata.pkl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        self.metadata = joblib.load(metadata_path)

        # Index batch files (don't load data yet)
        self.batch_files = sorted(self.saliency_dir.glob("batch_*.pkl"))
        print(f"Indexing {len(self.batch_files)} batch files...")

        # Build index: (batch_file_idx, item_idx_in_batch) -> label
        self.sample_index = []
        class_counts = defaultdict(int)

        for batch_idx, batch_file in enumerate(tqdm(self.batch_files, desc="Indexing batches")):
            batch_data = joblib.load(batch_file)

            for item_idx, item in enumerate(batch_data):
                label = item['true_label']

                # Limit samples per class if specified
                if max_samples_per_class is None or class_counts[label] < max_samples_per_class:
                    self.sample_index.append({
                        'batch_idx': batch_idx,
                        'item_idx': item_idx,
                        'label': label
                    })
                    class_counts[label] += 1

        print(f"Indexed {len(self.sample_index)} samples")
        print(f"Classes represented: {len(class_counts)}")
        print(f"Memory footprint: ~{len(self.sample_index) * 24 / 1024 / 1024:.2f} MB (index only)")

        # Cache for loaded batches (LRU cache to avoid reloading)
        self._batch_cache = {}
        # Keep more batches in memory since we have plenty of RAM
        # Each batch ~100 samples × 224×224×4 bytes ≈ 20MB
        # 50 batches ≈ 1GB which is reasonable
        self._cache_size = 50  # Keep 50 batches in memory
        print(f"Batch cache size: {self._cache_size} batches (~{self._cache_size * 20} MB)")

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        index_entry = self.sample_index[idx]
        batch_idx = index_entry['batch_idx']
        item_idx = index_entry['item_idx']

        # Load batch from cache or disk
        if batch_idx not in self._batch_cache:
            # Evict oldest batch if cache is full
            if len(self._batch_cache) >= self._cache_size:
                oldest_key = next(iter(self._batch_cache))
                del self._batch_cache[oldest_key]

            # Load batch from disk
            batch_file = self.batch_files[batch_idx]
            self._batch_cache[batch_idx] = joblib.load(batch_file)

        # Get item from cached batch
        item = self._batch_cache[batch_idx][item_idx]

        # Convert to tensor and add channel dimension
        saliency = torch.from_numpy(item['saliency_map']).float().unsqueeze(0)  # (1, H, W)
        label = torch.tensor(item['true_label'], dtype=torch.long)

        return saliency, label


# ==========================================
# Training
# ==========================================

class LDDMMTrainer:
    """Trainer for LDDMM pipeline."""

    def __init__(self, model, train_loader, val_loader=None,
                 lr=1e-3, device='cuda', checkpoint_dir='./checkpoints'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)

        # Loss
        self.criterion = LDDMMLoss(lambda_smooth=0.01, lambda_entropy=0.001)

        # History
        self.history = {
            'train_loss': [],
            'train_alignment': [],
            'train_smoothness': [],
            'train_entropy': []
        }

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()

        epoch_losses = defaultdict(float)
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, (saliency_maps, labels) in enumerate(pbar):
            saliency_maps = saliency_maps.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Forward pass
            h_aligned, cluster_probs, phi = self.model(
                saliency_maps, class_ids=labels, update_templates=True
            )

            # Get templates for this batch (optimized batched access on GPU)
            batch_templates = self.model.get_batch_templates(labels)  # (B, K, 1, H, W)

            # Compute velocity field (for smoothness loss)
            _, v_low = self.model.predictor(saliency_maps)

            # Compute loss
            losses = self.criterion(h_aligned, batch_templates, cluster_probs, phi)

            # Backward
            self.optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Accumulate losses
            for key, value in losses.items():
                epoch_losses[key] += value.item()
            num_batches += 1

            # Update progress bar with GPU memory usage
            postfix_dict = {
                'loss': losses['total'].item(),
                'align': losses['alignment'].item(),
            }
            if torch.cuda.is_available():
                gpu_mem_used = torch.cuda.memory_allocated() / (1024**3)
                postfix_dict['GPU_GB'] = f'{gpu_mem_used:.1f}'
            pbar.set_postfix(postfix_dict)

            # Periodic GPU memory cleanup
            if batch_idx % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Average losses
        avg_losses = {key: value / num_batches for key, value in epoch_losses.items()}

        # Update history
        self.history['train_loss'].append(avg_losses['total'])
        self.history['train_alignment'].append(avg_losses['alignment'])
        self.history['train_smoothness'].append(avg_losses['smoothness'])
        self.history['train_entropy'].append(avg_losses['entropy'])

        return avg_losses

    def train(self, num_epochs, save_frequency=5):
        """Train for multiple epochs."""
        print(f"\n{'='*80}")
        print(f"Starting LDDMM Training")
        print(f"{'='*80}")
        print(f"Epochs: {num_epochs}")
        print(f"Device: {self.device}")

        for epoch in range(1, num_epochs + 1):
            avg_losses = self.train_epoch(epoch)

            print(f"\nEpoch {epoch}/{num_epochs} Summary:")
            print(f"  Total Loss: {avg_losses['total']:.4f}")
            print(f"  Alignment: {avg_losses['alignment']:.4f}")
            print(f"  Smoothness: {avg_losses['smoothness']:.4f}")
            print(f"  Entropy: {avg_losses['entropy']:.4f}")

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
            checkpoint_path = self.checkpoint_dir / "lddmm_model_final.pth"
        else:
            checkpoint_path = self.checkpoint_dir / f"lddmm_model_epoch_{epoch}.pth"

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'templates': self.model.templates,
            'template_counts': self.model.template_counts
        }, checkpoint_path)

        print(f"  Checkpoint saved: {checkpoint_path}")

    def plot_training_curves(self, save_path='training_curves.png'):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].plot(self.history['train_loss'])
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].grid(True)

        axes[0, 1].plot(self.history['train_alignment'])
        axes[0, 1].set_title('Alignment Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].grid(True)

        axes[1, 0].plot(self.history['train_smoothness'])
        axes[1, 0].set_title('Smoothness Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].grid(True)

        axes[1, 1].plot(self.history['train_entropy'])
        axes[1, 1].set_title('Entropy Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Training curves saved to {save_path}")


# ==========================================
# Main Script
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description='Geodesic Pattern Learning via Amortized LDDMM'
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
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--max_samples_per_class', type=int, default=None,
                       help='Max samples per class (default: None = all)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"Geodesic Pattern Learning via LDDMM")
    print(f"{'='*80}")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total GPU Memory: {total_mem:.1f} GB")
        torch.cuda.empty_cache()  # Clear cache before starting

    # Load dataset
    print(f"\nLoading saliency maps from {args.data_dir}...")
    dataset = SaliencyMapDataset(
        data_dir=args.data_dir,
        max_samples_per_class=args.max_samples_per_class
    )

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        prefetch_factor=2 if args.num_workers > 0 else None,  # Prefetch 2 batches per worker
        persistent_workers=True if args.num_workers > 0 else False  # Keep workers alive between epochs
    )

    print(f"\nDataLoader Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num workers: {args.num_workers}")
    print(f"  Pin memory: {True if torch.cuda.is_available() else False}")
    print(f"  Prefetch factor: {2 if args.num_workers > 0 else None}")

    # Create model
    print(f"\nInitializing LDDMM model...")
    model = LDDMM_GlobalPatternPipeline(
        num_classes=args.num_classes,
        k_subpatterns=args.k_subpatterns,
        img_res=(224, 224),
        device=device
    )

    # Create trainer
    trainer = LDDMMTrainer(
        model=model,
        train_loader=train_loader,
        lr=args.lr,
        device=device,
        checkpoint_dir=args.checkpoint_dir
    )

    # Print expected memory usage
    if torch.cuda.is_available():
        print(f"\nEstimated GPU Memory Usage:")
        template_mem = args.num_classes * args.k_subpatterns * 1 * 224 * 224 * 4 / (1024**3)
        batch_mem = args.batch_size * 1 * 224 * 224 * 4 / (1024**3)
        model_mem = sum(p.numel() * 4 for p in model.parameters()) / (1024**3)
        print(f"  Template bank: ~{template_mem:.2f} GB")
        print(f"  Model parameters: ~{model_mem:.2f} GB")
        print(f"  Batch data: ~{batch_mem:.2f} GB")
        print(f"  Estimated total: ~{template_mem + model_mem + batch_mem * 3:.2f} GB")
        print(f"  Available: {total_mem:.1f} GB")

    # Train
    trainer.train(num_epochs=args.epochs, save_frequency=5)

    # Plot training curves
    trainer.plot_training_curves(
        save_path=Path(args.checkpoint_dir) / 'training_curves.png'
    )

    if torch.cuda.is_available():
        print(f"\nFinal GPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")

    print(f"\n✓ All done!")


if __name__ == "__main__":
    main()
