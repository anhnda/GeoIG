"""
Advanced Geodesic Pattern Learning v3.1 - Enhanced Signal Preservation for Thin Structures

KEY INNOVATION (V3.1):
Addresses the "mass body dominance" problem where thick body regions overshadow
thin but important features (neck, legs) in the aligned patterns.

SOLUTION - Multi-Scale Signal Enhancement:
1. Adaptive Log-Power Normalization: Compresses dynamic range to preserve weak signals
   - Before alignment: Apply log(1 + x^α) to reduce mass body dominance
   - α ∈ [0.3, 0.7]: Power < 1 further compresses high values

2. Percentile-Based Normalization: Prevent global max from dominating
   - Normalize by 95th percentile instead of max
   - Allows weak signals to have meaningful magnitude

3. Local Contrast Enhancement: Boost signal-to-noise for thin structures
   - Apply CLAHE-like approach: divide by local mean, then renormalize

4. Template Update with Signal-Aware Weighting:
   - Instead of top-K (which favors concentrated regions)
   - Use signal-strength weighted averaging that preserves extended structures

REMOVED from v3:
- Edge gating (as requested - not the right approach for this problem)
- Compactness loss (was favoring blob-like shapes over extended structures)
- Top-K sharpening (was removing thin features)

Based on: geo_s3.py
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
from functools import lru_cache


# ==========================================
# Signal Enhancement Module (NEW in v3.1)
# ==========================================

class AdaptiveSignalEnhancer(nn.Module):
    """
    Enhance weak signals (thin structures) without edge gating.

    Uses adaptive normalization to prevent mass body from dominating:
    1. Log-power transform: log(1 + x^α) compresses dynamic range
    2. Percentile normalization: Use 95th percentile instead of max
    3. Local contrast enhancement: Boost relative signal strength
    """

    def __init__(self, alpha=0.5, percentile=95.0, local_window=15):
        super().__init__()
        self.alpha = alpha  # Power for compression (< 1 compresses high values more)
        self.percentile = percentile  # Normalization percentile
        self.local_window = local_window  # Window for local contrast

        # Local averaging filter (for contrast enhancement)
        self.register_buffer('local_avg_kernel',
                           torch.ones(1, 1, local_window, local_window) / (local_window ** 2))

    def forward(self, saliency, enhance_mode='log_power'):
        """
        Args:
            saliency: (B, 1, H, W) - Input saliency map
            enhance_mode: 'log_power', 'percentile', 'local_contrast', 'combined', or 'none'

        Returns:
            enhanced: (B, 1, H, W) - Enhanced saliency map
        """
        if enhance_mode == 'none':
            return saliency
        elif enhance_mode == 'log_power':
            return self._log_power_transform(saliency)
        elif enhance_mode == 'percentile':
            return self._percentile_normalize(saliency)
        elif enhance_mode == 'local_contrast':
            return self._local_contrast_enhance(saliency)
        elif enhance_mode == 'combined':
            # Apply all three in sequence
            x = self._log_power_transform(saliency)
            x = self._local_contrast_enhance(x)
            x = self._percentile_normalize(x)
            return x
        else:
            return saliency

    def _log_power_transform(self, x):
        """
        Log-power transform: y = log(1 + x^α)

        - α < 1: Compresses high values more than low values
        - log: Further compresses dynamic range
        - Effect: Mass body gets compressed, thin features become more visible
        """
        # Apply power transform
        x_powered = torch.pow(x + 1e-8, self.alpha)

        # Apply log transform
        x_log = torch.log(1.0 + x_powered)

        # Normalize to [0, 1]
        x_min = x_log.amin(dim=(1, 2, 3), keepdim=True)
        x_max = x_log.amax(dim=(1, 2, 3), keepdim=True)
        x_norm = (x_log - x_min) / (x_max - x_min + 1e-8)

        return x_norm

    def _percentile_normalize(self, x):
        """
        Normalize by percentile instead of max.

        - Prevents a few extreme pixels from dominating
        - Allows weak signals to have meaningful magnitude
        """
        B = x.shape[0]
        enhanced = []

        for i in range(B):
            x_i = x[i]
            # Compute percentile
            perc_val = torch.quantile(x_i.flatten(), self.percentile / 100.0)

            # Clip and normalize
            x_clipped = torch.clamp(x_i, max=perc_val)
            x_norm = x_clipped / (perc_val + 1e-8)
            enhanced.append(x_norm)

        return torch.stack(enhanced, dim=0)

    def _local_contrast_enhance(self, x):
        """
        Local contrast enhancement (CLAHE-like).

        - Divide by local mean to boost relative signal
        - Helps thin structures stand out from background
        """
        # Compute local average
        local_mean = F.conv2d(x, self.local_avg_kernel, padding=self.local_window // 2)

        # Divide by local mean (with floor to prevent division by zero)
        local_mean_floor = torch.clamp(local_mean, min=0.01)
        contrast_enhanced = x / local_mean_floor

        # Renormalize
        x_min = contrast_enhanced.amin(dim=(1, 2, 3), keepdim=True)
        x_max = contrast_enhanced.amax(dim=(1, 2, 3), keepdim=True)
        normalized = (contrast_enhanced - x_min) / (x_max - x_min + 1e-8)

        return normalized


# ==========================================
# Multi-Scale Coarse-to-Fine Predictor
# ==========================================

class MultiScalePredictor(nn.Module):
    """
    Hierarchical velocity field predictor:
    Stage 1: Coarse (14×14) - Align global mass/location
    Stage 2: Fine (112×112) - Align local details
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

        # Head 2a: COARSE Velocity Field (14×14)
        self.coarse_v_res = (14, 14)
        self.coarse_v_dim = 2 * self.coarse_v_res[0] * self.coarse_v_res[1]
        self.coarse_v_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.coarse_v_dim),
            nn.Tanh()
        )

        # Head 2b: FINE Velocity Field (112×112)
        self.fine_v_res = (112, 112)
        self.fine_v_dim = 2 * self.fine_v_res[0] * self.fine_v_res[1]
        self.fine_v_intermediate = max(2048, self.fine_v_dim // 4)

        self.fine_v_head = nn.Sequential(
            nn.Linear(feature_dim, self.fine_v_intermediate),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.fine_v_intermediate, self.fine_v_dim),
            nn.Tanh()
        )

        # Scaling factors
        self.coarse_scale = 3.0
        self.fine_scale = 0.5

        # Gaussian blur for coarse stage
        self.gaussian_blur = kornia.filters.GaussianBlur2d((11, 11), (3.0, 3.0))

        # Initialize to near-zero
        nn.init.normal_(self.coarse_v_head[-2].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.coarse_v_head[-2].bias)
        nn.init.normal_(self.fine_v_head[3].weight, mean=0.0, std=0.005)
        nn.init.zeros_(self.fine_v_head[3].bias)

    def forward(self, h_i):
        """
        Args:
            h_i: (B, 1, H, W) - Saliency maps (already enhanced)

        Returns:
            cluster_logits: (B, K)
            v_coarse: (B, 2, 14, 14)
            v_fine: (B, 2, 112, 112)
            h_i_blurred: (B, 1, H, W)
        """
        B = h_i.shape[0]

        # Create blurred version for coarse alignment
        h_i_blurred = self.gaussian_blur(h_i)

        # Extract features
        features = self.backbone(h_i)

        # Predict cluster assignment
        cluster_logits = self.assignment_head(features)

        # Predict COARSE velocity field
        features_blurred = self.backbone(h_i_blurred)
        v_coarse_flat = self.coarse_v_head(features_blurred)
        v_coarse = v_coarse_flat.view(B, 2, *self.coarse_v_res) * self.coarse_scale

        # Predict FINE residual velocity field
        v_fine_flat = self.fine_v_head(features)
        v_fine = v_fine_flat.view(B, 2, *self.fine_v_res) * self.fine_scale

        return cluster_logits, v_coarse, v_fine, h_i_blurred


# ==========================================
# Diffeomorphic Warper
# ==========================================

class DiffeomorphicWarper(nn.Module):
    """Scaling-and-squaring exponential map for diffeomorphic warping."""

    def __init__(self, img_res=(224, 224), num_steps=7):
        super().__init__()
        self.res = img_res
        self.num_steps = num_steps

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

        # Upsample velocity field to full resolution
        v = F.interpolate(v_low, size=self.res, mode='bilinear', align_corners=True)

        # Scale by 2^(-num_steps)
        v = v / (2 ** self.num_steps)

        # Scaling-and-squaring
        for _ in range(self.num_steps):
            flow = v.permute(0, 2, 3, 1)  # (B, H, W, 2)
            identity = self.identity_grid.expand(B, -1, -1, -1)
            grid = identity + flow

            v_warped = F.grid_sample(
                v, grid, mode='bilinear',
                padding_mode='border', align_corners=True
            )

            v = v + v_warped

        return v  # (B, 2, H, W)

    def warp_image(self, image, phi, preserve_mass=True):
        """Warp image using diffeomorphism with mass conservation."""
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

        # Enforce mass conservation
        if preserve_mass:
            mass_original = image.sum(dim=(1, 2, 3), keepdim=True)
            mass_warped = warped.sum(dim=(1, 2, 3), keepdim=True)
            scale_factor = mass_original / (mass_warped + 1e-10)
            warped = warped * scale_factor

        return warped


# ==========================================
# Sinkhorn-Knopp Optimal Transport
# ==========================================

def sinkhorn_barycenter(samples, weights=None, reg=0.1, num_iter=50):
    """
    Compute Wasserstein barycenter using Sinkhorn-Knopp algorithm.

    Args:
        samples: (B, 1, H, W) - Batch of aligned saliency maps
        weights: (B,) - Optional sample weights
        reg: float - Entropic regularization strength
        num_iter: int - Number of Sinkhorn iterations

    Returns:
        barycenter: (1, H, W) - Optimal transport barycenter
    """
    B, _, H, W = samples.shape

    if weights is None:
        weights = torch.ones(B, device=samples.device) / B
    else:
        weights = weights / weights.sum()

    # Flatten samples: (B, H*W)
    samples_flat = samples.view(B, -1)

    # Initialize barycenter as weighted average
    barycenter = (weights.view(B, 1) * samples_flat).sum(dim=0)  # (H*W,)

    # Normalize to probability distributions
    samples_flat = samples_flat / (samples_flat.sum(dim=1, keepdim=True) + 1e-10)
    barycenter = barycenter / (barycenter.sum() + 1e-10)

    # Sinkhorn iterations
    for _ in range(num_iter):
        # Update barycenter as weighted geometric mean
        log_samples = torch.log(samples_flat + 1e-10)
        log_barycenter = (weights.view(B, 1) * log_samples).sum(dim=0)
        barycenter = torch.exp(log_barycenter)
        barycenter = barycenter / (barycenter.sum() + 1e-10)

    # Reshape back
    barycenter = barycenter.view(1, H, W)

    return barycenter


# ==========================================
# Complete Pipeline with Signal Enhancement
# ==========================================

class AdvancedLDDMM_Pipeline(nn.Module):
    """
    Advanced LDDMM pipeline v3.1 with adaptive signal enhancement.

    V3.1 CHANGES:
    - Added AdaptiveSignalEnhancer to preserve thin structures
    - Removed edge gating (not the right approach)
    - Removed top-K sharpening (was removing thin features)
    - Reduced compactness loss (was favoring blobs over extended structures)
    - Template updates use signal-aware weighting
    """

    def __init__(self, num_classes=1000, k_subpatterns=10, img_res=(224, 224),
                 device='cuda', temperature=0.1, use_ot=True,
                 enhance_mode='log_power', alpha=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.K = k_subpatterns
        self.res = img_res
        self.device = device
        self.temperature = temperature
        self.use_ot = use_ot
        self.enhance_mode = enhance_mode

        # 1. Signal Enhancer (NEW in v3.1)
        self.signal_enhancer = AdaptiveSignalEnhancer(alpha=alpha, percentile=95.0, local_window=15)

        # 2. Multi-scale Predictor
        self.predictor = MultiScalePredictor(k_subpatterns=k_subpatterns, img_res=img_res)

        # 3. Warper
        self.warper = DiffeomorphicWarper(img_res=img_res, num_steps=7)

        # 4. Template Bank
        templates_init = torch.rand((num_classes, k_subpatterns, 1, *img_res)) * 0.1
        templates_init = torch.where(templates_init > 0.07, templates_init, torch.zeros_like(templates_init))
        self.register_buffer('templates', templates_init)
        self.register_buffer('template_counts', torch.zeros(num_classes, k_subpatterns))

        print(f"\nAdvanced LDDMM Pipeline v3.1 Initialized:")
        print(f"  Signal Enhancement: {enhance_mode} (alpha={alpha})")
        print(f"  Multi-scale: Coarse (14×14) → Fine (112×112)")
        print(f"  Template update: {'Optimal Transport (Signal-Aware)' if use_ot else 'Running Average'}")
        print(f"  Classes: {num_classes}, Sub-patterns: {k_subpatterns}")

    def forward(self, h_i, class_ids=None, update_templates=True, original_images=None):
        """
        Forward pass with signal enhancement and multi-scale alignment.

        Args:
            h_i: (B, 1, H, W) - Input saliency maps
            class_ids: (B,) - Class labels
            update_templates: bool - Whether to update template bank
            original_images: Not used in v3.1 (no edge gating)

        Returns:
            h_aligned: (B, 1, H, W) - Aligned saliency maps
            cluster_probs: (B, K)
            phi_coarse: (B, 2, H, W)
            phi_fine: (B, 2, H, W)
            v_coarse: (B, 2, 14, 14)
            v_fine: (B, 2, 112, 112)
        """
        B = h_i.shape[0]

        # Step 0: Apply signal enhancement to preserve thin structures
        h_i_enhanced = self.signal_enhancer(h_i, enhance_mode=self.enhance_mode)

        # Step 1: Predict cluster assignment and multi-scale velocity fields
        cluster_logits, v_coarse, v_fine, h_i_blurred = self.predictor(h_i_enhanced)
        cluster_probs = F.softmax(cluster_logits / self.temperature, dim=-1)
        cluster_assigned = torch.argmax(cluster_probs, dim=-1)

        # Step 2: COARSE alignment
        phi_coarse = self.warper(v_coarse)
        h_coarse_aligned = self.warper.warp_image(h_i_blurred, phi_coarse)

        # Step 3: FINE alignment
        phi_fine = self.warper(v_fine)
        h_aligned = self.warper.warp_image(h_coarse_aligned, phi_fine)

        # Step 4: Update template bank
        if update_templates and class_ids is not None:
            if self.use_ot:
                self._update_template_bank_ot_signal_aware(h_aligned, class_ids, cluster_assigned)
            else:
                self._update_template_bank_avg(h_aligned, class_ids, cluster_assigned)

        return h_aligned, cluster_probs, phi_coarse, phi_fine, v_coarse, v_fine

    @torch.no_grad()
    def _update_template_bank_ot_signal_aware(self, h_aligned, class_ids, cluster_assigned,
                                              sparsity_percentile=50):
        """
        Update template bank using OT with signal-aware weighting.

        V3.1 CHANGES:
        - Removed top-K sharpening (was removing thin features)
        - Reduced sparsity threshold (90% -> 75%) to preserve more structure
        - Use gentle sparsity instead of aggressive cutting
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

            # Limit samples to prevent OOM
            if samples.shape[0] > 16:
                indices = torch.randperm(samples.shape[0], device=samples.device)[:16]
                samples = samples[indices]

            # Compute Wasserstein barycenter
            barycenter = sinkhorn_barycenter(samples, reg=0.02, num_iter=10)

            del samples

            # GENTLE sparsity thresholding (preserve more structure)
            threshold = torch.quantile(barycenter.flatten(), sparsity_percentile / 100.0)
            barycenter = torch.where(
                barycenter >= threshold,
                barycenter,
                torch.zeros_like(barycenter)
            )

            # Normalize
            barycenter_max = barycenter.max()
            if barycenter_max > 0:
                barycenter = barycenter / barycenter_max

            # Blend with existing template
            count = self.template_counts[c, k]
            if count == 0:
                self.templates[c, k] = barycenter
            else:
                eta = torch.clamp(torch.tensor(0.1, device=count.device),
                                  max=1.0 / (count + 1))
                self.templates[c, k] = (1 - eta) * self.templates[c, k] + eta * barycenter

            self.template_counts[c, k] += mask.sum()

            del barycenter

    @torch.no_grad()
    def _update_template_bank_avg(self, h_aligned, class_ids, cluster_assigned, sparsity_percentile=50):
        """Running average update with gentle sparsity."""
        B = h_aligned.shape[0]

        for i in range(B):
            c, k = class_ids[i], cluster_assigned[i]
            self.template_counts[c, k] += 1

            count = self.template_counts[c, k]
            eta = 1.0 / count

            updated_template = (1 - eta) * self.templates[c, k] + eta * h_aligned[i]

            threshold = torch.quantile(updated_template.flatten(), sparsity_percentile / 100.0)
            updated_template = torch.where(
                updated_template >= threshold,
                updated_template,
                torch.zeros_like(updated_template)
            )

            if updated_template.max() > 0:
                updated_template = updated_template / updated_template.max()

            self.templates[c, k] = updated_template

    def get_batch_templates(self, class_ids):
        """Get templates for a batch of classes."""
        return self.templates[class_ids]  # (B, K, 1, H, W)


# ==========================================
# Enhanced Loss Functions
# ==========================================

class AdvancedLDDMMLoss(nn.Module):
    """
    Loss functions for v3.1.

    V3.1 CHANGES:
    - Reduced compactness penalty (10.0 -> 2.0) to allow extended structures
    - Reduced template sparsity (5.0 -> 2.0) to preserve thin features
    - Kept smoothness low to preserve details
    """

    def __init__(self, lambda_smooth=0.05, lambda_entropy=1.0, lambda_magnitude=0.00005,
                 lambda_diversity=2.0, lambda_template_diversity=2.0, lambda_template_sparsity=2.0,
                 lambda_spatial_diversity=0.3, lambda_compactness=2.0, lambda_mass_conservation=5.0,
                 lambda_sparsity_match=10.0, lambda_tv=0.5, lambda_jacobian=75.0,
                 lambda_coarse_smooth=0.03):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_entropy = lambda_entropy
        self.lambda_magnitude = lambda_magnitude
        self.lambda_diversity = lambda_diversity
        self.lambda_template_diversity = lambda_template_diversity
        self.lambda_template_sparsity = lambda_template_sparsity
        self.lambda_spatial_diversity = lambda_spatial_diversity
        self.lambda_compactness = lambda_compactness
        self.lambda_mass_conservation = lambda_mass_conservation
        self.lambda_sparsity_match = lambda_sparsity_match
        self.lambda_tv = lambda_tv
        self.lambda_jacobian = lambda_jacobian
        self.lambda_coarse_smooth = lambda_coarse_smooth

    def alignment_loss(self, h_aligned, templates, cluster_probs):
        """Hard assignment alignment loss."""
        B, K = cluster_probs.shape
        cluster_assigned = torch.argmax(cluster_probs, dim=-1)
        batch_idx = torch.arange(B, device=h_aligned.device)
        assigned_templates = templates[batch_idx, cluster_assigned]
        distances = ((h_aligned - assigned_templates) ** 2).mean(dim=(1, 2, 3))
        return distances.mean()

    def smoothness_loss(self, v_field):
        """Spatial gradient penalty."""
        diff_h = torch.abs(v_field[:, :, 1:, :] - v_field[:, :, :-1, :])
        diff_w = torch.abs(v_field[:, :, :, 1:] - v_field[:, :, :, :-1])
        return diff_h.mean() + diff_w.mean()

    def mass_conservation_loss(self, h_original, h_aligned):
        """Enforce total intensity preservation."""
        mass_original = h_original.sum(dim=(1, 2, 3))
        mass_aligned = h_aligned.sum(dim=(1, 2, 3))
        relative_change = torch.abs(mass_aligned - mass_original) / (mass_original + 1e-8)
        return relative_change.mean()

    def entropy_loss(self, cluster_probs):
        """Entropy regularization."""
        entropy = -(cluster_probs * torch.log(cluster_probs + 1e-10)).sum(dim=1)
        return entropy.mean()

    def diversity_loss(self, cluster_probs):
        """Diversity loss."""
        avg_cluster_usage = cluster_probs.mean(dim=0)
        cluster_entropy = -(avg_cluster_usage * torch.log(avg_cluster_usage + 1e-10)).sum()
        max_entropy = np.log(cluster_probs.shape[1])
        return max_entropy - cluster_entropy

    def magnitude_loss(self, v_field):
        """L2 penalty on velocity field magnitude."""
        magnitude = (v_field ** 2).mean()
        return magnitude

    def jacobian_determinant_loss(self, v_field):
        """Penalize extreme local volume changes."""
        dphi_x_dx = v_field[:, 0:1, :, 1:] - v_field[:, 0:1, :, :-1]
        dphi_x_dy = v_field[:, 0:1, 1:, :] - v_field[:, 0:1, :-1, :]
        dphi_y_dx = v_field[:, 1:2, :, 1:] - v_field[:, 1:2, :, :-1]
        dphi_y_dy = v_field[:, 1:2, 1:, :] - v_field[:, 1:2, :-1, :]

        min_h = min(dphi_x_dx.shape[2], dphi_x_dy.shape[2], dphi_y_dx.shape[2], dphi_y_dy.shape[2])
        min_w = min(dphi_x_dx.shape[3], dphi_x_dy.shape[3], dphi_y_dx.shape[3], dphi_y_dy.shape[3])

        dphi_x_dx = dphi_x_dx[:, :, :min_h, :min_w]
        dphi_x_dy = dphi_x_dy[:, :, :min_h, :min_w]
        dphi_y_dx = dphi_y_dx[:, :, :min_h, :min_w]
        dphi_y_dy = dphi_y_dy[:, :, :min_h, :min_w]

        det_J = (1 + dphi_x_dx) * (1 + dphi_y_dy) - dphi_x_dy * dphi_y_dx
        jac_loss = ((det_J - 1) ** 2).mean()
        return jac_loss

    def template_diversity_loss(self, templates):
        """Penalize templates for being too similar."""
        B, K = templates.shape[0], templates.shape[1]
        templates_flat = templates.reshape(B, K, -1)
        templates_norm = F.normalize(templates_flat, p=2, dim=-1)
        similarity_matrix = torch.bmm(templates_norm, templates_norm.transpose(1, 2))
        mask = torch.eye(K, device=templates.device).unsqueeze(0).expand(B, -1, -1)
        similarity_matrix = similarity_matrix * (1 - mask)
        avg_similarity = torch.abs(similarity_matrix).sum() / (B * K * (K - 1))
        return avg_similarity

    def template_sparsity_loss(self, templates):
        """Encourage templates to be sparse."""
        avg_intensity = torch.abs(templates).mean()
        return avg_intensity

    def spatial_center_diversity_loss(self, templates):
        """Encourage templates to have different spatial centers."""
        B, K, _, H, W = templates.shape
        y_coords = torch.arange(H, device=templates.device, dtype=torch.float32).view(1, 1, 1, H, 1)
        x_coords = torch.arange(W, device=templates.device, dtype=torch.float32).view(1, 1, 1, 1, W)
        templates_norm = templates / (templates.sum(dim=(2, 3, 4), keepdim=True) + 1e-8)
        center_y = (templates_norm * y_coords).sum(dim=(2, 3, 4))
        center_x = (templates_norm * x_coords).sum(dim=(2, 3, 4))
        centers = torch.stack([center_y, center_x], dim=-1)
        centers_expanded1 = centers.unsqueeze(2)
        centers_expanded2 = centers.unsqueeze(1)
        pairwise_distances = torch.norm(centers_expanded1 - centers_expanded2, dim=-1)
        mask = torch.eye(K, device=templates.device).unsqueeze(0).expand(B, -1, -1)
        pairwise_distances = pairwise_distances * (1 - mask)
        avg_distance = pairwise_distances.sum() / (B * K * (K - 1))
        max_distance = np.sqrt(H**2 + W**2)
        return max_distance - avg_distance

    def shape_compactness_loss(self, templates):
        """Encourage compact shapes (REDUCED in v3.1)."""
        B, K, _, H, W = templates.shape
        y_coords = torch.arange(H, device=templates.device, dtype=torch.float32).view(1, 1, 1, H, 1) - H/2
        x_coords = torch.arange(W, device=templates.device, dtype=torch.float32).view(1, 1, 1, 1, W) - W/2
        templates_norm = templates / (templates.sum(dim=(2, 3, 4), keepdim=True) + 1e-8)
        var_y = (templates_norm * y_coords**2).sum(dim=(2, 3, 4))
        var_x = (templates_norm * x_coords**2).sum(dim=(2, 3, 4))
        ratio = torch.maximum(var_y, var_x) / (torch.minimum(var_y, var_x) + 1e-6)
        compactness_penalty = torch.log(ratio + 1.0).mean()
        return compactness_penalty

    def sparsity_matching_loss(self, h_original, h_aligned):
        """Preserve sparsity pattern."""
        threshold = 0.01
        sparsity_original = (h_original > threshold).float().mean(dim=(1, 2, 3))
        sparsity_aligned = (h_aligned > threshold).float().mean(dim=(1, 2, 3))
        sparsity_diff = torch.abs(sparsity_aligned - sparsity_original).mean()
        return sparsity_diff

    def total_variation_loss(self, templates):
        """Reduce fragmentation."""
        diff_h = torch.abs(templates[:, :, :, 1:, :] - templates[:, :, :, :-1, :])
        diff_w = torch.abs(templates[:, :, :, :, 1:] - templates[:, :, :, :, :-1])
        tv = diff_h.mean() + diff_w.mean()
        return tv

    def forward(self, h_original, h_aligned, templates, cluster_probs,
                v_coarse, v_fine):
        """
        Compute total loss.

        Returns:
            loss_dict: Dictionary of loss components
        """
        loss_align = self.alignment_loss(h_aligned, templates, cluster_probs)
        loss_smooth_coarse = self.smoothness_loss(v_coarse)
        loss_smooth_fine = self.smoothness_loss(v_fine)
        loss_entropy = self.entropy_loss(cluster_probs)
        loss_diversity = self.diversity_loss(cluster_probs)
        loss_magnitude_coarse = self.magnitude_loss(v_coarse)
        loss_magnitude_fine = self.magnitude_loss(v_fine)
        loss_template_div = self.template_diversity_loss(templates)
        loss_template_sparse = self.template_sparsity_loss(templates)
        loss_spatial_div = self.spatial_center_diversity_loss(templates)
        loss_compactness = self.shape_compactness_loss(templates)
        loss_mass_cons = self.mass_conservation_loss(h_original, h_aligned)
        loss_sparsity_match = self.sparsity_matching_loss(h_original, h_aligned)
        loss_tv = self.total_variation_loss(templates)
        loss_jacobian_coarse = self.jacobian_determinant_loss(v_coarse)
        loss_jacobian_fine = self.jacobian_determinant_loss(v_fine)

        total_loss = (
            loss_align +
            self.lambda_coarse_smooth * loss_smooth_coarse +
            self.lambda_smooth * loss_smooth_fine +
            self.lambda_entropy * loss_entropy +
            self.lambda_diversity * loss_diversity +
            self.lambda_magnitude * (loss_magnitude_coarse + loss_magnitude_fine) +
            self.lambda_template_diversity * loss_template_div +
            self.lambda_template_sparsity * loss_template_sparse +
            self.lambda_spatial_diversity * loss_spatial_div +
            self.lambda_compactness * loss_compactness +
            self.lambda_mass_conservation * loss_mass_cons +
            self.lambda_sparsity_match * loss_sparsity_match +
            self.lambda_tv * loss_tv +
            self.lambda_jacobian * (loss_jacobian_coarse + loss_jacobian_fine)
        )

        return {
            'total': total_loss,
            'alignment': loss_align,
            'smoothness_coarse': loss_smooth_coarse,
            'smoothness_fine': loss_smooth_fine,
            'entropy': loss_entropy,
            'diversity': loss_diversity,
            'magnitude': loss_magnitude_coarse + loss_magnitude_fine,
            'template_diversity': loss_template_div,
            'template_sparsity': loss_template_sparse,
            'spatial_diversity': loss_spatial_div,
            'compactness': loss_compactness,
            'mass_conservation': loss_mass_cons,
            'sparsity_match': loss_sparsity_match,
            'total_variation': loss_tv,
            'jacobian': loss_jacobian_coarse + loss_jacobian_fine
        }


# ==========================================
# Dataset
# ==========================================

def safe_collate_fn(batch):
    """Custom collate function."""
    if len(batch[0]) == 3:
        saliencies, labels, images = zip(*batch)
    else:
        raise ValueError(f"Unexpected batch item length: {len(batch[0])}")

    for i, sal in enumerate(saliencies):
        if sal.shape != (1, 224, 224):
            raise ValueError(f"Sample {i}: Invalid saliency shape {sal.shape}")

    has_any_none = any(img is None for img in images)

    saliency_batch = torch.stack(saliencies, dim=0)
    label_batch = torch.tensor(labels, dtype=torch.long)

    if not has_any_none and images[0] is not None:
        for i, img in enumerate(images):
            if img.shape != (3, 224, 224):
                raise ValueError(f"Sample {i}: Invalid image shape {img.shape}")
        image_batch = torch.stack(images, dim=0)
    else:
        image_batch = None

    assert saliency_batch.shape[1] == 1

    return saliency_batch, label_batch, image_batch


class SaliencyMapDataset(Dataset):
    """Ultra-optimized RAM dataset."""

    def __init__(self, data_dir, max_samples_per_class=None, load_images=False, max_samples=100000):
        self.data_dir = Path(data_dir)
        self.saliency_dir = self.data_dir / "saliency_maps"
        self.load_images = load_images

        metadata_path = self.data_dir / "metadata.pkl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        self.metadata = joblib.load(metadata_path)

        batch_files = sorted(self.saliency_dir.glob("batch_*.pkl"))
        print(f"Loading {len(batch_files)} batch files into RAM...")

        self.saliency_maps = torch.empty((max_samples, 1, 224, 224), dtype=torch.float16)
        self.labels = torch.empty(max_samples, dtype=torch.long)

        if load_images:
            self.rgb_images = torch.empty((max_samples, 3, 224, 224), dtype=torch.uint8)
        else:
            self.rgb_images = None

        class_counts = defaultdict(int)
        corrupted_samples = 0
        idx = 0

        for batch_file in tqdm(batch_files, desc="Filling RAM buffers"):
            if idx >= max_samples:
                break

            batch_data = joblib.load(batch_file)

            for item in batch_data:
                if idx >= max_samples:
                    break

                label = item.get('true_label')
                if label is None:
                    corrupted_samples += 1
                    continue

                if max_samples_per_class is not None and class_counts[label] >= max_samples_per_class:
                    continue

                if 'saliency_map' not in item:
                    corrupted_samples += 1
                    continue

                saliency_np = item['saliency_map']
                saliency_shape = saliency_np.shape

                if len(saliency_shape) < 2 or saliency_shape[-2:] != (224, 224) or 0 in saliency_shape:
                    corrupted_samples += 1
                    continue

                if len(saliency_np.shape) == 2:
                    saliency_np = saliency_np[np.newaxis, ...]
                elif len(saliency_np.shape) == 3:
                    if saliency_np.shape[0] in [1, 3]:
                        pass
                    elif saliency_np.shape[2] in [1, 3]:
                        saliency_np = saliency_np.transpose(2, 0, 1)
                    else:
                        corrupted_samples += 1
                        continue

                    if saliency_np.shape[0] > 1:
                        saliency_np = saliency_np.mean(axis=0, keepdims=True)

                if saliency_np.shape != (1, 224, 224):
                    corrupted_samples += 1
                    continue

                self.saliency_maps[idx] = torch.from_numpy(saliency_np).to(torch.float16)
                self.labels[idx] = label

                if load_images:
                    if 'rgb_image' not in item:
                        corrupted_samples += 1
                        continue

                    try:
                        rgb_np = item['rgb_image']

                        if len(rgb_np.shape) == 3:
                            if rgb_np.shape[0] == 3:
                                pass
                            elif rgb_np.shape[2] == 3:
                                rgb_np = rgb_np.transpose(2, 0, 1)
                            else:
                                corrupted_samples += 1
                                continue
                        else:
                            corrupted_samples += 1
                            continue

                        if rgb_np.dtype != np.uint8:
                            if rgb_np.max() <= 1.0:
                                rgb_np = (rgb_np * 255).astype(np.uint8)
                            else:
                                rgb_np = rgb_np.astype(np.uint8)

                        self.rgb_images[idx] = torch.from_numpy(rgb_np)
                    except Exception:
                        corrupted_samples += 1
                        continue

                class_counts[label] += 1
                idx += 1

            del batch_data
            gc.collect()

        self.saliency_maps = self.saliency_maps[:idx]
        self.labels = self.labels[:idx]
        if load_images:
            self.rgb_images = self.rgb_images[:idx]

        num_samples = len(self.saliency_maps)
        saliency_mb = num_samples * 1 * 224 * 224 * 2 / (1024**2)
        rgb_mb = num_samples * 3 * 224 * 224 * 1 / (1024**2) if load_images else 0

        print(f"\n✅ Loaded {num_samples} samples ({corrupted_samples} corrupted)")
        print(f"  Total RAM: {saliency_mb + rgb_mb:.1f} MB")

    def __len__(self):
        return len(self.saliency_maps)

    def __getitem__(self, idx):
        saliency = self.saliency_maps[idx].float()
        label = self.labels[idx].item()

        if self.load_images:
            image = self.rgb_images[idx].float() / 255.0
            return saliency, label, image
        else:
            return saliency, label, None


# ==========================================
# Training
# ==========================================

class AdvancedLDDMMTrainer:
    """Trainer for v3.1."""

    def __init__(self, model, train_loader, val_loader=None,
                 lr=1e-3, device='cuda', checkpoint_dir='./checkpoints_s3_v1', use_amp=True):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.use_amp = use_amp and torch.cuda.is_available()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5)

        self.criterion = AdvancedLDDMMLoss(
            lambda_smooth=0.05,
            lambda_entropy=1.0,
            lambda_magnitude=0.00005,
            lambda_diversity=2.0,
            lambda_template_diversity=2.0,
            lambda_template_sparsity=2.0,  # Reduced from 7.0
            lambda_spatial_diversity=1,
            lambda_compactness=2.0,  # Reduced from 15.0
            lambda_mass_conservation=100.0,  # CRITICAL: Increased from 5.0 to enforce mass preservation
            lambda_sparsity_match=10.0,
            lambda_tv=0.5,
            lambda_jacobian=75.0,
            lambda_coarse_smooth=0.03
        ).to(device)

        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print(f"  Using AMP")
        else:
            self.scaler = None

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

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                h_aligned, cluster_probs, phi_coarse, phi_fine, v_coarse, v_fine = self.model(
                    saliency_maps,
                    class_ids=labels,
                    update_templates=True,
                    original_images=rgb_images
                )

                batch_templates = self.model.get_batch_templates(labels)

                losses = self.criterion(saliency_maps, h_aligned, batch_templates,
                                      cluster_probs, v_coarse, v_fine)

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

            for key, value in losses.items():
                epoch_losses[key] += value.item()
            num_batches += 1

            postfix_dict = {
                'loss': losses['total'].item(),
                'align': losses['alignment'].item(),
            }
            if torch.cuda.is_available():
                gpu_mem_used = torch.cuda.memory_allocated() / (1024**3)
                postfix_dict['GPU_GB'] = f'{gpu_mem_used:.1f}'

            pbar.set_postfix(postfix_dict)

            if batch_idx % 50 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        avg_losses = {key: value / num_batches for key, value in epoch_losses.items()}

        for key, value in avg_losses.items():
            self.history[f'train_{key}'].append(value)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return avg_losses

    def train(self, num_epochs, save_frequency=5):
        """Train for multiple epochs."""
        print(f"\n{'='*80}")
        print(f"Starting Advanced LDDMM Training v3.1")
        print(f"{'='*80}")

        for epoch in range(1, num_epochs + 1):
            avg_losses = self.train_epoch(epoch)

            print(f"\nEpoch {epoch}/{num_epochs} Summary:")
            print(f"  Total Loss: {avg_losses['total']:.4f}")
            print(f"  Alignment: {avg_losses['alignment']:.4f}")
            print(f"  Mass Conservation: {avg_losses['mass_conservation']:.4f}")
            print(f"  Compactness: {avg_losses['compactness']:.4f}")

            self.scheduler.step(avg_losses['total'])

            if epoch % save_frequency == 0:
                self.save_checkpoint(epoch)

        self.save_checkpoint(num_epochs, final=True)

        print(f"\n✓ Training Complete!")

    def save_checkpoint(self, epoch, final=False):
        """Save model checkpoint."""
        if final:
            checkpoint_path = self.checkpoint_dir / "advanced_lddmm_model_final.pth"
        else:
            checkpoint_path = self.checkpoint_dir / f"advanced_lddmm_model_epoch_{epoch}.pth"

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': dict(self.history),
            'templates': self.model.templates,
            'template_counts': self.model.template_counts
        }, checkpoint_path)

        print(f"  Checkpoint saved: {checkpoint_path}")

    def plot_training_curves(self, save_path='training_curves_v31.png'):
        """Plot training curves."""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()

        metrics = [
            ('train_total', 'Total Loss'),
            ('train_alignment', 'Alignment Loss'),
            ('train_smoothness_coarse', 'Smoothness (Coarse)'),
            ('train_smoothness_fine', 'Smoothness (Fine)'),
            ('train_mass_conservation', 'Mass Conservation'),
            ('train_sparsity_match', 'Sparsity Match'),
            ('train_jacobian', 'Jacobian'),
            ('train_compactness', 'Compactness'),
            ('train_diversity', 'Diversity')
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
        description='Advanced Geodesic Pattern Learning v3.1 - Enhanced Signal Preservation',
        epilog="""
V3.1 IMPROVEMENTS:
  - Adaptive signal enhancement (log-power, percentile, local contrast)
  - Removed edge gating and top-K sharpening
  - Reduced compactness and sparsity penalties
  - Preserves thin structures (neck, legs) alongside mass body

USAGE:
  python geo_s3_v1.py --data_dir ./data/saliency_imagenet1k_resnet50_100 \\
                      --use_ot --epochs 50 --enhance_mode combined --alpha 0.5
        """
    )
    parser.add_argument('--data_dir', type=str,
                       default='./data/saliency_imagenet1k_resnet50_100')
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--k_subpatterns', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--use_ot', action='store_true')
    parser.add_argument('--enhance_mode', type=str, default='log_power',
                       choices=['log_power', 'percentile', 'local_contrast', 'combined', 'none'],
                       help='Signal enhancement mode (default: log_power, use none to disable)')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Power for log-power transform (default: 0.5)')
    parser.add_argument('--max_samples_per_class', type=int, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_s3_v1')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"Advanced Geodesic Pattern Learning v3.1 - Signal Enhancement")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"\nEnhancement: {args.enhance_mode} (alpha={args.alpha})")

    dataset = SaliencyMapDataset(
        data_dir=args.data_dir,
        max_samples_per_class=args.max_samples_per_class,
        load_images=False,  # No edge gating in v3.1
        max_samples=100000
    )

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        collate_fn=safe_collate_fn,
        persistent_workers=True
    )

    model = AdvancedLDDMM_Pipeline(
        num_classes=args.num_classes,
        k_subpatterns=args.k_subpatterns,
        img_res=(224, 224),
        device=device,
        temperature=0.1,
        use_ot=args.use_ot,
        enhance_mode=args.enhance_mode,
        alpha=args.alpha
    )
    model = torch.compile(model)

    trainer = AdvancedLDDMMTrainer(
        model=model,
        train_loader=train_loader,
        lr=args.lr,
        device=device,
        checkpoint_dir=args.checkpoint_dir
    )

    trainer.train(num_epochs=args.epochs, save_frequency=5)

    trainer.plot_training_curves(
        save_path=Path(args.checkpoint_dir) / 'training_curves_v31.png'
    )

    print(f"\n✓ All done!")


if __name__ == "__main__":
    main()
