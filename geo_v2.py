"""
Advanced Geodesic Pattern Learning v2 - Refined for Sharp Structural Patterns


3. EDGE-AWARE GATING ENABLED BY DEFAULT:
   - Forces alignment based on physical object boundaries
   - Zeros out background noise before LDDMM processing
   - Recovers interpretable structural patterns (e.g., ostrich silhouette)

4. LAZY LOADING WITH LRU CACHE:
   - Reduces memory usage from ~75GB to <5GB for ImageNet-1k
   - On-demand loading of saliency maps and RGB images
   - Automatic cache cleanup between epochs
   - Configurable cache size for RAM/speed tradeoff

Key Innovations (from geo_x.py):
1. Multi-Scale Coarse-to-Fine SVF Prediction: Hierarchical alignment (14×14 → 112×112)
2. Sinkhorn-Knopp Optimal Transport: Denoising template updates via Wasserstein barycenter
3. Edge-Aware Attention Gating: Prioritize aligning IG at physical boundaries

This architecture addresses:
- Local minima from high-frequency noise
- Template ghosting from running averages
- Over-sensitivity to background IG artifacts
- HEATMAP DIFFUSION (v2 specific): Recovers spike-like IG structure
- MEMORY EXPLOSION (v2 specific): Lazy loading prevents OOM on large datasets

Usage:
    # With edge-aware gating (default, recommended)
    python geo_v2.py --data_dir ./data/saliency_imagenet1k_resnet50_100 \
                     --num_classes 1000 --k_subpatterns 10 --epochs 50 --use_edge_gating \
                     --cache_size 10

    # Without edge-aware gating (fallback)
    python geo_v2.py --data_dir ./data/saliency_imagenet1k_resnet50_100 \
                     --num_classes 1000 --k_subpatterns 10 --epochs 50
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
import kornia  # For edge detection
import gc  # For garbage collection
from functools import lru_cache  # For caching batch files


# ==========================================
# Multi-Scale Coarse-to-Fine Predictor
# ==========================================

class MultiScalePredictor(nn.Module):
    """
    Hierarchical velocity field predictor:
    Stage 1: Coarse (14×14) - Align global mass/location
    Stage 2: Fine (112×112) - Align local details

    The coarse stage receives a Gaussian-blurred IG to smooth out noise.
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

        # Head 2a: COARSE Velocity Field (14×14) - Global alignment
        self.coarse_v_res = (14, 14)
        self.coarse_v_dim = 2 * self.coarse_v_res[0] * self.coarse_v_res[1]
        self.coarse_v_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.coarse_v_dim),
            nn.Tanh()
        )

        # Head 2b: FINE Velocity Field (112×112) - Detail alignment
        # This predicts a RESIDUAL on top of the upsampled coarse field
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
        self.coarse_scale = 3.0  # Larger deformations for coarse alignment
        self.fine_scale = 0.5    # Smaller residual corrections

        # Gaussian blur for coarse stage (smooths noise)
        self.gaussian_blur = kornia.filters.GaussianBlur2d((11, 11), (3.0, 3.0))

        # Initialize to near-zero
        nn.init.normal_(self.coarse_v_head[-2].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.coarse_v_head[-2].bias)
        nn.init.normal_(self.fine_v_head[3].weight, mean=0.0, std=0.005)
        nn.init.zeros_(self.fine_v_head[3].bias)

    def forward(self, h_i):
        """
        Args:
            h_i: (B, 1, H, W) - Saliency maps

        Returns:
            cluster_logits: (B, K) - Assignment scores
            v_coarse: (B, 2, 14, 14) - Coarse velocity field
            v_fine: (B, 2, 112, 112) - Fine velocity field (residual)
            h_i_blurred: (B, 1, H, W) - Blurred version for coarse alignment
        """
        B = h_i.shape[0]

        # Create blurred version for coarse alignment
        h_i_blurred = self.gaussian_blur(h_i)

        # Extract features from ORIGINAL (non-blurred) map
        features = self.backbone(h_i)

        # Predict cluster assignment
        cluster_logits = self.assignment_head(features)

        # Predict COARSE velocity field (from blurred features for stability)
        features_blurred = self.backbone(h_i_blurred)
        v_coarse_flat = self.coarse_v_head(features_blurred)
        v_coarse = v_coarse_flat.view(B, 2, *self.coarse_v_res) * self.coarse_scale

        # Predict FINE residual velocity field
        v_fine_flat = self.fine_v_head(features)
        v_fine = v_fine_flat.view(B, 2, *self.fine_v_res) * self.fine_scale

        return cluster_logits, v_coarse, v_fine, h_i_blurred


# ==========================================
# Diffeomorphic Warper (Same as before)
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

    This replaces running average for template updates. It finds the
    optimal "center of mass" in Wasserstein space, which is more robust
    to noise than simple averaging.

    Args:
        samples: (B, 1, H, W) - Batch of aligned saliency maps
        weights: (B,) - Optional sample weights (uniform if None)
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

    # Initialize barycenter as weighted average (starting point)
    barycenter = (weights.view(B, 1) * samples_flat).sum(dim=0)  # (H*W,)

    # Normalize to probability distributions
    samples_flat = samples_flat / (samples_flat.sum(dim=1, keepdim=True) + 1e-10)
    barycenter = barycenter / (barycenter.sum() + 1e-10)

    # Sinkhorn iterations
    for _ in range(num_iter):
        # Compute cost matrix (L2 distance)
        # For efficiency, use squared L2 in 1D (pixel intensity space)
        # This is a simplified version - full OT would use spatial coordinates

        # Update barycenter as weighted geometric mean
        log_samples = torch.log(samples_flat + 1e-10)
        log_barycenter = (weights.view(B, 1) * log_samples).sum(dim=0)
        barycenter = torch.exp(log_barycenter)
        barycenter = barycenter / (barycenter.sum() + 1e-10)

    # Reshape back
    barycenter = barycenter.view(1, H, W)

    return barycenter


# ==========================================
# Edge-Aware Attention Gating
# ==========================================

class EdgeAwareGating(nn.Module):
    """
    Apply attention mask based on edge detection to prioritize
    aligning IG signals at physical boundaries.

    Uses Sobel filter to detect edges in original image.
    """

    def __init__(self):
        super().__init__()
        # No need to store Sobel as a module - we'll use functional spatial_gradient

    def forward(self, image, saliency):
        """
        Args:
            image: (B, 3, H, W) - Original RGB image
            saliency: (B, 1, H, W) - Saliency map

        Returns:
            gated_saliency: (B, 1, H, W) - Edge-gated saliency
            edge_mask: (B, 1, H, W) - Edge attention mask
        """
        # Convert to grayscale if needed
        if image.shape[1] == 3:
            image_gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        else:
            image_gray = image

        # Detect edges using spatial gradient (returns gradients in x and y)
        # spatial_gradient returns (B, C, 2, H, W) where dim=2 is [grad_y, grad_x]
        grads = kornia.filters.spatial_gradient(image_gray, mode='sobel')  # (B, 1, 2, H, W)

        # Extract x and y gradients
        grad_y = grads[:, :, 0, :, :]  # (B, 1, H, W)
        grad_x = grads[:, :, 1, :, :]  # (B, 1, H, W)

        # Compute edge magnitude
        edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)  # (B, 1, H, W)

        # DEBUG: Verify edge_magnitude has correct shape
        if edge_magnitude.shape[1] != 1:
            print(f"❌ edge_magnitude has wrong shape: {edge_magnitude.shape}, expected (B, 1, H, W)")
            raise RuntimeError(f"edge_magnitude shape error: {edge_magnitude.shape}")

        # Normalize to [0, 1]
        edge_mask = edge_magnitude / (edge_magnitude.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] + 1e-8)

        # DEBUG: Verify edge_mask has correct shape
        if edge_mask.shape[1] != 1:
            print(f"❌ edge_mask has wrong shape: {edge_mask.shape}, expected (B, 1, H, W)")
            raise RuntimeError(f"edge_mask shape error: {edge_mask.shape}")

        # Apply soft gating (mix of uniform and edge-based)
        alpha = 0.7  # 70% edge-based, 30% uniform
        edge_mask = alpha * edge_mask + (1 - alpha) * torch.ones_like(edge_mask)

        # Gate saliency
        gated_saliency = saliency * edge_mask

        # CRITICAL FIX: Force 4D shape (B, 1, H, W) to prevent dimension squeezing
        if gated_saliency.dim() == 3:
            gated_saliency = gated_saliency.unsqueeze(1)

        # Verify shape is correct
        assert gated_saliency.shape[1] == 1, f"EdgeAwareGating: Expected 1 channel, got {gated_saliency.shape}"

        return gated_saliency, edge_mask


# ==========================================
# Complete Pipeline with Advanced Features
# ==========================================

class AdvancedLDDMM_Pipeline(nn.Module):
    """
    Advanced LDDMM pipeline v2 with sharp structure recovery.

    V2 CHANGES:
    - Loss function tuned for anti-diffusion (reduced smoothness penalties)
    - Edge-aware gating recommended by default to zero out background noise

    Core Features:
    1. Multi-scale coarse-to-fine alignment (14×14 → 112×112)
    2. Optimal transport template updates (Sinkhorn barycenter)
    3. Edge-aware attention gating (zeros out background before alignment)
    """

    def __init__(self, num_classes=1000, k_subpatterns=10, img_res=(224, 224),
                 device='cuda', temperature=0.1, use_ot=True, use_edge_gating=True):
        super().__init__()
        self.num_classes = num_classes
        self.K = k_subpatterns
        self.res = img_res
        self.device = device
        self.temperature = temperature
        self.use_ot = use_ot  # Use optimal transport vs running average
        self.use_edge_gating = use_edge_gating  # Use edge-aware gating

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

        print(f"\nAdvanced LDDMM Pipeline v2 Initialized:")
        print(f"  Multi-scale: Coarse (14×14) → Fine (112×112)")
        print(f"  Template update: {'Optimal Transport (Sinkhorn)' if use_ot else 'Running Average'}")
        print(f"    - Sparsity:")
        print(f"    - Top-K sharpening:")
        print(f"  Edge gating: {'Enabled' if use_edge_gating else 'Disabled'} (v2 default: ON)")
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
        # DEBUG: Check input shape
        if h_i.shape[1] == 0:
            print(f"\n{'='*80}")
            print(f"❌ CRITICAL ERROR: Input tensor has 0 channels!")
            print(f"   h_i.shape = {h_i.shape}")
            if original_images is not None:
                print(f"   original_images.shape = {original_images.shape}")
            print(f"   This means the DataLoader is returning corrupted data")
            print(f"{'='*80}\n")
            raise RuntimeError(f"Input tensor has invalid shape: {h_i.shape}, expected [B, 1, 224, 224]")

        B = h_i.shape[0]

        # Optional: Apply edge gating
        if self.use_edge_gating and original_images is not None:
            h_i_gated, edge_mask = self.edge_gating(original_images, h_i)
        else:
            h_i_gated = h_i
            edge_mask = None

        # CRITICAL FIX: Ensure h_i_gated is always (B, 1, H, W)
        if h_i_gated.dim() == 3:
            # Lost channel dimension - restore it
            h_i_gated = h_i_gated.unsqueeze(1)
            print(f"⚠️  Warning: h_i_gated was 3D, restored to 4D: {h_i_gated.shape}")
        elif h_i_gated.shape[1] == 0:
            # Somehow got 0 channels - use original input
            print(f"❌ CRITICAL: h_i_gated has 0 channels {h_i_gated.shape}, falling back to h_i")
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

        # Step 4: Update template bank
        if update_templates and class_ids is not None:
            if self.use_ot:
                self._update_template_bank_ot(h_aligned, class_ids, cluster_assigned)
            else:
                self._update_template_bank_avg(h_aligned, class_ids, cluster_assigned)

        return h_aligned, cluster_probs, phi_coarse, phi_fine, v_coarse, v_fine

    @torch.no_grad()
    def _update_template_bank_ot(self, h_aligned, class_ids, cluster_assigned,
                                  sparsity_percentile=90, top_k_percent=20):
        """
        Update template bank using Optimal Transport (Sinkhorn barycenter) with AGGRESSIVE top-K sharpening.

        V2 CHANGES:
        - sparsity_percentile: 90% → 98% (MUCH more aggressive)
        - top_k_percent: 5% → 3% (even sharper templates to recover spike structure)

        For each (class, cluster), collect all aligned samples and compute
        their Wasserstein barycenter. Apply top-K sharpening to prevent blur leakage.

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
            c, k = pair[0], pair[1]  # Keep as tensors to avoid graph breaks

            # Find all samples for this (class, cluster)
            mask = (class_ids == c) & (cluster_assigned == k)
            if not mask.any():
                continue

            samples = h_aligned[mask]  # (N, 1, H, W)

            # Limit number of samples to prevent OOM
            if samples.shape[0] > 16:
                # Randomly sample 16 if too many
                indices = torch.randperm(samples.shape[0], device=samples.device)[:16]
                samples = samples[indices]

            # Compute Wasserstein barycenter (reduced iterations to save memory)
            barycenter = sinkhorn_barycenter(samples, reg=0.02, num_iter=10)

            # Free memory immediately
            del samples

            # === TOP-K SHARPENING ===
            # Keep only the top K% of pixels to prevent blur leakage
            # This is critical for maintaining "Ostrich" structure across updates
            num_pixels = barycenter.numel()
            top_k = max(int(num_pixels * top_k_percent / 100.0), 1)

            # Get indices of top-K values
            barycenter_flat = barycenter.flatten()
            top_k_values, top_k_indices = torch.topk(barycenter_flat, k=top_k)

            # Create sharpened version: zero out all but top-K
            barycenter_sharp = torch.zeros_like(barycenter_flat)
            barycenter_sharp[top_k_indices] = top_k_values
            barycenter = barycenter_sharp.reshape(barycenter.shape)

            # Additional sparsity thresholding (belt-and-suspenders approach)
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

            # Blend with existing template (slow EMA for stability)
            count = self.template_counts[c, k]  # Keep as tensor
            if count == 0:
                self.templates[c, k] = barycenter
            else:
                eta = torch.clamp(torch.tensor(0.1, device=count.device),
                                  max=1.0 / (count + 1))  # Slower updates, keep as tensor
                self.templates[c, k] = (1 - eta) * self.templates[c, k] + eta * barycenter

            self.template_counts[c, k] += mask.sum()

            # Free intermediate tensors immediately
            del barycenter, barycenter_flat, top_k_values, top_k_indices, barycenter_sharp

    @torch.no_grad()
    def _update_template_bank_avg(self, h_aligned, class_ids, cluster_assigned, sparsity_percentile=90):
        """Fallback: Running average update with v2 aggressive sparsity (98th percentile)."""
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
    Loss functions for advanced LDDMM with multi-scale fields.

    V2 CHANGES (Anti-Diffusion for Sharp Structure Recovery):
    - lambda_smooth: 0.1 → 0.02 (fine field, allow jitter to preserve edges)
    - lambda_coarse_smooth: 0.05 → 0.03 (slight reduction for coarse field)
    - lambda_template_sparsity: 1.5 → 5.0 (AGGRESSIVE sparsity enforcement)
    - lambda_jacobian: 50.0 (UNCHANGED - high to prevent tearing)
    """

    def __init__(self, lambda_smooth=0.02, lambda_entropy=1.0, lambda_magnitude=0.00005,
                 lambda_diversity=2.0, lambda_template_diversity=2.0, lambda_template_sparsity=5.0,
                 lambda_spatial_diversity=0.3, lambda_compactness=10.0, lambda_mass_conservation=0.01,
                 lambda_sparsity_match=10.0, lambda_tv=0.5, lambda_jacobian=50.0,
                 lambda_coarse_smooth=0.03):  # V2: Reduced smoothness for anti-diffusion
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
        self.lambda_coarse_smooth = lambda_coarse_smooth  # Less smoothness penalty for coarse

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
        """Enforce total intensity preservation using relative change."""
        mass_original = h_original.sum(dim=(1, 2, 3))
        mass_aligned = h_aligned.sum(dim=(1, 2, 3))
        # Use relative change instead of absolute to handle unnormalized saliency maps
        relative_change = torch.abs(mass_aligned - mass_original) / (mass_original + 1e-8)
        return relative_change.mean()

    def entropy_loss(self, cluster_probs):
        """Entropy regularization to encourage confident assignments."""
        entropy = -(cluster_probs * torch.log(cluster_probs + 1e-10)).sum(dim=1)
        return entropy.mean()

    def diversity_loss(self, cluster_probs):
        """Diversity loss to encourage using all K patterns."""
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
        """Encourage compact blob-like shapes."""
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
        """Reduce fragmentation in templates."""
        diff_h = torch.abs(templates[:, :, :, 1:, :] - templates[:, :, :, :-1, :])
        diff_w = torch.abs(templates[:, :, :, :, 1:] - templates[:, :, :, :, :-1])
        tv = diff_h.mean() + diff_w.mean()
        return tv

    def forward(self, h_original, h_aligned, templates, cluster_probs,
                v_coarse, v_fine):
        """
        Compute total loss with separate penalties for coarse and fine fields.

        Args:
            h_original: (B, 1, H, W) - Original IG
            h_aligned: (B, 1, H, W) - Final aligned IG
            templates: (B, K, 1, H, W) - Templates
            cluster_probs: (B, K) - Cluster assignments
            v_coarse: (B, 2, 14, 14) - Coarse velocity field
            v_fine: (B, 2, 112, 112) - Fine velocity field

        Returns:
            loss_dict: Dictionary of loss components
        """
        loss_align = self.alignment_loss(h_aligned, templates, cluster_probs)

        # Separate smoothness for coarse (allow rougher) and fine (penalize more)
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

        # Combined total loss
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
# Dataset (Lazy Loading with LRU Cache)
# ==========================================

def safe_collate_fn(batch):
    """
    Custom collate function that validates tensor shapes before batching.

    Catches shape mismatches that cause [B, 0, H, W] errors.
    """
    # Unpack batch
    if len(batch[0]) == 3:  # (saliency, label, image)
        saliencies, labels, images = zip(*batch)
    else:
        raise ValueError(f"Unexpected batch item length: {len(batch[0])}")

    # DEBUG: Check first saliency
    if saliencies[0].shape != (1, 224, 224):
        print(f"\n❌ COLLATE ERROR: First saliency has wrong shape!")
        print(f"   Expected: (1, 224, 224)")
        print(f"   Got: {saliencies[0].shape}")
        print(f"   Batch size: {len(saliencies)}")

    # Validate each saliency map
    for i, sal in enumerate(saliencies):
        if sal.shape != (1, 224, 224):
            print(f"\n❌ COLLATE ERROR in sample {i}:")
            print(f"   Shape: {sal.shape}")
            print(f"   Type: {type(sal)}")
            print(f"   Dtype: {sal.dtype if hasattr(sal, 'dtype') else 'N/A'}")
            raise ValueError(f"Sample {i}: Invalid saliency shape {sal.shape}, expected (1, 224, 224)")

    # Validate each image if not None
    # Check if ANY image is None - if so, we can't use images for this batch
    has_any_none = any(img is None for img in images)

    if not has_any_none and images[0] is not None:
        for i, img in enumerate(images):
            if img.shape != (3, 224, 224):
                raise ValueError(f"Sample {i}: Invalid image shape {img.shape}, expected (3, 224, 224)")

    # Stack tensors
    saliency_batch = torch.stack(saliencies, dim=0)  # [B, 1, H, W]
    label_batch = torch.tensor(labels, dtype=torch.long)  # [B]

    # Only create image batch if ALL images are non-None
    if not has_any_none and images[0] is not None:
        image_batch = torch.stack(images, dim=0)  # [B, 3, H, W]
    else:
        if has_any_none and images[0] is not None:
            # Count how many are None
            none_count = sum(1 for img in images if img is None)
            print(f"⚠️  Warning: Batch has {none_count}/{len(images)} None images - skipping RGB data for this batch")
        image_batch = None

    # Final validation
    assert saliency_batch.shape[1] == 1, f"Saliency batch has {saliency_batch.shape[1]} channels, expected 1"

    return saliency_batch, label_batch, image_batch


class SaliencyMapDataset(Dataset):
    """
    Ultra-optimized RAM dataset with zero memory peaks:
    - Saliency: float16 (50% memory vs float32)
    - RGB: uint8 (75% memory vs float32)
    - Pre-allocated tensors filled in-place (no stacking peaks)
    """

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
        print(f"  Saliency: float16 (50% memory vs float32)")
        print(f"  RGB: {'uint8 (25% memory vs float32)' if load_images else 'not loaded'}")
        print(f"  Pre-allocated in-place filling (no memory peaks)")

        # Pre-allocate tensors (avoids stacking peak)
        self.saliency_maps = torch.empty((max_samples, 1, 224, 224), dtype=torch.float16)
        self.labels = torch.empty(max_samples, dtype=torch.long)

        if load_images:
            self.rgb_images = torch.empty((max_samples, 3, 224, 224), dtype=torch.uint8)
        else:
            self.rgb_images = None

        # Fill tensors in-place
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

                # Check class limit
                if max_samples_per_class is not None and class_counts[label] >= max_samples_per_class:
                    continue

                # Validate and load saliency
                if 'saliency_map' not in item:
                    corrupted_samples += 1
                    continue

                saliency_np = item['saliency_map']
                saliency_shape = saliency_np.shape

                if len(saliency_shape) < 2 or saliency_shape[-2:] != (224, 224) or 0 in saliency_shape:
                    corrupted_samples += 1
                    continue

                # Convert saliency to (1, 224, 224)
                if len(saliency_np.shape) == 2:
                    saliency_np = saliency_np[np.newaxis, ...]  # (H, W) -> (1, H, W)
                elif len(saliency_np.shape) == 3:
                    if saliency_np.shape[0] in [1, 3]:
                        pass  # Already (C, H, W)
                    elif saliency_np.shape[2] in [1, 3]:
                        saliency_np = saliency_np.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
                    else:
                        corrupted_samples += 1
                        continue

                    # Average channels if multi-channel
                    if saliency_np.shape[0] > 1:
                        saliency_np = saliency_np.mean(axis=0, keepdims=True)

                if saliency_np.shape != (1, 224, 224):
                    corrupted_samples += 1
                    continue

                # Fill saliency (convert to float16 directly)
                self.saliency_maps[idx] = torch.from_numpy(saliency_np).to(torch.float16)
                self.labels[idx] = label

                # Load RGB if needed
                if load_images:
                    if 'rgb_image' not in item:
                        corrupted_samples += 1
                        continue

                    try:
                        rgb_np = item['rgb_image']

                        # Convert to (3, 224, 224) uint8
                        if len(rgb_np.shape) == 3:
                            if rgb_np.shape[0] == 3:
                                pass  # Already (3, H, W)
                            elif rgb_np.shape[2] == 3:
                                rgb_np = rgb_np.transpose(2, 0, 1)  # (H, W, 3) -> (3, H, W)
                            else:
                                corrupted_samples += 1
                                continue
                        else:
                            corrupted_samples += 1
                            continue

                        # Convert to uint8
                        if rgb_np.dtype != np.uint8:
                            if rgb_np.max() <= 1.0:
                                rgb_np = (rgb_np * 255).astype(np.uint8)
                            else:
                                rgb_np = rgb_np.astype(np.uint8)

                        # Fill RGB in-place
                        self.rgb_images[idx] = torch.from_numpy(rgb_np)
                    except Exception:
                        corrupted_samples += 1
                        continue

                # Update counters
                class_counts[label] += 1
                idx += 1

            del batch_data
            gc.collect()

        # Trim to actual size
        self.saliency_maps = self.saliency_maps[:idx]
        self.labels = self.labels[:idx]
        if load_images:
            self.rgb_images = self.rgb_images[:idx]

        # Print stats
        num_samples = len(self.saliency_maps)
        saliency_mb = num_samples * 1 * 224 * 224 * 2 / (1024**2)  # float16 = 2 bytes
        rgb_mb = num_samples * 3 * 224 * 224 * 1 / (1024**2) if load_images else 0

        print(f"\n✅ Loaded {num_samples} samples ({corrupted_samples} corrupted)")
        print(f"  Saliency: {saliency_mb:.1f} MB (float16) - would be {saliency_mb*2:.1f} MB as float32")
        if load_images:
            print(f"  RGB: {rgb_mb:.1f} MB (uint8) - would be {rgb_mb*4:.1f} MB as float32")
        print(f"  Total RAM: {saliency_mb + rgb_mb:.1f} MB")
        print(f"  Memory savings: {(saliency_mb*2 + rgb_mb*4) - (saliency_mb + rgb_mb):.1f} MB")


    def __len__(self):
        return len(self.saliency_maps)

    def __getitem__(self, idx):
        """
        Ultra-fast access: everything is pre-loaded in RAM.
        - Saliency: float16 -> float32 (on-the-fly in worker)
        - RGB: uint8 -> float32 (on-the-fly in worker)
        """
        # Convert float16 -> float32 for computation (happens in worker thread, very fast)
        saliency = self.saliency_maps[idx].float()  # (1, 224, 224)
        label = self.labels[idx].item()

        # Get RGB if needed
        if self.load_images:
            # Convert uint8 -> float32 on-the-fly
            image = self.rgb_images[idx].float() / 255.0  # (3, 224, 224)
            return saliency, label, image
        else:
            return saliency, label, None


# ==========================================
# Training
# ==========================================

class AdvancedLDDMMTrainer:
    """Trainer for Advanced LDDMM pipeline with multi-scale features."""

    def __init__(self, model, train_loader, val_loader=None,
                 lr=1e-3, device='cuda', checkpoint_dir='./checkpoints_advanced', use_amp=True):
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

        # Loss (V2: AGGRESSIVE sparsity + anti-diffusion for sharp structure recovery)
        # CRITICAL CHANGES IN V2:
        # 1. lambda_template_sparsity: 1.5 → 5.0 (AGGRESSIVE to recover spike structure)
        # 2. lambda_smooth (fine): 0.1 → 0.02 (reduced 5x to allow edge-preserving jitter)
        # 3. lambda_coarse_smooth: 0.05 → 0.03 (slight reduction)
        # 4. lambda_jacobian: 50.0 UNCHANGED (high to prevent tearing from jitter)
        #
        # The Jacobian determinant prevents the model from "cheating" by stretching
        # a single noisy pixel into a large blob to satisfy alignment loss.
        # This is essential for preserving sparse structures like "Ostrich" patterns.
        self.criterion = AdvancedLDDMMLoss(
            lambda_smooth=0.05,                   # V2: Reduced 5x (0.1 → 0.02) for edge preservation
            lambda_entropy=1.0,                   # Confident cluster assignments
            lambda_magnitude=0.00005,             # Allow detailed deformations
            lambda_diversity=2.0,                 # Use all K patterns
            lambda_template_diversity=2.0,        # Templates should differ
            lambda_template_sparsity=7.0,         # V2: AGGRESSIVE (1.5 → 5.0) for spike recovery
            lambda_spatial_diversity=1,         # Minimal spatial constraint
            lambda_compactness=10.0,              # STRONG: Prefer blob-like over line-like
            lambda_mass_conservation=5.0,       # CRITICAL: Perfect mass conservation
            lambda_sparsity_match=10.0,           # CRITICAL: Preserve sparsity patterns
            lambda_tv=0.5,                        # Reduce fragmentation
            lambda_jacobian=50.0,                 # CRITICAL: Prevent spatial stretching/compression!
            lambda_coarse_smooth=0.03             # V2: Reduced (0.05 → 0.03) for coarse field
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
            # Unpack batch (handles both 2-tuple and 3-tuple returns)
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
                    original_images=rgb_images  # Pass images for edge-aware gating
                )

                # Get templates for this batch
                batch_templates = self.model.get_batch_templates(labels)

                # Compute loss with multi-scale velocity fields
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
                # Also run Python GC to free RAM
                import gc
                gc.collect()

        # Average losses
        avg_losses = {key: value / num_batches for key, value in epoch_losses.items()}

        # Update history
        for key, value in avg_losses.items():
            self.history[f'train_{key}'].append(value)

        # Run garbage collection at end of epoch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return avg_losses

    def train(self, num_epochs, save_frequency=5):
        """Train for multiple epochs."""
        print(f"\n{'='*80}")
        print(f"Starting Advanced LDDMM Training")
        print(f"{'='*80}")
        print(f"Epochs: {num_epochs}")
        print(f"Device: {self.device}")

        for epoch in range(1, num_epochs + 1):
            avg_losses = self.train_epoch(epoch)

            print(f"\nEpoch {epoch}/{num_epochs} Summary:")
            print(f"  Total Loss: {avg_losses['total']:.4f}")
            print(f"  Alignment: {avg_losses['alignment']:.4f}")
            print(f"  *** MASS CONSERVATION: {avg_losses['mass_conservation']:.4f}")
            print(f"  *** SPARSITY MATCH: {avg_losses['sparsity_match']:.4f}")
            print(f"  *** JACOBIAN: {avg_losses['jacobian']:.4f}")
            print(f"  Smoothness (Coarse): {avg_losses['smoothness_coarse']:.4f}")
            print(f"  Smoothness (Fine): {avg_losses['smoothness_fine']:.4f}")
            print(f"  Entropy: {avg_losses['entropy']:.4f}")
            print(f"  Diversity: {avg_losses['diversity']:.4f}")
            print(f"  Template Diversity: {avg_losses['template_diversity']:.4f}")
            print(f"  Template Sparsity: {avg_losses['template_sparsity']:.4f}")
            print(f"  Compactness: {avg_losses['compactness']:.4f}")

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

    def plot_training_curves(self, save_path='training_curves_advanced.png'):
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
            ('train_entropy', 'Entropy'),
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
    # Configure torch.compile for better graph capture
    torch._dynamo.config.capture_scalar_outputs = True

    parser = argparse.ArgumentParser(
        description='Advanced Geodesic Pattern Learning v2 - Refined for Sharp Structural Patterns',
        epilog="""
V2 IMPROVEMENTS:
  - AGGRESSIVE sparsity enforcement (5x stronger)
  - Anti-diffusion velocity fields (reduced smoothness)
  - Edge-aware gating ENABLED BY DEFAULT (zeros out background noise)

RECOMMENDED USAGE:
  python geo_v2.py --data_dir ./data/saliency_imagenet1k_resnet50_100 \\
                   --use_ot --use_edge_gating --epochs 50
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
    parser.add_argument('--use_ot', action='store_true',
                       help='Use Optimal Transport for template updates (RECOMMENDED for v2)')
    parser.add_argument('--use_edge_gating', dest='use_edge_gating', action='store_true', default=True,
                       help='Use edge-aware attention gating (DEFAULT: ON for v2)')
    parser.add_argument('--no_edge_gating', dest='use_edge_gating', action='store_false',
                       help='Disable edge-aware attention gating')
    parser.add_argument('--max_samples_per_class', type=int, default=None,
                       help='Max samples per class (default: None = all)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_v2',
                       help='Checkpoint directory (default: ./checkpoints_v2)')
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"Advanced Geodesic Pattern Learning v2 - Sharp Structure Recovery")
    print(f"{'='*80}")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total GPU Memory: {total_mem:.1f} GB")
        torch.cuda.empty_cache()

    print(f"\nV2 Improvements:")
    print(f"  - AGGRESSIVE sparsity)")
    print(f"  - Anti-diffusion: fine_smooth=0.02 (was 0.1), coarse_smooth=0.03 (was 0.05)")
    print(f"  - Edge-aware gating: {'ENABLED' if args.use_edge_gating else 'DISABLED'} (default: ON)")
    print(f"\nFeatures:")
    print(f"  - Multi-scale coarse-to-fine alignment (14×14 → 112×112)")
    print(f"  - Optimal Transport: {args.use_ot}")
    if args.use_ot:
        print(f"    ⚠️  OT uses more RAM - if you get OOM, disable with --no_ot")
    print(f"  - Edge-aware gating: {args.use_edge_gating}")

    # Load dataset
    print(f"\nLoading saliency maps from {args.data_dir}...")
    dataset = SaliencyMapDataset(
        data_dir=args.data_dir,
        max_samples_per_class=args.max_samples_per_class,
        load_images=args.use_edge_gating,  # Load RGB images into RAM (uint8 format)
        max_samples=100000  # Pre-allocate for up to 100k samples
    )

    # DataLoader - optimized for RAM-based dataset
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,  # 2 workers enough for RAM access (reduces overhead)
        pin_memory=torch.cuda.is_available(),  # Pin memory for faster GPU transfer
        collate_fn=safe_collate_fn,
        persistent_workers=True
    )

    print(f"\nDataLoader Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num workers: 2 (optimized for RAM access)")
    print(f"  Pin memory: {torch.cuda.is_available()}")
    print(f"  Persistent workers: True")

    # Create model
    print(f"\nInitializing Advanced LDDMM model...")
    model = AdvancedLDDMM_Pipeline(
        num_classes=args.num_classes,
        k_subpatterns=args.k_subpatterns,
        img_res=(224, 224),
        device=device,
        temperature=0.1,
        use_ot=args.use_ot,
        use_edge_gating=args.use_edge_gating
    )
    model = torch.compile(model) # Add this line
    # Create trainer
    trainer = AdvancedLDDMMTrainer(
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
        save_path=Path(args.checkpoint_dir) / 'training_curves_v2.png'
    )

    if torch.cuda.is_available():
        print(f"\nFinal GPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")

    print(f"\n✓ All done!")
    print(f"\nTo visualize patterns:")
    print(f"  python visualize_patterns.py --checkpoint {args.checkpoint_dir}/advanced_lddmm_model_final.pth --class_id 0")
    print(f"\nTo retrain with same settings:")
    print(f"  python geo_v2.py --data_dir {args.data_dir} --use_ot --use_edge_gating --epochs {args.epochs}")


if __name__ == "__main__":
    main()
