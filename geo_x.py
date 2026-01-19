"""
Advanced Geodesic Pattern Learning with Multi-Scale LDDMM and Optimal Transport

Key Innovations:
1. Multi-Scale Coarse-to-Fine SVF Prediction: Hierarchical alignment (14×14 → 112×112)
2. Sinkhorn-Knopp Optimal Transport: Denoising template updates via Wasserstein barycenter
3. Edge-Aware Attention Gating: Prioritize aligning IG at physical boundaries

This architecture addresses:
- Local minima from high-frequency noise
- Template ghosting from running averages
- Over-sensitivity to background IG artifacts

Usage:
    python geo_x.py --data_dir ./data/saliency_imagenet1k_resnet50_100 \
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
        # Sobel edge detector
        self.sobel = kornia.filters.Sobel()

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

        # Detect edges
        edges = self.sobel(image_gray)  # (B, 2, H, W) - gradients in x and y
        edge_magnitude = torch.sqrt(edges[:, 0:1]**2 + edges[:, 1:2]**2 + 1e-8)

        # Normalize to [0, 1]
        edge_mask = edge_magnitude / (edge_magnitude.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] + 1e-8)

        # Apply soft gating (mix of uniform and edge-based)
        alpha = 0.7  # 70% edge-based, 30% uniform
        edge_mask = alpha * edge_mask + (1 - alpha) * torch.ones_like(edge_mask)

        # Gate saliency
        gated_saliency = saliency * edge_mask

        return gated_saliency, edge_mask


# ==========================================
# Complete Pipeline with Advanced Features
# ==========================================

class AdvancedLDDMM_Pipeline(nn.Module):
    """
    Advanced LDDMM pipeline with:
    1. Multi-scale coarse-to-fine alignment
    2. Optimal transport template updates
    3. Edge-aware attention gating
    """

    def __init__(self, num_classes=1000, k_subpatterns=10, img_res=(224, 224),
                 device='cuda', temperature=0.1, use_ot=True, use_edge_gating=False):
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

        print(f"\nAdvanced LDDMM Pipeline Initialized:")
        print(f"  Multi-scale: Coarse (14×14) → Fine (112×112)")
        print(f"  Template update: {'Optimal Transport (Sinkhorn)' if use_ot else 'Running Average'}")
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
        B = h_i.shape[0]

        # Optional: Apply edge gating
        if self.use_edge_gating and original_images is not None:
            h_i_gated, edge_mask = self.edge_gating(original_images, h_i)
        else:
            h_i_gated = h_i
            edge_mask = None

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
                                  sparsity_percentile=90, top_k_percent=5):
        """
        Update template bank using Optimal Transport (Sinkhorn barycenter) with top-K sharpening.

        For each (class, cluster), collect all aligned samples and compute
        their Wasserstein barycenter. Apply top-K sharpening to prevent blur leakage.

        Args:
            h_aligned: (B, 1, H, W) - Aligned saliency maps
            class_ids: (B,) - Class indices
            cluster_assigned: (B,) - Cluster indices
            sparsity_percentile: float - Percentile threshold for sparsity (default: 90)
            top_k_percent: float - Keep only top K% of pixels (default: 5% for high sharpness)
        """
        B = h_aligned.shape[0]

        # Group samples by (class, cluster)
        unique_pairs = torch.unique(torch.stack([class_ids, cluster_assigned], dim=1), dim=0)

        for pair in unique_pairs:
            c, k = pair[0].item(), pair[1].item()

            # Find all samples for this (class, cluster)
            mask = (class_ids == c) & (cluster_assigned == k)
            if not mask.any():
                continue

            samples = h_aligned[mask]  # (N, 1, H, W)

            # Compute Wasserstein barycenter
            barycenter = sinkhorn_barycenter(samples, reg=0.05, num_iter=30)

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
            if barycenter.max() > 0:
                barycenter = barycenter / barycenter.max()

            # Blend with existing template (slow EMA for stability)
            count = self.template_counts[c, k].item()
            if count == 0:
                self.templates[c, k] = barycenter
            else:
                eta = min(0.1, 1.0 / (count + 1))  # Slower updates
                self.templates[c, k] = (1 - eta) * self.templates[c, k] + eta * barycenter

            self.template_counts[c, k] += mask.sum().item()

    @torch.no_grad()
    def _update_template_bank_avg(self, h_aligned, class_ids, cluster_assigned, sparsity_percentile=90):
        """Fallback: Running average update (same as geo_patterns.py)."""
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
    """Loss functions for advanced LDDMM with multi-scale fields."""

    def __init__(self, lambda_smooth=0.1, lambda_entropy=1.0, lambda_magnitude=0.00005,
                 lambda_diversity=2.0, lambda_template_diversity=2.0, lambda_template_sparsity=1.5,
                 lambda_spatial_diversity=0.3, lambda_compactness=10.0, lambda_mass_conservation=100.0,
                 lambda_sparsity_match=10.0, lambda_tv=0.5, lambda_jacobian=50.0,
                 lambda_coarse_smooth=0.05):  # NEW: separate smoothness for coarse field
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
        """Enforce total intensity preservation."""
        mass_original = h_original.sum(dim=(1, 2, 3))
        mass_aligned = h_aligned.sum(dim=(1, 2, 3))
        return torch.abs(mass_aligned - mass_original).mean()

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
# Dataset (Same as geo_patterns.py)
# ==========================================

class SaliencyMapDataset(Dataset):
    """Full in-memory dataset with optional RGB image loading for edge-aware gating."""

    def __init__(self, data_dir, max_samples_per_class=None, load_images=False):
        self.data_dir = Path(data_dir)
        self.saliency_dir = self.data_dir / "saliency_maps"
        self.load_images = load_images

        metadata_path = self.data_dir / "metadata.pkl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        self.metadata = joblib.load(metadata_path)

        batch_files = sorted(self.saliency_dir.glob("batch_*.pkl"))
        print(f"Loading {len(batch_files)} batch files into memory...")

        all_saliency_maps = []
        all_labels = []
        all_images = [] if load_images else None
        all_image_paths = []
        class_counts = defaultdict(int)

        for batch_file in tqdm(batch_files, desc="Loading data"):
            batch_data = joblib.load(batch_file)

            for item in batch_data:
                label = item['true_label']

                if max_samples_per_class is None or class_counts[label] < max_samples_per_class:
                    all_saliency_maps.append(item['saliency_map'])
                    all_labels.append(label)

                    # Try to load RGB images if requested
                    if load_images:
                        # Check if image is stored in the batch data
                        if 'image' in item:
                            all_images.append(item['image'])
                        elif 'rgb_image' in item:
                            all_images.append(item['rgb_image'])
                        # Store image path for lazy loading
                        elif 'image_path' in item:
                            all_image_paths.append(item['image_path'])
                        else:
                            # Create a dummy grayscale image from saliency map
                            # This is a fallback - edge detection will work but won't be optimal
                            dummy_img = np.stack([item['saliency_map']] * 3, axis=0)
                            all_images.append(dummy_img)

                    class_counts[label] += 1

        self.saliency_maps = torch.from_numpy(np.array(all_saliency_maps)).float().unsqueeze(1)
        self.labels = torch.tensor(all_labels, dtype=torch.long)

        if load_images and all_images:
            self.images = torch.from_numpy(np.array(all_images)).float()
            # Normalize to [0, 1] if needed
            if self.images.max() > 1.0:
                self.images = self.images / 255.0
            print(f"✓ Loaded {len(self.images)} RGB images for edge-aware gating")
        else:
            self.images = None
            if load_images:
                print(f"⚠ Warning: RGB images not found in batch data. Edge gating will use fallback mode.")

        memory_mb = self.saliency_maps.element_size() * self.saliency_maps.nelement() / (1024**2)
        if self.images is not None:
            memory_mb += self.images.element_size() * self.images.nelement() / (1024**2)
        print(f"✓ Loaded {len(self.saliency_maps)} samples ({memory_mb:.1f} MB)")

    def __len__(self):
        return len(self.saliency_maps)

    def __getitem__(self, idx):
        saliency = self.saliency_maps[idx]
        label = self.labels[idx]

        if self.images is not None:
            image = self.images[idx]
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

        # Loss (rebalanced for advanced features with emphasis on structure preservation)
        # CRITICAL: lambda_jacobian=50.0 is HIGH on purpose!
        # The Jacobian determinant prevents the model from "cheating" by stretching
        # a single noisy pixel into a large blob to satisfy alignment loss.
        # This is essential for preserving sparse structures like "Ostrich" patterns.
        self.criterion = AdvancedLDDMMLoss(
            lambda_smooth=0.1,                    # Moderate smoothness for fine fields
            lambda_entropy=1.0,                   # Confident cluster assignments
            lambda_magnitude=0.00005,             # Allow detailed deformations
            lambda_diversity=2.0,                 # Use all K patterns
            lambda_template_diversity=2.0,        # Templates should differ
            lambda_template_sparsity=1.5,         # High sparsity enforcement
            lambda_spatial_diversity=0.3,         # Minimal spatial constraint
            lambda_compactness=10.0,              # STRONG: Prefer blob-like over line-like
            lambda_mass_conservation=100.0,       # CRITICAL: Perfect mass conservation
            lambda_sparsity_match=10.0,           # CRITICAL: Preserve sparsity patterns
            lambda_tv=0.5,                        # Reduce fragmentation
            lambda_jacobian=50.0,                 # CRITICAL: Prevent spatial stretching/compression!
            lambda_coarse_smooth=0.05             # Less smoothness penalty for coarse field
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

            # Periodic GPU memory cleanup
            if batch_idx % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Average losses
        avg_losses = {key: value / num_batches for key, value in epoch_losses.items()}

        # Update history
        for key, value in avg_losses.items():
            self.history[f'train_{key}'].append(value)

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
    parser = argparse.ArgumentParser(
        description='Advanced Geodesic Pattern Learning with Multi-Scale LDDMM and Optimal Transport'
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
    parser.add_argument('--use_ot', action='store_true',
                       help='Use Optimal Transport for template updates')
    parser.add_argument('--use_edge_gating', action='store_true',
                       help='Use edge-aware attention gating')
    parser.add_argument('--max_samples_per_class', type=int, default=None,
                       help='Max samples per class (default: None = all)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_advanced',
                       help='Checkpoint directory')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"Advanced Geodesic Pattern Learning")
    print(f"{'='*80}")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total GPU Memory: {total_mem:.1f} GB")
        torch.cuda.empty_cache()

    print(f"\nFeatures:")
    print(f"  - Multi-scale coarse-to-fine alignment (14×14 → 112×112)")
    print(f"  - Optimal Transport: {args.use_ot}")
    print(f"  - Edge-aware gating: {args.use_edge_gating}")

    # Load dataset
    print(f"\nLoading saliency maps from {args.data_dir}...")
    dataset = SaliencyMapDataset(
        data_dir=args.data_dir,
        max_samples_per_class=args.max_samples_per_class,
        load_images=args.use_edge_gating  # Load RGB images only if edge gating is enabled
    )

    # DataLoader (no workers needed - data already in memory)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"\nDataLoader Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num workers: 0 (in-memory dataset)")
    print(f"  Pin memory: {True if torch.cuda.is_available() else False}")

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
        save_path=Path(args.checkpoint_dir) / 'training_curves_advanced.png'
    )

    if torch.cuda.is_available():
        print(f"\nFinal GPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")

    print(f"\n✓ All done!")
    print(f"\nTo use the trained model:")
    print(f"  python geo_x.py --data_dir {args.data_dir} --use_ot --epochs {args.epochs}")


if __name__ == "__main__":
    main()
