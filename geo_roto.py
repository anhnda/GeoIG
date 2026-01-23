"""
Roto-Translation Covariant Diffeomorphic Dictionary Learning (Roto-LDDMM)

This module implements a part-based approach to diffeomorphic pattern learning,
addressing the "ghosting" problem in global LDDMM where thin structures (e.g.,
ostrich necks/legs) get averaged out.

Key Innovation: LEGO Perspective
================================
Instead of a global warp ϕ, the model learns:
1. Atom Bank: Dictionary of K learnable "LEGO pieces" (atoms)
2. Pose Predictor: For each image, predict SE(2) transformations (x, y, θ, s) per atom
3. Local Refinement: Apply small diffeomorphic warps to each atom
4. Composition: Soft-max assembly of transformed atoms

Mathematical Framework:
h ≈ ∑_k w_k · (d_k ∘ g_k ∘ ϕ_k)

where:
- d_k: k-th atom (learnable)
- g_k ∈ SE(2): Lie group element (translation + rotation)
- ϕ_k: Local diffeomorphism (handles biological variation)
- w_k: Attention weight

Advantages over Global LDDMM:
- Preserves thin structures (legs, necks) as distinct atoms
- Sparse representation (only active atoms contribute)
- Rotation/translation equivariant
- No entropy loss from spatial averaging

Usage:
    python geo_roto.py --data_dir ./data/saliency_imagenet1k_resnet50_100 \\
                       --num_classes 1000 --k_atoms 15 --epochs 50
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
import gc


# ==========================================
# SE(2) Transformation Layer
# ==========================================

class SE2Transform(nn.Module):
    """
    Apply SE(2) transformation (translation + rotation + scale) to a 2D signal.

    This is the core of the "LEGO" approach: each atom can be independently
    positioned and oriented in the image plane.
    """

    def __init__(self, img_res=(224, 224)):
        super().__init__()
        self.res = img_res

    def forward(self, atoms, poses):
        """
        Transform atoms according to SE(2) poses.

        Args:
            atoms: (B, K, 1, H_atom, W_atom) - Atom templates
            poses: (B, K, 4) - [tx, ty, θ, scale] for each atom
                   tx, ty ∈ [-1, 1] (normalized coordinates)
                   θ ∈ [0, 2π] (rotation angle)
                   scale ∈ (0, 2] (scaling factor)

        Returns:
            transformed_atoms: (B, K, 1, H, W) - Atoms placed in image plane
        """
        B, K, C, H_atom, W_atom = atoms.shape
        H, W = self.res

        # Extract pose parameters
        tx = poses[:, :, 0]  # (B, K)
        ty = poses[:, :, 1]  # (B, K)
        theta = poses[:, :, 2]  # (B, K)
        scale = poses[:, :, 3]  # (B, K)

        # Create affine transformation matrices for each (batch, atom) pair
        # Affine matrix format for grid_sample: [2, 3] for each transform
        cos_theta = torch.cos(theta)  # (B, K)
        sin_theta = torch.sin(theta)  # (B, K)

        # Build affine matrices: [scale*cos(θ), -scale*sin(θ), tx]
        #                        [scale*sin(θ),  scale*cos(θ), ty]
        # Shape: (B, K, 2, 3)
        affine_matrices = torch.zeros(B, K, 2, 3, device=atoms.device)
        affine_matrices[:, :, 0, 0] = scale * cos_theta
        affine_matrices[:, :, 0, 1] = -scale * sin_theta
        affine_matrices[:, :, 0, 2] = tx
        affine_matrices[:, :, 1, 0] = scale * sin_theta
        affine_matrices[:, :, 1, 1] = scale * cos_theta
        affine_matrices[:, :, 1, 2] = ty

        # Flatten batch and atoms for grid_sample
        atoms_flat = atoms.view(B * K, C, H_atom, W_atom)
        affine_flat = affine_matrices.view(B * K, 2, 3)

        # Generate sampling grid
        grid = F.affine_grid(affine_flat, [B * K, C, H, W], align_corners=True)

        # Sample from atoms
        transformed = F.grid_sample(
            atoms_flat, grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )

        # Reshape back to (B, K, C, H, W)
        transformed = transformed.view(B, K, C, H, W)

        return transformed


# ==========================================
# Local Diffeomorphic Refinement
# ==========================================

class LocalDiffeomorphicRefinement(nn.Module):
    """
    Apply small local diffeomorphisms to atoms for biological variation.

    Unlike global LDDMM, these warps are applied AFTER the SE(2) transformation,
    allowing the model to handle slight distortions (e.g., bent leg) without
    losing the core structure.
    """

    def __init__(self, img_res=(224, 224), num_steps=5):
        super().__init__()
        self.res = img_res
        self.num_steps = num_steps

        # Identity grid for warping
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

    def forward(self, v_local):
        """
        Compute diffeomorphism via scaling-and-squaring.

        Args:
            v_local: (B*K, 2, v_H, v_W) - Local velocity fields

        Returns:
            phi: (B*K, 2, H, W) - Diffeomorphic displacement fields
        """
        BK = v_local.shape[0]

        # Upsample to full resolution
        v = F.interpolate(v_local, size=self.res, mode='bilinear', align_corners=True)

        # Scale by 2^(-num_steps)
        v = v / (2 ** self.num_steps)

        # Scaling-and-squaring
        for _ in range(self.num_steps):
            flow = v.permute(0, 2, 3, 1)  # (BK, H, W, 2)
            identity = self.identity_grid.expand(BK, -1, -1, -1)
            grid = identity + flow

            v_warped = F.grid_sample(
                v, grid, mode='bilinear',
                padding_mode='border', align_corners=True
            )

            v = v + v_warped

        return v

    def warp_atoms(self, atoms, phi):
        """
        Warp atoms using diffeomorphism.

        Args:
            atoms: (B*K, 1, H, W) - Transformed atoms
            phi: (B*K, 2, H, W) - Displacement fields

        Returns:
            warped: (B*K, 1, H, W) - Warped atoms
        """
        BK = atoms.shape[0]

        # Convert displacement to sampling grid
        flow = phi.permute(0, 2, 3, 1)  # (BK, H, W, 2)
        identity = self.identity_grid.expand(BK, -1, -1, -1)
        grid = identity + flow

        # Sample
        warped = F.grid_sample(
            atoms, grid, mode='bilinear',
            padding_mode='zeros', align_corners=True
        )

        return warped


# ==========================================
# Pose and Attention Predictor
# ==========================================

class PoseAttentionPredictor(nn.Module):
    """
    Predict SE(2) poses and attention weights for each atom.

    For each input image, this network outputs:
    1. Pose parameters (tx, ty, θ, scale) for each of K atoms
    2. Attention weights w_k indicating atom importance
    3. Local velocity field for refinement (optional)
    """

    def __init__(self, k_atoms=15, img_res=(224, 224), predict_local_warp=True):
        super().__init__()
        self.K = k_atoms
        self.res = img_res
        self.predict_local_warp = predict_local_warp

        # Shared backbone
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
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 28 -> 14
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten()
        )

        feature_dim = 256 * 7 * 7

        # Head 1: Pose parameters (tx, ty, θ, scale) for K atoms
        self.pose_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.K * 4)  # 4 params per atom
        )

        # Head 2: Attention weights
        self.attention_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.K)
        )

        # Head 3: Local velocity fields (optional, for refinement)
        if predict_local_warp:
            self.local_v_res = (28, 28)  # Lower resolution for local warps
            self.local_v_dim = 2 * self.local_v_res[0] * self.local_v_res[1]
            self.local_v_head = nn.Sequential(
                nn.Linear(feature_dim, 1024),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(1024, self.K * self.local_v_dim),
                nn.Tanh()
            )
            self.local_v_scale = 0.3  # Small deformations

        # Initialize pose head to identity transformations
        nn.init.normal_(self.pose_head[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.pose_head[-1].bias, 0.0)
        # Set scale bias to 0.5 (after sigmoid -> ~0.62)
        self.pose_head[-1].bias.data[3::4] = 0.0

        # Initialize attention to uniform
        nn.init.normal_(self.attention_head[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.attention_head[-1].bias, 0.0)

    def forward(self, h_i):
        """
        Args:
            h_i: (B, 1, H, W) - Input saliency maps

        Returns:
            poses: (B, K, 4) - SE(2) parameters [tx, ty, θ, scale]
            attention: (B, K) - Attention weights (softmax normalized)
            v_local: (B, K, 2, v_H, v_W) - Local velocity fields (optional)
        """
        B = h_i.shape[0]

        # Extract features
        features = self.backbone(h_i)

        # Predict poses
        poses_flat = self.pose_head(features)  # (B, K*4)
        poses = poses_flat.view(B, self.K, 4)

        # Apply activations to constrain pose parameters
        # tx, ty: tanh (range [-1, 1])
        poses[:, :, 0:2] = torch.tanh(poses[:, :, 0:2])
        # θ: map to [0, 2π]
        poses[:, :, 2] = torch.sigmoid(poses[:, :, 2]) * 2 * np.pi
        # scale: sigmoid mapped to [0.3, 1.5]
        poses[:, :, 3] = 0.3 + 1.2 * torch.sigmoid(poses[:, :, 3])

        # Predict attention weights
        attention_logits = self.attention_head(features)  # (B, K)
        attention = F.softmax(attention_logits, dim=-1)

        # Predict local velocity fields (optional)
        v_local = None
        if self.predict_local_warp:
            v_local_flat = self.local_v_head(features)  # (B, K*v_dim)
            v_local = v_local_flat.view(B, self.K, 2, *self.local_v_res) * self.local_v_scale

        return poses, attention, v_local


# ==========================================
# Atom Bank (Learnable Dictionary)
# ==========================================

class AtomBank(nn.Module):
    """
    Learnable dictionary of K atoms (LEGO pieces) - CLASS-SPECIFIC VERSION.

    CRITICAL DESIGN: Each class has its own independent set of K atoms!
    - Shape: (num_classes, k_atoms, 1, H_atom, W_atom)
    - Example: Class 0 (ostrich) has its own atoms for [head, neck, body, leg1, leg2, ...]
    - Example: Class 1 (dog) has different atoms for [head, body, tail, leg1, ...]

    This prevents "ghosting" because:
    - Ostrich neck atom is independent from dog neck atom
    - Each atom is positioned via SE(2) transformation
    - No averaging across different poses → no blur

    Alternative: Set shared_atoms=True for memory efficiency (all classes share atoms)
    """

    def __init__(self, num_classes=1000, k_atoms=15, atom_res=(56, 56), shared_atoms=False):
        super().__init__()
        self.num_classes = num_classes
        self.K = k_atoms
        self.atom_res = atom_res
        self.shared_atoms = shared_atoms

        if shared_atoms:
            # SHARED VERSION: All classes use the same K atoms
            # Shape: (k_atoms, 1, H_atom, W_atom)
            # Memory: ~3 MB for K=15, 56×56
            atoms_init = torch.rand(k_atoms, 1, *atom_res) * 0.2
            print(f"  Atom bank mode: SHARED (all classes use same {k_atoms} atoms)")
        else:
            # CLASS-SPECIFIC VERSION: Each class has its own K atoms
            # Shape: (num_classes, k_atoms, 1, H_atom, W_atom)
            # Memory: ~3 GB for 1000 classes, K=15, 56×56
            atoms_init = torch.rand(num_classes, k_atoms, 1, *atom_res) * 0.2
            print(f"  Atom bank mode: CLASS-SPECIFIC ({num_classes} classes × {k_atoms} atoms)")

        # Sparsify initialization
        atoms_init = torch.where(
            atoms_init > 0.15,
            atoms_init,
            torch.zeros_like(atoms_init)
        )

        # Make atoms learnable parameters
        self.atoms = nn.Parameter(atoms_init)

        # Track usage counts (not learnable)
        if shared_atoms:
            self.register_buffer('usage_counts', torch.zeros(k_atoms))
        else:
            self.register_buffer('usage_counts', torch.zeros(num_classes, k_atoms))

    def forward(self, class_ids):
        """
        Retrieve atoms for a batch of classes.

        Args:
            class_ids: (B,) - Class indices

        Returns:
            atoms: (B, K, 1, H_atom, W_atom) - Atom templates (non-negative)
        """
        if self.shared_atoms:
            # All classes use the same atoms
            B = class_ids.shape[0]
            atoms = self.atoms.unsqueeze(0).expand(B, -1, -1, -1, -1)
        else:
            # Each class has its own atoms
            atoms = self.atoms[class_ids]

        # CRITICAL: Ensure atoms are non-negative (saliency values must be >= 0)
        return F.relu(atoms)

    def get_atoms_for_class(self, class_id):
        """Get all atoms for a specific class (non-negative)."""
        if self.shared_atoms:
            atoms = self.atoms
        else:
            atoms = self.atoms[class_id]

        # Ensure atoms are non-negative
        return F.relu(atoms)


# ==========================================
# Roto-LDDMM Pipeline
# ==========================================

class RotoLDDMM_Pipeline(nn.Module):
    """
    Part-based diffeomorphic pattern learning with SE(2) equivariance.

    Architecture:
    1. Predict poses (tx, ty, θ, scale) and attention weights
    2. Transform atoms using SE(2) group actions
    3. Optionally refine with local diffeomorphisms
    4. Compose using soft-max aggregation
    """

    def __init__(self, num_classes=1000, k_atoms=15, img_res=(224, 224),
                 atom_res=(56, 56), device='cuda', use_local_warp=True,
                 composition_mode='softmax', shared_atoms=False):
        super().__init__()
        self.num_classes = num_classes
        self.K = k_atoms
        self.res = img_res
        self.atom_res = atom_res
        self.device = device
        self.use_local_warp = use_local_warp
        self.composition_mode = composition_mode  # 'softmax', 'max', 'sum'
        self.shared_atoms = shared_atoms

        # 1. Atom Bank (learnable dictionary)
        # IMPORTANT: By default, each class has its own atoms (shared_atoms=False)
        self.atom_bank = AtomBank(
            num_classes=num_classes,
            k_atoms=k_atoms,
            atom_res=atom_res,
            shared_atoms=shared_atoms
        )

        # 2. Pose and Attention Predictor
        self.predictor = PoseAttentionPredictor(
            k_atoms=k_atoms,
            img_res=img_res,
            predict_local_warp=use_local_warp
        )

        # 3. SE(2) Transformer
        self.se2_transform = SE2Transform(img_res=img_res)

        # 4. Local Diffeomorphic Refinement (optional)
        if use_local_warp:
            self.local_refiner = LocalDiffeomorphicRefinement(
                img_res=img_res,
                num_steps=5
            )

        print(f"\nRoto-LDDMM Pipeline Initialized:")
        if shared_atoms:
            print(f"  Atom dictionary: SHARED ({k_atoms} atoms for all classes)")
        else:
            print(f"  Atom dictionary: CLASS-SPECIFIC ({num_classes} classes × {k_atoms} atoms each)")
        print(f"  Atom resolution: {atom_res}")
        print(f"  Image resolution: {img_res}")
        print(f"  Local warping: {'Enabled' if use_local_warp else 'Disabled'}")
        print(f"  Composition mode: {composition_mode}")

    def forward(self, h_i, class_ids=None, update_atoms=True):
        """
        Forward pass: compose image from transformed atoms.

        Args:
            h_i: (B, 1, H, W) - Input saliency maps
            class_ids: (B,) - Class labels
            update_atoms: bool - Whether to update usage tracking

        Returns:
            h_composed: (B, 1, H, W) - Composed pattern
            poses: (B, K, 4) - SE(2) parameters
            attention: (B, K) - Attention weights
            atoms_transformed: (B, K, 1, H, W) - Transformed atoms
        """
        B = h_i.shape[0]

        # Step 1: Predict poses, attention, and local velocity fields
        poses, attention, v_local = self.predictor(h_i)

        # Step 2: Get atoms for this batch's classes
        if class_ids is not None:
            atoms = self.atom_bank(class_ids)  # (B, K, 1, H_atom, W_atom)
        else:
            # Use class 0 atoms as default
            atoms = self.atom_bank(torch.zeros(B, dtype=torch.long, device=self.device))

        # Step 3: Apply SE(2) transformations
        atoms_transformed = self.se2_transform(atoms, poses)  # (B, K, 1, H, W)

        # Step 4: Apply local diffeomorphic refinement (optional)
        if self.use_local_warp and v_local is not None:
            # Flatten batch and atoms
            atoms_flat = atoms_transformed.view(B * self.K, 1, *self.res)
            v_local_flat = v_local.view(B * self.K, 2, *v_local.shape[-2:])

            # Compute diffeomorphisms
            phi_local = self.local_refiner(v_local_flat)

            # Warp atoms
            atoms_warped = self.local_refiner.warp_atoms(atoms_flat, phi_local)

            # Reshape back
            atoms_transformed = atoms_warped.view(B, self.K, 1, *self.res)

        # Step 5: Compose final pattern
        h_composed = self._compose_atoms(atoms_transformed, attention)

        # Update usage tracking
        if update_atoms and class_ids is not None:
            with torch.no_grad():
                if self.shared_atoms:
                    # Shared atoms: aggregate attention across all samples
                    self.atom_bank.usage_counts += attention.sum(dim=0)
                else:
                    # Class-specific atoms: track per-class usage
                    for b in range(B):
                        self.atom_bank.usage_counts[class_ids[b]] += attention[b]

        return h_composed, poses, attention, atoms_transformed

    def _compose_atoms(self, atoms, attention):
        """
        Compose atoms using attention-weighted aggregation.

        Args:
            atoms: (B, K, 1, H, W) - Transformed atoms
            attention: (B, K) - Attention weights

        Returns:
            composed: (B, 1, H, W) - Composed pattern
        """
        B, K, _, _, _ = atoms.shape  # C, H, W not used in this method

        if self.composition_mode == 'softmax':
            # Weighted sum (standard)
            # attention: (B, K) -> (B, K, 1, 1, 1)
            weights = attention.view(B, K, 1, 1, 1)
            composed = (atoms * weights).sum(dim=1)  # (B, 1, H, W)

        elif self.composition_mode == 'max':
            # Max-pooling across atoms (preserves sharp structures)
            composed = atoms.max(dim=1)[0]  # (B, 1, H, W)

        elif self.composition_mode == 'sum':
            # Simple sum (no attention weighting)
            composed = atoms.sum(dim=1)  # (B, 1, H, W)

        else:
            raise ValueError(f"Unknown composition mode: {self.composition_mode}")

        return composed


# ==========================================
# Loss Functions
# ==========================================

class RotoLDDMMLoss(nn.Module):
    """
    Loss functions for Roto-LDDMM with improved structural priors.

    Key improvements:
    1. Total Variation (TV) on atoms for smooth, connected shapes
    2. Atom compactness to prevent fragmented atoms
    3. Corrected attention entropy (encourage sparse usage)
    4. Structural diversity enforcement
    """

    def __init__(self,
                 lambda_reconstruction=1.0,
                 lambda_atom_sparsity=3.0,  # Increased for cleaner atoms
                 lambda_atom_diversity=2.0,
                 lambda_pose_reg=0.01,
                 lambda_attention_sparsity=1.0,  # Renamed and fixed sign
                 lambda_local_smooth=0.05,
                 lambda_mass_conservation=1.0,
                 lambda_atom_tv=2.0,  # NEW: Total Variation on atoms
                 lambda_atom_compactness=1.0):  # NEW: Atom connectivity
        super().__init__()
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_atom_sparsity = lambda_atom_sparsity
        self.lambda_atom_diversity = lambda_atom_diversity
        self.lambda_pose_reg = lambda_pose_reg
        self.lambda_attention_sparsity = lambda_attention_sparsity
        self.lambda_local_smooth = lambda_local_smooth
        self.lambda_mass_conservation = lambda_mass_conservation
        self.lambda_atom_tv = lambda_atom_tv
        self.lambda_atom_compactness = lambda_atom_compactness

    def reconstruction_loss(self, h_original, h_composed):
        """MSE between input and composed pattern."""
        return F.mse_loss(h_composed, h_original)

    def atom_sparsity_loss(self, atoms):
        """Encourage atoms to be sparse (lots of zeros)."""
        # L1 penalty on atom intensities
        return torch.abs(atoms).mean()

    def atom_diversity_loss(self, atoms):
        """
        Penalize atoms for being too similar.

        Args:
            atoms: (B, K, 1, H, W) - Atoms
        """
        B, K, _, _, _ = atoms.shape  # C, H, W not used
        atoms_flat = atoms.view(B, K, -1)

        # Normalize
        atoms_norm = F.normalize(atoms_flat, p=2, dim=-1)

        # Compute pairwise similarities
        similarity = torch.bmm(atoms_norm, atoms_norm.transpose(1, 2))  # (B, K, K)

        # Mask diagonal (self-similarity)
        mask = torch.eye(K, device=atoms.device).unsqueeze(0)
        similarity = similarity * (1 - mask)

        # Penalize high similarity
        return torch.abs(similarity).mean()

    def pose_regularization_loss(self, poses):
        """
        Penalize extreme poses.

        Args:
            poses: (B, K, 4) - [tx, ty, θ, scale]
        """
        # Penalize large translations
        translation_penalty = (poses[:, :, 0:2] ** 2).mean()

        # Penalize extreme scales (prefer scale ≈ 1)
        scale_penalty = ((poses[:, :, 3] - 1.0) ** 2).mean()

        return translation_penalty + scale_penalty

    def attention_sparsity_loss(self, attention):
        """
        Encourage sparse attention (few active atoms per image).

        Uses negative entropy: lower entropy means peaky distribution
        (1-2 dominant atoms) rather than uniform distribution.

        Args:
            attention: (B, K) - Attention weights (softmax normalized)
        """
        # Compute entropy: H = -sum(p * log(p))
        # Add epsilon to avoid log(0)
        entropy = -(attention * torch.log(attention + 1e-10)).sum(dim=1)

        # Minimize entropy to encourage sparse attention
        # (peaky distribution = low entropy)
        return entropy.mean()

    def local_smoothness_loss(self, v_local):
        """
        Spatial gradient penalty on local velocity fields.

        Args:
            v_local: (B, K, 2, v_H, v_W) - Local velocity fields
        """
        if v_local is None:
            return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')

        # Flatten batch and atoms
        B, K = v_local.shape[0], v_local.shape[1]
        v_flat = v_local.view(B * K, 2, *v_local.shape[-2:])

        diff_h = torch.abs(v_flat[:, :, 1:, :] - v_flat[:, :, :-1, :])
        diff_w = torch.abs(v_flat[:, :, :, 1:] - v_flat[:, :, :, :-1])

        return diff_h.mean() + diff_w.mean()

    def atom_total_variation_loss(self, atoms):
        """
        Total Variation (TV) loss on atoms to encourage smooth, connected shapes.

        TV penalizes rapid changes in pixel values, encouraging atoms to form
        coherent blobs rather than scattered pixels.

        Args:
            atoms: (B, K, 1, H, W) - Atom templates
        """
        B, K, C, H, W = atoms.shape

        # Flatten batch and atoms
        atoms_flat = atoms.view(B * K, C, H, W)

        # Compute TV (sum of absolute gradients)
        diff_h = torch.abs(atoms_flat[:, :, 1:, :] - atoms_flat[:, :, :-1, :])
        diff_w = torch.abs(atoms_flat[:, :, :, 1:] - atoms_flat[:, :, :, :-1])

        tv = diff_h.sum() + diff_w.sum()

        # Normalize by number of pixels
        tv = tv / (B * K * C * H * W)

        return tv

    def atom_compactness_loss(self, atoms):
        """
        Encourage atoms to be compact (active pixels close together).

        Penalizes the variance of active pixel positions, encouraging atoms
        to form tight, connected regions rather than scattered activations.

        Args:
            atoms: (B, K, 1, H, W) - Atom templates
        """
        B, K, C, H, W = atoms.shape

        # Create coordinate grids
        y_coords = torch.arange(H, device=atoms.device, dtype=torch.float32).view(1, 1, 1, H, 1)
        x_coords = torch.arange(W, device=atoms.device, dtype=torch.float32).view(1, 1, 1, 1, W)

        # Use absolute values to ensure positive mass (atoms can be negative during training)
        atoms_abs = torch.abs(atoms)

        # Normalize as probability distributions (now guaranteed positive)
        total_mass = atoms_abs.sum(dim=(2, 3, 4), keepdim=True)
        atoms_norm = atoms_abs / (total_mass + 1e-8)

        # Compute centers of mass
        center_y = (atoms_norm * y_coords).sum(dim=(2, 3, 4))  # (B, K)
        center_x = (atoms_norm * x_coords).sum(dim=(2, 3, 4))  # (B, K)

        # Compute variance (spread) around center
        var_y = (atoms_norm * (y_coords - center_y.view(B, K, 1, 1, 1))**2).sum(dim=(2, 3, 4))
        var_x = (atoms_norm * (x_coords - center_x.view(B, K, 1, 1, 1))**2).sum(dim=(2, 3, 4))

        # Penalize large variance (encourage compact atoms)
        compactness = (var_y + var_x).mean()

        return compactness

    def mass_conservation_loss(self, h_original, h_composed):
        """Preserve total intensity."""
        mass_original = h_original.sum(dim=(1, 2, 3))
        mass_composed = h_composed.sum(dim=(1, 2, 3))
        relative_change = torch.abs(mass_composed - mass_original) / (mass_original + 1e-8)
        return relative_change.mean()

    def forward(self, h_original, h_composed, atoms, poses, attention, v_local=None):
        """
        Compute total loss with improved structural priors.

        Args:
            h_original: (B, 1, H, W) - Input saliency
            h_composed: (B, 1, H, W) - Composed pattern
            atoms: (B, K, 1, H_atom, W_atom) - Atoms
            poses: (B, K, 4) - SE(2) parameters
            attention: (B, K) - Attention weights
            v_local: (B, K, 2, v_H, v_W) - Local velocity fields (optional)

        Returns:
            loss_dict: Dictionary of loss components
        """
        loss_recon = self.reconstruction_loss(h_original, h_composed)
        loss_atom_sparse = self.atom_sparsity_loss(atoms)
        loss_atom_div = self.atom_diversity_loss(atoms)
        loss_pose_reg = self.pose_regularization_loss(poses)
        loss_attention_sparse = self.attention_sparsity_loss(attention)
        loss_local_smooth = self.local_smoothness_loss(v_local)
        loss_mass_cons = self.mass_conservation_loss(h_original, h_composed)
        loss_atom_tv = self.atom_total_variation_loss(atoms)
        loss_atom_compact = self.atom_compactness_loss(atoms)

        total_loss = (
            self.lambda_reconstruction * loss_recon +
            self.lambda_atom_sparsity * loss_atom_sparse +
            self.lambda_atom_diversity * loss_atom_div +
            self.lambda_pose_reg * loss_pose_reg +
            self.lambda_attention_sparsity * loss_attention_sparse +
            self.lambda_local_smooth * loss_local_smooth +
            self.lambda_mass_conservation * loss_mass_cons +
            self.lambda_atom_tv * loss_atom_tv +
            self.lambda_atom_compactness * loss_atom_compact
        )

        return {
            'total': total_loss,
            'reconstruction': loss_recon,
            'atom_sparsity': loss_atom_sparse,
            'atom_diversity': loss_atom_div,
            'pose_regularization': loss_pose_reg,
            'attention_sparsity': loss_attention_sparse,
            'local_smoothness': loss_local_smooth,
            'mass_conservation': loss_mass_cons,
            'atom_tv': loss_atom_tv,
            'atom_compactness': loss_atom_compact
        }


# ==========================================
# Dataset (Reuse from geo_s3.py)
# ==========================================

def safe_collate_fn(batch):
    """Custom collate function with validation."""
    if len(batch[0]) == 3:
        saliencies, labels, images = zip(*batch)
    else:
        raise ValueError(f"Unexpected batch item length: {len(batch[0])}")

    for i, sal in enumerate(saliencies):
        if sal.shape != (1, 224, 224):
            raise ValueError(f"Sample {i}: Invalid saliency shape {sal.shape}")

    saliency_batch = torch.stack(saliencies, dim=0)
    label_batch = torch.tensor(labels, dtype=torch.long)

    has_any_none = any(img is None for img in images)
    if not has_any_none and images[0] is not None:
        image_batch = torch.stack(images, dim=0)
    else:
        image_batch = None

    return saliency_batch, label_batch, image_batch


class SaliencyMapDataset(Dataset):
    """RAM-optimized dataset (float16 saliency, uint8 RGB)."""

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

        for batch_file in tqdm(batch_files, desc="Loading data"):
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
                    if saliency_np.shape[0] not in [1, 3]:
                        if saliency_np.shape[2] in [1, 3]:
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

        print(f"Loaded {len(self.saliency_maps)} samples ({corrupted_samples} corrupted)")

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
# Trainer
# ==========================================

class RotoLDDMMTrainer:
    """Trainer for Roto-LDDMM pipeline with warm-up phase."""

    def __init__(self, model, train_loader, val_loader=None,
                 lr=1e-3, device='cuda', checkpoint_dir='./checkpoints_roto',
                 warmup_epochs=0):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.warmup_epochs = warmup_epochs  # Freeze atoms for first N epochs

        # Optimizer (separate learning rates for atoms and predictor)
        self.optimizer = optim.Adam([
            {'params': self.model.atom_bank.parameters(), 'lr': lr * 0.3},  # Much slower for atoms
            {'params': self.model.predictor.parameters(), 'lr': lr},
        ], weight_decay=1e-5)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        # Improved loss with structural priors
        self.criterion = RotoLDDMMLoss(
            lambda_reconstruction=1.0,
            lambda_atom_sparsity=5.0,          # Increased for cleaner atoms
            lambda_atom_diversity=2.0,
            lambda_pose_reg=0.01,
            lambda_attention_sparsity=2.0,     # Encourage sparse attention
            lambda_local_smooth=0.05,
            lambda_mass_conservation=1.0,
            lambda_atom_tv=3.0,                # Strong TV for smooth atoms
            lambda_atom_compactness=2.0        # Encourage compact atoms
        ).to(device)

        self.history = defaultdict(list)

        print(f"\nTrainer Configuration:")
        print(f"  Warmup epochs: {warmup_epochs} (atoms frozen)")
        print(f"  Atom learning rate: {lr * 0.3:.6f}")
        print(f"  Predictor learning rate: {lr:.6f}")
        print(f"  Loss weights:")
        print(f"    - Reconstruction: 1.0")
        print(f"    - Atom Sparsity: 5.0 (high for clean atoms)")
        print(f"    - Atom TV: 3.0 (smooth, connected shapes)")
        print(f"    - Atom Compactness: 2.0 (prevent fragmentation)")
        print(f"    - Attention Sparsity: 2.0 (few atoms per image)")
        print(f"    - Atom Diversity: 2.0 (different atoms)")

    def train_epoch(self, epoch):
        """Train for one epoch with optional warmup."""
        self.model.train()

        # Warmup phase: freeze atoms
        if epoch <= self.warmup_epochs:
            print(f"  [Warmup Phase] Atoms frozen, training predictor only")
            for param in self.model.atom_bank.parameters():
                param.requires_grad = False
        else:
            # Unfreeze atoms after warmup
            for param in self.model.atom_bank.parameters():
                param.requires_grad = True

        epoch_losses = defaultdict(float)
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch_data in enumerate(pbar):
            if len(batch_data) == 3:
                saliency_maps, labels, _ = batch_data  # rgb_images not used
            else:
                saliency_maps, labels = batch_data

            saliency_maps = saliency_maps.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            h_composed, poses, attention, _ = self.model(  # atoms_transformed not used here
                saliency_maps,
                class_ids=labels,
                update_atoms=True
            )

            # Get atoms (for loss computation)
            atoms = self.model.atom_bank(labels)

            # Get v_local from predictor
            _, _, v_local = self.model.predictor(saliency_maps)

            # Compute loss
            losses = self.criterion(
                saliency_maps, h_composed, atoms, poses, attention, v_local
            )

            # Backward
            self.optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Accumulate losses
            for key, value in losses.items():
                epoch_losses[key] += value.item()
            num_batches += 1

            pbar.set_postfix({
                'loss': losses['total'].item(),
                'recon': losses['reconstruction'].item(),
                'tv': losses['atom_tv'].item(),
                'attn_sp': losses['attention_sparsity'].item()
            })

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
        print(f"Starting Roto-LDDMM Training")
        print(f"{'='*80}")

        for epoch in range(1, num_epochs + 1):
            avg_losses = self.train_epoch(epoch)

            print(f"\nEpoch {epoch}/{num_epochs} Summary:")
            print(f"  Total Loss: {avg_losses['total']:.4f}")
            print(f"  Reconstruction: {avg_losses['reconstruction']:.4f}")
            print(f"  Atom TV: {avg_losses['atom_tv']:.4f}")
            print(f"  Atom Compactness: {avg_losses['atom_compactness']:.4f}")
            print(f"  Atom Sparsity: {avg_losses['atom_sparsity']:.4f}")
            print(f"  Atom Diversity: {avg_losses['atom_diversity']:.4f}")
            print(f"  Attention Sparsity: {avg_losses['attention_sparsity']:.4f}")
            print(f"  Mass Conservation: {avg_losses['mass_conservation']:.4f}")

            self.scheduler.step(avg_losses['total'])

            if epoch % save_frequency == 0:
                self.save_checkpoint(epoch)

        self.save_checkpoint(num_epochs, final=True)

        print(f"\n{'='*80}")
        print(f"✓ Training Complete!")
        print(f"{'='*80}")

    def save_checkpoint(self, epoch, final=False):
        """Save model checkpoint."""
        if final:
            checkpoint_path = self.checkpoint_dir / "roto_lddmm_model_final.pth"
        else:
            checkpoint_path = self.checkpoint_dir / f"roto_lddmm_model_epoch_{epoch}.pth"

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': dict(self.history),
            'atoms': self.model.atom_bank.atoms,
            'usage_counts': self.model.atom_bank.usage_counts
        }, checkpoint_path)

        print(f"  Checkpoint saved: {checkpoint_path}")


# ==========================================
# Main Script
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description='Roto-Translation Covariant Diffeomorphic Dictionary Learning',
        epilog="""
INNOVATION: LEGO-Based Pattern Learning
  - Atoms: Learnable dictionary of body parts
  - SE(2): Rotation + translation equivariance
  - Local warps: Handle biological variation
  - Soft composition: Attention-weighted assembly

USAGE:
  python geo_roto.py --data_dir ./data/saliency_imagenet1k_resnet50_100 \\
                     --num_classes 1000 --k_atoms 15 --epochs 50
        """
    )
    parser.add_argument('--data_dir', type=str,
                       default='./data/saliency_imagenet1k_resnet50_100',
                       help='Directory with saliency maps')
    parser.add_argument('--num_classes', type=int, default=1000,
                       help='Number of classes')
    parser.add_argument('--k_atoms', type=int, default=15,
                       help='Number of atoms per class')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--atom_res', type=int, default=56,
                       help='Atom resolution (square)')
    parser.add_argument('--composition_mode', type=str, default='softmax',
                       choices=['softmax', 'max', 'sum'],
                       help='Atom composition mode')
    parser.add_argument('--no_local_warp', action='store_true',
                       help='Disable local diffeomorphic refinement')
    parser.add_argument('--shared_atoms', action='store_true',
                       help='Use shared atoms across all classes (saves memory)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                       help='Number of epochs to freeze atoms and train predictor only (default: 5)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_roto',
                       help='Checkpoint directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"Roto-LDDMM: Part-Based Diffeomorphic Learning (Improved)")
    print(f"{'='*80}")
    print(f"Device: {device}")

    # Memory estimate for atom bank
    atom_memory_gb = (args.num_classes * args.k_atoms * 1 * args.atom_res * args.atom_res * 4) / (1024**3)
    shared_memory_gb = (args.k_atoms * 1 * args.atom_res * args.atom_res * 4) / (1024**3)
    print(f"\nAtom Bank Configuration:")
    if args.shared_atoms:
        print(f"  Mode: SHARED atoms (all classes use same {args.k_atoms} atoms)")
        print(f"  Memory: ~{shared_memory_gb:.3f} GB")
    else:
        print(f"  Mode: CLASS-SPECIFIC atoms (each class has own {args.k_atoms} atoms)")
        print(f"  Memory: ~{atom_memory_gb:.3f} GB")
        print(f"  (Use --shared_atoms to reduce to ~{shared_memory_gb:.3f} GB)")

    print(f"\nTraining Strategy:")
    print(f"  Warmup epochs: {args.warmup_epochs} (atoms frozen, train predictor)")
    print(f"  Main training: Epochs {args.warmup_epochs + 1}-{args.epochs}")
    print(f"  Improved losses: TV + Compactness + Attention Sparsity")

    # Load dataset
    print(f"\nLoading saliency maps from {args.data_dir}...")
    dataset = SaliencyMapDataset(
        data_dir=args.data_dir,
        max_samples_per_class=None,
        load_images=False,
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

    # Create model
    print(f"\nInitializing Roto-LDDMM model...")
    model = RotoLDDMM_Pipeline(
        num_classes=args.num_classes,
        k_atoms=args.k_atoms,
        img_res=(224, 224),
        atom_res=(args.atom_res, args.atom_res),
        device=device,
        use_local_warp=not args.no_local_warp,
        composition_mode=args.composition_mode,
        shared_atoms=args.shared_atoms
    )

    # Create trainer
    trainer = RotoLDDMMTrainer(
        model=model,
        train_loader=train_loader,
        lr=args.lr,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        warmup_epochs=args.warmup_epochs
    )

    # Train
    trainer.train(num_epochs=args.epochs, save_frequency=5)

    print(f"\n✓ All done!")


if __name__ == "__main__":
    main()
