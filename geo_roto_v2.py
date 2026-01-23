"""
Roto-LDDMM V2.2: STRUCTURE OVER SMOOTHNESS
===========================================

Evolution from V2.0 → V2.2:
- V2.0: Captured dots perfectly but lost topological meaning (fragmented textures)
- V2.1: Re-introduced compactness to group dots → Result: Blobs, not anatomical structures
- V2.2: **RECONSTRUCTION DOMINATES** - let atoms stretch/deform to match skeletal structure

Core Philosophy (V2.2)
======================
**Problem**: V2.1 produced smooth blobs instead of bird-like skeletal structures because
smoothness priors (TV, compactness) dominated over reconstruction signal.

**Solution**: Shift power balance to reconstruction:
- Reconstruction weight: 1.0 → 15.0 (15× increase)
- TV loss: 0.5 → 0.01 (allow anatomical detail)
- Compactness: 0.3 → 0.005 (allow elongated atoms - necks, legs)
- Attention sparsity: 4.0 → 0.5 (allow 6-8 atoms to collaborate)
- Gaussian annealing: Keep σ ≥ 1.0 until epoch 40 (persistent scaffolding)

V2.2 Key Features
=================
1. **Anisotropic Scaling**: sx, sy parameters allow atoms to stretch independently
   - 56×56 square atoms can become 20×120 rectangles for long necks

2. **Persistent Gaussian Scaffolding**:
   - Refinement (21-40): σ from 2.0 → 1.0 (slower, was →0.5)
   - Finalize (41-50): σ from 1.0 → 0 (was from 0.5)
   - Keeps "hills" large enough for atoms to find distant structures

3. **Minimal Smoothness Constraints**:
   - Let atoms fragment/stretch if needed to match reconstruction
   - Multiple atoms (6-8) collaborate to build complex shapes

4. **Structure over Smoothness**:
   - Atoms should look like "smudges" or "lines", not blobs
   - Composed map forms rough skeletal structure, not smooth reconstruction

Expected Results
================
- Atom shapes: Elongated smudges/lines (not round blobs)
- Atom usage: 6-8 atoms active (not just 3-4)
- Composed map: Skeletal bird structure with overlapping parts

Usage:
    python geo_roto_v2.py --data_dir ./data/ig_maps \\
                          --num_classes 1000 --k_atoms 15 --epochs 50
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from collections import defaultdict
import gc

# Import utilities from roto_utils
from roto_utils import (
    GaussianBlur,
    SE2Transform,
    LocalDiffeomorphicRefinement,
    get_part_seeds,
    sample_seed_maps,
    safe_collate_fn,
    SaliencyMapDataset
)


# ==========================================
# Pose and Attention Predictor
# ==========================================

class PoseAttentionPredictor(nn.Module):
    """
    Predict SE(2) poses and attention weights for each atom.

    For each input image, this network outputs:
    1. Pose parameters (tx, ty, θ, sx, sy) for each of K atoms
       - tx, ty: translation
       - θ: rotation
       - sx, sy: anisotropic scaling (allows elongated structures like necks/legs)
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

        # Head 1: Pose parameters (tx, ty, θ, sx, sy) for K atoms
        # CHANGED: Now 5 params per atom to support anisotropic scaling
        self.pose_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.K * 5)  # 5 params per atom: tx, ty, θ, sx, sy
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
            # BALANCE: Set to 0.2 as middle ground between rigidity (0.05) and smearing (0.3+)
            # Allows organic curves (bent necks, legs) without over-smearing
            self.local_v_scale = 0.2  # Small-to-moderate deformations (was 0.3, then 0.05)

        # Initialize pose head to identity transformations
        nn.init.normal_(self.pose_head[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.pose_head[-1].bias, 0.0)
        # Set scale biases for sx and sy to 0.0 (after sigmoid -> ~0.5)
        self.pose_head[-1].bias.data[3::5] = 0.0  # sx (every 5th param starting at index 3)
        self.pose_head[-1].bias.data[4::5] = 0.0  # sy (every 5th param starting at index 4)

        # Initialize attention to uniform
        nn.init.normal_(self.attention_head[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.attention_head[-1].bias, 0.0)

    def forward(self, h_i):
        """
        Args:
            h_i: (B, 1, H, W) - Input saliency maps

        Returns:
            poses: (B, K, 5) - SE(2) parameters [tx, ty, θ, sx, sy]
            attention: (B, K) - Attention weights (softmax normalized)
            v_local: (B, K, 2, v_H, v_W) - Local velocity fields (optional)
        """
        B = h_i.shape[0]

        # Extract features
        features = self.backbone(h_i)

        # Predict poses
        poses_flat = self.pose_head(features)  # (B, K*5)
        poses = poses_flat.view(B, self.K, 5)

        # Apply activations to constrain pose parameters
        # tx, ty: tanh (range [-1, 1])
        poses[:, :, 0:2] = torch.tanh(poses[:, :, 0:2])
        # θ: map to [0, 2π]
        poses[:, :, 2] = torch.sigmoid(poses[:, :, 2]) * 2 * np.pi
        # sx (scale x): sigmoid mapped to [0.2, 2.0] for anisotropic scaling
        poses[:, :, 3] = 0.2 + 1.8 * torch.sigmoid(poses[:, :, 3])
        # sy (scale y): sigmoid mapped to [0.2, 2.0] for anisotropic scaling
        poses[:, :, 4] = 0.2 + 1.8 * torch.sigmoid(poses[:, :, 4])

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

    V2 IMPROVEMENT: Smart seeded initialization using greedy peak detection.

    Instead of random initialization, atoms are initialized from actual data:
    1. Sample representative images from each class
    2. Use greedy peak-finding to extract anatomical parts
    3. Initialize atoms with these discovered patches

    This dramatically improves convergence because atoms start near
    meaningful features (head, body, legs) rather than random noise.

    CRITICAL DESIGN: Each class has its own independent set of K atoms!
    - Shape: (num_classes, k_atoms, 1, H_atom, W_atom)
    - Example: Class 0 (ostrich) has atoms for [body, neck, head, leg1, leg2, ...]
    - Example: Class 1 (dog) has atoms for [body, head, tail, leg1, ...]
    """

    def __init__(self, num_classes=1000, k_atoms=15, atom_res=(56, 56),
                 shared_atoms=False, seed_maps=None):
        """
        Args:
            num_classes: int - Number of classes
            k_atoms: int - Number of atoms per class
            atom_res: tuple - Atom resolution (H, W)
            shared_atoms: bool - Whether to share atoms across classes
            seed_maps: dict or Tensor - Optional initialization seeds
                       If dict: {class_id: (1, H, W) tensor} - one sample per class
                       If Tensor: (num_classes, 1, H, W) - one sample per class
                       If None: fall back to random initialization
        """
        super().__init__()
        self.num_classes = num_classes
        self.K = k_atoms
        self.atom_res = atom_res
        self.shared_atoms = shared_atoms

        # Initialize atoms
        if seed_maps is not None:
            print(f"  Using SMART SEEDED initialization (greedy peak detection)")
            atoms_init = self._initialize_from_seeds(seed_maps, shared_atoms)
        else:
            print(f"  Using RANDOM initialization (fallback)")
            if shared_atoms:
                atoms_init = torch.rand(k_atoms, 1, *atom_res) * 0.2
            else:
                atoms_init = torch.rand(num_classes, k_atoms, 1, *atom_res) * 0.2

            # Sparsify random initialization
            atoms_init = torch.where(
                atoms_init > 0.15,
                atoms_init,
                torch.zeros_like(atoms_init)
            )

        if shared_atoms:
            print(f"  Atom bank mode: SHARED (all classes use same {k_atoms} atoms)")
        else:
            print(f"  Atom bank mode: CLASS-SPECIFIC ({num_classes} classes × {k_atoms} atoms)")

        # Make atoms learnable parameters
        self.atoms = nn.Parameter(atoms_init)

        # Track usage counts (not learnable)
        if shared_atoms:
            self.register_buffer('usage_counts', torch.zeros(k_atoms))
        else:
            self.register_buffer('usage_counts', torch.zeros(num_classes, k_atoms))

    def _initialize_from_seeds(self, seed_maps, shared_atoms):
        """
        Initialize atoms using greedy peak detection on seed maps.

        IMPORTANT: Applies spatial dilation to give atoms a "body" that can
        capture gradients during early training. Without this, tiny dot-atoms
        won't see the gradient signal from blurred targets.

        Args:
            seed_maps: dict or Tensor - Seed saliency maps
            shared_atoms: bool - Whether to share atoms across classes

        Returns:
            atoms_init: Tensor - Initialized atoms
        """
        if shared_atoms:
            # For shared atoms, average across all classes
            if isinstance(seed_maps, dict):
                # Average all seed maps
                all_maps = torch.stack(list(seed_maps.values()), dim=0)  # (C, 1, H, W)
                avg_map = all_maps.mean(dim=0)  # (1, H, W)
            else:
                # seed_maps is (num_classes, 1, H, W)
                avg_map = seed_maps.mean(dim=0)  # (1, H, W)

            # Find parts in averaged map
            _, patches = get_part_seeds(avg_map, k_atoms=self.K, patch_size=self.atom_res[0])
            atoms_init = patches  # (K, 1, patch_size, patch_size)

        else:
            # Class-specific atoms
            atoms_init = torch.zeros(self.num_classes, self.K, 1, *self.atom_res)

            if isinstance(seed_maps, dict):
                # Process each class that has a seed
                for class_id, seed_map in seed_maps.items():
                    if class_id < self.num_classes:
                        _, patches = get_part_seeds(seed_map, k_atoms=self.K,
                                                   patch_size=self.atom_res[0])
                        atoms_init[class_id] = patches
            else:
                # seed_maps is (num_classes, 1, H, W)
                for class_id in range(min(self.num_classes, seed_maps.shape[0])):
                    _, patches = get_part_seeds(seed_maps[class_id], k_atoms=self.K,
                                               patch_size=self.atom_res[0])
                    atoms_init[class_id] = patches

        # CRITICAL FIX: Dilate patches to give atoms a "body"
        # This transforms tiny dot-seeds into blob-seeds that can catch gradients
        # Use 7x7 max pooling to expand active regions
        print(f"  Applying spatial dilation (7x7 max pool) to atom seeds...")
        if shared_atoms:
            # atoms_init: (K, 1, H, W)
            atoms_init = F.max_pool2d(atoms_init, kernel_size=7, stride=1, padding=3)
        else:
            # atoms_init: (num_classes, K, 1, H, W)
            # Flatten to (num_classes*K, 1, H, W) for max_pool2d
            orig_shape = atoms_init.shape
            atoms_flat = atoms_init.view(-1, 1, *self.atom_res)
            atoms_flat = F.max_pool2d(atoms_flat, kernel_size=7, stride=1, padding=3)
            atoms_init = atoms_flat.view(orig_shape)

        # Add small noise to break symmetry
        atoms_init = atoms_init + torch.randn_like(atoms_init) * 0.01

        # Ensure non-negative
        atoms_init = torch.clamp(atoms_init, min=0.0)

        return atoms_init

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
                 composition_mode='max', shared_atoms=False, seed_maps=None):  # V2: seed_maps for smart init
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
        # V2: Use smart seeded initialization if seed_maps provided
        self.atom_bank = AtomBank(
            num_classes=num_classes,
            k_atoms=k_atoms,
            atom_res=atom_res,
            shared_atoms=shared_atoms,
            seed_maps=seed_maps  # V2: Smart initialization!
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
    Loss functions for Roto-LDDMM V2: Optimized for dot-point IG maps.

    Key Changes V2.3 (ANTI-COLLAPSE UPDATE):
    1. Weighted reconstruction loss: 10× weight for positive signal vs zeros
    2. Peak penalty: Prevent atoms from collapsing to sharp dots
    3. Removed compactness penalty: Allow atoms to spread naturally
    4. Increased TV (0.1): Encourage smooth, spread-out atoms
    5. Minimum blur (σ≥0.5): Maintain gradient signal for diffuse regions
    """

    def __init__(self,
                 lambda_reconstruction=1.0,
                 lambda_atom_sparsity=3.0,          # Moderate: allow atoms to capture regions
                 lambda_atom_diversity=2.0,         # Moderate: different atoms
                 lambda_pose_reg=0.01,              # Low: allow flexible poses
                 lambda_attention_sparsity=4.0,     # High: force fewer, better atoms per sample
                 lambda_local_smooth=0.8,           # Keep warps smooth but allow organic curves
                 lambda_mass_conservation=1.0,      # Standard
                 lambda_atom_tv=0.5,                # RE-ENABLED: group dots into connected parts
                 lambda_atom_compactness=0.3,       # RE-INTRODUCED: force dots to form a "part"
                 lambda_atom_usage_balance=2.0,     # Prevent mode collapse
                 lambda_atom_overlap=1.0,           # NEW: prevent atoms from stacking
                 lambda_atom_peak_penalty=0.5,      # NEW: prevent overly peaked atoms
                 signal_weight=10.0):               # NEW: weight for positive signal in reconstruction
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
        self.lambda_atom_usage_balance = lambda_atom_usage_balance
        self.lambda_atom_overlap = lambda_atom_overlap
        self.lambda_atom_peak_penalty = lambda_atom_peak_penalty
        self.signal_weight = signal_weight

    def reconstruction_loss(self, h_original, h_composed, signal_weight=10.0):
        """
        Weighted MSE between input and composed pattern.

        Gives higher weight to positive signal regions (where h_original > 0)
        to prevent the model from just matching the easy zero regions.

        Args:
            h_original: (B, 1, H, W) - Target saliency maps
            h_composed: (B, 1, H, W) - Reconstructed saliency maps
            signal_weight: float - Weight multiplier for positive signal regions (default: 10.0)

        Returns:
            Weighted MSE loss
        """
        # Compute per-pixel squared error
        squared_error = (h_composed - h_original) ** 2

        # Create weight map: higher weight for positive signal regions
        # Use a threshold to identify signal vs background (small values might be noise)
        signal_threshold = h_original.max() * 0.01  # 1% of max value
        signal_mask = (h_original > signal_threshold).float()

        # Weight map: 1.0 for background, signal_weight for signal regions
        weight_map = 1.0 + (signal_weight - 1.0) * signal_mask

        # Weighted MSE
        weighted_error = squared_error * weight_map

        # Normalize by total weight to keep loss magnitude comparable
        total_weight = weight_map.sum()
        loss = weighted_error.sum() / (total_weight + 1e-8)

        return loss

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
            poses: (B, K, 5) - [tx, ty, θ, sx, sy]
        """
        # Penalize large translations
        translation_penalty = (poses[:, :, 0:2] ** 2).mean()

        # Penalize extreme scales (prefer sx ≈ 1 and sy ≈ 1)
        scale_x_penalty = ((poses[:, :, 3] - 1.0) ** 2).mean()
        scale_y_penalty = ((poses[:, :, 4] - 1.0) ** 2).mean()

        return translation_penalty + scale_x_penalty + scale_y_penalty

    def attention_sparsity_loss(self, attention):
        """
        Encourage moderate attention diversity (3-5 active atoms per image).

        ANTI-COLLAPSE FIX: Instead of minimizing entropy (which causes collapse
        to 1 atom), we target an entropy corresponding to ~3-5 atoms.

        For K atoms:
        - Entropy = 0: Only 1 atom active (COLLAPSE!)
        - Entropy = log(K): All K atoms equally active (too diffuse)
        - Entropy = log(4): ~4 atoms active (SWEET SPOT)

        Args:
            attention: (B, K) - Attention weights (softmax normalized)
        """
        # Compute entropy: H = -sum(p * log(p))
        entropy = -(attention * torch.log(attention + 1e-10)).sum(dim=1)

        # Target entropy: log(4) ≈ 1.386 (encourages ~4 atoms per sample)
        # This prevents both collapse (H→0) and over-diffusion (H→log(K))
        target_entropy = np.log(4.0)

        # Penalize deviation from target
        return ((entropy - target_entropy) ** 2).mean()

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
        B, K, _, H, W = atoms.shape  # _ = C (channels, not used)

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

    def atom_usage_balance_loss(self, attention):
        """
        Encourage balanced usage of all atoms across the batch.

        Prevents mode collapse where only 1-2 atoms are used and others ignored.
        Computes the variance of mean attention across atoms - low variance means
        all atoms are used equally.

        Args:
            attention: (B, K) - Attention weights across batch

        Returns:
            loss: Scalar - Variance of atom usage (minimize for balance)
        """
        # Mean attention per atom across the batch
        mean_attention_per_atom = attention.mean(dim=0)  # (K,)

        # Compute variance - we want this to be small (uniform usage)
        # If all atoms used equally, each should have ~1/K attention
        target_uniform = 1.0 / attention.shape[1]
        variance = ((mean_attention_per_atom - target_uniform) ** 2).mean()

        return variance

    def atom_overlap_penalty(self, atoms_transformed, attention):
        """
        Penalize atoms that overlap significantly in the image plane.

        This forces atoms to "spread out" and find different anatomical parts
        (legs, neck, body) rather than all stacking on the same bright region.

        Args:
            atoms_transformed: (B, K, 1, H, W) - Transformed atoms in image plane
            attention: (B, K) - Attention weights

        Returns:
            loss: Scalar - Overlap penalty (minimize)
        """
        B, K, _, _, _ = atoms_transformed.shape

        if K <= 1:
            return torch.tensor(0.0, device=atoms_transformed.device)

        # Flatten spatial dimensions
        atoms_flat = atoms_transformed.view(B, K, -1)  # (B, K, H*W)

        # Normalize atoms as spatial distributions
        atoms_norm = atoms_flat / (atoms_flat.sum(dim=-1, keepdim=True) + 1e-8)

        # Compute pairwise overlaps using dot product (intersection)
        # overlap[i,j] = sum(atoms_norm[i] * atoms_norm[j])
        overlap = torch.bmm(atoms_norm, atoms_norm.transpose(1, 2))  # (B, K, K)

        # Mask out diagonal (self-overlap)
        mask = 1 - torch.eye(K, device=atoms_transformed.device).unsqueeze(0)  # (1, K, K)
        overlap = overlap * mask

        # Weight by attention - only penalize overlaps between active atoms
        attention_product = attention.unsqueeze(2) * attention.unsqueeze(1)  # (B, K, K)
        weighted_overlap = overlap * attention_product

        # Average across all pairs
        return weighted_overlap.sum() / (B * K * (K - 1))

    def atom_peak_penalty(self, atoms):
        """
        Penalize atoms that are too peaked/concentrated.

        Encourages atoms to distribute their mass more evenly rather than
        having a few very bright pixels. This helps prevent collapse to
        dot-like structures and encourages more spread-out anatomical parts.

        Args:
            atoms: (B, K, 1, H, W) - Atom templates

        Returns:
            loss: Scalar - Peak penalty (minimize)
        """
        B, K, _, _, _ = atoms.shape

        # Flatten spatial dimensions
        atoms_flat = atoms.view(B * K, -1)  # (B*K, H*W)

        # Get max value for each atom
        max_vals = atoms_flat.max(dim=-1)[0]  # (B*K,)

        # Get mean value for each atom (excluding zeros)
        atoms_abs = torch.abs(atoms_flat)
        mean_vals = atoms_abs.sum(dim=-1) / (atoms_abs > 0).sum(dim=-1).clamp(min=1)

        # Penalize large ratio between max and mean (peaked distributions)
        # A flat distribution has ratio ~1, peaked has ratio >> 1
        peak_ratio = max_vals / (mean_vals + 1e-8)

        # Penalize ratios above 2.0 (allow some variation but not extreme peaks)
        penalty = F.relu(peak_ratio - 2.0).mean()

        return penalty

    def forward(self, h_original, h_composed, atoms, poses, attention, v_local=None, atoms_transformed=None):
        """
        Compute total loss with improved structural priors.

        Args:
            h_original: (B, 1, H, W) - Input saliency
            h_composed: (B, 1, H, W) - Composed pattern
            atoms: (B, K, 1, H_atom, W_atom) - Atoms
            poses: (B, K, 5) - SE(2) parameters
            attention: (B, K) - Attention weights
            v_local: (B, K, 2, v_H, v_W) - Local velocity fields (optional)
            atoms_transformed: (B, K, 1, H, W) - Transformed atoms in image plane (optional)

        Returns:
            loss_dict: Dictionary of loss components
        """
        loss_recon = self.reconstruction_loss(h_original, h_composed, signal_weight=self.signal_weight)
        loss_atom_sparse = self.atom_sparsity_loss(atoms)
        loss_atom_div = self.atom_diversity_loss(atoms)
        loss_pose_reg = self.pose_regularization_loss(poses)
        loss_attention_sparse = self.attention_sparsity_loss(attention)
        loss_local_smooth = self.local_smoothness_loss(v_local)
        loss_mass_cons = self.mass_conservation_loss(h_original, h_composed)
        loss_atom_tv = self.atom_total_variation_loss(atoms)
        loss_atom_compact = self.atom_compactness_loss(atoms)
        loss_usage_balance = self.atom_usage_balance_loss(attention)
        loss_peak_penalty = self.atom_peak_penalty(atoms)

        # Compute overlap penalty if transformed atoms provided
        if atoms_transformed is not None:
            loss_overlap = self.atom_overlap_penalty(atoms_transformed, attention)
        else:
            loss_overlap = torch.tensor(0.0, device=h_original.device)

        total_loss = (
            self.lambda_reconstruction * loss_recon +
            self.lambda_atom_sparsity * loss_atom_sparse +
            self.lambda_atom_diversity * loss_atom_div +
            self.lambda_pose_reg * loss_pose_reg +
            self.lambda_attention_sparsity * loss_attention_sparse +
            self.lambda_local_smooth * loss_local_smooth +
            self.lambda_mass_conservation * loss_mass_cons +
            self.lambda_atom_tv * loss_atom_tv +
            self.lambda_atom_compactness * loss_atom_compact +
            self.lambda_atom_usage_balance * loss_usage_balance +
            self.lambda_atom_overlap * loss_overlap +
            self.lambda_atom_peak_penalty * loss_peak_penalty
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
            'atom_compactness': loss_atom_compact,
            'usage_balance': loss_usage_balance,
            'atom_overlap': loss_overlap,
            'atom_peak_penalty': loss_peak_penalty
        }


# ==========================================
# Trainer
# ==========================================

class RotoLDDMMTrainer:
    """
    Trainer for Roto-LDDMM V2.3 with 4-stage Gaussian annealing (anti-collapse).

    Training Stages:
    ----------------
    1. Warmup (1-5):       Atoms frozen, predictor learns with σ=2.5
    2. Discovery (6-20):   σ=2.0, atoms move to signal regions, high LR
    3. Refinement (21-40): σ→1.0 (slower drop), local warps enabled, moderate LR
    4. Finalize (41-50):   σ→0.5 (maintain minimum blur for diffuse signal), low LR

    V2.3 Features:
    - Weighted reconstruction: 10× weight for positive signal vs background
    - Peak penalty: Prevents atoms from collapsing to sharp dots
    - No compactness penalty: Allows atoms to spread naturally
    """

    def __init__(self, model, train_loader, val_loader=None,
                 lr=1e-3, device='cuda', checkpoint_dir='./checkpoints_roto_v2',
                 total_epochs=50):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.total_epochs = total_epochs

        # Gaussian blur module
        self.gaussian_blur = GaussianBlur(kernel_size=9).to(device)

        # Define 4-stage schedule
        self.stage_boundaries = {
            'warmup': (1, 5),
            'discovery': (6, 20),
            'refinement': (21, 40),
            'finalize': (41, total_epochs)
        }

        # Optimizer (separate learning rates for atoms and predictor)
        self.optimizer = optim.Adam([
            {'params': self.model.atom_bank.parameters(), 'lr': lr * 0.3},  # Slower for atoms
            {'params': self.model.predictor.parameters(), 'lr': lr},
        ], weight_decay=1e-5)

        # Manual LR scheduling (no auto scheduler)
        self.base_lr = lr

        # V2.2 Loss: STRUCTURE OVER SMOOTHNESS
        # Philosophy: Let reconstruction signal dominate, allow atoms to stretch/fragment if needed
        self.criterion = RotoLDDMMLoss(
            lambda_reconstruction=15.0,        # CRITICAL: 15× increase - force model to match dots!
            lambda_atom_sparsity=3.0,          # Moderate: allow atoms to capture regions
            lambda_atom_diversity=2.0,         # Moderate: force different atoms
            lambda_pose_reg=0.01,              # Low: flexible poses
            lambda_attention_sparsity=0.5,     # REDUCED: allow 6-8 atoms to collaborate (was 4.0)
            lambda_local_smooth=0.8,           # Keep warps smooth but allow organic curves
            lambda_mass_conservation=1.0,      # Standard
            lambda_atom_tv=0.1,                # INCREASED: encourage smoother, more spread-out atoms (was 0.01)
            lambda_atom_compactness=0.0,       # REMOVED: allow atoms to spread naturally (was 0.005)
            lambda_atom_usage_balance=2.0,     # Prevent mode collapse
            lambda_atom_overlap=0.1,           # REDUCED: neck can be near head (was 1.0)
            lambda_atom_peak_penalty=0.5,      # NEW: prevent overly peaked atoms (encourage distribution)
            signal_weight=10.0                 # NEW: 10× weight for positive signal regions in reconstruction
        ).to(device)

        self.history = defaultdict(list)

        print(f"\n{'='*80}")
        print(f"Roto-LDDMM V2.3 Trainer: ANTI-COLLAPSE WITH WEIGHTED RECONSTRUCTION")
        print(f"{'='*80}")
        print(f"\n4-Stage Training Schedule (UPDATED ANNEALING):")
        print(f"  Stage 1 - Warmup     (Epochs 1-5):   σ=2.5, Atoms Frozen")
        print(f"  Stage 2 - Discovery  (Epochs 6-20):  σ=2.0, High LR")
        print(f"  Stage 3 - Refinement (Epochs 21-40): σ→1.0 (slower drop, was →0.5)")
        print(f"  Stage 4 - Finalize   (Epochs 41-{total_epochs}): σ→0.5 (maintain minimum blur)")
        print(f"\nLoss Weights (V2.3 - ANTI-COLLAPSE UPDATE):")
        print(f"  - Reconstruction: 15.0 ★★★ CRITICAL - 15× increase to force dot matching!")
        print(f"    └─ Signal Weight: 10.0 (prioritize positive signal over background zeros)")
        print(f"  - Atom TV: 0.1 (encourage smooth/spread atoms, was 0.01)")
        print(f"  - Atom Compactness: 0.0 (DISABLED - allow atoms to spread naturally)")
        print(f"  - Atom Peak Penalty: 0.5 (NEW - prevent peaked/collapsed atoms)")
        print(f"  - Attention Sparsity: 0.5 (allow 6-8 atoms to collaborate)")
        print(f"  - Overlap Penalty: 0.1 (minimal - neck can be near head)")
        print(f"  - Atom Sparsity: 3.0 (moderate)")
        print(f"  - Atom Diversity: 2.0 (force different atoms)")
        print(f"  - Local Smooth: 0.8 (allow organic curves)")
        print(f"  - Usage Balance: 2.0 (prevent collapse)")
        print(f"\nPhilosophy Evolution:")
        print(f"  V2.1: Group dots into smooth, compact parts → Result: Blobs, not birds")
        print(f"  V2.2: Let reconstruction dominate, atoms stretch to match → Result: Still collapsed")
        print(f"  V2.3: Weighted reconstruction + anti-collapse penalties → Goal: Spread atoms, match signal")
        print(f"\nKey Features:")
        print(f"  - Weighted reconstruction: 10× priority for signal regions over background")
        print(f"  - Peak penalty: Forces atoms to distribute mass evenly (not concentrated)")
        print(f"  - Minimum blur (σ≥0.5): Maintains gradient signal for diffuse regions")
        print(f"  - Anisotropic scaling (sx, sy) for elongated structures")
        print(f"  - No compactness penalty: Atoms can spread naturally")
        print(f"  - Multiple atoms collaborate (6-8 instead of 3-4) to build complex shapes")
        print(f"{'='*80}\n")

    def get_stage_config(self, epoch):
        """
        Get training configuration for current epoch.

        Returns:
            dict with: stage_name, sigma, freeze_atoms, lr_multiplier
        """
        if self.stage_boundaries['warmup'][0] <= epoch <= self.stage_boundaries['warmup'][1]:
            return {
                'stage_name': 'Warmup',
                'sigma': 2.5,
                'freeze_atoms': True,
                'lr_multiplier': 1.0
            }
        elif self.stage_boundaries['discovery'][0] <= epoch <= self.stage_boundaries['discovery'][1]:
            return {
                'stage_name': 'Discovery',
                'sigma': 2.0,
                'freeze_atoms': False,
                'lr_multiplier': 1.0  # High LR
            }
        elif self.stage_boundaries['refinement'][0] <= epoch <= self.stage_boundaries['refinement'][1]:
            # UPDATED: Slower annealing - σ from 2.0 to 1.0 (was 2.0 → 0.5)
            # Keeps "scaffolding" large enough for atoms to stretch into necks/legs
            start_epoch, end_epoch = self.stage_boundaries['refinement']
            progress = (epoch - start_epoch) / (end_epoch - start_epoch)
            sigma = 2.0 - progress * 1.0  # 2.0 → 1.0 (slower drop)
            return {
                'stage_name': 'Refinement',
                'sigma': sigma,
                'freeze_atoms': False,
                'lr_multiplier': 0.5  # Moderate LR
            }
        else:  # Finalize
            # UPDATED: Final drop from 1.0 to 0.5 (maintain minimum blur for diffuse signal)
            # This prevents atoms from collapsing to only match sharp peaks
            start_epoch, end_epoch = self.stage_boundaries['finalize']
            progress = (epoch - start_epoch) / (end_epoch - start_epoch)
            sigma = 1.0 - progress * 0.5  # 1.0 → 0.5 (keep minimum blur)
            return {
                'stage_name': 'Finalize',
                'sigma': sigma,
                'freeze_atoms': False,
                'lr_multiplier': 0.2  # Low LR
            }

    def train_epoch(self, epoch):
        """Train for one epoch with Gaussian annealing."""
        self.model.train()

        # Get stage configuration
        config = self.get_stage_config(epoch)
        stage_name = config['stage_name']
        sigma = config['sigma']
        freeze_atoms = config['freeze_atoms']
        lr_mult = config['lr_multiplier']

        # Adjust learning rates
        for param_group in self.optimizer.param_groups:
            if 'atom_bank' in str(param_group['params'][0]):
                param_group['lr'] = self.base_lr * 0.3 * lr_mult
            else:
                param_group['lr'] = self.base_lr * lr_mult

        # Freeze/unfreeze atoms
        for param in self.model.atom_bank.parameters():
            param.requires_grad = not freeze_atoms

        print(f"\n  [{stage_name}] σ={sigma:.2f}, LR×{lr_mult:.1f}, Atoms {'Frozen' if freeze_atoms else 'Active'}")

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

            # Apply Gaussian blur (V2: annealing strategy)
            # Early training: blur creates "hills" for gradient flow
            # Later training: blur → 0 to learn exact dot patterns
            saliency_maps_blurred = self.gaussian_blur(saliency_maps, sigma=sigma)

            # Forward pass (use blurred input)
            h_composed, poses, attention, atoms_transformed = self.model(
                saliency_maps_blurred,
                class_ids=labels,
                update_atoms=True
            )

            # Get atoms (for loss computation)
            atoms = self.model.atom_bank(labels)

            # Get v_local from predictor (also use blurred input for consistency)
            _, _, v_local = self.model.predictor(saliency_maps_blurred)

            # Compute loss against BLURRED target
            # As σ→0, blurred→original, so final loss is against exact dots
            losses = self.criterion(
                saliency_maps_blurred, h_composed, atoms, poses, attention, v_local,
                atoms_transformed=atoms_transformed
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
                'recon★': losses['reconstruction'].item(),  # Primary signal
                'tv': losses['atom_tv'].item(),
                'peak': losses['atom_peak_penalty'].item()
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
        """Train for multiple epochs with 4-stage annealing."""
        print(f"\n{'='*80}")
        print(f"Starting Roto-LDDMM V2.1 Training (Organic Dot-Patterns)")
        print(f"{'='*80}")

        for epoch in range(1, num_epochs + 1):
            avg_losses = self.train_epoch(epoch)

            # Get stage info for display
            config = self.get_stage_config(epoch)

            print(f"\nEpoch {epoch}/{num_epochs} Summary [{config['stage_name']}], σ={config['sigma']:.2f}:")
            print(f"  Total Loss: {avg_losses['total']:.4f}")
            print(f"  ★ Reconstruction: {avg_losses['reconstruction']:.4f} ← PRIMARY SIGNAL (×15.0)")
            print(f"  Atom TV: {avg_losses['atom_tv']:.6f} (×0.1 - encourage smooth/spread atoms)")
            print(f"  Atom Compactness: {avg_losses['atom_compactness']:.6f} (×0.0 - DISABLED)")
            print(f"  Atom Peak Penalty: {avg_losses['atom_peak_penalty']:.4f} (×0.5 - prevent collapse)")
            print(f"  Atom Overlap: {avg_losses['atom_overlap']:.4f} (×0.1 - allow proximity)")
            print(f"  Attention Sparsity: {avg_losses['attention_sparsity']:.4f} (×0.5 - encourage collaboration)")
            print(f"  Usage Balance: {avg_losses['usage_balance']:.6f} (uniform atom usage)")
            print(f"  Atom Diversity: {avg_losses['atom_diversity']:.4f} (different atoms)")
            print(f"  Mass Conservation: {avg_losses['mass_conservation']:.4f}")

            # Stage transition alerts
            if epoch == self.stage_boundaries['discovery'][0]:
                print(f"\n  *** STAGE TRANSITION: Warmup → Discovery (atoms now active) ***")
            elif epoch == self.stage_boundaries['refinement'][0]:
                print(f"\n  *** STAGE TRANSITION: Discovery → Refinement (blur annealing starts) ***")
            elif epoch == self.stage_boundaries['finalize'][0]:
                print(f"\n  *** STAGE TRANSITION: Refinement → Finalize (final push to exact dots) ***")

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
        description='Roto-LDDMM V2.2: STRUCTURE OVER SMOOTHNESS',
        epilog="""
V2.2 PHILOSOPHY: Let reconstruction signal dominate to create skeletal structures

KEY CHANGES (V2.1 → V2.2):
  - Reconstruction weight: 1.0 → 15.0 (15× increase!)
  - TV loss: 0.5 → 0.01 (allow anatomical detail)
  - Compactness: 0.3 → 0.005 (allow elongated atoms - necks, legs)
  - Attention sparsity: 4.0 → 0.5 (allow 6-8 atoms to collaborate)
  - Gaussian annealing: Keep σ ≥ 1.0 until epoch 40 (persistent scaffolding)

ANISOTROPIC SCALING:
  - Atoms can stretch independently: sx, sy ∈ [0.2, 2.0]
  - 56×56 squares become 20×120 rectangles for necks/legs

TRAINING SCHEDULE (UPDATED):
  Epochs 1-5:   Warmup (atoms frozen, σ=2.5)
  Epochs 6-20:  Discovery (atoms active, σ=2.0, high LR)
  Epochs 21-40: Refinement (σ→1.0, slower annealing, was →0.5)
  Epochs 41-50: Finalize (σ→0, from 1.0, was from 0.5)

EXPECTED RESULTS:
  - Atoms: Elongated smudges/lines (not blobs)
  - Usage: 6-8 atoms active (not just 3-4)
  - Output: Skeletal bird structure with overlapping parts

USAGE:
  python geo_roto_v2.py --data_dir ./data/ig_maps \\
                        --num_classes 1000 --k_atoms 15 --epochs 50
        """
    )
    parser.add_argument('--data_dir', type=str,
                       default='./data/saliency_imagenet_sub_c20_s100',
                       help='Directory with saliency maps')
    parser.add_argument('--num_classes', type=int, default=20,
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
    parser.add_argument('--composition_mode', type=str, default='max',
                       choices=['softmax', 'max', 'sum'],
                       help='Atom composition mode (V2 default: max for dot preservation)')
    parser.add_argument('--no_local_warp', action='store_true',
                       help='Disable local diffeomorphic refinement')
    parser.add_argument('--shared_atoms', action='store_true',
                       help='Use shared atoms across all classes (saves memory)')
    parser.add_argument('--no_seed_init', action='store_true',
                       help='Disable smart seeded initialization (use random instead)')
    parser.add_argument('--seed_samples', type=int, default=3,
                       help='Number of samples per class for seeded initialization (default: 3)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_roto_v2',
                       help='Checkpoint directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"Roto-LDDMM V2.2: STRUCTURE OVER SMOOTHNESS")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"\n★ Key Philosophy: Reconstruction signal (×15.0) DOMINATES smoothness priors")

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

    print(f"\nV2.2 Training Strategy:")
    print(f"  4-Stage Gaussian Annealing (UPDATED):")
    print(f"    Stage 1 (Epochs 1-5):    Warmup - atoms frozen, σ=2.5")
    print(f"    Stage 2 (Epochs 6-20):   Discovery - σ=2.0, high LR")
    print(f"    Stage 3 (Epochs 21-40):  Refinement - σ→1.0 (slower, was →0.5)")
    print(f"    Stage 4 (Epochs 41-{args.epochs}): Finalize - σ→0 (from 1.0, was from 0.5)")
    print(f"  Loss optimization: Reconstruction (×15.0) DOMINATES, minimal smoothness")
    print(f"  Expected: Elongated atoms, 6-8 atoms active, skeletal structure")

    # Load dataset
    print(f"\nLoading saliency maps from {args.data_dir}...")
    dataset = SaliencyMapDataset(
        data_dir=args.data_dir,
        max_samples_per_class=None,
        load_images=False,
        max_samples=100000
    )

    # Sample seed maps for smart atom initialization (V2 innovation)
    if not args.no_seed_init:
        seed_maps = sample_seed_maps(
            dataset,
            num_classes=args.num_classes,
            samples_per_class=args.seed_samples
        )
    else:
        print(f"\n[WARNING] Seeded initialization DISABLED - using random initialization")
        print(f"          This may lead to slower convergence!\n")
        seed_maps = None

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        collate_fn=safe_collate_fn,
        persistent_workers=True
    )

    # Create model with seeded atoms
    print(f"\nInitializing Roto-LDDMM V2 model...")
    model = RotoLDDMM_Pipeline(
        num_classes=args.num_classes,
        k_atoms=args.k_atoms,
        img_res=(224, 224),
        atom_res=(args.atom_res, args.atom_res),
        device=device,
        use_local_warp=not args.no_local_warp,
        composition_mode=args.composition_mode,
        shared_atoms=args.shared_atoms,
        seed_maps=seed_maps  # V2: Smart initialization!
    )

    # Create trainer
    trainer = RotoLDDMMTrainer(
        model=model,
        train_loader=train_loader,
        lr=args.lr,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        total_epochs=args.epochs  # V2: pass total epochs for stage scheduling
    )

    # Train
    trainer.train(num_epochs=args.epochs, save_frequency=5)

    print(f"\n✓ All done!")


if __name__ == "__main__":
    main()
