"""
Roto-LDDMM Utilities
====================

Reusable components for Roto-LDDMM training:
- Data loading and preprocessing
- Geometric transformations (SE(2), diffeomorphisms)
- Gaussian blur with annealing
- Smart atom initialization (greedy seeding)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import joblib
from collections import defaultdict
import gc
from torch.utils.data import Dataset


# ==========================================
# Gaussian Blur with Annealing
# ==========================================

class GaussianBlur(nn.Module):
    """
    Apply Gaussian blur with dynamic sigma scheduling.

    This is crucial for dot-point IG maps: blur creates "gradient hills"
    that guide atoms to the right regions, then annealing sharpens the signal.
    """

    def __init__(self, kernel_size=9):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def _create_gaussian_kernel(self, sigma, channels=1):
        """Create 2D Gaussian kernel."""
        # Create 1D Gaussian
        k = self.kernel_size
        coords = torch.arange(k, dtype=torch.float32) - k // 2
        gauss_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gauss_1d = gauss_1d / gauss_1d.sum()

        # Create 2D kernel via outer product
        gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]

        # Expand for conv2d: (out_channels, in_channels/groups, H, W)
        kernel = gauss_2d.view(1, 1, k, k).repeat(channels, 1, 1, 1)

        return kernel

    def forward(self, x, sigma):
        """
        Apply Gaussian blur with given sigma.

        Args:
            x: (B, C, H, W) - Input tensor
            sigma: float - Gaussian standard deviation (0 = no blur)

        Returns:
            blurred: (B, C, H, W) - Blurred tensor
        """
        if sigma < 0.1:
            # Skip blur for very small sigma
            return x

        _, C, _, _ = x.shape  # B, H, W not needed

        # Create kernel
        kernel = self._create_gaussian_kernel(sigma, channels=C).to(x.device)

        # Apply depthwise convolution (separate filter per channel)
        blurred = F.conv2d(x, kernel, padding=self.padding, groups=C)

        return blurred


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
# Smart Initialization: Greedy Part Discovery
# ==========================================

def get_part_seeds(saliency_map, k_atoms=15, patch_size=56):
    """
    Greedy algorithm to find anatomical parts by sequential peak detection.

    This is MUCH better than random initialization because:
    1. First peak → body (highest intensity region)
    2. Mask body → second peak → head/neck
    3. Mask head → third peak → leg1
    4. Continue until all K parts are found

    Args:
        saliency_map: (H, W) or (1, H, W) - Single saliency map
        k_atoms: int - Number of parts to find
        patch_size: int - Size of patches to extract (default 56)

    Returns:
        seeds: List of (y, x) tuples - Centers of discovered parts
        patches: (K, 1, patch_size, patch_size) - Extracted patches
    """
    # Ensure 2D
    if len(saliency_map.shape) == 3:
        saliency_map = saliency_map[0]  # (1, H, W) → (H, W)

    H, W = saliency_map.shape
    map_residual = saliency_map.clone()
    seeds = []
    patches = []

    half_patch = patch_size // 2

    for _ in range(k_atoms):
        # Find the coordinate of the maximum remaining intensity
        idx = torch.argmax(map_residual)
        y, x = divmod(idx.item(), W)

        # Save the patch center
        seeds.append((y, x))

        # Extract patch (with boundary handling)
        y1, y2 = max(0, y - half_patch), min(H, y + half_patch)
        x1, x2 = max(0, x - half_patch), min(W, x + half_patch)

        patch = torch.zeros(1, patch_size, patch_size, device=saliency_map.device)
        patch_h = y2 - y1
        patch_w = x2 - x1

        # Center the extracted region in the patch
        offset_y = (patch_size - patch_h) // 2
        offset_x = (patch_size - patch_w) // 2

        patch[0, offset_y:offset_y + patch_h, offset_x:offset_x + patch_w] = \
            saliency_map[y1:y2, x1:x2]

        patches.append(patch)

        # Mask out a circular region so we don't pick the same part twice
        # Use slightly larger mask than patch to ensure separation
        mask_radius = int(half_patch * 1.2)
        y1_mask = max(0, y - mask_radius)
        y2_mask = min(H, y + mask_radius)
        x1_mask = max(0, x - mask_radius)
        x2_mask = min(W, x + mask_radius)

        map_residual[y1_mask:y2_mask, x1_mask:x2_mask] = 0

    patches = torch.stack(patches, dim=0)  # (K, 1, patch_size, patch_size)

    return seeds, patches


def sample_seed_maps(dataset, num_classes=1000, samples_per_class=1):
    """
    Sample representative saliency maps for atom initialization.

    Args:
        dataset: SaliencyMapDataset - The loaded dataset
        num_classes: int - Number of classes
        samples_per_class: int - How many samples to average per class

    Returns:
        seed_maps: dict - {class_id: (1, H, W) tensor} averaged saliency map per class
    """
    print(f"\n{'='*80}")
    print(f"Sampling Seed Maps for Smart Atom Initialization")
    print(f"{'='*80}")

    class_samples = defaultdict(list)

    # Collect samples
    print(f"Collecting {samples_per_class} sample(s) per class...")
    for idx in tqdm(range(len(dataset)), desc="Scanning dataset"):
        saliency, label, _ = dataset[idx]

        if len(class_samples[label]) < samples_per_class:
            class_samples[label].append(saliency)

        # Stop early if we have enough samples for all classes
        if len(class_samples) >= num_classes:
            if all(len(samples) >= samples_per_class for samples in class_samples.values()):
                break

    # Average samples per class
    seed_maps = {}
    for class_id, samples in class_samples.items():
        if samples:
            # Stack and average
            stacked = torch.stack(samples, dim=0)  # (N, 1, H, W)
            avg_map = stacked.mean(dim=0)  # (1, H, W)
            seed_maps[class_id] = avg_map

    print(f"Collected seeds for {len(seed_maps)} classes")
    print(f"{'='*80}\n")

    return seed_maps


# ==========================================
# Dataset
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
