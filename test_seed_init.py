"""
Quick test of the greedy part seeding algorithm.

This demonstrates how the algorithm discovers anatomical parts sequentially.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

# Import from geo_roto_v2
import sys
from geo_roto_v2 import get_part_seeds


def create_test_saliency_map():
    """Create a synthetic saliency map with multiple peaks."""
    H, W = 224, 224
    sal_map = torch.zeros(1, H, W)

    # Peak 1: Body (large, centered, brightest)
    y1, x1 = 112, 112
    for i in range(-20, 21):
        for j in range(-20, 21):
            if 0 <= y1+i < H and 0 <= x1+j < W:
                dist = np.sqrt(i**2 + j**2)
                sal_map[0, y1+i, x1+j] = max(0, 1.0 - dist/20)

    # Peak 2: Head (smaller, upper left)
    y2, x2 = 60, 80
    for i in range(-10, 11):
        for j in range(-10, 11):
            if 0 <= y2+i < H and 0 <= x2+j < W:
                dist = np.sqrt(i**2 + j**2)
                sal_map[0, y2+i, x2+j] = max(0, 0.7 - dist/10)

    # Peak 3: Tail (medium, right)
    y3, x3 = 120, 180
    for i in range(-15, 16):
        for j in range(-8, 9):
            if 0 <= y3+i < H and 0 <= x3+j < W:
                dist = np.sqrt(i**2 + j**2)
                sal_map[0, y3+i, x3+j] = max(0, 0.5 - dist/12)

    # Peak 4: Leg 1 (small, lower left)
    y4, x4 = 160, 90
    for i in range(-8, 9):
        for j in range(-5, 6):
            if 0 <= y4+i < H and 0 <= x4+j < W:
                dist = np.sqrt(i**2 + j**2)
                sal_map[0, y4+i, x4+j] = max(0, 0.4 - dist/8)

    # Peak 5: Leg 2 (small, lower right)
    y5, x5 = 165, 135
    for i in range(-8, 9):
        for j in range(-5, 6):
            if 0 <= y5+i < H and 0 <= x5+j < W:
                dist = np.sqrt(i**2 + j**2)
                sal_map[0, y5+i, x5+j] = max(0, 0.35 - dist/8)

    return sal_map


def visualize_seed_discovery(saliency_map, k_atoms=5):
    """Visualize the greedy part discovery process."""
    seeds, patches = get_part_seeds(saliency_map, k_atoms=k_atoms, patch_size=56)

    # Create visualization
    fig, axes = plt.subplots(2, k_atoms + 1, figsize=(3*(k_atoms+1), 6))

    # Original map
    axes[0, 0].imshow(saliency_map[0].cpu().numpy(), cmap='hot')
    axes[0, 0].set_title('Original Map')
    axes[0, 0].axis('off')

    # Mark seeds on original
    axes[1, 0].imshow(saliency_map[0].cpu().numpy(), cmap='hot')
    for idx, (y, x) in enumerate(seeds):
        axes[1, 0].plot(x, y, 'b+', markersize=15, markeredgewidth=2)
        axes[1, 0].text(x+10, y+10, f'{idx+1}', color='cyan', fontsize=12, fontweight='bold')
    axes[1, 0].set_title(f'Discovered Seeds (n={k_atoms})')
    axes[1, 0].axis('off')

    # Show discovered patches
    for idx in range(k_atoms):
        y, x = seeds[idx]

        # Top: circle on map showing extraction region
        axes[0, idx+1].imshow(saliency_map[0].cpu().numpy(), cmap='hot', alpha=0.5)
        circle = plt.Circle((x, y), 28, color='cyan', fill=False, linewidth=2)
        axes[0, idx+1].add_patch(circle)
        axes[0, idx+1].plot(x, y, 'b+', markersize=12, markeredgewidth=2)
        axes[0, idx+1].set_xlim(max(0, x-60), min(224, x+60))
        axes[0, idx+1].set_ylim(min(224, y+60), max(0, y-60))
        axes[0, idx+1].set_title(f'Part {idx+1} @ ({y},{x})')
        axes[0, idx+1].axis('off')

        # Bottom: extracted patch
        axes[1, idx+1].imshow(patches[idx, 0].cpu().numpy(), cmap='hot')
        axes[1, idx+1].set_title(f'Patch {idx+1}\n(56x56)')
        axes[1, idx+1].axis('off')

    plt.tight_layout()
    plt.savefig('seed_discovery_demo.png', dpi=150)
    print(f"\n✓ Visualization saved to: seed_discovery_demo.png")
    print(f"\nDiscovered {k_atoms} parts in order:")
    for idx, (y, x) in enumerate(seeds):
        peak_value = saliency_map[0, y, x].item()
        print(f"  Part {idx+1}: Position ({y:3d}, {x:3d}), Peak intensity: {peak_value:.3f}")


if __name__ == "__main__":
    print("Testing Greedy Part Seeding Algorithm")
    print("=" * 60)

    # Create test map
    print("\n1. Creating synthetic saliency map with 5 peaks...")
    test_map = create_test_saliency_map()

    # Run seeding
    print("2. Running greedy peak detection...")
    visualize_seed_discovery(test_map, k_atoms=5)

    print("\n" + "=" * 60)
    print("Test complete! Check seed_discovery_demo.png")
