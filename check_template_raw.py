"""
Directly inspect raw template values to see what's actually stored.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

def check_template(checkpoint_path, class_id=9, pattern_id=8):
    """Check raw template values."""

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    template = checkpoint['templates'][class_id, pattern_id, 0].numpy()  # (224, 224)

    print(f"\n{'='*80}")
    print(f"RAW TEMPLATE INSPECTION - Class {class_id}, Pattern {pattern_id}")
    print(f"{'='*80}\n")

    # Basic stats
    print(f"Shape: {template.shape}")
    print(f"Min: {template.min():.6f}")
    print(f"Max: {template.max():.6f}")
    print(f"Mean: {template.mean():.6f}")
    print(f"Std: {template.std():.6f}")

    # Histogram of values
    print(f"\nValue Distribution:")
    print(f"  Exactly 0: {(template == 0).sum():6d} ({(template == 0).sum()/template.size*100:5.2f}%)")
    print(f"  (0, 0.01]: {((template > 0) & (template <= 0.01)).sum():6d} ({((template > 0) & (template <= 0.01)).sum()/template.size*100:5.2f}%)")
    print(f"  (0.01, 0.1]: {((template > 0.01) & (template <= 0.1)).sum():6d} ({((template > 0.01) & (template <= 0.1)).sum()/template.size*100:5.2f}%)")
    print(f"  (0.1, 0.5]: {((template > 0.1) & (template <= 0.5)).sum():6d} ({((template > 0.1) & (template <= 0.5)).sum()/template.size*100:5.2f}%)")
    print(f"  (0.5, 0.9]: {((template > 0.5) & (template <= 0.9)).sum():6d} ({((template > 0.5) & (template <= 0.9)).sum()/template.size*100:5.2f}%)")
    print(f"  (0.9, 1.0]: {(template > 0.9).sum():6d} ({(template > 0.9).sum()/template.size*100:5.2f}%)")

    # Percentiles
    print(f"\nPercentiles:")
    for p in [50, 75, 90, 95, 99, 99.9]:
        print(f"  {p:5.1f}th: {np.percentile(template, p):.6f}")

    # Non-zero values
    nonzero = template[template > 0]
    if len(nonzero) > 0:
        print(f"\nNon-zero values only:")
        print(f"  Count: {len(nonzero):6d} ({len(nonzero)/template.size*100:5.2f}%)")
        print(f"  Min: {nonzero.min():.6f}")
        print(f"  Max: {nonzero.max():.6f}")
        print(f"  Mean: {nonzero.mean():.6f}")
        print(f"  Median: {np.median(nonzero):.6f}")
        print(f"  75th percentile: {np.percentile(nonzero, 75):.6f}")
        print(f"  95th percentile: {np.percentile(nonzero, 95):.6f}")

    # Active values (> 0.01)
    active = template[template > 0.01]
    if len(active) > 0:
        print(f"\nActive values (>0.01) only:")
        print(f"  Count: {len(active):6d} ({len(active)/template.size*100:5.2f}%)")
        print(f"  Min: {active.min():.6f}")
        print(f"  Max: {active.max():.6f}")
        print(f"  Mean: {active.mean():.6f}")
        print(f"  Median: {np.median(active):.6f}")
        print(f"  75th percentile: {np.percentile(active, 75):.6f}")

    # Spatial analysis - where are the non-zero values?
    print(f"\nSpatial Distribution:")
    h, w = template.shape

    # Divide into regions
    regions = {
        'Top-left': template[:h//3, :w//3],
        'Top-center': template[:h//3, w//3:2*w//3],
        'Top-right': template[:h//3, 2*w//3:],
        'Middle-left': template[h//3:2*h//3, :w//3],
        'Center': template[h//3:2*h//3, w//3:2*w//3],
        'Middle-right': template[h//3:2*h//3, 2*w//3:],
        'Bottom-left': template[2*h//3:, :w//3],
        'Bottom-center': template[2*h//3:, w//3:2*w//3],
        'Bottom-right': template[2*h//3:, 2*w//3:],
        'Border (10px)': np.concatenate([
            template[:10, :].flatten(),
            template[-10:, :].flatten(),
            template[:, :10].flatten(),
            template[:, -10:].flatten()
        ]),
    }

    for region_name, region_data in regions.items():
        nonzero_count = (region_data > 0.01).sum()
        avg_val = region_data[region_data > 0.01].mean() if nonzero_count > 0 else 0
        print(f"  {region_name:20s}: {nonzero_count:5d} active pixels ({nonzero_count/region_data.size*100:5.2f}%), avg={avg_val:.4f}")

    # Create visualization with multiple vmaxes
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    vmaxes = [
        ('Max value', template.max()),
        ('99th percentile', np.percentile(template, 99)),
        ('95th percentile', np.percentile(template, 95)),
        ('75th %ile (nonzero)', np.percentile(active, 75) if len(active) > 0 else template.max()),
        ('Median (nonzero)', np.median(active) if len(active) > 0 else template.max()),
        ('Fixed 0.3', 0.3),
    ]

    for ax, (title, vmax) in zip(axes.flatten(), vmaxes):
        im = ax.imshow(template, cmap='hot', vmin=0, vmax=vmax, interpolation='nearest')
        ax.set_title(f'{title}\nvmax={vmax:.4f}', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(f'Pattern {pattern_id} - Class {class_id} (Ostrich)\nDifferent Normalizations',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = f'template_inspection_class{class_id}_pattern{pattern_id}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {output_path}")
    plt.show()

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--class_id', type=int, default=9)
    parser.add_argument('--pattern_id', type=int, default=8)

    args = parser.parse_args()
    check_template(args.checkpoint, args.class_id, args.pattern_id)
