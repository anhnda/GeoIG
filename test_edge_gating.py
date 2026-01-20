#!/usr/bin/env python3
"""
Quick test to verify EdgeAwareGating doesn't drop channel dimensions.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from geo_v2 import EdgeAwareGating


def test_edge_gating():
    """Test that EdgeAwareGating preserves channel dimension."""
    print("Testing EdgeAwareGating...")

    # Create module
    gating = EdgeAwareGating()

    # Test with normal inputs
    batch_size = 32
    image = torch.randn(batch_size, 3, 224, 224)
    saliency = torch.randn(batch_size, 1, 224, 224)

    print(f"Input shapes:")
    print(f"  image: {image.shape}")
    print(f"  saliency: {saliency.shape}")

    # Forward pass
    try:
        gated_saliency, edge_mask = gating(image, saliency)

        print(f"\nOutput shapes:")
        print(f"  gated_saliency: {gated_saliency.shape}")
        print(f"  edge_mask: {edge_mask.shape}")

        # Verify shapes
        assert gated_saliency.shape == (batch_size, 1, 224, 224), \
            f"Wrong gated_saliency shape: {gated_saliency.shape}"
        assert edge_mask.shape == (batch_size, 1, 224, 224), \
            f"Wrong edge_mask shape: {edge_mask.shape}"

        print("\n✅ EdgeAwareGating test PASSED - channel dimension preserved!")
        return True

    except Exception as e:
        print(f"\n❌ EdgeAwareGating test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_edge_gating()
    sys.exit(0 if success else 1)
