"""
Pattern Predictability Verification for Geodesic LDDMM Model

This script verifies that learned patterns are predictive by:
1. Loading the top patterns for a specific class
2. Filtering out zero-usage patterns (0%)
3. Converting patterns back to image space
4. Feeding them to the model to verify correct predictions
5. Outputting statistics on prediction accuracy

Usage:
    # Verify patterns for class 0
    python verify_pattern_predictable.py --checkpoint checkpoints/lddmm_model_final.pth --class_id 0

    # Verify specific class with custom data
    python verify_pattern_predictable.py --checkpoint checkpoints/lddmm_model_epoch_10.pth \
                                          --class_id 281 \
                                          --data_dir ./data/saliency_imagenet1k_resnet50_100
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse
from torchvision import models, transforms
from PIL import Image

# Import model components
import sys
sys.path.append('.')
from geo_patterns import LDDMM_GlobalPatternPipeline
from geo_x import AdvancedLDDMM_Pipeline
from full_classes import IMAGENET2012_CLASSES


class PatternPredictor:
    """Verify pattern predictability by feeding them back to a classifier."""

    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device

        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract configuration
        self.num_classes = checkpoint['templates'].shape[0]
        self.k_subpatterns = checkpoint['templates'].shape[1]

        # Auto-detect velocity field resolution and model type
        v_head_keys = [k for k in checkpoint['model_state_dict'].keys()
                       if 'predictor.fine_v_head' in k and 'weight' in k and 'Linear' not in k]

        if not v_head_keys:
            v_head_keys = [k for k in checkpoint['model_state_dict'].keys()
                           if 'predictor.v_head' in k and 'weight' in k and 'Linear' not in k]

        if not v_head_keys:
            v_head_keys = [k for k in checkpoint['model_state_dict'].keys()
                           if 'predictor.coarse_v_head' in k and 'weight' in k and 'Linear' not in k]

        if not v_head_keys:
            raise ValueError("Could not find velocity head in checkpoint.")

        v_head_keys.sort()
        v_head_key = v_head_keys[-1]

        v_head_weight_shape = checkpoint['model_state_dict'][v_head_key].shape
        v_dim = v_head_weight_shape[0]
        v_res_size = int(np.sqrt(v_dim / 2))
        v_res = (v_res_size, v_res_size)

        # Detect model type
        has_fine_v_head = any('fine_v_head' in k for k in checkpoint['model_state_dict'].keys())
        has_coarse_v_head = any('coarse_v_head' in k for k in checkpoint['model_state_dict'].keys())
        is_advanced_model = has_fine_v_head or has_coarse_v_head

        print(f"Loaded model:")
        print(f"  Type: {'Advanced (Multi-Scale)' if is_advanced_model else 'Simple (Single-Scale)'}")
        print(f"  Classes: {self.num_classes}")
        print(f"  Sub-patterns per class: {self.k_subpatterns}")
        print(f"  Velocity field resolution: {v_res}")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")

        # Create model
        self.is_advanced_model = is_advanced_model
        if is_advanced_model:
            self.model = AdvancedLDDMM_Pipeline(
                num_classes=self.num_classes,
                k_subpatterns=self.k_subpatterns,
                img_res=(224, 224),
                device=device,
                use_ot=True,
                use_edge_gating=False
            ).to(device)
        else:
            self.model = LDDMM_GlobalPatternPipeline(
                num_classes=self.num_classes,
                k_subpatterns=self.k_subpatterns,
                img_res=(224, 224),
                device=device,
                v_res=v_res
            ).to(device)

        # Load weights (handle torch.compile() prefix)
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            print("  Detected torch.compile() checkpoint - stripping _orig_mod. prefix...")
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict)
        self.model.eval()

        print(f"✓ Model loaded successfully!")

        # Load pre-trained ResNet50 classifier for verification
        print(f"\nLoading ResNet50 classifier for verification...")
        self.classifier = models.resnet50(pretrained=True).to(device)
        self.classifier.eval()
        print(f"✓ ResNet50 loaded successfully!")

        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Get class names
        self.class_names = {idx: name for idx, (wnid, name) in enumerate(IMAGENET2012_CLASSES.items())}

    def pattern_to_image(self, pattern):
        """
        Convert a pattern (saliency map) back to image space.

        This is a heuristic approach:
        1. Normalize the pattern to [0, 1]
        2. Convert from single channel to RGB by replicating
        3. Apply smoothing to create more natural images

        Args:
            pattern: (1, H, W) pattern tensor

        Returns:
            (3, H, W) image tensor in [0, 1] range
        """
        # Ensure pattern is on CPU for processing
        pattern_cpu = pattern.squeeze().cpu()

        # Normalize to [0, 1] range
        pattern_min = pattern_cpu.min()
        pattern_max = pattern_cpu.max()

        if pattern_max > pattern_min:
            pattern_norm = (pattern_cpu - pattern_min) / (pattern_max - pattern_min)
        else:
            pattern_norm = pattern_cpu

        # Apply smoothing to make it more image-like
        pattern_tensor = pattern_norm.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        # Gaussian blur for smoothing
        kernel_size = 5
        sigma = 1.0
        kernel = self._gaussian_kernel(kernel_size, sigma).to(pattern_tensor.device)
        pattern_smooth = F.conv2d(pattern_tensor, kernel, padding=kernel_size//2)

        # Convert to RGB by replicating the channel
        pattern_rgb = pattern_smooth.repeat(1, 3, 1, 1)  # (1, 3, H, W)

        return pattern_rgb.squeeze(0)  # (3, H, W)

    def _gaussian_kernel(self, kernel_size, sigma):
        """Create a 2D Gaussian kernel for smoothing."""
        x = torch.arange(kernel_size).float() - kernel_size // 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)

    def verify_class_patterns(self, class_id, min_usage_threshold=0.0):
        """
        Verify that patterns for a class can be correctly predicted.

        Args:
            class_id: Class index to verify
            min_usage_threshold: Minimum usage percentage to include (0.0 = exclude zero patterns)

        Returns:
            dict: Results including correct predictions and statistics
        """
        class_name = self.class_names.get(class_id, f"Class {class_id}")
        print(f"\n{'='*80}")
        print(f"Verifying patterns for: {class_name} (Class {class_id})")
        print(f"{'='*80}")

        # Get templates and counts for this class
        templates = self.model.templates[class_id]  # (K, 1, H, W)
        counts = self.model.template_counts[class_id].cpu().numpy()  # (K,)

        # Calculate usage percentages
        total_count = counts.sum()
        usage_percentages = (counts / total_count * 100) if total_count > 0 else np.zeros_like(counts)

        # Filter patterns by usage threshold
        valid_patterns = []
        for k in range(self.k_subpatterns):
            if usage_percentages[k] > min_usage_threshold:
                valid_patterns.append({
                    'pattern_id': k,
                    'usage_count': int(counts[k]),
                    'usage_percent': usage_percentages[k],
                    'template': templates[k]
                })

        print(f"\nPattern Statistics:")
        print(f"  Total patterns: {self.k_subpatterns}")
        print(f"  Patterns with usage > {min_usage_threshold}%: {len(valid_patterns)}")
        print(f"  Filtered out: {self.k_subpatterns - len(valid_patterns)}")

        if not valid_patterns:
            print(f"\n⚠ No patterns found with usage > {min_usage_threshold}%")
            return {
                'class_id': class_id,
                'class_name': class_name,
                'total_patterns': self.k_subpatterns,
                'tested_patterns': 0,
                'correct_predictions': 0,
                'accuracy': 0.0,
                'pattern_results': []
            }

        # Test each valid pattern
        print(f"\n{'='*80}")
        print(f"Testing patterns by feeding to classifier...")
        print(f"{'='*80}")

        pattern_results = []
        correct_count = 0

        with torch.no_grad():
            for i, pattern_info in enumerate(valid_patterns):
                pattern_id = pattern_info['pattern_id']
                pattern = pattern_info['template']

                # Convert pattern to image
                img = self.pattern_to_image(pattern).to(self.device)

                # Normalize for classifier
                img_normalized = self.normalize(img).unsqueeze(0)  # (1, 3, H, W)

                # Get prediction from classifier
                logits = self.classifier(img_normalized)
                pred_class = logits.argmax(dim=1).item()
                pred_prob = F.softmax(logits, dim=1)[0, class_id].item()
                top5_probs, top5_classes = torch.topk(F.softmax(logits, dim=1), k=5, dim=1)

                is_correct = (pred_class == class_id)
                is_in_top5 = class_id in top5_classes[0].cpu().numpy()

                if is_correct:
                    correct_count += 1

                result = {
                    'pattern_id': pattern_id,
                    'usage_percent': pattern_info['usage_percent'],
                    'predicted_class': pred_class,
                    'predicted_class_name': self.class_names.get(pred_class, f"Class {pred_class}"),
                    'target_class_prob': pred_prob,
                    'is_correct': is_correct,
                    'is_in_top5': is_in_top5,
                    'top5_classes': top5_classes[0].cpu().numpy().tolist(),
                    'top5_probs': top5_probs[0].cpu().numpy().tolist()
                }

                pattern_results.append(result)

                # Print result
                status = "✓ CORRECT" if is_correct else "✗ WRONG"
                top5_status = "(in Top-5)" if is_in_top5 and not is_correct else ""
                print(f"\nPattern {pattern_id} (usage: {pattern_info['usage_percent']:.1f}%):")
                print(f"  {status} {top5_status}")
                print(f"  Predicted: {result['predicted_class_name']} (class {pred_class})")
                print(f"  Target class probability: {pred_prob:.4f}")
                if not is_correct:
                    print(f"  Top prediction probability: {result['top5_probs'][0]:.4f}")

        # Calculate statistics
        accuracy = correct_count / len(valid_patterns) if valid_patterns else 0.0

        print(f"\n{'='*80}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"Class: {class_name} (Class {class_id})")
        print(f"Total patterns tested: {len(valid_patterns)}")
        print(f"Correct predictions: {correct_count}")
        print(f"Accuracy: {accuracy*100:.2f}%")

        # Top-5 accuracy
        top5_correct = sum(1 for r in pattern_results if r['is_in_top5'])
        top5_accuracy = top5_correct / len(valid_patterns) if valid_patterns else 0.0
        print(f"Top-5 Accuracy: {top5_accuracy*100:.2f}%")

        return {
            'class_id': class_id,
            'class_name': class_name,
            'total_patterns': self.k_subpatterns,
            'tested_patterns': len(valid_patterns),
            'correct_predictions': correct_count,
            'top5_predictions': top5_correct,
            'accuracy': accuracy,
            'top5_accuracy': top5_accuracy,
            'pattern_results': pattern_results
        }


def main():
    parser = argparse.ArgumentParser(
        description='Verify pattern predictability for LDDMM model'
    )
    parser.add_argument('--checkpoint', type=str, default='checkpoints/lddmm_model_final.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--class_id', type=int, required=True,
                       help='Class ID to verify patterns for')
    parser.add_argument('--min_usage', type=float, default=0.0,
                       help='Minimum usage percentage to include (default: 0.0 = exclude zero patterns)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--output', type=str, default=None,
                       help='Optional output file to save results (JSON format)')

    args = parser.parse_args()

    # Create predictor
    print(f"\n{'='*80}")
    print(f"Pattern Predictability Verification")
    print(f"{'='*80}")

    predictor = PatternPredictor(
        checkpoint_path=args.checkpoint,
        device=args.device
    )

    # Verify patterns
    results = predictor.verify_class_patterns(
        class_id=args.class_id,
        min_usage_threshold=args.min_usage
    )

    # Save results if requested
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(exist_ok=True, parents=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to {output_path}")

    print(f"\n{'='*80}")
    print(f"Verification Complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
