"""
Pattern Predictability Verification for Geodesic LDDMM Model

This script verifies that learned patterns are predictive using Pattern-Injection:
1. Loading the top patterns for a specific class
2. Filtering out zero-usage patterns (0%)
3. Taking random images from DIFFERENT classes
4. Injecting/overlaying the target class pattern onto these images
5. Checking if the classifier's probability for the target class increases significantly

This tests whether the patterns capture discriminative features that can shift
predictions toward the target class.

Usage:
    # Verify patterns for class 0 (tabby cat)
    python verify_pattern_predictable.py --checkpoint checkpoints/lddmm_model_final.pth --class_id 0

    # Verify with custom number of test images
    python verify_pattern_predictable.py --checkpoint checkpoints/lddmm_model_final.pth \
                                          --class_id 281 \
                                          --num_test_images 50 \
                                          --data_dir ./data/saliency_imagenet1k_resnet50_100
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse
from torchvision import models, transforms
import joblib
import random

# Import model components
import sys
sys.path.append('.')
from geo_patterns import LDDMM_GlobalPatternPipeline
from geo_x import AdvancedLDDMM_Pipeline
from full_classes import IMAGENET2012_CLASSES
from saliency_map_export import ImageNet1kSaliencyDataset, IMAGENET_RAW_DIR, IMAGENET_SAMPLED_DIR


class PatternPredictor:
    """Verify pattern predictability using pattern injection on real images."""

    def __init__(self, checkpoint_path, data_dir, device='cuda', images_per_class=100):
        self.device = device
        self.data_dir = Path(data_dir)

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

        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        # Transform for loading ImageNet images
        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        # Normalization transform for classifier
        self.normalize = transforms.Normalize(mean=self.mean.tolist(), std=self.std.tolist())

        # Load ImageNet dataset for accessing original images
        print(f"\nLoading ImageNet dataset for test images...")
        try:
            self.imagenet_dataset = ImageNet1kSaliencyDataset(
                raw_dir=IMAGENET_RAW_DIR,
                sampled_dir=IMAGENET_SAMPLED_DIR,
                output_dir=self.data_dir,
                images_per_class=images_per_class,
                force_resample=False
            )
            print(f"✓ ImageNet dataset loaded with {len(self.imagenet_dataset)} images")
        except Exception as e:
            raise RuntimeError(f"Could not load ImageNet dataset: {e}")

        # Build class index for fast sample lookup
        self.saliency_dir = self.data_dir / "saliency_maps"
        self.cache_file = self.data_dir / "class_index_cache.pkl"
        print(f"\nLoading class index for sample lookup...")
        self.class_index = self._load_index_cache()
        print(f"✓ Class index ready!")

        # Get class names
        self.class_names = {idx: name for idx, name in enumerate(IMAGENET2012_CLASSES.values())}

    def _load_index_cache(self):
        """Load the class index from cache file."""
        if not self.cache_file.exists():
            raise RuntimeError(f"Class index cache not found at {self.cache_file}. "
                             f"Please run visualize_patterns.py first to build the cache.")
        print(f"  Loading index from cache...")
        class_index = joblib.load(self.cache_file)
        total_samples = sum(len(samples) for samples in class_index.values())
        print(f"  ✓ Cache loaded: {len(class_index)} classes, {total_samples} samples")
        return class_index

    def get_similar_classes(self, target_class_id, num_similar=50):
        """
        Get similar classes to the target class based on WordNet hierarchy.

        For stronger verification, we want to test on "adjacent" classes
        (e.g., other birds if target is ostrich) rather than random classes.

        Args:
            target_class_id: Target class
            num_similar: Number of similar classes to return

        Returns:
            List of class IDs that are semantically similar
        """
        # Simple heuristic: Get classes with nearby IDs (often semantically related in ImageNet)
        # and some random ones for diversity
        nearby_classes = []

        # Get nearby classes (within ±100 IDs, often related in ImageNet structure)
        for offset in range(1, 101):
            for delta in [offset, -offset]:
                candidate = target_class_id + delta
                if 0 <= candidate < self.num_classes and candidate != target_class_id:
                    nearby_classes.append(candidate)
                    if len(nearby_classes) >= num_similar // 2:
                        break
            if len(nearby_classes) >= num_similar // 2:
                break

        # Add some random classes for diversity
        all_other_classes = [c for c in range(self.num_classes) if c != target_class_id]
        random_classes = random.sample(all_other_classes, min(num_similar // 2, len(all_other_classes)))

        similar_classes = nearby_classes + random_classes
        return similar_classes[:num_similar]

    def get_random_images_from_other_classes(self, target_class_id, num_images=20, use_similar_classes=False):
        """
        Get random images from classes OTHER than the target class.

        Args:
            target_class_id: Class to exclude
            num_images: Number of random images to retrieve
            use_similar_classes: If True, sample from semantically similar classes only

        Returns:
            List of (image_tensor, original_class_id) tuples
        """
        # Get class pool
        if use_similar_classes:
            other_class_ids = self.get_similar_classes(target_class_id)
            print(f"  Using {len(other_class_ids)} similar/adjacent classes for harder test")
        else:
            other_class_ids = [c for c in range(self.num_classes) if c != target_class_id]
            print(f"  Using all {len(other_class_ids)} other classes (random sampling)")

        # Sample random images
        sampled_images = []
        attempts = 0
        max_attempts = num_images * 10

        while len(sampled_images) < num_images and attempts < max_attempts:
            attempts += 1

            # Pick a random class
            random_class = random.choice(other_class_ids)

            # Get samples for this class
            if random_class not in self.class_index or not self.class_index[random_class]:
                continue

            # Pick a random sample from this class
            batch_file, item_idx = random.choice(self.class_index[random_class])

            try:
                # Load the batch
                batch_data = joblib.load(batch_file)
                item = batch_data[item_idx]

                # Get the sample index
                sample_idx = item.get('sample_index', item.get('index'))

                # Load image from dataset
                image_pil, _ = self.imagenet_dataset[sample_idx]
                img_tensor = self.image_transform(image_pil)  # (3, 224, 224)

                sampled_images.append((img_tensor, random_class))

            except Exception as e:
                print(f"Warning: Could not load sample: {e}")
                continue

        return sampled_images

    def inject_pattern_into_image(self, image, pattern, alpha=0.3, method='additive'):
        """
        Inject a pattern into an image.

        Args:
            image: (3, H, W) image tensor in [0, 1]
            pattern: (1, H, W) pattern tensor
            alpha: Strength of injection (0-1)
            method: 'additive' or 'multiplicative'

        Returns:
            (3, H, W) modified image tensor
        """
        # Normalize pattern to [0, 1]
        pattern_np = pattern.squeeze().cpu().numpy()
        pattern_min = pattern_np.min()
        pattern_max = pattern_np.max()

        if pattern_max > pattern_min:
            pattern_norm = (pattern_np - pattern_min) / (pattern_max - pattern_min)
        else:
            pattern_norm = np.ones_like(pattern_np) * 0.5

        # Convert to tensor and expand to 3 channels
        pattern_tensor = torch.from_numpy(pattern_norm).float()
        pattern_3ch = pattern_tensor.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)

        image_np = image.numpy()

        if method == 'additive':
            # Add pattern to image (weighted)
            injected = image_np + alpha * pattern_3ch.numpy()
            injected = np.clip(injected, 0, 1)
        elif method == 'multiplicative':
            # Multiply image by pattern (enhances where pattern is strong)
            injected = image_np * (1 + alpha * pattern_3ch.numpy())
            injected = np.clip(injected, 0, 1)
        else:
            raise ValueError(f"Unknown injection method: {method}")

        return torch.from_numpy(injected).float()

    def verify_class_patterns(self, class_id, num_test_images=20, min_usage_threshold=0.0,
                            alpha=0.3, injection_method='additive', use_similar_classes=False):
        """
        Verify that patterns for a class are predictive using pattern injection.

        Takes random images from OTHER classes, injects the target class patterns,
        and checks if the classifier's probability for the target class increases.

        Args:
            class_id: Class index to verify
            num_test_images: Number of random test images from other classes
            min_usage_threshold: Minimum usage percentage to include (0.0 = exclude zero patterns)
            alpha: Pattern injection strength (0-1)
            injection_method: 'additive' or 'multiplicative'
            use_similar_classes: If True, test on semantically similar classes (harder test)

        Returns:
            dict: Results including probability changes and statistics
        """
        class_name = self.class_names.get(class_id, f"Class {class_id}")
        print(f"\n{'='*80}")
        print(f"Verifying patterns for: {class_name} (Class {class_id})")
        print(f"Pattern Injection Method: {injection_method} (alpha={alpha})")
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
        print(f"  Filtered out (zero usage): {self.k_subpatterns - len(valid_patterns)}")

        if not valid_patterns:
            print(f"\n⚠ No patterns found with usage > {min_usage_threshold}%")
            return {
                'class_id': class_id,
                'class_name': class_name,
                'total_patterns': self.k_subpatterns,
                'tested_patterns': 0,
                'successful_injections': 0,
                'pattern_results': []
            }

        # Get random test images from other classes
        print(f"\n{'='*80}")
        print(f"Sampling {num_test_images} random images from OTHER classes...")
        print(f"{'='*80}")

        test_images = self.get_random_images_from_other_classes(
            class_id, num_test_images, use_similar_classes=use_similar_classes
        )
        print(f"✓ Sampled {len(test_images)} test images")

        if not test_images:
            print(f"\n⚠ Could not load test images")
            return None

        # Test each valid pattern
        print(f"\n{'='*80}")
        print(f"Testing Pattern Injection...")
        print(f"{'='*80}")

        pattern_results = []

        with torch.no_grad():
            for pattern_info in valid_patterns:
                pattern_id = pattern_info['pattern_id']
                pattern = pattern_info['template']

                print(f"\nPattern {pattern_id} (usage: {pattern_info['usage_percent']:.1f}%):")

                prob_increases = []
                rank_improvements = []
                log_odds_increases = []
                prediction_flips = 0  # Count when target becomes top-1

                for _, (img_tensor, orig_class) in enumerate(test_images):
                    # Get baseline prediction (without pattern)
                    img_normalized = self.normalize(img_tensor).unsqueeze(0).to(self.device)
                    baseline_logits = self.classifier(img_normalized)
                    baseline_probs = F.softmax(baseline_logits, dim=1)[0]
                    baseline_prob = baseline_probs[class_id].item()

                    # Get baseline rank of target class
                    sorted_indices = torch.argsort(baseline_probs, descending=True)
                    baseline_rank = (sorted_indices == class_id).nonzero(as_tuple=True)[0].item()
                    baseline_top1 = sorted_indices[0].item()

                    # Inject pattern into image
                    injected_img = self.inject_pattern_into_image(
                        img_tensor, pattern, alpha=alpha, method=injection_method
                    )

                    # Get prediction with pattern
                    injected_normalized = self.normalize(injected_img).unsqueeze(0).to(self.device)
                    injected_logits = self.classifier(injected_normalized)
                    injected_probs = F.softmax(injected_logits, dim=1)[0]
                    injected_prob = injected_probs[class_id].item()

                    # Get new rank of target class
                    sorted_indices_inj = torch.argsort(injected_probs, descending=True)
                    injected_rank = (sorted_indices_inj == class_id).nonzero(as_tuple=True)[0].item()
                    injected_top1 = sorted_indices_inj[0].item()

                    # Calculate changes
                    prob_increase = injected_prob - baseline_prob
                    rank_improvement = baseline_rank - injected_rank  # Positive = better rank

                    # Calculate log-odds change (more sensitive to tail probability changes)
                    # log-odds = ln(p / (1-p))
                    eps = 1e-10  # Avoid log(0)
                    baseline_log_odds = np.log((baseline_prob + eps) / (1 - baseline_prob + eps))
                    injected_log_odds = np.log((injected_prob + eps) / (1 - injected_prob + eps))
                    log_odds_increase = injected_log_odds - baseline_log_odds

                    # Check if prediction flipped to target class
                    if injected_top1 == class_id and baseline_top1 != class_id:
                        prediction_flips += 1

                    prob_increases.append(prob_increase)
                    rank_improvements.append(rank_improvement)
                    log_odds_increases.append(log_odds_increase)

                # Calculate statistics for this pattern
                avg_prob_increase = np.mean(prob_increases)
                median_prob_increase = np.median(prob_increases)
                positive_increases = sum(1 for x in prob_increases if x > 0)
                avg_rank_improvement = np.mean(rank_improvements)
                avg_log_odds_increase = np.mean(log_odds_increases)
                median_log_odds_increase = np.median(log_odds_increases)

                result = {
                    'pattern_id': pattern_id,
                    'usage_percent': pattern_info['usage_percent'],
                    'avg_prob_increase': avg_prob_increase,
                    'median_prob_increase': median_prob_increase,
                    'positive_increases': positive_increases,
                    'success_rate': positive_increases / len(test_images),
                    'avg_rank_improvement': avg_rank_improvement,
                    'avg_log_odds_increase': avg_log_odds_increase,
                    'median_log_odds_increase': median_log_odds_increase,
                    'prediction_flips': prediction_flips,
                    'flip_rate': prediction_flips / len(test_images),
                    'prob_increases': prob_increases,
                    'rank_improvements': rank_improvements,
                    'log_odds_increases': log_odds_increases
                }

                pattern_results.append(result)

                # Print result
                success_rate = positive_increases / len(test_images) * 100
                flip_rate = prediction_flips / len(test_images) * 100
                status = "✓ EFFECTIVE" if success_rate > 50 else "~ WEAK" if success_rate > 25 else "✗ INEFFECTIVE"
                print(f"  {status}")
                print(f"  Avg probability increase: {avg_prob_increase:+.6f} ({avg_prob_increase*100:+.4f}%)")
                print(f"  Median probability increase: {median_prob_increase:+.6f} ({median_prob_increase*100:+.4f}%)")
                print(f"  Avg log-odds increase: {avg_log_odds_increase:+.4f}")
                print(f"  Median log-odds increase: {median_log_odds_increase:+.4f}")
                print(f"  Success rate: {positive_increases}/{len(test_images)} ({success_rate:.1f}%)")
                print(f"  Avg rank improvement: {avg_rank_improvement:+.1f}")
                print(f"  Prediction flips (→ Top-1): {prediction_flips}/{len(test_images)} ({flip_rate:.1f}%)")

        # Calculate overall statistics
        successful_patterns = sum(1 for r in pattern_results if r['success_rate'] > 0.5)
        avg_success_rate = np.mean([r['success_rate'] for r in pattern_results])
        avg_log_odds = np.mean([r['avg_log_odds_increase'] for r in pattern_results])
        total_flips = sum(r['prediction_flips'] for r in pattern_results)
        avg_flip_rate = np.mean([r['flip_rate'] for r in pattern_results])

        print(f"\n{'='*80}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"Class: {class_name} (Class {class_id})")
        print(f"Injection strength (alpha): {alpha}")
        print(f"Test regime: {'Similar/Adjacent classes (harder)' if use_similar_classes else 'Random classes (easier)'}")
        print(f"\nPattern Statistics:")
        print(f"  Total patterns tested: {len(valid_patterns)}")
        print(f"  Effective patterns (>50% success): {successful_patterns}")
        print(f"  Average success rate: {avg_success_rate*100:.2f}%")
        print(f"  Average log-odds increase: {avg_log_odds:+.4f}")
        print(f"\nPrediction Flips (→ Top-1):")
        print(f"  Total flips across all patterns: {total_flips}")
        print(f"  Average flip rate: {avg_flip_rate*100:.2f}%")
        print(f"\nConclusion: {'✓ Patterns are PREDICTIVE' if successful_patterns > 0 else '✗ Patterns are NOT predictive'}")

        if successful_patterns > 0 and avg_flip_rate > 0.1:
            print(f"✓✓ Patterns can FLIP predictions to target class!")
        elif successful_patterns > 0:
            print(f"✓ Patterns shift probabilities but rarely flip predictions (try higher alpha)")

        return {
            'class_id': class_id,
            'class_name': class_name,
            'total_patterns': self.k_subpatterns,
            'tested_patterns': len(valid_patterns),
            'successful_injections': successful_patterns,
            'avg_success_rate': avg_success_rate,
            'avg_log_odds_increase': avg_log_odds,
            'total_prediction_flips': total_flips,
            'avg_flip_rate': avg_flip_rate,
            'injection_params': {
                'alpha': alpha,
                'method': injection_method,
                'num_test_images': len(test_images),
                'use_similar_classes': use_similar_classes
            },
            'pattern_results': pattern_results
        }


def main():
    parser = argparse.ArgumentParser(
        description='Verify pattern predictability using pattern injection'
    )
    parser.add_argument('--checkpoint', type=str, default='checkpoints/lddmm_model_final.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str,
                       default='./data/saliency_imagenet1k_resnet50_100',
                       help='Directory with saliency maps')
    parser.add_argument('--class_id', type=int, required=True,
                       help='Class ID to verify patterns for')
    parser.add_argument('--num_test_images', type=int, default=20,
                       help='Number of random test images from other classes')
    parser.add_argument('--min_usage', type=float, default=0.0,
                       help='Minimum usage percentage to include (default: 0.0 = exclude zero patterns)')
    parser.add_argument('--alpha', type=float, default=0.3,
                       help='Pattern injection strength (0-1, default: 0.3)')
    parser.add_argument('--injection_method', type=str, default='additive',
                       choices=['additive', 'multiplicative'],
                       help='Pattern injection method (default: additive)')
    parser.add_argument('--use_similar_classes', action='store_true',
                       help='Test on semantically similar/adjacent classes (harder test)')
    parser.add_argument('--images_per_class', type=int, default=100,
                       help='Number of images per class (must match saliency generation)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--output', type=str, default=None,
                       help='Optional output file to save results (JSON format)')

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Create predictor
    print(f"\n{'='*80}")
    print(f"Pattern Predictability Verification (Pattern Injection)")
    print(f"{'='*80}")

    predictor = PatternPredictor(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        device=args.device,
        images_per_class=args.images_per_class
    )

    # Verify patterns
    results = predictor.verify_class_patterns(
        class_id=args.class_id,
        num_test_images=args.num_test_images,
        min_usage_threshold=args.min_usage,
        alpha=args.alpha,
        injection_method=args.injection_method,
        use_similar_classes=args.use_similar_classes
    )

    # Save results if requested
    if args.output and results:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(exist_ok=True, parents=True)

        # Convert numpy arrays and types to native Python types for JSON serialization
        def convert_to_native(obj):
            """Recursively convert numpy types to native Python types."""
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return [convert_to_native(item) for item in obj.tolist()]
            else:
                return obj

        results_json = convert_to_native(results)

        with open(output_path, 'w') as f:
            json.dump(results_json, f, indent=2)

        print(f"\n✓ Results saved to {output_path}")

    print(f"\n{'='*80}")
    print(f"Verification Complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
