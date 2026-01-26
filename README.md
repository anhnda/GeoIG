# GeoIG - Subset Version for Fast Prototyping

This is a subset version of the Geodesic Pattern Learning pipeline, optimized for fast prototyping and experimentation with smaller datasets.

## Overview

The subset version works with a reduced number of ImageNet classes (default: 20 classes) instead of the full 1000 classes, making it ideal for:
- Quick experimentation and testing
- Development and debugging
- Limited computational resources
- Proof-of-concept demonstrations

## Three-Step Workflow

### Step 1: Generate Saliency Maps (Subset)

Export saliency maps for the first N classes:

```bash
python saliency_clean_export_sub.py
```

**Default Configuration:**
- Classes: First 20 classes (0-19)
- Samples per class: 100 (randomly sampled)
- Output: `./data/saliency_imagenet_sub_c20_s100/`

**What this does:**
- Loads ResNet50 pretrained on ImageNet
- Randomly samples images from the first N classes
- Computes Integrated Gradients saliency maps
- Exports saliency maps + RGB images (for edge-aware gating)
- Saves metadata and batch files

**Customization:**
```bash
# Custom subset: 50 classes, 50 samples each
python saliency_clean_export_sub.py --num_classes 50 --samples_per_class 50

# With Expected Gradients (5 baselines, slower but more accurate)
python saliency_clean_export_sub.py --num_classes 20 --use_mean
```

**Expected Output:**
```
✅ Loaded 1847 samples (153 corrupted)
  Saliency: 184.0 MB (float16)
  RGB: 292.6 MB (uint8)
  Total RAM: 476.6 MB

Output: ./data/saliency_imagenet_sub_c20_s100
```

---

### Step 2: Train Geodesic Pattern Model (Subset)

Train the LDDMM model to learn sub-patterns:

```bash
python geo_s2_sub.py \
    --data_dir ./data/saliency_imagenet_sub_c20_s100 \
    --num_classes 20 \
    --use_ot \
    --no_edge_gating \
    --k_subpatterns 15
```

**Key Parameters:**
- `--data_dir`: Path to saliency data from Step 1
- `--num_classes`: Must match the number exported in Step 1 (default: 20)
- `--k_subpatterns`: Number of sub-patterns per class (default: 10, recommended: 10-15)
- `--use_ot`: Use Optimal Transport for template updates (recommended)
- `--no_edge_gating`: Disable edge-aware gating (faster, uses less memory)
- `--epochs`: Training epochs (default: 50)

**What this does:**
- Multi-scale coarse-to-fine alignment (14×14 → 112×112)
- Learns K sub-patterns (templates) per class
- Uses diffeomorphic warping (LDDMM)
- Sinkhorn-Knopp optimal transport for template updates
- Saves checkpoints every 5 epochs

**Expected Training Time:**
- **With GPU (RTX 3080)**: ~5-10 minutes for 50 epochs (20 classes)
- **With CPU**: ~1-2 hours

**Output Files:**
```
checkpoints_sub/
├── advanced_lddmm_model_epoch_5.pth
├── advanced_lddmm_model_epoch_10.pth
├── ...
├── advanced_lddmm_model_final.pth
└── training_curves_sub.png
```

**Training Configuration:**
```bash
# Faster training (fewer epochs, no OT)
python geo_s2_sub.py \
    --data_dir ./data/saliency_imagenet_sub_c20_s100 \
    --num_classes 20 \
    --epochs 20 \
    --k_subpatterns 10

# More patterns per class
python geo_s2_sub.py \
    --data_dir ./data/saliency_imagenet_sub_c20_s100 \
    --num_classes 20 \
    --k_subpatterns 20 \
    --use_ot

# Larger subset (50 classes)
python geo_s2_sub.py \
    --data_dir ./data/saliency_imagenet_sub_c50_s50 \
    --num_classes 50 \
    --k_subpatterns 15 \
    --use_ot
```

---

### Step 3: Visualize Learned Patterns

Visualize the learned sub-patterns and decompositions:

```bash
python visualize_patterns_sub.py \
    --checkpoint checkpoints_sub/advanced_lddmm_model_final.pth \
    --class_id 9
```

**What this does:**
- Loads the trained model
- Generates three visualizations:
  1. **Class Patterns**: All K sub-patterns for the specified class
  2. **Sample Decomposition**: How a single sample is aligned and assigned to patterns
  3. **Sample Comparison**: Multiple samples from the same class

**Output Files:**
```
visualizations_sub/
├── class_9_patterns_sub.png              # All K sub-patterns
├── class_9_decomposition_sample_0_sub.png # Detailed decomposition
└── class_9_comparison_sub.png             # Multiple sample comparison
```

**Customization:**
```bash
# Visualize different class
python visualize_patterns_sub.py \
    --checkpoint checkpoints_sub/advanced_lddmm_model_final.pth \
    --class_id 5

# Visualize specific sample
python visualize_patterns_sub.py \
    --checkpoint checkpoints_sub/advanced_lddmm_model_final.pth \
    --class_id 9 \
    --sample_idx 3

# Compare more samples
python visualize_patterns_sub.py \
    --checkpoint checkpoints_sub/advanced_lddmm_model_final.pth \
    --class_id 9 \
    --num_samples 10

# Use intermediate checkpoint
python visualize_patterns_sub.py \
    --checkpoint checkpoints_sub/advanced_lddmm_model_epoch_20.pth \
    --class_id 9
```

---

## Complete Example Workflow

```bash
# 1. Generate saliency maps (20 classes, 100 samples each)
python saliency_clean_export_sub.py \
    --num_classes 20 \
    --samples_per_class 100

# 2. Train model (50 epochs, 15 sub-patterns per class)
python geo_s2_sub.py \
    --data_dir ./data/saliency_imagenet_sub_c20_s100 \
    --num_classes 20 \
    --k_subpatterns 15 \
    --use_ot \
    --no_edge_gating \
    --epochs 50

# 3. Visualize multiple classes
for class_id in {0..19}; do
    python visualize_patterns_sub.py \
        --checkpoint checkpoints_sub/advanced_lddmm_model_final.pth \
        --class_id $class_id \
        --output_dir visualizations_sub/class_$class_id
done
```

---

## Requirements

### Python Dependencies
```bash
pip install torch torchvision numpy scipy matplotlib kornia joblib tqdm pandas pillow
```

### Hardware Requirements
- **Minimum**: 8GB RAM, any GPU with 4GB VRAM
- **Recommended**: 16GB RAM, GPU with 8GB+ VRAM (RTX 3080, A100, etc.)
- **CPU-only**: Supported but much slower

### Data Requirements
- ImageNet raw data: `/data/imagenet_raw/data` (Parquet format)
- Or modify paths in `saliency_clean_export_sub.py`

---

## Key Differences from Full Version

| Feature | Full Version | Subset Version |
|---------|-------------|----------------|
| **Classes** | 1000 | 20 (default) |
| **Samples** | 50k-100k | 1k-2k |
| **Training Time** | 4-8 hours | 5-10 minutes |
| **Memory Usage** | 30-50GB | 5-10GB |
| **Checkpoint Dir** | `checkpoints_v2/` | `checkpoints_sub/` |
| **Output Dir** | `visualizations/` | `visualizations_sub/` |

---

## Understanding the Output

### 1. Class Patterns Visualization
Shows all K learned sub-patterns for a class:
- Each pattern is a learned template
- Usage percentage shows how often each pattern is used
- Patterns should look distinct and meaningful

### 2. Sample Decomposition
Shows detailed analysis of a single sample:
- **Original Image**: Input RGB image
- **Original IG Map**: Raw saliency map
- **Aligned IG Map**: After diffeomorphic warping
- **Deformation Magnitude**: How much warping was applied
- **Deformation Field**: Vector field of warping
- **Cluster Assignment**: Probability distribution over K patterns
- **Top-3 Patterns**: Most likely patterns for this sample
- **Difference Maps**: Residual error for each pattern
- **Reconstruction Error**: Overall alignment quality

### 3. Sample Comparison
Shows how different samples from the same class are assigned to different patterns:
- Helps understand pattern diversity
- Shows robustness of pattern learning

---

## Troubleshooting

### Issue: `IndexError: index 14 is out of bounds`
**Solution**: This was fixed in the latest version. Make sure you're using the updated `visualize_patterns_sub.py`.

### Issue: `FileNotFoundError: Metadata not found`
**Solution**: Run Step 1 (saliency export) first before Step 2 (training).

### Issue: Out of memory during training
**Solutions:**
- Use `--no_edge_gating` to disable edge-aware gating
- Reduce batch size: `--batch_size 32`
- Reduce number of sub-patterns: `--k_subpatterns 10`
- Disable optimal transport: Remove `--use_ot` flag

### Issue: Training is too slow
**Solutions:**
- Reduce epochs: `--epochs 20`
- Use smaller subset: `--num_classes 10 --samples_per_class 50`
- Disable optimal transport (faster): Remove `--use_ot` flag

### Issue: Patterns look random/noisy
**Solutions:**
- Train for more epochs: `--epochs 100`
- Use optimal transport: `--use_ot`
- Increase samples per class in Step 1
- Try different hyperparameters (learning rate, loss weights)

---

## Advanced Configuration

### Custom Loss Weights
Edit `geo_s2_sub.py` line ~1131-1143 to adjust loss weights:
```python
self.criterion = AdvancedLDDMMLoss(
    lambda_smooth=0.05,              # Smoothness (lower = sharper patterns)
    lambda_template_sparsity=7.0,    # Sparsity (higher = sparser patterns)
    lambda_jacobian=75.0,            # Prevent tearing (keep high)
    lambda_mass_conservation=5.0,    # Mass conservation (keep high)
    # ... other parameters
)
```

### Multi-GPU Training
```bash
CUDA_VISIBLE_DEVICES=0,1 python geo_s2_sub.py \
    --data_dir ./data/saliency_imagenet_sub_c20_s100 \
    --num_classes 20 \
    --batch_size 128  # Larger batch size for multi-GPU
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{geodesic_ig_patterns,
  title={Geodesic Pattern Learning for Interpretable Saliency Maps},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

---

## License

MIT License

---

## Related Files

- **Full Version**: `geo_s2.py`, `visualize_patterns.py`
- **Subset Version**: `geo_s2_sub.py`, `visualize_patterns_sub.py`, `saliency_clean_export_sub.py`
- **Utilities**: `full_classes.py` (ImageNet class names)

---

## FAQ

**Q: How many classes should I use for prototyping?**
A: Start with 10-20 classes. This is enough to validate your approach while keeping training fast.

**Q: How many sub-patterns (K) should I use?**
A: Start with K=10-15. More patterns capture more diversity but take longer to train.

**Q: Should I use Optimal Transport (`--use_ot`)?**
A: Yes, if you have enough memory. OT produces cleaner, more interpretable patterns.

**Q: Should I use edge-aware gating (`--use_edge_gating`)?**
A: Only if you want to focus on object boundaries. It uses more memory and is slower.

**Q: Can I use this with other models (not ResNet50)?**
A: Yes! Modify `saliency_clean_export_sub.py` to use a different model. The rest of the pipeline is model-agnostic.

**Q: How do I scale to the full 1000 classes?**
A: Use the full version: `geo_s2.py` and `visualize_patterns.py`. Expect 4-8 hours of training time.

---

## Next Steps

1. **Experiment with different hyperparameters**: Try different K values, loss weights, and architectures
2. **Analyze pattern semantics**: Do patterns correspond to meaningful object parts?
3. **Compare with baselines**: How do learned patterns compare to raw IG maps?
4. **Scale to full ImageNet**: Use `geo_s2.py` for 1000 classes
5. **Apply to other domains**: Medical imaging, object detection, etc.

---

For questions or issues, please open a GitHub issue or contact the authors.
