# Quick Validation Grid Guide

## TL;DR

Generate a visual comparison grid of model predictions vs ground truth:

```bash
python create_validation_grid.py \
    output/onion_model_defect_segmentation_oct_7/checkpoint_best_ema.pth \
    datasets/onion_defect_coco \
    validation_grid.png
```

## What it does

Creates a grid with 10 validation images (4 columns each):
1. **Original Image** - The raw input
2. **Ground Truth Mask** - What the model should predict (color-coded by class)
3. **Prediction Mask** - What the model actually predicted (color-coded by class)
4. **Overlay** - Predictions overlaid on the original image

**Plus:** A color legend at the top showing which color = which class!

## Quick Options

```bash
# More samples (20 instead of 10)
python create_validation_grid.py ... --num-samples=20

# Use CPU instead of GPU
python create_validation_grid.py ... --device=cpu

# Lower threshold to see more predictions
python create_validation_grid.py ... --conf-threshold=0.3

# Different random samples
python create_validation_grid.py ... --seed=123
```

## Full Documentation

See [VALIDATION_GRID_README.md](VALIDATION_GRID_README.md) for complete details.

## Output Example

The script generates a single PNG image with a grid layout:
```
┌─────────────────────────────────────────────────────────────┐
│  Original    Ground Truth    Prediction    Overlay          │
├─────────────────────────────────────────────────────────────┤
│  [Image 1]   [GT Mask 1]     [Pred 1]      [Overlay 1]      │
│  [Image 2]   [GT Mask 2]     [Pred 2]      [Overlay 2]      │
│  [Image 3]   [GT Mask 3]     [Pred 3]      [Overlay 3]      │
│  ...         ...             ...           ...              │
│  [Image 10]  [GT Mask 10]    [Pred 10]     [Overlay 10]     │
└─────────────────────────────────────────────────────────────┘
```

## Notes

- Only uses **positive examples** (images with annotations)
- **Color-coded masks** make it easy to see which classes are predicted correctly
- Each class gets a distinct color (legend shown at top)
- Random sampling ensures variety (use `--seed` for reproducibility)
- High resolution output (150 DPI) for detailed inspection
- Works with any COCO format dataset

