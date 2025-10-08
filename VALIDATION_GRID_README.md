# Validation Grid Visualization

This script creates a visual grid comparing model predictions with ground truth annotations on validation images.

## Overview

The `create_validation_grid.py` script generates a grid visualization with 4 columns:
1. **Original Image**: The input image
2. **Ground Truth Mask**: Color-coded mask from annotations (each class has a unique color)
3. **Prediction Mask**: Color-coded mask from model predictions (each class has a unique color)
4. **Overlay**: Original image with predicted masks overlaid (semi-transparent, color-coded)

A **color legend** is automatically added at the top of the grid showing which color corresponds to each class.

## Usage

### Basic Usage

```bash
python create_validation_grid.py <weights_path> <dataset_dir> <output_path>
```

### Example Commands

**With trained onion model:**
```bash
python create_validation_grid.py \
    output/onion_model_defect_segmentation_oct_7/checkpoint_best_ema.pth \
    datasets/onion_defect_coco \
    validation_grid.png
```

**With more samples (20 instead of 10):**
```bash
python create_validation_grid.py \
    output/onion_model_defect_segmentation_oct_7/checkpoint_best_ema.pth \
    datasets/onion_defect_coco \
    validation_grid.png \
    --num-samples=20
```

**Using CPU instead of GPU:**
```bash
python create_validation_grid.py \
    output/onion_model_defect_segmentation_oct_7/checkpoint_best_ema.pth \
    datasets/onion_defect_coco \
    validation_grid.png \
    --device=cpu
```

**Lower confidence threshold to see more predictions:**
```bash
python create_validation_grid.py \
    output/onion_model_defect_segmentation_oct_7/checkpoint_best_ema.pth \
    datasets/onion_defect_coco \
    validation_grid.png \
    --conf-threshold=0.3
```

## Arguments

### Required Arguments

- `<weights_path>`: Path to trained RF-DETR weights (.pth file)
- `<dataset_dir>`: Path to COCO dataset directory (should contain `valid/_annotations.coco.json`)
- `<output_path>`: Path to save the validation grid image (e.g., `validation_grid.png`)

### Optional Arguments

- `--num-samples=<n>`: Number of validation samples to display (default: 10)
- `--model-size=<size>`: Model size: nano, small, medium, or large (default: small)
- `--device=<device>`: Device to run inference on: cpu, cuda, or mps (default: cuda)
- `--conf-threshold=<th>`: Confidence threshold for predictions (default: 0.5)
- `--seed=<seed>`: Random seed for sample selection (default: 42)

## Output

The script generates a single image file containing:
- Multiple rows (one per validation sample, default 10)
- 4 columns per row (original, ground truth, prediction, overlay)
- **Color legend** at the top showing class colors
- High resolution (150 DPI) for detailed inspection
- Color-coded masks make it easy to identify which classes are predicted correctly

### Color Coding

- Each class is assigned a distinct color from matplotlib's color palette
- Ground truth and predictions use the same color scheme for easy comparison
- Background (no mask) is shown as black
- The overlay column shows predictions with 50% transparency over the original image

## Requirements

The script requires:
- Trained RF-DETR model weights
- COCO format dataset with validation split
- Python packages: `docopt`, `pycocotools`, `matplotlib`, `opencv-python`, `PIL`, `torch`, `numpy`

## Dataset Format

The dataset should follow the Roboflow/COCO structure:
```
dataset_dir/
├── class_names.txt (optional)
└── valid/
    ├── _annotations.coco.json
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## Notes

- The script only selects **positive examples** (images with at least one annotation)
- If fewer positive examples exist than requested, all available samples are used
- Masks are color-coded by class for easy visual comparison
- Each class gets a distinct color from matplotlib's color palette (tab10 for ≤10 classes, tab20 for more)
- Overlay uses 50% transparency for predictions
- Images are randomly sampled from the validation set (use `--seed` for reproducibility)
- The legend at the top shows which color corresponds to each class name

## Troubleshooting

### CUDA Out of Memory

If you encounter GPU memory issues, try:
```bash
# Use CPU instead
python create_validation_grid.py ... --device=cpu

# Or reduce the number of samples
python create_validation_grid.py ... --num-samples=5
```

### No Positive Examples Found

If the validation set has no annotated images, you'll see:
```
Found 0 positive examples in validation set
```

Make sure your dataset has annotations in the validation split.

### Model Not Loading

Ensure the weights file matches the model size:
```bash
# If using a different model size
python create_validation_grid.py ... --model-size=medium
```

