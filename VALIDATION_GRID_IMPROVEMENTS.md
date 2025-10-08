# Validation Grid Improvements

## What Changed

The `create_validation_grid.py` script has been enhanced with **color-coded class visualization**.

## Before (Grayscale Masks)

Previously, the script showed:
- Ground truth masks: Binary/grayscale (white = mask, black = background)
- Prediction masks: Binary/grayscale (white = mask, black = background)
- Overlay: Green overlay on original image

**Problem:** When you have multiple classes, you couldn't tell which class was which in the masks.

## After (Color-Coded Masks)

Now the script shows:
- Ground truth masks: **Color-coded by class** (each class has a unique color)
- Prediction masks: **Color-coded by class** (matching the ground truth colors)
- Overlay: **Color-coded overlay** on original image (50% transparency)
- **Color legend** at the top showing which color = which class

**Benefits:**
- ✅ Instantly see which classes are being predicted
- ✅ Easy visual comparison between ground truth and predictions
- ✅ Identify which specific classes the model struggles with
- ✅ Professional-looking visualizations with legend

## Example Usage

```bash
# Generate validation grid with color-coded masks
python create_validation_grid.py \
    output/onion_model_defect_segmentation_oct_7/checkpoint_best_ema.pth \
    datasets/onion_defect_coco \
    validation_grid.png

# Result: A grid showing 10 validation images with:
#   - Original images
#   - Color-coded ground truth masks
#   - Color-coded prediction masks
#   - Color-coded overlays
#   - Legend showing: mechanical_damage (blue), peeled_skin (orange)
```

## Technical Details

### Color Palette
- Uses matplotlib's `tab10` colormap for ≤10 classes
- Uses matplotlib's `tab20` colormap for 11-20 classes
- Each class gets a consistent, distinct color
- Background is always black (0, 0, 0)

### Functions Added
1. `generate_color_palette(num_classes)` - Creates distinct colors for each class
2. `create_colored_mask(class_mask, num_classes)` - Converts class IDs to RGB colors
3. Updated `get_ground_truth_mask()` - Now returns colored mask + class mask
4. Updated `get_prediction_mask()` - Now returns colored mask + class mask
5. Updated `create_overlay()` - Now handles colored masks with alpha blending

### Legend
- Automatically added at the top of the figure
- Shows class name with its assigned color
- Displays up to 5 classes per row
- Uses matplotlib patches for clear visualization

## Backward Compatibility

The script maintains the same command-line interface. All existing commands work as before, just with improved visualization.

## Testing

Tested with:
- 2-class dataset (onion_defect: mechanical_damage, peeled_skin)
- 854 positive validation examples
- Multiple sample sizes (2, 3, 10 samples)
- Both CUDA and CPU devices

All tests passed successfully.

