# RF-DETR Inference - Quick Start Guide

## One-Minute Overview

The `inference.py` script now works with **both instance segmentation AND detection-only models**. It automatically detects which type you have and handles it appropriately.

## Basic Usage

```bash
# Works with both model types - auto-detects!
python inference.py images/ model.pth output/ --save-coco

# Test with just 10 images first
python inference.py images/ model.pth output/ --save-coco --sample-size=10
```

## Common Commands

### Save COCO Format (All Models)
```bash
python inference.py images/ model.pth output/ --save-coco
# Output: coco_annotations.json with all detections
```

### Visualize Detections (All Models)
```bash
python inference.py images/ model.pth output/ --visualize
# Detection model: shows boxes and labels
# Segmentation model: shows masks, boxes, and labels
```

### Save Masks (Segmentation Models Only)
```bash
python inference.py images/ seg_model.pth output/ --save-masks --save-instances
# Output: masks/ and instances/ directories
```

### Quick Test (10 Images)
```bash
python inference.py images/ model.pth output/ \
  --visualize --save-coco --sample-size=10
```

### Full Analysis (Segmentation)
```bash
python inference.py images/ seg_model.pth output/ \
  --save-coco --save-txt --save-masks --save-instances --visualize
```

## What Gets Created

### For Detection Models
- `coco_annotations.json` - Detections with bboxes and scores
- `labels/` - TXT files with bboxes (optional)
- `visualizations/` - PNG images with boxes (optional)

### For Segmentation Models
- `coco_annotations.json` - Detections with bboxes, scores, AND masks
- `labels/` - TXT files with mask polygons (optional)
- `instances/` - Individual mask images (optional)
- `classes/` - Per-class combined masks (optional)
- `visualizations/` - PNG images with masks (optional)

## Key Options

| Option | Purpose | Works With |
|--------|---------|-----------|
| `--save-coco` | Save COCO format JSON | Both |
| `--save-txt` | Save YOLO format labels | Both |
| `--visualize` | Create visual output | Both |
| `--save-masks` | Save per-class masks | Segmentation only |
| `--save-instances` | Save instance masks | Segmentation only |
| `--sample-size=N` | Process only first N images | Both |
| `--conf-threshold=0.5` | Confidence threshold | Both |
| `--device=cuda:0` | Use GPU | Both |

## Important Notes

1. **Auto-Detection**: The script automatically figures out if your model is segmentation or detection-only. You don't need to specify anything!

2. **COCO Always Saves All**: With `--save-coco`, ALL detections are saved (not just high-confidence ones). Filter by score field downstream.

3. **Segmentation Options Protected**: Using `--save-masks` or `--save-instances` with a detection-only model will give a clear error.

4. **Fast Testing**: Use `--sample-size=10` to test on a small set before running the full dataset.

## Testing Your Model

### Step 1: Test Auto-Detection
```bash
python inference.py images/ model.pth output/ --visualize --sample-size=5
```

If it works and shows appropriate visualizations (boxes for detection, masks+boxes for segmentation), your model is working!

### Step 2: Save COCO
```bash
python inference.py images/ model.pth output/ --save-coco --sample-size=10
```

Check the JSON output to verify format is correct.

### Step 3: Full Run
```bash
# For detection models
python inference.py images/ model.pth output/ --save-coco --save-txt --visualize

# For segmentation models
python inference.py images/ model.pth output/ \
  --save-coco --save-txt --save-masks --save-instances --visualize
```

## Error Solutions

### "ERROR: --save-masks is only available for segmentation models!"
**Solution**: Your model is detection-only. Remove `--save-masks`:
```bash
python inference.py images/ model.pth output/ --visualize --save-coco
```

### "Model type: Detection-only" but I expected segmentation
**Solution**: Check if model was trained with `--no-segmentation`. You can still use it, just without mask-related options.

### Script runs slowly
**Solution**: Use GPU with `--device=cuda:0` or reduce sample with `--sample-size=100`

## Example: Meat FM Boxes Detection Model

```bash
python inference.py \
  ~/drive/datasets/test_meat_fm_boxes/images_and_yolo_labels \
  output/meat_fm_boxes_nov_4_2025/checkpoint_best_ema.pth \
  results/ \
  --model-size=medium \
  --save-coco \
  --visualize \
  --sample-size=10 \
  --device=cpu
```

## Example: Onion Defect Segmentation Model

```bash
python inference.py \
  ~/drive/datasets/test_onion_defect_segmentation_legacy/images \
  output/onion_model_defect_segmentation_oct_7/checkpoint_best_ema.pth \
  results/ \
  --model-size=small \
  --save-coco \
  --save-masks \
  --save-instances \
  --visualize \
  --sample-size=10 \
  --device=cpu
```

## More Help

For detailed information, see:
- `INFERENCE_EXAMPLES.md` - Comprehensive examples and workflows
- `TEST_INFERENCE.md` - Detailed test results
- `IMPLEMENTATION_COMPLETE.md` - Complete technical documentation

## Quick Facts

✅ Works with both segmentation and detection-only models
✅ Automatic model type detection
✅ Clear error messages for invalid options
✅ COCO format output for both types
✅ Sample-size option for fast testing
✅ Backward compatible with existing scripts
✅ Production ready

**That's it! You're ready to use inference.py**

