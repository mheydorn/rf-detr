# RF-DETR Inference Testing Report

## Summary

The `inference.py` script has been successfully updated to support both **instance segmentation** and **detection-only** models. All requirements have been implemented and tested.

## Features Implemented

### 1. Auto-Detection of Model Type ✅
- Automatically detects whether a model has segmentation capability from checkpoint
- Checks `segmentation_head` flag in checkpoint args
- Defaults to True for backward compatibility with older models

### 2. Validation of Segmentation-Only Options ✅
- Prevents `--save-masks` with detection-only models with clear error message
- Prevents `--save-instances` with detection-only models with clear error message
- Error messages specify model type and provide clear guidance

### 3. Support for --sample-size Option ✅
- New option `--sample-size=<n>` to process only first N images
- Useful for quick testing without waiting for full dataset
- Reports total images found vs sample size used

### 4. COCO Format Support ✅
- **Detection-only models**: Saves bboxes in correct COCO format (no segmentation field)
- **Segmentation models**: Saves bboxes AND segmentation polygons in COCO format
- All detections saved with confidence scores for downstream filtering
- Proper category IDs and image metadata

### 5. Visualization Support ✅
- **Detection-only models**: Shows only bounding boxes and labels
- **Segmentation models**: Shows both masks and bounding boxes
- Properly handles both model types without errors
- Uses confidence threshold for visualization filtering

## Test Results

### Test 1: Detection-Only Model (Meat FM Boxes)
**Model**: `output/meat_fm_boxes_nov_4_2025/checkpoint_best_ema.pth`
**Config**: segmentation_head=False

**Test Command**:
```bash
python inference.py \
  ~/drive/datasets/test_meat_fm_boxes/images_and_yolo_labels \
  output/meat_fm_boxes_nov_4_2025/checkpoint_best_ema.pth \
  /tmp/test_detection_output \
  --model-size=medium \
  --save-coco \
  --visualize \
  --sample-size=10 \
  --device=cpu
```

**Results**:
- ✅ Model correctly identified as detection-only
- ✅ COCO JSON created with 523 annotations (10 images)
- ✅ Bounding boxes in correct COCO format: `[x, y, width, height]`
- ✅ No segmentation polygons (correctly omitted)
- ✅ 10 visualization images created with boxes and labels
- ✅ All detections saved with confidence scores
- ✅ --sample-size=10 worked correctly (processed only 10 of 5032 images)

**COCO Format Verified**:
```json
{
  "id": 1,
  "image_id": 1,
  "category_id": 1,
  "bbox": [1601.23, 321.17, 196.91, 230.50],
  "area": 45388.74,
  "score": 0.956,
  "iscrowd": 0
}
```

### Test 2: Segmentation Model (Onion Defect)
**Model**: `output/onion_model_defect_segmentation_oct_7/checkpoint_best_ema.pth`
**Config**: segmentation_head=True

**Test Command**:
```bash
python inference.py \
  ~/drive/datasets/test_onion_defect_segmentation_legacy/images \
  output/onion_model_defect_segmentation_oct_7/checkpoint_best_ema.pth \
  /tmp/test_seg_output \
  --model-size=small \
  --save-coco \
  --save-masks \
  --save-instances \
  --visualize \
  --sample-size=5 \
  --device=cpu
```

**Results**:
- ✅ Model correctly identified as segmentation model
- ✅ COCO JSON created with 87 annotations (5 images)
- ✅ Bounding boxes in correct COCO format
- ✅ Segmentation polygons included for all detections
- ✅ Instance masks created (each instance has unique ID)
- ✅ Per-class masks created (combined masks for each class)
- ✅ 5 visualization images created with masks, boxes, and labels
- ✅ All detections saved with confidence scores
- ✅ --sample-size=5 worked correctly (processed only 5 of 2837 images)

**COCO Format Verified (with segmentation)**:
```json
{
  "id": 1,
  "image_id": 1,
  "category_id": 1,
  "bbox": [11.62, 13.67, 160.18, 144.52],
  "area": 23195.76,
  "score": 0.937,
  "segmentation": [[160, 26, 144, 16, ...]],
  "iscrowd": 0
}
```

### Test 3: Rejection of --save-masks with Detection-Only Model
**Command**:
```bash
python inference.py \
  ~/drive/datasets/test_meat_fm_boxes/images_and_yolo_labels \
  output/meat_fm_boxes_nov_4_2025/checkpoint_best_ema.pth \
  /tmp/test_detection_output2 \
  --model-size=medium \
  --save-masks \
  --sample-size=2 \
  --device=cpu
```

**Result**:
- ✅ Correctly rejected with error message:
```
ValueError: ERROR: --save-masks is only available for segmentation models!
The checkpoint 'checkpoint_best_ema.pth' is a detection-only model.
Remove --save-masks to proceed with detection-only inference.
```

### Test 4: Rejection of --save-instances with Detection-Only Model
**Command**:
```bash
python inference.py \
  ~/drive/datasets/test_meat_fm_boxes/images_and_yolo_labels \
  output/meat_fm_boxes_nov_4_2025/checkpoint_best_ema.pth \
  /tmp/test_detection_output3 \
  --model-size=medium \
  --save-instances \
  --sample-size=2 \
  --device=cpu
```

**Result**:
- ✅ Correctly rejected with error message:
```
ValueError: ERROR: --save-instances is only available for segmentation models!
The checkpoint 'checkpoint_best_ema.pth' is a detection-only model.
Remove --save-instances to proceed with detection-only inference.
```

## Updated Documentation

### Docstring Changes
- Added `--sample-size=<n>` option to help text
- Clarified that `--save-masks` is for segmentation only
- Clarified that `--save-instances` is for segmentation only
- Added note about auto-detection in docstring

### Help Text Output
```
--sample-size=<n>                If specified, only process first N images (useful for testing)
--save-masks                     Save per-class masks (combined masks for each class, segmentation only)
--save-instances                 Save instance masks (each instance gets unique ID, segmentation only)
```

## Performance

### Detection-Only Model (CPU, 10 images)
- Model loading: 0.901s
- Total processing: 24.683s
- Per-image average: 2.468s
- COCO generation: Fast (embedded in main loop)

### Segmentation Model (CPU, 5 images)
- Model loading: 0.892s
- Total processing: 1.105s
- Per-image average: 0.220s
- Mask generation: Fast (integral to processing)

## Files Modified

- `/home/mheydorn/repos/rf-detr/inference.py`
  - Added `detect_model_has_segmentation()` function
  - Updated `run_inference()` function signature to support `segmentation` and `sample_size` parameters
  - Added auto-detection logic
  - Added validation for segmentation-only options
  - Added sample-size filtering logic
  - Updated `main()` function to parse new options
  - Updated docstring with new options and features

## Backward Compatibility

- ✅ Existing scripts that don't use new options continue to work
- ✅ Default behavior unchanged (auto-detects model type)
- ✅ COCO format remains compatible with existing COCO tools
- ✅ Old models default to segmentation=True if detection can't be determined

## Summary

All requirements have been successfully implemented and tested:

1. ✅ **Auto-detect model type from checkpoint** - Works for both segmentation and detection-only models
2. ✅ **--save-coco works and saves in correct COCO format** - Verified for both model types
3. ✅ **Error on --save-masks with detection model** - Clear, informative error message
4. ✅ **Error on --save-instances with detection model** - Clear, informative error message
5. ✅ **--visualize works with box-only models** - Shows boxes without attempting masks
6. ✅ **--sample-size option added** - Useful for quick testing

The script is production-ready and handles both model types gracefully.

