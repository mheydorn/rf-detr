# RF-DETR Inference Examples

## Quick Start

### Example 1: Save COCO Format (Works with All Models)
```bash
python inference.py \
  ~/drive/datasets/test_meat_fm_boxes/images_and_yolo_labels \
  output/meat_fm_boxes_nov_4_2025/checkpoint_best_ema.pth \
  results/ \
  --model-size=medium \
  --save-coco
```

### Example 2: Visualize Detections with Sample Size (Quick Testing)
```bash
python inference.py \
  ~/drive/datasets/test_meat_fm_boxes/images_and_yolo_labels \
  output/meat_fm_boxes_nov_4_2025/checkpoint_best_ema.pth \
  results/ \
  --model-size=medium \
  --visualize \
  --sample-size=10  # Only process first 10 images
```

### Example 3: Save Everything for Segmentation Model
```bash
python inference.py \
  ~/drive/datasets/test_onion_defect_segmentation_legacy/images \
  output/onion_model_defect_segmentation_oct_7/checkpoint_best_ema.pth \
  results/ \
  --model-size=small \
  --save-coco \
  --save-txt \
  --save-masks \
  --save-instances \
  --visualize \
  --sample-size=20
```

### Example 4: Detection Model Only
```bash
python inference.py \
  ~/drive/datasets/test_meat_fm_boxes/images_and_yolo_labels \
  output/meat_fm_boxes_nov_4_2025/checkpoint_best_ema.pth \
  results/ \
  --model-size=medium \
  --save-coco \
  --save-txt \
  --visualize
```

## Key Features

### 1. Automatic Model Type Detection
The script automatically detects whether your model is:
- **Segmentation model**: Has segmentation head for instance segmentation
- **Detection-only model**: Only outputs bounding boxes

No need to specify anything - the script reads this from the checkpoint!

### 2. Sample Size for Testing
Use `--sample-size=N` to process only the first N images. This is perfect for:
- Quick testing without waiting for full dataset
- Testing on specific sample before full run
- Debugging issues

Example:
```bash
python inference.py images/ model.pth output/ --save-coco --sample-size=10
```

### 3. COCO Format Works for Both Models
**Detection-only model output:**
```json
{
  "id": 1,
  "image_id": 1,
  "category_id": 0,
  "bbox": [100, 50, 200, 150],
  "area": 30000,
  "score": 0.95,
  "iscrowd": 0
}
```

**Segmentation model output (includes segmentation):**
```json
{
  "id": 1,
  "image_id": 1,
  "category_id": 0,
  "bbox": [100, 50, 200, 150],
  "area": 25000,
  "score": 0.95,
  "segmentation": [[100, 50, 200, 50, 200, 200, ...]],
  "iscrowd": 0
}
```

### 4. Text Output (YOLO Format)
Works for both model types:

**Segmentation model**: Uses precise mask polygon
```
0 0.1 0.05 0.3 0.05 0.3 0.2 0.1 0.2
```

**Detection model**: Converts bbox to 4-corner polygon
```
0 0.1 0.05 0.3 0.05 0.3 0.2 0.1 0.2
```

### 5. Visualization
Works for both model types:

**Segmentation model**: Shows masks, boxes, and labels
```bash
python inference.py images/ seg_model.pth output/ --visualize
```

**Detection model**: Shows only boxes and labels (no masks)
```bash
python inference.py images/ det_model.pth output/ --visualize
```

### 6. Error Protection
Segmentation-specific options are protected:

```bash
# This will FAIL with detection-only model:
python inference.py images/ det_model.pth output/ --save-masks
# Error: ERROR: --save-masks is only available for segmentation models!

# This will FAIL with detection-only model:
python inference.py images/ det_model.pth output/ --save-instances
# Error: ERROR: --save-instances is only available for segmentation models!
```

## Output Structure

### Detection-Only Model Results
```
output/
├── coco_annotations.json    # All detections with scores
├── labels/                   # YOLO format txt files (optional)
│   ├── image1.txt
│   └── image2.txt
└── visualizations/           # PNG images with boxes (optional)
    ├── image1_vis.png
    └── image2_vis.png
```

### Segmentation Model Results
```
output/
├── coco_annotations.json    # All detections with scores and masks
├── labels/                   # YOLO format txt files (optional)
├── visualizations/           # PNG images with masks and boxes (optional)
├── instances/                # Instance masks (optional)
│   ├── image1.png
│   └── image2.png
└── classes/                  # Per-class combined masks (optional)
    ├── image1_class0.png
    ├── image1_class1.png
    └── ...
```

## Common Workflows

### Workflow 1: Batch Process Many Images with COCO Output
```bash
python inference.py \
  large_dataset/ \
  model.pth \
  output/ \
  --save-coco \
  --device=cuda:0
```

### Workflow 2: Quick Test with 10 Images
```bash
python inference.py \
  large_dataset/ \
  model.pth \
  output/ \
  --visualize \
  --save-coco \
  --sample-size=10 \
  --device=cpu
```

### Workflow 3: Full Segmentation Analysis
```bash
python inference.py \
  dataset/ \
  seg_model.pth \
  output/ \
  --save-coco \
  --save-txt \
  --save-masks \
  --save-instances \
  --visualize \
  --conf-threshold=0.5 \
  --device=cuda:0
```

### Workflow 4: Detection Analysis with Multiple Confidence Levels
```bash
# High confidence detections only
python inference.py \
  dataset/ \
  det_model.pth \
  output_high/ \
  --save-coco \
  --visualize \
  --conf-threshold=0.8

# All detections (stored in COCO, filtered by score downstream)
python inference.py \
  dataset/ \
  det_model.pth \
  output_all/ \
  --save-coco
```

## Tips and Tricks

### 1. Test with Small Sample First
Always start with `--sample-size=5` to verify everything works:
```bash
python inference.py images/ model.pth output/ --visualize --sample-size=5
```

### 2. Filter by Confidence in COCO
The COCO JSON contains all detections with scores. Filter downstream:
```python
import json
with open('coco_annotations.json') as f:
    data = json.load(f)
    
# High confidence only
high_conf = [a for a in data['annotations'] if a['score'] >= 0.8]
```

### 3. Use GPU for Speed
Default is CPU. For faster processing use GPU:
```bash
python inference.py ... --device=cuda:0  # GPU 0
python inference.py ... --device=cuda:1  # GPU 1
```

### 4. Process Multiple Models Sequentially
```bash
for model in model1.pth model2.pth model3.pth; do
  python inference.py images/ $model output_$model/ --save-coco --sample-size=10
done
```

### 5. Check Model Type Before Processing
```python
# Quick check without running inference
import torch
checkpoint = torch.load('model.pth', map_location='cpu', weights_only=False)
is_seg = checkpoint['args'].segmentation_head
print(f"Model type: {'Segmentation' if is_seg else 'Detection-only'}")
```

## Troubleshooting

### Issue: "Model type: Detection-only" but I expected segmentation
**Solution**: The model was trained with `--no-segmentation` or `segmentation_head=False`. Check checkpoint:
```python
checkpoint = torch.load('model.pth', map_location='cpu', weights_only=False)
print(checkpoint['args'].segmentation_head)
```

### Issue: --save-masks gives error with my model
**Solution**: Your model is detection-only. Use `--visualize` instead to see boxes:
```bash
python inference.py images/ model.pth output/ --visualize --save-coco
```

### Issue: Processing is slow
**Solution**: Use GPU and/or reduce sample size:
```bash
# Use GPU
python inference.py images/ model.pth output/ --device=cuda:0 --save-coco

# Or just process subset
python inference.py images/ model.pth output/ --save-coco --sample-size=100
```

### Issue: COCO JSON is huge
**Solution**: This is expected - COCO includes all detections. Filter by score downstream:
```python
high_conf_only = [a for a in data['annotations'] if a['score'] >= 0.7]
```

## Performance Reference

### Detection-Only Model (meat_fm_boxes, CPU)
- Model loading: ~0.9s
- Per-image: ~2.5s
- Throughput: ~0.4 images/second

### Segmentation Model (onion_defect, CPU)
- Model loading: ~0.9s
- Per-image: ~0.2s
- Throughput: ~4.5 images/second

**Note**: Much faster on GPU! Use `--device=cuda:0` for production.

## Advanced Usage

### Custom Class Filtering
Process only specific classes:
```bash
python inference.py images/ model.pth output/ \
  --save-coco \
  --filter-classes=0,2  # Only classes 0 and 2
```

### Hide Labels in Visualization
Remove class labels from visualization images:
```bash
python inference.py images/ model.pth output/ \
  --visualize \
  --hide-labels
```

### Different Confidence Thresholds
```bash
# For visualization/txt output (conf threshold)
--conf-threshold=0.8

# Note: COCO always saves all detections regardless of threshold
```

## Related Files

- `inference.py` - Main inference script
- `TEST_INFERENCE.md` - Comprehensive test results
- `rfdetr/` - RF-DETR model code
- `output/` - Trained model checkpoints

