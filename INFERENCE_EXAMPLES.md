# RF-DETR Inference Examples

This document provides examples of how to use the `inference.py` script for running RF-DETR models on your images.

## Basic Usage

```bash
./inference.py <images_dir> <weights_path> <output_dir> [options]
```

## Examples with Onion Defect Dataset

### 1. Basic Detection Inference

Run detection-only inference (no masks):
```bash
python inference.py \
    datasets/onion_defect_coco/test \
    rf-detr-small.pth \
    inference_output \
    --num-classes=2 \
    --conf=0.5
```

### 2. Segmentation Inference with Visualization

Run segmentation inference with visual output:
```bash
python inference.py \
    datasets/onion_defect_coco/test \
    output/onion_model_defect_segmentation_oct_7/checkpoint_best_regular.pth \
    inference_output_seg \
    --segmentation \
    --visualize \
    --class-names=datasets/onion_defect_coco/class_names.txt \
    --conf=0.3
```

### 3. Save All Outputs (Masks, COCO, Text, Visualization)

Complete inference with all output formats:
```bash
python inference.py \
    datasets/onion_defect_coco/test \
    output/onion_model_defect_segmentation_oct_7/checkpoint_best_regular.pth \
    inference_output_full \
    --segmentation \
    --visualize \
    --save-txt \
    --save-conf \
    --save-coco-labels \
    --class-names=datasets/onion_defect_coco/class_names.txt \
    --conf=0.3
```

### 4. GPU Inference

Run on CUDA GPU:
```bash
python inference.py \
    datasets/onion_defect_coco/test \
    output/onion_model_defect_segmentation_oct_7/checkpoint_best_regular.pth \
    inference_output_gpu \
    --segmentation \
    --visualize \
    --device=cuda:0 \
    --class-names=datasets/onion_defect_coco/class_names.txt
```

### 5. Filter Specific Classes

Only detect specific classes (e.g., only class 0 - mechanical_damage):
```bash
python inference.py \
    datasets/onion_defect_coco/test \
    output/onion_model_defect_segmentation_oct_7/checkpoint_best_regular.pth \
    inference_output_filtered \
    --segmentation \
    --visualize \
    --filter-classes=0 \
    --class-names=datasets/onion_defect_coco/class_names.txt \
    --conf=0.25
```

### 6. Inference on a Subset of Images

Create a temporary directory with a few test images:
```bash
mkdir -p test_images
cp datasets/onion_defect_coco/test/2023-01-25-21-48-09.528215_SVW-C8E447_34_run.png test_images/
cp datasets/onion_defect_coco/test/2023-01-25-21-48-10.278337_SVW-C8E447_20_run.png test_images/

python inference.py \
    test_images \
    output/onion_model_defect_segmentation_oct_7/checkpoint_best_regular.pth \
    quick_test_output \
    --segmentation \
    --visualize \
    --class-names=datasets/onion_defect_coco/class_names.txt
```

## Output Structure

After running inference, your output directory will contain:

```
output_dir/
├── masks/                      # Individual mask images (PNG)
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── labels/                     # Text files with segmentation polygons
│   ├── image1.txt
│   ├── image2.txt
│   └── ...
├── visualizations/            # Annotated images with overlaid masks/boxes
│   ├── image1_vis.png
│   ├── image2_vis.png
│   └── ...
└── coco_annotations.json      # COCO format annotations (if --save-coco-labels)
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--conf=<conf>` | Confidence threshold | 0.5 |
| `--model-size=<size>` | Model size (nano, small, medium, large) | small |
| `--device=<device>` | Device (cpu, cuda, mps, cuda:0, etc.) | cpu |
| `--num-classes=<n>` | Number of classes | (required if no class-names) |
| `--class-names=<path>` | Path to class names file | (optional) |
| `--save-txt` | Save segmentation polygons as text | false |
| `--save-conf` | Include confidence scores in text output | false |
| `--save-coco-labels` | Save COCO format JSON | false |
| `--visualize` | Save visualization images | false |
| `--hide-labels` | Hide labels on visualization | false |
| `--filter-classes=<ids>` | Filter by class IDs (comma-separated) | (none) |
| `--segmentation` | Enable segmentation head | false |

## Tips

1. **Lower confidence threshold** if you're missing detections: `--conf=0.25`
2. **Use GPU** for faster inference on large image batches: `--device=cuda:0`
3. **Save COCO format** for easy evaluation: `--save-coco-labels`
4. **Filter classes** to focus on specific defect types: `--filter-classes=0,1`
5. **Use class names file** for better visualization labels: `--class-names=path/to/class_names.txt`

## Comparison to YOLO Reference

The RF-DETR inference script follows a similar CLI pattern to the YOLO reference script:

**Similar:**
- Same argument structure: `<images_dir> <weights_path> <output_dir>`
- Similar options: `--conf`, `--device`, `--visualize`, `--save-txt`, `--save-coco-labels`
- Same output directory structure

**Different:**
- RF-DETR uses `--model-size` instead of auto-detecting from weights
- RF-DETR requires `--segmentation` flag for segmentation models
- RF-DETR uses `--class-names` or `--num-classes` instead of auto-detecting
- RF-DETR uses `--filter-classes` instead of `--classes`
- No `--iou` or `--imgsz` options (RF-DETR handles these internally)
- Omitted YOLO-specific output format options

## Performance Notes

- **CPU inference**: Expect ~500ms-2s per image depending on image size
- **GPU inference**: Expect ~50-200ms per image (after warmup)
- **Batch processing**: The script processes images sequentially; for large batches, GPU is recommended


