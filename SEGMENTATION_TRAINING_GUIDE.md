# RF-DETR Segmentation Training Guide

This guide explains how to train RF-DETR segmentation models on custom datasets using the provided conversion and training scripts.

## 📋 Overview

This toolkit enables you to:
1. Convert datasets with individual JSON annotations to COCO format
2. Train RF-DETR segmentation models (Nano, Small, Medium)
3. Evaluate and test your trained models
4. Deploy models for inference

## 🎯 Supported Dataset Format

Your dataset should have this structure:
```
your_dataset/
├── image1.png
├── image1.json
├── image2.png
├── image2.json
└── ...
```

Each JSON file should follow this format:
```json
{
  "image_info": {
    "filename": "image1.png"
  },
  "annotations": [
    {
      "class": "class_name",
      "polygons": [
        [[x1, y1], [x2, y2], [x3, y3], ...]
      ],
      "holes": []
    }
  ]
}
```

**Key points:**
- One JSON file per image with matching filename (except extension)
- Polygons are arrays of [x, y] coordinate pairs
- Multiple polygons per class are supported
- Multiple classes per image are supported

---

## 🚀 Quick Start

### Step 1: Convert Your Dataset

```bash
python convert_dataset_to_coco.py \
    --source-dir path/to/your/dataset \
    --output-dir datasets/your_dataset_coco
```

This will:
- Scan all JSON files and discover classes automatically
- Split data into train (70%), valid (20%), test (10%)
- Convert polygons to COCO format
- Copy images to organized folders
- Create `_annotations.coco.json` files

### Step 2: Verify the Conversion

```bash
python verify_dataset.py --dataset-dir datasets/your_dataset_coco
```

This checks that everything is properly formatted.

### Step 3: Train Your Model

```bash
python train_segmentation.py \
    --dataset-dir datasets/your_dataset_coco \
    --output-dir output/my_model
```

### Step 4: Monitor Training

In a separate terminal:
```bash
tensorboard --logdir output/my_model
```

Open http://localhost:6006 in your browser.

### Step 5: Test Your Model

```bash
python test_segmentation_inference.py \
    --checkpoint output/my_model/checkpoint_best_regular.pth \
    --dataset-dir datasets/your_dataset_coco
```

---

## 📚 Detailed Instructions

### Dataset Conversion Options

```bash
python convert_dataset_to_coco.py \
    --source-dir path/to/source \
    --output-dir path/to/output \
    --train-split 0.7 \
    --valid-split 0.2 \
    --test-split 0.1 \
    --seed 42 \
    --supercategory "object"
```

**Arguments:**
- `--source-dir` (required): Path to source dataset with JSON files
- `--output-dir` (required): Path where COCO format will be saved
- `--train-split`: Fraction of data for training (default: 0.7)
- `--valid-split`: Fraction of data for validation (default: 0.2)
- `--test-split`: Fraction of data for testing (default: 0.1)
- `--seed`: Random seed for reproducibility (default: 42)
- `--supercategory`: Category name for all classes (default: "object")

**Output structure:**
```
your_dataset_coco/
├── train/
│   ├── *.png (or *.jpg)
│   └── _annotations.coco.json
├── valid/
│   ├── *.png
│   └── _annotations.coco.json
├── test/
│   ├── *.png
│   └── _annotations.coco.json
└── class_names.txt
```

---

### Training Options

```bash
python train_segmentation.py \
    --dataset-dir datasets/your_dataset_coco \
    --output-dir output/my_model \
    --model-size small \
    --batch-size 4 \
    --epochs 100 \
    --lr 1e-4 \
    --num-workers 4
```

**Arguments:**
- `--dataset-dir` (required): Path to COCO-format dataset
- `--output-dir` (required): Path to save training outputs
- `--model-size`: Model size - `nano`, `small`, or `medium` (default: small)
- `--batch-size`: Batch size for training (default: 4)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 1e-4)
- `--resolution`: Input resolution (uses model default if not specified)
- `--num-workers`: Number of data loading workers (default: 2)
- `--no-pretrained`: Don't use pretrained weights

**Model Sizes:**

| Model  | Resolution | Parameters | Speed    | Use Case                    |
|--------|-----------|------------|----------|-----------------------------|
| Nano   | 384×384   | 30.5M      | Fastest  | Edge devices, quick tests   |
| Small  | 512×512   | 32.1M      | Fast     | Production (recommended)    |
| Medium | 576×576   | 33.7M      | Moderate | Maximum accuracy            |

**Training Configuration:**

The training script uses optimized defaults:
- Optimizer: AdamW with layer-wise LR decay
- EMA (Exponential Moving Average): Enabled
- Multi-scale training: Enabled
- Data augmentation: Random flips, crops, resizes
- Early stopping: 15 epochs patience
- Gradient accumulation: Effective batch size of 16
- Loss weights optimized for segmentation

**Training Outputs:**
```
output/my_model/
├── checkpoint_best_regular.pth    # Best model (use this!)
├── checkpoint_last.pth            # Latest checkpoint
├── checkpoint_epoch_10.pth        # Periodic checkpoints
├── checkpoint_epoch_20.pth
├── ...
├── metrics.json                   # Training metrics
└── events.out.tfevents.*          # TensorBoard logs
```

---

### Testing and Inference

Test on a random image from your test set:

```bash
python test_segmentation_inference.py \
    --checkpoint output/my_model/checkpoint_best_regular.pth \
    --dataset-dir datasets/your_dataset_coco \
    --model-size small \
    --threshold 0.3 \
    --output test_result.jpg
```

**Arguments:**
- `--checkpoint` (required): Path to trained model checkpoint
- `--dataset-dir` (required): Path to dataset directory
- `--model-size`: Model size used for training (default: small)
- `--threshold`: Confidence threshold (default: 0.3)
- `--output`: Output path for annotated image (default: test_inference_result.jpg)

---

### Custom Inference Code

```python
from rfdetr import RFDETRSmall
from PIL import Image
import supervision as sv

# Load your trained model
model = RFDETRSmall(
    num_classes=YOUR_NUM_CLASSES,
    pretrain_weights="output/my_model/checkpoint_best_regular.pth",
    segmentation_head=True
)

# Optimize for faster inference
model.optimize_for_inference()

# Load and predict
image = Image.open("your_image.jpg")
detections = model.predict(image, threshold=0.5)

# Visualize (requires supervision package)
class_names = ['class1', 'class2', ...]  # Your class names
labels = [
    f"{class_names[class_id]} {confidence:.2f}"
    for class_id, confidence in zip(detections.class_id, detections.confidence)
]

annotated = image.copy()
if detections.mask is not None:
    annotated = sv.MaskAnnotator().annotate(annotated, detections)
annotated = sv.BoxAnnotator().annotate(annotated, detections)
annotated = sv.LabelAnnotator().annotate(annotated, detections, labels)

annotated.save("result.jpg")
```

---

## 💡 Tips and Best Practices

### Choosing Model Size

- **Nano**: Use for edge devices, mobile deployment, or quick experiments
- **Small**: Best for production - good balance of speed and accuracy
- **Medium**: Use when accuracy is critical and speed is less important

### Adjusting Batch Size

Based on GPU memory:
- **24GB+ GPU**: `--batch-size 8` or higher
- **12GB GPU**: `--batch-size 4` (default)
- **8GB GPU**: `--batch-size 2`
- **4GB GPU**: `--batch-size 1` with `--model-size nano`

### Training Time Estimates

On NVIDIA RTX 4090:
- **Nano**: ~2-3 hours for 100 epochs
- **Small**: ~3-4 hours for 100 epochs
- **Medium**: ~4-6 hours for 100 epochs

*Scales with dataset size and GPU performance*

### Improving Model Performance

1. **More training data**: The more diverse annotated images, the better
2. **Balance classes**: Ensure all classes have sufficient examples
3. **Quality annotations**: Accurate polygon annotations are crucial
4. **Longer training**: Increase epochs if model is still improving
5. **Learning rate**: Try `--lr 5e-5` or `--lr 2e-4` if default doesn't converge
6. **Data augmentation**: Enabled by default, helps generalization

---

## 🔧 Troubleshooting

### CUDA Out of Memory

**Solution 1**: Reduce batch size
```bash
python train_segmentation.py ... --batch-size 2
```

**Solution 2**: Use smaller model
```bash
python train_segmentation.py ... --model-size nano
```

**Solution 3**: Reduce resolution (if needed)
```bash
python train_segmentation.py ... --resolution 384
```

### Training is Slow

**Solution 1**: Increase workers
```bash
python train_segmentation.py ... --num-workers 8
```

**Solution 2**: Check GPU usage
```bash
nvidia-smi  # Should show GPU utilization near 100%
```

**Solution 3**: Enable any available optimizations in your environment

### Model Not Converging

1. Check your data with verify_dataset.py
2. Try different learning rate: `--lr 5e-5` or `--lr 2e-4`
3. Train longer: `--epochs 150`
4. Check annotation quality
5. Ensure sufficient training data

### No Detections During Inference

1. Lower threshold: `--threshold 0.1`
2. Check if model finished training
3. Verify you're using the best checkpoint
4. Test on training images first to verify model works

### Import Errors

Install required packages:
```bash
pip install rfdetr supervision pillow tqdm tensorboard
```

---

## 📊 Example Workflow

Here's a complete example using a hypothetical defect detection dataset:

```bash
# Step 1: Convert dataset
python convert_dataset_to_coco.py \
    --source-dir datasets/defect_raw \
    --output-dir datasets/defect_coco

# Step 2: Verify
python verify_dataset.py --dataset-dir datasets/defect_coco

# Step 3: Train
python train_segmentation.py \
    --dataset-dir datasets/defect_coco \
    --output-dir output/defect_model \
    --model-size small \
    --epochs 100

# Step 4: Monitor (separate terminal)
tensorboard --logdir output/defect_model

# Step 5: Test
python test_segmentation_inference.py \
    --checkpoint output/defect_model/checkpoint_best_regular.pth \
    --dataset-dir datasets/defect_coco
```

---

## 📈 Understanding Training Metrics

### Key Metrics to Watch

**Loss curves (should decrease):**
- `loss_ce`: Classification loss
- `loss_bbox`: Bounding box regression loss
- `loss_giou`: GIoU loss for box quality
- `loss_mask_ce`: Mask classification loss
- `loss_mask_dice`: Dice loss for segmentation

**Validation metrics (should increase):**
- `coco_eval/bbox/AP`: Box detection mAP
- `coco_eval/segm/AP`: Segmentation mAP
- `coco_eval/bbox/AP50`: Box AP at IoU 0.5
- `coco_eval/segm/AP50`: Segmentation AP at IoU 0.5

**Early stopping:**
- Training automatically stops if validation AP doesn't improve for 15 epochs
- Best model is always saved as `checkpoint_best_regular.pth`

---

## 🎓 Advanced Usage

### Custom Data Splits

```bash
python convert_dataset_to_coco.py \
    --source-dir datasets/my_data \
    --output-dir datasets/my_data_coco \
    --train-split 0.8 \
    --valid-split 0.15 \
    --test-split 0.05
```

### Resume Training

If training was interrupted, resume from last checkpoint:

```python
from rfdetr import RFDETRSmall
from rfdetr.config import SegmentationTrainConfig

model = RFDETRSmall(...)
config = SegmentationTrainConfig(
    dataset_dir="datasets/my_data_coco",
    output_dir="output/my_model",
    resume="output/my_model/checkpoint_last.pth",  # Resume from here
    ...
)
model.train(config)
```

### Fine-tuning from Another Dataset

```python
model = RFDETRSmall(
    num_classes=YOUR_NUM_CLASSES,
    pretrain_weights="output/previous_model/checkpoint_best_regular.pth",
    segmentation_head=True
)
# Then train as usual
```

---

## 🛠️ Script Reference

### `convert_dataset_to_coco.py`
Converts individual JSON annotations to COCO format.

### `train_segmentation.py`
Main training script with optimized defaults.

### `test_segmentation_inference.py`
Tests trained model on sample images.

### `verify_dataset.py`
Verifies COCO dataset structure and format.

---

## 📚 Additional Resources

- **RF-DETR Documentation**: https://rfdetr.roboflow.com
- **RF-DETR GitHub**: https://github.com/roboflow/rf-detr
- **COCO Format**: https://cocodataset.org/#format-data
- **Supervision Library**: https://github.com/roboflow/supervision

---

## ✅ Checklist

Before training:
- [ ] Dataset converted to COCO format
- [ ] Dataset verified with verify_dataset.py
- [ ] GPU available (`nvidia-smi` shows GPU)
- [ ] Required packages installed

During training:
- [ ] TensorBoard running for monitoring
- [ ] Training losses decreasing
- [ ] Validation metrics improving
- [ ] Checkpoints being saved

After training:
- [ ] Best model checkpoint exists
- [ ] Test inference produces good results
- [ ] Model ready for deployment

---

## 🤝 Support

For questions or issues:
1. Check this guide thoroughly
2. Verify dataset format with verify_dataset.py
3. Check RF-DETR documentation
4. Open an issue on GitHub if needed

---

**Happy Training!** 🚀

