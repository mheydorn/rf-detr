# Onion Defect Segmentation Training Guide

This guide will help you train an RF-DETR segmentation model on the onion defect dataset.

## Dataset Overview

- **Total Images**: 21,326
- **Classes**: 2
  - `mechanical_damage`
  - `peeled_skin`
- **Split**:
  - Train: 14,928 images (4,277 annotations)
  - Valid: 4,265 images (1,189 annotations)
  - Test: 2,133 images (583 annotations)

## Quick Start

The dataset has already been converted to COCO format. To start training immediately:

```bash
python train_onion_segmentation.py
```

This will train an RF-DETR-Small segmentation model with default settings.

## Training Options

### Model Sizes

You can choose from three model sizes:

1. **Nano** (fastest, smallest)
   - Resolution: 384x384
   - ~30.5M parameters
   - Best for: Quick experiments, edge devices

2. **Small** (recommended, balanced)
   - Resolution: 512x512
   - ~32.1M parameters
   - Best for: Production use, good balance of speed and accuracy

3. **Medium** (most accurate)
   - Resolution: 576x576
   - ~33.7M parameters
   - Best for: Maximum accuracy

### Training Commands

#### Train with default settings (RF-DETR-Small):
```bash
python train_onion_segmentation.py
```

#### Train with RF-DETR-Nano (faster, smaller):
```bash
python train_onion_segmentation.py --model-size nano --batch-size 8
```

#### Train with RF-DETR-Medium (more accurate):
```bash
python train_onion_segmentation.py --model-size medium --batch-size 2
```

#### Customize training parameters:
```bash
python train_onion_segmentation.py \
    --model-size small \
    --batch-size 4 \
    --epochs 100 \
    --lr 1e-4 \
    --num-workers 4 \
    --output-dir output/my_onion_model
```

### Available Arguments

- `--dataset-dir`: Path to COCO-format dataset (default: `datasets/onion_defect_coco`)
- `--output-dir`: Path to save training outputs (default: `output/onion_segmentation`)
- `--model-size`: Model size: `nano`, `small`, or `medium` (default: `small`)
- `--batch-size`: Batch size for training (default: 4)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 1e-4)
- `--resolution`: Input resolution (defaults to model default)
- `--num-workers`: Number of data loading workers (default: 2)
- `--no-pretrained`: Don't use pretrained weights

## Training Configuration

The training script uses these optimized settings:

- **Optimizer**: AdamW with layer-wise learning rate decay
- **Learning Rate**: 1e-4 (backbone: 1.5e-4)
- **Data Augmentation**: 
  - Multi-scale training
  - Random horizontal flips
  - Random crops and resizes
- **Regularization**:
  - EMA (Exponential Moving Average) with decay 0.993
  - Weight decay: 1e-4
- **Early Stopping**: Enabled with patience of 15 epochs
- **Loss Weights**:
  - Classification loss: 5.0
  - Mask CE loss: 5.0
  - Mask Dice loss: 5.0
  - BBox loss: 5.0
  - GIoU loss: 2.0

## Monitoring Training

### TensorBoard

The training script enables TensorBoard by default. To monitor training progress:

```bash
tensorboard --logdir output/onion_segmentation
```

Then open http://localhost:6006 in your browser.

### Training Outputs

Training outputs are saved to the output directory:

```
output/onion_segmentation/
├── checkpoint_best_regular.pth    # Best model checkpoint
├── checkpoint_last.pth            # Latest checkpoint
├── checkpoint_epoch_10.pth        # Periodic checkpoints
├── checkpoint_epoch_20.pth
├── ...
├── metrics.json                   # Training metrics
└── events.out.tfevents.*          # TensorBoard logs
```

## Expected Training Time

Approximate training times (on NVIDIA RTX 4090):

- **Nano**: ~2-3 hours for 100 epochs
- **Small**: ~3-4 hours for 100 epochs
- **Medium**: ~4-6 hours for 100 epochs

*Note: Times vary based on GPU, batch size, and dataset size.*

## Performance Tips

### If you have limited GPU memory:

1. Reduce batch size:
   ```bash
   python train_onion_segmentation.py --batch-size 2
   ```

2. Use gradient accumulation (automatically set in config to effective batch size of 16)

3. Use the Nano model:
   ```bash
   python train_onion_segmentation.py --model-size nano
   ```

### If you want faster training:

1. Increase batch size (if you have GPU memory):
   ```bash
   python train_onion_segmentation.py --batch-size 8
   ```

2. Increase number of workers:
   ```bash
   python train_onion_segmentation.py --num-workers 8
   ```

3. Reduce epochs if you see early convergence:
   ```bash
   python train_onion_segmentation.py --epochs 50
   ```

## Evaluation

After training, you can evaluate your model on the test set:

```python
from rfdetr import RFDETRSmall
from pathlib import Path

# Load your trained model
model = RFDETRSmall(
    num_classes=2,
    pretrain_weights="output/onion_segmentation/checkpoint_best_regular.pth",
    segmentation_head=True
)

# Run evaluation
# (Add evaluation code here)
```

## Inference

To use your trained model for inference:

```python
import io
import requests
from PIL import Image
from rfdetr import RFDETRSmall
import supervision as sv

# Load model
model = RFDETRSmall(
    num_classes=2,
    pretrain_weights="output/onion_segmentation/checkpoint_best_regular.pth",
    segmentation_head=True
)

model.optimize_for_inference()

# Load image
image_path = "path/to/your/onion/image.png"
image = Image.open(image_path)

# Run inference
detections = model.predict(image, threshold=0.5)

# Visualize results
class_names = ['mechanical_damage', 'peeled_skin']
labels = [
    f"{class_names[class_id]} {confidence:.2f}"
    for class_id, confidence in zip(detections.class_id, detections.confidence)
]

annotated_image = image.copy()
if detections.mask is not None:
    annotated_image = sv.MaskAnnotator().annotate(annotated_image, detections)
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

sv.plot_image(annotated_image)
```

## Troubleshooting

### Out of Memory Error

If you get a CUDA out of memory error:
1. Reduce batch size: `--batch-size 1` or `--batch-size 2`
2. Use smaller model: `--model-size nano`
3. Reduce resolution (if needed)

### Training is too slow

1. Increase number of workers: `--num-workers 8`
2. Check if you're using GPU: Training should show CUDA device
3. Enable mixed precision (already enabled by default)

### Model not converging

1. Try adjusting learning rate: `--lr 5e-5` or `--lr 2e-4`
2. Increase training epochs: `--epochs 150`
3. Check data quality and annotations

### Import errors

Make sure you have the required packages installed:
```bash
pip install rfdetr supervision pillow tqdm
```

## Advanced: Manual Training with Config

For more control, you can use the Python API directly:

```python
from rfdetr import RFDETRSmall
from rfdetr.config import SegmentationTrainConfig

# Create model
model = RFDETRSmall(
    num_classes=2,
    segmentation_head=True,
)

# Configure training
config = SegmentationTrainConfig(
    dataset_dir="datasets/onion_defect_coco",
    output_dir="output/onion_custom",
    batch_size=4,
    epochs=100,
    lr=1e-4,
    class_names=['mechanical_damage', 'peeled_skin'],
    # Add more custom parameters here
)

# Train
model.train(config)
```

## Dataset Re-conversion

If you need to reconvert the dataset with different splits:

```bash
python convert_onion_to_coco.py \
    --source-dir datasets/onion_defect_segmentation \
    --output-dir datasets/onion_defect_coco_custom \
    --train-split 0.8 \
    --valid-split 0.15 \
    --test-split 0.05
```

## Citation

If you use RF-DETR in your research, please cite:

```bibtex
@software{rf-detr,
  author = {Robinson, Isaac and Robicheaux, Peter and Popov, Matvei and Ramanan, Deva and Peri, Neehar},
  license = {Apache-2.0},
  title = {RF-DETR},
  howpublished = {\url{https://github.com/roboflow/rf-detr}},
  year = {2025},
  note = {SOTA Real-Time Object Detection Model}
}
```

## Support

For issues and questions:
- RF-DETR GitHub: https://github.com/roboflow/rf-detr
- RF-DETR Documentation: https://rfdetr.roboflow.com
- Roboflow Forum: https://discuss.roboflow.com

