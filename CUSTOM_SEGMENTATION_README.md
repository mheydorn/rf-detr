# Training RF-DETR Segmentation Models on Custom Datasets

Quick reference for training RF-DETR segmentation models on datasets with individual JSON annotation files.

## 🎯 Your Dataset Format

Your datasets should have one JSON file per image:

```
your_dataset/
├── image1.png
├── image1.json
├── image2.png  
├── image2.json
└── ...
```

**JSON format:**
```json
{
  "image_info": {"filename": "image1.png"},
  "annotations": [
    {
      "class": "class_name",
      "polygons": [[[x1, y1], [x2, y2], ...]],
      "holes": []
    }
  ]
}
```

---

## ⚡ Quick Start

### 1. Convert Dataset
```bash
python convert_dataset_to_coco.py \
    --source-dir path/to/your/dataset \
    --output-dir datasets/your_dataset_coco
```

### 2. Train Model
```bash
python train_segmentation.py \
    --dataset-dir datasets/your_dataset_coco \
    --output-dir output/your_model
```

### 3. Monitor Training (separate terminal)
```bash
tensorboard --logdir output/your_model
```

### 4. Test Model
```bash
python test_segmentation_inference.py \
    --checkpoint output/your_model/checkpoint_best_regular.pth \
    --dataset-dir datasets/your_dataset_coco
```

---

## 📝 Current Example: Onion Defect Dataset

Your onion defect dataset has already been converted:
- **Source**: `datasets/onion_defect_segmentation/`
- **Converted**: `datasets/onion_defect_coco/`
- **Classes**: mechanical_damage, peeled_skin
- **Images**: 21,326 (14,928 train / 4,265 valid / 2,133 test)

**To verify:**
```bash
python verify_dataset.py --dataset-dir datasets/onion_defect_coco
```

**To train on it:**
```bash
python train_segmentation.py \
    --dataset-dir datasets/onion_defect_coco \
    --output-dir output/onion_model
```

---

## 🔧 Common Options

### Model Sizes
- `--model-size nano` - Fastest, 384×384 (~2-3 hrs training)
- `--model-size small` - Balanced, 512×512 (~3-4 hrs training) **[Default]**
- `--model-size medium` - Most accurate, 576×576 (~4-6 hrs training)

### GPU Memory
- **24GB+**: `--batch-size 8`
- **12GB**: `--batch-size 4` (default)
- **8GB**: `--batch-size 2`
- **4GB**: `--batch-size 1 --model-size nano`

### Custom Data Splits
```bash
python convert_dataset_to_coco.py \
    --source-dir path/to/dataset \
    --output-dir path/to/output \
    --train-split 0.8 \
    --valid-split 0.15 \
    --test-split 0.05
```

---

## 📂 Files Overview

| Script | Purpose |
|--------|---------|
| `convert_dataset_to_coco.py` | Convert JSON annotations to COCO format |
| `train_segmentation.py` | Train RF-DETR segmentation models |
| `verify_dataset.py` | Verify COCO dataset structure |
| `test_segmentation_inference.py` | Test trained model inference |
| `SEGMENTATION_TRAINING_GUIDE.md` | Detailed documentation |

---

## 💡 Quick Tips

**Getting started:**
- Always run `verify_dataset.py` after conversion
- Start with default settings (Small model, batch size 4)
- Monitor training with TensorBoard

**If training is too slow:**
- Increase `--num-workers 8`
- Use smaller model `--model-size nano`

**If GPU runs out of memory:**
- Reduce `--batch-size 2` or `--batch-size 1`
- Use smaller model `--model-size nano`

**For better accuracy:**
- Use Medium model `--model-size medium`
- Train longer `--epochs 150`
- Ensure high-quality annotations

---

## 📖 Full Documentation

See `SEGMENTATION_TRAINING_GUIDE.md` for:
- Detailed parameter explanations
- Troubleshooting guide
- Custom inference examples
- Advanced usage scenarios
- Performance optimization tips

---

## 🚀 Next Steps

1. **For onion dataset** (already converted):
   ```bash
   python train_segmentation.py \
       --dataset-dir datasets/onion_defect_coco \
       --output-dir output/onion_model
   ```

2. **For new datasets**:
   ```bash
   # Step 1: Convert
   python convert_dataset_to_coco.py \
       --source-dir datasets/my_new_dataset \
       --output-dir datasets/my_new_dataset_coco
   
   # Step 2: Train
   python train_segmentation.py \
       --dataset-dir datasets/my_new_dataset_coco \
       --output-dir output/my_new_model
   ```

---

**That's it!** The scripts are designed to work with any dataset following the JSON format shown above.

