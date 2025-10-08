# Quick Start: Onion Defect Segmentation Training

This is a quick reference guide for training RF-DETR on your onion defect segmentation dataset.

## 🎯 Dataset Status

✅ **Dataset converted and ready for training!**

- **Source**: `datasets/onion_defect_segmentation/` (original format)
- **Converted**: `datasets/onion_defect_coco/` (COCO format)
- **Total images**: 21,326
- **Classes**: 2 (mechanical_damage, peeled_skin)
- **Split**: 70% train / 20% valid / 10% test

## 🚀 Quick Commands

### 1. Verify Dataset (optional)
```bash
python verify_dataset.py
```

### 2. Start Training
```bash
# Default (RF-DETR-Small, recommended)
python train_onion_segmentation.py

# Or with RF-DETR-Nano (faster)
python train_onion_segmentation.py --model-size nano --batch-size 8

# Or with RF-DETR-Medium (more accurate)
python train_onion_segmentation.py --model-size medium --batch-size 2
```

### 3. Monitor Training
```bash
# In a separate terminal
tensorboard --logdir output/onion_segmentation
```

### 4. Test Inference (after training)
```bash
python test_inference.py
```

## 📊 What to Expect

### Training Time (on RTX 4090)
- **Nano**: ~2-3 hours for 100 epochs
- **Small**: ~3-4 hours for 100 epochs  
- **Medium**: ~4-6 hours for 100 epochs

### Training Progress
- Model saves checkpoints every 10 epochs
- Early stopping enabled (patience: 15 epochs)
- Best model saved as `checkpoint_best_regular.pth`
- Metrics logged to TensorBoard

### Output Structure
```
output/onion_segmentation/
├── checkpoint_best_regular.pth    # Best model (use this for inference)
├── checkpoint_last.pth            # Latest checkpoint
├── checkpoint_epoch_*.pth         # Periodic checkpoints
├── metrics.json                   # Training metrics
└── events.out.tfevents.*          # TensorBoard logs
```

## 🔧 Common Adjustments

### If GPU memory is low:
```bash
python train_onion_segmentation.py --batch-size 2 --model-size nano
```

### If training is too slow:
```bash
python train_onion_segmentation.py --num-workers 8
```

### Custom output directory:
```bash
python train_onion_segmentation.py --output-dir output/my_model
```

### Different learning rate:
```bash
python train_onion_segmentation.py --lr 5e-5
```

## 📝 Files Created

1. **convert_onion_to_coco.py** - Converts dataset to COCO format
2. **train_onion_segmentation.py** - Training script
3. **verify_dataset.py** - Dataset verification script
4. **test_inference.py** - Test inference on sample images
5. **ONION_SEGMENTATION_TRAINING.md** - Detailed training guide

## 🆘 Troubleshooting

**Problem**: CUDA out of memory
```bash
python train_onion_segmentation.py --batch-size 1 --model-size nano
```

**Problem**: Training not starting
- Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
- Check dataset exists: `python verify_dataset.py`

**Problem**: No detections during inference
- Try lowering threshold: `python test_inference.py --threshold 0.1`
- Check if model finished training

**Problem**: Import errors
```bash
pip install rfdetr supervision pillow tqdm tensorboard
```

## 📖 More Information

For detailed documentation, see:
- **ONION_SEGMENTATION_TRAINING.md** - Full training guide
- **RF-DETR Docs**: https://rfdetr.roboflow.com

## 🎓 Next Steps After Training

1. **Evaluate on test set**: The training script automatically evaluates on test set
2. **Test on new images**: Use `test_inference.py` or write custom inference code
3. **Deploy model**: Export to ONNX or use with Roboflow Inference
4. **Fine-tune**: Adjust hyperparameters if needed

## 💡 Tips

- Start with default settings (Small model, 100 epochs)
- Monitor TensorBoard for loss curves and validation metrics
- Use early stopping to avoid overfitting
- Save best checkpoint for deployment
- Test on diverse images to verify generalization

---

**Ready to train?** Just run:
```bash
python train_onion_segmentation.py
```

Good luck! 🚀

