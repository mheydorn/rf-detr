#!/usr/bin/env python3
"""
Training script for onion defect segmentation using RF-DETR.
"""

import json
from pathlib import Path
from rfdetr import RFDETRSmall
from rfdetr.config import SegmentationTrainConfig


def load_class_names(dataset_dir):
    """Load class names from the dataset."""
    class_names_file = Path(dataset_dir) / "class_names.txt"
    if class_names_file.exists():
        with open(class_names_file, 'r') as f:
            return [line.strip() for line in f.readlines()]
    
    # Fallback: read from COCO annotation file
    ann_file = Path(dataset_dir) / "train" / "_annotations.coco.json"
    if ann_file.exists():
        with open(ann_file, 'r') as f:
            data = json.load(f)
            categories = sorted(data['categories'], key=lambda x: x['id'])
            return [cat['name'] for cat in categories]
    
    raise ValueError(f"Could not find class names in {dataset_dir}")


def train_segmentation_model(
    dataset_dir="datasets/onion_defect_coco",
    output_dir="output/onion_segmentation",
    model_size="small",  # Options: "nano", "small", "medium"
    batch_size=4,
    epochs=100,
    learning_rate=1e-4,
    resolution=None,
    num_workers=2,
    use_pretrained=True,
):
    """
    Train an RF-DETR segmentation model on the onion defect dataset.
    
    Args:
        dataset_dir: Path to the COCO-format dataset directory
        output_dir: Path to save training outputs
        model_size: Model size ("nano", "small", "medium")
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate
        resolution: Input resolution (if None, uses model default)
        num_workers: Number of data loading workers
        use_pretrained: Whether to use pretrained weights
    """
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    
    print("=" * 80)
    print("RF-DETR Onion Defect Segmentation Training")
    print("=" * 80)
    
    # Load class names
    class_names = load_class_names(dataset_dir)
    num_classes = len(class_names)
    
    print(f"\nDataset: {dataset_path}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {', '.join(class_names)}")
    print(f"Output directory: {output_path}")
    print(f"Model size: {model_size}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    
    # Select model based on size
    if model_size.lower() == "nano":
        from rfdetr import RFDETRNano
        model_class = RFDETRNano
        default_resolution = 384
    elif model_size.lower() == "small":
        from rfdetr import RFDETRSmall
        model_class = RFDETRSmall
        default_resolution = 512
    elif model_size.lower() == "medium":
        from rfdetr import RFDETRMedium
        model_class = RFDETRMedium
        default_resolution = 576
    else:
        raise ValueError(f"Invalid model size: {model_size}. Choose from 'nano', 'small', 'medium'")
    
    resolution = resolution or default_resolution
    print(f"Resolution: {resolution}x{resolution}")
    
    # Initialize model
    print("\nInitializing model...")
    model = model_class(
        num_classes=num_classes,
        resolution=resolution,
        pretrain_weights=None if not use_pretrained else model_class.default_config.pretrain_weights,
        segmentation_head=True,
    )
    
    # Configure training
    train_config = SegmentationTrainConfig(
        dataset_dir=str(dataset_path),
        output_dir=str(output_path),
        batch_size=batch_size,
        epochs=epochs,
        lr=learning_rate,
        lr_encoder=learning_rate * 1.5,
        num_workers=num_workers,
        class_names=class_names,
        checkpoint_interval=10,
        use_ema=True,
        ema_decay=0.993,
        early_stopping=True,
        early_stopping_patience=15,
        tensorboard=True,
        wandb=False,
        multi_scale=True,
        expanded_scales=True,
        square_resize_div_64=True,
        # Segmentation-specific parameters
        mask_ce_loss_coef=5.0,
        mask_dice_loss_coef=5.0,
        cls_loss_coef=5.0,
    )
    
    print("\nTraining configuration:")
    print(f"  Batch size: {train_config.batch_size}")
    print(f"  Gradient accumulation steps: {train_config.grad_accum_steps}")
    print(f"  Effective batch size: {train_config.batch_size * train_config.grad_accum_steps}")
    print(f"  Learning rate: {train_config.lr}")
    print(f"  Encoder learning rate: {train_config.lr_encoder}")
    print(f"  EMA: {train_config.use_ema}")
    print(f"  Early stopping: {train_config.early_stopping}")
    print(f"  Multi-scale: {train_config.multi_scale}")
    print(f"  TensorBoard: {train_config.tensorboard}")
    
    # Start training
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    try:
        model.train(train_config)
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print(f"Model saved to: {output_path}")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print(f"Partial results saved to: {output_path}")
        
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train RF-DETR segmentation model on onion defect dataset")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="datasets/onion_defect_coco",
        help="Path to COCO-format dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/onion_segmentation",
        help="Path to save training outputs"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["nano", "small", "medium"],
        help="Model size to use"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Input resolution (defaults to model default)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Don't use pretrained weights"
    )
    
    args = parser.parse_args()
    
    train_segmentation_model(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        model_size=args.model_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        resolution=args.resolution,
        num_workers=args.num_workers,
        use_pretrained=not args.no_pretrained,
    )

