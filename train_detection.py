#!/usr/bin/env python3
"""
Training script for detection models using RF-DETR.
Works with any dataset converted to COCO format using convert_dataset_to_coco.py
"""

import json
from pathlib import Path
from rfdetr import RFDETRSmall


def load_class_names(dataset_dir):
    """Load class names from the dataset."""
    class_names_file = Path(dataset_dir) / "class_names.txt"
    if class_names_file.exists():
        with open(class_names_file, 'r') as f:
            return [line.strip() for line in f.readlines()]
    
    # Try different COCO annotation file locations
    possible_ann_files = [
        Path(dataset_dir) / "train" / "_annotations.coco.json",  # Roboflow format
        Path(dataset_dir) / "train_annotations.json",  # Standard COCO format
    ]
    
    for ann_file in possible_ann_files:
        if ann_file.exists():
            with open(ann_file, 'r') as f:
                data = json.load(f)
                categories = sorted(data['categories'], key=lambda x: x['id'])
                return [cat['name'] for cat in categories]
    
    raise ValueError(f"Could not find class names in {dataset_dir}. Tried: {possible_ann_files}")


def train_detection_model(
    dataset_dir,
    output_dir,
    model_size="small",  # Options: "nano", "small", "medium"
    batch_size=4,
    grad_accum_steps=4,
    epochs=100,
    learning_rate=1e-4,
    resolution=None,
    num_workers=2,
    use_pretrained=True,
    device="cuda",
    resume=None,
):
    """
    Train an RF-DETR detection model on any COCO-format dataset.
    
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
        resume: Path to checkpoint file to resume training from
    """
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    
    print("=" * 80)
    print("RF-DETR Detection Model Training")
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
    if resume:
        print(f"Resuming from checkpoint: {resume}")
    
    # Select model based on size and get default settings
    if model_size.lower() == "nano":
        from rfdetr import RFDETRNano
        model_class = RFDETRNano
        default_resolution = 384
        default_pretrain_weights = "rf-detr-nano.pth"
    elif model_size.lower() == "small":
        from rfdetr import RFDETRSmall
        model_class = RFDETRSmall
        default_resolution = 512
        default_pretrain_weights = "rf-detr-small.pth"
    elif model_size.lower() == "medium":
        from rfdetr import RFDETRMedium
        model_class = RFDETRMedium
        default_resolution = 576
        default_pretrain_weights = "rf-detr-medium.pth"
    else:
        raise ValueError(f"Invalid model size: {model_size}. Choose from 'nano', 'small', 'medium'")
    
    resolution = resolution or default_resolution
    print(f"Resolution: {resolution}x{resolution}")
    
    # Initialize model
    print("\nInitializing model...")
    model = model_class(
        num_classes=num_classes,
        resolution=resolution,
        pretrain_weights=None if not use_pretrained else default_pretrain_weights,
    )
    
    # Configure training parameters as kwargs
    train_kwargs = {
        'dataset_dir': str(dataset_path),
        'output_dir': str(output_path),
        'batch_size': batch_size,
        'epochs': epochs,
        'grad_accum_steps': grad_accum_steps,  # Gradient accumulation steps
        'lr': learning_rate,
        'lr_encoder': learning_rate * 1.5,
        'num_workers': num_workers,
        'class_names': class_names,
        'checkpoint_interval': 10,
        'use_ema': True,
        'ema_decay': 0.993,
        'early_stopping': True,
        'early_stopping_patience': 15,
        'tensorboard': True,
        'wandb': False,
        'multi_scale': True,
        'expanded_scales': True,
        'square_resize_div_64': True,
        'device': device,
        'cls_loss_coef': 5.0,
    }
    
    # Add resume parameter if provided
    if resume:
        train_kwargs['resume'] = resume
    
    print("\nTraining configuration:")
    print(f"  Batch size: {train_kwargs['batch_size']}")
    print(f"  Epochs: {train_kwargs['epochs']}")
    print(f"  Learning rate: {train_kwargs['lr']}")
    print(f"  Encoder learning rate: {train_kwargs['lr_encoder']}")
    print(f"  EMA: {train_kwargs['use_ema']}")
    print(f"  Early stopping: {train_kwargs['early_stopping']}")
    print(f"  Multi-scale: {train_kwargs['multi_scale']}")
    print(f"  TensorBoard: {train_kwargs['tensorboard']}")
    
    # Start training
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    try:
        model.train(**train_kwargs)
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
    
    parser = argparse.ArgumentParser(description="Train RF-DETR detection model on COCO-format dataset")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to COCO-format dataset directory (output from convert_dataset_to_coco.py)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
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
        "--grad-accum-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (effective batch = batch_size * grad_accum_steps)"
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
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training (e.g., 'cuda', 'cuda:0', 'cuda:2', 'cpu')"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from (e.g., output_dir/checkpoint.pth)"
    )
    
    args = parser.parse_args()
    
    train_detection_model(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        model_size=args.model_size,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        epochs=args.epochs,
        learning_rate=args.lr,
        resolution=args.resolution,
        num_workers=args.num_workers,
        use_pretrained=not args.no_pretrained,
        device=args.device,
        resume=args.resume,
    )

