#!/usr/bin/env python3
"""
Test inference on a sample image from a segmentation dataset.
This script loads a trained model and runs inference on a test image.
"""

import random
from pathlib import Path
from PIL import Image
import supervision as sv


def test_inference(
    model_checkpoint,
    dataset_dir,
    model_size="small",
    output_path="test_inference_result.jpg",
    threshold=0.3
):
    """
    Test inference on a random image from the test set.
    
    Args:
        model_checkpoint: Path to trained model checkpoint
        dataset_dir: Path to dataset directory
        model_size: Model size used for training
        output_path: Path to save annotated output image
        threshold: Confidence threshold for detections
    """
    print("=" * 80)
    print("Testing Segmentation Model Inference")
    print("=" * 80)
    
    # Import the appropriate model class
    if model_size.lower() == "nano":
        from rfdetr import RFDETRNano
        model_class = RFDETRNano
    elif model_size.lower() == "small":
        from rfdetr import RFDETRSmall
        model_class = RFDETRSmall
    elif model_size.lower() == "medium":
        from rfdetr import RFDETRMedium
        model_class = RFDETRMedium
    else:
        raise ValueError(f"Invalid model size: {model_size}")
    
    # Check if checkpoint exists
    checkpoint_path = Path(model_checkpoint)
    if not checkpoint_path.exists():
        print(f"\n❌ Model checkpoint not found: {checkpoint_path}")
        print("\nPlease train a model first using train_segmentation.py")
        return
    
    print(f"\nModel checkpoint: {checkpoint_path}")
    print(f"Model size: {model_size}")
    
    # Load class names to determine number of classes
    dataset_path = Path(dataset_dir)
    class_names_file = dataset_path / "class_names.txt"
    if class_names_file.exists():
        with open(class_names_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    else:
        import json
        ann_file = dataset_path / "train" / "_annotations.coco.json"
        with open(ann_file, 'r') as f:
            data = json.load(f)
            categories = sorted(data['categories'], key=lambda x: x['id'])
            class_names = [cat['name'] for cat in categories]
    
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {', '.join(class_names)}")
    
    # Load model
    print("\nLoading model...")
    model = model_class(
        num_classes=num_classes,
        pretrain_weights=str(checkpoint_path),
        segmentation_head=True
    )
    
    # Optimize for inference
    print("Optimizing model for inference...")
    model.optimize_for_inference()
    
    # Find a random test image
    dataset_path = Path(dataset_dir)
    test_images = list((dataset_path / "test").glob("*.png"))
    
    if len(test_images) == 0:
        print(f"\n❌ No test images found in {dataset_path / 'test'}")
        return
    
    image_path = random.choice(test_images)
    print(f"\nTest image: {image_path.name}")
    
    # Load image
    image = Image.open(image_path)
    print(f"Image size: {image.size}")
    
    # Run inference
    print(f"\nRunning inference (threshold={threshold})...")
    detections = model.predict(image, threshold=threshold)
    
    print(f"Detected {len(detections)} instances")
    
    if len(detections) > 0:
        print("\nDetections:")
        for i, (class_id, confidence) in enumerate(zip(detections.class_id, detections.confidence)):
            print(f"  {i+1}. {class_names[class_id]}: {confidence:.3f}")
        
        # Create annotations
        labels = [
            f"{class_names[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]
        
        # Annotate image
        annotated_image = image.copy()
        
        # Add masks if available
        if detections.mask is not None:
            mask_annotator = sv.MaskAnnotator()
            annotated_image = mask_annotator.annotate(annotated_image, detections)
        
        # Add boxes
        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(annotated_image, detections)
        
        # Add labels
        label_annotator = sv.LabelAnnotator()
        annotated_image = label_annotator.annotate(annotated_image, detections, labels)
        
        # Save result
        output_file = Path(output_path)
        annotated_image.save(output_file)
        print(f"\n✅ Annotated image saved to: {output_file.absolute()}")
        
    else:
        print("\nNo detections found (try lowering the threshold)")
        # Still save the original image
        output_file = Path(output_path)
        image.save(output_file)
        print(f"Original image saved to: {output_file.absolute()}")
    
    print("\n" + "=" * 80)
    print("Inference test complete!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test inference on segmentation dataset")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to COCO-format dataset directory"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["nano", "small", "medium"],
        help="Model size"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_inference_result.jpg",
        help="Output path for annotated image"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Confidence threshold"
    )
    
    args = parser.parse_args()
    
    test_inference(
        model_checkpoint=args.checkpoint,
        dataset_dir=args.dataset_dir,
        model_size=args.model_size,
        output_path=args.output,
        threshold=args.threshold
    )

