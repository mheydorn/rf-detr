#!/usr/bin/env python3
"""Debug a single image to see what's actually being predicted."""

import sys
import json
from pathlib import Path
from PIL import Image
import torch
import numpy as np

# Import RF-DETR models
from rfdetr import RFDETRSmall

def debug_image(image_path, weights_path, num_classes=2, conf_thres=0.5):
    """Debug a single image to see raw predictions."""
    
    print(f"\n{'='*80}")
    print(f"Debugging Single Image")
    print(f"{'='*80}\n")
    
    print(f"Image: {image_path}")
    print(f"Weights: {weights_path}")
    print(f"Num classes: {num_classes}")
    print(f"Confidence threshold: {conf_thres}")
    print()
    
    # Load model
    print("Loading model...")
    model = RFDETRSmall(
        num_classes=num_classes,
        pretrain_weights=str(weights_path),
        segmentation_head=True,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    
    # Load image
    print("Loading image...")
    image = Image.open(image_path).convert('RGB')
    img_width, img_height = image.size
    print(f"Image size: {img_width}x{img_height}")
    print()
    
    # Run inference
    print("Running inference...")
    detections = model.predict(image, threshold=conf_thres)
    
    print(f"Number of detections: {len(detections)}")
    print()
    
    if len(detections) > 0:
        print(f"{'='*80}")
        print(f"Raw Detection Data")
        print(f"{'='*80}\n")
        
        print(f"Class IDs: {detections.class_id}")
        print(f"Class ID dtype: {detections.class_id.dtype}")
        print(f"Class ID shape: {detections.class_id.shape}")
        print(f"Min class ID: {np.min(detections.class_id)}")
        print(f"Max class ID: {np.max(detections.class_id)}")
        print(f"Unique class IDs: {np.unique(detections.class_id)}")
        print()
        
        print(f"Confidences: {detections.confidence}")
        print(f"Min confidence: {np.min(detections.confidence):.4f}")
        print(f"Max confidence: {np.max(detections.confidence):.4f}")
        print()
        
        print(f"Boxes shape: {detections.xyxy.shape}")
        print()
        
        if detections.mask is not None:
            print(f"Masks shape: {detections.mask.shape}")
            print(f"Mask dtype: {detections.mask.dtype}")
            print(f"Masks range: [{np.min(detections.mask)}, {np.max(detections.mask)}]")
        else:
            print("No masks in detections")
        print()
        
        # Check which would be filtered
        print(f"{'='*80}")
        print(f"Filtering Analysis")
        print(f"{'='*80}\n")
        
        valid_mask = detections.class_id < num_classes
        invalid_mask = ~valid_mask
        
        num_valid = np.sum(valid_mask)
        num_invalid = np.sum(invalid_mask)
        
        print(f"Valid detections (class_id < {num_classes}): {num_valid}")
        print(f"Invalid detections (class_id >= {num_classes}): {num_invalid}")
        
        if num_invalid > 0:
            print(f"\nInvalid class IDs: {detections.class_id[invalid_mask]}")
            print(f"Invalid confidences: {detections.confidence[invalid_mask]}")
        
        # Detailed per-detection info
        print(f"\n{'='*80}")
        print(f"Per-Detection Breakdown")
        print(f"{'='*80}\n")
        
        for i in range(min(10, len(detections))):  # Show first 10
            class_id = int(detections.class_id[i])
            conf = float(detections.confidence[i])
            box = detections.xyxy[i]
            has_mask = detections.mask is not None
            
            valid = "✅ VALID" if class_id < num_classes else "❌ INVALID (out of range)"
            
            print(f"Detection {i}:")
            print(f"  Class ID: {class_id} {valid}")
            print(f"  Confidence: {conf:.4f}")
            print(f"  Box: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
            print(f"  Has mask: {has_mask}")
            print()
        
        if len(detections) > 10:
            print(f"... and {len(detections) - 10} more detections")
    else:
        print("No detections found above threshold")
    
    print(f"{'='*80}\n")

if __name__ == '__main__':
    # Get a problematic image from the diagnosis
    coco_path = Path('latest_out/coco_annotations.json')
    
    if not coco_path.exists():
        print("Error: COCO annotations not found. Please provide image path manually.")
        print(f"Usage: {sys.argv[0]} <image_path> <weights_path> [num_classes] [conf_thres]")
        sys.exit(1)
    
    # Load COCO to find an image with visualization but no annotation
    with open(coco_path, 'r') as f:
        coco_data = json.load(f)
    
    image_id_to_name = {img['id']: img['file_name'] for img in coco_data['images']}
    images_with_annotations = {ann['image_id'] for ann in coco_data['annotations']}
    
    vis_dir = Path('latest_out/visualizations')
    vis_files = list(vis_dir.glob('*_vis.png'))
    
    # Find an image with vis but no annotation
    problematic_image = None
    for vf in vis_files:
        original_name = vf.stem.replace('_vis', '') + '.png'
        # Find this image in COCO
        for img_id, img_name in image_id_to_name.items():
            if img_name == original_name:
                if img_id not in images_with_annotations:
                    problematic_image = img_name
                    break
        if problematic_image:
            break
    
    if not problematic_image:
        print("No problematic images found (all visualizations have annotations)")
        sys.exit(0)
    
    print(f"Found problematic image: {problematic_image}")
    print("This image has a visualization but NO COCO annotation")
    print()
    
    # Find the actual image file
    images_dir = Path('latest_out').parent / 'images'  # Guess
    if not images_dir.exists():
        # Try to find it from the command line args or ask user
        if len(sys.argv) > 1:
            image_path = Path(sys.argv[1])
            weights_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('model.pth')
        else:
            print(f"Cannot find images directory. Please run:")
            print(f"  python {sys.argv[0]} <path/to/{problematic_image}> <weights_path>")
            sys.exit(1)
    else:
        image_path = images_dir / problematic_image
        weights_path = Path('model.pth')  # Default
    
    # Use command line args if provided
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        weights_path = Path(sys.argv[2])
    
    num_classes = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    conf_thres = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        print(f"\nPlease run:")
        print(f"  python {sys.argv[0]} <path/to/{problematic_image}> <weights_path> [num_classes] [conf_thres]")
        sys.exit(1)
    
    if not weights_path.exists():
        print(f"Error: Weights not found at {weights_path}")
        sys.exit(1)
    
    debug_image(image_path, weights_path, num_classes, conf_thres)

