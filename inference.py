#!/usr/bin/env python3
"""RF-DETR Instance Segmentation and Detection Inference

Usage:
    inference.py <images_dir> <weights_path> <output_dir> [options]
    inference.py -h | --help
    inference.py --version

Arguments:
    <images_dir>    Directory containing input images
    <weights_path>  Path to trained RF-DETR weights (.pth file)
    <output_dir>    Directory to save output masks

Options:
    -h --help                        Show this help message and exit
    --version                        Show version and exit
    --conf-threshold=<th>            Confidence threshold for visualizations and txt output [default: 0.5]
                                     Note: COCO JSON (--save-coco) always saves ALL detections with scores,
                                     allowing downstream software to apply its own thresholding
    --model-size=<size>              Model size (nano, small, medium, large) [default: small]
    --device=<device>                Device to run inference on (cpu, cuda, mps, or cuda:0, cuda:1, etc.) [default: cuda]
    --num-classes=<classes>          Number of classes (required if not using class-names file)
    --class-names=<path>             Path to class names file (one class per line)
    --save-txt                       Save detection results as txt files (YOLO format, uses conf-threshold)
    --save-coco                      Save COCO format annotations JSON file (includes ALL detections with scores)
    --visualize                      Save visualization images with overlaid masks/boxes (uses conf-threshold)
    --hide-labels                    Hide class labels on bounding boxes in visualization
    --filter-classes=<ids>           Filter by class indices (comma-separated)
    --no-segmentation                Disable segmentation head (detection only)

Examples:
    # Save all detections to COCO (downstream processing can filter by score)
    inference.py images/ model.pth output/ --num-classes=2 --save-coco
    
    # Visualize only high-confidence detections while saving all to COCO
    inference.py images/ model.pth output/ --num-classes=2 --visualize --save-coco --conf-threshold=0.7
    
    # Using class names file with lower threshold for visualization
    inference.py images/ model.pth output/ --class-names=classes.txt --visualize --conf-threshold=0.3
    
    # Save txt labels for high-confidence detections only
    inference.py images/ model.pth output/ --num-classes=2 --save-txt --conf-threshold=0.6
    
    # Detection only (no segmentation)
    inference.py images/ model.pth output/ --num-classes=80 --no-segmentation --visualize
    
    # GPU inference with txt output
    inference.py images/ model.pth output/ --device=cuda:0 --class-names=classes.txt --save-txt

"""

import os
import sys
import cv2
import json
import numpy as np
import torch
from pathlib import Path
from docopt import docopt
from datetime import datetime
from PIL import Image
from typing import List, Optional, Tuple
import supervision as sv

# Import RF-DETR models
try:
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge
except ImportError as e:
    print(f"Error importing RF-DETR modules: {e}")
    print("Make sure RF-DETR is installed: pip install -e .")
    sys.exit(1)


def load_class_names(class_names_path: Optional[str] = None, num_classes: Optional[int] = None) -> Tuple[List[str], int]:
    """
    Load class names from file or generate default names.
    
    Args:
        class_names_path: Path to class names file
        num_classes: Number of classes if no file provided
        
    Returns:
        Tuple of (class_names_list, num_classes)
    """
    if class_names_path and Path(class_names_path).exists():
        with open(class_names_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
        num_classes = len(class_names)
        print(f"Loaded {num_classes} class names from {class_names_path}")
    elif num_classes:
        class_names = [f"class_{i}" for i in range(num_classes)]
        print(f"Using default class names for {num_classes} classes")
    else:
        raise ValueError("Either --class-names or --num-classes must be provided")
    
    return class_names, num_classes


def create_category_mapping(detections, num_classes: int) -> dict:
    """
    Create mapping from model's category IDs to 0-indexed class indices.
    
    RF-DETR models may output category IDs that don't start at 0 (e.g., 1, 2, 3...)
    depending on how the training dataset was formatted. This function creates
    a mapping to remap those to 0-indexed class indices for consistent handling.
    
    Args:
        detections: Detection results from model
        num_classes: Number of classes expected
        
    Returns:
        Dictionary mapping category_id -> class_index
    """
    if len(detections) == 0:
        # No detections, assume 1-indexed categories (1, 2, ..., num_classes)
        # This is common in COCO format datasets
        return {i + 1: i for i in range(num_classes)}
    
    # Get unique category IDs from detections
    unique_cats = np.unique(detections.class_id)
    
    # Check if already 0-indexed (all category IDs < num_classes)
    if np.all(unique_cats < num_classes):
        # Already 0-indexed (0, 1, 2, ...)
        return {i: i for i in range(num_classes)}
    
    # Check if 1-indexed (all category IDs in range [1, num_classes])
    if np.all((unique_cats >= 1) & (unique_cats <= num_classes)):
        # 1-indexed categories (1, 2, 3...) - shift down by 1
        # Create mapping for ALL expected categories, not just detected ones
        return {i + 1: i for i in range(num_classes)}
    
    # For other cases, try to infer based on the minimum category ID
    min_cat = int(np.min(unique_cats))
    if min_cat > 0:
        # Assume offset by min_cat
        return {i + min_cat: i for i in range(num_classes)}
    
    # Fallback: identity mapping
    return {i: i for i in range(num_classes)}


def mask_to_polygon(mask: np.ndarray, epsilon_factor: float = 0.002) -> Optional[List[float]]:
    """
    Convert a binary mask to polygon coordinates using improved algorithm.
    
    Args:
        mask: Binary mask (H, W), can be boolean or uint8
        epsilon_factor: Polygon approximation factor (relative to perimeter)
        
    Returns:
        List of polygon coordinates [x1, y1, x2, y2, ...] or None if conversion fails
    """
    # Ensure mask is uint8 binary (0 or 255)
    if mask.dtype == bool:
        mask_uint8 = mask.astype(np.uint8) * 255
    else:
        mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
    
    # Skip if mask is empty
    if np.sum(mask_uint8) == 0:
        return None
    
    # Find contours - use CHAIN_APPROX_SIMPLE to reduce points
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Use the largest contour by area
    contour = max(contours, key=cv2.contourArea)
    
    # Check minimum area (at least 1 pixel)
    area = cv2.contourArea(contour)
    if area < 1.0:
        return None
    
    # Ensure contour has enough points
    if len(contour) < 3:
        return None
    
    # Approximate polygon to reduce complexity while maintaining shape
    perimeter = cv2.arcLength(contour, True)
    if perimeter > 0:
        epsilon = epsilon_factor * perimeter
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)
    else:
        approx_contour = contour
    
    # Flatten to list of coordinates [x1, y1, x2, y2, ...]
    polygon = approx_contour.reshape(-1, 2).flatten().tolist()
    
    # COCO format requires at least 6 values (3 points)
    if len(polygon) < 6:
        # Use original contour if approximation reduced too much
        polygon = contour.reshape(-1, 2).flatten().tolist()
        if len(polygon) < 6:
            return None
    
    return polygon


def save_txt_detection(
    class_id: int,
    polygon: List[float],
    confidence: float,
    img_width: int,
    img_height: int,
    save_conf: bool = False
) -> str:
    """
    Create a text line for detection in YOLO segmentation format.
    
    Format: class_id x1 y1 x2 y2 ... [confidence]
    Coordinates are normalized to [0, 1]
    
    Args:
        class_id: Class index
        polygon: Polygon coordinates in pixels [x1, y1, x2, y2, ...]
        confidence: Detection confidence score
        img_width: Image width in pixels
        img_height: Image height in pixels
        save_conf: Whether to include confidence score
        
    Returns:
        Formatted text line
    """
    # Normalize coordinates
    normalized = []
    for i, coord in enumerate(polygon):
        if i % 2 == 0:  # x coordinate
            normalized.append(coord / img_width)
        else:  # y coordinate
            normalized.append(coord / img_height)
    
    # Format line
    parts = [str(class_id)] + [f"{c:.6f}" for c in normalized]
    if save_conf:
        parts.append(f"{confidence:.6f}")
    
    return ' '.join(parts)


def create_coco_annotation(
    annotation_id: int,
    image_id: int,
    class_id: int,
    bbox: np.ndarray,
    confidence: float,
    polygon: Optional[List[float]] = None,
    mask: Optional[np.ndarray] = None
) -> dict:
    """
    Create a COCO format annotation.
    
    Args:
        annotation_id: Unique annotation ID
        image_id: Image ID this annotation belongs to
        class_id: Class index
        bbox: Bounding box [x1, y1, x2, y2] in pixels
        confidence: Detection confidence score
        polygon: Segmentation polygon [x1, y1, x2, y2, ...]
        mask: Binary mask for calculating area (optional)
        
    Returns:
        COCO annotation dictionary
    """
    x1, y1, x2, y2 = bbox
    bbox_w = float(x2 - x1)
    bbox_h = float(y2 - y1)
    
    # Calculate area from mask if available, otherwise use bbox area
    if mask is not None:
        area = int(np.sum(mask > 0))
    else:
        area = bbox_w * bbox_h
    
    annotation = {
        "id": int(annotation_id),
        "image_id": int(image_id),
        "category_id": int(class_id),
        "bbox": [float(x1), float(y1), bbox_w, bbox_h],
        "area": float(area),
        "iscrowd": 0,
        "score": float(confidence)
    }
    
    # Add segmentation if available
    if polygon is not None and len(polygon) >= 6:
        annotation["segmentation"] = [polygon]
    
    return annotation


def save_combined_masks(
    masks: List[np.ndarray],
    class_ids: List[int],
    output_dir: Path,
    img_name: str,
    num_classes: int
):
    """
    Save combined instance mask and per-class masks.
    
    Args:
        masks: List of binary masks
        class_ids: List of class IDs for each mask
        output_dir: Output directory
        img_name: Base image name (without extension)
        num_classes: Total number of classes
    """
    if not masks:
        return
    
    h, w = masks[0].shape
    
    # Save instance mask (each instance gets unique ID)
    # Use uint16 to support up to 65535 instances (uint8 only supports 255)
    instance_mask = np.zeros((h, w), dtype=np.uint16)
    for i, mask in enumerate(masks):
        # Ensure mask has correct shape (resize if needed)
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            mask = mask.astype(bool) if masks[0].dtype == bool else mask
        instance_mask[mask > 0] = i + 1
    
    instance_path = output_dir / 'instances' / f"{img_name}.png"
    instance_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(instance_path), instance_mask)
    
    # Save per-class masks (all instances of each class combined)
    classes_dir = output_dir / 'classes'
    classes_dir.mkdir(parents=True, exist_ok=True)
    
    for class_id in range(num_classes):
        class_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Combine all masks for this class
        for mask, cid in zip(masks, class_ids):
            if cid == class_id:
                # Ensure mask has correct shape (resize if needed)
                if mask.shape != (h, w):
                    mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    mask = mask.astype(bool) if masks[0].dtype == bool else mask
                class_mask[mask > 0] = 255
        
        class_path = classes_dir / f"{img_name}_class{class_id}.png"
        cv2.imwrite(str(class_path), class_mask)


def run_inference(
    images_dir: str,
    weights_path: str,
    output_dir: str,
    model_size: str = 'small',
    conf_threshold: float = 0.5,
    num_classes: Optional[int] = None,
    class_names_path: Optional[str] = None,
    device: str = 'cpu',
    save_txt: bool = False,
    save_coco: bool = False,
    visualize: bool = False,
    hide_labels: bool = False,
    filter_classes: Optional[List[int]] = None,
    segmentation: bool = True,
):
    """
    Run RF-DETR inference on a directory of images.
    
    Args:
        images_dir: Directory containing input images
        weights_path: Path to trained model weights
        output_dir: Directory to save outputs
        model_size: Model size (nano, small, medium, large)
        conf_threshold: Confidence threshold for detections
        num_classes: Number of classes
        class_names_path: Path to class names file
        device: Device to run inference on
        save_txt: Save detection results as txt files
        save_coco: Save COCO format annotations
        visualize: Save visualization images
        hide_labels: Hide class labels on bounding boxes
        filter_classes: Filter by specific class indices
        segmentation: Enable segmentation head
    """
    # Validate inputs
    images_dir = Path(images_dir)
    weights_path = Path(weights_path)
    output_dir = Path(output_dir)
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    # Load class names
    class_names, num_classes = load_class_names(class_names_path, num_classes)
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if save_txt:
        labels_dir = output_dir / 'labels'
        labels_dir.mkdir(exist_ok=True)
    
    if visualize:
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
    
    # Initialize COCO data structure if needed
    coco_data = None
    if save_coco:
        coco_data = {
            "info": {
                "description": "RF-DETR Inference Results",
                "version": "1.0",
                "date_created": datetime.now().isoformat()
            },
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Add categories
        for class_id, class_name in enumerate(class_names):
            coco_data["categories"].append({
                "id": int(class_id),
                "name": class_name,
                "supercategory": "object"
            })
    
    # Check device availability
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            print(f"\nWarning: CUDA device '{device}' requested but CUDA is not available.")
            print("Falling back to CPU.")
            device = 'cpu'
        else:
            if ':' in device:
                gpu_idx = int(device.split(':')[1])
                if gpu_idx >= torch.cuda.device_count():
                    print(f"\nWarning: GPU {gpu_idx} requested but only {torch.cuda.device_count()} GPU(s) available.")
                    print(f"Using cuda:0 instead.")
                    device = 'cuda:0'
            gpu_name = torch.cuda.get_device_name(0 if ':' not in device else int(device.split(':')[1]))
            print(f"\nUsing GPU: {gpu_name}")
    elif device == 'mps':
        if not torch.backends.mps.is_available():
            print(f"\nWarning: MPS device requested but MPS is not available.")
            print("Falling back to CPU.")
            device = 'cpu'
    
    # Initialize model
    print(f"\n{'='*80}")
    print(f"RF-DETR Inference")
    print(f"{'='*80}")
    print(f"Model: {weights_path}")
    print(f"Model size: {model_size}")
    print(f"Segmentation: {segmentation}")
    print(f"Device: {device}")
    print(f"Confidence threshold: {conf_threshold} (for visualizations/txt only)")
    if save_coco:
        print(f"COCO output: ALL detections saved with scores (threshold filtering in downstream tools)")
    print(f"Classes: {num_classes} ({', '.join(class_names[:5])}{'...' if len(class_names) > 5 else ''})")
    print(f"Output directory: {output_dir}")
    
    # Select model class
    model_classes = {
        'nano': RFDETRNano,
        'small': RFDETRSmall,
        'medium': RFDETRMedium,
        'large': RFDETRLarge,
    }
    
    if model_size.lower() not in model_classes:
        raise ValueError(f"Invalid model size: {model_size}. Choose from {list(model_classes.keys())}")
    
    model_class = model_classes[model_size.lower()]
    
    print("\nLoading model...")
    model = model_class(
        num_classes=num_classes,
        pretrain_weights=str(weights_path),
        segmentation_head=segmentation,
        device=device,
    )
    
    # Optimize for inference
    print("Optimizing model for inference...")
    try:
        model.optimize_for_inference()
        print("Model optimization successful")
    except Exception as e:
        print(f"Warning: Could not optimize model: {e}")
    
    # Get list of images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(images_dir.glob(f"*{ext}"))
        image_paths.extend(images_dir.glob(f"*{ext.upper()}"))
    
    image_paths = sorted(set(image_paths))
    
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")
    
    print(f"\nFound {len(image_paths)} images")
    print(f"\n{'='*80}")
    print("Running inference...")
    print(f"{'='*80}\n")
    
    annotation_id = 1
    image_id = 1
    
    total_detections = 0
    
    # Process each image
    for img_path in image_paths:
        print(f"Processing: {img_path.name}")
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        img_width, img_height = image.size
        
        # Run inference with very low threshold to get ALL detections
        # This ensures COCO output contains all predictions with scores
        all_detections = model.predict(image, threshold=0.01)
        
        # Create mapping from model's category IDs to class indices
        # This handles cases where training data used non-zero-indexed categories (e.g., 1, 2 instead of 0, 1)
        category_mapping = create_category_mapping(all_detections, num_classes)
        
        # Filter by class if specified (after mapping)
        if filter_classes is not None and len(all_detections) > 0:
            # Map filter_classes to original category IDs for filtering
            reverse_mapping = {v: k for k, v in category_mapping.items()}
            filter_cats = [reverse_mapping.get(c, c) for c in filter_classes]
            mask = np.isin(all_detections.class_id, filter_cats)
            all_detections = all_detections[mask]
        
        # Show category mapping if detections present
        if len(all_detections) > 0 and image_id == 1:  # Only show once
            unique_cats = np.unique(all_detections.class_id)
            print(f"  Category ID mapping: {', '.join([f'{cat}->{category_mapping.get(cat, cat)}' for cat in sorted(unique_cats)])}")
        
        # Create filtered detections for visualization and txt output
        if len(all_detections) > 0:
            conf_mask = all_detections.confidence >= conf_threshold
            filtered_detections = all_detections[conf_mask]
        else:
            filtered_detections = all_detections
        
        num_all_detections = len(all_detections)
        num_filtered_detections = len(filtered_detections)
        total_detections += num_all_detections
        
        print(f"  All detections: {num_all_detections}")
        if num_filtered_detections < num_all_detections:
            print(f"  Above threshold ({conf_threshold}): {num_filtered_detections}")
        
        # Add image info to COCO
        if save_coco:
            coco_data["images"].append({
                "id": image_id,
                "file_name": img_path.name,
                "width": int(img_width),
                "height": int(img_height)
            })
        
        # Process ALL detections for COCO output and mask saving
        txt_lines = []
        masks_list = []
        class_ids_list = []
        
        # Process all detections for COCO output
        if num_all_detections > 0:
            all_boxes = all_detections.xyxy
            all_confidences = all_detections.confidence
            all_class_ids_raw = all_detections.class_id  # Raw category IDs from model
            all_masks = all_detections.mask if segmentation and all_detections.mask is not None else None
            
            if segmentation and all_masks is not None:
                print(f"  Masks: {all_masks.shape}")
            
            # Process each detection for COCO and mask output
            for i in range(num_all_detections):
                category_id = int(all_class_ids_raw[i])  # Original category ID from model
                class_id = category_mapping.get(category_id, category_id)  # Mapped to class index
                confidence = float(all_confidences[i])
                box = all_boxes[i]
                mask = all_masks[i] if all_masks is not None else None
                
                # Store mask for combined output (using mapped class_id)
                if mask is not None:
                    masks_list.append(mask)
                    class_ids_list.append(class_id)
                
                # Convert mask to polygon
                polygon = None
                if mask is not None:
                    polygon = mask_to_polygon(mask)
                    if polygon is None and save_coco:
                        print(f"    Warning: Failed to convert mask {i} to polygon")
                
                # Add COCO annotation (always saves all detections with scores)
                if save_coco:
                    annotation = create_coco_annotation(
                        annotation_id, image_id, class_id,
                        box, confidence, polygon, mask
                    )
                    coco_data["annotations"].append(annotation)
                    annotation_id += 1
        
        # Process filtered detections for txt output (only above threshold)
        if save_txt and num_filtered_detections > 0:
            filt_boxes = filtered_detections.xyxy
            filt_confidences = filtered_detections.confidence
            filt_class_ids_raw = filtered_detections.class_id
            filt_masks = filtered_detections.mask if segmentation and filtered_detections.mask is not None else None
            
            for i in range(num_filtered_detections):
                category_id = int(filt_class_ids_raw[i])
                class_id = category_mapping.get(category_id, category_id)
                confidence = float(filt_confidences[i])
                mask = filt_masks[i] if filt_masks is not None else None
                
                # Convert mask to polygon
                if mask is not None:
                    polygon = mask_to_polygon(mask)
                    if polygon is not None:
                        line = save_txt_detection(
                            class_id, polygon, confidence,
                            img_width, img_height, save_conf=True
                        )
                        txt_lines.append(line)
        
        # Save combined masks
        if segmentation and masks_list:
            save_combined_masks(
                masks_list, class_ids_list, output_dir,
                img_path.stem, num_classes
            )
        
        # Save txt file
        if save_txt and txt_lines:
            txt_path = labels_dir / f"{img_path.stem}.txt"
            with open(txt_path, 'w') as f:
                f.write('\n'.join(txt_lines))
        
        # Save visualization (only shows detections above threshold)
        if visualize and num_filtered_detections > 0:
            image_np = np.array(image)
            
            # Get filtered detection data for visualization
            filt_class_ids_raw = filtered_detections.class_id
            filt_confidences = filtered_detections.confidence
            filt_masks = filtered_detections.mask if segmentation and filtered_detections.mask is not None else None
            
            # Create annotated image
            if segmentation and filt_masks is not None:
                mask_annotator = sv.MaskAnnotator()
                image_np = mask_annotator.annotate(image_np, filtered_detections)
            
            box_annotator = sv.BoxAnnotator()
            image_np = box_annotator.annotate(image_np, filtered_detections)
            
            if not hide_labels:
                labels = []
                for category_id, conf in zip(filt_class_ids_raw, filt_confidences):
                    # Map category ID to class index for label lookup
                    class_id = category_mapping.get(category_id, category_id)
                    if 0 <= class_id < len(class_names):
                        labels.append(f"{class_names[class_id]} {conf:.2f}")
                    else:
                        labels.append(f"class_{class_id} {conf:.2f}")
                
                label_annotator = sv.LabelAnnotator()
                image_np = label_annotator.annotate(image_np, filtered_detections, labels)
            
            vis_path = vis_dir / f"{img_path.stem}_vis.png"
            Image.fromarray(image_np).save(str(vis_path))
        
        image_id += 1
    
    # Save COCO JSON
    if save_coco:
        coco_json_path = output_dir / 'coco_annotations.json'
        with open(coco_json_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        print(f"\nCOCO annotations saved: {coco_json_path}")
        print(f"  Total annotations: {len(coco_data['annotations'])} (ALL detections with scores)")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Inference completed!")
    print(f"{'='*80}")
    print(f"Processed: {len(image_paths)} images")
    print(f"Total detections: {total_detections} (all confidence levels)")
    print(f"Results saved to: {output_dir}")
    
    if segmentation:
        print(f"\nMasks (all detections):")
        print(f"  Instance masks: {output_dir / 'instances'}")
        print(f"  Per-class masks: {output_dir / 'classes'}")
    
    if save_txt:
        print(f"\nLabels (YOLO format, conf >= {conf_threshold}): {labels_dir}")
    
    if save_coco:
        print(f"\nCOCO Annotations (all detections): {coco_json_path}")
        print(f"  Use 'score' field to filter by confidence in downstream tools")
    
    if visualize:
        print(f"\nVisualizations (conf >= {conf_threshold}): {vis_dir}")
    
    print(f"{'='*80}\n")


def main():
    """Main entry point."""
    args = docopt(__doc__, version='RF-DETR Inference 2.0')
    
    # Parse arguments
    images_dir = args['<images_dir>']
    weights_path = args['<weights_path>']
    output_dir = args['<output_dir>']
    
    # Parse options
    conf_threshold = float(args['--conf-threshold'])
    model_size = args['--model-size']
    device = args['--device']
    num_classes = int(args['--num-classes']) if args['--num-classes'] else None
    class_names_path = args['--class-names']
    save_txt = args['--save-txt']
    save_coco = args['--save-coco']
    visualize = args['--visualize']
    hide_labels = args['--hide-labels']
    segmentation = not args['--no-segmentation']
    
    # Parse class filter
    filter_classes = None
    if args['--filter-classes']:
        filter_classes = [int(x) for x in args['--filter-classes'].split(',')]
    
    run_inference(
        images_dir=images_dir,
        weights_path=weights_path,
        output_dir=output_dir,
        model_size=model_size,
        conf_threshold=conf_threshold,
        num_classes=num_classes,
        class_names_path=class_names_path,
        device=device,
        save_txt=save_txt,
        save_coco=save_coco,
        visualize=visualize,
        hide_labels=hide_labels,
        filter_classes=filter_classes,
        segmentation=segmentation,
    )


if __name__ == '__main__':
    main()
