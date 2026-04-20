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
    --coco-threshold=<th>            Confidence threshold specifically for COCO labels (separate from conf-threshold) [default: 0]
    --model-size=<size>              Model size (nano, small, medium, large) [default: small]
    --device=<device>                Device to run inference on (cpu, cuda, mps, or cuda:0, cuda:1, etc.) [default: cuda]
    --num-classes=<classes>          Number of classes (only required if checkpoint doesn't contain class names)
    --sample-size=<n>                If specified, only process first N images (useful for testing)
    --save-txt                       Save detection results as txt files (YOLO format, uses conf-threshold)
    --save-coco                      Save COCO format annotations JSON file (includes ALL detections with scores)
    --save-masks                     Save per-class masks (combined masks for each class, segmentation only)
    --save-instances                 Save instance masks (each instance gets unique ID, segmentation only)
    --visualize                      Save visualization images with overlaid masks/boxes (uses conf-threshold)
    --hide-labels                    Hide class labels on bounding boxes in visualization
    --filter-classes=<ids>           Filter by class indices (comma-separated)
    --no-segmentation                Disable segmentation head (detection only)
    --min=<size>                     Minimum detection size in square pixels (exclude smaller detections)
    --max=<size>                     Maximum detection size in square pixels (exclude larger detections)
    --all-fm                         Convert all predictions to "fm" class regardless of original class
    --additional-weights=<path>      Additional weights file to run inference with (detections pooled together)

Note: Image resolution is automatically loaded from the checkpoint file.

Note: At least one output option (--save-txt, --save-coco, --save-masks, --save-instances, --visualize) must be specified

Examples:
    # Save all detections to COCO (resolution and class names from checkpoint)
    inference.py images/ model.pth output/ --save-coco
    
    # Visualize only high-confidence detections while saving all to COCO
    inference.py images/ model.pth output/ --visualize --save-coco --conf-threshold=0.7
    
    # Save instance masks and per-class masks
    inference.py images/ model.pth output/ --save-instances --save-masks
    
    # Save txt labels for high-confidence detections only
    inference.py images/ model.pth output/ --save-txt --conf-threshold=0.6
    
    # Detection only (no segmentation) - save COCO and visualizations
    inference.py images/ model.pth output/ --no-segmentation --visualize --save-coco
    
    # GPU inference with multiple outputs
    inference.py images/ model.pth output/ --device=cuda:0 --save-txt --save-coco

"""

import os
import sys
import cv2
import json
import numpy as np
import torch
import time
from pathlib import Path
from docopt import docopt
from datetime import datetime
from PIL import Image
from typing import List, Optional, Tuple, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
import supervision as sv
from tqdm import tqdm

# Import RF-DETR models
try:
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge
except ImportError as e:
    print(f"Error importing RF-DETR modules: {e}")
    print("Make sure RF-DETR is installed: pip install -e .")
    sys.exit(1)


def load_image(img_path: Path) -> Tuple[Path, Optional[Image.Image], Optional[Tuple[int, int]], Optional[str]]:
    """
    Load a single image from disk.
    
    Args:
        img_path: Path to image file
        
    Returns:
        Tuple of (path, PIL Image, (width, height), error_message)
        If loading fails, image and size will be None and error_message will be set
    """
    try:
        image = Image.open(img_path).convert('RGB')
        return (img_path, image, image.size, None)
    except Exception as e:
        return (img_path, None, None, str(e))


def prefetch_images(
    image_paths: List[Path],
    num_workers: int = 4,
    prefetch_factor: int = 2
) -> Iterator[Tuple[Path, Optional[Image.Image], Optional[Tuple[int, int]], Optional[str]]]:
    """
    Generator that prefetches images in background threads.
    
    Uses a thread pool to load images ahead of consumption, overlapping
    I/O with compute. Maintains order of images.
    
    Args:
        image_paths: List of image paths to load
        num_workers: Number of worker threads for loading
        prefetch_factor: How many images to prefetch per worker
        
    Yields:
        Tuple of (path, PIL Image, (width, height), error_message)
    """
    max_prefetch = num_workers * prefetch_factor
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit initial batch of tasks
        futures = {}
        path_iter = iter(enumerate(image_paths))
        
        # Fill the prefetch buffer
        for _ in range(min(max_prefetch, len(image_paths))):
            try:
                idx, path = next(path_iter)
                future = executor.submit(load_image, path)
                futures[future] = idx
            except StopIteration:
                break
        
        # Results buffer to maintain order
        results = {}
        next_idx = 0
        
        while futures or next_idx < len(image_paths):
            # Wait for any future to complete
            if futures:
                done_futures = []
                for future in as_completed(futures):
                    done_futures.append(future)
                    break  # Just get one at a time to maintain flow
                
                for future in done_futures:
                    idx = futures.pop(future)
                    results[idx] = future.result()
                    
                    # Submit next task if available
                    try:
                        new_idx, path = next(path_iter)
                        new_future = executor.submit(load_image, path)
                        futures[new_future] = new_idx
                    except StopIteration:
                        pass
            
            # Yield results in order
            while next_idx in results:
                yield results.pop(next_idx)
                next_idx += 1


def detect_model_has_segmentation(weights_path: str, checkpoint: Optional[dict] = None) -> bool:
    """
    Detect if a model checkpoint has segmentation capability.
    
    Checks the checkpoint args for segmentation_head flag. This is the most
    reliable indicator of whether a model was trained with segmentation.
    
    Args:
        weights_path: Path to checkpoint file
        checkpoint: Pre-loaded checkpoint dict (optional, avoids redundant loading)
        
    Returns:
        True if model has segmentation head, False if detection-only
    """
    try:
        if checkpoint is None:
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        
        # Check args for segmentation flag (most reliable)
        if 'args' in checkpoint:
            args = checkpoint['args']
            if hasattr(args, 'segmentation_head'):
                has_seg = args.segmentation_head
                return has_seg
        
        # Fallback: check if model state dict contains segmentation head parameters
        # Look for decoder.mask_head or similar segmentation-specific layers
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            # Look for decoder.mask_head or segmentation head parameters
            seg_keys = [k for k in state_dict.keys() if 'mask_head' in k.lower() or 
                       ('decoder' in k and 'mask' in k.lower() and 'token' not in k.lower())]
            if seg_keys:
                return True
        
        # Default to True for backward compatibility (assume older models have segmentation)
        return True
        
    except Exception as e:
        print(f"Warning: Could not determine if model has segmentation head: {e}")
        print("Assuming model has segmentation capability")
        return True


def load_class_names(class_names_path: Optional[str] = None, num_classes: Optional[int] = None) -> Tuple[List[str], int]:
    """
    Load class names from file or generate default names.
    
    Supports two file formats:
    1. Text file with one class name per line
    2. JSON file with {"class_name": index, ...} pairs
    
    Args:
        class_names_path: Path to class names file
        num_classes: Number of classes if no file provided
        
    Returns:
        Tuple of (class_names_list, num_classes)
    """
    if class_names_path and Path(class_names_path).exists():
        path = Path(class_names_path)
        
        # Try to load as JSON if it has .json extension or if the content is JSON
        if path.suffix.lower() == '.json':
            with open(class_names_path, 'r') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, dict):
                        # Check if it's a class_name: index dictionary
                        if all(isinstance(k, str) and isinstance(v, int) for k, v in data.items()):
                            # Sort by index to create ordered list
                            max_idx = max(data.values())
                            class_names = [''] * (max_idx + 1)
                            for class_name, idx in data.items():
                                class_names[idx] = class_name
                            # Check for gaps in indices (would cause misalignment)
                            if any(name == '' for name in class_names):
                                missing_indices = [i for i, name in enumerate(class_names) if name == '']
                                raise ValueError(
                                    f"JSON class mapping has gaps at indices {missing_indices}. "
                                    f"Class indices must be contiguous (0, 1, 2, ..., n-1)."
                                )
                            
                            # Check if first class is "background" - RF-DETR doesn't use background class
                            # Remove it and shift all other classes down by 1
                            if class_names[0].lower() == 'background':
                                print(f"Removing 'background' at index 0 - RF-DETR doesn't predict background")
                                class_names = class_names[1:]  # Remove background
                                print(f"Shifted class indices down by 1")
                            
                            num_classes = len(class_names)
                            print(f"Loaded {num_classes} class names from JSON file")
                            return class_names, num_classes
                except json.JSONDecodeError:
                    pass  # Fall through to text file handling
        
        # Load as text file (one class per line)
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


def merge_detections(detections1: sv.Detections, detections2: sv.Detections) -> sv.Detections:
    """
    Merge two supervision Detections objects into one.
    
    Args:
        detections1: First set of detections
        detections2: Second set of detections
        
    Returns:
        Merged Detections object containing all detections from both inputs
    """
    # Handle empty cases
    if len(detections1) == 0:
        return detections2
    if len(detections2) == 0:
        return detections1
    
    # Merge bounding boxes
    merged_xyxy = np.concatenate([detections1.xyxy, detections2.xyxy], axis=0)
    
    # Merge confidence scores
    merged_confidence = np.concatenate([detections1.confidence, detections2.confidence], axis=0)
    
    # Merge class IDs
    merged_class_id = np.concatenate([detections1.class_id, detections2.class_id], axis=0)
    
    # Merge masks if present
    merged_mask = None
    if detections1.mask is not None and detections2.mask is not None:
        merged_mask = np.concatenate([detections1.mask, detections2.mask], axis=0)
    elif detections1.mask is not None:
        merged_mask = detections1.mask
    elif detections2.mask is not None:
        merged_mask = detections2.mask
    
    # Create merged detections
    return sv.Detections(
        xyxy=merged_xyxy,
        confidence=merged_confidence,
        class_id=merged_class_id,
        mask=merged_mask
    )


def save_combined_masks(
    masks: List[np.ndarray],
    class_ids: List[int],
    output_dir: Path,
    img_name: str,
    num_classes: int,
    save_instances: bool = False,
    save_classes: bool = False
):
    """
    Save combined instance mask and/or per-class masks.
    
    Args:
        masks: List of binary masks
        class_ids: List of class IDs for each mask
        output_dir: Output directory
        img_name: Base image name (without extension)
        num_classes: Total number of classes
        save_instances: Whether to save instance masks
        save_classes: Whether to save per-class masks
    """
    if not masks or (not save_instances and not save_classes):
        return
    
    h, w = masks[0].shape
    
    # Save instance mask (each instance gets unique ID)
    if save_instances:
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
    if save_classes:
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
    coco_threshold: float = 0,
    num_classes: Optional[int] = None,
    device: str = 'cpu',
    save_txt: bool = False,
    save_coco: bool = False,
    save_masks: bool = False,
    save_instances: bool = False,
    visualize: bool = False,
    hide_labels: bool = False,
    filter_classes: Optional[List[int]] = None,
    segmentation: Optional[bool] = None,
    sample_size: Optional[int] = None,
    min_size: Optional[float] = None,
    max_size: Optional[float] = None,
    all_fm: bool = False,
    additional_weights: Optional[str] = None,
):
    """
    Run RF-DETR inference on a directory of images.
    
    Image resolution is automatically loaded from checkpoint.
    
    Args:
        images_dir: Directory containing input images
        weights_path: Path to trained model weights (must contain resolution)
        output_dir: Directory to save outputs
        model_size: Model size (nano, small, medium, large)
        conf_threshold: Confidence threshold for detections
        coco_threshold: Confidence threshold specifically for COCO labels (separate from conf_threshold)
        num_classes: Number of classes (only needed if not in checkpoint)
        device: Device to run inference on
        save_txt: Save detection results as txt files
        save_coco: Save COCO format annotations
        save_masks: Save per-class masks
        save_instances: Save instance masks
        visualize: Save visualization images
        hide_labels: Hide class labels on bounding boxes
        filter_classes: Filter by specific class indices
        segmentation: Enable segmentation head (auto-detected if None)
        sample_size: If specified, only process first N images
        min_size: Minimum detection size in square pixels (exclude smaller)
        max_size: Maximum detection size in square pixels (exclude larger)
        all_fm: Convert all predictions to "fm" class
        additional_weights: Path to additional weights file (detections pooled with primary)
    """
    # Validate that at least one output is specified
    if not any([save_txt, save_coco, save_masks, save_instances, visualize]):
        raise ValueError(
            "No output format specified! You must specify at least one of:\n"
            "  --save-txt          Save YOLO format text files\n"
            "  --save-coco         Save COCO JSON annotations\n"
            "  --save-masks        Save per-class mask images\n"
            "  --save-instances    Save instance mask images\n"
            "  --visualize         Save visualization images\n"
        )
    
    # Validate inputs
    images_dir = Path(images_dir)
    weights_path = Path(weights_path)
    output_dir = Path(output_dir)
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    # Validate additional weights if provided
    additional_weights_path = None
    if additional_weights:
        additional_weights_path = Path(additional_weights)
        if not additional_weights_path.exists():
            raise FileNotFoundError(f"Additional weights file not found: {additional_weights_path}")
    
    # Load checkpoint once and reuse for all metadata extraction
    checkpoint = None
    checkpoint_class_names = None
    checkpoint_resolution = None
    try:
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        if 'args' in checkpoint:
            if hasattr(checkpoint['args'], 'class_names'):
                checkpoint_class_names = checkpoint['args'].class_names
                print(f"Found class names in checkpoint: {len(checkpoint_class_names)} classes")
            if hasattr(checkpoint['args'], 'resolution'):
                checkpoint_resolution = checkpoint['args'].resolution
                print(f"Found resolution in checkpoint: {checkpoint_resolution}")
    except Exception as e:
        print(f"Warning: Could not load data from checkpoint: {e}")
    
    # Auto-detect segmentation capability from checkpoint if not explicitly specified
    if segmentation is None:
        print("Auto-detecting segmentation capability from checkpoint...")
        has_segmentation = detect_model_has_segmentation(str(weights_path), checkpoint=checkpoint)
        segmentation = has_segmentation
        print(f"Model type: {'Segmentation' if segmentation else 'Detection-only'}")
    
    # Validate that segmentation-only options are not used with detection-only models
    if not segmentation:
        if save_masks:
            raise ValueError(
                "ERROR: --save-masks is only available for segmentation models!\n"
                f"The checkpoint '{weights_path.name}' is a detection-only model.\n"
                "Remove --save-masks to proceed with detection-only inference."
            )
        if save_instances:
            raise ValueError(
                "ERROR: --save-instances is only available for segmentation models!\n"
                f"The checkpoint '{weights_path.name}' is a detection-only model.\n"
                "Remove --save-instances to proceed with detection-only inference."
            )
    
    # Use checkpoint class names or generate default names
    if checkpoint_class_names is not None:
        class_names = checkpoint_class_names
        num_classes = len(class_names)
        print(f"Using class names from model checkpoint")
    elif num_classes is not None:
        class_names = [f"class_{i}" for i in range(num_classes)]
        num_classes = num_classes
        print(f"Using default class names for {num_classes} classes")
    else:
        raise ValueError(
            "Could not determine class names!\n"
            "  - Checkpoint doesn't contain class names\n"
            "  - You must provide --num-classes\n"
        )
    
    # Use checkpoint resolution (required)
    if checkpoint_resolution is None:
        raise ValueError(
            "Could not determine image resolution!\n"
            "  - Checkpoint doesn't contain resolution\n"
        )
    imgz = checkpoint_resolution
    print(f"Using resolution from checkpoint: {imgz}")
    
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
    if additional_weights_path:
        print(f"Additional model: {additional_weights_path}")
    print(f"Model size: {model_size}")
    print(f"Segmentation: {segmentation}")
    print(f"Device: {device}")
    print(f"Confidence threshold: {conf_threshold} (for visualizations/txt only)")
    if save_coco:
        print(f"COCO threshold: {coco_threshold} (filters COCO labels before saving)")
        if coco_threshold == 0:
            print(f"COCO output: ALL detections saved with scores (no threshold filtering)")
        else:
            print(f"COCO output: detections >= {coco_threshold} threshold saved")
    print(f"Classes: {num_classes} ({', '.join(class_names[:5])}{'...' if len(class_names) > 5 else ''})")
    print(f"Output directory: {output_dir}")
    if min_size is not None:
        print(f"Min size filter: {min_size} sq pixels")
    if max_size is not None:
        print(f"Max size filter: {max_size} sq pixels")
    
    # Find "fm" class index if --all-fm is specified
    fm_class_index = None
    if all_fm:
        # Look for "fm" class in class names (case-insensitive)
        for idx, name in enumerate(class_names):
            if name.lower() == 'fm':
                fm_class_index = idx
                break
        if fm_class_index is None:
            raise ValueError(
                "Cannot use --all-fm: 'fm' class not found in class names!\n"
                f"Available classes: {class_names}"
            )
        print(f"All-FM mode: Converting all detections to class 'fm' (index {fm_class_index})")
    
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
    t_load_start = time.perf_counter()
    print(f"Using input resolution: {imgz}x{imgz}")
    model = model_class(
        num_classes=num_classes,
        pretrain_weights=str(weights_path),
        segmentation_head=segmentation,
        device=device,
        resolution=imgz,
    )
    t_load_end = time.perf_counter()
    print(f"Model loaded in {t_load_end - t_load_start:.3f}s")
    
    # Optimize for inference
    print("Optimizing model for inference...")
    t_opt_start = time.perf_counter()
    try:
        model.optimize_for_inference()
        t_opt_end = time.perf_counter()
        print(f"Model optimization successful ({t_opt_end - t_opt_start:.3f}s)")
    except Exception as e:
        print(f"Warning: Could not optimize model: {e}")
    
    # Load additional model if specified
    additional_model = None
    if additional_weights_path:
        print(f"\nLoading additional model from {additional_weights_path}...")
        t_load_add_start = time.perf_counter()
        
        # Get resolution from additional checkpoint (load once, reuse for segmentation detection)
        add_checkpoint = None
        add_resolution = None
        try:
            add_checkpoint = torch.load(additional_weights_path, map_location='cpu', weights_only=False)
            if 'args' in add_checkpoint:
                if hasattr(add_checkpoint['args'], 'resolution'):
                    add_resolution = add_checkpoint['args'].resolution
            if add_resolution is None:
                add_resolution = imgz  # Use primary model's resolution as fallback
                print(f"  Using primary model resolution: {add_resolution}")
            else:
                print(f"  Resolution from checkpoint: {add_resolution}")
        except Exception as e:
            print(f"  Warning: Could not load resolution from additional checkpoint: {e}")
            add_resolution = imgz
        
        # Auto-detect segmentation for additional model (reuse loaded checkpoint)
        add_has_seg = detect_model_has_segmentation(str(additional_weights_path), checkpoint=add_checkpoint)
        add_segmentation = add_has_seg if segmentation is None else segmentation
        print(f"  Segmentation: {add_segmentation}")
        
        additional_model = model_class(
            num_classes=num_classes,
            pretrain_weights=str(additional_weights_path),
            segmentation_head=add_segmentation,
            device=device,
            resolution=add_resolution,
        )
        t_load_add_end = time.perf_counter()
        print(f"  Additional model loaded in {t_load_add_end - t_load_add_start:.3f}s")
        
        # Optimize additional model
        try:
            additional_model.optimize_for_inference()
            print(f"  Additional model optimization successful")
        except Exception as e:
            print(f"  Warning: Could not optimize additional model: {e}")
    
    # Get list of images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(images_dir.glob(f"*{ext}"))
        image_paths.extend(images_dir.glob(f"*{ext.upper()}"))
    
    image_paths = sorted(set(image_paths))
    
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")
    
    # Apply sample size if specified
    total_images = len(image_paths)
    if sample_size is not None and sample_size > 0:
        image_paths = image_paths[:sample_size]
        print(f"\nFound {total_images} images, using sample size: {len(image_paths)} images")
    else:
        print(f"\nFound {len(image_paths)} images")
    print(f"\n{'='*80}")
    print("Running inference...")
    print(f"{'='*80}\n")

    annotation_id = 1
    image_id = 1

    total_detections = 0

    # Timing statistics
    image_load_times = []
    inference_times = []
    postprocess_times = []
    total_times = []

    # Pre-create annotators once (reused for all images) - significant speedup
    if visualize:
        mask_annotator = sv.MaskAnnotator() if segmentation else None
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator() if not hide_labels else None

    t_total_start = time.perf_counter()

    # Process each image with progress bar (using prefetch for faster loading)
    print(f"Using prefetch with 4 workers to overlap image loading with inference")
    with tqdm(total=len(image_paths), desc="Processing images", unit="img") as pbar:
        for img_path, image, img_size, load_error in prefetch_images(image_paths, num_workers=4, prefetch_factor=2):
            t_img_start = time.perf_counter()

            # Check if image loaded successfully
            if image is None:
                pbar.set_postfix_str(f"Skipped {img_path.name}: {load_error}")
                pbar.update(1)
                continue
            
            img_width, img_height = img_size
            # Note: image load time is now overlapped with previous inference
            # We track a nominal time for statistics but actual wall time is hidden
            image_load_times.append(0.0)  # Overlapped, so effective time is ~0

            # Run inference with very low threshold to get ALL detections
            # This ensures COCO output contains all predictions with scores
            t_infer_start = time.perf_counter()
            all_detections = model.predict(image, threshold=0.01)
            
            # Run inference with additional model and merge detections
            if additional_model is not None:
                additional_detections = additional_model.predict(image, threshold=0.01)
                all_detections = merge_detections(all_detections, additional_detections)
            
            t_infer_end = time.perf_counter()
            inference_times.append(t_infer_end - t_infer_start)

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

            # Filter by size (area in square pixels)
            if len(all_detections) > 0 and (min_size is not None or max_size is not None):
                boxes = all_detections.xyxy
                # Calculate area: (x2 - x1) * (y2 - y1)
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                size_mask = np.ones(len(all_detections), dtype=bool)
                if min_size is not None:
                    size_mask &= (areas >= min_size)
                if max_size is not None:
                    size_mask &= (areas <= max_size)
                all_detections = all_detections[size_mask]

            # Show category mapping if detections present
            if len(all_detections) > 0 and image_id == 1:  # Only show once
                unique_cats = np.unique(all_detections.class_id)
                print(f"Category ID mapping: {', '.join([f'{cat}->{category_mapping.get(cat, cat)}' for cat in sorted(unique_cats)])}")

            # Create filtered detections for visualization and txt output
            if len(all_detections) > 0:
                conf_mask = all_detections.confidence >= conf_threshold
                filtered_detections = all_detections[conf_mask]
            else:
                filtered_detections = all_detections

            num_all_detections = len(all_detections)
            num_filtered_detections = len(filtered_detections)
            total_detections += num_all_detections

            # Add image info to COCO
            if save_coco:
                coco_data["images"].append({
                    "id": image_id,
                    "file_name": img_path.name,
                    "width": int(img_width),
                    "height": int(img_height)
                })

            # Process ALL detections for COCO output and mask saving
            t_postprocess_start = time.perf_counter()
            txt_lines = []
            masks_list = []
            class_ids_list = []

            # Process all detections for COCO output
            if num_all_detections > 0:
                all_boxes = all_detections.xyxy
                all_confidences = all_detections.confidence
                all_class_ids_raw = all_detections.class_id  # Raw category IDs from model
                all_masks = all_detections.mask if segmentation and all_detections.mask is not None else None

                # Process each detection for COCO and mask output
                for i in range(num_all_detections):
                    category_id = int(all_class_ids_raw[i])  # Original category ID from model
                    class_id = category_mapping.get(category_id, category_id)  # Mapped to class index
                    # Override class to "fm" if --all-fm is specified
                    if all_fm and fm_class_index is not None:
                        class_id = fm_class_index
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

                    # Add COCO annotation (apply coco_threshold filter)
                    if save_coco and confidence >= coco_threshold:
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
                    # Override class to "fm" if --all-fm is specified
                    if all_fm and fm_class_index is not None:
                        class_id = fm_class_index
                    confidence = float(filt_confidences[i])
                    mask = filt_masks[i] if filt_masks is not None else None
                    box = filt_boxes[i]

                    # For segmentation models, try to use polygon from mask
                    # For detection-only models, convert bbox to YOLO polygon format
                    if mask is not None:
                        # Segmentation model - use mask polygon
                        polygon = mask_to_polygon(mask)
                        if polygon is not None:
                            line = save_txt_detection(
                                class_id, polygon, confidence,
                                img_width, img_height, save_conf=True
                            )
                            txt_lines.append(line)
                    elif not segmentation:
                        # Detection-only model - convert bbox to polygon format
                        x1, y1, x2, y2 = box
                        # Create polygon as bbox corners in normalized coordinates
                        polygon = [x1, y1, x2, y1, x2, y2, x1, y2]
                        line = save_txt_detection(
                            class_id, polygon, confidence,
                            img_width, img_height, save_conf=True
                        )
                        txt_lines.append(line)

            # Save combined masks
            if segmentation and masks_list and (save_masks or save_instances):
                save_combined_masks(
                    masks_list, class_ids_list, output_dir,
                    img_path.stem, num_classes,
                    save_instances=save_instances,
                    save_classes=save_masks
                )

            # Save txt file
            if save_txt and txt_lines:
                txt_path = labels_dir / f"{img_path.stem}.txt"
                with open(txt_path, 'w') as f:
                    f.write('\n'.join(txt_lines))

            # Save visualization (saves all images, annotates those with detections above threshold)
            if visualize:
                image_np = np.array(image)

                # Only annotate if there are detections above threshold
                if num_filtered_detections > 0:
                    # Get filtered detection data for visualization
                    filt_class_ids_raw = filtered_detections.class_id
                    filt_confidences = filtered_detections.confidence
                    filt_masks = filtered_detections.mask if segmentation and filtered_detections.mask is not None else None

                    # Annotate using pre-created annotators (faster than creating new ones each time)
                    if segmentation and filt_masks is not None and mask_annotator is not None:
                        image_np = mask_annotator.annotate(image_np, filtered_detections)

                    image_np = box_annotator.annotate(image_np, filtered_detections)

                    if label_annotator is not None:
                        labels = []
                        for category_id, conf in zip(filt_class_ids_raw, filt_confidences):
                            # Map category ID to class index for label lookup
                            class_id = category_mapping.get(category_id, category_id)
                            # Override class to "fm" if --all-fm is specified
                            if all_fm and fm_class_index is not None:
                                class_id = fm_class_index
                            if 0 <= class_id < len(class_names):
                                labels.append(f"{class_names[class_id]} {conf:.2f}")
                            else:
                                labels.append(f"class_{class_id} {conf:.2f}")

                        image_np = label_annotator.annotate(image_np, filtered_detections, labels)

                vis_path = vis_dir / f"{img_path.stem}_vis.png"
                # Use cv2.imwrite (faster than PIL) - convert RGB to BGR for cv2
                cv2.imwrite(str(vis_path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            t_postprocess_end = time.perf_counter()
            t_img_end = time.perf_counter()

            postprocess_times.append(t_postprocess_end - t_postprocess_start)
            total_times.append(t_img_end - t_img_start)

            image_id += 1

            # Incremental COCO saving every 100 images (no indent for speed)
            if save_coco and image_id % 100 == 0:
                coco_json_path = output_dir / 'coco_annotations.json'
                with open(coco_json_path, 'w') as f:
                    json.dump(coco_data, f)

            # Update progress bar
            pbar.set_postfix_str(f"{num_all_detections} detections")
            pbar.update(1)
    
    t_total_end = time.perf_counter()
    
    # Final COCO JSON save
    if save_coco:
        coco_json_path = output_dir / 'coco_annotations.json'
        with open(coco_json_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        print(f"COCO annotations saved: {coco_json_path}")
        print(f"  Total annotations: {len(coco_data['annotations'])} (ALL detections with scores)")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Inference completed!")
    print(f"{'='*80}")
    print(f"Processed: {len(image_paths)} images")
    print(f"Total detections: {total_detections} (all confidence levels)")
    print(f"Results saved to: {output_dir}")
    
    # Print timing statistics
    print(f"\n{'='*80}")
    print(f"Performance Metrics")
    print(f"{'='*80}")
    print(f"Total processing time: {t_total_end - t_total_start:.3f}s")
    print(f"\nPer-image averages:")
    print(f"  Image loading:    (overlapped with inference via prefetch)")
    print(f"  Inference:        {np.mean(inference_times):.3f}s (±{np.std(inference_times):.3f}s)")
    print(f"  Postprocessing:   {np.mean(postprocess_times):.3f}s (±{np.std(postprocess_times):.3f}s)")
    print(f"  Total per image:  {np.mean(total_times):.3f}s (±{np.std(total_times):.3f}s)")
    print(f"\nThroughput:")
    print(f"  Images per second: {len(image_paths) / (t_total_end - t_total_start):.2f}")
    print(f"  FPS (inference only): {1.0 / np.mean(inference_times):.2f}")
    print(f"{'='*80}")
    
    if segmentation and (save_instances or save_masks):
        print(f"\nMasks (all detections):")
        if save_instances:
            print(f"  Instance masks: {output_dir / 'instances'}")
        if save_masks:
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
    model_size = args['--model-size']
    conf_threshold = float(args['--conf-threshold'])
    coco_threshold = float(args['--coco-threshold'])
    device = args['--device']
    num_classes = int(args['--num-classes']) if args['--num-classes'] else None
    sample_size = int(args['--sample-size']) if args['--sample-size'] else None
    save_txt = args['--save-txt']
    save_coco = args['--save-coco']
    save_masks = args['--save-masks']
    save_instances = args['--save-instances']
    visualize = args['--visualize']
    hide_labels = args['--hide-labels']
    
    # Auto-detect segmentation if --no-segmentation is not specified
    # If --no-segmentation is specified, set segmentation to False
    # Otherwise, leave as None to auto-detect
    if args['--no-segmentation']:
        segmentation = False
    else:
        segmentation = None  # Will be auto-detected from checkpoint
    
    # Parse class filter
    filter_classes = None
    if args['--filter-classes']:
        filter_classes = [int(x) for x in args['--filter-classes'].split(',')]
    
    # Parse size filters
    min_size = float(args['--min']) if args['--min'] else None
    max_size = float(args['--max']) if args['--max'] else None
    all_fm = args['--all-fm']
    additional_weights = args['--additional-weights']
    
    run_inference(
        images_dir=images_dir,
        weights_path=weights_path,
        output_dir=output_dir,
        model_size=model_size,
        conf_threshold=conf_threshold,
        coco_threshold=coco_threshold,
        num_classes=num_classes,
        device=device,
        save_txt=save_txt,
        save_coco=save_coco,
        save_masks=save_masks,
        save_instances=save_instances,
        visualize=visualize,
        hide_labels=hide_labels,
        filter_classes=filter_classes,
        segmentation=segmentation,
        sample_size=sample_size,
        min_size=min_size,
        max_size=max_size,
        all_fm=all_fm,
        additional_weights=additional_weights,
    )


if __name__ == '__main__':
    main()
