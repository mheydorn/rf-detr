#!/usr/bin/env python3
"""Create a validation grid showing original images, ground truth masks, predictions, and overlays.

This script generates a visual comparison grid with 4 columns:
  1. Original Image - The input image
  2. Ground Truth Mask - Color-coded mask from annotations (each class has a unique color)
  3. Prediction Mask - Color-coded mask from model predictions (each class has a unique color)
  4. Overlay - Original image with predicted masks overlaid (semi-transparent, color-coded)

A color legend is automatically added at the top showing which color corresponds to each class.

Usage:
    create_validation_grid.py <weights_path> <dataset_dir> <output_path> [options]
    create_validation_grid.py -h | --help

Arguments:
    <weights_path>  Path to trained RF-DETR weights (.pth file)
    <dataset_dir>   Path to COCO dataset directory (should contain valid/_annotations.coco.json)
    <output_path>   Path to save the validation grid image

Options:
    -h --help                    Show this help message and exit
    --num-samples=<n>            Number of validation samples to display [default: 10]
    --model-size=<size>          Model size (nano, small, medium, large) [default: small]
    --device=<device>            Device to run inference on (cpu, cuda, mps) [default: cuda]
    --conf-threshold=<th>        Confidence threshold for predictions [default: 0.5]
    --seed=<seed>                Random seed for sample selection [default: 42]

Examples:
    # Basic usage
    create_validation_grid.py model.pth datasets/onion_defect_coco validation_grid.png
    
    # With more samples
    create_validation_grid.py model.pth datasets/onion_defect_coco validation_grid.png --num-samples=20
    
    # Using CPU
    create_validation_grid.py model.pth datasets/onion_defect_coco validation_grid.png --device=cpu

"""

import os
import sys
import json
import cv2
import numpy as np
import torch
from pathlib import Path
from docopt import docopt
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools import mask as mask_utils
from typing import List, Dict, Tuple
import random

# Import RF-DETR models
try:
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge
except ImportError as e:
    print(f"Error importing RF-DETR modules: {e}")
    print("Make sure RF-DETR is installed: pip install -e .")
    sys.exit(1)


def load_coco_annotations(coco_json_path: Path) -> Dict:
    """Load COCO format annotations."""
    with open(coco_json_path, 'r') as f:
        return json.load(f)


def polygon_to_mask(polygons: List[List[float]], height: int, width: int) -> np.ndarray:
    """Convert COCO polygon to binary mask."""
    if not polygons:
        return np.zeros((height, width), dtype=np.uint8)
    
    # COCO uses RLE format
    rles = mask_utils.frPyObjects(polygons, height, width)
    rle = mask_utils.merge(rles)
    mask = mask_utils.decode(rle)
    
    return mask


def generate_color_palette(num_classes: int) -> np.ndarray:
    """Generate distinct colors for each class.
    
    Returns:
        Array of shape (num_classes + 1, 3) with RGB colors.
        Index 0 is black (background), indices 1+ are class colors.
    """
    # Create color palette with distinct colors
    colors = np.zeros((num_classes + 1, 3), dtype=np.uint8)
    
    # Background is black
    colors[0] = [0, 0, 0]
    
    # Use matplotlib's colormap for distinct colors
    try:
        cmap = plt.colormaps.get_cmap('tab10' if num_classes <= 10 else 'tab20')
    except AttributeError:
        # Fallback for older matplotlib versions
        cmap = plt.cm.get_cmap('tab10' if num_classes <= 10 else 'tab20')
    
    for i in range(num_classes):
        # Get color from colormap
        rgba = cmap(i / max(num_classes, 1))
        # Convert to RGB (0-255)
        colors[i + 1] = [int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)]
    
    return colors


def create_colored_mask(class_mask: np.ndarray, num_classes: int) -> np.ndarray:
    """Create RGB colored mask from class mask.
    
    Args:
        class_mask: 2D array where 0=background, 1+ = class_id + 1
        num_classes: Number of classes
        
    Returns:
        RGB image (H, W, 3) with colored classes
    """
    height, width = class_mask.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Generate color palette
    colors = generate_color_palette(num_classes)
    
    # Apply colors based on class mask
    for class_id in range(num_classes + 1):
        mask = class_mask == class_id
        colored_mask[mask] = colors[class_id]
    
    return colored_mask


def get_ground_truth_mask(coco_data: Dict, image_id: int, height: int, width: int, num_classes: int, class_names: List[str], debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Get combined ground truth mask for an image with class IDs.

    Returns:
        Tuple of (colored_mask RGB image, class_mask with class IDs)
    """
    # Find all annotations for this image
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]

    # Create class mask (0 = background, 1+ = class index + 1)
    class_mask = np.zeros((height, width), dtype=np.uint8)

    # Create mapping from category_id to class index
    category_to_class_idx = {}
    for cat in coco_data['categories']:
        category_to_class_idx[cat['id']] = class_names.index(cat['name'])

    if debug:
        print(f"  Ground truth category mapping: {category_to_class_idx}")
        print(f"  Class names: {class_names}")
        print(f"  Found {len(annotations)} annotations")

    for ann in annotations:
        if 'segmentation' in ann and ann['segmentation']:
            mask = polygon_to_mask(ann['segmentation'], height, width)
            category_id = ann['category_id']
            if category_id in category_to_class_idx:
                class_idx = category_to_class_idx[category_id]
                # Use class_idx + 1 so background stays 0
                class_mask[mask > 0] = class_idx + 1
                if debug:
                    print(f"    Category {category_id} -> class_idx {class_idx} -> mask value {class_idx + 1}")

    # Create colored mask
    colored_mask = create_colored_mask(class_mask, num_classes)

    return colored_mask, class_mask


def get_prediction_mask(model, image: Image.Image, conf_threshold: float, num_classes: int, coco_data: Dict, class_names: List[str], debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Get prediction mask from model with class IDs.

    Returns:
        Tuple of (colored_mask RGB image, class_mask with class IDs)
    """
    # Run inference
    detections = model.predict(image, threshold=conf_threshold)

    # Create combined mask from all detections
    height, width = image.size[1], image.size[0]
    class_mask = np.zeros((height, width), dtype=np.uint8)

    # Create mapping from category_id to class index (same as ground truth)
    category_to_class_idx = {}
    for cat in coco_data['categories']:
        category_to_class_idx[cat['id']] = class_names.index(cat['name'])

    if len(detections) > 0 and detections.mask is not None:
        if debug:
            print(f"    Found {len(detections)} detections")
            print(f"    Model output class IDs: {detections.class_id}")
            print(f"    Confidences: {detections.confidence}")
        
        for i, mask in enumerate(detections.mask):
            model_class_id = detections.class_id[i]
            
            # Map model's class_id to class index
            # Model outputs COCO category IDs, need to map to class indices
            if model_class_id in category_to_class_idx:
                class_idx = category_to_class_idx[model_class_id]
            else:
                # If model outputs 0-indexed, use it directly
                class_idx = model_class_id
            
            # Use class_idx + 1 so background stays 0
            mask_area = mask > 0
            # Only set pixels that aren't already assigned (no overwriting)
            class_mask[mask_area & (class_mask == 0)] = class_idx + 1
            
            if debug:
                print(f"    Detection {i}: model_class_id {model_class_id} -> class_idx {class_idx} -> mask value {class_idx + 1}")
        
        if debug:
            unique_classes = np.unique(class_mask)
            print(f"    Unique class values in mask: {unique_classes}")

    # Create colored mask
    colored_mask = create_colored_mask(class_mask, num_classes)

    return colored_mask, class_mask


def create_overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = None, alpha: float = 0.5) -> np.ndarray:
    """Create overlay of mask on image.
    
    Args:
        image: Original image (H, W, 3)
        mask: Either RGB colored mask (H, W, 3) or binary mask (H, W)
        color: Color to use for binary masks. If None, assumes mask is already colored.
        alpha: Transparency (0=transparent, 1=opaque)
    """
    overlay = image.copy().astype(np.float32)
    
    # Check if mask is colored (3 channels) or binary (2D)
    if len(mask.shape) == 3 and mask.shape[2] == 3:
        # Colored mask - blend where mask is not black
        mask_binary = np.any(mask > 0, axis=2)
        mask_float = mask.astype(np.float32)
        overlay[mask_binary] = (1 - alpha) * overlay[mask_binary] + alpha * mask_float[mask_binary]
    else:
        # Binary mask - use specified color
        if color is None:
            color = (255, 0, 0)  # Default to red
        colored_mask = np.zeros_like(overlay)
        colored_mask[mask > 0] = color
        overlay = (1 - alpha) * overlay + alpha * colored_mask
    
    return overlay.astype(np.uint8)


def create_validation_grid(
    weights_path: str,
    dataset_dir: str,
    output_path: str,
    num_samples: int = 10,
    model_size: str = 'small',
    device: str = 'cuda',
    conf_threshold: float = 0.5,
    seed: int = 42,
):
    """
    Create a validation grid showing original images, ground truth, predictions, and overlays.
    
    Args:
        weights_path: Path to trained model weights
        dataset_dir: Path to COCO dataset directory
        output_path: Path to save the grid image
        num_samples: Number of samples to display
        model_size: Model size (nano, small, medium, large)
        device: Device to run inference on
        conf_threshold: Confidence threshold for predictions
        seed: Random seed for sample selection
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Validate inputs
    weights_path = Path(weights_path)
    dataset_dir = Path(dataset_dir)
    output_path = Path(output_path)
    
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    # Load COCO annotations
    coco_json_path = dataset_dir / 'valid' / '_annotations.coco.json'
    if not coco_json_path.exists():
        raise FileNotFoundError(f"COCO annotations not found: {coco_json_path}")
    
    print(f"\n{'='*80}")
    print("Creating Validation Grid")
    print(f"{'='*80}")
    print(f"Model: {weights_path}")
    print(f"Model size: {model_size}")
    print(f"Device: {device}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"Number of samples: {num_samples}")
    print(f"Dataset: {dataset_dir}")
    
    coco_data = load_coco_annotations(coco_json_path)
    
    # Load class names
    class_names_path = dataset_dir / 'class_names.txt'
    if class_names_path.exists():
        with open(class_names_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
    else:
        # Get class names from COCO categories
        class_names = [cat['name'] for cat in sorted(coco_data['categories'], key=lambda x: x['id'])]
    
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {', '.join(class_names)}")
    
    # Check device availability
    if device.startswith('cuda') and not torch.cuda.is_available():
        print(f"\nWarning: CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'
    
    # Initialize model
    print("\nLoading model...")
    model_classes = {
        'nano': RFDETRNano,
        'small': RFDETRSmall,
        'medium': RFDETRMedium,
        'large': RFDETRLarge,
    }
    
    if model_size.lower() not in model_classes:
        raise ValueError(f"Invalid model size: {model_size}. Choose from {list(model_classes.keys())}")
    
    model_class = model_classes[model_size.lower()]
    model = model_class(
        num_classes=num_classes,
        pretrain_weights=str(weights_path),
        segmentation_head=True,
        device=device,
    )
    
    # Optimize for inference
    try:
        model.optimize_for_inference()
        print("Model optimization successful")
    except Exception as e:
        print(f"Warning: Could not optimize model: {e}")
    
    # Find positive examples (images with annotations)
    positive_images = []
    for img_info in coco_data['images']:
        image_id = img_info['id']
        # Check if image has annotations
        has_annotations = any(ann['image_id'] == image_id for ann in coco_data['annotations'])
        if has_annotations:
            positive_images.append(img_info)
    
    print(f"\nFound {len(positive_images)} positive examples in validation set")
    
    if len(positive_images) < num_samples:
        print(f"Warning: Only {len(positive_images)} positive examples available. Using all of them.")
        num_samples = len(positive_images)
    
    # Randomly sample positive examples
    sampled_images = random.sample(positive_images, num_samples)
    
    print(f"\n{'='*80}")
    print("Processing samples...")
    print(f"{'='*80}\n")
    
    # Create figure with subplots (num_samples rows, 4 columns)
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    
    # Handle case where num_samples = 1
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    valid_images_dir = dataset_dir / 'valid'
    
    for idx, img_info in enumerate(sampled_images):
        print(f"Processing sample {idx + 1}/{num_samples}: {img_info['file_name']}")
        
        # Load image
        image_path = valid_images_dir / img_info['file_name']
        if not image_path.exists():
            print(f"  Warning: Image not found: {image_path}")
            continue
        
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        height, width = img_info['height'], img_info['width']
        
        # Get ground truth mask (colored)
        gt_mask_colored, gt_mask_class = get_ground_truth_mask(coco_data, img_info['id'], height, width, num_classes, class_names, debug=False)
        
        # Get prediction mask (colored)
        pred_mask_colored, pred_mask_class = get_prediction_mask(model, image, conf_threshold, num_classes, coco_data, class_names, debug=False)
        
        # Create overlay (colored prediction mask on original image)
        overlay = create_overlay(image_np, pred_mask_colored, color=None, alpha=0.5)
        
        # Plot original image
        axes[idx, 0].imshow(image_np)
        axes[idx, 0].set_title(f'Original\n{img_info["file_name"]}', fontsize=10)
        axes[idx, 0].axis('off')
        
        # Plot ground truth mask (colored)
        axes[idx, 1].imshow(gt_mask_colored)
        axes[idx, 1].set_title('Ground Truth Mask', fontsize=10)
        axes[idx, 1].axis('off')
        
        # Plot prediction mask (colored)
        axes[idx, 2].imshow(pred_mask_colored)
        axes[idx, 2].set_title('Prediction Mask', fontsize=10)
        axes[idx, 2].axis('off')
        
        # Plot overlay
        axes[idx, 3].imshow(overlay)
        axes[idx, 3].set_title('Overlay', fontsize=10)
        axes[idx, 3].axis('off')
    
    plt.tight_layout()
    
    # Add legend showing class colors
    colors = generate_color_palette(num_classes)
    legend_elements = []
    for i in range(num_classes):
        color = colors[i + 1] / 255.0  # Normalize to [0, 1] for matplotlib
        legend_elements.append(patches.Patch(facecolor=color, label=class_names[i]))
    
    # Add legend to the figure
    fig.legend(handles=legend_elements, loc='upper center', ncol=min(num_classes, 5), 
               bbox_to_anchor=(0.5, 1.0), fontsize=12, frameon=True)
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n{'='*80}")
    print(f"Validation grid saved to: {output_path}")
    print(f"{'='*80}\n")
    
    plt.close()


def main():
    """Main entry point."""
    args = docopt(__doc__)
    
    # Parse arguments
    weights_path = args['<weights_path>']
    dataset_dir = args['<dataset_dir>']
    output_path = args['<output_path>']
    
    # Parse options
    num_samples = int(args['--num-samples'])
    model_size = args['--model-size']
    device = args['--device']
    conf_threshold = float(args['--conf-threshold'])
    seed = int(args['--seed'])
    
    create_validation_grid(
        weights_path=weights_path,
        dataset_dir=dataset_dir,
        output_path=output_path,
        num_samples=num_samples,
        model_size=model_size,
        device=device,
        conf_threshold=conf_threshold,
        seed=seed,
    )


if __name__ == '__main__':
    main()

