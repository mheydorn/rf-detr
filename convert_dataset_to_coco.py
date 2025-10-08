#!/usr/bin/env python3
"""
Convert segmentation dataset to COCO format for RF-DETR training.

This script converts datasets with individual JSON annotation files per image to COCO format.

Expected JSON format:
{
  "image_info": {"filename": "image.png"},
  "annotations": [
    {
      "class": "class_name",
      "polygons": [[[x1, y1], [x2, y2], ...]],
      "holes": []
    }
  ]
}
"""

import json
import os
from pathlib import Path
from datetime import datetime
import random
from collections import defaultdict
from PIL import Image
import shutil
from tqdm import tqdm


def load_annotation(json_path):
    """Load annotation from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def polygon_to_coco_format(polygons):
    """
    Convert polygon format from [[x, y], [x, y], ...] to COCO format [x, y, x, y, ...]
    """
    coco_polygons = []
    for polygon in polygons:
        if len(polygon) == 0:
            continue
        # Flatten the polygon: [[x, y], [x, y], ...] -> [x, y, x, y, ...]
        flattened = []
        for point in polygon:
            flattened.extend(point)
        coco_polygons.append(flattened)
    return coco_polygons


def get_bbox_from_polygon(polygon):
    """
    Calculate bounding box from polygon.
    Returns [x_min, y_min, width, height] in COCO format.
    """
    if len(polygon) == 0:
        return [0, 0, 0, 0]
    
    x_coords = [point[0] for point in polygon]
    y_coords = [point[1] for point in polygon]
    
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    
    width = x_max - x_min
    height = y_max - y_min
    
    return [x_min, y_min, width, height]


def calculate_area(bbox):
    """Calculate area from bounding box."""
    return bbox[2] * bbox[3]


def convert_dataset_to_coco(source_dir, output_dir, splits={'train': 0.7, 'valid': 0.2, 'test': 0.1}, seed=42, supercategory="object"):
    """
    Convert a segmentation dataset to COCO format.
    
    Args:
        source_dir: Path to the source dataset directory
        output_dir: Path to the output directory where COCO format will be saved
        splits: Dictionary with train/valid/test split ratios
        seed: Random seed for reproducibility
        supercategory: Supercategory name for all classes (default: "object")
    """
    random.seed(seed)
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    print(f"Converting dataset from {source_path} to {output_path}")
    
    # Get all JSON files
    json_files = sorted(list(source_path.glob("*.json")))
    print(f"Found {len(json_files)} annotation files")
    
    if len(json_files) == 0:
        raise ValueError(f"No JSON files found in {source_path}")
    
    # Collect all unique class names
    class_names = set()
    for json_file in tqdm(json_files, desc="Scanning for classes"):
        annotation = load_annotation(json_file)
        for ann in annotation.get('annotations', []):
            class_names.add(ann['class'])
    
    class_names = sorted(list(class_names))
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Create category mapping
    categories = []
    category_name_to_id = {}
    for idx, class_name in enumerate(class_names, start=1):
        categories.append({
            "id": idx,
            "name": class_name,
            "supercategory": supercategory
        })
        category_name_to_id[class_name] = idx
    
    # Shuffle and split the files
    random.shuffle(json_files)
    
    total = len(json_files)
    train_end = int(total * splits['train'])
    valid_end = train_end + int(total * splits['valid'])
    
    split_files = {
        'train': json_files[:train_end],
        'valid': json_files[train_end:valid_end],
        'test': json_files[valid_end:]
    }
    
    print(f"\nDataset split:")
    print(f"  Train: {len(split_files['train'])} images")
    print(f"  Valid: {len(split_files['valid'])} images")
    print(f"  Test: {len(split_files['test'])} images")
    
    # Process each split
    for split_name, files in split_files.items():
        print(f"\nProcessing {split_name} split...")
        
        # Create output directory for this split
        split_dir = output_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize COCO format structure
        coco_output = {
            "info": {
                "description": "Segmentation Dataset (converted to COCO format)",
                "url": "",
                "version": "1.0",
                "year": 2025,
                "contributor": "",
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": categories
        }
        
        annotation_id = 1
        
        # Process each file in this split
        for image_id, json_file in enumerate(tqdm(files, desc=f"Converting {split_name}"), start=1):
            annotation = load_annotation(json_file)
            
            # Get image filename
            image_filename = annotation['image_info']['filename']
            image_path = source_path / image_filename
            
            # Skip if image doesn't exist
            if not image_path.exists():
                print(f"Warning: Image {image_filename} not found, skipping")
                continue
            
            # Copy image to split directory
            dest_image_path = split_dir / image_filename
            shutil.copy2(image_path, dest_image_path)
            
            # Get image dimensions
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"Error opening image {image_filename}: {e}")
                continue
            
            # Add image info
            coco_output['images'].append({
                "id": image_id,
                "file_name": image_filename,
                "width": width,
                "height": height,
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": ""
            })
            
            # Process annotations
            for ann in annotation['annotations']:
                class_name = ann['class']
                polygons = ann.get('polygons', [])
                
                # Skip empty annotations
                if len(polygons) == 0:
                    continue
                
                # Process each polygon instance
                for polygon in polygons:
                    if len(polygon) < 3:  # Need at least 3 points for a polygon
                        continue
                    
                    # Convert polygon to COCO format
                    segmentation = [[]]
                    for point in polygon:
                        segmentation[0].extend(point)
                    
                    # Calculate bounding box
                    bbox = get_bbox_from_polygon(polygon)
                    area = calculate_area(bbox)
                    
                    # Skip invalid bounding boxes
                    if area == 0:
                        continue
                    
                    # Add annotation
                    coco_output['annotations'].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_name_to_id[class_name],
                        "segmentation": segmentation,
                        "area": area,
                        "bbox": bbox,
                        "iscrowd": 0
                    })
                    annotation_id += 1
        
        # Save COCO annotation file
        annotation_file = split_dir / "_annotations.coco.json"
        with open(annotation_file, 'w') as f:
            json.dump(coco_output, f, indent=2)
        
        print(f"Saved {split_name} annotations to {annotation_file}")
        print(f"  Images: {len(coco_output['images'])}")
        print(f"  Annotations: {len(coco_output['annotations'])}")
    
    # Save class names for reference
    class_names_file = output_path / "class_names.txt"
    with open(class_names_file, 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    print(f"\nConversion complete!")
    print(f"Output directory: {output_path}")
    print(f"Class names saved to: {class_names_file}")
    print(f"Categories: {class_names}")
    
    return class_names


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert segmentation dataset to COCO format")
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Path to source dataset directory with JSON annotations"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to output directory for COCO format"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.7,
        help="Ratio of data for training"
    )
    parser.add_argument(
        "--valid-split",
        type=float,
        default=0.2,
        help="Ratio of data for validation"
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Ratio of data for testing"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--supercategory",
        type=str,
        default="object",
        help="Supercategory name for all classes"
    )
    
    args = parser.parse_args()
    
    # Validate splits sum to 1
    total_split = args.train_split + args.valid_split + args.test_split
    if abs(total_split - 1.0) > 0.001:
        raise ValueError(f"Splits must sum to 1.0, got {total_split}")
    
    splits = {
        'train': args.train_split,
        'valid': args.valid_split,
        'test': args.test_split
    }
    
    convert_dataset_to_coco(
        args.source_dir,
        args.output_dir,
        splits=splits,
        seed=args.seed,
        supercategory=args.supercategory
    )

