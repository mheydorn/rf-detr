#!/usr/bin/env python3
"""
Save sample images from COCO dataset with annotations visualized.

Usage:
    python save_coco_samples.py --dataset-dir datasets/onion_defect_coco --split train --num-samples 10
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils


def save_annotated_samples(dataset_dir: str, split: str, num_samples: int = 10, output_dir: str = "sample_visualizations"):
    """Save sample images with annotations drawn."""
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir) / split
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load COCO
    ann_file = dataset_path / split / "_annotations.coco.json"
    print(f"Loading annotations from {ann_file}...")
    coco = COCO(str(ann_file))
    
    # Get categories
    categories = {cat['id']: cat for cat in coco.dataset['categories']}
    
    # Generate colors
    np.random.seed(42)
    colors = {}
    for cat_id in categories.keys():
        colors[cat_id] = tuple((np.random.rand(3) * 255).astype(int).tolist())
    
    # Get random images
    image_ids = coco.getImgIds()
    sample_ids = random.sample(image_ids, min(num_samples, len(image_ids)))
    
    print(f"Saving {len(sample_ids)} annotated samples to {output_path}...")
    
    for idx, image_id in enumerate(sample_ids):
        # Load image
        img_info = coco.loadImgs(image_id)[0]
        img_path = dataset_path / split / img_info['file_name']
        img = cv2.imread(str(img_path))
        
        if img is None:
            print(f"Warning: Could not load {img_path}")
            continue
        
        # Get annotations
        ann_ids = coco.getAnnIds(imgIds=image_id)
        annotations = coco.loadAnns(ann_ids)
        
        # Create overlay for masks
        overlay = img.copy()
        
        # Draw each annotation
        for ann in annotations:
            cat_id = ann['category_id']
            cat_name = categories[cat_id]['name']
            color = colors[cat_id]
            
            # Draw segmentation mask
            if 'segmentation' in ann and isinstance(ann['segmentation'], list):
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(overlay, [poly], color)
            
            # Draw bounding box
            if 'bbox' in ann:
                x, y, w, h = [int(v) for v in ann['bbox']]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                
                # Draw label
                label = f"{cat_name}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x, y - label_size[1] - 10), 
                            (x + label_size[0] + 10, y), color, -1)
                cv2.putText(img, label, (x + 5, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Blend overlay with original image
        alpha = 0.4
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        
        # Add info text
        info_text = f"Image {idx + 1}/{len(sample_ids)} | ID: {image_id} | Annotations: {len(annotations)}"
        cv2.putText(img, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Save
        output_file = output_path / f"sample_{idx+1:03d}_{img_info['file_name']}"
        cv2.imwrite(str(output_file), img)
        print(f"  Saved: {output_file}")
    
    print(f"\n✓ Done! Saved {len(sample_ids)} samples to {output_path}")
    print(f"\nSummary:")
    for cat_id, cat in categories.items():
        print(f"  - {cat['name']}: {colors[cat_id]} (BGR color)")


def main():
    parser = argparse.ArgumentParser(description="Save sample images with annotations")
    parser.add_argument("--dataset-dir", required=True, help="Path to COCO dataset")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to save")
    parser.add_argument("--output-dir", default="sample_visualizations", help="Output directory")
    
    args = parser.parse_args()
    
    save_annotated_samples(
        dataset_dir=args.dataset_dir,
        split=args.split,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

