#!/usr/bin/env python3
"""Inspect COCO annotations to check segmentation data."""

import json
import sys
from pathlib import Path
from collections import defaultdict

def analyze_coco_annotations(coco_json_path):
    """Analyze COCO annotations to check segmentation coverage."""
    
    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"COCO Annotations Analysis")
    print(f"{'='*80}")
    print(f"File: {coco_json_path}")
    print(f"\n{'='*80}")
    print(f"Summary Statistics")
    print(f"{'='*80}")
    
    # Basic stats
    num_images = len(coco_data.get('images', []))
    num_annotations = len(coco_data.get('annotations', []))
    num_categories = len(coco_data.get('categories', []))
    
    print(f"Total Images: {num_images}")
    print(f"Total Annotations: {num_annotations}")
    print(f"Total Categories: {num_categories}")
    
    # Analyze annotations
    annotations_with_seg = 0
    annotations_without_seg = 0
    segmentation_point_counts = []
    segmentation_areas = []
    
    # Track per-image data
    image_annotation_count = defaultdict(int)
    image_seg_count = defaultdict(int)
    
    for ann in coco_data.get('annotations', []):
        image_id = ann['image_id']
        image_annotation_count[image_id] += 1
        
        if 'segmentation' in ann and ann['segmentation']:
            annotations_with_seg += 1
            image_seg_count[image_id] += 1
            
            # Analyze segmentation polygons
            for polygon in ann['segmentation']:
                num_points = len(polygon) // 2  # polygon is [x1, y1, x2, y2, ...]
                segmentation_point_counts.append(num_points)
            
            # Track area
            if 'area' in ann:
                segmentation_areas.append(ann['area'])
        else:
            annotations_without_seg += 1
    
    print(f"\n{'='*80}")
    print(f"Annotation Breakdown")
    print(f"{'='*80}")
    print(f"Annotations WITH segmentation: {annotations_with_seg} ({annotations_with_seg/max(num_annotations,1)*100:.1f}%)")
    print(f"Annotations WITHOUT segmentation: {annotations_without_seg} ({annotations_without_seg/max(num_annotations,1)*100:.1f}%)")
    
    # Per-image statistics
    images_with_annotations = len(image_annotation_count)
    images_with_segmentations = len([img_id for img_id, count in image_seg_count.items() if count > 0])
    images_without_segmentations = images_with_annotations - images_with_segmentations
    
    print(f"\n{'='*80}")
    print(f"Image-Level Statistics")
    print(f"{'='*80}")
    print(f"Images with annotations: {images_with_annotations}")
    print(f"Images with at least 1 segmentation: {images_with_segmentations}")
    print(f"Images with NO segmentations: {images_without_segmentations}")
    
    if segmentation_point_counts:
        avg_points = sum(segmentation_point_counts) / len(segmentation_point_counts)
        min_points = min(segmentation_point_counts)
        max_points = max(segmentation_point_counts)
        
        print(f"\n{'='*80}")
        print(f"Polygon Complexity")
        print(f"{'='*80}")
        print(f"Average points per polygon: {avg_points:.1f}")
        print(f"Min points: {min_points}")
        print(f"Max points: {max_points}")
    
    if segmentation_areas:
        avg_area = sum(segmentation_areas) / len(segmentation_areas)
        min_area = min(segmentation_areas)
        max_area = max(segmentation_areas)
        
        print(f"\n{'='*80}")
        print(f"Segmentation Areas")
        print(f"{'='*80}")
        print(f"Average area: {avg_area:.1f} pixels")
        print(f"Min area: {min_area:.1f} pixels")
        print(f"Max area: {max_area:.1f} pixels")
        
        # Count "significantly sized" segmentations (e.g., > 100 pixels)
        significant_threshold = 100
        significant_segs = sum(1 for area in segmentation_areas if area > significant_threshold)
        print(f"Segmentations > {significant_threshold} pixels: {significant_segs} ({significant_segs/len(segmentation_areas)*100:.1f}%)")
    
    # Detailed per-image breakdown
    print(f"\n{'='*80}")
    print(f"Per-Image Breakdown (showing first 20 images)")
    print(f"{'='*80}")
    print(f"{'Image ID':<12} {'Filename':<40} {'Annotations':<13} {'Segmentations':<15}")
    print(f"{'-'*80}")
    
    image_lookup = {img['id']: img for img in coco_data.get('images', [])}
    
    for i, (img_id, ann_count) in enumerate(sorted(image_annotation_count.items())):
        if i >= 20:
            print(f"... and {len(image_annotation_count) - 20} more images")
            break
        
        seg_count = image_seg_count.get(img_id, 0)
        img_info = image_lookup.get(img_id, {})
        filename = img_info.get('file_name', 'unknown')
        
        print(f"{img_id:<12} {filename:<40} {ann_count:<13} {seg_count:<15}")
    
    # Find problematic images (have annotations but no segmentations)
    problematic_images = [(img_id, image_annotation_count[img_id]) 
                         for img_id in image_annotation_count 
                         if image_seg_count.get(img_id, 0) == 0]
    
    if problematic_images:
        print(f"\n{'='*80}")
        print(f"PROBLEMATIC: Images with annotations but NO segmentations")
        print(f"{'='*80}")
        print(f"{'Image ID':<12} {'Filename':<40} {'Annotations':<13}")
        print(f"{'-'*80}")
        
        for img_id, ann_count in problematic_images[:10]:
            img_info = image_lookup.get(img_id, {})
            filename = img_info.get('file_name', 'unknown')
            print(f"{img_id:<12} {filename:<40} {ann_count:<13}")
        
        if len(problematic_images) > 10:
            print(f"... and {len(problematic_images) - 10} more problematic images")
    
    # Sample a few annotations to show structure
    print(f"\n{'='*80}")
    print(f"Sample Annotations (first 3)")
    print(f"{'='*80}")
    
    for i, ann in enumerate(coco_data.get('annotations', [])[:3]):
        print(f"\nAnnotation {i+1}:")
        print(f"  ID: {ann.get('id')}")
        print(f"  Image ID: {ann.get('image_id')}")
        print(f"  Category ID: {ann.get('category_id')}")
        print(f"  Bbox: {ann.get('bbox')}")
        print(f"  Area: {ann.get('area')}")
        print(f"  Has Segmentation: {'Yes' if 'segmentation' in ann and ann['segmentation'] else 'No'}")
        if 'segmentation' in ann and ann['segmentation']:
            for j, seg in enumerate(ann['segmentation']):
                print(f"  Polygon {j+1}: {len(seg)//2} points, first few coords: {seg[:10]}...")
    
    print(f"\n{'='*80}\n")
    
    return {
        'images_with_segmentations': images_with_segmentations,
        'images_without_segmentations': images_without_segmentations,
        'annotations_with_seg': annotations_with_seg,
        'annotations_without_seg': annotations_without_seg
    }


if __name__ == '__main__':
    # Check if path provided as argument
    if len(sys.argv) > 1:
        coco_path = Path(sys.argv[1])
    else:
        # Default to latest_out
        coco_path = Path('latest_out/coco_annotations.json')
    
    if not coco_path.exists():
        print(f"Error: COCO annotations file not found: {coco_path}")
        print(f"\nUsage: {sys.argv[0]} [path/to/coco_annotations.json]")
        sys.exit(1)
    
    analyze_coco_annotations(coco_path)

