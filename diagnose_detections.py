#!/usr/bin/env python3
"""Diagnose detection issues by comparing visualizations to COCO annotations."""

import json
from pathlib import Path
from collections import defaultdict

def diagnose_detections(output_dir='latest_out'):
    """Compare visualizations directory to COCO annotations."""
    
    output_path = Path(output_dir)
    coco_json_path = output_path / 'coco_annotations.json'
    vis_dir = output_path / 'visualizations'
    
    print(f"\n{'='*80}")
    print(f"Detection Diagnosis")
    print(f"{'='*80}\n")
    
    # Load COCO data
    if not coco_json_path.exists():
        print(f"ERROR: COCO annotations not found at {coco_json_path}")
        return
    
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Get image names with annotations
    image_id_to_name = {img['id']: img['file_name'] for img in coco_data['images']}
    images_with_annotations = set()
    annotation_counts = defaultdict(int)
    
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        images_with_annotations.add(img_id)
        annotation_counts[img_id] += 1
    
    # Get visualization file names
    if vis_dir.exists():
        vis_files = list(vis_dir.glob('*_vis.png'))
        vis_image_names = set()
        for vf in vis_files:
            # Remove '_vis' suffix to get original image name
            original_name = vf.stem.replace('_vis', '') + '.png'
            vis_image_names.add(original_name)
    else:
        print(f"ERROR: Visualizations directory not found at {vis_dir}")
        return
    
    print(f"Total images in COCO JSON: {len(coco_data['images'])}")
    print(f"Images with annotations: {len(images_with_annotations)}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print(f"Visualization files created: {len(vis_files)}")
    print()
    
    # Find images that have visualizations but no annotations
    images_with_vis_but_no_ann = []
    
    for img_id, img_name in image_id_to_name.items():
        has_annotation = img_id in images_with_annotations
        has_visualization = img_name in vis_image_names
        
        if has_visualization and not has_annotation:
            images_with_vis_but_no_ann.append((img_id, img_name))
    
    print(f"{'='*80}")
    print(f"MISMATCH ANALYSIS")
    print(f"{'='*80}")
    print(f"Images with visualizations but NO annotations: {len(images_with_vis_but_no_ann)}")
    print()
    
    if images_with_vis_but_no_ann:
        print("Sample of problematic images (first 20):")
        print(f"{'Image ID':<12} {'Filename':<50}")
        print(f"{'-'*80}")
        for img_id, img_name in images_with_vis_but_no_ann[:20]:
            print(f"{img_id:<12} {img_name:<50}")
        
        if len(images_with_vis_but_no_ann) > 20:
            print(f"... and {len(images_with_vis_but_no_ann) - 20} more")
    
    # Find images with annotations but no visualizations (shouldn't happen)
    images_with_ann_but_no_vis = []
    
    for img_id in images_with_annotations:
        img_name = image_id_to_name[img_id]
        has_visualization = img_name in vis_image_names
        
        if not has_visualization:
            images_with_ann_but_no_vis.append((img_id, img_name))
    
    print(f"\nImages with annotations but NO visualizations: {len(images_with_ann_but_no_vis)}")
    
    if images_with_ann_but_no_vis:
        print("Sample:")
        for img_id, img_name in images_with_ann_but_no_vis[:10]:
            print(f"  {img_id}: {img_name}")
    
    print(f"\n{'='*80}")
    print(f"HYPOTHESIS")
    print(f"{'='*80}")
    print()
    print("Based on the data:")
    print(f"- {len(vis_files)} visualizations were created (detections found)")
    print(f"- {len(images_with_annotations)} images have COCO annotations")
    print(f"- {len(images_with_vis_but_no_ann)} images have detections but no annotations")
    print()
    print("Possible causes:")
    print("1. Detections have out-of-range class IDs (gets visualized but skipped in COCO)")
    print("2. Mask-to-polygon conversion is failing (but annotation should still be created)")
    print("3. Some other filtering is happening between visualization and annotation")
    print()
    print("To verify: Check inference output for 'Warning: Detected class_id X is out of range'")
    print(f"{'='*80}\n")
    
    return {
        'vis_count': len(vis_files),
        'ann_count': len(images_with_annotations),
        'mismatch_count': len(images_with_vis_but_no_ann)
    }

if __name__ == '__main__':
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else 'latest_out'
    diagnose_detections(output_dir)

