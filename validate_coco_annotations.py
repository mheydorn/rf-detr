#!/usr/bin/env python3
"""
Validate COCO format dataset annotations for segmentation tasks.
Identifies and reports malformed segmentation polygons that could cause training crashes.

Usage:
    validate_coco_annotations.py <dataset_dir> [--fix]
    
Arguments:
    <dataset_dir>  Path to COCO format dataset directory
    
Options:
    --fix          Fix malformed annotations by removing invalid polygons
"""

import json
from pathlib import Path
from docopt import docopt


def validate_polygon(poly):
    """
    Validate a single polygon.
    Returns (is_valid, error_message)
    """
    if not isinstance(poly, list):
        return False, "Polygon is not a list"
    
    if len(poly) < 6:
        return False, f"Polygon has only {len(poly)} coordinates (needs at least 6 for 3 points)"
    
    if len(poly) % 2 != 0:
        return False, f"Polygon has odd number of coordinates ({len(poly)})"
    
    # Check for invalid coordinate values
    for coord in poly:
        if not isinstance(coord, (int, float)):
            return False, f"Invalid coordinate type: {type(coord)}"
        if coord < 0:
            return False, f"Negative coordinate: {coord}"
    
    return True, None


def validate_segmentation(segm, ann_id):
    """
    Validate a segmentation annotation.
    Returns (is_valid, valid_polygons, issues)
    """
    issues = []
    
    # RLE format is already encoded, skip validation
    if isinstance(segm, dict):
        return True, segm, []
    
    # Must be a list of polygons
    if not isinstance(segm, list):
        return False, None, [f"Segmentation is not a list (type: {type(segm)})"]
    
    if len(segm) == 0:
        return False, None, [f"Segmentation is empty"]
    
    valid_polygons = []
    for i, poly in enumerate(segm):
        is_valid, error = validate_polygon(poly)
        if is_valid:
            valid_polygons.append(poly)
        else:
            issues.append(f"Polygon {i}: {error}")
    
    if len(valid_polygons) == 0:
        return False, None, issues
    
    return True, valid_polygons, issues


def validate_dataset(dataset_dir, split, fix=False):
    """
    Validate annotations for a specific split.
    Returns (total_anns, invalid_anns, fixed_anns, issues)
    """
    ann_file = Path(dataset_dir) / split / "_annotations.coco.json"
    
    if not ann_file.exists():
        return None, None, None, f"Annotation file not found: {ann_file}"
    
    print(f"\nValidating {split} split...")
    print(f"Annotation file: {ann_file}")
    
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    if 'annotations' not in data:
        return 0, 0, 0, []
    
    annotations = data['annotations']
    total_anns = len(annotations)
    invalid_anns = 0
    fixed_anns = 0
    all_issues = []
    
    valid_annotations = []
    
    for ann in annotations:
        ann_id = ann.get('id', 'unknown')
        
        if 'segmentation' not in ann:
            valid_annotations.append(ann)
            continue
        
        segm = ann['segmentation']
        is_valid, valid_polygons, issues = validate_segmentation(segm, ann_id)
        
        if not is_valid:
            invalid_anns += 1
            all_issues.append({
                'annotation_id': ann_id,
                'image_id': ann.get('image_id', 'unknown'),
                'category_id': ann.get('category_id', 'unknown'),
                'issues': issues
            })
            if not fix:
                # Keep the annotation even if invalid (unless fixing)
                valid_annotations.append(ann)
        else:
            if len(issues) > 0:
                # Some polygons were invalid but at least one is valid
                fixed_anns += 1
                all_issues.append({
                    'annotation_id': ann_id,
                    'image_id': ann.get('image_id', 'unknown'),
                    'category_id': ann.get('category_id', 'unknown'),
                    'issues': issues,
                    'fixed': True
                })
                if fix:
                    ann['segmentation'] = valid_polygons
            valid_annotations.append(ann)
    
    # Save fixed annotations if requested
    if fix and (invalid_anns > 0 or fixed_anns > 0):
        # Backup original file
        backup_file = ann_file.with_suffix('.json.backup')
        if not backup_file.exists():
            print(f"Creating backup: {backup_file}")
            with open(backup_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        # Save fixed annotations
        data['annotations'] = valid_annotations
        print(f"Saving fixed annotations to: {ann_file}")
        with open(ann_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Fixed file saved. Original backed up to {backup_file}")
    
    return total_anns, invalid_anns, fixed_anns, all_issues


def main():
    args = docopt(__doc__)
    dataset_dir = args['<dataset_dir>']
    fix = args['--fix']
    
    print("=" * 80)
    print("COCO Segmentation Annotation Validator")
    print("=" * 80)
    print(f"Dataset directory: {dataset_dir}")
    print(f"Fix mode: {'ON - will fix and save' if fix else 'OFF - validation only'}")
    
    # Validate each split
    splits = ['train', 'valid', 'test']
    total_stats = {'total': 0, 'invalid': 0, 'fixed': 0}
    all_split_issues = {}
    
    for split in splits:
        result = validate_dataset(dataset_dir, split, fix)
        
        if result[0] is None:
            print(f"\n{split}: {result[-1]}")
            continue
        
        total_anns, invalid_anns, fixed_anns, issues = result
        total_stats['total'] += total_anns
        total_stats['invalid'] += invalid_anns
        total_stats['fixed'] += fixed_anns
        
        if len(issues) > 0:
            all_split_issues[split] = issues
        
        print(f"  Total annotations: {total_anns}")
        print(f"  Invalid annotations: {invalid_anns}")
        print(f"  Partially fixed annotations: {fixed_anns}")
        
        if invalid_anns > 0 or fixed_anns > 0:
            print(f"  ⚠️  Found {invalid_anns + fixed_anns} problematic annotations")
    
    # Print detailed issues
    if all_split_issues:
        print("\n" + "=" * 80)
        print("DETAILED ISSUES")
        print("=" * 80)
        
        for split, issues in all_split_issues.items():
            print(f"\n{split.upper()} split:")
            for issue_data in issues[:10]:  # Show first 10 issues per split
                ann_id = issue_data['annotation_id']
                img_id = issue_data['image_id']
                cat_id = issue_data['category_id']
                fixed = issue_data.get('fixed', False)
                status = "[FIXED]" if fixed else "[INVALID]"
                
                print(f"  {status} Annotation {ann_id} (image: {img_id}, category: {cat_id}):")
                for issue in issue_data['issues']:
                    print(f"    - {issue}")
            
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more issues")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total annotations: {total_stats['total']}")
    print(f"Invalid annotations: {total_stats['invalid']} ({100*total_stats['invalid']/max(1,total_stats['total']):.1f}%)")
    print(f"Partially fixed annotations: {total_stats['fixed']} ({100*total_stats['fixed']/max(1,total_stats['total']):.1f}%)")
    
    if total_stats['invalid'] > 0 or total_stats['fixed'] > 0:
        print("\n⚠️  Issues found!")
        if not fix:
            print("\nTo fix these issues, run with --fix flag:")
            print(f"  python validate_coco_annotations.py {dataset_dir} --fix")
        else:
            print("\n✓ Annotations have been fixed and saved.")
            print("  Original files backed up with .backup extension.")
            print("  You can now run training without segmentation errors.")
    else:
        print("\n✓ All annotations are valid!")


if __name__ == '__main__':
    main()

