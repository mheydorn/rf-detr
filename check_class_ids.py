#!/usr/bin/env python3
"""Check what class IDs are in the COCO annotations to verify the hypothesis."""

import json
import sys
from pathlib import Path
from collections import Counter

def check_class_ids(coco_json_path):
    """Check class IDs in COCO annotations."""
    
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"Class ID Analysis")
    print(f"{'='*80}\n")
    
    # Get all class IDs from annotations
    class_ids = [ann['category_id'] for ann in coco_data['annotations']]
    class_id_counts = Counter(class_ids)
    
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print(f"Unique class IDs found: {sorted(class_id_counts.keys())}")
    print(f"\nClass ID distribution:")
    for class_id in sorted(class_id_counts.keys()):
        count = class_id_counts[class_id]
        print(f"  Class {class_id}: {count} annotations ({count/len(coco_data['annotations'])*100:.1f}%)")
    
    # Get categories from COCO data
    print(f"\n{'='*80}")
    print(f"Categories Defined in COCO JSON")
    print(f"{'='*80}\n")
    
    categories = coco_data.get('categories', [])
    print(f"Number of categories: {len(categories)}")
    for cat in categories:
        cat_id = cat['id']
        cat_name = cat['name']
        count = class_id_counts.get(cat_id, 0)
        print(f"  ID {cat_id}: {cat_name} ({count} annotations)")
    
    # Check for consistency
    print(f"\n{'='*80}")
    print(f"Consistency Check")
    print(f"{'='*80}\n")
    
    max_class_id = max(class_ids) if class_ids else -1
    num_categories = len(categories)
    
    if max_class_id >= num_categories:
        print(f"❌ ERROR: Max class ID ({max_class_id}) >= num_categories ({num_categories})")
        print(f"   This should not happen with the fix applied.")
    else:
        print(f"✅ OK: All class IDs are within valid range (0-{num_categories-1})")
    
    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    coco_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('latest_out/coco_annotations.json')
    
    if not coco_path.exists():
        print(f"Error: COCO annotations file not found: {coco_path}")
        sys.exit(1)
    
    check_class_ids(coco_path)

