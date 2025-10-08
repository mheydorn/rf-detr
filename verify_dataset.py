#!/usr/bin/env python3
"""
Verify that a COCO-format segmentation dataset is properly formatted for RF-DETR training.
"""

import json
from pathlib import Path
from PIL import Image
import sys


def verify_dataset(dataset_dir):
    """Verify the dataset structure and format."""
    dataset_path = Path(dataset_dir)
    
    print("=" * 80)
    print("Dataset Verification")
    print("=" * 80)
    print(f"\nDataset directory: {dataset_path.absolute()}")
    
    if not dataset_path.exists():
        print(f"❌ Error: Dataset directory does not exist: {dataset_path}")
        return False
    
    # Check directory structure
    print("\n1. Checking directory structure...")
    required_dirs = ['train', 'valid', 'test']
    all_dirs_exist = True
    
    for dir_name in required_dirs:
        dir_path = dataset_path / dir_name
        if dir_path.exists():
            print(f"   ✓ {dir_name}/ exists")
        else:
            print(f"   ❌ {dir_name}/ missing")
            all_dirs_exist = False
    
    if not all_dirs_exist:
        print("\n❌ Dataset structure is incomplete")
        return False
    
    # Check class names file
    print("\n2. Checking class names...")
    class_names_file = dataset_path / "class_names.txt"
    if class_names_file.exists():
        with open(class_names_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        print(f"   ✓ Found {len(class_names)} classes: {', '.join(class_names)}")
    else:
        print(f"   ⚠️  class_names.txt not found (will read from annotations)")
    
    # Verify each split
    print("\n3. Verifying splits...")
    stats = {}
    
    for split in required_dirs:
        print(f"\n   Checking {split} split...")
        split_dir = dataset_path / split
        
        # Check annotation file
        ann_file = split_dir / "_annotations.coco.json"
        if not ann_file.exists():
            print(f"      ❌ _annotations.coco.json missing in {split}/")
            return False
        
        print(f"      ✓ Annotation file exists")
        
        # Load and verify annotations
        try:
            with open(ann_file, 'r') as f:
                coco_data = json.load(f)
        except Exception as e:
            print(f"      ❌ Error loading annotation file: {e}")
            return False
        
        # Check required fields
        required_fields = ['images', 'annotations', 'categories']
        for field in required_fields:
            if field not in coco_data:
                print(f"      ❌ Missing required field: {field}")
                return False
        
        num_images = len(coco_data['images'])
        num_annotations = len(coco_data['annotations'])
        num_categories = len(coco_data['categories'])
        
        print(f"      ✓ Images: {num_images}")
        print(f"      ✓ Annotations: {num_annotations}")
        print(f"      ✓ Categories: {num_categories}")
        
        stats[split] = {
            'images': num_images,
            'annotations': num_annotations,
            'categories': num_categories
        }
        
        # Verify a few images exist
        print(f"      Verifying image files...")
        missing_images = 0
        for img_info in coco_data['images'][:10]:  # Check first 10
            img_path = split_dir / img_info['file_name']
            if not img_path.exists():
                missing_images += 1
        
        if missing_images > 0:
            print(f"      ⚠️  {missing_images} out of first 10 images are missing")
        else:
            print(f"      ✓ Sample images verified")
        
        # Check if annotations have segmentation masks
        has_segmentation = False
        for ann in coco_data['annotations'][:10]:
            if 'segmentation' in ann and len(ann['segmentation']) > 0:
                has_segmentation = True
                break
        
        if has_segmentation:
            print(f"      ✓ Annotations contain segmentation masks")
        else:
            print(f"      ⚠️  No segmentation masks found in annotations")
    
    # Summary
    print("\n" + "=" * 80)
    print("Verification Summary")
    print("=" * 80)
    
    total_images = sum(s['images'] for s in stats.values())
    total_annotations = sum(s['annotations'] for s in stats.values())
    
    print(f"\nTotal images: {total_images}")
    print(f"Total annotations: {total_annotations}")
    print(f"Average annotations per image: {total_annotations / total_images:.2f}")
    
    print(f"\nSplit distribution:")
    for split, split_stats in stats.items():
        percentage = (split_stats['images'] / total_images) * 100
        print(f"  {split:6s}: {split_stats['images']:5d} images ({percentage:5.1f}%)")
    
    print("\n✅ Dataset verification complete!")
    print("\nYou can now start training with:")
    print(f"  python train_segmentation.py --dataset-dir {dataset_dir} --output-dir output/my_model")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify COCO-format segmentation dataset")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to COCO-format dataset directory"
    )
    
    args = parser.parse_args()
    
    success = verify_dataset(args.dataset_dir)
    sys.exit(0 if success else 1)

