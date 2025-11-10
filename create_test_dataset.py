#!/usr/bin/env python3
"""
Create a small test COCO dataset for quick testing.
"""

import json
import shutil
from pathlib import Path
from PIL import Image, ImageDraw
import random

def create_test_dataset(output_dir, num_train=5, num_val=2, num_test=2, num_classes=2):
    """
    Create a minimal COCO dataset with synthetic images.
    
    Args:
        output_dir: Output directory for the dataset
        num_train: Number of training images
        num_val: Number of validation images
        num_test: Number of test images
        num_classes: Number of object classes
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating test dataset in {output_path}")
    
    # Define classes
    class_names = [f"class_{i}" for i in range(num_classes)]
    categories = [
        {"id": i + 1, "name": name, "supercategory": "object"}
        for i, name in enumerate(class_names)
    ]
    
    # Save class names
    with open(output_path / "class_names.txt", 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    
    print(f"Classes: {class_names}")
    
    def create_split(split_name, num_images):
        """Create a single split (train/val)."""
        split_dir = output_path / split_name
        split_dir.mkdir(exist_ok=True)
        
        images = []
        annotations = []
        annotation_id = 1
        
        for img_id in range(1, num_images + 1):
            # Create synthetic image
            width, height = 640, 480
            img = Image.new('RGB', (width, height), color=(random.randint(200, 255), random.randint(200, 255), random.randint(200, 255)))
            draw = ImageDraw.Draw(img)
            
            # Add 1-3 random boxes per image
            num_boxes = random.randint(1, 3)
            for _ in range(num_boxes):
                # Random box
                x1 = random.randint(50, width - 150)
                y1 = random.randint(50, height - 150)
                box_w = random.randint(50, 150)
                box_h = random.randint(50, 150)
                x2 = min(x1 + box_w, width - 10)
                y2 = min(y1 + box_h, height - 10)
                
                # Random class
                class_id = random.randint(1, num_classes)
                
                # Draw box
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
                draw.rectangle([x1, y1, x2, y2], outline=colors[class_id % len(colors)], width=3)
                
                # Calculate area
                area = (x2 - x1) * (y2 - y1)
                
                # Add annotation
                annotations.append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": class_id,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "area": float(area),
                    "iscrowd": 0
                })
                annotation_id += 1
            
            # Save image
            img_filename = f"image_{img_id:04d}.jpg"
            img.save(split_dir / img_filename)
            
            # Add image info
            images.append({
                "id": img_id,
                "file_name": img_filename,
                "width": width,
                "height": height
            })
        
        # Create COCO annotation file
        coco_data = {
            "images": images,
            "annotations": annotations,
            "categories": categories
        }
        
        # Save annotations
        ann_file = output_path / f"{split_name}_annotations.json"
        with open(ann_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"  {split_name}: {len(images)} images, {len(annotations)} annotations")
        return len(images), len(annotations)
    
    # Create splits
    train_imgs, train_anns = create_split("train", num_train)
    val_imgs, val_anns = create_split("val", num_val)
    test_imgs, test_anns = create_split("test", num_test)
    
    print(f"\nDataset created successfully!")
    print(f"  Total: {train_imgs + val_imgs + test_imgs} images, {train_anns + val_anns + test_anns} annotations")
    print(f"  Location: {output_path}")
    print(f"\nYou can now train with:")
    print(f"  python train_detection.py --dataset-dir {output_path} --output-dir output/test --epochs 2 --batch-size 2")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create a small test COCO dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_dataset",
        help="Output directory for the test dataset"
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=5,
        help="Number of training images"
    )
    parser.add_argument(
        "--num-val",
        type=int,
        default=2,
        help="Number of validation images"
    )
    parser.add_argument(
        "--num-test",
        type=int,
        default=2,
        help="Number of test images"
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of object classes"
    )
    
    args = parser.parse_args()
    
    create_test_dataset(
        output_dir=args.output_dir,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        num_classes=args.num_classes
    )

