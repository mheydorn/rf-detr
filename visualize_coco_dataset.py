#!/usr/bin/env python3
"""
Interactive COCO Dataset Viewer

A GUI tool for visualizing and examining COCO format datasets with segmentation annotations.
Navigate through images, view masks, bounding boxes, and annotation details.

Usage:
    python visualize_coco_dataset.py --dataset-dir datasets/onion_defect_coco --split train
    python visualize_coco_dataset.py --dataset-dir datasets/onion_defect_coco --split train --show-only-positive
    
Controls:
    - Next/Previous buttons or arrow keys to navigate
    - Click on image to see details
    - Toggle masks/boxes visibility
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Rectangle, Polygon
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils


class COCOViewer:
    """Interactive viewer for COCO datasets with segmentation annotations."""
    
    def __init__(self, dataset_dir: str, split: str = "train", shuffle: bool = False, 
                 show_only_positive: bool = False):
        """
        Initialize the COCO viewer.
        
        Args:
            dataset_dir: Path to COCO dataset directory
            split: Dataset split to visualize ('train', 'valid', 'test')
            shuffle: Whether to shuffle the image order
            show_only_positive: If True, only show images with at least one segmentation annotation
        """
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        
        # Load COCO annotations
        ann_file = self.dataset_dir / split / "_annotations.coco.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")
        
        print(f"Loading COCO dataset from {ann_file}...")
        self.coco = COCO(str(ann_file))
        
        # Get all image IDs
        self.image_ids = self.coco.getImgIds()
        
        # Filter to only positive images if requested
        if show_only_positive:
            print("Filtering to show only images with segmentation annotations...")
            positive_image_ids = []
            for img_id in self.image_ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anns = self.coco.loadAnns(ann_ids)
                # Check if any annotation has a segmentation field
                has_segmentation = any('segmentation' in ann and ann['segmentation'] for ann in anns)
                if has_segmentation:
                    positive_image_ids.append(img_id)
            
            original_count = len(self.image_ids)
            self.image_ids = positive_image_ids
            print(f"Filtered {original_count} images → {len(self.image_ids)} images with segmentations")
            
            if len(self.image_ids) == 0:
                raise ValueError("No images found with segmentation annotations!")
        
        if shuffle:
            random.shuffle(self.image_ids)
        
        self.current_idx = 0
        self.show_masks = True
        self.show_boxes = True
        self.show_labels = True
        
        # Get category information
        self.categories = {cat['id']: cat for cat in self.coco.dataset['categories']}
        
        # Generate random colors for each category
        np.random.seed(42)
        self.colors = {}
        for cat_id in self.categories.keys():
            self.colors[cat_id] = np.random.rand(3)
        
        # Setup the figure
        self._setup_figure()
        
        # Display first image
        self._display_image()
        
        print(f"\nDataset loaded successfully!")
        print(f"Total images: {len(self.image_ids)}")
        print(f"Categories: {', '.join([cat['name'] for cat in self.categories.values()])}")
        print(f"\nControls:")
        print(f"  - Use Next/Previous buttons or Left/Right arrow keys")
        print(f"  - Press 'm' to toggle masks")
        print(f"  - Press 'b' to toggle bounding boxes")
        print(f"  - Press 'l' to toggle labels")
        print(f"  - Press 'r' to jump to random image")
        print(f"  - Press 'q' to quit")
    
    def _setup_figure(self):
        """Setup the matplotlib figure with controls."""
        self.fig = plt.figure(figsize=(14, 10))
        
        # Main image axes
        self.ax_img = plt.axes([0.1, 0.25, 0.8, 0.7])
        
        # Info text axes
        self.ax_info = plt.axes([0.1, 0.12, 0.8, 0.1])
        self.ax_info.axis('off')
        
        # Control buttons
        btn_height = 0.04
        btn_width = 0.12
        y_pos = 0.05
        
        ax_prev = plt.axes([0.1, y_pos, btn_width, btn_height])
        ax_next = plt.axes([0.23, y_pos, btn_width, btn_height])
        ax_random = plt.axes([0.36, y_pos, btn_width, btn_height])
        ax_masks = plt.axes([0.49, y_pos, btn_width, btn_height])
        ax_boxes = plt.axes([0.62, y_pos, btn_width, btn_height])
        ax_labels = plt.axes([0.75, y_pos, btn_width, btn_height])
        
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_random = Button(ax_random, 'Random')
        self.btn_masks = Button(ax_masks, 'Toggle Masks')
        self.btn_boxes = Button(ax_boxes, 'Toggle Boxes')
        self.btn_labels = Button(ax_labels, 'Toggle Labels')
        
        # Connect button callbacks
        self.btn_prev.on_clicked(lambda event: self._show_prev())
        self.btn_next.on_clicked(lambda event: self._show_next())
        self.btn_random.on_clicked(lambda event: self._show_random())
        self.btn_masks.on_clicked(lambda event: self._toggle_masks())
        self.btn_boxes.on_clicked(lambda event: self._toggle_boxes())
        self.btn_labels.on_clicked(lambda event: self._toggle_labels())
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
    
    def _load_image(self, image_id: int) -> Tuple[np.ndarray, Dict]:
        """Load an image and its metadata."""
        img_info = self.coco.loadImgs(image_id)[0]
        img_path = self.dataset_dir / self.split / img_info['file_name']
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img, img_info
    
    def _get_annotations(self, image_id: int) -> List[Dict]:
        """Get all annotations for an image."""
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        return self.coco.loadAnns(ann_ids)
    
    def _display_image(self):
        """Display the current image with annotations."""
        self.ax_img.clear()
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        # Get current image
        image_id = self.image_ids[self.current_idx]
        img, img_info = self._load_image(image_id)
        annotations = self._get_annotations(image_id)
        
        # Display image
        self.ax_img.imshow(img)
        self.ax_img.axis('off')
        
        # Draw annotations
        for ann in annotations:
            cat_id = ann['category_id']
            cat_name = self.categories[cat_id]['name']
            color = self.colors[cat_id]
            
            # Draw segmentation mask
            if self.show_masks and 'segmentation' in ann:
                if isinstance(ann['segmentation'], list):
                    # Polygon format
                    for seg in ann['segmentation']:
                        poly = np.array(seg).reshape(-1, 2)
                        polygon = Polygon(poly, fill=True, alpha=0.4, 
                                        facecolor=color, edgecolor=color, linewidth=2)
                        self.ax_img.add_patch(polygon)
                elif isinstance(ann['segmentation'], dict):
                    # RLE format
                    mask = mask_utils.decode(ann['segmentation'])
                    masked = np.zeros((*mask.shape, 4))
                    masked[:, :, :3] = color
                    masked[:, :, 3] = mask * 0.4
                    self.ax_img.imshow(masked)
            
            # Draw bounding box
            if self.show_boxes and 'bbox' in ann:
                x, y, w, h = ann['bbox']
                rect = Rectangle((x, y), w, h, fill=False, 
                               edgecolor=color, linewidth=2)
                self.ax_img.add_patch(rect)
                
                # Draw label
                if self.show_labels:
                    label = f"{cat_name}"
                    self.ax_img.text(x, y - 5, label, 
                                   bbox=dict(boxstyle='round,pad=0.3', 
                                           facecolor=color, alpha=0.8),
                                   fontsize=10, color='white', weight='bold')
        
        # Update info text
        info_lines = [
            f"Image {self.current_idx + 1}/{len(self.image_ids)} | "
            f"ID: {image_id} | "
            f"File: {img_info['file_name']}",
            f"Size: {img_info['width']}x{img_info['height']} | "
            f"Annotations: {len(annotations)}",
        ]
        
        # Add annotation details
        if annotations:
            cat_counts = {}
            total_area = 0
            for ann in annotations:
                cat_name = self.categories[ann['category_id']]['name']
                cat_counts[cat_name] = cat_counts.get(cat_name, 0) + 1
                total_area += ann.get('area', 0)
            
            cat_str = ', '.join([f"{name}: {count}" for name, count in cat_counts.items()])
            info_lines.append(f"Categories: {cat_str}")
            info_lines.append(f"Total annotation area: {total_area:.0f} px²")
        
        info_text = '\n'.join(info_lines)
        self.ax_info.text(0.5, 0.5, info_text, 
                         transform=self.ax_info.transAxes,
                         ha='center', va='center',
                         fontsize=10, family='monospace',
                         bbox=dict(boxstyle='round,pad=0.5', 
                                 facecolor='lightgray', alpha=0.8))
        
        plt.draw()
    
    def _show_next(self):
        """Show next image."""
        self.current_idx = (self.current_idx + 1) % len(self.image_ids)
        self._display_image()
    
    def _show_prev(self):
        """Show previous image."""
        self.current_idx = (self.current_idx - 1) % len(self.image_ids)
        self._display_image()
    
    def _show_random(self):
        """Show random image."""
        self.current_idx = random.randint(0, len(self.image_ids) - 1)
        self._display_image()
    
    def _toggle_masks(self):
        """Toggle mask visibility."""
        self.show_masks = not self.show_masks
        print(f"Masks: {'ON' if self.show_masks else 'OFF'}")
        self._display_image()
    
    def _toggle_boxes(self):
        """Toggle bounding box visibility."""
        self.show_boxes = not self.show_boxes
        print(f"Boxes: {'ON' if self.show_boxes else 'OFF'}")
        self._display_image()
    
    def _toggle_labels(self):
        """Toggle label visibility."""
        self.show_labels = not self.show_labels
        print(f"Labels: {'ON' if self.show_labels else 'OFF'}")
        self._display_image()
    
    def _on_key_press(self, event):
        """Handle keyboard events."""
        if event.key == 'right' or event.key == 'n':
            self._show_next()
        elif event.key == 'left' or event.key == 'p':
            self._show_prev()
        elif event.key == 'r':
            self._show_random()
        elif event.key == 'm':
            self._toggle_masks()
        elif event.key == 'b':
            self._toggle_boxes()
        elif event.key == 'l':
            self._toggle_labels()
        elif event.key == 'q':
            plt.close(self.fig)
    
    def show(self):
        """Show the viewer window."""
        plt.show()


def print_dataset_stats(dataset_dir: str, split: str):
    """Print detailed statistics about the dataset."""
    dataset_path = Path(dataset_dir)
    ann_file = dataset_path / split / "_annotations.coco.json"
    
    if not ann_file.exists():
        print(f"Error: Annotation file not found: {ann_file}")
        return
    
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    print("=" * 80)
    print(f"COCO Dataset Statistics - {split.upper()} Split")
    print("=" * 80)
    
    # Basic stats
    print(f"\nDataset: {dataset_dir}")
    print(f"Split: {split}")
    print(f"Total images: {len(data['images'])}")
    print(f"Total annotations: {len(data['annotations'])}")
    print(f"Categories: {len(data['categories'])}")
    
    # Category details
    print(f"\nCategory details:")
    for cat in data['categories']:
        cat_id = cat['id']
        cat_name = cat['name']
        cat_anns = [ann for ann in data['annotations'] if ann['category_id'] == cat_id]
        print(f"  - {cat_name} (ID: {cat_id}): {len(cat_anns)} annotations")
    
    # Image size distribution
    widths = [img['width'] for img in data['images']]
    heights = [img['height'] for img in data['images']]
    print(f"\nImage dimensions:")
    print(f"  Width:  min={min(widths)}, max={max(widths)}, avg={sum(widths)/len(widths):.1f}")
    print(f"  Height: min={min(heights)}, max={max(heights)}, avg={sum(heights)/len(heights):.1f}")
    
    # Annotation statistics
    if data['annotations']:
        areas = [ann.get('area', 0) for ann in data['annotations']]
        print(f"\nAnnotation areas:")
        print(f"  Min: {min(areas):.0f} px²")
        print(f"  Max: {max(areas):.0f} px²")
        print(f"  Avg: {sum(areas)/len(areas):.0f} px²")
        
        # Annotations per image
        from collections import Counter
        img_ann_counts = Counter(ann['image_id'] for ann in data['annotations'])
        ann_counts = list(img_ann_counts.values())
        print(f"\nAnnotations per image:")
        print(f"  Min: {min(ann_counts)}")
        print(f"  Max: {max(ann_counts)}")
        print(f"  Avg: {sum(ann_counts)/len(ann_counts):.2f}")
    
    # Segmentation format check
    seg_types = set()
    for ann in data['annotations']:
        if 'segmentation' in ann:
            if isinstance(ann['segmentation'], list):
                seg_types.add('polygon')
            elif isinstance(ann['segmentation'], dict):
                seg_types.add('RLE')
    print(f"\nSegmentation formats: {', '.join(seg_types)}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive COCO dataset viewer with segmentation support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # View training set
    python visualize_coco_dataset.py --dataset-dir datasets/onion_defect_coco --split train
    
    # View validation set with shuffled order
    python visualize_coco_dataset.py --dataset-dir datasets/onion_defect_coco --split valid --shuffle
    
    # Show only images with segmentation annotations
    python visualize_coco_dataset.py --dataset-dir datasets/onion_defect_coco --split train --show-only-positive
    
    # Print statistics only
    python visualize_coco_dataset.py --dataset-dir datasets/onion_defect_coco --split train --stats-only

Controls in viewer:
    Next/Previous buttons or Left/Right arrow keys - Navigate images
    Press 'm' - Toggle masks on/off
    Press 'b' - Toggle bounding boxes on/off
    Press 'l' - Toggle labels on/off
    Press 'r' - Jump to random image
    Press 'q' - Quit viewer
        """
    )
    
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to COCO dataset directory"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "valid", "test"],
        help="Dataset split to visualize (default: train)"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle image order"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only print dataset statistics without showing viewer"
    )
    parser.add_argument(
        "--show-only-positive",
        action="store_true",
        help="Only show images with at least one segmentation annotation"
    )
    
    args = parser.parse_args()
    
    # Print statistics
    print_dataset_stats(args.dataset_dir, args.split)
    
    if not args.stats_only:
        print(f"\nStarting interactive viewer...")
        print("=" * 80)
        
        # Create and show viewer
        viewer = COCOViewer(
            dataset_dir=args.dataset_dir,
            split=args.split,
            shuffle=args.shuffle,
            show_only_positive=args.show_only_positive
        )
        viewer.show()
    else:
        print("\n(Use without --stats-only flag to launch interactive viewer)")


if __name__ == "__main__":
    main()

