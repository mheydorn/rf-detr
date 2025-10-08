#!/usr/bin/env python3
"""Debug why mask to polygon conversion is failing."""

import json
import sys
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

# Import the mask_to_polygon function from inference
sys.path.insert(0, str(Path(__file__).parent))
from inference import mask_to_polygon

def analyze_mask_file(mask_path):
    """Analyze a single mask file to see why polygon conversion might fail."""
    
    # Load mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        print(f"ERROR: Could not load mask from {mask_path}")
        return
    
    print(f"\n{'='*80}")
    print(f"Analyzing: {mask_path.name}")
    print(f"{'='*80}\n")
    
    # Mask properties
    print(f"Mask shape: {mask.shape}")
    print(f"Mask dtype: {mask.dtype}")
    print(f"Unique values: {np.unique(mask)}")
    print(f"Min: {np.min(mask)}, Max: {np.max(mask)}")
    print(f"Non-zero pixels: {np.sum(mask > 0)}")
    print()
    
    # For combined masks, there might be multiple instance IDs
    unique_vals = np.unique(mask)
    instance_ids = unique_vals[unique_vals > 0]
    
    print(f"Number of instances in this mask: {len(instance_ids)}")
    print(f"Instance IDs: {instance_ids.tolist()}")
    print()
    
    # Try converting each instance
    for inst_id in instance_ids:
        print(f"Instance {inst_id}:")
        instance_mask = (mask == inst_id).astype(np.uint8)
        
        num_pixels = np.sum(instance_mask > 0)
        print(f"  Pixels: {num_pixels}")
        
        # Try to convert to polygon
        polygon = mask_to_polygon(instance_mask)
        
        if polygon:
            print(f"  ✅ Polygon conversion SUCCESS: {len(polygon)//2} points")
        else:
            print(f"  ❌ Polygon conversion FAILED")
            
            # Debug why it failed
            mask_uint8 = (instance_mask > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                print(f"     Reason: No contours found")
            else:
                print(f"     Found {len(contours)} contours")
                for i, contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    print(f"     Contour {i}: {len(contour)} points, area={area:.1f}")
                    
                    if len(contour) < 3:
                        print(f"       Issue: Too few points (< 3)")
                    else:
                        flat = contour.flatten()
                        if len(flat) < 6:
                            print(f"       Issue: Flattened contour < 6 values ({len(flat)})")
        print()

def main():
    masks_dir = Path('latest_out/masks')
    
    if not masks_dir.exists():
        print(f"ERROR: Masks directory not found: {masks_dir}")
        sys.exit(1)
    
    mask_files = list(masks_dir.glob('*.png'))
    
    if not mask_files:
        print(f"ERROR: No mask files found in {masks_dir}")
        sys.exit(1)
    
    print(f"Found {len(mask_files)} mask files")
    
    # Analyze first few masks
    num_to_analyze = 5
    print(f"\nAnalyzing first {num_to_analyze} mask files...\n")
    
    for mask_file in sorted(mask_files)[:num_to_analyze]:
        analyze_mask_file(mask_file)

if __name__ == '__main__':
    main()

