#!/usr/bin/env python3
"""
PREPROCESS MODULE - With Stamp Removal

Steps:
1. Load image from manifest
2. Remove stamp (if enabled)
3. Apply rotation correction
4. Save processed image
"""

import json
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


# ============================
# STAMP REMOVAL FUNCTIONS
# ============================

RED_STAMP_HSV_RANGES = [
    {"lower": np.array([0, 50, 50]), "upper": np.array([10, 255, 255])},
    {"lower": np.array([160, 50, 50]), "upper": np.array([180, 255, 255])},
]

PINK_STAMP_HSV_RANGES = [
    {"lower": np.array([150, 30, 100]), "upper": np.array([180, 200, 255])},
    {"lower": np.array([0, 30, 100]), "upper": np.array([10, 200, 255])},
]


def remove_stamp_by_color(img, color_ranges=None, fill_color=(255, 255, 255), aggressive=False):
    """Xóa dấu mộc dựa trên màu HSV"""
    if color_ranges is None:
        if aggressive:
            color_ranges = RED_STAMP_HSV_RANGES + PINK_STAMP_HSV_RANGES
        else:
            color_ranges = RED_STAMP_HSV_RANGES
    
    # Chuyển sang HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Tạo mask
    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    
    for color_range in color_ranges:
        lower = color_range["lower"]
        upper = color_range["upper"]
        mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Morphological operations
    kernel_size = 5 if aggressive else 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Thay thế vùng đã detect
    result = img.copy()
    result[combined_mask > 0] = fill_color
    
    return result


def remove_stamp_inpaint(img, color_ranges=None, inpaint_radius=3, aggressive=False):
    """Xóa dấu mộc bằng inpainting"""
    if color_ranges is None:
        if aggressive:
            color_ranges = RED_STAMP_HSV_RANGES + PINK_STAMP_HSV_RANGES
        else:
            color_ranges = RED_STAMP_HSV_RANGES
    
    # Chuyển sang HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Tạo mask
    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    
    for color_range in color_ranges:
        lower = color_range["lower"]
        upper = color_range["upper"]
        mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Morphological operations
    kernel_size = 5 if aggressive else 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Inpainting
    inpaint_radius = 5 if aggressive else 3
    result = cv2.inpaint(img, combined_mask, inpaint_radius, cv2.INPAINT_TELEA)
    
    return result


# ============================
# Utility: rotate image
# ============================
def rotate_img(img, angle):
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


# ============================================
# MAIN FUNCTION
# ============================================
def preprocess(
    json_path, 
    out_dir, 
    remove_stamp=False,
    stamp_method="color",
    stamp_aggressive=False
):
    """
    Preprocess images: remove stamp + rotation correction
    
    Parameters:
    -----------
    json_path : str or Path
        Path to manifest.json from step1
    out_dir : str or Path
        Output directory
    remove_stamp : bool
        Enable stamp removal (default: False)
    stamp_method : str
        Stamp removal method: "color" or "inpaint" (default: "color")
    stamp_aggressive : bool
        Use aggressive stamp removal (may remove red text) (default: False)
    """
    json_path = Path(json_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf8") as f:
        manifest = json.load(f)

    pages = manifest["pages"]

    new_manifest = {
        "file": manifest.get("file", ""),
        "dpi": manifest.get("dpi", 200),
        "pages": [],
        "preprocessing": {
            "stamp_removal": remove_stamp,
            "stamp_method": stamp_method if remove_stamp else None,
            "stamp_aggressive": stamp_aggressive if remove_stamp else None
        }
    }

    print("\n=== PREPROCESS START ===")
    print(f"Stamp removal: {'ENABLED' if remove_stamp else 'DISABLED'}")
    if remove_stamp:
        print(f"  Method: {stamp_method}")
        print(f"  Aggressive: {stamp_aggressive}")

    for page in pages:
        page_number   = page["page_number"]
        img_path      = Path(page["file"])
        meta_rotation = page.get("meta_rotation", 0)
        width         = page["width"]
        height        = page["height"]

        print(f"\nPage {page_number}: meta_rotation = {meta_rotation}")

        # Load image (BGR)
        img = cv2.imread(str(img_path))
        if img is None:
            print(f" Cannot read image: {img_path}")
            continue

        # STEP 1: Remove stamp (if enabled)
        if remove_stamp:
            print(f"  Removing stamp using {stamp_method} method...")
            try:
                if stamp_method == "inpaint":
                    img = remove_stamp_inpaint(img, aggressive=stamp_aggressive)
                else:  # default to color
                    img = remove_stamp_by_color(img, aggressive=stamp_aggressive)
                print(f"  ✓ Stamp removed")
            except Exception as e:
                print(f"  ✗ Stamp removal failed: {e}")
                print(f"  Continuing without stamp removal...")

        # STEP 2: Apply reverse rotation
        applied_rotation = (360 - meta_rotation) % 360
        print(f"  Applied rotation => {applied_rotation}")
        img_fixed = rotate_img(img, applied_rotation)

        # Save new corrected image
        out_file = out_dir / f"page_{page_number:03d}.png"
        Image.fromarray(cv2.cvtColor(img_fixed, cv2.COLOR_BGR2RGB)).save(out_file)

        h, w = img_fixed.shape[:2]

        # Write to new manifest
        new_manifest["pages"].append({
            "page_number": page_number,
            "file": str(out_file.resolve()),
            "width": w,
            "height": h,
            "meta_rotation": meta_rotation,
            "applied_rotation": applied_rotation,
            "final_rotation": 0,
            "stamp_removed": remove_stamp
        })

    # Save new manifest
    with open(out_dir / "manifest_fixed.json", "w", encoding="utf8") as f:
        json.dump(new_manifest, f, indent=2, ensure_ascii=False)

    print("\n=== DONE! Saved to:", out_dir, "===")


# ============================================
# CLI ENTRY
# ============================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Preprocess images: remove stamp + rotation correction"
    )
    parser.add_argument(
        "manifest",
        help="Path to manifest.json from step1"
    )
    parser.add_argument(
        "output",
        help="Output directory"
    )
    parser.add_argument(
        "--remove-stamp",
        action="store_true",
        help="Enable stamp removal"
    )
    parser.add_argument(
        "--stamp-method",
        choices=["color", "inpaint"],
        default="color",
        help="Stamp removal method (default: color)"
    )
    parser.add_argument(
        "--stamp-aggressive",
        action="store_true",
        help="Use aggressive stamp removal (may remove red text)"
    )
    
    args = parser.parse_args()
    
    preprocess(
        args.manifest,
        args.output,
        remove_stamp=args.remove_stamp,
        stamp_method=args.stamp_method,
        stamp_aggressive=args.stamp_aggressive
    )