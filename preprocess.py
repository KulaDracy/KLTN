#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


# ============================
# Utility: rotate image
# ============================
def rotate_img(img, angle):
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)   # undo scan rotation
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


# ============================================
# MAIN FUNCTION
# ============================================
def preprocess(json_path, out_dir):
    json_path = Path(json_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf8") as f:
        manifest = json.load(f)

    pages = manifest["pages"]

    new_manifest = {
        "file": manifest.get("file", ""),
        "dpi": manifest.get("dpi", 200),
        "pages": []
    }

    print("\n=== PREPROCESS START ===")

    for page in pages:
        page_number   = page["page_number"]
        img_path      = Path(page["file"])
        meta_rotation = page.get("meta_rotation", 0)  # rotation stored in PDF
        width         = page["width"]
        height        = page["height"]

        print(f"\nPage {page_number}: meta_rotation = {meta_rotation}")

        # Load image (BGR)
        img = cv2.imread(str(img_path))
        if img is None:
            print(f" Cannot read image: {img_path}")
            continue

        # Apply reverse rotation:
        # meta_rotation = 90  -> image must be rotated -90
        applied_rotation = (360 - meta_rotation) % 360

        print(f" Applied rotation => {applied_rotation}")

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
            "final_rotation": 0
        })

    # Save new manifest
    with open(out_dir / "manifest_fixed.json", "w", encoding="utf8") as f:
        json.dump(new_manifest, f, indent=2, ensure_ascii=False)

    print("\n=== DONE! Saved to:", out_dir, "===")


# ============================================
# CLI ENTRY
# ============================================
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python preprocess.py manifest.json output_folder")
        sys.exit(1)

    preprocess(sys.argv[1], sys.argv[2])
