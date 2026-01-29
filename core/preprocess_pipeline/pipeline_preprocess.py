#!/usr/bin/env python3
"""
PIPELINE PREPROCESS PDF - With Stamp Removal

Steps:
1) pdf2img.py        → PDF → images + manifest.json
2) preprocess.py     → remove stamp (optional) + undo PDF rotation → manifest_fixed.json
3) scan.py           → detect table + OCR rotate → manifest_scanned.json

All working directories are inside ./temp

USAGE:
    # Without stamp removal (default)
    python pipeline_preprocess.py input.pdf
    
    # With stamp removal
    python pipeline_preprocess.py input.pdf --remove-stamp
    
    # With stamp removal (aggressive mode)
    python pipeline_preprocess.py input.pdf --remove-stamp --stamp-aggressive
    
    # With stamp removal (inpaint method)
    python pipeline_preprocess.py input.pdf --remove-stamp --stamp-method inpaint
"""

import sys
import subprocess
from pathlib import Path
import shutil
import time
import argparse


# ============================================================
# CONFIG
# ============================================================

TEMP_DIR = Path("temp")

STEP1_DIR = TEMP_DIR / "step1_pdf2img"
STEP2_DIR = TEMP_DIR / "step2_preprocess"
STEP3_DIR = TEMP_DIR / "step3_scan"

PDF2IMG_SCRIPT = "pdf2img.py"
PREPROCESS_SCRIPT = "preprocess.py"
SCAN_SCRIPT = "scan.py"


# ============================================================
# UTILS
# ============================================================

def run_cmd(cmd, title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(" ".join(cmd))
    print()

    start = time.time()
    result = subprocess.run(cmd, shell=False)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n ERROR in step: {title}")
        sys.exit(1)

    print(f"\n DONE ({elapsed:.2f}s)")


# ============================================================
# PIPELINE
# ============================================================

def run_pipeline(
    pdf_path, 
    remove_stamp=False,
    stamp_method="color",
    stamp_aggressive=False
):
    """
    Run full preprocessing pipeline
    
    Parameters:
    -----------
    pdf_path : str or Path
        Path to input PDF file
    remove_stamp : bool
        Enable stamp removal in step 2 (default: False)
    stamp_method : str
        Stamp removal method: "color" or "inpaint" (default: "color")
    stamp_aggressive : bool
        Use aggressive stamp removal (may remove red text) (default: False)
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        print(f" ERROR: PDF file not found: {pdf_path}")
        sys.exit(1)

    # Clean temp directory
    if TEMP_DIR.exists():
        print(" Cleaning temp directory...")
        shutil.rmtree(TEMP_DIR)

    STEP1_DIR.mkdir(parents=True)
    STEP2_DIR.mkdir(parents=True)
    STEP3_DIR.mkdir(parents=True)

    print("\n" + "=" * 70)
    print("PIPELINE CONFIGURATION")
    print("=" * 70)
    print(f" Working directory: {TEMP_DIR.resolve()}")
    print(f" Input PDF: {pdf_path}")
    print(f" Stamp removal: {'ENABLED' if remove_stamp else 'DISABLED'}")
    if remove_stamp:
        print(f"   - Method: {stamp_method}")
        print(f"   - Aggressive: {stamp_aggressive}")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1: PDF → IMAGE
    # --------------------------------------------------------
    run_cmd(
        [
            sys.executable,
            PDF2IMG_SCRIPT,
            str(pdf_path),
            str(STEP1_DIR)
        ],
        "STEP 1: PDF -> IMAGE (pdf2img.py)"
    )

    manifest_step1 = STEP1_DIR / "manifest.json"

    # --------------------------------------------------------
    # STEP 2: PREPROCESS (STAMP REMOVAL + UNDO META ROTATION)
    # --------------------------------------------------------
    preprocess_cmd = [
        sys.executable,
        PREPROCESS_SCRIPT,
        str(manifest_step1),
        str(STEP2_DIR)
    ]
    
    # Add stamp removal options if enabled
    if remove_stamp:
        preprocess_cmd.append("--remove-stamp")
        preprocess_cmd.append("--stamp-method")
        preprocess_cmd.append(stamp_method)
        if stamp_aggressive:
            preprocess_cmd.append("--stamp-aggressive")
    
    run_cmd(
        preprocess_cmd,
        f"STEP 2: PREPROCESS ({'with stamp removal' if remove_stamp else 'rotation only'})"
    )

    manifest_step2 = STEP2_DIR / "manifest_fixed.json"

    # --------------------------------------------------------
    # STEP 3: SCAN + OCR + TABLE DETECT
    # --------------------------------------------------------
    run_cmd(
        [
            sys.executable,
            SCAN_SCRIPT,
            str(manifest_step2),
            str(STEP3_DIR)
        ],
        "STEP 3: SCAN + OCR + TABLE (scan.py)"
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("✓ PIPELINE FINISHED SUCCESSFULLY")
    print("=" * 70)
    print(f"\n Output directories:")
    print(f"   Step 1 (PDF->Images): {STEP1_DIR.resolve()}")
    print(f"   Step 2 (Preprocessed): {STEP2_DIR.resolve()}")
    print(f"   Step 3 (Scanned): {STEP3_DIR.resolve()}")
    print(f"\n Final manifest: {(STEP3_DIR / 'manifest_scanned.json').resolve()}")
    if remove_stamp:
        print(f"\n ⚠ Stamp removal was ENABLED")
        print(f"   Check {STEP2_DIR.resolve()} for results")
    print("=" * 70)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PDF preprocessing pipeline with optional stamp removal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (no stamp removal)
  python pipeline_preprocess.py input.pdf
  
  # With stamp removal
  python pipeline_preprocess.py input.pdf --remove-stamp
  
  # With stamp removal (aggressive mode)
  python pipeline_preprocess.py input.pdf --remove-stamp --stamp-aggressive
  
  # With stamp removal (inpaint method)
  python pipeline_preprocess.py input.pdf --remove-stamp --stamp-method inpaint
        """
    )
    parser.add_argument(
        "pdf",
        help="Input PDF file"
    )
    parser.add_argument(
        "--remove-stamp",
        action="store_true",
        help="Enable stamp removal in preprocessing step"
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
    
    run_pipeline(
        args.pdf,
        remove_stamp=args.remove_stamp,
        stamp_method=args.stamp_method,
        stamp_aggressive=args.stamp_aggressive
    )