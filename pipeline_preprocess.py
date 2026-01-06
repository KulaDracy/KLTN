#!/usr/bin/env python3
"""
PIPELINE PREPROCESS PDF

Steps:
1) pdf2img.py        → PDF → images + manifest.json
2) preprocess.py     → undo PDF rotation → manifest_fixed.json
3) scan.py            → detect table + OCR rotate → manifest_scanned.json

All working directories are inside ./temp
"""

import sys
import subprocess
from pathlib import Path
import shutil
import time


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

def run_pipeline(pdf_path):
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        print(" PDF file not found:", pdf_path)
        sys.exit(1)

    # Clean temp directory
    if TEMP_DIR.exists():
        print(" Cleaning temp directory...")
        shutil.rmtree(TEMP_DIR)

    STEP1_DIR.mkdir(parents=True)
    STEP2_DIR.mkdir(parents=True)
    STEP3_DIR.mkdir(parents=True)

    print("\n Working directory:", TEMP_DIR.resolve())

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
    # STEP 2: PREPROCESS (UNDO META ROTATION)
    # --------------------------------------------------------
    run_cmd(
        [
            sys.executable,
            PREPROCESS_SCRIPT,
            str(manifest_step1),
            str(STEP2_DIR)
        ],
        "STEP 2: PREPROCESS (preprocess.py)"
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
    print("PIPELINE FINISHED SUCCESSFULLY")
    print("=" * 70)
    print(" Final output directory:")
    print(" ", STEP3_DIR.resolve())
    print("\n Final manifest:")
    print(" ", (STEP3_DIR / "manifest_scanned.json").resolve())
    print("=" * 70)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline_preprocess.py input.pdf")
        sys.exit(1)

    run_pipeline(sys.argv[1])
