#!/usr/bin/env python3
"""
FASTAPI PIPELINE PREPROCESS SERVICE

Pipeline:
1) pdf2img.py    → PDF → images + manifest.json
2) preprocess.py → undo PDF rotation → manifest_fixed.json
3) scan.py       → detect table + OCR rotate → manifest_scanned.json
"""

import sys
import uuid
import time
import shutil
import subprocess
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse


# ============================================================
# CONFIG
# ============================================================

BASE_TEMP_DIR = Path("temp")
BASE_TEMP_DIR.mkdir(exist_ok=True)

PDF2IMG_SCRIPT = "pdf2img.py"
PREPROCESS_SCRIPT = "preprocess.py"
SCAN_SCRIPT = "scan.py"


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="PDF Preprocess Pipeline Service",
    description="Run full PDF preprocess pipeline (pdf2img → preprocess → scan)",
    version="1.0.0"
)


# ============================================================
# UTILS
# ============================================================

def run_cmd(cmd, title):
    start = time.time()

    proc = subprocess.run(
        cmd,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    elapsed = time.time() - start

    if proc.returncode != 0:
        raise RuntimeError(
            f"{title} FAILED\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )

    return {
        "title": title,
        "time": round(elapsed, 2),
        "stdout": proc.stdout
    }


# ============================================================
# API ENDPOINT
# ============================================================

@app.post("/pipeline/preprocess")
async def pipeline_preprocess(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    job_id = str(uuid.uuid4())
    job_dir = BASE_TEMP_DIR / job_id

    step1_dir = job_dir / "step1_pdf2img"
    step2_dir = job_dir / "step2_preprocess"
    step3_dir = job_dir / "step3_scan"

    input_pdf = job_dir / file.filename

    try:
        # Prepare dirs
        job_dir.mkdir(parents=True, exist_ok=True)
        step1_dir.mkdir()
        step2_dir.mkdir()
        step3_dir.mkdir()

        # Save PDF
        with open(input_pdf, "wb") as f:
            shutil.copyfileobj(file.file, f)

        logs = []

        # ----------------------------------------------------
        # STEP 1
        # ----------------------------------------------------
        logs.append(run_cmd(
            [
                sys.executable,
                PDF2IMG_SCRIPT,
                str(input_pdf),
                str(step1_dir)
            ],
            "STEP 1: PDF → IMAGE"
        ))

        manifest_step1 = step1_dir / "manifest.json"
        if not manifest_step1.exists():
            raise RuntimeError("manifest.json not found after STEP 1")

        # ----------------------------------------------------
        # STEP 2
        # ----------------------------------------------------
        logs.append(run_cmd(
            [
                sys.executable,
                PREPROCESS_SCRIPT,
                str(manifest_step1),
                str(step2_dir)
            ],
            "STEP 2: PREPROCESS"
        ))

        manifest_step2 = step2_dir / "manifest_fixed.json"
        if not manifest_step2.exists():
            raise RuntimeError("manifest_fixed.json not found after STEP 2")

        # ----------------------------------------------------
        # STEP 3
        # ----------------------------------------------------
        logs.append(run_cmd(
            [
                sys.executable,
                SCAN_SCRIPT,
                str(manifest_step2),
                str(step3_dir)
            ],
            "STEP 3: SCAN + OCR + TABLE"
        ))

        manifest_final = step3_dir / "manifest_scanned.json"
        if not manifest_final.exists():
            raise RuntimeError("manifest_scanned.json not found after STEP 3")

    except Exception as e:
        # Cleanup on failure
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse({
        "job_id": job_id,
        "input_pdf": file.filename,
        "output_dir": str(step3_dir.resolve()),
        "final_manifest": str(manifest_final.resolve()),
        "steps": logs
    })


@app.get("/health")
def health():
    return {"status": "ok"}
