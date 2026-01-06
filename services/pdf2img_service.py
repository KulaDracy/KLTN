#!/usr/bin/env python3
"""
PDF → Image API Service
"""

import uuid
import json
import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from pdf2image import convert_from_path
from PIL import Image
import PyPDF2


# ============================================================
# CONFIG
# ============================================================

BASE_TEMP_DIR = Path("temp")
BASE_TEMP_DIR.mkdir(exist_ok=True)

DPI = 200
POPPLER_PATH = None   # Windows: set Poppler/bin if needed


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="PDF to Image Service",
    description="Convert PDF to images and extract rotation metadata",
    version="1.0.0"
)


# ============================================================
# UTILS
# ============================================================

def read_pdf_rotations(pdf_path: Path) -> List[int]:
    rotations = []
    reader = PyPDF2.PdfReader(str(pdf_path))

    for page in reader.pages:
        rotate = page.get("/Rotate", 0)
        try:
            rotate = int(rotate) % 360
        except Exception:
            rotate = 0
        rotations.append(rotate)

    return rotations


# ============================================================
# CORE LOGIC
# ============================================================

def process_pdf(pdf_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    rotations = read_pdf_rotations(pdf_path)

    pages = convert_from_path(
        pdf_path,
        dpi=DPI,
        poppler_path=POPPLER_PATH
    )

    manifest = {
        "file": pdf_path.name,
        "dpi": DPI,
        "pages": []
    }

    for idx, pil_img in enumerate(pages, start=1):
        meta_rotation = rotations[idx - 1]

        width, height = pil_img.size
        orientation = "portrait" if height >= width else "landscape"

        out_file = out_dir / f"page_{idx:03d}.png"
        pil_img.save(out_file)

        manifest["pages"].append({
            "page_number": idx,
            "file": str(out_file.resolve()),
            "width": width,
            "height": height,
            "dpi": DPI,
            "meta_rotation": meta_rotation,
            "orientation": orientation,
            "pdf_portrait": height >= width,
            "pdf_landscape": width > height
        })

    with open(out_dir / "manifest.json", "w", encoding="utf8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


# ============================================================
# API ENDPOINTS
# ============================================================

@app.post("/pdf2img")
async def pdf_to_img_api(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    job_id = str(uuid.uuid4())
    job_dir = BASE_TEMP_DIR / job_id
    input_pdf = job_dir / file.filename
    output_dir = job_dir / "images"

    job_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded PDF
    with open(input_pdf, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        manifest = process_pdf(input_pdf, output_dir)
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse({
        "job_id": job_id,
        "pages": len(manifest["pages"]),
        "output_dir": str(output_dir.resolve()),
        "manifest": manifest
    })


@app.get("/health")
def health():
    return {"status": "ok"}
