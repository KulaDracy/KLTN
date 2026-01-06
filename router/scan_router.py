from fastapi import APIRouter, UploadFile, File
from services.scan_service import process_pdf
from services.mark_service import process_manifest
from pathlib import Path
import shutil

router = APIRouter()

@router.post("/upload")
async def upload(pdf: UploadFile = File(...)):
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)

    pdf_path = temp_dir / pdf.filename
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(pdf.file, f)

    # Step 1 – Scan PDF
    scan_output = temp_dir / "scan_output"
    manifest_path = process_pdf(str(pdf_path), str(scan_output))

    # Step 2 – OCR & flag
    marked_json = scan_output / "marked.json"
    final_json = process_manifest(manifest_path, str(marked_json))

    return {"status": "ok", "result": final_json}
