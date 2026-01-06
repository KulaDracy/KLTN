from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pathlib import Path
import json
import shutil
import uuid
import asyncio

from .scanandmark import process_manifest   # import hàm đã viết


router = APIRouter(prefix="/mark", tags=["Mark OCR"])


# ---------------- UTILS ----------------

async def save_uploaded_file(file: UploadFile, dest: Path):
    """Lưu file upload vào thư mục tạm"""
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return dest


# ---------------- ROUTES ----------------


@router.post("/manifest")
async def mark_manifest(
    manifest_file: UploadFile = File(...),
    output_name: str = Form(None)
):
    """
    Nhận manifest.json → chạy OCR gắn cờ → trả JSON kết quả
    """

    # tạo tên file random nếu không có output_name
    if not output_name:
        output_name = f"marked_{uuid.uuid4().hex}.json"

    temp_dir = Path("temp_manifests")
    temp_dir.mkdir(exist_ok=True)

    uploaded_manifest_path = temp_dir / f"upload_{uuid.uuid4().hex}.json"

    # lưu file upload
    await save_uploaded_file(manifest_file, uploaded_manifest_path)

    output_dir = Path("processed")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / output_name

    # chạy OCR & gắn flag — chạy dưới thread để không block event loop
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: process_manifest(str(uploaded_manifest_path), str(output_path))
    )

    # đọc kết quả
    with open(output_path, "r", encoding="utf8") as f:
        data = json.load(f)

    return JSONResponse(content=data)



@router.post("/path")
async def mark_manifest_from_path(
    manifest_path: str = Form(...),
    output_name: str = Form(None)
):
    """
    API nhận đường dẫn manifest.json có sẵn trên server
    """

    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return JSONResponse({"error": "Manifest not found"}, status_code=404)

    if not output_name:
        output_name = f"marked_{uuid.uuid4().hex}.json"

    output_dir = Path("processed")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / output_name

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: process_manifest(str(manifest_path), str(output_path))
    )

    with open(output_path, "r", encoding="utf8") as f:
        data = json.load(f)

    return JSONResponse(content=data)
