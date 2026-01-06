#!/usr/bin/env python3
import sys
import json
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path

try:
    import PyPDF2
except ImportError:
    print("Bạn cần cài đặt PyPDF2: pip install PyPDF2")
    sys.exit(1)

# ==========================
# CONFIG
# ==========================
DPI = 200
POPPLER_PATH = None   # Windows thì set đường dẫn Poppler/bin

# ==========================
# READ PDF ROTATION METADATA
# ==========================
def get_pdf_rotations(pdf_path):
    rotations = []
    pdf = PyPDF2.PdfReader(pdf_path)
    for i, page in enumerate(pdf.pages):
        rotate = page.get("/Rotate", 0)
        try:
            rotate = int(rotate)
        except:
            rotate = 0
        rotations.append(rotate)
    return rotations

# ==========================
# MAIN EXPORT FUNCTION
# ==========================
def process_pdf(pdf_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Đọc rotation metadata trong PDF...")
    pdf_rotations = get_pdf_rotations(pdf_path)

    print("Chuyển PDF sang ảnh...")
    pages = convert_from_path(pdf_path, dpi=DPI, poppler_path=POPPLER_PATH)

    manifest = {
        "file": Path(pdf_path).name,
        "dpi": DPI,
        "pages": []
    }

    for idx, pil_img in enumerate(pages, start=1):
        print(f"Trang {idx}/{len(pages)}")

        meta_rotate = pdf_rotations[idx - 1]

        # pdf2image đã tự xoay ảnh theo meta_rotate → ảnh nhận được là đúng chiều
        width, height = pil_img.size
        orientation = "portrait" if height >= width else "landscape"

        # Save the image
        out_file = out / f"page_{idx:03d}.png"
        pil_img.save(out_file)

        manifest["pages"].append({
            "page_number": idx,
            "file": str(out_file.resolve()),
            "width": width,
            "height": height,
            "dpi": DPI,
            "meta_rotation": meta_rotate,
            "orientation": orientation,
            "pdf_portrait": height >= width,
            "pdf_landscape": width > height
        })

    # Save manifest
    with open(out / "manifest.json", "w", encoding="utf8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\nDONE", out)


# ==========================
# CLI
# ==========================
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python pdftoimg.py input.pdf output_folder")
        sys.exit(1)

    process_pdf(sys.argv[1], sys.argv[2])
