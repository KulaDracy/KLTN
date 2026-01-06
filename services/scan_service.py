import json
import time
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path
from paddleocr import PaddleOCR


DEFAULT_DPI = 150
MAX_DPI = 600
DPI_STEP = 50
MIN_WIDTH_PX = 1200

POPPLER_PATH = None


def pdf_page_to_image(pdf_path, page_number, dpi):
    images = convert_from_path(
        pdf_path,
        dpi=dpi,
        first_page=page_number,
        last_page=page_number,
        poppler_path=POPPLER_PATH
    )
    return images[0]


def best_dpi_render(pdf, page, min_width):
    dpi = DEFAULT_DPI
    final_image = None

    while dpi <= MAX_DPI:
        img = pdf_page_to_image(pdf, page, dpi)
        if img.size[0] >= min_width:
            return img, dpi
        final_image = img
        dpi += DPI_STEP

    return final_image, dpi - DPI_STEP


def rotate_image_cv2(img, angle):
    if angle == 0:
        return img
    elif angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def detect_rotation_paddle(img_cv2, ocr_angle_model):
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

    result = ocr_angle_model.ocr(img_rgb, det=False, rec=False, cls=True)

    angle = 0
    if result and result[0]:
        angle = int(result[0][0])
    return angle


def process_pdf(pdf_path: str, out_dir: str) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    angle_model = PaddleOCR(use_angle_cls=True, lang="en", det=False, rec=False)

    pages = convert_from_path(pdf_path, dpi=DEFAULT_DPI, poppler_path=POPPLER_PATH)
    total_pages = len(pages)
    del pages

    manifest = {
        "file": Path(pdf_path).name,
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pages": []
    }

    for idx in range(total_pages):
        page_num = idx + 1

        img_pil, dpi_used = best_dpi_render(pdf_path, page_num, MIN_WIDTH_PX)
        img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        angle = detect_rotation_paddle(img_cv2, angle_model)
        img_rot = rotate_image_cv2(img_cv2, angle)
        final_pil = Image.fromarray(cv2.cvtColor(img_rot, cv2.COLOR_BGR2RGB))

        save_path = out / f"page_{page_num:03d}.png"
        final_pil.save(save_path)

        manifest["pages"].append({
            "page_number": page_num,
            "dpi": dpi_used,
            "rotation": angle,
            "file": str(save_path),
            "width": final_pil.size[0],
            "height": final_pil.size[1]
        })

    json_path = out / "manifest.json"
    with open(json_path, "w", encoding="utf8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return str(json_path)
