# Full pipeline: PDF -> crop table cells -> generate PaddleOCR dataset -> fine-tune PaddleOCR
# Works on Windows
# Requires: pdf2image, opencv-python, paddleocr, paddlepaddle, camelot-py[cv]

import os
import cv2
import json
import shutil
from pdf2image import convert_from_path
import camelot

# =============================
# 1. Convert PDF to images
# =============================
def pdf_to_images(pdf_path, dpi=300):
    pages = convert_from_path(pdf_path, dpi=dpi)
    imgs = []
    for i, page in enumerate(pages):
        out = f"temp/page_{i}.png"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        page.save(out)
        imgs.append(out)
    return imgs

# =============================
# 2. Extract tables using Camelot
# =============================
def extract_tables(page_img):
    tables = camelot.read_pdf(page_img.replace(".png", ".pdf"), flavor='lattice')
    return tables

# =============================
# 3. Crop each cell as image
# =============================
def crop_cells_from_table(table, page_img, out_dir="dataset/train/images"):
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(page_img)
    h, w, _ = img.shape

    cell_items = []
    for r in range(len(table.cells)):
        for c in range(len(table.cells[r])):
            cell = table.cells[r][c]
            x1, y1, x2, y2 = cell.x1, cell.y1, cell.x2, cell.y2
            cx1, cy1 = int(x1), int(h - y2)
            cx2, cy2 = int(x2), int(h - y1)
            crop = img[cy1:cy2, cx1:cx2]

            out_path = os.path.join(out_dir, f"cell_{r}_{c}_{os.path.basename(page_img)}")
            cv2.imwrite(out_path, crop)

            text = table.df.iloc[r, c]
            cell_items.append((out_path, text))

    return cell_items

# =============================
# 4. Build PaddleOCR labels file
# =============================
def write_paddle_labels(items, label_file):
    with open(label_file, "w", encoding="utf-8") as f:
        for img, txt in items:
            f.write(f"images/{os.path.basename(img)}\t{txt}\n")

# =============================
# 5. Create config for PaddleOCR fine-tune
# =============================
def generate_paddle_config():
    cfg = """
Global:
  use_gpu: False
  epoch_num: 20
  save_model_dir: ./output/rec_financial
  pretrained_model: ./pretrained/ch_ppocr_server_v2.0_rec_pretrained

Train:
  dataset:
    name: SimpleDataSet
    data_dir: dataset/train
    label_file_list:
      - dataset/train/labels.txt

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: dataset/val
    label_file_list:
      - dataset/val/labels.txt
"""
    with open("rec_config.yml", "w", encoding="utf-8") as f:
        f.write(cfg)

# =============================
# 6. MAIN PIPELINE
# =============================
def pipeline(pdf_path):
    # 1. PDF -> images
    pages = pdf_to_images(pdf_path)

    all_items = []

    for p in pages:
        # 2. Extract table
        tables = camelot.read_pdf(pdf_path, pages=str(pages.index(p)+1), flavor='lattice')

        for t in tables:
            # 3. Crop cells
            items = crop_cells_from_table(t, p)
            all_items.extend(items)

    # 4. Generate labels
    os.makedirs("dataset/train", exist_ok=True)
    write_paddle_labels(all_items, "dataset/train/labels.txt")

    # 5. Auto-generate config
    generate_paddle_config()

    print("Dataset ready. Run training:")
    print("python -m paddleocr.tools.train -c rec_config.yml")

# =============================
pipeline("financial_report.pdf")