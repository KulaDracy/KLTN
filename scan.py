#!/usr/bin/env python3
"""
Multiprocessing PDF Scanner
Xử lý song song nhiều trang PDF để tăng tốc độ
Giữ nguyên độ chính xác của thuật toán detection
"""

import sys
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from multiprocessing import Pool, cpu_count
import time
from functools import partial

# ============================================================
# GLOBAL OCR INITIALIZATION (Lazy Loading per Process)
# ============================================================

_ocr_instance = None

def get_ocr():
    """
    Lazy initialization of OCR per process
    Mỗi worker process sẽ có OCR instance riêng
    """
    global _ocr_instance
    if _ocr_instance is None:
        try:
            _ocr_instance = PaddleOCR(
                det=True, 
                rec=True, 
                use_textline_orientation=True, 
                lang="vi", 
                show_log=False
            )
        except:
            _ocr_instance = PaddleOCR(
                det=True, 
                rec=True, 
                use_angle_cls=False, 
                lang="vi", 
                show_log=False
            )
    return _ocr_instance


# ============================================================
# 1) TABLE DETECTOR
# ============================================================

def detect_table(img):
    """Detect bảng bằng pattern nhiều hàng + nhiều cột"""
    ocr = get_ocr()
    
    try:
        res = ocr.ocr(img, det=True, rec=False, cls=False)
        boxes = res[0] if res else []
    except:
        return False
    
    if not boxes or len(boxes) < 15:
        return False
    
    lefts, centers = [], []
    for b in boxes:
        pts = np.array(b, float)
        xs = pts[:, 0]
        ys = pts[:, 1]
        lefts.append(xs.min())
        centers.append((ys.min() + ys.max()) / 2)
    
    # Detect columns
    lefts_sorted = np.sort(lefts)
    col = 1
    for i in range(1, len(lefts_sorted)):
        if abs(lefts_sorted[i] - lefts_sorted[i - 1]) > 25:
            col += 1
    
    # Detect rows
    centers_sorted = np.sort(centers)
    row = 1
    for i in range(1, len(centers_sorted)):
        if abs(centers_sorted[i] - centers_sorted[i - 1]) > 18:
            row += 1
    
    is_table = col >= 3 and row >= 5
    return is_table


# ============================================================
# 2) IMPROVED OCR ORIENTATION SCORE
# ============================================================

def score_orientation_improved(img, verbose=False):
    """
    Score orientation với multiple metrics
    verbose: False để tắt output trong multiprocessing
    """
    ocr = get_ocr()
    
    angles = [0, 90, 180, 270]
    scores = {}
    details = {}
    
    # Crop để bỏ viền
    h_orig, w_orig = img.shape[:2]
    h_start = int(h_orig * 0.01)
    h_end = int(h_orig * 0.99)
    w_start = int(w_orig * 0.01)
    w_end = int(w_orig * 0.99)
    img_cropped = img[h_start:h_end, w_start:w_end]
    
    # Resize
    h, w = img_cropped.shape[:2]
    scale = 800 / max(h, w)
    small = cv2.resize(img_cropped, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_LINEAR)
    
    for angle in angles:
        # Rotate
        if angle == 0:
            rot = small
        elif angle == 90:
            rot = cv2.rotate(small, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rot = cv2.rotate(small, cv2.ROTATE_180)
        else:
            rot = cv2.rotate(small, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        try:
            res = ocr.ocr(rot, det=True, rec=True, cls=False)
            
            if not res or not res[0]:
                scores[angle] = 0
                details[angle] = {
                    'text_len': 0,
                    'confidence': 0,
                    'horizontal_ratio': 0,
                    'box_count': 0,
                    'score': 0
                }
                continue
            
            boxes = res[0]
            
            # Metric 1: Text length
            text_len = sum(len(x[1][0]) for x in boxes)
            
            # Metric 2: Average confidence
            confidences = [x[1][1] for x in boxes]
            avg_confidence = np.mean(confidences) if confidences else 0
            
            # Metric 3: Horizontal ratio
            horizontal_count = 0
            for box in boxes:
                bbox = box[0]
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                box_w = max(x_coords) - min(x_coords)
                box_h = max(y_coords) - min(y_coords)
                
                if box_w > box_h * 1.5:
                    horizontal_count += 1
            
            horizontal_ratio = horizontal_count / len(boxes) if boxes else 0
            
            # Metric 4: Box count
            box_count = len(boxes)
            
            # Combined score
            combined_score = (
                text_len * 0.40 +
                avg_confidence * 1000 * 0.20 +
                horizontal_ratio * 1000 * 0.35 +
                box_count * 5 * 0.05
            )
            
            scores[angle] = combined_score
            details[angle] = {
                'text_len': text_len,
                'confidence': round(avg_confidence, 3),
                'horizontal_ratio': round(horizontal_ratio, 3),
                'box_count': box_count,
                'score': round(combined_score, 1)
            }
            
        except Exception as e:
            scores[angle] = 0
            details[angle] = {
                'text_len': 0,
                'confidence': 0,
                'horizontal_ratio': 0,
                'box_count': 0,
                'score': 0,
                'error': str(e)
            }
    
    # Print detailed scores if verbose
    if verbose:
        print("  OCR Detailed Scores:")
        for angle in angles:
            d = details[angle]
            print(f"    {angle:3d}: text={d['text_len']:4d}, "
                  f"conf={d['confidence']:.2f}, horiz={d['horizontal_ratio']:.2f}, "
                  f"boxes={d['box_count']:3d}, score={d.get('score', 0):.1f}")
    
    # Find best and second best
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_angle, best_score = sorted_scores[0]
    second_angle, second_score = sorted_scores[1]
    
    total_text = sum(details[a]['text_len'] for a in angles)
    
    if verbose:
        print(f"  Best: {best_angle} (score: {best_score:.1f})")
        print(f"  2nd: {second_angle} (score: {second_score:.1f})")
        print(f"  Margin: {best_score - second_score:.1f}")
    
    return best_angle, best_score, second_score, total_text, details


# ============================================================
# 3) IMPROVED DECISION ENGINE
# ============================================================

def decide_rotation_improved(img, is_table, verbose=False):
    """
    Decision logic với verbose control
    """
    
    # 1) Nếu là bảng
    if is_table:
        if verbose:
            print("  TABLE DETECTED  Checking orientation carefully...")
        
        best_angle, best_score, second_score, total_text, details = \
            score_orientation_improved(img, verbose=verbose)
        
        margin = best_score - second_score
        
        if best_angle == 270 and margin > 50:
            if verbose:
                print(f"  TABLE: 270 is CONFIDENT (margin {margin:.1f})  ROTATE 270°")
            return 270
        
        elif details[270]['score'] > details[0]['score'] and margin < 100:
            if verbose:
                print(f"  TABLE: 270 Score > 0 Score (conflict)  ROTATE 270°")
            return 270
        
        else:
            if verbose:
                print(f"  TABLE: Low confidence or 0 best  NO ROTATION")
            return 0
    
    # 2) Không phải bảng
    best_angle, best_score, second_score, total_text, details = \
        score_orientation_improved(img, verbose=verbose)
    
    margin = best_score - second_score
    
    # 3) OCR tự tin
    if margin > 50:
        if verbose:
            print(f"  OCR CONFIDENT (margin: {margin:.1f})  ROTATE {best_angle}°")
        return best_angle
    
    # 4) Ít text → fallback geometry
    if total_text < 200:
        if verbose:
            print(f"  LOW TEXT ({total_text} chars)  fallback to geometry")
        
        h, w = img.shape[:2]
        
        if w > h * 1.3:
            if verbose:
                print("  Image is wide (landscape)  ROTATE 90")
            return 90
        else:
            if verbose:
                print("  Image is portrait or square  NO ROTATION")
            return 0
    
    # 5) OCR không tự tin
    if verbose:
        print(f"  OCR LOW CONFIDENCE (margin: {margin:.1f})  NO ROTATION")
    return 0


# ============================================================
# 4) PROCESS SINGLE PAGE (Worker Function)
# ============================================================

def process_page(page_info, out_dir, verbose=False):
    """
    Process một trang PDF - Chạy trong worker process
    
    Args:
        page_info: Dict chứa thông tin trang
        out_dir: Output directory
        verbose: Print debug info
    
    Returns:
        Dict with result
    """
    num = page_info["page_number"]
    img_path = Path(page_info["file"])
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"PAGE {num}")
        print('='*70)
    
    # Load image
    img = cv2.imread(str(img_path))
    
    if img is None:
        return {
            "page_number": num,
            "error": "Failed to load image",
            "file": None,
            "rotation": 0,
            "is_table": False
        }
    
    # Detect table
    is_table = detect_table(img)
    if verbose:
        print(f"  Detecting table... {'YES' if is_table else 'NO'}")
    
    # Decide rotation
    angle = decide_rotation_improved(img, is_table, verbose=verbose)
    
    # Apply rotation
    if angle == 90:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    if verbose:
        print(f"  FINAL ROTATION: {angle}°")
    
    # Save
    out_file = Path(out_dir) / f"page_{num:03d}.png"
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(out_file)
    
    if verbose:
        print(f"  Saved: {out_file.name}")
    
    return {
        "page_number": num,
        "file": str(out_file.resolve()),
        "rotation": angle,
        "is_table": is_table
    }


# ============================================================
# 5) MAIN WITH MULTIPROCESSING
# ============================================================

def scan_multiprocessing(manifest_file, out_dir, num_workers=None, verbose=False):
    """
    Process PDF với multiprocessing
    
    Args:
        manifest_file: Path to manifest JSON
        out_dir: Output directory
        num_workers: Số worker processes (None = auto detect)
        verbose: Print detailed progress
    """
    
    # Load manifest
    manifest = json.load(open(manifest_file, "r", encoding="utf8"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Leave 1 CPU free
    
    total_pages = len(manifest["pages"])
    
    print("\n" + "="*70)
    print("PDF SCANNER WITH MULTIPROCESSING")
    print("="*70)
    print(f"Total pages: {total_pages}")
    print(f"Workers: {num_workers}")
    print(f"Output: {out}")
    print("="*70)
    
    # Start timer
    start_time = time.time()
    
    # Create worker function with fixed out_dir
    worker_func = partial(process_page, out_dir=out_dir, verbose=verbose)
    
    # Process with multiprocessing
    if num_workers > 1:
        print("\n Processing pages in parallel...")
        with Pool(processes=num_workers) as pool:
            # Map với progress
            results = []
            for i, result in enumerate(pool.imap(worker_func, manifest["pages"]), 1):
                results.append(result)
                if not verbose:  # Chỉ show progress bar nếu không verbose
                    progress = i / total_pages * 100
                    bar_length = 50
                    filled = int(bar_length * i / total_pages)
                    bar = ' ' * filled + ' ' * (bar_length - filled)
                    print(f"\r  [{bar}] {progress:.1f}% ({i}/{total_pages})", end='')
            
            if not verbose:
                print()  # New line after progress bar
    else:
        # Single process mode (for debugging)
        print("\n Processing pages sequentially...")
        results = []
        for page_info in manifest["pages"]:
            result = worker_func(page_info)
            results.append(result)
    
    # Sort results by page number
    results.sort(key=lambda x: x["page_number"])
    
    # Create new manifest
    new_manifest = {
        "file": manifest["file"],
        "pages": results
    }
    
    # Save manifest
    manifest_out = out / "manifest_scanned.json"
    with open(manifest_out, "w", encoding="utf8") as f:
        json.dump(new_manifest, f, indent=2, ensure_ascii=False)
    
    # End timer
    elapsed = time.time() - start_time
    
    # Summary
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
    print(f"Processed: {len(results)} pages")
    print(f"Time: {elapsed:.2f}s ({elapsed/total_pages:.2f}s per page)")
    print(f"Speed: {total_pages/elapsed:.2f} pages/sec")
    print(f"Manifest: {manifest_out}")
    
    # Statistics
    rotated = sum(1 for r in results if r.get('rotation', 0) != 0)
    tables = sum(1 for r in results if r.get('is_table', False))
    errors = sum(1 for r in results if 'error' in r)
    
    print(f"\n Statistics:")
    print(f"   Rotated pages: {rotated}/{total_pages} ({rotated/total_pages*100:.1f}%)")
    print(f"   Tables detected: {tables}/{total_pages} ({tables/total_pages*100:.1f}%)")
    if errors > 0:
        print(f"   Errors: {errors}")
    print("="*70)


# ============================================================
# 6) SINGLE PROCESS MODE (for comparison/debugging)
# ============================================================

def scan_sequential(manifest_file, out_dir):
    """
    Process PDF tuần tự (single process) - for debugging
    """
    manifest = json.load(open(manifest_file, "r", encoding="utf8"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("PDF SCANNER (SEQUENTIAL MODE)")
    print("="*70)
    
    start_time = time.time()
    
    results = []
    for page_info in manifest["pages"]:
        result = process_page(page_info, out_dir, verbose=True)
        results.append(result)
    
    # Save manifest
    new_manifest = {"file": manifest["file"], "pages": results}
    manifest_out = out / "manifest_scanned.json"
    with open(manifest_out, "w", encoding="utf8") as f:
        json.dump(new_manifest, f, indent=2, ensure_ascii=False)
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print(f"DONE in {elapsed:.2f}s")
    print("="*70)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scan_mp.py manifest.json output_dir [options]")
        print("\nOptions:")
        print("  --workers N    Number of worker processes (default: auto)")
        print("  --sequential   Run in sequential mode (for debugging)")
        print("  --verbose      Print detailed progress")
        print("\nExamples:")
        print("  python scan_mp.py manifest.json output/")
        print("  python scan_mp.py manifest.json output/ --workers 4")
        print("  python scan_mp.py manifest.json output/ --sequential --verbose")
        sys.exit(1)
    
    manifest_file = sys.argv[1]
    out_dir = sys.argv[2]
    
    # Parse options
    sequential = '--sequential' in sys.argv
    verbose = '--verbose' in sys.argv
    
    num_workers = None
    if '--workers' in sys.argv:
        idx = sys.argv.index('--workers')
        if idx + 1 < len(sys.argv):
            num_workers = int(sys.argv[idx + 1])
    
    # Run
    if sequential:
        scan_sequential(manifest_file, out_dir)
    else:
        scan_multiprocessing(manifest_file, out_dir, num_workers=num_workers, verbose=verbose)