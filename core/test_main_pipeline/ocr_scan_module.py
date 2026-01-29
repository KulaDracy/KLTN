import cv2
import numpy as np
import threading
import queue
import time
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from PIL import Image

# Import Models
from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

@dataclass
class OCRScanResult:
    """Kết quả trả về tương thích với Main Pipeline"""
    image_path: str
    lines: list
    average_confidence: float
    has_low_confidence: bool

class OCRScanModule:
    def __init__(self, use_gpu: bool = False, num_workers: int = 2, confidence_threshold: float = 0.6):
        self.use_gpu = use_gpu
        self.num_workers = num_workers
        self.confidence_threshold = confidence_threshold
        
        # Thread-local storage: Mỗi worker sẽ giữ một bản copy của model để tránh xung đột
        self._local = threading.local()
        
        # Queue hệ thống
        self.task_queue = queue.PriorityQueue()
        self.results_map = {}
        self.results_lock = threading.Lock()
        self.stop_event = threading.Event()

    def _init_models(self):
        """Khởi tạo models cho từng luồng riêng biệt"""
        if not hasattr(self._local, 'detector'):
            # Paddle chỉ dùng để Detect (vùng chứa chữ)
            self._local.detector = PaddleOCR(
                use_angle_cls=True, lang='vi', det=True, rec=False, 
                use_gpu=self.use_gpu, show_log=False
            )
            # VietOCR dùng để nhận diện chữ tiếng Việt chính xác
            config = Cfg.load_config_from_name('vgg_transformer')
            config['device'] = 'cuda' if self.use_gpu else 'cpu'
            self._local.recognizer = Predictor(config)
        return self._local.detector, self._local.recognizer

    def _quick_analyze_priority(self, image_path: str) -> int:
        """Phân tích nhanh để đưa vào hàng đợi VIP (Ưu tiên trang có nhiều số)"""
        try:
            img = cv2.imread(image_path)
            if img is None: return 10
            # Kiểm tra tên file hoặc nội dung sơ bộ
            name = Path(image_path).stem.lower()
            if any(k in name for k in ['page_001', 'balance', 'kqkd', 'tc']):
                return 1  # Rất ưu tiên
            return 5      # Bình thường
        except:
            return 10

    def _process_logic(self, image_path: str, detector, recognizer) -> OCRScanResult:
        """Core logic Hybrid OCR"""
        img = cv2.imread(image_path)
        # 1. Detection
        raw_result = detector.ocr(img, cls=True)
        if not raw_result or raw_result[0] is None:
            return OCRScanResult(image_path, [], 0.0, True)

        lines_data = []
        result = raw_result[0]

        for i, box in enumerate(result):
            try:
                # Xử lý tọa độ an toàn (Fix lỗi inhomogeneous shape)
                points_raw = box[0] if isinstance(box, list) and len(box) == 2 else box
                points = np.array(points_raw, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                
                x_min, y_min = np.min(points, axis=0)
                x_max, y_max = np.max(points, axis=0)
                
                # Cắt ảnh (Crop)
                h_img, w_img = img.shape[:2]
                crop = img[max(0, y_min-2):min(h_img, y_max+2), max(0, x_min-2):min(w_img, x_max+2)]
                
                if crop.size > 0:
                    # 2. Recognition với VietOCR
                    pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    text = recognizer.predict(pil_img)
                    
                    lines_data.append({
                        "text": text,
                        "confidence": 0.95,
                        "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
                        "y_position": float(y_min),
                        "line_number": i
                    })
            except:
                continue

        lines_data.sort(key=lambda x: x['y_position'])
        return OCRScanResult(image_path, lines_data, 0.95, False)

    def _worker(self):
        """Worker thread loop"""
        det, rec = self._init_models()
        while not self.stop_event.is_set():
            try:
                priority, img_path = self.task_queue.get(timeout=1)
                result = self._process_logic(img_path, det, rec)
                
                with self.results_lock:
                    self.results_map[img_path] = result
                
                self.task_queue.task_done()
                print(f"  ✓ Processed: {Path(img_path).name} (Priority: {priority})")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"  ❌ Worker Error: {e}")
                self.task_queue.task_done()

    def scan_batch(self, image_folder: str, verbose: bool = True) -> List[OCRScanResult]:
        """Hàm chính để quét cả folder bằng đa luồng"""
        img_paths = [str(f) for f in Path(image_folder).glob("*") if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        if not img_paths: return []

        print(f"\n🚀 Starting Multi-thread Hybrid OCR ({self.num_workers} workers)...")
        
        # 1. Đưa vào Queue với Priority
        for path in img_paths:
            priority = self._quick_analyze_priority(path)
            self.task_queue.put((priority, path))

        # 2. Chạy Workers
        threads = []
        for i in range(self.num_workers):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            threads.append(t)

        # 3. Đợi hoàn thành
        self.task_queue.join()
        self.stop_event.set()
        for t in threads: t.join()

        # Trả về kết quả theo đúng thứ tự file ban đầu
        return [self.results_map[p] for p in img_paths if p in self.results_map]

    def save_scan_result(self, result: OCRScanResult, output_path: str):
        """Lưu JSON tương thích module Extract"""
        data = {
            "image_path": result.image_path,
            "average_confidence": result.average_confidence,
            "lines": result.lines
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)