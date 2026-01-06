"""
=============================================================================
OCR Scanner với Vietnamese Support và Confidence Scoring
- Tối ưu OCR tiếng Việt
- Đánh giá confidence cho mỗi trang
- Đánh dấu trang có confidence thấp
- Dual Queue với VIP priority
=============================================================================
"""

import json
import threading
import queue
import time
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import cv2
import numpy as np
from postprocess_financial import FinancialPostProcessor


# Import PaddleOCR
try:
    from paddleocr import PaddleOCR
    print("✓ Using pip installed PaddleOCR")
except ImportError:
    print("❌ PaddleOCR not found. Please install: pip install paddleocr")
    sys.exit(1)


class QueueType(Enum):
    """Loại queue"""
    VIP = "VIP"
    NORMAL = "NORMAL"


class ConfidenceLevel(Enum):
    """Mức độ confidence"""
    HIGH = "HIGH"        # >= 0.85
    MEDIUM = "MEDIUM"    # 0.70 - 0.84
    LOW = "LOW"          # 0.50 - 0.69
    VERY_LOW = "VERY_LOW"  # < 0.50

class BlockType(Enum):
    """Loại block content"""
    TEXT = "TEXT"
    TABLE = "TABLE"
    HEADER = "HEADER"
    FOOTER = "FOOTER"


@dataclass
class BoundingBox:
    """Bounding box của một block"""
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    
    @property
    def width(self):
        return self.x_max - self.x_min
    
    @property
    def height(self):
        return self.y_max - self.y_min
    
    @property
    def center_x(self):
        return (self.x_min + self.x_max) // 2
    
    @property
    def center_y(self):
        return (self.y_min + self.y_max) // 2
    
    @property
    def area(self):
        return self.width * self.height
@dataclass
class TextBlock:
    """Text block với vị trí"""
    text: str
    confidence: float
    bbox: BoundingBox
    block_type: BlockType = BlockType.TEXT


@dataclass
class TableCell:
    """Cell trong bảng"""
    row: int
    col: int
    text: str
    confidence: float
    bbox: BoundingBox


@dataclass
class Table:
    """Bảng được phát hiện"""
    bbox: BoundingBox
    rows: int
    cols: int
    cells: List[TableCell]
    confidence: float
    
    def to_2d_array(self) -> List[List[str]]:
        """Convert cells thành mảng 2D"""
        array = [["" for _ in range(self.cols)] for _ in range(self.rows)]
        for cell in self.cells:
            if 0 <= cell.row < self.rows and 0 <= cell.col < self.cols:
                array[cell.row][cell.col] = cell.text
        return array
    
    def to_markdown(self) -> str:
        """Convert bảng sang Markdown format"""
        array = self.to_2d_array()
        if not array:
            return ""
        
        lines = []
        # Header row
        lines.append("| " + " | ".join(array[0]) + " |")
        # Separator
        lines.append("|" + "|".join(["---" for _ in range(self.cols)]) + "|")
        # Data rows
        for row in array[1:]:
            lines.append("| " + " | ".join(row) + " |")
        
        return "\n".join(lines)
    
    def to_html(self) -> str:
        """Convert bảng sang HTML format"""
        array = self.to_2d_array()
        if not array:
            return ""
        
        html = ['<table border="1">']
        # Header
        html.append('  <thead>')
        html.append('    <tr>')
        for cell in array[0]:
            html.append(f'      <th>{cell}</th>')
        html.append('    </tr>')
        html.append('  </thead>')
        # Body
        html.append('  <tbody>')
        for row in array[1:]:
            html.append('    <tr>')
            for cell in row:
                html.append(f'      <td>{cell}</td>')
            html.append('    </tr>')
        html.append('  </tbody>')
        html.append('</table>')
        
        return '\n'.join(html)
@dataclass
class PageLayout:
    """Layout của trang"""
    width: int
    height: int
    text_blocks: List[TextBlock] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    
    def get_reading_order_blocks(self) -> List[TextBlock]:
        """Sắp xếp blocks theo thứ tự đọc (top-to-bottom, left-to-right)"""
        return sorted(self.text_blocks, 
                     key=lambda b: (b.bbox.y_min // 50, b.bbox.x_min))
@dataclass
class PageTask:
    """Task để xử lý một trang"""
    page_number: int
    image_path: str
    priority: int = 0
    
    def __lt__(self, other):
        return self.priority > other.priority


@dataclass
class PageResult:
    """Kết quả OCR cho một trang"""
    page_number: int
    priority: int
    queue_type: str
    content: str
    layout: Optional[PageLayout]
    char_count: int
    number_count: int
    total_blocks: int
    table_count: int
    avg_confidence: float
    confidence_level: str
    needs_review: bool
    low_confidence_blocks: List[Dict]
    processing_time: float
    financial_data: Optional[dict] = None


class TableDetector:
    """Phát hiện và trích xuất bảng từ ảnh"""
    
    @staticmethod
    def detect_lines(image: np.ndarray) -> Tuple[List, List]:
        """Phát hiện đường kẻ ngang và dọc trong ảnh"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        
        # Find contours
        h_contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
        v_contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
        
        return h_contours, v_contours
    
    @staticmethod
    def find_table_regions(image: np.ndarray, h_contours, v_contours) -> List[BoundingBox]:
        """Tìm vùng bảng từ các đường kẻ"""
        if not h_contours or not v_contours:
            return []
        
        # Combine lines
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        table_mask = np.zeros((h, w), dtype=np.uint8)
        
        cv2.drawContours(table_mask, h_contours, -1, 255, 2)
        cv2.drawContours(table_mask, v_contours, -1, 255, 2)
        
        # Find intersections and table regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        table_mask = cv2.morphologyEx(table_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        tables = []
        min_table_area = 5000  # Minimum area for a table
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            if area > min_table_area and w > 100 and h > 100:
                tables.append(BoundingBox(x, y, x + w, y + h))
        
        return tables
    
    @staticmethod
    def cluster_cells_by_position(text_blocks: List[TextBlock], 
                                  table_bbox: BoundingBox) -> Tuple[int, int, List[TableCell]]:
        """Gom các text blocks thành cells của bảng"""
        # Filter blocks inside table
        cells_in_table = [b for b in text_blocks 
                         if (b.bbox.x_min >= table_bbox.x_min and 
                             b.bbox.x_max <= table_bbox.x_max and
                             b.bbox.y_min >= table_bbox.y_min and
                             b.bbox.y_max <= table_bbox.y_max)]
        
        if not cells_in_table:
            return 0, 0, []
        
        # Sort by position
        cells_in_table.sort(key=lambda b: (b.bbox.y_min, b.bbox.x_min))
        
        # Cluster into rows
        rows = []
        current_row = [cells_in_table[0]]
        row_y_threshold = 20
        
        for block in cells_in_table[1:]:
            if abs(block.bbox.y_min - current_row[0].bbox.y_min) < row_y_threshold:
                current_row.append(block)
            else:
                rows.append(sorted(current_row, key=lambda b: b.bbox.x_min))
                current_row = [block]
        
        if current_row:
            rows.append(sorted(current_row, key=lambda b: b.bbox.x_min))
        
        # Determine number of columns (max cells in any row)
        num_cols = max(len(row) for row in rows) if rows else 0
        num_rows = len(rows)
        
        # Create TableCells
        table_cells = []
        for row_idx, row in enumerate(rows):
            for col_idx, block in enumerate(row):
                cell = TableCell(
                    row=row_idx,
                    col=col_idx,
                    text=block.text,
                    confidence=block.confidence,
                    bbox=block.bbox
                )
                table_cells.append(cell)
        
        return num_rows, num_cols, table_cells

class VietnameseOCRService:
    """OCR Service tối ưu cho tiếng Việt"""
    _local = threading.local()
    
    def __init__(self):
        print("✓ Vietnamese OCR Service initialized")
    
    def get_ocr(self):
        """Get thread-local OCR instance với config tối ưu cho tiếng Việt"""
        if not hasattr(self._local, 'ocr'):
            self._local.ocr = PaddleOCR(
                lang="vi",  # Vietnamese language
                use_textline_orientation=True,
                show_log=False,
                # Tối ưu cho tiếng Việt
                det_db_thresh=0.3,  # Detection threshold
                det_db_box_thresh=0.6,  # Box threshold
                rec_batch_num=6,  # Batch processing
                drop_score=0.3,  # Ngưỡng thấp để detect nhiều text hơn
                use_dilation=True,  # Mở rộng vùng text
            )
            print(f"  [Thread {threading.current_thread().name}] Vietnamese OCR instance created")
        return self._local.ocr
    
    def ocr(self, image, det=True, rec=True, cls=False):
        """Perform OCR"""
        ocr_instance = self.get_ocr()
        return ocr_instance.ocr(image, det=det, rec=rec, cls=cls)


class DualQueueOCRScanner:
    """OCR Scanner với Vietnamese support và Confidence scoring"""
    
    # Confidence thresholds
    CONFIDENCE_HIGH = 0.85
    CONFIDENCE_MEDIUM = 0.70
    CONFIDENCE_LOW = 0.50
    
    def __init__(
        self,
        image_folder: str,
        vip_workers: int = 2,
        normal_workers: int = 2,
        output_file: str = "ocr_results.json",
        confidence_threshold: float = 0.70,  # Ngưỡng để đánh dấu needs_review
        enable_table_detection: bool = True
    ):
        self.post_processor = FinancialPostProcessor()
        """Initialize Dual Queue OCR Scanner"""
        self.image_folder = Path(image_folder)
        self.output_file = Path(output_file)
        self.confidence_threshold = confidence_threshold
        self.enable_table_detection = enable_table_detection
        
        if not self.image_folder.exists():
            raise FileNotFoundError(f"Image folder not found: {self.image_folder}")
        
        # Initialize Vietnamese OCR service
        self.ocr_service = VietnameseOCRService()
        self.table_detector = TableDetector()
        
        # Queues
        self.vip_queue = queue.PriorityQueue()
        self.normal_queue = queue.Queue()
        self.results = []
        self.results_lock = threading.Lock()
        
        # Workers
        self.vip_workers = vip_workers
        self.normal_workers = normal_workers
        
        # Statistics
        self.stats = {
            "total_pages": 0,
            "vip_pages": 0,
            "normal_pages": 0,
            "processed": 0,
            "total_tables": 0,
            "high_confidence": 0,
            "medium_confidence": 0,
            "low_confidence": 0,
            "very_low_confidence": 0,
            "needs_review": 0,
            "start_time": None,
            "end_time": None
        }
        
        # Control
        self.stop_event = threading.Event()
        self.workers = []
        
        print(f" DualQueueOCRScanner initialized")
        print(f"  Image folder: {self.image_folder}")
        print(f"  VIP Workers: {vip_workers}")
        print(f"  Normal Workers: {normal_workers}")
        print(f"  Confidence threshold: {confidence_threshold}")
        print(f"  Output: {self.output_file}")
    
    def count_chars_and_numbers(self, text: str) -> Tuple[int, int]:
        """Đếm số lượng ký tự chữ và số trong text"""
        # Đếm chữ cái tiếng Việt (bao gồm dấu)
        char_count = len(re.findall(r'[a-zA-ZÀ-ỹ]', text))
        # Đếm chữ số
        number_count = len(re.findall(r'\d', text))
        return char_count, number_count
    
    def classify_confidence(self, confidence: float) -> ConfidenceLevel:
        """Phân loại mức độ confidence"""
        if confidence >= self.CONFIDENCE_HIGH:
            return ConfidenceLevel.HIGH
        elif confidence >= self.CONFIDENCE_MEDIUM:
            return ConfidenceLevel.MEDIUM
        elif confidence >= self.CONFIDENCE_LOW:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def preprocess_image(self, image):
        """Preprocessing ảnh"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, 
                                           templateWindowSize=7, searchWindowSize=21)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        binary = cv2.adaptiveThreshold(enhanced, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        return processed
    
    def clean_text(self, text: str) -> str:
        """Clean text"""
        text = text.replace('đ', 'đ').replace('Đ', 'Đ')
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_layout(self, image: np.ndarray, ocr_result) -> PageLayout:
        """Trích xuất layout từ kết quả OCR"""
        h, w = image.shape[:2]
        layout = PageLayout(width=w, height=h)
        
        if not ocr_result or not ocr_result[0]:
            return layout
        
        # Extract text blocks
        text_blocks = []
        for line in ocr_result[0]:
            bbox_points, (text, confidence) = line
            
            # Calculate bounding box
            xs = [p[0] for p in bbox_points]
            ys = [p[1] for p in bbox_points]
            bbox = BoundingBox(
                x_min=int(min(xs)),
                y_min=int(min(ys)),
                x_max=int(max(xs)),
                y_max=int(max(ys))
            )
            
            cleaned_text = self.clean_text(text)
            text_block = TextBlock(
                text=cleaned_text,
                confidence=confidence,
                bbox=bbox
            )
            text_blocks.append(text_block)
        
        layout.text_blocks = text_blocks
        
        # Detect tables if enabled
        if self.enable_table_detection and len(text_blocks) > 5:
            h_contours, v_contours = self.table_detector.detect_lines(image)
            
            if h_contours and v_contours:
                table_regions = self.table_detector.find_table_regions(
                    image, h_contours, v_contours
                )
                
                for table_bbox in table_regions:
                    num_rows, num_cols, cells = self.table_detector.cluster_cells_by_position(
                        text_blocks, table_bbox
                    )
                    
                    if num_rows > 1 and num_cols > 1:
                        avg_conf = sum(c.confidence for c in cells) / len(cells) if cells else 0
                        table = Table(
                            bbox=table_bbox,
                            rows=num_rows,
                            cols=num_cols,
                            cells=cells,
                            confidence=avg_conf
                        )
                        layout.tables.append(table)
        
        return layout
    
    def layout_to_text(self, layout: PageLayout) -> str:
        """Convert layout thành text có cấu trúc"""
        lines = []
        
        # Get blocks in reading order
        ordered_blocks = layout.get_reading_order_blocks()
        
        # Track which blocks are in tables
        blocks_in_tables = set()
        for table in layout.tables:
            for cell in table.cells:
                # Find corresponding block
                for block in ordered_blocks:
                    if (block.bbox.x_min == cell.bbox.x_min and 
                        block.bbox.y_min == cell.bbox.y_min):
                        blocks_in_tables.add(id(block))
        
        # Add tables first
        for i, table in enumerate(layout.tables, 1):
            lines.append(f"\n[TABLE {i}]")
            lines.append(table.to_markdown())
            lines.append(f"[/TABLE {i}]\n")
        
        # Add text blocks not in tables
        lines.append("\n[TEXT CONTENT]")
        for block in ordered_blocks:
            if id(block) not in blocks_in_tables:
                lines.append(block.text)
        lines.append("[/TEXT CONTENT]")
        
        return "\n".join(lines)
    
    def load_images(self) -> List[str]:
        """Load danh sách ảnh từ thư mục"""
        print(f"\n📁 Loading images from: {self.image_folder}")
        
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_files = []
        
        for ext in extensions:
            image_files.extend(self.image_folder.glob(ext))
        
        image_files = sorted(image_files, key=lambda x: x.name)
        
        print(f"✓ Found {len(image_files)} images")
        
        return [str(f) for f in image_files]
    
    def quick_analyze_image(self, image_path: str) -> Tuple[bool, int]:
        """Quick analyze for VIP detection"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return False, 0
            
            h, w = image.shape[:2]
            small_image = cv2.resize(image, (int(w * 0.25), int(h * 0.25)))
            quick_result = self.ocr_service.ocr(small_image, det=True, rec=True, cls=False)
            
            if not quick_result or not quick_result[0]:
                return False, 0
            
            all_text = " ".join([line[1][0] for line in quick_result[0]])
            char_count, number_count = self.count_chars_and_numbers(all_text)
            
            financial_keywords = [
                'tài sản', 'nợ', 'vốn', 'doanh thu', 'lợi nhuận',
                'kế toán', 'báo cáo', 'bảng cân đối', 'thuyết minh',
                'đồng', 'vnd', 'vnđ', 'nghìn', 'triệu', 'tỷ',
                'số dư', 'phải thu', 'phải trả', 'chi phí'
            ]
            
            text_lower = all_text.lower()
            has_financial = any(kw in text_lower for kw in financial_keywords)
            number_blocks = re.findall(r'\d[\d,.\s]{3,}', all_text)
            
            is_financial = (
                (number_count > char_count * 0.7) or
                (has_financial and number_count > 15) or
                (len(number_blocks) > 3 and number_count > 25)
            )
            
            return is_financial, number_count
        except Exception as e:
            print(f"  ⚠️  Quick analyze error: {e}")
            return False, 0
    
    def create_initial_tasks(self, image_paths: List[str]):
        """Tạo initial tasks với VIP detection"""
        print(f"\n📋 Creating initial tasks with VIP detection...")
        
        vip_detected = 0
        
        for page_num, image_path in enumerate(image_paths, start=1):
            # Quick analyze
            is_financial, number_count = self.quick_analyze_image(image_path)
            
            if is_financial:
                priority = number_count
                task = PageTask(
                    page_number=page_num,
                    image_path=image_path,
                    priority=priority
                )
                self.vip_queue.put((priority, task))
                vip_detected += 1
                print(f"  ⭐ Page {page_num}: Detected FINANCIAL → VIP queue (priority: {priority})")
            else:
                task = PageTask(
                    page_number=page_num,
                    image_path=image_path,
                    priority=0
                )
                self.normal_queue.put(task)
        
        self.stats['total_pages'] = len(image_paths)
        self.stats['vip_pages'] = vip_detected
        
        print(f" Created {len(image_paths)} tasks")
        print(f"  • VIP queue: {vip_detected} pages")
        print(f"  • Normal queue: {len(image_paths) - vip_detected} pages")
    
    def ocr_worker(self, worker_id: int, queue_type: QueueType):
        """Worker để xử lý OCR từ queue"""
        worker_name = f"{queue_type.value}-Worker-{worker_id}"
        print(f" {worker_name} started")
        
        work_queue = self.vip_queue if queue_type == QueueType.VIP else self.normal_queue
        
        while not self.stop_event.is_set():
            try:
                if queue_type == QueueType.VIP:
                    priority, task = work_queue.get(timeout=1)
                else:
                    task = work_queue.get(timeout=1)
                
                result = self.process_page_ocr(task, worker_name)
                
                with self.results_lock:
                    self.results.append(result)
                    self.stats['processed'] += 1
                    
                    # Update confidence stats
                    if result.confidence_level == "HIGH":
                        self.stats['high_confidence'] += 1
                    elif result.confidence_level == "MEDIUM":
                        self.stats['medium_confidence'] += 1
                    elif result.confidence_level == "LOW":
                        self.stats['low_confidence'] += 1
                    else:
                        self.stats['very_low_confidence'] += 1
                    
                    if result.needs_review:
                        self.stats['needs_review'] += 1
                
                work_queue.task_done()
                
                total = self.stats['total_pages']
                done = self.stats['processed']
                progress = (done / total * 100) if total > 0 else 0
                
                conf_icon = "✓" if result.avg_confidence >= self.confidence_threshold else "⚠️"
                print(f"  {conf_icon} [{worker_name}] Progress: {done}/{total} ({progress:.1f}%) "
                      f"- Page {task.page_number}: conf={result.avg_confidence:.2f} ({result.confidence_level})")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f" {worker_name} error: {e}")
                work_queue.task_done()
        
        print(f" {worker_name} stopped")
    
    def process_page_ocr(self, task: PageTask, worker_name: str) -> PageResult:
        """Process OCR for page"""
        start_time = time.time()
        
        try:
            queue_type = "VIP" if task.priority > 0 else "NORMAL"
            image = cv2.imread(task.image_path)
            
            if image is None:
                raise ValueError(f"Cannot load: {task.image_path}")
            
            processed = self.preprocess_image(image)
            ocr_result = self.ocr_service.ocr(processed, det=True, rec=True, cls=False)
            
            # Extract layout
            layout = self.extract_layout(image, ocr_result)
            
            # Convert to text
            content = self.layout_to_text(layout)
            # ===== POST-PROCESS FINANCIAL =====
            raw_blocks = [
            {
                "text": b.text,
                "confidence": b.confidence,
                "bbox": [b.bbox.x_min, b.bbox.y_min, b.bbox.x_max, b.bbox.y_max]
            }
            for b in layout.text_blocks
            ]

            financial_structured = self.post_processor.process_page({
                "content": content,                 # full page text
                "text_blocks": raw_blocks,           # line-level blocks
                "tables": [
                {
                    "bbox": [t.bbox.x_min, t.bbox.y_min, t.bbox.x_max, t.bbox.y_max],
                    "rows": t.rows,
                    "cols": t.cols,
                    "rows_data": [
                    [
                        {
                            "text": c.text,
                            "confidence": c.confidence,
                            "bbox": [
                                c.bbox.x_min,
                                c.bbox.y_min,
                                c.bbox.x_max,
                                c.bbox.y_max
                            ]
                        }
                        for c in t.cells if c.row == r
                    ]
                    for r in range(t.rows)
                    ]
                }
                for t in layout.tables
                ]
            })

            # Calculate metrics
            char_count, number_count = self.count_chars_and_numbers(content)
            
            all_confidences = [b.confidence for b in layout.text_blocks]
            avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
            
            confidence_level = self.classify_confidence(avg_confidence)
            needs_review = avg_confidence < self.confidence_threshold
            
            low_conf_blocks = [
                {
                    "text": b.text,
                    "confidence": round(float(b.confidence), 3),
                    "bbox": [b.bbox.x_min, b.bbox.y_min, b.bbox.x_max, b.bbox.y_max]
                }
                for b in layout.text_blocks
                if b.confidence < self.confidence_threshold
            ]
            
            return PageResult(
                page_number=task.page_number,
                priority=task.priority,
                queue_type=queue_type,
                content=content,
                layout=layout,
                financial_data=financial_structured,
                char_count=char_count,
                number_count=number_count,
                total_blocks=len(layout.text_blocks),
                table_count=len(layout.tables),
                avg_confidence=avg_confidence,
                confidence_level=confidence_level.value,
                needs_review=needs_review,
                low_confidence_blocks=low_conf_blocks,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            print(f"  ❌ [{worker_name}] page {task.page_number}: {e}")
            return PageResult(
                page_number=task.page_number,
                priority=task.priority,
                queue_type="NORMAL",
                content=f"ERROR: {str(e)}",
                layout=None,
                char_count=0,
                number_count=0,
                total_blocks=0,
                table_count=0,
                avg_confidence=0.0,
                confidence_level="VERY_LOW",
                needs_review=True,
                low_confidence_blocks=[],
                processing_time=time.time() - start_time
            )
    
    def start_workers(self):
        """Khởi động tất cả workers"""
        print(f"\n Starting workers...")
        
        for i in range(self.vip_workers):
            worker = threading.Thread(
                target=self.ocr_worker,
                args=(i + 1, QueueType.VIP),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        for i in range(self.normal_workers):
            worker = threading.Thread(
                target=self.ocr_worker,
                args=(i + 1, QueueType.NORMAL),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        print(f"v Started {len(self.workers)} workers")
    
    def wait_completion(self):
        """Đợi tất cả tasks hoàn thành"""
        print("\n Waiting for all tasks to complete...")
        self.normal_queue.join()
        self.vip_queue.join()
        print("v All tasks completed")
    
    def stop_workers(self):
        """Dừng tất cả workers"""
        print("\nx Stopping workers...")
        self.stop_event.set()
        for worker in self.workers:
            worker.join(timeout=2)
        print("v All workers stopped")
    
    def save_results(self):
        """Lưu kết quả OCR ra JSON (phù hợp pipeline mới)"""
        print(f"\n💾 Saving results to: {self.output_file}")

        # Sort theo page_number
        sorted_results = sorted(self.results, key=lambda r: r.page_number)

        pages_json = {}
        total_tables = 0

        for result in sorted_results:
            page_key = f"page_{result.page_number}"
            total_tables += result.table_count

            pages_json[page_key] = {
                "page_number": result.page_number,
                "queue_type": result.queue_type,
                "priority": result.priority,
                # ===== CONTENT =====
                "content": result.content,
                # ===== STATISTICS =====
                "statistics": {
                    "char_count": result.char_count,
                    "number_count": result.number_count,
                    "text_blocks": result.total_blocks,
                    "table_count": result.table_count,
                    "processing_time_sec": round(result.processing_time, 3)
                },

                # ===== CONFIDENCE =====
                "confidence": {
                    "average": round(result.avg_confidence, 4),
                    "level": result.confidence_level,
                    "needs_review": result.needs_review
                }
            }
            if result.financial_data:
                pages_json[page_key]["financial"] = result.financial_data
            # Chỉ ghi block confidence thấp nếu có
            if result.low_confidence_blocks:
                pages_json[page_key]["low_confidence_blocks"] = result.low_confidence_blocks

        # ===== FINAL OUTPUT =====
        output = {
            "metadata": {
                "total_pages": self.stats["total_pages"],
                "vip_pages": self.stats["vip_pages"],
                "normal_pages": self.stats["total_pages"] - self.stats["vip_pages"],
                "confidence_threshold": self.confidence_threshold,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            },

            "pages": pages_json,

            "summary": {
                "confidence_distribution": {
                    "high": self.stats["high_confidence"],
                    "medium": self.stats["medium_confidence"],
                    "low": self.stats["low_confidence"],
                    "very_low": self.stats["very_low_confidence"]
                },
                "needs_review_pages": self.stats["needs_review"],
                "total_tables_detected": total_tables,
                "total_processing_time_sec": round(
                    self.stats["end_time"] - self.stats["start_time"], 2
                ),
                "avg_time_per_page_sec": round(
                    (self.stats["end_time"] - self.stats["start_time"]) / self.stats["total_pages"],3) 
                    if self.stats["total_pages"] else 0
            }
        }

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print("✅ Results saved successfully")

    def run(self):
        """Chạy toàn bộ quy trình OCR"""
        print("\n" + "="*70)
        print("VIETNAMESE OCR SCANNER WITH CONFIDENCE SCORING")
        print("="*70)
        
        try:
            self.stats['start_time'] = time.time()
            
            image_paths = self.load_images()
            
            if not image_paths:
                print(" No images found")
                return
            
            self.create_initial_tasks(image_paths)
            self.start_workers()
            self.wait_completion()
            self.stop_workers()
            
            self.stats['end_time'] = time.time()
            
            self.save_results()
            self.print_statistics()
            
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
            self.stop_workers()
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            self.stop_workers()
    
    def print_statistics(self):
        """In thống kê cuối cùng"""
        print("\n" + "="*70)
        print("STATISTICS")
        print("="*70)
        
        total_time = self.stats['end_time'] - self.stats['start_time']
        
        print(f"Total Pages:          {self.stats['total_pages']}")
        print(f"  • VIP Pages:        {self.stats['vip_pages']}")
        print(f"  • Normal Pages:     {self.stats['total_pages'] - self.stats['vip_pages']}")
        print(f"\nConfidence Distribution:")
        print(f"  • High (≥0.85):     {self.stats['high_confidence']}")
        print(f"  • Medium (0.70-0.84): {self.stats['medium_confidence']}")
        print(f"  • Low (0.50-0.69):  {self.stats['low_confidence']}")
        print(f"  • Very Low (<0.50): {self.stats['very_low_confidence']}")
        print(f"\n⚠️  Needs Review:      {self.stats['needs_review']} pages")
        print(f"\nProcessing Time:")
        print(f"  • Total:            {total_time:.2f}s")
        print(f"  • Avg/Page:         {total_time / self.stats['total_pages']:.2f}s")
        print("="*70)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Vietnamese OCR Scanner with Confidence Scoring')
    parser.add_argument('image_folder', help='Folder containing images to scan')
    parser.add_argument('--vip-workers', type=int, default=2, help='Number of VIP workers')
    parser.add_argument('--normal-workers', type=int, default=2, help='Number of Normal workers')
    parser.add_argument('--output', default='ocr_results.json', help='Output JSON file')
    parser.add_argument('--confidence-threshold', type=float, default=0.70, 
                       help='Confidence threshold for needs_review flag (default: 0.70)')
    
    args = parser.parse_args()
    
    scanner = DualQueueOCRScanner(
        image_folder=args.image_folder,
        vip_workers=args.vip_workers,
        normal_workers=args.normal_workers,
        output_file=args.output,
        confidence_threshold=args.confidence_threshold
    )
    
    scanner.run()


if __name__ == "__main__":
    main()