"""
ocr_extract.py
Module trích xuất dữ liệu từ ảnh - Tích hợp với Vietnamese OCR Scanner
Tối ưu tốc độ với batch processing và cache
"""

import re
import json
import threading
from typing import List, Dict, Any, Optional
from pathlib import Path
import cv2
import numpy as np
# Import từ scanner module
try:
    from ocr_scan import VietnameseOCRService
    SCANNER_AVAILABLE = True
    print("✓ Using optimized VietnameseOCRService from scanner")
except ImportError:
    SCANNER_AVAILABLE = False
    print("⚠️  Scanner not available, using fallback PaddleOCR")
    from paddleocr import PaddleOCR

from postprocess_financial import PostProcessor


class OptimizedOCRExtractor:
    """
    Module trích xuất OCR tối ưu tốc độ
    - Sử dụng VietnameseOCRService từ scanner (thread-safe)
    - Batch processing
    - Cache kết quả
    - Preprocessing được tối ưu
    """
    
    def __init__(self, 
                 lang='vi', 
                 use_gpu=False, 
                 confidence_threshold=0.5,
                 use_scanner_service=True,
                 enable_cache=True):
        """
        Khởi tạo OCR Extractor tối ưu
        
        Args:
            lang: Ngôn ngữ ('vi' cho tiếng Việt)
            use_gpu: Sử dụng GPU hay không
            confidence_threshold: Ngưỡng độ tin cậy tối thiểu
            use_scanner_service: Sử dụng VietnameseOCRService (nhanh hơn)
            enable_cache: Bật cache kết quả OCR
        """
        self.confidence_threshold = confidence_threshold
        self.enable_cache = enable_cache
        self.cache = {} if enable_cache else None
        self.cache_lock = threading.Lock() if enable_cache else None
        
        # Khởi tạo OCR service
        if use_scanner_service and SCANNER_AVAILABLE:
            self.ocr_service = VietnameseOCRService()
            self.use_scanner = True
            print("✓ Using optimized VietnameseOCRService (thread-safe)")
        else:
            self.ocr_service = None
            self.use_scanner = False
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=lang,
                use_gpu=use_gpu,
                show_log=False,
                det_db_thresh=0.3,
                det_db_box_thresh=0.6,
                rec_batch_num=6,
                drop_score=0.3,
                use_dilation=True,
            )
            print("✓ Using standard PaddleOCR")
        
        self.post_processor = PostProcessor()
        print(f"✓ Confidence threshold: {confidence_threshold}")
        print(f"✓ Cache: {'Enabled' if enable_cache else 'Disabled'}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocessing ảnh được tối ưu từ scanner
        
        Args:
            image: Ảnh đầu vào (BGR)
            
        Returns:
            Ảnh đã được xử lý
        """
        # Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Denoising
        denoised = cv2.fastNlMeansDenoising(
            gray, None, 
            h=10, 
            templateWindowSize=7, 
            searchWindowSize=21
        )
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Adaptive Thresholding
        binary = cv2.adaptiveThreshold(
            enhanced, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to BGR
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        
        return processed
    
    def get_cache_key(self, image_path: str) -> str:
        """Tạo cache key từ đường dẫn ảnh"""
        return str(Path(image_path).absolute())
    
    def get_from_cache(self, image_path: str) -> Optional[List[str]]:
        """Lấy kết quả từ cache"""
        if not self.enable_cache:
            return None
        
        cache_key = self.get_cache_key(image_path)
        
        with self.cache_lock:
            return self.cache.get(cache_key)
    
    def save_to_cache(self, image_path: str, lines: List[str]):
        """Lưu kết quả vào cache"""
        if not self.enable_cache:
            return
        
        cache_key = self.get_cache_key(image_path)
        
        with self.cache_lock:
            self.cache[cache_key] = lines
    
    def extract_text_from_image(self, image_path: str, use_preprocessing=True) -> List[str]:
        """
        Trích xuất văn bản từ ảnh (tối ưu tốc độ)
        
        Args:
            image_path: Đường dẫn đến file ảnh
            use_preprocessing: Sử dụng preprocessing hay không
            
        Returns:
            Danh sách các dòng văn bản đã được sắp xếp
        """
        # Kiểm tra cache trước
        cached_result = self.get_from_cache(image_path)
        if cached_result is not None:
            print(f"  ✓ Cache hit: {image_path}")
            return cached_result
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Preprocessing (optional)
        if use_preprocessing:
            image = self.preprocess_image(image)
        
        # OCR
        if self.use_scanner:
            result = self.ocr_service.ocr(image, det=True, rec=True, cls=False)
        else:
            result = self.ocr.ocr(image, det=True, rec=True, cls=False)
        
        # Extract lines
        lines = []
        if result and result[0]:
            # Sắp xếp theo tọa độ y (từ trên xuống dưới)
            sorted_result = sorted(result[0], key=lambda x: x[0][0][1])
            
            for line in sorted_result:
                text = line[1][0]
                confidence = line[1][1]
                
                # Chỉ lấy các dòng có độ tin cậy cao
                if confidence > self.confidence_threshold:
                    lines.append(text)
        
        # Save to cache
        self.save_to_cache(image_path, lines)
        
        return lines
    
    def extract_from_scanner_result(self, scanner_result_path: str, page_number: int) -> List[str]:
        """
        Trích xuất text từ kết quả đã scan (tái sử dụng kết quả)
        
        Args:
            scanner_result_path: Đường dẫn đến file JSON kết quả scanner
            page_number: Số trang cần trích xuất
            
        Returns:
            Danh sách các dòng văn bản
        """
        with open(scanner_result_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        page_key = f"page_{page_number}"
        
        if page_key not in data.get('pages', {}):
            raise ValueError(f"Page {page_number} not found in scanner result")
        
        page_data = data['pages'][page_key]
        content = page_data.get('content', '')
        
        # Parse content thành lines
        lines = []
        
        # Extract từ [TEXT CONTENT]
        text_match = re.search(r'\[TEXT CONTENT\](.*?)\[/TEXT CONTENT\]', content, re.DOTALL)
        if text_match:
            text_content = text_match.group(1).strip()
            lines = [line.strip() for line in text_content.split('\n') if line.strip()]
        
        return lines
    
    def extract_dates_from_header(self, lines: List[str], max_lines_to_check=10) -> List[str]:
        """
        Trích xuất các ngày từ header của bảng
        
        Args:
            lines: Danh sách các dòng văn bản
            max_lines_to_check: Số dòng đầu tiên cần kiểm tra
            
        Returns:
            Danh sách các ngày tháng (tối đa 2 ngày)
        """
        dates = []
        
        # Tìm các dòng chứa ngày tháng (format dd/mm/yyyy)
        date_pattern = r'\d{2}/\d{2}/\d{4}'
        
        for line in lines[:max_lines_to_check]:
            found_dates = re.findall(date_pattern, line)
            dates.extend(found_dates)
        
        # Nếu không tìm thấy, sử dụng giá trị mặc định
        if not dates:
            dates = ["31/03/2025", "01/01/2025"]
        
        return dates[:2]
    
    def batch_extract_text(self, image_paths: List[str], max_workers=4) -> Dict[str, List[str]]:
        """
        Trích xuất text từ nhiều ảnh đồng thời (parallel processing)
        
        Args:
            image_paths: Danh sách đường dẫn ảnh
            max_workers: Số worker đồng thời
            
        Returns:
            Dict {image_path: lines}
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = {}
        
        print(f"📦 Batch processing {len(image_paths)} images with {max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tất cả tasks
            future_to_path = {
                executor.submit(self.extract_text_from_image, path): path 
                for path in image_paths
            }
            
            # Collect results
            for i, future in enumerate(as_completed(future_to_path), 1):
                path = future_to_path[future]
                try:
                    lines = future.result()
                    results[path] = lines
                    print(f"  ✓ [{i}/{len(image_paths)}] {Path(path).name}: {len(lines)} lines")
                except Exception as e:
                    print(f"  ✗ [{i}/{len(image_paths)}] {Path(path).name}: {e}")
                    results[path] = []
        
        return results
    
    def process_and_export(self, 
                          image_path: str, 
                          output_json_path: str = None, 
                          verbose: bool = True,
                          use_preprocessing: bool = True) -> Dict[str, Any]:
        """
        Xử lý ảnh và xuất ra file JSON
        
        Args:
            image_path: Đường dẫn đến file ảnh
            output_json_path: Đường dẫn file JSON đầu ra (optional)
            verbose: Hiển thị log chi tiết
            use_preprocessing: Sử dụng preprocessing
            
        Returns:
            Dữ liệu JSON đã được cấu trúc
        """
        if verbose:
            print(f"📄 Processing: {image_path}")
        
        # Bước 1: Trích xuất văn bản từ ảnh
        lines = self.extract_text_from_image(image_path, use_preprocessing)
        if verbose:
            print(f"  ✓ Extracted {len(lines)} lines")
        
        # Bước 2: Trích xuất ngày tháng
        dates = self.extract_dates_from_header(lines)
        if verbose:
            print(f"  ✓ Dates: {dates}")
        
        # Bước 3: Xử lý hậu kỳ và xây dựng cấu trúc
        structured_data = self.post_processor.build_structure(lines, dates)
        
        # Bước 4: Tạo cấu trúc JSON cuối cùng
        result = {
            'metadata': {
                'source_image': image_path,
                'dates': dates,
                'total_sections': len(structured_data),
                'total_lines': len(lines)
            },
            'sections': structured_data
        }
        
        # Bước 5: Lưu file JSON (nếu có đường dẫn)
        if output_json_path:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            if verbose:
                print(f"  ✓ Saved to: {output_json_path}")
        
        return result
    
    def process_from_scanner_result(self,
                                   scanner_result_path: str,
                                   page_number: int,
                                   output_json_path: str = None) -> Dict[str, Any]:
        """
        Xử lý từ kết quả scanner có sẵn (tái sử dụng, cực nhanh)
        
        Args:
            scanner_result_path: Đường dẫn file JSON kết quả scanner
            page_number: Số trang cần xử lý
            output_json_path: Đường dẫn file JSON đầu ra
            
        Returns:
            Dữ liệu JSON đã được cấu trúc
        """
        print(f"♻️  Reusing scanner result for page {page_number}")
        
        # Bước 1: Trích xuất text từ scanner result
        lines = self.extract_from_scanner_result(scanner_result_path, page_number)
        print(f"  ✓ Extracted {len(lines)} lines from scanner result")
        
        # Bước 2: Trích xuất ngày tháng
        dates = self.extract_dates_from_header(lines)
        
        # Bước 3: Xử lý hậu kỳ
        structured_data = self.post_processor.build_structure(lines, dates)
        
        # Bước 4: Tạo cấu trúc JSON
        result = {
            'metadata': {
                'source': 'scanner_result',
                'page_number': page_number,
                'dates': dates,
                'total_sections': len(structured_data),
                'total_lines': len(lines)
            },
            'sections': structured_data
        }
        
        # Bước 5: Lưu file JSON
        if output_json_path:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"  ✓ Saved to: {output_json_path}")
        
        return result
    
    def batch_process_from_scanner(self,
                                  scanner_result_path: str,
                                  output_dir: str = None) -> List[Dict[str, Any]]:
        """
        Xử lý batch từ kết quả scanner (cực nhanh - không cần OCR lại)
        
        Args:
            scanner_result_path: Đường dẫn file JSON kết quả scanner
            output_dir: Thư mục lưu các file JSON
            
        Returns:
            Danh sách kết quả đã xử lý
        """
        # Load scanner result
        with open(scanner_result_path, 'r', encoding='utf-8') as f:
            scanner_data = json.load(f)
        
        pages = scanner_data.get('pages', {})
        total_pages = len(pages)
        
        print(f"♻️  Batch processing {total_pages} pages from scanner result")
        
        results = []
        
        for i, page_key in enumerate(sorted(pages.keys()), 1):
            page_number = int(page_key.split('_')[1])
            
            output_path = None
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                output_path = str(Path(output_dir) / f"page_{page_number}.json")
            
            try:
                result = self.process_from_scanner_result(
                    scanner_result_path, 
                    page_number, 
                    output_path
                )
                results.append(result)
                print(f"  ✓ [{i}/{total_pages}] Page {page_number}")
            except Exception as e:
                print(f"  ✗ [{i}/{total_pages}] Page {page_number}: {e}")
                results.append(None)
        
        return results
    
    def clear_cache(self):
        """Xóa cache"""
        if self.enable_cache:
            with self.cache_lock:
                self.cache.clear()
                print("✓ Cache cleared")


# Ví dụ sử dụng
if __name__ == "__main__":
    print("="*70)
    print("OPTIMIZED OCR EXTRACTOR - EXAMPLES")
    print("="*70)
    
    # ===== CÁCH 1: Extract trực tiếp từ ảnh (có cache) =====
    print("\n=== CÁCH 1: Extract từ ảnh với cache ===")
    extractor = OptimizedOCRExtractor(
        lang='vi',
        use_scanner_service=True,  # Sử dụng service từ scanner
        enable_cache=True           # Bật cache
    )
    
    result = extractor.process_and_export(
        image_path='balance_sheet.png',
        output_json_path='output.json'
    )
    
    print(f"\n✓ Processed successfully")
    print(f"  Sections: {len(result['sections'])}")
    
    
    # ===== CÁCH 2: Batch processing nhiều ảnh =====
    print("\n\n=== CÁCH 2: Batch processing ===")
    
    image_files = [
        'image1.png',
        'image2.png',
        'image3.png'
    ]
    
    batch_results = extractor.batch_extract_text(
        image_paths=image_files,
        max_workers=4  # 4 workers đồng thời
    )
    
    print(f"\n✓ Batch processed {len(batch_results)} images")
    
    
    # ===== CÁCH 3: Tái sử dụng kết quả từ scanner (NHANH NHẤT) =====
    print("\n\n=== CÁCH 3: Reuse scanner result (FASTEST) ===")
    
    # Giả sử bạn đã chạy scanner trước và có file ocr_results.json
    result_from_scanner = extractor.process_from_scanner_result(
        scanner_result_path='ocr_results.json',
        page_number=1,
        output_json_path='page_1_structured.json'
    )
    
    print(f"\n✓ Reused scanner result")
    print(f"  Sections: {len(result_from_scanner['sections'])}")
    
    
    # ===== CÁCH 4: Batch processing từ scanner result =====
    print("\n\n=== CÁCH 4: Batch from scanner (ALL PAGES) ===")
    
    all_results = extractor.batch_process_from_scanner(
        scanner_result_path='ocr_results.json',
        output_dir='./structured_output'
    )
    
    print(f"\n✓ Processed {len([r for r in all_results if r])} pages")
    
    
    # Clear cache
    print("\n\n=== Cache Management ===")
    extractor.clear_cache()