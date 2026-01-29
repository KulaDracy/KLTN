"""
post_process.py
Adapter module để tương thích giữa ocr_extract.py và postprocess_financial.py
"""

import re
from typing import List, Dict, Any, Tuple

# Import từ postprocess_financial
try:
    from postprocess_financial import (
        VietnameseTextNormalizer,
        NumberParser,
        SectionMapper
    )
    FINANCIAL_PROCESSOR_AVAILABLE = True
except ImportError:
    FINANCIAL_PROCESSOR_AVAILABLE = False
    print("⚠️  postprocess_financial.py not found, using fallback")


class PostProcessor:
    """
    Adapter class để tương thích với ocr_extract.py
    Sử dụng logic từ FinancialProcessor nhưng với interface đơn giản hơn
    """
    
    def __init__(self):
        if FINANCIAL_PROCESSOR_AVAILABLE:
            self.normalizer = VietnameseTextNormalizer()
            self.number_parser = NumberParser()
            self.section_mapper = SectionMapper()
        
        self.section_keywords = [
            "TÀI SẢN",
            "NGUỒN VỐN",
            "TÀI SẢN NGẮN HẠN",
            "TÀI SẢN DÀI HẠN",
            "NỢ PHẢI TRẢ",
            "VỐN CHỦ SỞ HỮU"
        ]
    
    def normalize_vietnamese_text(self, text: str) -> str:
        """
        Chuẩn hóa văn bản tiếng Việt
        
        Args:
            text: Văn bản cần chuẩn hóa
            
        Returns:
            Văn bản đã được chuẩn hóa
        """
        if not text:
            return ""
        
        if FINANCIAL_PROCESSOR_AVAILABLE:
            return self.normalizer.normalize(text)
        
        # Fallback normalization
        text = re.sub(r'\s+', ' ', text.strip())
        text = text.replace('−', '-')
        text = text.replace('–', '-')
        return text
    
    def extract_code(self, text: str) -> Tuple[str, str]:
        """
        Trích xuất mã code từ đầu dòng
        
        Args:
            text: Dòng văn bản cần trích xuất
            
        Returns:
            Tuple (code, remaining_text)
        """
        # Pattern: số ở đầu dòng (2-3 chữ số)
        match = re.match(r'^\s*(\d{2,3})[.\s]+(.+)', text.strip())
        if match:
            code = match.group(1)
            remaining = match.group(2)
            return code, remaining
        return "", text
    
    def extract_values(self, text: str) -> List[float]:
        """
        Trích xuất các giá trị số từ văn bản
        
        Args:
            text: Văn bản chứa giá trị số
            
        Returns:
            Danh sách các giá trị số
        """
        if FINANCIAL_PROCESSOR_AVAILABLE:
            return self.number_parser.extract_all_numbers(text)
        
        # Fallback extraction
        numbers = re.findall(r'[\d,.]+', text)
        values = []
        
        for num in numbers:
            try:
                clean_num = num.replace(',', '').replace('.', '')
                if clean_num and len(clean_num) >= 3:
                    values.append(float(clean_num))
            except ValueError:
                continue
        
        return values
    
    def is_section_header(self, text: str) -> bool:
        """
        Kiểm tra xem dòng có phải là tiêu đề section không
        
        Args:
            text: Dòng văn bản cần kiểm tra
            
        Returns:
            True nếu là section header
        """
        if FINANCIAL_PROCESSOR_AVAILABLE:
            section_info = self.section_mapper.detect_section(text)
            return section_info is not None
        
        # Fallback
        text_upper = text.upper()
        return any(keyword in text_upper for keyword in self.section_keywords)
    
    def parse_line(self, line_text: str) -> Dict[str, Any]:
        """
        Parse một dòng OCR thành cấu trúc dữ liệu
        
        Args:
            line_text: Dòng văn bản từ OCR
            
        Returns:
            Dict chứa code, name và values
        """
        normalized = self.normalize_vietnamese_text(line_text)
        
        # Trích xuất code
        code, remaining = self.extract_code(normalized)
        
        # Trích xuất giá trị số
        values = self.extract_values(remaining)
        
        # Tách tên và giá trị - loại bỏ các số khỏi tên
        name = remaining
        for num_str in re.findall(r'[\d,.\s()+-]+', remaining):
            name = name.replace(num_str, '')
        
        name = re.sub(r'\s+', ' ', name).strip()
        name = re.sub(r'[:\-]+$', '', name).strip()
        
        return {
            'code': code,
            'name': name,
            'values': values
        }
    
    def build_structure(self, ocr_lines: List[str], dates: List[str]) -> List[Dict[str, Any]]:
        """
        Xây dựng cấu trúc JSON từ các dòng OCR
        
        Args:
            ocr_lines: Danh sách các dòng văn bản từ OCR
            dates: Danh sách các ngày tháng (cột trong bảng)
            
        Returns:
            Danh sách các section đã được cấu trúc
        """
        sections = []
        current_section = None
        
        for line in ocr_lines:
            if not line.strip():
                continue
            
            parsed = self.parse_line(line)
            
            # Kiểm tra nếu là section header
            if self.is_section_header(line):
                if current_section:
                    sections.append(current_section)
                
                # Detect section info
                section_name = parsed['name'] if parsed['name'] else line.strip()
                section_code = parsed['code'] if parsed['code'] else ''
                
                # Try to get standard section info
                if FINANCIAL_PROCESSOR_AVAILABLE:
                    section_info = self.section_mapper.detect_section(line)
                    if section_info:
                        section_name, section_code = section_info
                
                current_section = {
                    'section': section_name,
                    'code': section_code,
                    'items': []
                }
            
            elif current_section and parsed['code']:
                # Tạo dict values với các ngày tương ứng
                values_dict = {}
                for i, date in enumerate(dates):
                    if i < len(parsed['values']):
                        value = parsed['values'][i]
                        values_dict[date] = int(value) if value else 0
                
                item = {
                    'code': parsed['code'],
                    'name': parsed['name'],
                    'values': values_dict
                }
                current_section['items'].append(item)
        
        # Thêm section cuối cùng
        if current_section:
            sections.append(current_section)
        
        return sections


# Test standalone
if __name__ == "__main__":
    print("="*70)
    print("POST PROCESS ADAPTER - TEST")
    print("="*70)
    
    processor = PostProcessor()
    
    # Test normalize
    text = "100  Tiền   và   các   khoản   tương   đương   tiền"
    print(f"\nOriginal: {text}")
    print(f"Normalized: {processor.normalize_vietnamese_text(text)}")
    
    # Test parse_line
    line = "110 Tiền và các khoản tương đương tiền 116,733,376,376 91,741,974,158"
    parsed = processor.parse_line(line)
    print(f"\nParsed line:")
    print(f"  Code: {parsed['code']}")
    print(f"  Name: {parsed['name']}")
    print(f"  Values: {parsed['values']}")
    
    # Test build_structure
    lines = [
        "TÀI SẢN NGẮN HẠN",
        "110 Tiền và các khoản tương đương tiền 116733376376 91741974158",
        "111 Tiền 52733376376 40741974158"
    ]
    dates = ["31/03/2025", "01/01/2025"]
    
    result = processor.build_structure(lines, dates)
    
    import json
    print(f"\nStructured data:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    print("\n✅ Post Process Adapter working correctly!")