"""
ocr_extract_module.py
Module OCR Extract - Parse OCR scan results thành JSON có cấu trúc tài chính
Giữ nguyên ngữ cảnh số liệu, xử lý tiếng Việt
"""

import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class FinancialItem:
    """Một khoản mục trong báo cáo tài chính"""
    code: str
    name: str
    values: Dict[str, Any]  # {date: value}
    confidence: float
    source_lines: List[int]
    context: str  # Ngữ cảnh: thuộc section nào
    
    def to_dict(self):
        return {
            'code': self.code,
            'name': self.name,
            'values': self.values,
            'confidence': self.confidence,
            'source_lines': self.source_lines,
            'context': self.context
        }


@dataclass
class FinancialSection:
    """Một section trong báo cáo tài chính"""
    section_name: str
    section_code: str
    items: List[FinancialItem]
    section_confidence: float
    hierarchy_level: int  # Cấp độ phân cấp
    parent_section: Optional[str] = None
    
    def to_dict(self):
        return {
            'section_name': self.section_name,
            'section_code': self.section_code,
            'items': [item.to_dict() for item in self.items],
            'section_confidence': self.section_confidence,
            'hierarchy_level': self.hierarchy_level,
            'parent_section': self.parent_section
        }


@dataclass
class FinancialReport:
    """Báo cáo tài chính hoàn chỉnh"""
    report_type: str  # "balance_sheet", "income_statement", etc.
    company_name: Optional[str]
    report_dates: List[str]
    sections: List[FinancialSection]
    metadata: Dict[str, Any]
    
    def to_dict(self):
        return {
            'report_type': self.report_type,
            'company_name': self.company_name,
            'report_dates': self.report_dates,
            'sections': [section.to_dict() for section in self.sections],
            'metadata': self.metadata
        }


class OCRExtractModule:
    """
    Module OCR Extract - Parse scan results thành cấu trúc tài chính
    
    Features:
    - Parse sections với hierarchy
    - Trích xuất items với code + name + values
    - Giữ nguyên ngữ cảnh (context)
    - Xử lý tiếng Việt
    - Nhận dạng report type
    """
    
    # Keywords để nhận dạng report type
    REPORT_TYPE_KEYWORDS = {
        'balance_sheet': [
            'bảng cân đối kế toán',
            'balance sheet',
            'tài sản',
            'nguồn vốn',
            'assets',
            'liabilities'
        ],
        'income_statement': [
            'báo cáo kết quả hoạt động kinh doanh',
            'income statement',
            'doanh thu',
            'chi phí',
            'lợi nhuận',
            'revenue',
            'profit'
        ],
        'cash_flow': [
            'báo cáo lưu chuyển tiền tệ',
            'cash flow',
            'lưu chuyển tiền',
            'cash flow statement'
        ]
    }
    
    # Keywords để nhận dạng sections
    SECTION_KEYWORDS = [
        'tài sản',
        'nguồn vốn',
        'nợ phải trả',
        'vốn chủ sở hữu',
        'doanh thu',
        'chi phí',
        'lợi nhuận',
        'assets',
        'liabilities',
        'equity',
        'revenue',
        'expenses'
    ]
    
    def __init__(self):
        """Khởi tạo OCR Extract Module"""
        print("✓ OCR Extract Module initialized")
    
    def load_scan_result(self, scan_json_path: str) -> Dict[str, Any]:
        """Load scan result từ JSON"""
        with open(scan_json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_dates(self, lines: List[Dict]) -> List[str]:
        """
        Trích xuất các ngày tháng từ header
        
        Args:
            lines: List OCR lines
            
        Returns:
            List dates
        """
        dates = []
        
        # Patterns
        date_patterns = [
            r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4})',
            r'(\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4})',
            r'(Ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4})',
        ]
        
        # Tìm trong 15 dòng đầu
        for line in lines[:15]:
            text = line.get('text', '')
            
            for pattern in date_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Normalize
                    date_str = re.sub(r'[/\-\.]', '/', match)
                    date_str = re.sub(r'Ngày\s+', '', date_str, flags=re.IGNORECASE)
                    date_str = date_str.strip()
                    
                    if date_str not in dates:
                        dates.append(date_str)
        
        return dates
    
    def extract_company_name(self, lines: List[Dict]) -> Optional[str]:
        """Trích xuất tên công ty từ header"""
        # Tìm trong 10 dòng đầu
        for line in lines[:10]:
            text = line.get('text', '').strip()
            
            # Pattern: "Công ty ...", "CÔNG TY ..."
            if re.search(r'(công ty|CÔNG TY)', text, re.IGNORECASE):
                # Clean up
                company = re.sub(r'(công ty|CÔNG TY)', 'Công ty', text, flags=re.IGNORECASE)
                return company.strip()
        
        return None
    
    def detect_report_type(self, lines: List[Dict]) -> str:
        """
        Nhận dạng loại báo cáo
        
        Args:
            lines: List OCR lines
            
        Returns:
            Report type
        """
        # Combine text từ 20 dòng đầu
        text = ' '.join([line.get('text', '') for line in lines[:20]]).lower()
        
        # Check keywords
        for report_type, keywords in self.REPORT_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    return report_type
        
        return 'unknown'
    
    def is_section_header(self, text: str) -> bool:
        """
        Kiểm tra xem có phải section header không
        
        Args:
            text: Text cần check
            
        Returns:
            True nếu là section header
        """
        text_lower = text.lower().strip()
        
        # Check 1: Chữ IN HOA (nhiều hơn 70%)
        if text.isupper() and len(text) > 5:
            return True
        
        # Check 2: Có keywords
        for keyword in self.SECTION_KEYWORDS:
            if keyword in text_lower:
                return True
        
        # Check 3: Pattern dạng "A. TÊN SECTION"
        if re.match(r'^[A-Z]\.\s+[A-ZẮẰẲẴẶĂÂẤẦẨẪẬÊẾỀỂỄỆÔỐỒỔỖỘƠỚỜỞỠỢƯỨỪỬỮỰĐ\s]+$', text):
            return True
        
        return False
    
    def parse_code_and_name(self, text: str) -> Optional[tuple]:
        """
        Parse code và name từ text
        
        Args:
            text: Text line
            
        Returns:
            (code, name) hoặc None
        """
        # Patterns
        patterns = [
            r'^(\d{2,4})\s*[-.\s]\s*(.+)$',  # "100 - Tài sản ngắn hạn"
            r'^(\d{2,4})\.\s*(.+)$',          # "100. Tài sản ngắn hạn"
            r'^(\d{2,4})\s+(.+)$',            # "100 Tài sản ngắn hạn"
        ]
        
        for pattern in patterns:
            match = re.match(pattern, text.strip())
            if match:
                code = match.group(1)
                name = match.group(2).strip()
                return (code, name)
        
        return None
    
    def extract_numbers_from_text(self, text: str) -> List[float]:
        """
        Trích xuất các số từ text
        
        Args:
            text: Text chứa số
            
        Returns:
            List các số
        """
        # Pattern cho số (có thể có dấu phẩy, chấm, khoảng trắng)
        pattern = r'[\d\s,\.]+\d'
        
        matches = re.findall(pattern, text)
        
        numbers = []
        for match in matches:
            # Clean và convert
            clean = match.replace(' ', '').replace(',', '')
            try:
                # Check if integer or float
                if '.' in clean:
                    num = float(clean)
                else:
                    num = int(clean)
                numbers.append(num)
            except:
                continue
        
        return numbers
    
    def parse_financial_structure(self,
                                  lines: List[Dict],
                                  dates: List[str]) -> List[FinancialSection]:
        """
        Parse cấu trúc tài chính từ OCR lines
        
        Args:
            lines: List OCR lines
            dates: List dates trong báo cáo
            
        Returns:
            List FinancialSection
        """
        sections = []
        current_section = None
        current_section_name = ""
        current_section_code = ""
        hierarchy_level = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            text = line.get('text', '').strip()
            confidence = line.get('confidence', 0.0)
            line_number = line.get('line_number', i + 1)
            
            # Check if section header
            if self.is_section_header(text):
                # Save previous section
                if current_section:
                    sections.append(current_section)
                
                # Start new section
                current_section_name = text
                current_section_code = ""
                hierarchy_level += 1
                
                current_section = FinancialSection(
                    section_name=current_section_name,
                    section_code=current_section_code,
                    items=[],
                    section_confidence=confidence,
                    hierarchy_level=hierarchy_level
                )
                
                i += 1
                continue
            
            # Parse item (code + name + values)
            parsed = self.parse_code_and_name(text)
            
            if parsed and current_section:
                code, name = parsed
                
                # Extract values từ dòng này và có thể các dòng tiếp theo
                values = {}
                source_lines = [line_number]
                
                # Tìm số trong dòng này
                numbers = self.extract_numbers_from_text(text)
                
                # Nếu không đủ số, tìm trong dòng tiếp theo
                j = i + 1
                while j < len(lines) and len(numbers) < len(dates):
                    next_text = lines[j].get('text', '')
                    
                    # Stop nếu gặp code mới
                    if self.parse_code_and_name(next_text):
                        break
                    
                    # Stop nếu gặp section mới
                    if self.is_section_header(next_text):
                        break
                    
                    # Extract numbers
                    next_numbers = self.extract_numbers_from_text(next_text)
                    if next_numbers:
                        numbers.extend(next_numbers)
                        source_lines.append(lines[j].get('line_number', j + 1))
                    
                    j += 1
                
                # Map numbers to dates
                for idx, date in enumerate(dates):
                    if idx < len(numbers):
                        values[date] = numbers[idx]
                    else:
                        values[date] = None
                
                # Create item
                item = FinancialItem(
                    code=code,
                    name=name,
                    values=values,
                    confidence=confidence,
                    source_lines=source_lines,
                    context=current_section_name
                )
                
                current_section.items.append(item)
                
                # Update section code nếu chưa có
                if not current_section.section_code:
                    current_section.section_code = code
            
            i += 1
        
        # Add last section
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def extract(self,
               scan_json_path: str,
               output_json_path: Optional[str] = None,
               verbose: bool = True) -> FinancialReport:
        """
        Extract cấu trúc tài chính từ scan result
        
        Args:
            scan_json_path: Path to scan JSON
            output_json_path: Path to save output
            verbose: Show logs
            
        Returns:
            FinancialReport
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"OCR EXTRACTING: {Path(scan_json_path).name}")
            print(f"{'='*70}")
        
        # Load scan result
        scan_data = self.load_scan_result(scan_json_path)
        lines = scan_data.get('lines', [])
        
        if verbose:
            print(f"Total lines: {len(lines)}")
        
        # Extract metadata
        dates = self.extract_dates(lines)
        company_name = self.extract_company_name(lines)
        report_type = self.detect_report_type(lines)
        
        if verbose:
            print(f"Report type: {report_type}")
            print(f"Company: {company_name or 'Unknown'}")
            print(f"Dates: {dates}")
        
        # Parse structure
        if verbose:
            print("Parsing financial structure...")
        
        sections = self.parse_financial_structure(lines, dates)
        
        # Create report
        report = FinancialReport(
            report_type=report_type,
            company_name=company_name,
            report_dates=dates,
            sections=sections,
            metadata={
                'source_image': scan_data.get('image_path', ''),
                'extraction_time': datetime.now().isoformat(),
                'total_lines': len(lines),
                'average_confidence': scan_data.get('average_confidence', 0.0),
                'total_sections': len(sections),
                'total_items': sum(len(s.items) for s in sections)
            }
        )
        
        if verbose:
            print(f"\n📊 Extract Results:")
            print(f"  Sections: {len(sections)}")
            print(f"  Items: {report.metadata['total_items']}")
            print(f"{'='*70}\n")
        
        # Save
        if output_json_path:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
            
            if verbose:
                print(f"✓ Extract result saved: {output_json_path}")
        
        return report


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("OCR EXTRACT MODULE - EXAMPLE")
    print("="*70)
    
    # Initialize
    extractor = OCRExtractModule()
    
    # Extract from scan result
    report = extractor.extract(
        scan_json_path='scan_result.json',
        output_json_path='extracted_report.json',
        verbose=True
    )
    
    print("\n📄 Report Summary:")
    print(f"  Type: {report.report_type}")
    print(f"  Company: {report.company_name}")
    print(f"  Dates: {report.report_dates}")
    print(f"  Sections: {len(report.sections)}")
