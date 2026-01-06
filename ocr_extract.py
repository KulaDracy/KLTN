"""
=============================================================================
OCR Extract Module
- Parse OCR_scan JSON output
- Extract structured financial data (label + value pairs)
- Calculate extraction confidence
- Identify items needing VLM supervision
=============================================================================
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence levels"""
    HIGH = "HIGH"          # >= 0.85
    MEDIUM = "MEDIUM"      # 0.70 - 0.84
    LOW = "LOW"            # 0.50 - 0.69
    VERY_LOW = "VERY_LOW"  # < 0.50


@dataclass
class ExtractedItem:
    """A single extracted financial item"""
    label: str
    label_normalized: str
    value: Optional[str]
    value_numeric: Optional[float]
    confidence: float
    page_number: int
    source: str  # "ocr_extract"
    parse_error: bool = False
    metadata: Dict = None


@dataclass
class OCRExtractResult:
    """Result from OCR extraction"""
    items: List[ExtractedItem]
    total_items: int
    avg_confidence: float
    confidence_level: ConfidenceLevel
    needs_vlm: bool
    issues: List[Dict]
    metadata: Dict


class FinancialLabelNormalizer:
    """Normalize Vietnamese financial labels to standard keys"""
    
    # Comprehensive label mappings
    LABEL_MAPPINGS = {
        # Assets - Tài sản
        'tài sản': 'assets',
        'tai san': 'assets',
        'tổng tài sản': 'total_assets',
        'tong tai san': 'total_assets',
        
        # Current Assets - Tài sản ngắn hạn
        'tài sản ngắn hạn': 'current_assets',
        'tai san ngan han': 'current_assets',
        'taisan ngan han': 'current_assets',
        
        # Cash - Tiền
        'tiền': 'cash',
        'tien': 'cash',
        'tiền và các khoản tương đương tiền': 'cash_equivalents',
        'tien va cac khoan tuong duong tien': 'cash_equivalents',
        'các khoản tương đương tiền': 'cash_equivalents',
        'cac khoan tuong duong tien': 'cash_equivalents',
        
        # Short-term investments - Đầu tư tài chính ngắn hạn
        'các khoản đầu tư tài chính ngắn hạn': 'short_term_investments',
        'cac khoan dau tu tai chinh ngan han': 'short_term_investments',
        'đầu tư ngắn hạn': 'short_term_investments',
        'chứng khoán kinh doanh': 'trading_securities',
        'chung khoan kinh doanh': 'trading_securities',
        'đầu tư nắm giữ đến ngày đáo hạn': 'held_to_maturity_investments',
        'dau tu nam giu den ngay dao han': 'held_to_maturity_investments',
        
        # Receivables - Phải thu
        'các khoản phải thu': 'receivables',
        'cac khoan phai thu': 'receivables',
        'phải thu ngắn hạn': 'short_term_receivables',
        'phai thu ngan han': 'short_term_receivables',
        'phải thu ngắn hạn của khách hàng': 'accounts_receivable',
        'phai thu ngan han cua khach hang': 'accounts_receivable',
        'phải thu của khách hàng': 'accounts_receivable',
        'trả trước cho người bán': 'prepaid_to_suppliers',
        'tra truoc cho nguoi ban': 'prepaid_to_suppliers',
        'phải thu nội bộ': 'internal_receivables',
        'phai thu noi bo': 'internal_receivables',
        'phải thu về cho vay': 'loan_receivables',
        'phai thu ve cho vay': 'loan_receivables',
        'phải thu khác': 'other_receivables',
        'phai thu khac': 'other_receivables',
        'dự phòng phải thu khó đòi': 'allowance_for_doubtful_accounts',
        'du phong phai thu kho doi': 'allowance_for_doubtful_accounts',
        
        # Inventory - Hàng tồn kho
        'hàng tồn kho': 'inventory',
        'hang ton kho': 'inventory',
        'dự phòng giảm giá hàng tồn kho': 'inventory_provision',
        'du phong giam gia hang ton kho': 'inventory_provision',
        
        # Other current assets - Tài sản ngắn hạn khác
        'tài sản ngắn hạn khác': 'other_current_assets',
        'tai san ngan han khac': 'other_current_assets',
        'chi phí trả trước': 'prepaid_expenses',
        'chi phi tra truoc': 'prepaid_expenses',
        'thuế gtgt được khấu trừ': 'vat_deductible',
        'thue gtgt duoc khau tru': 'vat_deductible',
        'thuế và các khoản khác phải thu': 'tax_receivables',
        'thue va cac khoan khac phai thu': 'tax_receivables',
        
        # Non-current assets - Tài sản dài hạn
        'tài sản dài hạn': 'non_current_assets',
        'tai san dai han': 'non_current_assets',
        
        # Liabilities - Nợ phải trả
        'nợ phải trả': 'liabilities',
        'no phai tra': 'liabilities',
        'tổng nợ phải trả': 'total_liabilities',
        
        # Equity - Vốn chủ sở hữu
        'vốn chủ sở hữu': 'equity',
        'von chu so huu': 'equity',
        
        # Income statement items
        'doanh thu': 'revenue',
        'lợi nhuận': 'profit',
        'loi nhuan': 'profit',
    }
    
    @classmethod
    def normalize(cls, label: str) -> str:
        """
        Normalize Vietnamese label to standard key
        
        Args:
            label: Original Vietnamese label
            
        Returns:
            Normalized key or cleaned label
        """
        if not label:
            return ""
        
        # Clean label
        cleaned = label.lower().strip()
        
        # Remove Roman numerals at start: I. II. III. etc
        cleaned = re.sub(r'^[ivxIVX]+\.?\s*', '', cleaned)
        
        # Remove numbering: 1. 2. 3. etc
        cleaned = re.sub(r'^\d+\.?\s*', '', cleaned)
        
        # Remove excess whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Try exact match first
        if cleaned in cls.LABEL_MAPPINGS:
            return cls.LABEL_MAPPINGS[cleaned]
        
        # Try partial match
        for key, value in cls.LABEL_MAPPINGS.items():
            if key in cleaned or cleaned in key:
                return value
        
        # Return cleaned label as fallback
        return cleaned.replace(' ', '_')


class NumberParser:
    """Parse Vietnamese number formats"""
    
    @staticmethod
    def parse(value_str: str) -> Optional[float]:
        """
        Parse Vietnamese number formats
        
        Handles:
        - 1,234,567 or 1.234.567 (thousand separators)
        - (123) for negative
        - -123
        - 123,456.78 (decimal)
        
        Args:
            value_str: String containing number
            
        Returns:
            Float value or None if parse fails
        """
        if not value_str or not isinstance(value_str, str):
            return None
        
        # Remove currency symbols and units
        cleaned = re.sub(
            r'[đĐ$€£¥₫]|VNĐ|USD|EUR|triệu|tỷ|nghìn|đồng|Dong',
            '',
            value_str,
            flags=re.IGNORECASE
        )
        cleaned = cleaned.strip()
        
        if not cleaned:
            return None
        
        # Handle negative in parentheses: (123) → -123
        is_negative = False
        if cleaned.startswith('(') and cleaned.endswith(')'):
            cleaned = cleaned[1:-1]
            is_negative = True
        
        # Handle explicit negative sign
        if cleaned.startswith('-'):
            is_negative = True
            cleaned = cleaned[1:]
        
        # Detect format
        # Count dots and commas
        dot_count = cleaned.count('.')
        comma_count = cleaned.count(',')
        
        # Case 1: No separators
        if dot_count == 0 and comma_count == 0:
            try:
                value = float(cleaned)
                return -value if is_negative else value
            except ValueError:
                return None
        
        # Case 2: European format (1.234.567,89)
        if dot_count > 1:
            cleaned = cleaned.replace('.', '')
            cleaned = cleaned.replace(',', '.')
        
        # Case 3: US format (1,234,567.89)
        elif comma_count > 1:
            cleaned = cleaned.replace(',', '')
        
        # Case 4: Single separator
        elif dot_count == 1 and comma_count == 0:
            # Could be decimal or thousand separator
            parts = cleaned.split('.')
            if len(parts[-1]) == 3:  # 1.234 (thousand)
                cleaned = cleaned.replace('.', '')
            # else: 1.5 (decimal) - keep as is
        
        elif comma_count == 1 and dot_count == 0:
            # Could be decimal or thousand separator
            parts = cleaned.split(',')
            if len(parts[-1]) == 3:  # 1,234 (thousand)
                cleaned = cleaned.replace(',', '')
            else:  # 1,5 (decimal)
                cleaned = cleaned.replace(',', '.')
        
        # Try final conversion
        try:
            # Remove any remaining non-digit chars except dot
            cleaned = re.sub(r'[^\d.]', '', cleaned)
            value = float(cleaned)
            return -value if is_negative else value
        except ValueError:
            return None
    
    @staticmethod
    def looks_like_number(text: str) -> bool:
        """Check if text looks like it should be a number"""
        if not text:
            return False
        # Has at least one digit
        return bool(re.search(r'\d', text))


class OCRExtractor:
    """
    Extract structured financial data from OCR_scan JSON output
    """
    
    # Confidence thresholds
    CONFIDENCE_HIGH = 0.85
    CONFIDENCE_MEDIUM = 0.70
    CONFIDENCE_LOW = 0.50
    
    def __init__(self, confidence_threshold: float = 0.85):
        """
        Initialize OCR Extractor
        
        Args:
            confidence_threshold: Threshold to trigger VLM (default: 0.85)
        """
        self.confidence_threshold = confidence_threshold
        self.normalizer = FinancialLabelNormalizer()
        self.parser = NumberParser()
        
        logger.info(f"OCR Extractor initialized (threshold: {confidence_threshold})")
    
    def extract_from_json(self, json_path: str) -> OCRExtractResult:
        """
        Extract structured data from OCR_scan JSON file
        
        Args:
            json_path: Path to OCR_scan output JSON
            
        Returns:
            OCRExtractResult with extracted items
        """
        logger.info(f"Extracting from: {json_path}")
        
        # Load JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)
        
        # Extract metadata
        metadata = ocr_data.get('metadata', {})
        pages = ocr_data.get('pages', {})
        
        logger.info(f"  Total pages: {metadata.get('total_pages', len(pages))}")
        
        # Process all pages
        all_items = []
        all_confidences = []
        issues = []
        
        for page_key, page_data in pages.items():
            page_num = int(page_key.split()[-1])
            
            # Extract items from page
            page_items, page_issues = self._extract_from_page(
                page_num=page_num,
                page_data=page_data
            )
            
            all_items.extend(page_items)
            all_confidences.extend([item.confidence for item in page_items])
            issues.extend(page_issues)
        
        # Calculate overall metrics
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
        confidence_level = self._classify_confidence(avg_confidence)
        needs_vlm = self._should_use_vlm(avg_confidence, issues)
        
        logger.info(f"  Extracted {len(all_items)} items")
        logger.info(f"  Average confidence: {avg_confidence:.3f}")
        logger.info(f"  Needs VLM: {needs_vlm}")
        
        return OCRExtractResult(
            items=all_items,
            total_items=len(all_items),
            avg_confidence=avg_confidence,
            confidence_level=confidence_level,
            needs_vlm=needs_vlm,
            issues=issues,
            metadata={
                'source': 'ocr_extract',
                'input_file': json_path,
                'total_pages': metadata.get('total_pages'),
                'processing_date': metadata.get('processing_date')
            }
        )
    
    def _extract_from_page(
        self,
        page_num: int,
        page_data: Dict
    ) -> Tuple[List[ExtractedItem], List[Dict]]:
        """
        Extract financial items from a single page
        
        Args:
            page_num: Page number
            page_data: Page data from OCR_scan JSON
            
        Returns:
            (extracted_items, issues)
        """
        content = page_data.get('content', '')
        page_confidence = page_data.get('confidence', {}).get('average', 0.0)
        
        items = []
        issues = []
        
        # Split content into lines
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to parse as financial item (label + value)
            parsed = self._parse_line_to_item(line, page_num, page_confidence)
            
            if parsed:
                items.append(parsed)
                
                # Check for issues
                if parsed.parse_error:
                    issues.append({
                        'page': page_num,
                        'item': parsed.label,
                        'reason': 'parse_error',
                        'confidence': parsed.confidence
                    })
                
                if parsed.confidence < self.CONFIDENCE_MEDIUM:
                    issues.append({
                        'page': page_num,
                        'item': parsed.label,
                        'reason': 'low_confidence',
                        'confidence': parsed.confidence
                    })
        
        return items, issues
    
    def _parse_line_to_item(
        self,
        line: str,
        page_num: int,
        page_confidence: float
    ) -> Optional[ExtractedItem]:
        """
        Parse a line of text into financial item
        
        Expected formats:
        - "Tài sản ngắn hạn 100,000,000"
        - "I. Tài sản ngắn hạn: 100,000,000"
        - "1. Tiền 50,000,000"
        
        Args:
            line: Text line
            page_num: Page number
            page_confidence: Page-level confidence
            
        Returns:
            ExtractedItem or None
        """
        # Pattern: label followed by number(s)
        # Try multiple patterns
        
        # Pattern 1: Label : Number (with colon)
        match = re.search(r'^(.+?)[:：]\s*([\d,.\s()+-]+(?:\s*[\d,.\s()+-]+)*)$', line)
        
        if not match:
            # Pattern 2: Label  Number (spaces)
            match = re.search(r'^(.+?)\s{2,}([\d,.\s()+-]+(?:\s*[\d,.\s()+-]+)*)$', line)
        
        if not match:
            # Pattern 3: Label Number (at least one space)
            match = re.search(r'^(.+?)\s+([\d,.\s()+-]+)$', line)
        
        if not match:
            # No number found - could be just a label
            # Still extract if it's a financial term
            label = line.strip()
            if self._is_financial_label(label):
                return ExtractedItem(
                    label=label,
                    label_normalized=self.normalizer.normalize(label),
                    value=None,
                    value_numeric=None,
                    confidence=page_confidence,
                    page_number=page_num,
                    source='ocr_extract',
                    parse_error=False,
                    metadata={'has_value': False}
                )
            return None
        
        # Extract label and value
        label = match.group(1).strip()
        value_str = match.group(2).strip()
        
        # Skip if label is too short or not meaningful
        if len(label) < 3 or not self._is_financial_label(label):
            return None
        
        # Parse number
        value_numeric = self.parser.parse(value_str)
        
        # Determine item confidence
        # If we successfully parsed the number, boost confidence
        item_confidence = page_confidence
        if value_numeric is not None:
            item_confidence = min(page_confidence + 0.05, 1.0)
        
        return ExtractedItem(
            label=label,
            label_normalized=self.normalizer.normalize(label),
            value=value_str,
            value_numeric=value_numeric,
            confidence=item_confidence,
            page_number=page_num,
            source='ocr_extract',
            parse_error=(value_numeric is None and self.parser.looks_like_number(value_str)),
            metadata={
                'has_value': True,
                'original_line': line
            }
        )
    
    def _is_financial_label(self, label: str) -> bool:
        """
        Check if label looks like a financial term
        
        Args:
            label: Text label
            
        Returns:
            True if looks like financial term
        """
        # Check if contains financial keywords
        financial_keywords = [
            'tài sản', 'tai san', 'nợ', 'no', 'vốn', 'von',
            'doanh thu', 'lợi nhuận', 'loi nhuan', 'chi phí', 'chi phi',
            'tiền', 'tien', 'hàng', 'hang', 'phải thu', 'phai thu',
            'phải trả', 'phai tra', 'đầu tư', 'dau tu', 'khoản', 'khoan',
            'assets', 'liabilities', 'equity', 'revenue', 'profit', 'cash'
        ]
        
        label_lower = label.lower()
        return any(keyword in label_lower for keyword in financial_keywords)
    
    def _classify_confidence(self, confidence: float) -> ConfidenceLevel:
        """Classify confidence level"""
        if confidence >= self.CONFIDENCE_HIGH:
            return ConfidenceLevel.HIGH
        elif confidence >= self.CONFIDENCE_MEDIUM:
            return ConfidenceLevel.MEDIUM
        elif confidence >= self.CONFIDENCE_LOW:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _should_use_vlm(self, avg_confidence: float, issues: List[Dict]) -> bool:
        """
        Decide if VLM supervision is needed
        
        Args:
            avg_confidence: Average confidence
            issues: List of detected issues
            
        Returns:
            True if VLM needed
        """
        # Trigger 1: Overall confidence below threshold
        if avg_confidence < self.confidence_threshold:
            return True
        
        # Trigger 2: Too many parse errors
        parse_errors = [i for i in issues if i['reason'] == 'parse_error']
        if len(parse_errors) > 5:
            return True
        
        # Trigger 3: Too many low confidence items
        low_conf_items = [i for i in issues if i['reason'] == 'low_confidence']
        if len(low_conf_items) > 10:
            return True
        
        return False
    
    def export_to_json(self, result: OCRExtractResult, output_path: str):
        """
        Export extraction result to JSON
        
        Args:
            result: OCRExtractResult
            output_path: Output file path
        """
        output = {
            'metadata': {
                **result.metadata,
                'total_items': result.total_items,
                'avg_confidence': round(result.avg_confidence, 3),
                'confidence_level': result.confidence_level.value,
                'needs_vlm': result.needs_vlm,
                'threshold_used': self.confidence_threshold
            },
            'items': [
                {
                    'label': item.label,
                    'label_normalized': item.label_normalized,
                    'value': item.value,
                    'value_numeric': item.value_numeric,
                    'confidence': round(item.confidence, 3),
                    'page_number': item.page_number,
                    'source': item.source,
                    'parse_error': item.parse_error
                }
                for item in result.items
            ],
            'issues': result.issues,
            'summary': {
                'total_items': result.total_items,
                'items_with_values': len([i for i in result.items if i.value_numeric is not None]),
                'items_with_errors': len([i for i in result.items if i.parse_error]),
                'avg_confidence': round(result.avg_confidence, 3),
                'needs_vlm_supervision': result.needs_vlm,
                'issue_count': len(result.issues)
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Exported to: {output_path}")


# ============================================================================
# MAIN / USAGE EXAMPLE
# ============================================================================

def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ocr_extract.py <ocr_scan_json_path>")
        print("Example: python ocr_extract.py ocr_results.json")
        sys.exit(1)
    
    input_json = sys.argv[1]
    output_json = input_json.replace('.json', '_extracted.json')
    
    print("="*70)
    print("OCR EXTRACT MODULE")
    print("="*70)
    
    # Initialize extractor
    extractor = OCRExtractor(confidence_threshold=0.85)
    
    # Extract
    result = extractor.extract_from_json(input_json)
    
    # Export
    extractor.export_to_json(result, output_json)
    
    # Print summary
    print("\n" + "="*70)
    print("EXTRACTION SUMMARY")
    print("="*70)
    print(f"Total items extracted:     {result.total_items}")
    print(f"Items with numeric values: {len([i for i in result.items if i.value_numeric])}")
    print(f"Items with parse errors:   {len([i for i in result.items if i.parse_error])}")
    print(f"Average confidence:        {result.avg_confidence:.3f}")
    print(f"Confidence level:          {result.confidence_level.value}")
    print(f"Needs VLM supervision:     {'YES ⚠️' if result.needs_vlm else 'NO ✓'}")
    print(f"Total issues detected:     {len(result.issues)}")
    print("="*70)
    
    # Print sample items
    print("\nSample extracted items (first 10):")
    for i, item in enumerate(result.items[:10], 1):
        status = "✓" if item.value_numeric is not None else "⚠️"
        print(f"{i:2d}. {status} [{item.confidence:.2f}] {item.label}")
        print(f"    → {item.label_normalized}: {item.value_numeric}")


if __name__ == "__main__":
    main()