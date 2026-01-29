"""
fusion_module.py
Module Fusion - Kết hợp kết quả từ OCR Extract và VLMs Extract
Tạo ra file JSON hoàn chỉnh + CSV với cấu trúc tài chính
"""

import json
import csv
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import re
from datetime import datetime


class FusionStrategy(Enum):
    """Chiến lược fusion"""
    VLM_PRIORITY = "VLM_PRIORITY"           # Ưu tiên VLM
    OCR_PRIORITY = "OCR_PRIORITY"           # Ưu tiên OCR
    CONFIDENCE_BASED = "CONFIDENCE_BASED"   # Dựa trên confidence
    HYBRID = "HYBRID"                       # Kết hợp thông minh


@dataclass
class FusionConfig:
    """Cấu hình cho Fusion"""
    strategy: FusionStrategy = FusionStrategy.HYBRID
    ocr_confidence_threshold: float = 0.85
    prefer_vlm_for_numbers: bool = True      # Ưu tiên VLM cho số liệu
    prefer_ocr_for_structure: bool = True    # Ưu tiên OCR cho cấu trúc
    verify_consistency: bool = True          # Kiểm tra tính nhất quán
    verbose: bool = True
    export_csv: bool = True                  # Export CSV
    csv_include_metadata: bool = True        # Include metadata trong CSV


@dataclass
class FusionMetrics:
    """Metrics về quá trình fusion"""
    total_sections: int = 0
    ocr_sections: int = 0
    vlm_sections: int = 0
    merged_sections: int = 0
    conflicts_resolved: int = 0
    confidence_improvements: int = 0
    value_corrections: int = 0


@dataclass
class SectionMatch:
    """Kết quả matching giữa OCR và VLM section"""
    ocr_section: Optional[Dict] = None
    vlm_section: Optional[Dict] = None
    match_score: float = 0.0
    has_conflict: bool = False
    conflict_details: List[str] = field(default_factory=list)


class FusionModule:
    """
    Module Fusion - Kết hợp OCR và VLMs results
    Tạo output JSON hoàn chỉnh với độ chính xác cao nhất
    """
    
    def __init__(self, config: FusionConfig = None):
        """
        Khởi tạo Fusion Module
        
        Args:
            config: Cấu hình fusion
        """
        self.config = config or FusionConfig()
        self.metrics = FusionMetrics()
        
        print("✓ Fusion Module initialized")
        print(f"  Strategy: {self.config.strategy.value}")
        print(f"  OCR confidence threshold: {self.config.ocr_confidence_threshold}")
    
    def load_ocr_result(self, ocr_path: str) -> Dict[str, Any]:
        """Load OCR result từ file JSON"""
        with open(ocr_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_vlm_result(self, vlm_path: str) -> Dict[str, Any]:
        """Load VLM result từ file JSON"""
        with open(vlm_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def normalize_text(self, text: str) -> str:
        """Chuẩn hóa text để so sánh"""
        if not text:
            return ""
        # Lowercase, remove extra spaces
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        # Remove special chars for comparison
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Tính độ tương đồng giữa 2 text
        
        Returns:
            Score từ 0.0 đến 1.0
        """
        norm1 = self.normalize_text(text1)
        norm2 = self.normalize_text(text2)
        
        if not norm1 or not norm2:
            return 0.0
        
        if norm1 == norm2:
            return 1.0
        
        # Simple word-based similarity
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def match_sections(self, 
                      ocr_sections: List[Dict], 
                      vlm_sections: List[Dict]) -> List[SectionMatch]:
        """
        Match các sections giữa OCR và VLM
        
        Args:
            ocr_sections: Sections từ OCR
            vlm_sections: Sections từ VLM
            
        Returns:
            List of SectionMatch
        """
        matches = []
        used_vlm = set()
        
        for ocr_sec in ocr_sections:
            best_match = None
            best_score = 0.0
            best_vlm_idx = -1
            
            ocr_name = ocr_sec.get('section', '')
            ocr_code = ocr_sec.get('code', '')
            
            # Tìm VLM section tương ứng
            for i, vlm_sec in enumerate(vlm_sections):
                if i in used_vlm:
                    continue
                
                vlm_name = vlm_sec.get('section', '')
                vlm_code = vlm_sec.get('code', '')
                
                # Match by code first
                if ocr_code and vlm_code and ocr_code == vlm_code:
                    score = 1.0
                else:
                    # Match by name similarity
                    score = self.calculate_text_similarity(ocr_name, vlm_name)
                
                if score > best_score:
                    best_score = score
                    best_match = vlm_sec
                    best_vlm_idx = i
            
            # Create match
            if best_score > 0.5:  # Threshold cho match
                used_vlm.add(best_vlm_idx)
                match = SectionMatch(
                    ocr_section=ocr_sec,
                    vlm_section=best_match,
                    match_score=best_score
                )
            else:
                # OCR section không có match
                match = SectionMatch(
                    ocr_section=ocr_sec,
                    vlm_section=None,
                    match_score=0.0
                )
            
            matches.append(match)
        
        # Thêm VLM sections không được match
        for i, vlm_sec in enumerate(vlm_sections):
            if i not in used_vlm:
                match = SectionMatch(
                    ocr_section=None,
                    vlm_section=vlm_sec,
                    match_score=0.0
                )
                matches.append(match)
        
        return matches
    
    def merge_items(self, 
                   ocr_items: List[Dict], 
                   vlm_items: List[Dict],
                   ocr_confidence: float) -> List[Dict]:
        """
        Merge items từ OCR và VLM
        
        Args:
            ocr_items: Items từ OCR
            vlm_items: Items từ VLM
            ocr_confidence: OCR confidence
            
        Returns:
            Merged items
        """
        merged = []
        used_vlm = set()
        
        for ocr_item in ocr_items:
            ocr_code = ocr_item.get('code', '')
            ocr_name = ocr_item.get('name', '')
            
            # Tìm VLM item tương ứng
            best_vlm = None
            best_vlm_idx = -1
            best_score = 0.0
            
            for i, vlm_item in enumerate(vlm_items):
                if i in used_vlm:
                    continue
                
                vlm_code = vlm_item.get('code', '')
                vlm_name = vlm_item.get('name', '')
                
                # Match by code
                if ocr_code and vlm_code and ocr_code == vlm_code:
                    score = 1.0
                else:
                    # Match by name
                    score = self.calculate_text_similarity(ocr_name, vlm_name)
                
                if score > best_score:
                    best_score = score
                    best_vlm = vlm_item
                    best_vlm_idx = i
            
            # Merge item
            if best_score > 0.6:
                used_vlm.add(best_vlm_idx)
                merged_item = self.merge_single_item(ocr_item, best_vlm, ocr_confidence)
            else:
                # Chỉ có OCR
                merged_item = self.select_item_by_strategy(ocr_item, None, ocr_confidence)
            
            merged.append(merged_item)
        
        # Thêm VLM items không match
        for i, vlm_item in enumerate(vlm_items):
            if i not in used_vlm:
                merged_item = self.select_item_by_strategy(None, vlm_item, 0.0)
                merged.append(merged_item)
        
        return merged
    
    def merge_single_item(self, 
                         ocr_item: Dict, 
                         vlm_item: Dict,
                         ocr_confidence: float) -> Dict:
        """
        Merge một item từ OCR và VLM
        
        Args:
            ocr_item: Item từ OCR
            vlm_item: Item từ VLM
            ocr_confidence: OCR confidence
            
        Returns:
            Merged item
        """
        merged = {}
        
        # Code - ưu tiên VLM nếu có
        merged['code'] = vlm_item.get('code') or ocr_item.get('code', '')
        
        # Name - ưu tiên VLM nếu OCR confidence thấp
        if ocr_confidence >= self.config.ocr_confidence_threshold:
            merged['name'] = ocr_item.get('name', '')
            if vlm_item.get('name') and vlm_item['name'] != ocr_item.get('name'):
                merged['name_alternative'] = vlm_item['name']
        else:
            merged['name'] = vlm_item.get('name', '')
        
        # Values - merge thông minh
        ocr_values = ocr_item.get('values', {})
        vlm_values = vlm_item.get('values', {})
        
        merged['values'] = self.merge_values(ocr_values, vlm_values, ocr_confidence)
        
        # Metadata
        merged['sources'] = {
            'ocr': True,
            'vlm': True,
            'primary_source': 'vlm' if self.config.prefer_vlm_for_numbers else 'ocr'
        }
        
        return merged
    
    def merge_values(self, 
                    ocr_values: Dict, 
                    vlm_values: Dict,
                    ocr_confidence: float) -> Dict:
        """
        Merge values từ OCR và VLM
        
        Args:
            ocr_values: Values từ OCR
            vlm_values: Values từ VLM
            ocr_confidence: OCR confidence
            
        Returns:
            Merged values dict
        """
        merged = {}
        
        # Get all dates
        all_dates = set(ocr_values.keys()).union(set(vlm_values.keys()))
        
        for date in all_dates:
            ocr_val = ocr_values.get(date)
            vlm_val = vlm_values.get(date)
            
            if ocr_val is not None and vlm_val is not None:
                # Có cả 2 - chọn theo strategy
                if self.config.strategy == FusionStrategy.VLM_PRIORITY:
                    merged[date] = vlm_val
                elif self.config.strategy == FusionStrategy.OCR_PRIORITY:
                    merged[date] = ocr_val
                elif self.config.strategy == FusionStrategy.CONFIDENCE_BASED:
                    if ocr_confidence >= self.config.ocr_confidence_threshold:
                        merged[date] = ocr_val
                    else:
                        merged[date] = vlm_val
                else:  # HYBRID
                    # Ưu tiên VLM cho số liệu nếu được config
                    if self.config.prefer_vlm_for_numbers:
                        merged[date] = vlm_val
                        # Check nếu khác nhau quá nhiều
                        if abs(vlm_val - ocr_val) / max(abs(vlm_val), abs(ocr_val), 1) > 0.1:
                            self.metrics.value_corrections += 1
                    else:
                        merged[date] = ocr_val
            
            elif ocr_val is not None:
                merged[date] = ocr_val
            elif vlm_val is not None:
                merged[date] = vlm_val
        
        return merged
    
    def select_item_by_strategy(self, 
                                ocr_item: Optional[Dict], 
                                vlm_item: Optional[Dict],
                                ocr_confidence: float) -> Dict:
        """
        Chọn item theo strategy khi chỉ có một source
        
        Args:
            ocr_item: Item từ OCR (có thể None)
            vlm_item: Item từ VLM (có thể None)
            ocr_confidence: OCR confidence
            
        Returns:
            Selected item
        """
        if ocr_item and not vlm_item:
            item = ocr_item.copy()
            item['sources'] = {'ocr': True, 'vlm': False, 'primary_source': 'ocr'}
            return item
        
        if vlm_item and not ocr_item:
            item = vlm_item.copy()
            item['sources'] = {'ocr': False, 'vlm': True, 'primary_source': 'vlm'}
            return item
        
        # Should not reach here
        return {}
    
    def merge_section(self, match: SectionMatch, ocr_confidence: float) -> Dict:
        """
        Merge một section từ match
        
        Args:
            match: SectionMatch object
            ocr_confidence: OCR confidence
            
        Returns:
            Merged section
        """
        if match.ocr_section and match.vlm_section:
            # Có cả 2
            merged = {
                'section': match.vlm_section.get('section') or match.ocr_section.get('section', ''),
                'code': match.vlm_section.get('code') or match.ocr_section.get('code', ''),
                'items': self.merge_items(
                    match.ocr_section.get('items', []),
                    match.vlm_section.get('items', []),
                    ocr_confidence
                ),
                'match_score': round(match.match_score, 3),
                'sources': {
                    'ocr': True,
                    'vlm': True
                }
            }
            self.metrics.merged_sections += 1
        
        elif match.ocr_section:
            # Chỉ có OCR
            merged = match.ocr_section.copy()
            merged['sources'] = {'ocr': True, 'vlm': False}
            self.metrics.ocr_sections += 1
        
        else:
            # Chỉ có VLM
            merged = match.vlm_section.copy()
            merged['sources'] = {'ocr': False, 'vlm': True}
            self.metrics.vlm_sections += 1
        
        return merged
    
    def fuse_page(self,
                  ocr_data: Dict,
                  vlm_data: Dict,
                  page_number: int) -> Dict:
        """
        Fuse data cho một trang
        
        Args:
            ocr_data: OCR result cho trang
            vlm_data: VLM result cho trang
            page_number: Số trang
            
        Returns:
            Fused data
        """
        if self.config.verbose:
            print(f"\n🔄 Fusing page {page_number}...")
        
        # Extract sections
        ocr_sections = ocr_data.get('sections', [])
        vlm_sections = vlm_data.get('data', {}).get('sections', [])
        
        if self.config.verbose:
            print(f"  OCR sections: {len(ocr_sections)}")
            print(f"  VLM sections: {len(vlm_sections)}")
        
        # Get OCR confidence
        ocr_confidence = ocr_data.get('metadata', {}).get('avg_confidence', 0.5)
        if 'confidence' in ocr_data:
            ocr_confidence = ocr_data['confidence'].get('average', 0.5)
        
        # Match sections
        matches = self.match_sections(ocr_sections, vlm_sections)
        
        if self.config.verbose:
            print(f"  Matched: {len(matches)} sections")
        
        # Merge sections
        fused_sections = []
        for match in matches:
            merged_section = self.merge_section(match, ocr_confidence)
            fused_sections.append(merged_section)
        
        self.metrics.total_sections += len(fused_sections)
        
        # Create result
        result = {
            'page_number': page_number,
            'sections': fused_sections,
            'fusion_metadata': {
                'strategy': self.config.strategy.value,
                'ocr_confidence': round(ocr_confidence, 3),
                'total_sections': len(fused_sections),
                'ocr_only_sections': sum(1 for s in fused_sections 
                                        if s.get('sources', {}).get('ocr') 
                                        and not s.get('sources', {}).get('vlm')),
                'vlm_only_sections': sum(1 for s in fused_sections 
                                        if s.get('sources', {}).get('vlm') 
                                        and not s.get('sources', {}).get('ocr')),
                'merged_sections': sum(1 for s in fused_sections 
                                      if s.get('sources', {}).get('ocr') 
                                      and s.get('sources', {}).get('vlm'))
            }
        }
        
        if self.config.verbose:
            print(f"  ✓ Fused: {len(fused_sections)} sections")
        
        return result
    
    def fuse_folder(self,
                   ocr_folder: str,
                   vlm_folder: str,
                   output_path: str,
                   ocr_summary_path: str = None,
                   vlm_summary_path: str = None) -> Dict:
        """
        Fuse toàn bộ folder results
        
        Args:
            ocr_folder: Folder chứa OCR results
            vlm_folder: Folder chứa VLM results
            output_path: Path để lưu fused result
            ocr_summary_path: Path đến OCR summary (optional)
            vlm_summary_path: Path đến VLM summary (optional)
            
        Returns:
            Fused result dict
        """
        print("\n" + "="*70)
        print("FUSION MODULE - MERGING OCR + VLM RESULTS")
        print("="*70)
        
        ocr_path = Path(ocr_folder)
        vlm_path = Path(vlm_folder)
        
        # Find all OCR files
        ocr_files = sorted(ocr_path.glob('page_*_ocr.json'))
        if not ocr_files:
            ocr_files = sorted(ocr_path.glob('page_*.json'))
        
        print(f"\n📁 OCR folder: {ocr_folder}")
        print(f"   Found {len(ocr_files)} OCR files")
        
        print(f"\n📁 VLM folder: {vlm_folder}")
        vlm_files = sorted(vlm_path.glob('page_*_vlm.json'))
        print(f"   Found {len(vlm_files)} VLM files")
        
        # Create mapping
        fused_pages = {}
        
        for ocr_file in ocr_files:
            # Extract page number
            match = re.search(r'page_(\d+)', ocr_file.stem)
            if not match:
                continue
            
            page_num = int(match.group(1))
            
            # Find corresponding VLM file
            vlm_candidates = [
                vlm_path / f'page_{page_num}_vlm.json',
                vlm_path / f'page_{page_num}.json',
            ]
            
            vlm_file = None
            for candidate in vlm_candidates:
                if candidate.exists():
                    vlm_file = candidate
                    break
            
            if not vlm_file:
                print(f"\n⚠️  Page {page_num}: No VLM result found, using OCR only")
                ocr_data = self.load_ocr_result(str(ocr_file))
                fused_pages[f'page_{page_num}'] = {
                    'page_number': page_num,
                    'sections': ocr_data.get('sections', []),
                    'sources': {'ocr': True, 'vlm': False}
                }
                continue
            
            # Load both
            ocr_data = self.load_ocr_result(str(ocr_file))
            vlm_data = self.load_vlm_result(str(vlm_file))
            
            # Fuse
            fused = self.fuse_page(ocr_data, vlm_data, page_num)
            fused_pages[f'page_{page_num}'] = fused
        
        # Create final result
        final_result = {
            'metadata': {
                'fusion_strategy': self.config.strategy.value,
                'total_pages': len(fused_pages),
                'ocr_confidence_threshold': self.config.ocr_confidence_threshold,
                'prefer_vlm_for_numbers': self.config.prefer_vlm_for_numbers,
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'pages': fused_pages,
            'summary': {
                'total_sections': self.metrics.total_sections,
                'ocr_only_sections': self.metrics.ocr_sections,
                'vlm_only_sections': self.metrics.vlm_sections,
                'merged_sections': self.metrics.merged_sections,
                'value_corrections': self.metrics.value_corrections
            }
        }
        
        # Save result
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Saved fused result to: {output_path}")
        
        # Print summary
        self.print_summary(final_result)
        
        # Export CSV if configured
        if self.config.export_csv:
            base_path = Path(output_path).with_suffix('')
            csv_outputs = self.export_all_formats(final_result, str(base_path))
        
        return final_result
    
    def print_summary(self, result: Dict):
        """In summary của fusion"""
        print("\n" + "="*70)
        print("FUSION SUMMARY")
        print("="*70)
        
        summary = result.get('summary', {})
        metadata = result.get('metadata', {})
        
        print(f"Strategy:        {metadata.get('fusion_strategy')}")
        print(f"Total Pages:     {metadata.get('total_pages')}")
        print(f"\nSections:")
        print(f"  Total:         {summary.get('total_sections')}")
        print(f"  OCR only:      {summary.get('ocr_only_sections')}")
        print(f"  VLM only:      {summary.get('vlm_only_sections')}")
        print(f"  Merged:        {summary.get('merged_sections')}")
        print(f"\nCorrections:")
        print(f"  Values:        {summary.get('value_corrections')}")
        print("="*70)
    
    def extract_company_info(self, fused_data: Dict) -> Dict[str, str]:
        """
        Trích xuất thông tin cơ bản công ty từ fused data
        
        Args:
            fused_data: Dữ liệu đã fusion
            
        Returns:
            Dict chứa thông tin công ty
        """
        company_info = {
            'Tên công ty': '',
            'Mã chứng khoán': '',
            'Địa chỉ': '',
            'Ngành nghề': '',
            'Năm tài chính': ''
        }
        
        # Tìm thông tin từ các pages
        pages = fused_data.get('pages', {})
        
        for page_key, page_data in pages.items():
            sections = page_data.get('sections', [])
            
            for section in sections:
                section_name = section.get('section', '').upper()
                items = section.get('items', [])
                
                # Tìm các pattern
                for item in items:
                    name = item.get('name', '').upper()
                    
                    if 'TÊN CÔNG TY' in name or 'DOANH NGHIỆP' in name:
                        company_info['Tên công ty'] = item.get('name', '')
                    elif 'MÃ CK' in name or 'TICKER' in name:
                        company_info['Mã chứng khoán'] = item.get('name', '')
        
        return company_info
    
    def extract_financial_categories(self, fused_data: Dict) -> Dict[str, List[Dict]]:
        """
        Phân loại dữ liệu tài chính thành các category
        
        Args:
            fused_data: Dữ liệu đã fusion
            
        Returns:
            Dict với các category: assets, liabilities, equity
        """
        categories = {
            'assets_short_term': [],      # Tài sản ngắn hạn
            'assets_long_term': [],       # Tài sản dài hạn
            'liabilities_short_term': [], # Nợ ngắn hạn
            'liabilities_long_term': [],  # Nợ dài hạn
            'equity': [],                 # Vốn chủ sở hữu
            'revenue': [],                # Doanh thu
            'expenses': [],               # Chi phí
            'other': []                   # Khác
        }
        
        # Keywords để phân loại
        keywords_map = {
            'assets_short_term': ['TÀI SẢN NGẮN HẠN', 'TÀI SẢN LƯU ĐỘNG', 'TIỀN', 'PHẢI THU NGẮN HẠN'],
            'assets_long_term': ['TÀI SẢN DÀI HẠN', 'TÀI SẢN CỐ ĐỊNH', 'BẤT ĐỘNG SẢN'],
            'liabilities_short_term': ['NỢ NGẮN HẠN', 'PHẢI TRẢ NGẮN HẠN', 'NỢ LƯU ĐỘNG'],
            'liabilities_long_term': ['NỢ DÀI HẠN', 'PHẢI TRẢ DÀI HẠN'],
            'equity': ['VỐN CHỦ SỞ HỮU', 'NGUỒN VỐN', 'VỐN GÓP'],
            'revenue': ['DOANH THU', 'BÁN HÀNG'],
            'expenses': ['CHI PHÍ', 'GIÁ VỐN']
        }
        
        pages = fused_data.get('pages', {})
        
        for page_key, page_data in pages.items():
            sections = page_data.get('sections', [])
            
            for section in sections:
                section_name = section.get('section', '').upper()
                items = section.get('items', [])
                
                # Phân loại section
                category = 'other'
                for cat, keywords in keywords_map.items():
                    if any(kw in section_name for kw in keywords):
                        category = cat
                        break
                
                # Thêm items vào category
                for item in items:
                    item_with_section = item.copy()
                    item_with_section['section'] = section.get('section', '')
                    item_with_section['section_code'] = section.get('code', '')
                    categories[category].append(item_with_section)
        
        return categories
    
    def export_to_csv(self, 
                     fused_data: Dict, 
                     output_csv_path: str,
                     company_info: Dict = None) -> str:
        """
        Export dữ liệu ra CSV với cấu trúc tài chính
        
        Args:
            fused_data: Dữ liệu đã fusion
            output_csv_path: Đường dẫn file CSV output
            company_info: Thông tin công ty (optional)
            
        Returns:
            Path to CSV file
        """
        if self.config.verbose:
            print(f"\n📊 Exporting to CSV: {output_csv_path}")
        
        # Extract company info nếu chưa có
        if not company_info:
            company_info = self.extract_company_info(fused_data)
        
        # Extract categories
        categories = self.extract_financial_categories(fused_data)
        
        # Collect all dates
        all_dates = set()
        pages = fused_data.get('pages', {})
        for page_data in pages.values():
            for section in page_data.get('sections', []):
                for item in section.get('items', []):
                    all_dates.update(item.get('values', {}).keys())
        
        dates = sorted(list(all_dates))
        
        # Create CSV
        csv_path = Path(output_csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            
            # Header - Company Info
            if self.config.csv_include_metadata:
                writer.writerow(['THÔNG TIN CÔNG TY'])
                writer.writerow([])
                for key, value in company_info.items():
                    writer.writerow([key, value])
                writer.writerow([])
                writer.writerow([])
            
            # Header - Column names
            header = ['Phân loại', 'Mã mục', 'Tên mục', 'Section'] + dates
            writer.writerow(header)
            
            # Write data by category
            category_names = {
                'assets_short_term': 'TÀI SẢN NGẮN HẠN',
                'assets_long_term': 'TÀI SẢN DÀI HẠN',
                'liabilities_short_term': 'NỢ PHẢI TRẢ NGẮN HẠN',
                'liabilities_long_term': 'NỢ PHẢI TRẢ DÀI HẠN',
                'equity': 'VỐN CHỦ SỞ HỮU',
                'revenue': 'DOANH THU',
                'expenses': 'CHI PHÍ',
                'other': 'KHÁC'
            }
            
            for category, category_label in category_names.items():
                items = categories.get(category, [])
                
                if not items:
                    continue
                
                # Category header
                writer.writerow([])
                writer.writerow([category_label])
                writer.writerow([])
                
                # Write items
                for item in items:
                    row = [
                        category_label,
                        item.get('code', ''),
                        item.get('name', ''),
                        item.get('section', '')
                    ]
                    
                    # Add values for each date
                    values = item.get('values', {})
                    for date in dates:
                        value = values.get(date, '')
                        # Format number
                        if isinstance(value, (int, float)):
                            row.append(f"{value:,.0f}")
                        else:
                            row.append(value)
                    
                    writer.writerow(row)
            
            # Summary section
            if self.config.csv_include_metadata:
                writer.writerow([])
                writer.writerow([])
                writer.writerow(['TỔNG KẾT'])
                writer.writerow([])
                
                summary = fused_data.get('summary', {})
                writer.writerow(['Tổng số sections', summary.get('total_sections', 0)])
                writer.writerow(['Merged sections', summary.get('merged_sections', 0)])
                writer.writerow(['Value corrections', summary.get('value_corrections', 0)])
        
        if self.config.verbose:
            print(f"  ✓ CSV exported successfully")
            print(f"  Categories: {len([c for c in categories.values() if c])}")
            print(f"  Date columns: {len(dates)}")
        
        return str(csv_path)
    
    def export_balance_sheet_csv(self,
                                 fused_data: Dict,
                                 output_csv_path: str) -> str:
        """
        Export Balance Sheet format (Bảng cân đối kế toán)
        
        Args:
            fused_data: Dữ liệu đã fusion
            output_csv_path: Path to output CSV
            
        Returns:
            Path to CSV file
        """
        if self.config.verbose:
            print(f"\n📊 Exporting Balance Sheet CSV: {output_csv_path}")
        
        # Extract categories
        categories = self.extract_financial_categories(fused_data)
        
        # Get all dates
        all_dates = set()
        pages = fused_data.get('pages', {})
        for page_data in pages.values():
            for section in page_data.get('sections', []):
                for item in section.get('items', []):
                    all_dates.update(item.get('values', {}).keys())
        
        dates = sorted(list(all_dates))
        
        # Create CSV
        csv_path = Path(output_csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            
            # Title
            writer.writerow(['BẢNG CÂN ĐỐI KẾ TOÁN'])
            writer.writerow([])
            
            # Header
            header = ['TÀI SẢN', 'Mã số'] + dates + ['', 'NGUỒN VỐN', 'Mã số'] + dates
            writer.writerow(header)
            writer.writerow([])
            
            # Assets (left side)
            all_assets = categories['assets_short_term'] + categories['assets_long_term']
            
            # Liabilities + Equity (right side)
            all_liabilities = categories['liabilities_short_term'] + categories['liabilities_long_term']
            all_equity = categories['equity']
            sources = all_liabilities + all_equity
            
            # Write rows
            max_rows = max(len(all_assets), len(sources))
            
            for i in range(max_rows):
                row = []
                
                # Left side - Assets
                if i < len(all_assets):
                    asset = all_assets[i]
                    row.append(asset.get('name', ''))
                    row.append(asset.get('code', ''))
                    
                    values = asset.get('values', {})
                    for date in dates:
                        value = values.get(date, '')
                        if isinstance(value, (int, float)):
                            row.append(f"{value:,.0f}")
                        else:
                            row.append(value)
                else:
                    row.extend(['', ''] + [''] * len(dates))
                
                row.append('')  # Separator
                
                # Right side - Sources
                if i < len(sources):
                    source = sources[i]
                    row.append(source.get('name', ''))
                    row.append(source.get('code', ''))
                    
                    values = source.get('values', {})
                    for date in dates:
                        value = values.get(date, '')
                        if isinstance(value, (int, float)):
                            row.append(f"{value:,.0f}")
                        else:
                            row.append(value)
                else:
                    row.extend(['', ''] + [''] * len(dates))
                
                writer.writerow(row)
        
        if self.config.verbose:
            print(f"  ✓ Balance Sheet CSV exported")
        
        return str(csv_path)
    
    def export_all_formats(self,
                          fused_data: Dict,
                          base_output_path: str) -> Dict[str, str]:
        """
        Export tất cả formats: JSON + CSV + Balance Sheet
        
        Args:
            fused_data: Dữ liệu đã fusion
            base_output_path: Base path (without extension)
            
        Returns:
            Dict with paths to all generated files
        """
        base_path = Path(base_output_path)
        base_dir = base_path.parent
        base_name = base_path.stem
        
        outputs = {}
        
        # JSON (already saved in fuse_folder)
        outputs['json'] = str(base_path.with_suffix('.json'))
        
        # CSV - Detailed format
        if self.config.export_csv:
            csv_detailed_path = base_dir / f"{base_name}_detailed.csv"
            outputs['csv_detailed'] = self.export_to_csv(fused_data, str(csv_detailed_path))
            
            # CSV - Balance Sheet format
            csv_balance_path = base_dir / f"{base_name}_balance_sheet.csv"
            outputs['csv_balance_sheet'] = self.export_balance_sheet_csv(
                fused_data, str(csv_balance_path)
            )
        
        if self.config.verbose:
            print(f"\n📦 All formats exported:")
            for format_name, path in outputs.items():
                print(f"  • {format_name}: {path}")
        
        return outputs


# Ví dụ sử dụng
if __name__ == "__main__":
    print("="*70)
    print("FUSION MODULE - EXAMPLES WITH CSV EXPORT")
    print("="*70)
    
    # ===== CÁCH 1: Fusion với CSV export =====
    print("\n=== CÁCH 1: Fusion with CSV Export ===")
    
    config_with_csv = FusionConfig(
        strategy=FusionStrategy.HYBRID,
        prefer_vlm_for_numbers=True,
        export_csv=True,              # Enable CSV export
        csv_include_metadata=True,
        verbose=True
    )
    
    fusion1 = FusionModule(config_with_csv)
    
    result1 = fusion1.fuse_folder(
        ocr_folder='./ocr_results',
        vlm_folder='./vlm_results',
        output_path='./output/fused_with_csv.json'
    )
    # Sẽ tự động tạo:
    # - fused_with_csv.json
    # - fused_with_csv_detailed.csv
    # - fused_with_csv_balance_sheet.csv
    
    
    # ===== CÁCH 2: Export CSV manually từ JSON có sẵn =====
    print("\n\n=== CÁCH 2: Manual CSV Export ===")
    
    config2 = FusionConfig(strategy=FusionStrategy.HYBRID)
    fusion2 = FusionModule(config2)
    
    # Load JSON đã có
    with open('./output/fused_with_csv.json', 'r', encoding='utf-8') as f:
        existing_data = json.load(f)
    
    # Export to CSV
    fusion2.export_to_csv(
        existing_data,
        './output/custom_detailed.csv'
    )
    
    fusion2.export_balance_sheet_csv(
        existing_data,
        './output/custom_balance_sheet.csv'
    )
    
    
    # ===== CÁCH 3: Export tất cả formats =====
    print("\n\n=== CÁCH 3: Export All Formats ===")
    
    outputs = fusion2.export_all_formats(
        existing_data,
        './output/complete_report'
    )
    
    print("\n✅ Generated files:")
    for format_type, path in outputs.items():
        print(f"  • {format_type}: {path}")
    
    
    # ===== CÁCH 4: Extract company info =====
    print("\n\n=== CÁCH 4: Extract Company Info ===")
    
    company_info = fusion2.extract_company_info(existing_data)
    print("\nThông tin công ty:")
    for key, value in company_info.items():
        print(f"  {key}: {value}")
    
    
    # ===== CÁCH 5: Extract by categories =====
    print("\n\n=== CÁCH 5: Financial Categories ===")
    
    categories = fusion2.extract_financial_categories(existing_data)
    print("\nPhân loại tài chính:")
    for category, items in categories.items():
        if items:
            print(f"  • {category}: {len(items)} items")
    
    print("\n✅ Fusion with CSV export completed!")
    print("="*70)