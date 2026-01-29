"""
Module kiểm tra confidence và phân loại pages
- Load JSON results từ OCR Scanner
- Tính average confidence
- Phân loại pages: HIGH (≥0.9) → VLMs_detect, LOW (<0.9) → VLMs_supervisor
- Export danh sách pages cho từng module
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class ProcessingRoute(Enum):
    """Route xử lý dựa trên confidence"""
    VLMS_DETECT = "VLMs_detect"      # High confidence ≥ 0.9
    VLMS_SUPERVISOR = "VLMs_supervisor"  # Low confidence < 0.9


@dataclass
class PageConfidenceInfo:
    """Thông tin confidence của 1 page"""
    page_number: int
    avg_confidence: float
    confidence_level: str
    needs_review: bool
    queue_type: str
    priority: int
    char_count: int
    number_count: int
    total_blocks: int
    low_confidence_blocks: int
    route: ProcessingRoute


class ConfidenceChecker:
    """Module kiểm tra confidence và phân loại pages"""
    
    # Threshold để phân loại
    HIGH_CONFIDENCE_THRESHOLD = 0.9
    
    def __init__(
        self, 
        json_file: str,
        high_threshold: float = 0.9,
        export_lists: bool = True
    ):
        """
        Initialize Confidence Checker
        
        Args:
            json_file: Path tới JSON output từ OCR Scanner
            high_threshold: Ngưỡng confidence cao (default: 0.9)
            export_lists: Export danh sách pages ra file riêng
        """
        self.json_file = Path(json_file)
        self.high_threshold = high_threshold
        self.export_lists = export_lists
        
        if not self.json_file.exists():
            raise FileNotFoundError(f"JSON file not found: {self.json_file}")
        
        # Data
        self.ocr_data = None
        self.pages_info: List[PageConfidenceInfo] = []
        
        # Statistics
        self.stats = {
            "total_pages": 0,
            "high_confidence_pages": 0,  # ≥ 0.9 → VLMs_detect
            "low_confidence_pages": 0,   # < 0.9 → VLMs_supervisor
            "avg_confidence_all": 0.0,
            "vlms_detect_pages": [],
            "vlms_supervisor_pages": []
        }
        
        print(f" ConfidenceChecker initialized")
        print(f"  JSON file: {self.json_file}")
        print(f"  High confidence threshold: {self.high_threshold}")
    
    def load_ocr_results(self):
        """Load JSON results từ OCR Scanner"""
        print(f"\n Loading OCR results from: {self.json_file}")
        
        with open(self.json_file, 'r', encoding='utf-8') as f:
            self.ocr_data = json.load(f)
        
        total_pages = self.ocr_data.get('metadata', {}).get('total_pages', 0)
        print(f" Loaded {total_pages} pages")
        
        return self.ocr_data
    
    def analyze_pages(self):
        """Phân tích confidence của từng page"""
        print(f"\n Analyzing page confidence...")
        
        pages = self.ocr_data.get('pages', {})
        
        for page_key, page_data in pages.items():
            # Extract page number
            page_num = int(page_key.replace('page ', ''))
            
            # Get confidence info
            conf_data = page_data.get('confidence', {})
            avg_conf = conf_data.get('average', 0.0)
            conf_level = conf_data.get('level', 'UNKNOWN')
            needs_review = conf_data.get('needs_review', True)
            
            # Get other info
            stats = page_data.get('statistics', {})
            low_conf_blocks = page_data.get('low_confidence_blocks', [])
            
            # Determine route based on threshold
            if avg_conf >= self.high_threshold:
                route = ProcessingRoute.VLMS_DETECT
                self.stats['high_confidence_pages'] += 1
                self.stats['vlms_detect_pages'].append(page_num)
            else:
                route = ProcessingRoute.VLMS_SUPERVISOR
                self.stats['low_confidence_pages'] += 1
                self.stats['vlms_supervisor_pages'].append(page_num)
            
            # Create PageConfidenceInfo
            page_info = PageConfidenceInfo(
                page_number=page_num,
                avg_confidence=avg_conf,
                confidence_level=conf_level,
                needs_review=needs_review,
                queue_type=page_data.get('queue_type', 'UNKNOWN'),
                priority=page_data.get('priority', 0),
                char_count=stats.get('char_count', 0),
                number_count=stats.get('number_count', 0),
                total_blocks=stats.get('total_blocks', 0),
                low_confidence_blocks=len(low_conf_blocks),
                route=route
            )
            
            self.pages_info.append(page_info)
        
        # Sort by page number
        self.pages_info.sort(key=lambda x: x.page_number)
        
        # Calculate overall average
        if self.pages_info:
            total_conf = sum(p.avg_confidence for p in self.pages_info)
            self.stats['avg_confidence_all'] = total_conf / len(self.pages_info)
        
        self.stats['total_pages'] = len(self.pages_info)
        
        print(f"Analyzed {len(self.pages_info)} pages")
        print(f"   High confidence (≥{self.high_threshold}): {self.stats['high_confidence_pages']} pages → VLMs_detect")
        print(f"   Low confidence (<{self.high_threshold}): {self.stats['low_confidence_pages']} pages → VLMs_supervisor")
    
    def get_pages_by_route(self, route: ProcessingRoute) -> List[PageConfidenceInfo]:
        """Lấy danh sách pages theo route"""
        return [p for p in self.pages_info if p.route == route]
    
    def get_vlms_detect_pages(self) -> List[int]:
        """Lấy danh sách page numbers cho VLMs_detect (high confidence)"""
        return [p.page_number for p in self.pages_info if p.route == ProcessingRoute.VLMS_DETECT]
    
    def get_vlms_supervisor_pages(self) -> List[int]:
        """Lấy danh sách page numbers cho VLMs_supervisor (low confidence)"""
        return [p.page_number for p in self.pages_info if p.route == ProcessingRoute.VLMS_SUPERVISOR]
    
    def export_routing_lists(self, output_dir: str = "."):
        """Export danh sách pages cho từng module"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n Exporting routing lists to: {output_dir}")
        
        # 1. Export VLMs_detect list
        vlms_detect_file = output_dir / "vlms_detect_pages.json"
        vlms_detect_data = {
            "module": "VLMs_detect",
            "description": f"High confidence pages (≥{self.high_threshold})",
            "total_pages": self.stats['high_confidence_pages'],
            "pages": []
        }
        
        for page_info in self.get_pages_by_route(ProcessingRoute.VLMS_DETECT):
            vlms_detect_data['pages'].append({
                "page_number": page_info.page_number,
                "confidence": round(page_info.avg_confidence, 3),
                "confidence_level": page_info.confidence_level,
                "queue_type": page_info.queue_type,
                "priority": page_info.priority,
                "char_count": page_info.char_count,
                "number_count": page_info.number_count
            })
        
        with open(vlms_detect_file, 'w', encoding='utf-8') as f:
            json.dump(vlms_detect_data, f, ensure_ascii=False, indent=2)
        
        print(f"   VLMs_detect: {vlms_detect_file} ({self.stats['high_confidence_pages']} pages)")
        
        # 2. Export VLMs_supervisor list
        vlms_supervisor_file = output_dir / "vlms_supervisor_pages.json"
        vlms_supervisor_data = {
            "module": "VLMs_supervisor",
            "description": f"Low confidence pages (<{self.high_threshold}) - Needs review",
            "total_pages": self.stats['low_confidence_pages'],
            "pages": []
        }
        
        for page_info in self.get_pages_by_route(ProcessingRoute.VLMS_SUPERVISOR):
            vlms_supervisor_data['pages'].append({
                "page_number": page_info.page_number,
                "confidence": round(page_info.avg_confidence, 3),
                "confidence_level": page_info.confidence_level,
                "needs_review": page_info.needs_review,
                "queue_type": page_info.queue_type,
                "priority": page_info.priority,
                "low_confidence_blocks": page_info.low_confidence_blocks,
                "char_count": page_info.char_count,
                "number_count": page_info.number_count
            })
        
        with open(vlms_supervisor_file, 'w', encoding='utf-8') as f:
            json.dump(vlms_supervisor_data, f, ensure_ascii=False, indent=2)
        
        print(f"   VLMs_supervisor: {vlms_supervisor_file} ({self.stats['low_confidence_pages']} pages)")
        
        # 3. Export routing summary
        summary_file = output_dir / "routing_summary.json"
        summary_data = {
            "metadata": {
                "source_file": str(self.json_file),
                "high_confidence_threshold": self.high_threshold,
                "total_pages": self.stats['total_pages'],
                "avg_confidence_all": round(self.stats['avg_confidence_all'], 3)
            },
            "routing": {
                "vlms_detect": {
                    "count": self.stats['high_confidence_pages'],
                    "percentage": round(self.stats['high_confidence_pages'] / self.stats['total_pages'] * 100, 1),
                    "pages": self.stats['vlms_detect_pages']
                },
                "vlms_supervisor": {
                    "count": self.stats['low_confidence_pages'],
                    "percentage": round(self.stats['low_confidence_pages'] / self.stats['total_pages'] * 100, 1),
                    "pages": self.stats['vlms_supervisor_pages']
                }
            }
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        print(f"  Summary: {summary_file}")
    
    def print_detailed_report(self):
        """In báo cáo chi tiết"""
        print("\n" + "="*80)
        print("CONFIDENCE ANALYSIS REPORT")
        print("="*80)
        
        print(f"\n Overall Statistics:")
        print(f"  Total pages:           {self.stats['total_pages']}")
        print(f"  Average confidence:    {self.stats['avg_confidence_all']:.3f}")
        print(f"  Threshold:             {self.high_threshold}")
        
        print(f"\n Routing Distribution:")
        high_pct = (self.stats['high_confidence_pages'] / self.stats['total_pages'] * 100) if self.stats['total_pages'] > 0 else 0
        low_pct = (self.stats['low_confidence_pages'] / self.stats['total_pages'] * 100) if self.stats['total_pages'] > 0 else 0
        
        print(f"   VLMs_detect (≥{self.high_threshold}):      {self.stats['high_confidence_pages']:3d} pages ({high_pct:5.1f}%)")
        print(f"    VLMs_supervisor (<{self.high_threshold}):  {self.stats['low_confidence_pages']:3d} pages ({low_pct:5.1f}%)")
        
        print(f"\nPages by Route:")
        
        # VLMs_detect pages
        detect_pages = self.get_pages_by_route(ProcessingRoute.VLMS_DETECT)
        if detect_pages:
            print(f"\n  VLMs_detect ({len(detect_pages)} pages):")
            for page in detect_pages[:5]:  # Show first 5
                print(f"     Page {page.page_number:3d}: conf={page.avg_confidence:.3f} ({page.confidence_level})")
            if len(detect_pages) > 5:
                print(f"     ... and {len(detect_pages) - 5} more pages")
        
        # VLMs_supervisor pages
        supervisor_pages = self.get_pages_by_route(ProcessingRoute.VLMS_SUPERVISOR)
        if supervisor_pages:
            print(f"\n  VLMs_supervisor ({len(supervisor_pages)} pages) - NEEDS REVIEW:")
            for page in supervisor_pages:
                review_flag = "v"if page.needs_review else "x"
                print(f"     {review_flag} Page {page.page_number:3d}: conf={page.avg_confidence:.3f} ({page.confidence_level}) "
                      f"- {page.low_confidence_blocks} low blocks")
        
        print("="*80)
    
    def run(self, output_dir: str = "."):
        """Chạy toàn bộ quy trình check confidence"""
        print("\n" + "="*80)
        print("CONFIDENCE CHECKER MODULE")
        print("="*80)
        
        try:
            # Load OCR results
            self.load_ocr_results()
            
            # Analyze pages
            self.analyze_pages()
            
            # Export routing lists
            if self.export_lists:
                self.export_routing_lists(output_dir)
            
            # Print report
            self.print_detailed_report()
            
            print(f"\nConfidence checking completed!")
            
            return {
                "vlms_detect_pages": self.get_vlms_detect_pages(),
                "vlms_supervisor_pages": self.get_vlms_supervisor_pages(),
                "stats": self.stats
            }
            
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Check confidence and route pages to VLMs_detect or VLMs_supervisor'
    )
    parser.add_argument('json_file', help='JSON file from OCR Scanner')
    parser.add_argument('--threshold', type=float, default=0.9,
                       help='High confidence threshold (default: 0.9)')
    parser.add_argument('--output-dir', default='routing_output',
                       help='Output directory for routing lists')
    parser.add_argument('--no-export', action='store_true',
                       help='Do not export routing lists')
    
    args = parser.parse_args()
    
    checker = ConfidenceChecker(
        json_file=args.json_file,
        high_threshold=args.threshold,
        export_lists=not args.no_export
    )
    
    result = checker.run(output_dir=args.output_dir)
    
    if result:
        print(f"\nQuick Access:")
        print(f"  VLMs_detect pages:     {result['vlms_detect_pages']}")
        print(f"  VLMs_supervisor pages: {result['vlms_supervisor_pages']}")


if __name__ == "__main__":
    main()