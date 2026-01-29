"""
Main Pipeline Module
====================

Module chính cho pipeline xử lý tài liệu tài chính:
- OCR Pipeline: Trích xuất văn bản từ ảnh
- VLMs Pipeline: Sử dụng Vision Language Models
- Confidence Checker: Kiểm tra độ tin cậy và phân loại
- Fusion: Kết hợp kết quả OCR và VLM

Architecture:
-------------
    Input Images
         │
         ├─→ OCR Pipeline ─→ OCR Results (JSON)
         │                        │
         └─→ Confidence Checker ─→ Classification
                                   │
                ┌──────────────────┴───────────────┐
                │                                  │
         HIGH Confidence                    LOW Confidence
         (≥0.9)                             (<0.9)
                │                                  │
         VLMs Detect Mode              VLMs Supervisor Mode
                │                                  │
                └──────────────┬───────────────────┘
                               │
                        Fusion Module
                               │
                      Final Results (JSON + CSV)

Components:
-----------
1. OCR_pipeline/
   - ocr_scan: OCR scanning với PaddleOCR
   - ocr_extract: Trích xuất và xử lý văn bản
   - post_process: Hậu xử lý text
   - postprocess_financial: Xử lý tài liệu tài chính

2. VLMs_pipeline/
   - vlm_extract: Trích xuất bằng Vision Language Models
   - Support: Claude (Anthropic), GPT-4V (OpenAI)

3. confidence_checker.py
   - Phân tích confidence score
   - Phân loại pages: HIGH/LOW confidence
   - Route pages tới VLMs_detect hoặc VLMs_supervisor

4. fusion.py
   - Kết hợp OCR + VLM results
   - Giải quyết conflicts
   - Export JSON + CSV

Usage Examples:
---------------
    # Full pipeline
    >>> from main_pipeline import run_full_pipeline
    >>> result = run_full_pipeline(
    ...     images_dir="output/step3_scanned/",
    ...     ocr_json="output/ocr_results.json",
    ...     output_dir="final_results/"
    ... )
    
    # Step by step
    >>> from main_pipeline import OCRExtractor, ConfidenceChecker, VLMsExtractor, FusionModule
    
    >>> # Step 1: OCR
    >>> ocr = OCRExtractor()
    >>> ocr_results = ocr.process_directory("images/")
    
    >>> # Step 2: Check confidence
    >>> checker = ConfidenceChecker("ocr_results.json")
    >>> classification = checker.process()
    
    >>> # Step 3: VLM processing
    >>> vlm = VLMsExtractor(provider="anthropic", api_key="...")
    >>> vlm_results = vlm.process_pages(classification['high_confidence_pages'])
    
    >>> # Step 4: Fusion
    >>> fusion = FusionModule()
    >>> final = fusion.fuse_results("ocr_results.json", "vlm_results.json")
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

__version__ = "1.0.0"
__author__ = "Your Team"

# ===========================
# PATH CONFIGURATION
# ===========================
_MODULE_DIR = Path(__file__).parent
_PARENT_DIR = _MODULE_DIR.parent

if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

# ===========================
# IMPORT OCR PIPELINE
# ===========================
from .OCR_pipeline import (
    # OCR Extraction
    OCRExtractor,
    OptimizedOCRExtractor,
    extract_text,
    
    # OCR Scanning
    OCRScanner,
    VietnameseOCRService,
    scan_document,
    
    # Post Processing
    PostProcessor,
    clean_text,
    post_process_ocr,
    
    # Financial Processing
    FinancialProcessor,
    process_financial_doc,
)

# ===========================
# IMPORT VLMS PIPELINE
# ===========================
from .VLMs_pipeline import (
    VLMsExtractor,
    VLMConfig,
    extract_with_vlm,
)

# ===========================
# IMPORT CONFIDENCE CHECKER
# ===========================
from .confidence_checker import (
    ConfidenceChecker,
    PageConfidenceInfo,
    ProcessingRoute,
    check_confidence,
    classify_pages,
)

# ===========================
# IMPORT FUSION MODULE
# ===========================
from .fusion import (
    FusionModule,
    FusionConfig,
    FusionStrategy,
    FusionMetrics,
    fusion_results,
    merge_ocr_vlm,
)


# ===========================
# HIGH-LEVEL PIPELINE FUNCTIONS
# ===========================

def run_full_pipeline(
    images_dir: str,
    output_dir: str = "pipeline_output",
    ocr_config: Optional[Dict] = None,
    vlm_config: Optional[Dict] = None,
    fusion_config: Optional[Dict] = None,
    confidence_threshold: float = 0.9,
    export_csv: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Chạy full pipeline: OCR → Confidence Check → VLM → Fusion
    
    Args:
        images_dir: Thư mục chứa ảnh đã được preprocess
        output_dir: Thư mục output chính
        ocr_config: Cấu hình OCR (dict)
        vlm_config: Cấu hình VLM - bắt buộc có 'provider' và 'api_key'
        fusion_config: Cấu hình Fusion (dict)
        confidence_threshold: Ngưỡng phân loại HIGH/LOW (default: 0.9)
        export_csv: Export CSV cùng với JSON
        verbose: In log chi tiết
    
    Returns:
        dict: {
            'ocr_results': OCR results path,
            'confidence_report': Confidence analysis path,
            'vlm_results': VLM results path,
            'final_results': Fused results path,
            'csv_output': CSV path (if export_csv=True),
            'statistics': Pipeline statistics
        }
    
    Example:
        >>> result = run_full_pipeline(
        ...     images_dir="preprocessed_images/",
        ...     output_dir="final_output/",
        ...     vlm_config={'provider': 'anthropic', 'api_key': 'sk-...'},
        ...     confidence_threshold=0.9
        ... )
        >>> print(f"Final results: {result['final_results']}")
    """
    import json
    from datetime import datetime
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("MAIN PIPELINE - FINANCIAL DOCUMENT PROCESSING")
    print("="*70)
    print(f"Input: {images_dir}")
    print(f"Output: {output_dir}")
    print(f"Confidence threshold: {confidence_threshold}")
    print("="*70)
    
    results = {
        'start_time': datetime.now().isoformat(),
        'statistics': {}
    }
    
    # ========================================
    # STEP 1: OCR Processing
    # ========================================
    print("\n[STEP 1/4] OCR Processing...")
    ocr_output = output_path / "ocr_results.json"
    
    ocr_cfg = ocr_config or {}
    ocr = OptimizedOCRExtractor(**ocr_cfg)
    
    ocr_results = ocr.process_directory(
        images_dir,
        output_json=str(ocr_output),
        verbose=verbose
    )
    
    results['ocr_results'] = str(ocr_output)
    results['statistics']['total_pages'] = len(ocr_results.get('pages', {}))
    print(f"✅ OCR complete: {ocr_output}")
    
    # ========================================
    # STEP 2: Confidence Analysis
    # ========================================
    print("\n[STEP 2/4] Confidence Analysis...")
    confidence_output = output_path / "confidence_report.json"
    
    checker = ConfidenceChecker(
        json_file=str(ocr_output),
        high_threshold=confidence_threshold,
        export_lists=True
    )
    
    classification = checker.process(output_file=str(confidence_output))
    
    results['confidence_report'] = str(confidence_output)
    results['statistics']['high_confidence'] = len(classification['vlms_detect_pages'])
    results['statistics']['low_confidence'] = len(classification['vlms_supervisor_pages'])
    print(f"✅ Confidence analysis complete")
    print(f"   HIGH confidence: {len(classification['vlms_detect_pages'])} pages")
    print(f"   LOW confidence: {len(classification['vlms_supervisor_pages'])} pages")
    
    # ========================================
    # STEP 3: VLM Processing
    # ========================================
    print("\n[STEP 3/4] VLM Processing...")
    vlm_output = output_path / "vlm_results.json"
    
    if not vlm_config or 'provider' not in vlm_config or 'api_key' not in vlm_config:
        print("⚠️  VLM config missing. Skipping VLM processing.")
        print("   Please provide: {'provider': 'anthropic', 'api_key': 'sk-...'}")
        results['vlm_results'] = None
    else:
        vlm_cfg = VLMConfig(**vlm_config)
        vlm = VLMsExtractor(config=vlm_cfg)
        
        # Process all pages (both HIGH and LOW confidence)
        all_pages = (classification['vlms_detect_pages'] + 
                    classification['vlms_supervisor_pages'])
        
        vlm_results = vlm.process_pages(
            pages=all_pages,
            output_json=str(vlm_output),
            verbose=verbose
        )
        
        results['vlm_results'] = str(vlm_output)
        print(f"✅ VLM processing complete: {vlm_output}")
    
    # ========================================
    # STEP 4: Fusion
    # ========================================
    print("\n[STEP 4/4] Fusing Results...")
    fusion_output = output_path / "final_results.json"
    csv_output = output_path / "final_results.csv" if export_csv else None
    
    fusion_cfg = FusionConfig(**(fusion_config or {}))
    fusion_cfg.export_csv = export_csv
    
    fusion = FusionModule(config=fusion_cfg)
    
    if results.get('vlm_results'):
        final_results = fusion.fuse_results(
            ocr_path=str(ocr_output),
            vlm_path=str(vlm_output),
            output_json=str(fusion_output),
            output_csv=str(csv_output) if csv_output else None
        )
        
        results['final_results'] = str(fusion_output)
        if export_csv:
            results['csv_output'] = str(csv_output)
        
        results['statistics']['fusion_metrics'] = fusion.metrics.__dict__
        print(f"✅ Fusion complete: {fusion_output}")
        if export_csv:
            print(f"✅ CSV exported: {csv_output}")
    else:
        # No VLM results, use OCR only
        print("⚠️  Using OCR results only (no VLM fusion)")
        results['final_results'] = str(ocr_output)
    
    # ========================================
    # SUMMARY
    # ========================================
    results['end_time'] = datetime.now().isoformat()
    
    summary_file = output_path / "pipeline_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"Total pages processed: {results['statistics']['total_pages']}")
    print(f"Final output: {results['final_results']}")
    print(f"Summary: {summary_file}")
    print("="*70)
    
    return results


def run_ocr_only(
    images_dir: str,
    output_file: str = "ocr_results.json",
    **ocr_kwargs
) -> Dict[str, Any]:
    """
    Chạy OCR processing only
    
    Args:
        images_dir: Thư mục chứa ảnh
        output_file: File output JSON
        **ocr_kwargs: Arguments cho OCRExtractor
    
    Returns:
        dict: OCR results
        
    Example:
        >>> results = run_ocr_only("images/", "output.json")
    """
    ocr = OptimizedOCRExtractor(**ocr_kwargs)
    return ocr.process_directory(images_dir, output_json=output_file)


def run_confidence_check(
    ocr_json: str,
    threshold: float = 0.9,
    output_file: str = "confidence_report.json"
) -> Dict[str, Any]:
    """
    Chạy confidence analysis only
    
    Args:
        ocr_json: Path to OCR results JSON
        threshold: Confidence threshold
        output_file: Output report file
    
    Returns:
        dict: Classification results
        
    Example:
        >>> classification = run_confidence_check("ocr.json", threshold=0.85)
    """
    checker = ConfidenceChecker(ocr_json, high_threshold=threshold)
    return checker.process(output_file=output_file)


def run_vlm_only(
    pages: List[Dict],
    provider: str,
    api_key: str,
    output_file: str = "vlm_results.json",
    **vlm_kwargs
) -> Dict[str, Any]:
    """
    Chạy VLM processing only
    
    Args:
        pages: List of page info dicts
        provider: 'anthropic' or 'openai'
        api_key: API key
        output_file: Output JSON file
        **vlm_kwargs: Additional VLM config
    
    Returns:
        dict: VLM results
        
    Example:
        >>> pages = [{'page_number': 1, 'image_path': 'img1.png'}]
        >>> results = run_vlm_only(pages, 'anthropic', 'sk-...')
    """
    config = VLMConfig(provider=provider, api_key=api_key, **vlm_kwargs)
    vlm = VLMsExtractor(config=config)
    return vlm.process_pages(pages, output_json=output_file)


# ===========================
# UTILITY FUNCTIONS
# ===========================

def get_pipeline_status(output_dir: str) -> Dict[str, Any]:
    """
    Kiểm tra trạng thái pipeline output
    
    Args:
        output_dir: Pipeline output directory
    
    Returns:
        dict: Status information
    """
    output_path = Path(output_dir)
    
    status = {
        'exists': output_path.exists(),
        'ocr_completed': (output_path / "ocr_results.json").exists(),
        'confidence_completed': (output_path / "confidence_report.json").exists(),
        'vlm_completed': (output_path / "vlm_results.json").exists(),
        'fusion_completed': (output_path / "final_results.json").exists(),
        'csv_exported': (output_path / "final_results.csv").exists(),
    }
    
    if (output_path / "pipeline_summary.json").exists():
        import json
        with open(output_path / "pipeline_summary.json", 'r') as f:
            status['summary'] = json.load(f)
    
    return status


def load_pipeline_results(output_dir: str) -> Dict[str, Any]:
    """
    Load all pipeline results
    
    Args:
        output_dir: Pipeline output directory
    
    Returns:
        dict: All results
    """
    import json
    output_path = Path(output_dir)
    
    results = {}
    
    files = {
        'ocr': 'ocr_results.json',
        'confidence': 'confidence_report.json',
        'vlm': 'vlm_results.json',
        'final': 'final_results.json',
        'summary': 'pipeline_summary.json'
    }
    
    for key, filename in files.items():
        filepath = output_path / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                results[key] = json.load(f)
    
    return results


def get_module_info():
    """Print module information"""
    print(f"""
{'='*70}
Main Pipeline Module v{__version__}
{'='*70}

Pipeline Flow:
  Input Images → OCR → Confidence Check → VLM → Fusion → Final Results

Components:
  • OCR_pipeline: Text extraction with PaddleOCR
  • VLMs_pipeline: Vision Language Models (Claude, GPT-4V)
  • Confidence Checker: Quality analysis and routing
  • Fusion: Combine and optimize results

Quick Start:
  from main_pipeline import run_full_pipeline
  
  result = run_full_pipeline(
      images_dir="preprocessed/",
      output_dir="final/",
      vlm_config={{'provider': 'anthropic', 'api_key': 'sk-...'}}
  )

Functions:
  • run_full_pipeline(): Complete pipeline
  • run_ocr_only(): OCR processing only
  • run_confidence_check(): Confidence analysis only
  • run_vlm_only(): VLM processing only
  • get_pipeline_status(): Check pipeline status
  • load_pipeline_results(): Load all results

{'='*70}
For detailed help: help(run_full_pipeline)
{'='*70}
""")


# ===========================
# EXPORTS
# ===========================

__all__ = [
    # Version
    '__version__',
    '__author__',
    
    # OCR Pipeline
    'OCRExtractor',
    'OptimizedOCRExtractor',
    'OCRScanner',
    'VietnameseOCRService',
    'PostProcessor',
    'FinancialProcessor',
    'extract_text',
    'scan_document',
    'clean_text',
    'post_process_ocr',
    'process_financial_doc',
    
    # VLMs Pipeline
    'VLMsExtractor',
    'VLMConfig',
    'extract_with_vlm',
    
    # Confidence Checker
    'ConfidenceChecker',
    'PageConfidenceInfo',
    'ProcessingRoute',
    'check_confidence',
    'classify_pages',
    
    # Fusion Module
    'FusionModule',
    'FusionConfig',
    'FusionStrategy',
    'FusionMetrics',
    'fusion_results',
    'merge_ocr_vlm',
    
    # High-level Pipeline
    'run_full_pipeline',
    'run_ocr_only',
    'run_confidence_check',
    'run_vlm_only',
    
    # Utilities
    'get_pipeline_status',
    'load_pipeline_results',
    'get_module_info',
]