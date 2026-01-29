"""
OCR Pipeline Module
===================

Module xử lý OCR (Optical Character Recognition) cho tài liệu tài chính tiếng Việt.
Sử dụng PaddleOCR với các tối ưu hóa cho tiếng Việt và tài liệu tài chính.

Components:
-----------
1. ocr_scan.py
   - VietnameseOCRService: OCR service tối ưu cho tiếng Việt
   - OCRScanner: Scanner với table detection và rotation
   - Thread-safe, batch processing support

2. ocr_extract.py
   - OCRExtractor: Basic OCR extraction
   - OptimizedOCRExtractor: Optimized với cache và batch processing
   - Support nhiều định dạng ảnh

3. post_process.py
   - PostProcessor: Text cleaning và normalization
   - Vietnamese text processing
   - Remove noise, fix common errors

4. postprocess_financial.py
   - FinancialProcessor: Xử lý tài liệu tài chính
   - Extract sections, codes, values
   - Structure financial data

Features:
---------
- ✅ Vietnamese language optimized
- ✅ High accuracy for financial documents
- ✅ Table detection and structure preservation
- ✅ Batch processing with multiprocessing
- ✅ Confidence scoring
- ✅ Cache support for performance
- ✅ GPU acceleration support
- ✅ Thread-safe operations

Usage Examples:
---------------
    # Quick OCR scanning
    >>> from OCR_pipeline import scan_document
    >>> results = scan_document("document.png")
    
    # Optimized extraction with cache
    >>> from OCR_pipeline import OptimizedOCRExtractor
    >>> ocr = OptimizedOCRExtractor(use_scanner_service=True, enable_cache=True)
    >>> text = ocr.extract_from_image("image.png")
    
    # Process directory
    >>> results = ocr.process_directory("images/", output_json="results.json")
    
    # Financial document processing
    >>> from OCR_pipeline import FinancialProcessor
    >>> processor = FinancialProcessor()
    >>> structured = processor.process_document("financial_ocr_results.json")
    
    # Post-processing
    >>> from OCR_pipeline import clean_text
    >>> cleaned = clean_text(raw_ocr_text)
"""

from pathlib import Path
import sys

__version__ = "1.0.0"

# ===========================
# IMPORT OCR SCANNING
# ===========================
try:
    from .ocr_scan import (
        VietnameseOCRService,
        OCRScanner,
        scan_document,
        scan_with_table_detection,
        process_image,
    )
    OCR_SCAN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  OCR scan module import warning: {e}")
    OCR_SCAN_AVAILABLE = False
    VietnameseOCRService = None
    OCRScanner = None
    scan_document = None
    scan_with_table_detection = None
    process_image = None

# ===========================
# IMPORT OCR EXTRACTION
# ===========================
try:
    from .ocr_extract import (
        OCRExtractor,
        OptimizedOCRExtractor,
        extract_text,
        extract_from_image,
        extract_from_directory,
    )
    OCR_EXTRACT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  OCR extract module import warning: {e}")
    OCR_EXTRACT_AVAILABLE = False
    OCRExtractor = None
    OptimizedOCRExtractor = None
    extract_text = None
    extract_from_image = None
    extract_from_directory = None

# ===========================
# IMPORT POST PROCESSING
# ===========================
try:
    from .post_process import (
        PostProcessor,
        clean_text,
        post_process_ocr,
        normalize_vietnamese,
        remove_noise,
        fix_common_errors,
    )
    POST_PROCESS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Post process module import warning: {e}")
    POST_PROCESS_AVAILABLE = False
    PostProcessor = None
    clean_text = None
    post_process_ocr = None
    normalize_vietnamese = None
    remove_noise = None
    fix_common_errors = None

# ===========================
# IMPORT FINANCIAL PROCESSING
# ===========================
try:
    from .postprocess_financial import (
        FinancialProcessor,
        process_financial_doc,
        extract_financial_sections,
        parse_financial_values,
        validate_financial_structure,
    )
    FINANCIAL_PROCESS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Financial process module import warning: {e}")
    FINANCIAL_PROCESS_AVAILABLE = False
    FinancialProcessor = None
    process_financial_doc = None
    extract_financial_sections = None
    parse_financial_values = None
    validate_financial_structure = None


# ===========================
# CONVENIENCE FUNCTIONS
# ===========================

def quick_ocr(image_path: str, language: str = 'vi', use_gpu: bool = False) -> dict:
    """
    Quick OCR cho một ảnh
    
    Args:
        image_path: Đường dẫn ảnh
        language: Ngôn ngữ ('vi' hoặc 'en')
        use_gpu: Sử dụng GPU
    
    Returns:
        dict: OCR results với text và confidence
        
    Example:
        >>> result = quick_ocr("document.png")
        >>> print(result['text'])
    """
    if not OCR_EXTRACT_AVAILABLE:
        raise ImportError("OCR extract module not available")
    
    ocr = OptimizedOCRExtractor(
        lang=language,
        use_gpu=use_gpu,
        use_scanner_service=True
    )
    return ocr.extract_from_image(image_path)


def batch_ocr(
    images_dir: str,
    output_json: str = None,
    language: str = 'vi',
    num_workers: int = None,
    verbose: bool = True
) -> dict:
    """
    Batch OCR cho nhiều ảnh trong thư mục
    
    Args:
        images_dir: Thư mục chứa ảnh
        output_json: File JSON output (optional)
        language: Ngôn ngữ
        num_workers: Số workers (None = auto)
        verbose: In log
    
    Returns:
        dict: Kết quả OCR cho tất cả ảnh
        
    Example:
        >>> results = batch_ocr("images/", "output.json")
        >>> print(f"Processed {len(results['pages'])} pages")
    """
    if not OCR_EXTRACT_AVAILABLE:
        raise ImportError("OCR extract module not available")
    
    ocr = OptimizedOCRExtractor(
        lang=language,
        use_scanner_service=True,
        enable_cache=True
    )
    
    return ocr.process_directory(
        images_dir,
        output_json=output_json,
        num_workers=num_workers,
        verbose=verbose
    )


def ocr_with_cleanup(
    image_path: str,
    apply_financial_rules: bool = False
) -> dict:
    """
    OCR + auto cleanup
    
    Args:
        image_path: Đường dẫn ảnh
        apply_financial_rules: Áp dụng rules cho tài liệu tài chính
    
    Returns:
        dict: Cleaned OCR results
        
    Example:
        >>> result = ocr_with_cleanup("financial_doc.png", apply_financial_rules=True)
    """
    if not OCR_EXTRACT_AVAILABLE or not POST_PROCESS_AVAILABLE:
        raise ImportError("Required modules not available")
    
    # OCR
    ocr = OptimizedOCRExtractor(use_scanner_service=True)
    raw_result = ocr.extract_from_image(image_path)
    
    # Cleanup
    if clean_text:
        cleaned_text = clean_text(raw_result.get('text', ''))
        raw_result['cleaned_text'] = cleaned_text
    
    # Financial processing
    if apply_financial_rules and FINANCIAL_PROCESS_AVAILABLE:
        processor = FinancialProcessor()
        financial_data = processor.extract_from_text(cleaned_text)
        raw_result['financial_data'] = financial_data
    
    return raw_result


def get_ocr_service(
    language: str = 'vi',
    use_gpu: bool = False,
    enable_cache: bool = True,
    use_optimized: bool = True
):
    """
    Get OCR service instance
    
    Args:
        language: Ngôn ngữ
        use_gpu: Sử dụng GPU
        enable_cache: Enable cache
        use_optimized: Sử dụng OptimizedOCRExtractor
    
    Returns:
        OCR service instance
        
    Example:
        >>> ocr = get_ocr_service(use_gpu=True)
        >>> result = ocr.extract_from_image("image.png")
    """
    if not OCR_EXTRACT_AVAILABLE:
        raise ImportError("OCR extract module not available")
    
    if use_optimized:
        return OptimizedOCRExtractor(
            lang=language,
            use_gpu=use_gpu,
            enable_cache=enable_cache,
            use_scanner_service=True
        )
    else:
        return OCRExtractor(
            lang=language,
            use_gpu=use_gpu
        )


# ===========================
# VALIDATION
# ===========================

def validate_ocr_result(result: dict) -> dict:
    """
    Validate OCR result structure
    
    Args:
        result: OCR result dict
    
    Returns:
        dict: Validation report
    """
    validation = {
        'valid': True,
        'issues': [],
        'warnings': []
    }
    
    # Check required fields
    if 'text' not in result:
        validation['valid'] = False
        validation['issues'].append("Missing 'text' field")
    
    if 'confidence' not in result:
        validation['warnings'].append("Missing 'confidence' field")
    
    # Check confidence value
    if 'confidence' in result:
        conf = result['confidence']
        if not isinstance(conf, (int, float)):
            validation['issues'].append("Confidence must be numeric")
        elif conf < 0 or conf > 1:
            validation['issues'].append("Confidence must be between 0 and 1")
        elif conf < 0.5:
            validation['warnings'].append(f"Low confidence: {conf:.2f}")
    
    # Check text content
    if 'text' in result and not result['text'].strip():
        validation['warnings'].append("Empty text result")
    
    return validation


# ===========================
# MODULE INFO
# ===========================

def get_module_info():
    """Print OCR Pipeline module information"""
    
    status = {
        'OCR Scan': '✓' if OCR_SCAN_AVAILABLE else '✗',
        'OCR Extract': '✓' if OCR_EXTRACT_AVAILABLE else '✗',
        'Post Process': '✓' if POST_PROCESS_AVAILABLE else '✗',
        'Financial Process': '✓' if FINANCIAL_PROCESS_AVAILABLE else '✗',
    }
    
    print(f"""
{'='*70}
OCR Pipeline Module v{__version__}
{'='*70}

Module Status:
  OCR Scan:           {status['OCR Scan']}
  OCR Extract:        {status['OCR Extract']}
  Post Process:       {status['Post Process']}
  Financial Process:  {status['Financial Process']}

Features:
  • Vietnamese language optimized
  • Table detection and structure preservation
  • Batch processing with multiprocessing
  • GPU acceleration support
  • Confidence scoring
  • Financial document processing

Quick Start:
  from OCR_pipeline import quick_ocr, batch_ocr
  
  # Single image
  result = quick_ocr("document.png")
  
  # Multiple images
  results = batch_ocr("images_dir/", "output.json")

Classes:
  • OptimizedOCRExtractor: Fast OCR with cache
  • VietnameseOCRService: Vietnamese optimized
  • FinancialProcessor: Financial document processor
  • PostProcessor: Text cleanup

Functions:
  • quick_ocr(): Quick single image OCR
  • batch_ocr(): Batch process directory
  • ocr_with_cleanup(): OCR + auto cleanup
  • clean_text(): Text post-processing
  • process_financial_doc(): Financial processing

{'='*70}
For detailed help: help(OptimizedOCRExtractor)
{'='*70}
""")


# ===========================
# EXPORTS
# ===========================

__all__ = [
    # Version
    '__version__',
    
    # OCR Scanning
    'VietnameseOCRService',
    'OCRScanner',
    'scan_document',
    'scan_with_table_detection',
    'process_image',
    
    # OCR Extraction
    'OCRExtractor',
    'OptimizedOCRExtractor',
    'extract_text',
    'extract_from_image',
    'extract_from_directory',
    
    # Post Processing
    'PostProcessor',
    'clean_text',
    'post_process_ocr',
    'normalize_vietnamese',
    'remove_noise',
    'fix_common_errors',
    
    # Financial Processing
    'FinancialProcessor',
    'process_financial_doc',
    'extract_financial_sections',
    'parse_financial_values',
    'validate_financial_structure',
    
    # Convenience Functions
    'quick_ocr',
    'batch_ocr',
    'ocr_with_cleanup',
    'get_ocr_service',
    'validate_ocr_result',
    
    # Module Info
    'get_module_info',
    
    # Availability Flags
    'OCR_SCAN_AVAILABLE',
    'OCR_EXTRACT_AVAILABLE',
    'POST_PROCESS_AVAILABLE',
    'FINANCIAL_PROCESS_AVAILABLE',
]