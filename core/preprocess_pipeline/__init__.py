"""
Preprocessing Pipeline Module
==============================

Module tiền xử lý tài liệu OCR với đầy đủ chức năng:

Components:
-----------
1. pdf2img.py: Chuyển đổi PDF sang ảnh với metadata rotation
   - process_pdf(): Convert PDF -> images + manifest.json
   - get_pdf_rotations(): Lấy rotation metadata từ PDF

2. preprocess.py: Xử lý rotation và chuẩn hóa ảnh
   - preprocess(): Fix rotation từ manifest -> manifest_fixed.json
   - rotate_img(): Rotate ảnh theo góc

3. scan.py: OCR scanning với table detection (multiprocessing)
   - scan_multiprocessing(): Parallel processing với PaddleOCR
   - scan_sequential(): Sequential mode cho debugging
   - detect_table(): Phát hiện bảng biểu
   - decide_rotation_improved(): Quyết định rotation thông minh

4. pipeline_preprocess.py: Full pipeline orchestrator
   - run_pipeline(): Chạy toàn bộ pipeline tự động
   Steps: PDF -> Images -> Fix Rotation -> OCR Scan

Workflow chuẩn:
---------------
    Step 1: pdf2img.process_pdf() -> temp/step1_pdf2img/
    Step 2: preprocess.preprocess() -> temp/step2_preprocess/
    Step 3: scan.scan_multiprocessing() -> temp/step3_scan/

Usage Examples:
---------------
    # Quick start - Full pipeline
    >>> from preprocess_pipeline import PDFPreprocessor
    >>> preprocessor = PDFPreprocessor(dpi=300, num_workers=4)
    >>> result = preprocessor.process("document.pdf", "output/")
    
    # Or use convenience function
    >>> from preprocess_pipeline import full_pipeline
    >>> result = full_pipeline("doc.pdf", "output/", dpi=300)
    
    # Step by step processing
    >>> from preprocess_pipeline import convert_pdf_to_images, fix_rotation, scan_document
    >>> manifest1 = convert_pdf_to_images("doc.pdf", "step1/")
    >>> manifest2 = fix_rotation(str(manifest1), "step2/")
    >>> manifest3 = scan_document(str(manifest2), "step3/")
    
    # Quick pipeline with temp directory
    >>> from preprocess_pipeline import QuickPipeline
    >>> result = QuickPipeline.run("document.pdf")
"""

import sys
import json
from pathlib import Path


# ===========================
# PATH CONFIGURATION
# ===========================
# Get the directory of this __init__.py file
_MODULE_DIR = Path(__file__).parent

# Add parent directory to Python path if needed (for imports from sibling packages)
_PARENT_DIR = _MODULE_DIR.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))


# ===========================
# IMPORT PDF2IMG MODULE
# ===========================
from .pdf2img import (
    process_pdf,
    get_pdf_rotations,
    DPI,
    POPPLER_PATH
)

# ===========================
# IMPORT PREPROCESS MODULE
# ===========================
from .preprocess import (
    preprocess,
    rotate_img
)

# ===========================
# IMPORT SCAN MODULE
# ===========================
from .scan import (
    scan_multiprocessing,
    scan_sequential,
    process_page,
    detect_table,
    score_orientation_improved,
    decide_rotation_improved,
    get_ocr
)

# ===========================
# IMPORT PIPELINE MODULE
# ===========================
from .pipeline_preprocess import (
    run_pipeline,
    TEMP_DIR,
    STEP1_DIR,
    STEP2_DIR,
    STEP3_DIR,
    PDF2IMG_SCRIPT,
    PREPROCESS_SCRIPT,
    SCAN_SCRIPT
)


# ===========================
# CONVENIENCE FUNCTIONS
# ===========================

def convert_pdf_to_images(pdf_path, output_dir, dpi=200, poppler_path=None):
    """
    Convert PDF thành ảnh với DPI tùy chỉnh
    
    Args:
        pdf_path (str): Đường dẫn file PDF
        output_dir (str): Thư mục output
        dpi (int): DPI cho ảnh (default: 200)
        poppler_path (str): Đường dẫn Poppler trên Windows (optional)
    
    Returns:
        Path: Đường dẫn manifest.json
    
    Example:
        >>> manifest_path = convert_pdf_to_images("doc.pdf", "output/")
        >>> print(f"Manifest: {manifest_path}")
        
        >>> # Với custom DPI
        >>> manifest_path = convert_pdf_to_images("doc.pdf", "output/", dpi=300)
    """
    # Import module để có thể modify config
    from . import pdf2img as pdf2img_module
    
    # Backup original config
    original_dpi = pdf2img_module.DPI
    original_poppler = pdf2img_module.POPPLER_PATH
    
    # Set custom config
    pdf2img_module.DPI = dpi
    if poppler_path:
        pdf2img_module.POPPLER_PATH = poppler_path
    
    try:
        # Process PDF
        process_pdf(pdf_path, output_dir)
    finally:
        # Restore original config
        pdf2img_module.DPI = original_dpi
        pdf2img_module.POPPLER_PATH = original_poppler
    
    return Path(output_dir) / "manifest.json"


def fix_rotation(manifest_path, output_dir):
    """
    Fix rotation của ảnh dựa vào manifest
    
    Args:
        manifest_path (str): Đường dẫn manifest.json
        output_dir (str): Thư mục output
    
    Returns:
        Path: Đường dẫn manifest_fixed.json
    
    Example:
        >>> fixed = fix_rotation("step1/manifest.json", "step2/")
        >>> print(f"Fixed manifest: {fixed}")
    """
    preprocess(manifest_path, output_dir)
    return Path(output_dir) / "manifest_fixed.json"


def scan_document(manifest_path, output_dir, num_workers=None, verbose=False):
    """
    Scan document với OCR + table detection (multiprocessing)
    
    Args:
        manifest_path (str): Đường dẫn manifest_fixed.json
        output_dir (str): Thư mục output
        num_workers (int): Số worker processes (None = auto)
        verbose (bool): Print detailed logs
    
    Returns:
        Path: Đường dẫn manifest_scanned.json
    
    Example:
        >>> scanned = scan_document("step2/manifest_fixed.json", "step3/")
        
        >>> # Với custom workers
        >>> scanned = scan_document("manifest.json", "output/", num_workers=4, verbose=True)
    """
    scan_multiprocessing(manifest_path, output_dir, 
                        num_workers=num_workers, verbose=verbose)
    return Path(output_dir) / "manifest_scanned.json"


def full_pipeline(pdf_path, output_dir="output", dpi=200, num_workers=None, 
                  poppler_path=None, verbose=False):
    """
    Chạy full pipeline: PDF -> Images -> Fix Rotation -> OCR Scan
    
    Args:
        pdf_path (str): File PDF cần xử lý
        output_dir (str): Thư mục base output (default: "output")
        dpi (int): DPI cho ảnh (default: 200)
        num_workers (int): Số worker cho scan (None = auto detect)
        poppler_path (str): Đường dẫn Poppler (Windows only, optional)
        verbose (bool): Print detailed logs (default: False)
    
    Returns:
        dict: {
            'step1_dir': Images directory,
            'step2_dir': Fixed images directory,
            'step3_dir': Scanned results directory,
            'manifest_original': manifest.json path,
            'manifest_fixed': manifest_fixed.json path,
            'manifest_scanned': manifest_scanned.json path,
            'manifest_data': final manifest data (dict)
        }
    
    Example:
        >>> result = full_pipeline("document.pdf", "output/", dpi=300, num_workers=4)
        >>> print(f"Final manifest: {result['manifest_scanned']}")
        >>> print(f"Total pages: {len(result['manifest_data']['pages'])}")
        
        >>> # Simple usage
        >>> result = full_pipeline("doc.pdf")
    """
    base_dir = Path(output_dir)
    step1_dir = base_dir / "step1_images"
    step2_dir = base_dir / "step2_fixed"
    step3_dir = base_dir / "step3_scanned"
    
    print("\n" + "="*70)
    print("FULL PREPROCESSING PIPELINE")
    print("="*70)
    print(f"Input PDF: {pdf_path}")
    print(f"Output base: {base_dir}")
    print(f"DPI: {dpi}")
    print(f"Workers: {num_workers if num_workers else 'auto'}")
    print("="*70)
    
    # Step 1: PDF -> Images
    print("\n[STEP 1/3] Converting PDF to images...")
    manifest1 = convert_pdf_to_images(pdf_path, str(step1_dir), dpi, poppler_path)
    print(f"✅ Step 1 complete: {manifest1}")
    
    # Step 2: Fix rotation
    print("\n[STEP 2/3] Fixing rotation...")
    manifest2 = fix_rotation(str(manifest1), str(step2_dir))
    print(f"✅ Step 2 complete: {manifest2}")
    
    # Step 3: OCR scan
    print("\n[STEP 3/3] Scanning with OCR...")
    manifest3 = scan_document(str(manifest2), str(step3_dir), num_workers, verbose)
    print(f"✅ Step 3 complete: {manifest3}")
    
    # Load final manifest
    with open(manifest3, 'r', encoding='utf-8') as f:
        final_manifest = json.load(f)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"Total pages: {len(final_manifest['pages'])}")
    print(f"Final output: {step3_dir}")
    print(f"Final manifest: {manifest3}")
    print("="*70)
    
    return {
        'step1_dir': step1_dir,
        'step2_dir': step2_dir,
        'step3_dir': step3_dir,
        'manifest_original': manifest1,
        'manifest_fixed': manifest2,
        'manifest_scanned': manifest3,
        'manifest_data': final_manifest
    }


def load_manifest(manifest_path):
    """
    Load manifest JSON file
    
    Args:
        manifest_path (str): Path to manifest JSON
    
    Returns:
        dict: Manifest data
        
    Example:
        >>> data = load_manifest("output/manifest.json")
        >>> print(f"Pages: {len(data['pages'])}")
    """
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_page_images(manifest_path):
    """
    Get list of image paths from manifest
    
    Args:
        manifest_path (str): Path to manifest JSON
    
    Returns:
        list: List of image file paths
        
    Example:
        >>> images = get_page_images("output/manifest.json")
        >>> for img in images:
        ...     print(img)
    """
    manifest = load_manifest(manifest_path)
    return [page['file'] for page in manifest['pages']]


# ===========================
# UNIFIED PREPROCESSOR CLASS
# ===========================

class PDFPreprocessor:
    """
    Unified class cho toàn bộ preprocessing workflow
    Hỗ trợ từng bước riêng lẻ hoặc full pipeline
    
    Attributes:
        dpi (int): DPI cho output images
        num_workers (int): Số workers cho scanning
        poppler_path (str): Đường dẫn Poppler (Windows)
        verbose (bool): Enable detailed logging
    
    Example:
        >>> # Full pipeline
        >>> preprocessor = PDFPreprocessor(dpi=300, num_workers=4)
        >>> result = preprocessor.process("document.pdf", "output/")
        >>> print(f"Done! Output: {result['step3_dir']}")
        
        >>> # Step by step
        >>> preprocessor = PDFPreprocessor()
        >>> m1 = preprocessor.pdf_to_images("doc.pdf", "step1/")
        >>> m2 = preprocessor.fix_rotation(str(m1), "step2/")
        >>> m3 = preprocessor.scan(str(m2), "step3/")
        
        >>> # Batch processing
        >>> pdfs = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
        >>> results = preprocessor.process_batch(pdfs, "batch_output/")
    """
    
    def __init__(self, dpi=200, num_workers=None, poppler_path=None, verbose=False):
        """
        Initialize preprocessor
        
        Args:
            dpi (int): DPI cho output images (default: 200)
            num_workers (int): Số workers cho scan (None = auto)
            poppler_path (str): Đường dẫn Poppler (Windows only)
            verbose (bool): Print detailed logs (default: False)
        """
        self.dpi = dpi
        self.num_workers = num_workers
        self.poppler_path = poppler_path
        self.verbose = verbose
    
    def pdf_to_images(self, pdf_path, output_dir):
        """
        Step 1: Convert PDF to images
        
        Args:
            pdf_path (str): PDF file path
            output_dir (str): Output directory
            
        Returns:
            Path: manifest.json path
        """
        return convert_pdf_to_images(pdf_path, output_dir, 
                                     self.dpi, self.poppler_path)
    
    def fix_rotation(self, manifest_path, output_dir):
        """
        Step 2: Fix rotation
        
        Args:
            manifest_path (str): manifest.json path
            output_dir (str): Output directory
            
        Returns:
            Path: manifest_fixed.json path
        """
        return fix_rotation(manifest_path, output_dir)
    
    def scan(self, manifest_path, output_dir):
        """
        Step 3: OCR scan with table detection
        
        Args:
            manifest_path (str): manifest_fixed.json path
            output_dir (str): Output directory
            
        Returns:
            Path: manifest_scanned.json path
        """
        return scan_document(manifest_path, output_dir, 
                           self.num_workers, self.verbose)
    
    def get_rotations(self, pdf_path):
        """
        Get rotation metadata from PDF
        
        Args:
            pdf_path (str): PDF file path
            
        Returns:
            list: List of rotation angles for each page
        """
        return get_pdf_rotations(pdf_path)
    
    def process(self, pdf_path, output_dir="output"):
        """
        Full pipeline processing
        
        Args:
            pdf_path (str): PDF file path
            output_dir (str): Base output directory (default: "output")
        
        Returns:
            dict: Complete processing results
        """
        return full_pipeline(
            pdf_path, 
            output_dir, 
            dpi=self.dpi,
            num_workers=self.num_workers,
            poppler_path=self.poppler_path,
            verbose=self.verbose
        )
    
    def process_batch(self, pdf_paths, output_base_dir="output"):
        """
        Process multiple PDFs
        
        Args:
            pdf_paths (list): List of PDF file paths
            output_base_dir (str): Base directory (default: "output")
        
        Returns:
            list: Results for each PDF (dict with status, paths, etc.)
            
        Example:
            >>> preprocessor = PDFPreprocessor(dpi=300)
            >>> pdfs = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
            >>> results = preprocessor.process_batch(pdfs)
            >>> successful = [r for r in results if r['status'] == 'success']
            >>> print(f"Processed: {len(successful)}/{len(results)}")
        """
        results = []
        base_dir = Path(output_base_dir)
        
        for i, pdf_path in enumerate(pdf_paths, 1):
            print(f"\n{'='*70}")
            print(f"PROCESSING PDF {i}/{len(pdf_paths)}")
            print(f"File: {Path(pdf_path).name}")
            print(f"{'='*70}")
            
            pdf_name = Path(pdf_path).stem
            pdf_output_dir = base_dir / pdf_name
            
            try:
                result = self.process(pdf_path, str(pdf_output_dir))
                result['status'] = 'success'
                result['pdf_path'] = pdf_path
            except Exception as e:
                print(f"❌ Error: {e}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()
                result = {
                    'status': 'failed',
                    'pdf_path': pdf_path,
                    'error': str(e)
                }
            
            results.append(result)
        
        # Summary
        success = sum(1 for r in results if r['status'] == 'success')
        failed = len(results) - success
        
        print(f"\n{'='*70}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Total: {len(results)} PDFs")
        print(f"Success: {success}")
        print(f"Failed: {failed}")
        print(f"{'='*70}")
        
        return results


# ===========================
# QUICK PIPELINE CLASS
# ===========================

class QuickPipeline:
    """
    Simplified interface using run_pipeline() from pipeline_preprocess.py
    Uses temp/ directory automatically
    
    Example:
        >>> # Run pipeline
        >>> pipeline = QuickPipeline()
        >>> result_manifest = pipeline.run("document.pdf")
        >>> print(f"Results: {result_manifest}")
        
        >>> # Get results
        >>> data = pipeline.get_results()
        >>> print(f"Pages: {len(data['pages'])}")
        
        >>> # Clean temp directory
        >>> pipeline.clean_temp()
    """
    
    @staticmethod
    def run(pdf_path):
        """
        Run pipeline với temp directory
        
        Args:
            pdf_path (str): PDF file path
        
        Returns:
            Path: Final manifest path (temp/step3_scan/manifest_scanned.json)
        """
        run_pipeline(pdf_path)
        return STEP3_DIR / "manifest_scanned.json"
    
    @staticmethod
    def clean_temp():
        """
        Clean temp directory
        
        Example:
            >>> QuickPipeline.clean_temp()
            ✅ Cleaned: temp
        """
        import shutil
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
            print(f"✅ Cleaned: {TEMP_DIR}")
    
    @staticmethod
    def get_results():
        """
        Get results from last run
        
        Returns:
            dict: Manifest data from step 3
            
        Raises:
            FileNotFoundError: If no results found
            
        Example:
            >>> data = QuickPipeline.get_results()
            >>> for page in data['pages']:
            ...     print(f"Page {page['page_number']}: {page['file']}")
        """
        manifest_path = STEP3_DIR / "manifest_scanned.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No results found: {manifest_path}")
        return load_manifest(manifest_path)


# ===========================
# MODULE INFO
# ===========================

def get_module_info():
    """
    Print module information and usage guide
    
    Example:
        >>> from preprocess_pipeline import get_module_info
        >>> get_module_info()
    """
    print(f"""
{'='*70}
Preprocessing Pipeline Module v{__version__}
{'='*70}

Available Components:
  • pdf2img: PDF to image conversion
  • preprocess: Rotation fixing
  • scan: OCR scanning with multiprocessing
  • pipeline_preprocess: Full pipeline orchestrator

Quick Start:
  from preprocess_pipeline import PDFPreprocessor
  
  preprocessor = PDFPreprocessor(dpi=300, num_workers=4)
  result = preprocessor.process("document.pdf", "output/")

Classes:
  • PDFPreprocessor: Full-featured preprocessing
  • QuickPipeline: Simple interface using temp/

Functions:
  • convert_pdf_to_images(): PDF -> Images
  • fix_rotation(): Fix image rotation
  • scan_document(): OCR scan with table detection
  • full_pipeline(): Complete workflow
  • run_pipeline(): Auto pipeline in temp/

Workflow:
  Step 1: PDF -> Images (with rotation metadata)
  Step 2: Fix rotation based on metadata
  Step 3: OCR scan + table detection

{'='*70}
For detailed documentation, use help() on any class or function:
  >>> help(PDFPreprocessor)
  >>> help(full_pipeline)
{'='*70}
""")


# ===========================
# EXPORTS
# ===========================

__all__ = [
    # Version info
    '__version__',
    '__author__',
    
    # PDF to Image
    'process_pdf',
    'get_pdf_rotations',
    'DPI',
    'POPPLER_PATH',
    
    # Preprocessing
    'preprocess',
    'rotate_img',
    
    # Scanning (multiprocessing)
    'scan_multiprocessing',
    'scan_sequential',
    'process_page',
    'detect_table',
    'score_orientation_improved',
    'decide_rotation_improved',
    'get_ocr',
    
    # Pipeline orchestrator
    'run_pipeline',
    'TEMP_DIR',
    'STEP1_DIR',
    'STEP2_DIR',
    'STEP3_DIR',
    
    # Convenience classes
    'PDFPreprocessor',
    'QuickPipeline',
    
    # Helper functions
    'convert_pdf_to_images',
    'fix_rotation',
    'scan_document',
    'full_pipeline',
    'load_manifest',
    'get_page_images',
    
    # Module info
    'get_module_info',
]


# ===========================
# MODULE INITIALIZATION
# ===========================

# Print welcome message if module is imported interactively
if __name__ != "__main__":
    # Module imported, not run as script
    pass