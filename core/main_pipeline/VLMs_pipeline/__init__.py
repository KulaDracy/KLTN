"""
VLMs Pipeline Module
====================

Module sử dụng Vision Language Models (VLMs) để trích xuất thông tin từ tài liệu.
Hỗ trợ nhiều VLM providers: Claude (Anthropic), GPT-4V (OpenAI).

Features:
---------
- ✅ Multi-provider support (Anthropic Claude, OpenAI GPT-4V)
- ✅ Batch processing with parallel execution
- ✅ Auto image loading from directories
- ✅ OCR-assisted extraction (hybrid mode)
- ✅ Financial document optimized prompts
- ✅ Structured JSON output
- ✅ Error handling and retry logic
- ✅ Cost tracking and rate limiting

Components:
-----------
1. vlm_extract.py
   - VLMsExtractor: Main VLM extraction class
   - VLMConfig: Configuration dataclass
   - Support cho cả HIGH và LOW confidence modes
   - Auto load images + OCR results

VLM Modes:
----------
1. **VLMs_detect** (HIGH confidence ≥0.9)
   - OCR đã chính xác, VLM verify và bổ sung
   - Focus: Table structure, values verification
   - Faster, lower cost

2. **VLMs_supervisor** (LOW confidence <0.9)
   - OCR kém chất lượng, VLM full extraction
   - Focus: Complete re-extraction from image
   - More thorough, higher accuracy

Supported Providers:
-------------------
- **Anthropic Claude**
  - Models: claude-3-opus, claude-3-sonnet, claude-3-haiku
  - Best for: Vietnamese text, complex documents
  - Vision capabilities: Excellent

- **OpenAI GPT-4V**
  - Models: gpt-4-vision-preview, gpt-4-turbo
  - Best for: General documents
  - Vision capabilities: Very good

Usage Examples:
---------------
    # Basic VLM extraction
    >>> from VLMs_pipeline import VLMsExtractor, VLMConfig
    >>> 
    >>> config = VLMConfig(
    ...     provider='anthropic',
    ...     model='claude-3-sonnet-20240229',
    ...     api_key='sk-ant-...'
    ... )
    >>> 
    >>> vlm = VLMsExtractor(config=config)
    >>> result = vlm.extract_from_image("document.png")
    
    # Process multiple pages
    >>> pages = [
    ...     {'page_number': 1, 'image_path': 'page1.png'},
    ...     {'page_number': 2, 'image_path': 'page2.png'},
    ... ]
    >>> results = vlm.process_pages(pages, output_json="vlm_results.json")
    
    # With OCR assistance (hybrid mode)
    >>> pages_with_ocr = [
    ...     {
    ...         'page_number': 1,
    ...         'image_path': 'page1.png',
    ...         'ocr_json_path': 'page1_ocr.json',
    ...         'has_ocr': True
    ...     }
    ... ]
    >>> results = vlm.process_pages(pages_with_ocr)
    
    # Quick extraction (convenience function)
    >>> from VLMs_pipeline import extract_with_vlm
    >>> result = extract_with_vlm(
    ...     image_path="doc.png",
    ...     provider="anthropic",
    ...     api_key="sk-ant-..."
    ... )

Integration with Confidence Checker:
------------------------------------
    >>> from main_pipeline import ConfidenceChecker
    >>> from VLMs_pipeline import VLMsExtractor, VLMConfig
    >>> 
    >>> # Classify pages
    >>> checker = ConfidenceChecker("ocr_results.json")
    >>> classification = checker.process()
    >>> 
    >>> # Process HIGH confidence pages (VLMs_detect mode)
    >>> config_detect = VLMConfig(
    ...     provider='anthropic',
    ...     api_key='...',
    ...     mode='detect'  # Faster, focus on verification
    ... )
    >>> vlm_detect = VLMsExtractor(config=config_detect)
    >>> high_results = vlm_detect.process_pages(
    ...     classification['vlms_detect_pages']
    ... )
    >>> 
    >>> # Process LOW confidence pages (VLMs_supervisor mode)
    >>> config_supervisor = VLMConfig(
    ...     provider='anthropic',
    ...     api_key='...',
    ...     mode='supervisor'  # Thorough, full extraction
    ... )
    >>> vlm_supervisor = VLMsExtractor(config=config_supervisor)
    >>> low_results = vlm_supervisor.process_pages(
    ...     classification['vlms_supervisor_pages']
    ... )

Cost Optimization:
-----------------
    # Use Haiku for simple documents (cheaper)
    >>> config = VLMConfig(
    ...     provider='anthropic',
    ...     model='claude-3-haiku-20240307',  # Cheapest
    ...     api_key='...'
    ... )
    
    # Use Sonnet for complex financial docs (balanced)
    >>> config = VLMConfig(
    ...     provider='anthropic',
    ...     model='claude-3-sonnet-20240229',  # Recommended
    ...     api_key='...'
    ... )
    
    # Use Opus for highest accuracy (expensive)
    >>> config = VLMConfig(
    ...     provider='anthropic',
    ...     model='claude-3-opus-20240229',  # Most accurate
    ...     api_key='...'
    ... )
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

__version__ = "1.0.0"

# ===========================
# IMPORTS
# ===========================
try:
    from .vlm_extract import (
        VLMsExtractor,
        VLMConfig,
        ProcessedPage,
        extract_with_vlm,
        extract_batch,
        load_image_base64,
    )
    VLM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  VLM extract module import warning: {e}")
    VLM_AVAILABLE = False
    VLMsExtractor = None
    VLMConfig = None
    ProcessedPage = None
    extract_with_vlm = None
    extract_batch = None
    load_image_base64 = None


# ===========================
# PROVIDER VALIDATION
# ===========================

def check_provider_available(provider: str) -> dict:
    """
    Kiểm tra provider có available không
    
    Args:
        provider: 'anthropic' hoặc 'openai'
    
    Returns:
        dict: {
            'available': bool,
            'package_installed': bool,
            'message': str
        }
    """
    result = {
        'provider': provider,
        'available': False,
        'package_installed': False,
        'message': ''
    }
    
    if provider == 'anthropic':
        try:
            import anthropic
            result['package_installed'] = True
            result['available'] = True
            result['message'] = 'Anthropic Claude available'
        except ImportError:
            result['message'] = 'Install: pip install anthropic'
    
    elif provider == 'openai':
        try:
            import openai
            result['package_installed'] = True
            result['available'] = True
            result['message'] = 'OpenAI GPT-4V available'
        except ImportError:
            result['message'] = 'Install: pip install openai'
    
    else:
        result['message'] = f'Unknown provider: {provider}'
    
    return result


def get_available_providers() -> List[str]:
    """
    Get list of available VLM providers
    
    Returns:
        list: List of available provider names
    """
    providers = []
    
    for provider in ['anthropic', 'openai']:
        if check_provider_available(provider)['available']:
            providers.append(provider)
    
    return providers


# ===========================
# CONVENIENCE FUNCTIONS
# ===========================

def quick_vlm_extract(
    image_path: str,
    provider: str = 'anthropic',
    api_key: str = None,
    model: str = None,
    include_ocr: bool = False,
    ocr_json_path: str = None
) -> dict:
    """
    Quick VLM extraction cho một ảnh
    
    Args:
        image_path: Đường dẫn ảnh
        provider: 'anthropic' hoặc 'openai'
        api_key: API key (required)
        model: Model name (optional, uses default)
        include_ocr: Include OCR data
        ocr_json_path: Path to OCR JSON (if include_ocr=True)
    
    Returns:
        dict: Extraction results
        
    Example:
        >>> result = quick_vlm_extract(
        ...     "document.png",
        ...     provider="anthropic",
        ...     api_key="sk-ant-..."
        ... )
    """
    if not VLM_AVAILABLE:
        raise ImportError("VLM extract module not available")
    
    if not api_key:
        raise ValueError("API key is required")
    
    # Default models
    if not model:
        if provider == 'anthropic':
            model = 'claude-3-sonnet-20240229'
        elif provider == 'openai':
            model = 'gpt-4-vision-preview'
    
    config = VLMConfig(
        provider=provider,
        model=model,
        api_key=api_key
    )
    
    vlm = VLMsExtractor(config=config)
    
    page_info = {
        'page_number': 1,
        'image_path': image_path,
        'has_ocr': include_ocr,
        'ocr_json_path': ocr_json_path if include_ocr else None
    }
    
    return vlm.extract_from_page(page_info)


def batch_vlm_extract(
    images_dir: str,
    provider: str,
    api_key: str,
    output_json: str = "vlm_results.json",
    model: str = None,
    max_workers: int = 3,
    verbose: bool = True
) -> dict:
    """
    Batch VLM extraction cho nhiều ảnh
    
    Args:
        images_dir: Thư mục chứa ảnh
        provider: VLM provider
        api_key: API key
        output_json: Output JSON file
        model: Model name (optional)
        max_workers: Số workers song song
        verbose: In log
    
    Returns:
        dict: Batch results
        
    Example:
        >>> results = batch_vlm_extract(
        ...     "images/",
        ...     provider="anthropic",
        ...     api_key="sk-ant-...",
        ...     output_json="vlm_output.json"
        ... )
    """
    if not VLM_AVAILABLE:
        raise ImportError("VLM extract module not available")
    
    # Default model
    if not model:
        if provider == 'anthropic':
            model = 'claude-3-sonnet-20240229'
        elif provider == 'openai':
            model = 'gpt-4-vision-preview'
    
    config = VLMConfig(
        provider=provider,
        model=model,
        api_key=api_key
    )
    
    vlm = VLMsExtractor(config=config)
    
    # Load images from directory
    images_path = Path(images_dir)
    image_files = sorted(images_path.glob('*.png')) + \
                  sorted(images_path.glob('*.jpg')) + \
                  sorted(images_path.glob('*.jpeg'))
    
    pages = [
        {
            'page_number': i + 1,
            'image_path': str(img),
            'has_ocr': False
        }
        for i, img in enumerate(image_files)
    ]
    
    return vlm.process_pages(
        pages,
        output_json=output_json,
        max_workers=max_workers,
        verbose=verbose
    )


def vlm_verify_ocr(
    image_path: str,
    ocr_json_path: str,
    provider: str,
    api_key: str,
    model: str = None
) -> dict:
    """
    Sử dụng VLM để verify và bổ sung OCR results
    
    Args:
        image_path: Đường dẫn ảnh
        ocr_json_path: Đường dẫn OCR JSON
        provider: VLM provider
        api_key: API key
        model: Model name (optional)
    
    Returns:
        dict: Verified and enhanced results
        
    Example:
        >>> result = vlm_verify_ocr(
        ...     "doc.png",
        ...     "doc_ocr.json",
        ...     provider="anthropic",
        ...     api_key="sk-ant-..."
        ... )
    """
    if not VLM_AVAILABLE:
        raise ImportError("VLM extract module not available")
    
    if not model:
        if provider == 'anthropic':
            model = 'claude-3-sonnet-20240229'
        elif provider == 'openai':
            model = 'gpt-4-vision-preview'
    
    config = VLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        mode='detect'  # Verification mode
    )
    
    vlm = VLMsExtractor(config=config)
    
    page_info = {
        'page_number': 1,
        'image_path': image_path,
        'ocr_json_path': ocr_json_path,
        'has_ocr': True
    }
    
    return vlm.extract_from_page(page_info)


def estimate_cost(
    num_images: int,
    provider: str = 'anthropic',
    model: str = None,
    image_size_mb: float = 1.0
) -> dict:
    """
    Ước tính chi phí VLM processing
    
    Args:
        num_images: Số lượng ảnh
        provider: Provider name
        model: Model name
        image_size_mb: Kích thước trung bình ảnh (MB)
    
    Returns:
        dict: Cost estimate
        
    Example:
        >>> cost = estimate_cost(100, 'anthropic', 'claude-3-sonnet-20240229')
        >>> print(f"Estimated cost: ${cost['total_usd']:.2f}")
    """
    # Pricing estimates (approximate, as of 2024)
    pricing = {
        'anthropic': {
            'claude-3-opus-20240229': {'input': 15.0, 'output': 75.0},  # per 1M tokens
            'claude-3-sonnet-20240229': {'input': 3.0, 'output': 15.0},
            'claude-3-haiku-20240307': {'input': 0.25, 'output': 1.25},
        },
        'openai': {
            'gpt-4-vision-preview': {'input': 10.0, 'output': 30.0},
            'gpt-4-turbo': {'input': 10.0, 'output': 30.0},
        }
    }
    
    if not model:
        if provider == 'anthropic':
            model = 'claude-3-sonnet-20240229'
        elif provider == 'openai':
            model = 'gpt-4-vision-preview'
    
    # Rough estimates
    tokens_per_image = 1000 + (image_size_mb * 500)  # Image + text tokens
    output_tokens = 2000  # Estimated output
    
    prices = pricing.get(provider, {}).get(model, {'input': 5.0, 'output': 15.0})
    
    input_cost = (num_images * tokens_per_image * prices['input']) / 1_000_000
    output_cost = (num_images * output_tokens * prices['output']) / 1_000_000
    total_cost = input_cost + output_cost
    
    return {
        'provider': provider,
        'model': model,
        'num_images': num_images,
        'estimated_input_tokens': int(num_images * tokens_per_image),
        'estimated_output_tokens': int(num_images * output_tokens),
        'input_cost_usd': round(input_cost, 2),
        'output_cost_usd': round(output_cost, 2),
        'total_usd': round(total_cost, 2),
        'note': 'This is a rough estimate. Actual costs may vary.'
    }


# ===========================
# MODULE INFO
# ===========================

def get_module_info():
    """Print VLMs Pipeline module information"""
    
    providers = get_available_providers()
    
    print(f"""
{'='*70}
VLMs Pipeline Module v{__version__}
{'='*70}

Available Providers:
  {'  '.join(['✓ ' + p for p in providers]) if providers else '✗ No providers available'}

Supported VLMs:
  Anthropic:
    • claude-3-opus-20240229    (Most accurate, expensive)
    • claude-3-sonnet-20240229  (Balanced, recommended)
    • claude-3-haiku-20240307   (Fast, cheap)
  
  OpenAI:
    • gpt-4-vision-preview      (Good vision capabilities)
    • gpt-4-turbo              (Fast, improved)

Features:
  • Multi-provider support (Anthropic, OpenAI)
  • Batch processing with parallelization
  • OCR-assisted extraction (hybrid mode)
  • Financial document optimized
  • Error handling and retry logic
  • Cost tracking

Quick Start:
  from VLMs_pipeline import quick_vlm_extract
  
  result = quick_vlm_extract(
      "document.png",
      provider="anthropic",
      api_key="sk-ant-..."
  )

Functions:
  • quick_vlm_extract(): Single image extraction
  • batch_vlm_extract(): Batch process directory
  • vlm_verify_ocr(): Verify OCR with VLM
  • estimate_cost(): Cost estimation
  • check_provider_available(): Check provider status

Classes:
  • VLMsExtractor: Main extraction class
  • VLMConfig: Configuration dataclass
  • ProcessedPage: Page info dataclass

{'='*70}
For detailed help: help(VLMsExtractor)
{'='*70}
""")


# ===========================
# EXPORTS
# ===========================

__all__ = [
    # Version
    '__version__',
    
    # Main Classes
    'VLMsExtractor',
    'VLMConfig',
    'ProcessedPage',
    
    # Core Functions
    'extract_with_vlm',
    'extract_batch',
    'load_image_base64',
    
    # Convenience Functions
    'quick_vlm_extract',
    'batch_vlm_extract',
    'vlm_verify_ocr',
    
    # Provider Management
    'check_provider_available',
    'get_available_providers',
    
    # Utilities
    'estimate_cost',
    'get_module_info',
    
    # Availability Flag
    'VLM_AVAILABLE',
]