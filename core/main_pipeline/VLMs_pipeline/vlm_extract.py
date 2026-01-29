"""
vlms_extract.py
Module VLMs Extract - Sử dụng Vision Language Models để extract dữ liệu
Auto load nhiều ảnh từ folder đã được xử lý
"""

import json
import base64
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# VLMs imports
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("⚠️  anthropic not installed. Install: pip install anthropic")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  openai not installed. Install: pip install openai")


@dataclass
class VLMConfig:
    """Cấu hình cho VLM"""
    provider: str  # 'anthropic' hoặc 'openai'
    model: str
    api_key: str
    max_tokens: int = 4096
    temperature: float = 0.0


@dataclass
class ProcessedPage:
    """Thông tin về một trang đã xử lý"""
    page_number: int
    image_path: str
    ocr_json_path: Optional[str] = None
    has_ocr: bool = False


class VLMsExtractor:
    """
    VLMs Extractor - Sử dụng Vision Language Models
    Auto load và xử lý nhiều ảnh từ folder
    """
    
    # Prompts templates
    FINANCIAL_EXTRACT_PROMPT = """Bạn là chuyên gia phân tích báo cáo tài chính.

Tôi đang cung cấp cho bạn:
1. Kết quả OCR từ một trang báo cáo tài chính (có thể có lỗi nhận dạng)
2. Ảnh gốc của trang đó

Nhiệm vụ của bạn:
- Phân tích ảnh và OCR text để xác định chính xác cấu trúc bảng
- Sửa lỗi OCR nếu có (dựa vào ảnh)
- Trích xuất dữ liệu theo cấu trúc JSON yêu cầu

Dữ liệu OCR đã có:
{ocr_data}

Yêu cầu output JSON format:
{{
  "sections": [
    {{
      "section": "TÀI SẢN NGẮN HẠN",
      "code": "100",
      "items": [
        {{
          "code": "110",
          "name": "Tiền và các khoản tương đương tiền",
          "values": {{
            "31/03/2025": 116733376376,
            "01/01/2025": 91741974158
          }}
        }}
      ]
    }}
  ]
}}

Lưu ý:
- Kiểm tra lại số liệu từ ảnh nếu OCR có vẻ sai
- Đảm bảo code và name chính xác
- Values phải là số nguyên (integer)
- Giữ nguyên dấu tiếng Việt

Chỉ trả về JSON, không giải thích thêm."""

    FINANCIAL_DIRECT_EXTRACT_PROMPT = """Bạn là chuyên gia phân tích báo cáo tài chính.

Phân tích ảnh báo cáo tài chính này và trích xuất dữ liệu thành JSON.

Yêu cầu output JSON format:
{{
  "sections": [
    {{
      "section": "TÀI SẢN NGẮN HẠN",
      "code": "100",
      "items": [
        {{
          "code": "110",
          "name": "Tiền và các khoản tương đương tiền",
          "values": {{
            "31/03/2025": 116733376376,
            "01/01/2025": 91741974158
          }}
        }}
      ]
    }}
  ]
}}

Lưu ý:
- Đọc chính xác tất cả số liệu từ ảnh
- Code và name phải chính xác
- Values phải là số nguyên (integer)
- Giữ nguyên dấu tiếng Việt
- Đọc cẩn thận các cột ngày tháng

Chỉ trả về JSON, không giải thích thêm."""

    def __init__(self, config: VLMConfig):
        """
        Khởi tạo VLMs Extractor
        
        Args:
            config: Cấu hình VLM (provider, model, API key)
        """
        self.config = config
        
        # Initialize client
        if config.provider == 'anthropic':
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic package not installed")
            self.client = anthropic.Anthropic(api_key=config.api_key)
        elif config.provider == 'openai':
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package not installed")
            self.client = openai.OpenAI(api_key=config.api_key)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
        
        print(f"✓ VLMs Extractor initialized")
        print(f"  Provider: {config.provider}")
        print(f"  Model: {config.model}")
    
    def scan_folder(self, 
                    image_folder: str,
                    ocr_folder: str = None,
                    image_extensions: List[str] = None) -> List[ProcessedPage]:
        """
        Scan folder để tìm ảnh và OCR JSON tương ứng
        
        Args:
            image_folder: Folder chứa ảnh
            ocr_folder: Folder chứa OCR JSON (optional)
            image_extensions: List extensions được chấp nhận
            
        Returns:
            List ProcessedPage
        """
        if image_extensions is None:
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        
        image_path = Path(image_folder)
        if not image_path.exists():
            raise FileNotFoundError(f"Image folder not found: {image_folder}")
        
        print(f"\n📁 Scanning folder: {image_folder}")
        
        # Tìm tất cả ảnh
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_path.glob(f'*{ext}'))
        
        image_files = sorted(image_files, key=lambda x: x.name)
        
        print(f"  ✓ Found {len(image_files)} images")
        
        # Map với OCR JSON
        processed_pages = []
        
        for i, img_file in enumerate(image_files, 1):
            ocr_json_path = None
            has_ocr = False
            
            # Tìm OCR JSON tương ứng
            if ocr_folder:
                ocr_path = Path(ocr_folder)
                # Thử nhiều pattern
                patterns = [
                    img_file.stem + '.json',           # same_name.json
                    f'page_{i}.json',                  # page_1.json
                    f'page_{i}_ocr.json',              # page_1_ocr.json
                    img_file.stem + '_ocr.json',       # image_name_ocr.json
                ]
                
                for pattern in patterns:
                    candidate = ocr_path / pattern
                    if candidate.exists():
                        ocr_json_path = str(candidate)
                        has_ocr = True
                        break
            
            page = ProcessedPage(
                page_number=i,
                image_path=str(img_file),
                ocr_json_path=ocr_json_path,
                has_ocr=has_ocr
            )
            
            processed_pages.append(page)
        
        # Statistics
        with_ocr = sum(1 for p in processed_pages if p.has_ocr)
        without_ocr = len(processed_pages) - with_ocr
        
        print(f"\n📊 Scan results:")
        print(f"  • Total pages: {len(processed_pages)}")
        print(f"  • With OCR: {with_ocr}")
        print(f"  • Without OCR: {without_ocr}")
        
        return processed_pages
    
    def load_from_scanner_result(self,
                                 scanner_result_path: str,
                                 image_folder: str) -> List[ProcessedPage]:
        """
        Load pages từ kết quả scanner JSON
        
        Args:
            scanner_result_path: Đường dẫn file JSON kết quả scanner
            image_folder: Folder chứa ảnh
            
        Returns:
            List ProcessedPage
        """
        print(f"\n📥 Loading from scanner result: {scanner_result_path}")
        
        # Load scanner result
        with open(scanner_result_path, 'r', encoding='utf-8') as f:
            scanner_data = json.load(f)
        
        pages_data = scanner_data.get('pages', {})
        
        # Map với ảnh
        image_path = Path(image_folder)
        image_files = sorted(image_path.glob('*.png')) + \
                     sorted(image_path.glob('*.jpg')) + \
                     sorted(image_path.glob('*.jpeg'))
        
        processed_pages = []
        
        for page_key in sorted(pages_data.keys()):
            page_number = int(page_key.split('_')[1])
            
            # Tìm ảnh tương ứng
            img_file = None
            if page_number <= len(image_files):
                img_file = image_files[page_number - 1]
            
            if img_file:
                page = ProcessedPage(
                    page_number=page_number,
                    image_path=str(img_file),
                    ocr_json_path=scanner_result_path,  # Lưu path đến scanner result
                    has_ocr=True
                )
                processed_pages.append(page)
        
        print(f"  ✓ Loaded {len(processed_pages)} pages from scanner result")
        
        return processed_pages
    
    def encode_image_base64(self, image_path: str) -> str:
        """
        Encode ảnh thành base64 string
        
        Args:
            image_path: Đường dẫn đến file ảnh
            
        Returns:
            Base64 encoded string
        """
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        return base64.b64encode(image_data).decode('utf-8')
    
    def load_ocr_result(self, ocr_json_path: str, page_number: int = None) -> Dict[str, Any]:
        """
        Load kết quả OCR từ file JSON
        
        Args:
            ocr_json_path: Đường dẫn file JSON kết quả OCR
            page_number: Số trang (nếu là scanner result)
            
        Returns:
            Dict chứa OCR data
        """
        with open(ocr_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Nếu là scanner result, extract page specific data
        if 'pages' in data and page_number:
            page_key = f'page_{page_number}'
            if page_key in data['pages']:
                return data['pages'][page_key]
        
        return data
    
    def call_anthropic_vision(self, 
                             prompt: str, 
                             image_base64: str,
                             media_type: str = "image/png") -> str:
        """
        Gọi Anthropic Claude với vision
        
        Args:
            prompt: Text prompt
            image_base64: Base64 encoded image
            media_type: Loại ảnh (image/png, image/jpeg)
            
        Returns:
            Response text từ model
        """
        message = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        
        return message.content[0].text
    
    def call_openai_vision(self, 
                          prompt: str, 
                          image_base64: str) -> str:
        """
        Gọi OpenAI GPT-4 Vision
        
        Args:
            prompt: Text prompt
            image_base64: Base64 encoded image
            
        Returns:
            Response text từ model
        """
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
        
        return response.choices[0].message.content
    
    def call_vlm(self, prompt: str, image_base64: str, media_type: str = "image/png") -> str:
        """
        Gọi VLM (tự động chọn provider)
        
        Args:
            prompt: Text prompt
            image_base64: Base64 encoded image
            media_type: Loại ảnh
            
        Returns:
            Response text từ model
        """
        if self.config.provider == 'anthropic':
            return self.call_anthropic_vision(prompt, image_base64, media_type)
        elif self.config.provider == 'openai':
            return self.call_openai_vision(prompt, image_base64)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """
        Trích xuất JSON từ response của VLM
        
        Args:
            response: Response text từ VLM (có thể có markdown code blocks)
            
        Returns:
            Dict từ JSON
        """
        # Remove markdown code blocks nếu có
        response = response.strip()
        
        # Try to find JSON in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Assume entire response is JSON
            json_str = response
        
        # Parse JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parse error: {e}")
            print(f"Response preview: {response[:500]}...")
            raise
    
    def extract_page(self,
                    page: ProcessedPage,
                    verbose: bool = True) -> Dict[str, Any]:
        """
        Extract một trang
        
        Args:
            page: ProcessedPage object
            verbose: Hiển thị log
            
        Returns:
            Dict chứa extracted data
        """
        if verbose:
            print(f"\n🔍 Page {page.page_number}: {Path(page.image_path).name}")
        
        start_time = time.time()
        
        # Encode image
        image_base64 = self.encode_image_base64(page.image_path)
        
        # Determine media type
        suffix = Path(page.image_path).suffix.lower()
        media_type_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg'
        }
        media_type = media_type_map.get(suffix, 'image/png')
        
        # Create prompt based on OCR availability
        if page.has_ocr and page.ocr_json_path:
            # Load OCR data
            ocr_data = self.load_ocr_result(page.ocr_json_path, page.page_number)
            ocr_sections_str = json.dumps(ocr_data.get('sections', []), 
                                         ensure_ascii=False, indent=2)
            prompt = self.FINANCIAL_EXTRACT_PROMPT.format(ocr_data=ocr_sections_str)
            if verbose:
                print(f"  📝 Using OCR data")
        else:
            # Direct extraction without OCR
            prompt = self.FINANCIAL_DIRECT_EXTRACT_PROMPT
            if verbose:
                print(f"  🖼️  Direct extraction (no OCR)")
        
        # Call VLM
        if verbose:
            print(f"  🤖 Calling {self.config.provider}...")
        
        response = self.call_vlm(prompt, image_base64, media_type)
        
        # Extract JSON from response
        result = self.extract_json_from_response(response)
        
        processing_time = time.time() - start_time
        
        # Create final result
        final_result = {
            'page_number': page.page_number,
            'image_path': page.image_path,
            'has_ocr': page.has_ocr,
            'vlm_provider': self.config.provider,
            'vlm_model': self.config.model,
            'processing_time': round(processing_time, 2),
            'data': result
        }
        
        if verbose:
            sections_count = len(result.get('sections', []))
            print(f"  ✓ Extracted {sections_count} sections in {processing_time:.2f}s")
        
        return final_result
    
    def extract_folder(self,
                      image_folder: str,
                      ocr_folder: str = None,
                      scanner_result_path: str = None,
                      output_dir: str = None,
                      max_workers: int = 1,
                      verbose: bool = True) -> Dict[str, Any]:
        """
        Extract toàn bộ folder
        
        Args:
            image_folder: Folder chứa ảnh
            ocr_folder: Folder chứa OCR JSON (optional)
            scanner_result_path: Path đến scanner result JSON (optional)
            output_dir: Folder output
            max_workers: Số workers concurrent (mặc định 1 để tránh rate limit)
            verbose: Hiển thị log
            
        Returns:
            Dict chứa tất cả results
        """
        print("\n" + "="*70)
        print("VLMs FOLDER EXTRACTION")
        print("="*70)
        
        # Scan folder
        if scanner_result_path:
            pages = self.load_from_scanner_result(scanner_result_path, image_folder)
        else:
            pages = self.scan_folder(image_folder, ocr_folder)
        
        if not pages:
            print("❌ No pages found")
            return {}
        
        # Create output dir
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Process pages
        print(f"\n🚀 Processing {len(pages)} pages with {max_workers} worker(s)...")
        
        results = {}
        
        if max_workers == 1:
            # Sequential processing
            for i, page in enumerate(pages, 1):
                print(f"\n[{i}/{len(pages)}]")
                try:
                    result = self.extract_page(page, verbose=verbose)
                    results[f'page_{page.page_number}'] = result
                    
                    # Save individual result
                    if output_dir:
                        output_path = Path(output_dir) / f'page_{page.page_number}_vlm.json'
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(result, f, ensure_ascii=False, indent=2)
                        if verbose:
                            print(f"  💾 Saved to: {output_path.name}")
                
                except Exception as e:
                    print(f"  ❌ Error: {e}")
                    results[f'page_{page.page_number}'] = {
                        'error': str(e),
                        'page_number': page.page_number,
                        'image_path': page.image_path
                    }
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_page = {
                    executor.submit(self.extract_page, page, False): page
                    for page in pages
                }
                
                for i, future in enumerate(as_completed(future_to_page), 1):
                    page = future_to_page[future]
                    print(f"\n[{i}/{len(pages)}] Page {page.page_number}")
                    
                    try:
                        result = future.result()
                        results[f'page_{page.page_number}'] = result
                        
                        # Save
                        if output_dir:
                            output_path = Path(output_dir) / f'page_{page.page_number}_vlm.json'
                            with open(output_path, 'w', encoding='utf-8') as f:
                                json.dump(result, f, ensure_ascii=False, indent=2)
                        
                        print(f"  ✓ Success")
                    except Exception as e:
                        print(f"  ❌ Error: {e}")
                        results[f'page_{page.page_number}'] = {
                            'error': str(e),
                            'page_number': page.page_number
                        }
        
        # Create summary
        success_count = sum(1 for r in results.values() if 'error' not in r)
        total_sections = sum(len(r.get('data', {}).get('sections', [])) 
                           for r in results.values() if 'error' not in r)
        
        summary = {
            'metadata': {
                'total_pages': len(pages),
                'successful': success_count,
                'failed': len(pages) - success_count,
                'total_sections_extracted': total_sections,
                'vlm_provider': self.config.provider,
                'vlm_model': self.config.model
            },
            'pages': results
        }
        
        # Save summary
        if output_dir:
            summary_path = Path(output_dir) / 'vlm_extraction_summary.json'
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Summary saved to: {summary_path}")
        
        # Print statistics
        print("\n" + "="*70)
        print("EXTRACTION SUMMARY")
        print("="*70)
        print(f"Total Pages:     {len(pages)}")
        print(f"  ✓ Successful:  {success_count}")
        print(f"  ✗ Failed:      {len(pages) - success_count}")
        print(f"Total Sections:  {total_sections}")
        print("="*70)
        
        return summary


# Ví dụ sử dụng
if __name__ == "__main__":
    print("="*70)
    print("VLMs EXTRACTOR - FOLDER PROCESSING")
    print("="*70)
    
    # Setup
    config = VLMConfig(
        provider='anthropic',
        model='claude-3-5-sonnet-20241022',
        api_key='your-anthropic-api-key',
        max_tokens=4096,
        temperature=0.0
    )
    
    extractor = VLMsExtractor(config)
    
    
    # ===== CÁCH 1: Extract folder với OCR có sẵn =====
    print("\n=== CÁCH 1: Extract folder với OCR ===")
    
    results = extractor.extract_folder(
        image_folder='./images',
        ocr_folder='./ocr_results',
        output_dir='./vlm_results',
        max_workers=1,  # Sequential để tránh rate limit
        verbose=True
    )
    
    
    # ===== CÁCH 2: Extract từ scanner result =====
    print("\n\n=== CÁCH 2: Extract từ scanner result ===")
    
    results_from_scanner = extractor.extract_folder(
        image_folder='./images',
        scanner_result_path='./ocr_results.json',  # Kết quả từ scanner
        output_dir='./vlm_results',
        max_workers=1,
        verbose=True
    )
    
    
    # ===== CÁCH 3: Extract folder không có OCR (direct) =====
    print("\n\n=== CÁCH 3: Direct extraction (no OCR) ===")
    
    results_direct = extractor.extract_folder(
        image_folder='./images',
        output_dir='./vlm_results_direct',
        max_workers=1,
        verbose=True
    )
    
    
    # ===== CÁCH 4: Parallel processing (cẩn thận rate limit) =====
    print("\n\n=== CÁCH 4: Parallel processing ===")
    
    results_parallel = extractor.extract_folder(
        image_folder='./images',
        ocr_folder='./ocr_results',
        output_dir='./vlm_results_parallel',
        max_workers=2,  # 2 concurrent requests
        verbose=False
    )