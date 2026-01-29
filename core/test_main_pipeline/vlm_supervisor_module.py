"""
vlm_supervisor_module.py
VLM Supervisor - Fallback khi OCR có confidence thấp
Đọc lại ảnh và tạo JSON có cấu trúc giống OCR Extract
"""

import json
import base64
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class VLMSupervisorModule:
    """
    VLM Supervisor - Fallback module khi OCR confidence thấp
    
    Features:
    - Detect low confidence từ scan result
    - Fallback về VLM để re-extract
    - Generate JSON giống format của OCR Extract
    - Support cả Anthropic và OpenAI
    """
    
    # Prompt template để extract cấu trúc tài chính
    FINANCIAL_EXTRACT_PROMPT = """Bạn là chuyên gia phân tích báo cáo tài chính.

Tôi có một ảnh báo cáo tài chính. Nhiệm vụ của bạn:
1. Đọc và phân tích ảnh kỹ lưỡng
2. Trích xuất TOÀN BỘ dữ liệu theo cấu trúc JSON bên dưới
3. Giữ nguyên ngữ cảnh và cấu trúc phân cấp

**YÊU CẦU OUTPUT JSON:**

```json
{{
  "report_type": "balance_sheet",  // hoặc "income_statement", "cash_flow"
  "company_name": "Tên công ty",
  "report_dates": ["31/03/2025", "01/01/2025"],  // Tất cả các cột ngày
  "sections": [
    {{
      "section_name": "TÀI SẢN NGẮN HẠN",
      "section_code": "100",
      "items": [
        {{
          "code": "110",
          "name": "Tiền và các khoản tương đương tiền",
          "values": {{
            "31/03/2025": 116733376376,
            "01/01/2025": 91741974158
          }},
          "confidence": 1.0,
          "source_lines": [1, 2],
          "context": "TÀI SẢN NGẮN HẠN"
        }},
        {{
          "code": "111",
          "name": "Tiền",
          "values": {{
            "31/03/2025": 50000000000,
            "01/01/2025": 45000000000
          }},
          "confidence": 1.0,
          "source_lines": [3],
          "context": "TÀI SẢN NGẮN HẠN"
        }}
      ],
      "section_confidence": 1.0,
      "hierarchy_level": 1,
      "parent_section": null
    }}
  ],
  "metadata": {{
    "source": "vlm_supervisor",
    "extraction_time": "2025-01-28T...",
    "total_sections": 3,
    "total_items": 25
  }}
}}
```

**LƯU Ý QUAN TRỌNG:**

1. **Đọc KỸ ảnh** - Không bỏ sót bất kỳ section hay item nào
2. **Code chính xác** - Đọc đúng mã số (100, 110, 111, etc.)
3. **Tên đầy đủ** - Giữ nguyên tên tiếng Việt có dấu
4. **Số chính xác** - Values phải là INTEGER, không dấu phẩy/chấm
5. **Ngày đúng format** - Theo format DD/MM/YYYY
6. **Context** - Mỗi item phải có context (thuộc section nào)
7. **Hierarchy** - Giữ đúng cấu trúc phân cấp

**QUAN TRỌNG:**
- Chỉ trả về JSON, KHÔNG giải thích
- Đảm bảo JSON valid (có thể parse được)
- Đọc hết tất cả các section và items trong ảnh

Hãy trích xuất dữ liệu từ ảnh báo cáo tài chính."""

    def __init__(self, 
                 provider: str = 'anthropic',
                 model: str = None,
                 api_key: str = None):
        """
        Khởi tạo VLM Supervisor
        
        Args:
            provider: 'anthropic' hoặc 'openai'
            model: Model name
            api_key: API key
        """
        self.provider = provider
        
        # Default models
        if model is None:
            if provider == 'anthropic':
                model = 'claude-3-5-sonnet-20241022'
            elif provider == 'openai':
                model = 'gpt-4o'
        
        self.model = model
        
        # Initialize client
        if provider == 'anthropic':
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic package not installed")
            if api_key is None:
                raise ValueError("API key required for Anthropic")
            self.client = anthropic.Anthropic(api_key=api_key)
        
        elif provider == 'openai':
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package not installed")
            if api_key is None:
                raise ValueError("API key required for OpenAI")
            self.client = openai.OpenAI(api_key=api_key)
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        print(f"✓ VLM Supervisor initialized")
        print(f"  Provider: {provider}")
        print(f"  Model: {model}")
    
    def should_fallback(self, scan_result_path: str) -> bool:
        """
        Kiểm tra xem có nên fallback về VLM không
        
        Args:
            scan_result_path: Path to scan result JSON
            
        Returns:
            True nếu cần fallback
        """
        with open(scan_result_path, 'r', encoding='utf-8') as f:
            scan_data = json.load(f)
        
        return scan_data.get('has_low_confidence', False)
    
    def encode_image(self, image_path: str) -> Tuple[str, str]:
        """
        Encode image to base64
        
        Returns:
            (base64_string, media_type)
        """
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Determine media type
        ext = Path(image_path).suffix.lower()
        media_type_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.webp': 'image/webp',
            '.gif': 'image/gif'
        }
        media_type = media_type_map.get(ext, 'image/png')
        
        return base64_image, media_type
    
    def call_vlm(self, 
                prompt: str,
                image_base64: str,
                media_type: str,
                max_tokens: int = 8192) -> str:
        """
        Call VLM API
        
        Args:
            prompt: Extraction prompt
            image_base64: Base64 encoded image
            media_type: Image media type
            max_tokens: Max tokens in response
            
        Returns:
            VLM response text
        """
        if self.provider == 'anthropic':
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=0.0,
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
        
        elif self.provider == 'openai':
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.0
            )
            return response.choices[0].message.content
    
    def extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract và parse JSON từ VLM response"""
        # Try to find JSON in code block
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON directly
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON from VLM response")
            print(f"Error: {e}")
            print(f"Response preview: {response[:500]}...")
            raise
    
    def extract_with_vlm(self,
                        image_path: str,
                        output_json_path: Optional[str] = None,
                        verbose: bool = True) -> Dict[str, Any]:
        """
        Extract dữ liệu từ ảnh sử dụng VLM
        
        Args:
            image_path: Path to image
            output_json_path: Path to save output
            verbose: Show logs
            
        Returns:
            Extracted financial data (same format as OCR Extract)
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"VLM SUPERVISOR EXTRACTING: {Path(image_path).name}")
            print(f"{'='*70}")
        
        # Encode image
        if verbose:
            print("Encoding image...")
        image_base64, media_type = self.encode_image(image_path)
        
        # Call VLM
        if verbose:
            print(f"Calling {self.provider} VLM ({self.model})...")
        
        response = self.call_vlm(
            self.FINANCIAL_EXTRACT_PROMPT,
            image_base64,
            media_type
        )
        
        if verbose:
            print("Parsing response...")
        
        # Parse JSON
        extracted_data = self.extract_json_from_response(response)
        
        # Add VLM metadata
        if 'metadata' not in extracted_data:
            extracted_data['metadata'] = {}
        
        extracted_data['metadata'].update({
            'source': 'vlm_supervisor',
            'vlm_provider': self.provider,
            'vlm_model': self.model,
            'extraction_time': datetime.now().isoformat(),
            'source_image': str(image_path)
        })
        
        if verbose:
            sections = len(extracted_data.get('sections', []))
            items = extracted_data.get('metadata', {}).get('total_items', 0)
            print(f"\n📊 VLM Extract Results:")
            print(f"  Sections: {sections}")
            print(f"  Items: {items}")
            print(f"{'='*70}\n")
        
        # Save
        if output_json_path:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(extracted_data, f, ensure_ascii=False, indent=2)
            
            if verbose:
                print(f"✓ VLM result saved: {output_json_path}")
        
        return extracted_data
    
    def smart_extract(self,
                     scan_result_path: str,
                     image_path: str,
                     ocr_extract_path: Optional[str] = None,
                     output_json_path: Optional[str] = None,
                     verbose: bool = True) -> Dict[str, Any]:
        """
        Smart extract - Tự động quyết định dùng OCR hay VLM
        
        Args:
            scan_result_path: Path to OCR scan result
            image_path: Path to original image
            ocr_extract_path: Path to OCR extract result (if exists)
            output_json_path: Path to save final result
            verbose: Show logs
            
        Returns:
            Final extracted data
        """
        if verbose:
            print(f"\n{'='*70}")
            print("VLM SUPERVISOR - SMART EXTRACT")
            print(f"{'='*70}")
        
        # Check if should fallback
        should_fallback = self.should_fallback(scan_result_path)
        
        if verbose:
            if should_fallback:
                print("⚠️  OCR confidence LOW - Falling back to VLM")
            else:
                print("✓ OCR confidence OK - Using OCR extract")
        
        if should_fallback:
            # Use VLM
            result = self.extract_with_vlm(
                image_path,
                output_json_path,
                verbose=verbose
            )
            result['metadata']['extraction_method'] = 'vlm_supervisor'
        else:
            # Use OCR extract
            if ocr_extract_path and Path(ocr_extract_path).exists():
                with open(ocr_extract_path, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                
                if 'metadata' not in result:
                    result['metadata'] = {}
                result['metadata']['extraction_method'] = 'ocr_extract'
                
                if verbose:
                    print(f"✓ Using OCR extract: {ocr_extract_path}")
                
                # Save to output path if different
                if output_json_path and output_json_path != ocr_extract_path:
                    with open(output_json_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
            else:
                if verbose:
                    print("⚠️  OCR extract not found - Falling back to VLM")
                result = self.extract_with_vlm(
                    image_path,
                    output_json_path,
                    verbose=verbose
                )
                result['metadata']['extraction_method'] = 'vlm_fallback'
        
        return result


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("VLM SUPERVISOR MODULE - EXAMPLE")
    print("="*70)
    
    # Initialize
    supervisor = VLMSupervisorModule(
        provider='anthropic',
        model='claude-3-5-sonnet-20241022',
        api_key='your-api-key-here'
    )
    
    # Example 1: Direct VLM extract
    print("\n=== Example 1: Direct VLM Extract ===")
    result = supervisor.extract_with_vlm(
        image_path='balance_sheet.png',
        output_json_path='vlm_extracted.json',
        verbose=True
    )
    
    # Example 2: Smart extract (auto decide OCR vs VLM)
    print("\n=== Example 2: Smart Extract ===")
    result = supervisor.smart_extract(
        scan_result_path='scan_result.json',
        image_path='balance_sheet.png',
        ocr_extract_path='extracted_report.json',
        output_json_path='final_extract.json',
        verbose=True
    )
    
    print(f"\n✓ Extraction method: {result['metadata']['extraction_method']}")
