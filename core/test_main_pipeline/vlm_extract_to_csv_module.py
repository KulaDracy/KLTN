"""
vlm_extract_to_csv_module.py
VLM Extract - Đọc JSON từ bước trước, verify với ảnh gốc, xuất CSV có cấu trúc
"""

import json
import csv
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


class VLMExtractToCSVModule:
    """
    VLM Extract to CSV - Verify JSON và export CSV
    
    Features:
    - Load JSON từ OCR/VLM Supervisor
    - Verify accuracy với ảnh gốc sử dụng VLM
    - Sửa lỗi nếu có
    - Export sang CSV có cấu trúc
    """
    
    VERIFICATION_PROMPT_TEMPLATE = """Bạn là chuyên gia kiểm tra dữ liệu tài chính.

Tôi đã extract dữ liệu từ báo cáo tài chính (bằng OCR hoặc VLM). Bây giờ tôi cần bạn:
1. Xem ảnh gốc báo cáo tài chính
2. So sánh với dữ liệu đã extract (JSON bên dưới)
3. Kiểm tra và sửa các lỗi:
   - Số liệu sai
   - Tên khoản mục sai
   - Mã code sai
   - Ngày tháng sai
   - Thiếu sections/items

**DỮ LIỆU ĐÃ EXTRACT:**
```json
{extracted_data}
```

**YÊU CẦU OUTPUT:**

Trả về JSON với format sau:

```json
{{
  "verification_status": "verified",  // "verified" hoặc "corrected"
  "corrections": [
    {{
      "location": "Section: TÀI SẢN NGẮN HẠN, Item: 110",
      "field": "values.31/03/2025",
      "original_value": "116733376376",
      "corrected_value": "116733376375",
      "reason": "Số cuối cùng là 5, không phải 6"
    }}
  ],
  "verified_data": {{
    "report_type": "balance_sheet",
    "company_name": "...",
    "report_dates": ["31/03/2025", "01/01/2025"],
    "sections": [
      {{
        "section_name": "TÀI SẢN NGẮN HẠN",
        "section_code": "100",
        "items": [
          {{
            "code": "110",
            "name": "Tiền và các khoản tương đương tiền",
            "values": {{
              "31/03/2025": 116733376375,
              "01/01/2025": 91741974158
            }},
            "confidence": 1.0,
            "source_lines": [1, 2],
            "context": "TÀI SẢN NGẮN HẠN",
            "verified": true
          }}
        ],
        "section_confidence": 1.0,
        "hierarchy_level": 1
      }}
    ],
    "metadata": {{
      "total_sections": 3,
      "total_items": 25,
      "verification_time": "2025-01-28T...",
      "vlm_verified": true
    }}
  }},
  "confidence_score": 0.98  // 0.0 - 1.0
}}
```

**LƯU Ý:**
- Đọc KỸ ảnh gốc để kiểm tra
- Nếu dữ liệu đúng 100%, verification_status = "verified", corrections = []
- Nếu có sửa, verification_status = "corrected", liệt kê trong corrections
- Values phải là INTEGER
- Đảm bảo JSON valid

Chỉ trả về JSON, KHÔNG giải thích."""

    def __init__(self,
                 provider: str = 'anthropic',
                 model: str = None,
                 api_key: str = None):
        """
        Khởi tạo VLM Extract to CSV Module
        
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
        
        print(f"✓ VLM Extract to CSV Module initialized")
        print(f"  Provider: {provider}")
        print(f"  Model: {model}")
    
    def load_extracted_data(self, json_path: str) -> Dict[str, Any]:
        """Load extracted data từ JSON"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def encode_image(self, image_path: str) -> Tuple[str, str]:
        """Encode image to base64"""
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        ext = Path(image_path).suffix.lower()
        media_type_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.webp': 'image/webp'
        }
        media_type = media_type_map.get(ext, 'image/png')
        
        return base64_image, media_type
    
    def call_vlm(self,
                prompt: str,
                image_base64: str,
                media_type: str,
                max_tokens: int = 8192) -> str:
        """Call VLM API"""
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
        """Extract JSON từ VLM response"""
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON")
            print(f"Error: {e}")
            raise
    
    def verify_with_vlm(self,
                       extracted_json_path: str,
                       image_path: str,
                       verbose: bool = True) -> Dict[str, Any]:
        """
        Verify extracted data với VLM
        
        Args:
            extracted_json_path: Path to extracted JSON
            image_path: Path to original image
            verbose: Show logs
            
        Returns:
            Verified data với corrections
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"VLM VERIFYING: {Path(extracted_json_path).name}")
            print(f"{'='*70}")
        
        # Load extracted data
        extracted_data = self.load_extracted_data(extracted_json_path)
        
        # Prepare for prompt (remove unnecessary fields)
        data_for_prompt = {
            'report_type': extracted_data.get('report_type'),
            'company_name': extracted_data.get('company_name'),
            'report_dates': extracted_data.get('report_dates'),
            'sections': extracted_data.get('sections', [])
        }
        
        data_str = json.dumps(data_for_prompt, ensure_ascii=False, indent=2)
        
        # Create prompt
        prompt = self.VERIFICATION_PROMPT_TEMPLATE.format(
            extracted_data=data_str
        )
        
        # Encode image
        if verbose:
            print("Encoding image...")
        image_base64, media_type = self.encode_image(image_path)
        
        # Call VLM
        if verbose:
            print(f"Calling {self.provider} VLM for verification...")
        
        response = self.call_vlm(prompt, image_base64, media_type)
        
        # Parse response
        if verbose:
            print("Parsing verification results...")
        
        verification_result = self.extract_json_from_response(response)
        
        # Stats
        status = verification_result.get('verification_status', 'unknown')
        corrections = verification_result.get('corrections', [])
        confidence = verification_result.get('confidence_score', 0.0)
        
        if verbose:
            print(f"\n📊 Verification Results:")
            print(f"  Status: {status}")
            print(f"  Corrections: {len(corrections)}")
            print(f"  Confidence: {confidence:.2%}")
            
            if corrections:
                print(f"\n  Corrections made:")
                for i, corr in enumerate(corrections[:5], 1):  # Show first 5
                    print(f"    {i}. {corr.get('location')}")
                    print(f"       {corr.get('field')}: "
                          f"{corr.get('original_value')} → {corr.get('corrected_value')}")
                
                if len(corrections) > 5:
                    print(f"    ... and {len(corrections) - 5} more")
            
            print(f"{'='*70}\n")
        
        return verification_result
    
    def export_to_csv(self,
                     verified_data: Dict[str, Any],
                     output_csv_path: str,
                     include_metadata: bool = True,
                     verbose: bool = True):
        """
        Export verified data ra CSV
        
        Args:
            verified_data: Verified data (từ verify_with_vlm)
            output_csv_path: Path to CSV output
            include_metadata: Include verification metadata
            verbose: Show logs
        """
        if verbose:
            print(f"Exporting to CSV: {output_csv_path}")
        
        # Get data
        report_dates = verified_data.get('report_dates', [])
        sections = verified_data.get('sections', [])
        
        with open(output_csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['Section', 'Code', 'Item Name']
            header.extend(report_dates)
            
            if include_metadata:
                header.extend(['Context', 'Verified', 'Confidence'])
            
            writer.writerow(header)
            
            # Data rows
            for section in sections:
                section_name = section.get('section_name', '')
                
                for item in section.get('items', []):
                    row = [
                        section_name,
                        item.get('code', ''),
                        item.get('name', '')
                    ]
                    
                    # Add values
                    values = item.get('values', {})
                    for date in report_dates:
                        value = values.get(date, '')
                        row.append(value)
                    
                    # Add metadata if requested
                    if include_metadata:
                        row.append(item.get('context', section_name))
                        row.append('Yes' if item.get('verified', False) else 'No')
                        row.append(item.get('confidence', 0.0))
                    
                    writer.writerow(row)
        
        if verbose:
            print(f"✓ CSV exported successfully")
    
    def process_complete(self,
                        extracted_json_path: str,
                        image_path: str,
                        output_csv_path: str,
                        output_verified_json_path: Optional[str] = None,
                        include_metadata: bool = True,
                        verbose: bool = True) -> Dict[str, Any]:
        """
        Complete workflow: Verify JSON + Export CSV
        
        Args:
            extracted_json_path: Path to extracted JSON (from OCR or VLM Supervisor)
            image_path: Path to original image
            output_csv_path: Path to CSV output
            output_verified_json_path: Path to save verified JSON
            include_metadata: Include metadata in CSV
            verbose: Show logs
            
        Returns:
            Complete result with verification + CSV path
        """
        if verbose:
            print(f"\n{'='*70}")
            print("VLM EXTRACT TO CSV - COMPLETE PROCESS")
            print(f"{'='*70}")
        
        # Step 1: Verify with VLM
        verification_result = self.verify_with_vlm(
            extracted_json_path,
            image_path,
            verbose=verbose
        )
        
        # Step 2: Export to CSV
        verified_data = verification_result.get('verified_data', {})
        
        self.export_to_csv(
            verified_data,
            output_csv_path,
            include_metadata=include_metadata,
            verbose=verbose
        )
        
        # Step 3: Save verified JSON if requested
        if output_verified_json_path:
            with open(output_verified_json_path, 'w', encoding='utf-8') as f:
                json.dump(verification_result, f, ensure_ascii=False, indent=2)
            
            if verbose:
                print(f"✓ Verified JSON saved: {output_verified_json_path}")
        
        # Complete result
        result = {
            'verification_status': verification_result.get('verification_status'),
            'corrections_count': len(verification_result.get('corrections', [])),
            'confidence_score': verification_result.get('confidence_score', 0.0),
            'csv_path': output_csv_path,
            'verified_json_path': output_verified_json_path,
            'total_sections': len(verified_data.get('sections', [])),
            'total_items': sum(
                len(s.get('items', []))
                for s in verified_data.get('sections', [])
            )
        }
        
        if verbose:
            print(f"\n{'='*70}")
            print("✅ PROCESS COMPLETED")
            print(f"{'='*70}")
            print(f"Status: {result['verification_status']}")
            print(f"Corrections: {result['corrections_count']}")
            print(f"Confidence: {result['confidence_score']:.2%}")
            print(f"CSV: {result['csv_path']}")
            print(f"{'='*70}\n")
        
        return result


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("VLM EXTRACT TO CSV MODULE - EXAMPLE")
    print("="*70)
    
    # Initialize
    vlm_extractor = VLMExtractToCSVModule(
        provider='anthropic',
        model='claude-3-5-sonnet-20241022',
        api_key='your-api-key-here'
    )
    
    # Complete process
    result = vlm_extractor.process_complete(
        extracted_json_path='extracted_report.json',
        image_path='balance_sheet.png',
        output_csv_path='final_output.csv',
        output_verified_json_path='verified_report.json',
        include_metadata=True,
        verbose=True
    )
    
    print(f"\n✓ Final CSV ready: {result['csv_path']}")
    print(f"  Sections: {result['total_sections']}")
    print(f"  Items: {result['total_items']}")
    print(f"  Confidence: {result['confidence_score']:.2%}")
