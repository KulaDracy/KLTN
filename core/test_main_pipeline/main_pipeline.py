"""
main_pipeline.py
Complete fixed version - Ready to run
"""

import json
import os
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass

# Import modules
from ocr_scan_module import OCRScanModule
from ocr_extract_module import OCRExtractModule
from vlm_supervisor_module import VLMSupervisorModule
from vlm_extract_to_csv_module import VLMExtractToCSVModule


@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    # OCR settings
    ocr_lang: str = 'vi'
    ocr_use_gpu: bool = False
    ocr_confidence_threshold: float = 0.6
    ocr_low_confidence_threshold: float = 0.5
    ocr_use_preprocessing: bool = True
    ocr_num_workers: int = 2  # For batch processing
    
    # VLM settings
    vlm_provider: str = 'anthropic' 
    vlm_model: Optional[str] = 'claude-3-5-sonnet-20241022'
    vlm_api_key: str = ''
    
    # Pipeline settings
    enable_vlm_supervisor: bool = False
    enable_vlm_verification: bool = False
    keep_intermediate_files: bool = True
    include_csv_metadata: bool = True
    verbose: bool = True


@dataclass
class PipelineResult:
    """Pipeline result"""
    success: bool
    image_path: str
    extraction_method: str
    verification_status: str
    corrections_count: int
    confidence_score: float
    total_sections: int
    total_items: int
    files: Dict[str, str]
    processing_time: float
    error: Optional[str] = None


class MainPipeline:
    """
    Main Pipeline with proper error handling and compatibility
    """
    
    def __init__(self, config: PipelineConfig):
        """Initialize pipeline"""
        self.config = config
        
        print("="*70)
        print("INITIALIZING MAIN PIPELINE")
        print("="*70)
        
        # Module 1: OCR Scanner
        print("\n📝 Initializing OCR Scan Module...")
        try:
            self.ocr_scanner = OCRScanModule(
                lang=config.ocr_lang,
                use_gpu=config.ocr_use_gpu,
                num_workers=config.ocr_num_workers,
                confidence_threshold=config.ocr_confidence_threshold,
                low_confidence_threshold=config.ocr_low_confidence_threshold
            )
        except TypeError as e:
            # Fallback if ocr_scan_module doesn't support all parameters
            print(f"⚠️  Warning: {e}")
            print("⚠️  Using simplified initialization")
            self.ocr_scanner = OCRScanModule(
                use_gpu=config.ocr_use_gpu,
                num_workers=config.ocr_num_workers,
                confidence_threshold=config.ocr_confidence_threshold
            )
        
        # Module 2: OCR Extractor
        print("\n🔍 Initializing OCR Extract Module...")
        self.ocr_extractor = OCRExtractModule()
        
        # Module 3: VLM Supervisor (optional)
        if config.enable_vlm_supervisor:
            print("\n🤖 Initializing VLM Supervisor...")
            self.vlm_supervisor = VLMSupervisorModule(
                provider=config.vlm_provider,
                model=config.vlm_model,
                api_key=config.vlm_api_key
            )
        else:
            self.vlm_supervisor = None
            print("\n⚠️  VLM Supervisor disabled")
        
        # Module 4: VLM CSV Extractor (optional)
        if config.enable_vlm_verification:
            print("\n✅ Initializing VLM Extract to CSV...")
            self.vlm_csv_extractor = VLMExtractToCSVModule(
                provider=config.vlm_provider,
                model=config.vlm_model,
                api_key=config.vlm_api_key
            )
        else:
            self.vlm_csv_extractor = None
            print("\n⚠️  VLM Verification disabled")
        
        print("\n" + "="*70)
        print("✅ PIPELINE INITIALIZED")
        print("="*70)
    
    def process_single_image(self,
                           image_path: str,
                           output_dir: str = './output',
                           output_name: Optional[str] = None) -> PipelineResult:
        """
        Process single image through pipeline
        
        Args:
            image_path: Path to image
            output_dir: Output directory
            output_name: Output name (without extension)
            
        Returns:
            PipelineResult
        """
        start_time = time.time()
        
        # Setup
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_name = Path(image_path).stem
        if output_name is None:
            output_name = image_name
        
        files = {}
        
        if self.config.verbose:
            print(f"\n{'='*70}")
            print(f"PROCESSING: {Path(image_path).name}")
            print(f"{'='*70}\n")
        
        try:
            # ===== STAGE 1: OCR SCAN =====
            if self.config.verbose:
                print("🔍 STAGE 1: OCR SCAN")
                print("-" * 70)
            
            # Check if scan_image method exists
            if hasattr(self.ocr_scanner, 'scan_image'):
                scan_result = self.ocr_scanner.scan_image(
                    image_path=image_path,
                    use_preprocessing=self.config.ocr_use_preprocessing,
                    verbose=self.config.verbose
                )
            else:
                # Fallback: use batch with single image
                scan_results = self.ocr_scanner.scan_batch(
                    os.path.dirname(image_path),
                    verbose=False
                )
                scan_result = next((r for r in scan_results if r.image_path == image_path), None)
                if scan_result is None:
                    raise ValueError("Failed to scan image")
            
            # Save scan result
            scan_path = output_path / f"{output_name}_scan.json"
            self.ocr_scanner.save_scan_result(
                scan_result,
                str(scan_path),
                verbose=self.config.verbose
            )
            files['scan_json'] = str(scan_path)
            
            # ===== STAGE 2: OCR EXTRACT =====
            if self.config.verbose:
                print(f"\n🔍 STAGE 2: OCR EXTRACT")
                print("-" * 70)
            
            extract_path = output_path / f"{output_name}_extracted.json"
            ocr_report = self.ocr_extractor.extract(
                scan_json_path=str(scan_path),
                output_json_path=str(extract_path),
                verbose=self.config.verbose
            )
            files['extracted_json'] = str(extract_path)
            
            # ===== STAGE 3: VLM SUPERVISOR (if needed) =====
            extraction_method = 'ocr'
            final_extract_path = extract_path
            
            # Check if has_low_confidence attribute exists
            has_low_conf = getattr(scan_result, 'has_low_confidence', False)
            
            if has_low_conf and self.vlm_supervisor:
                if self.config.verbose:
                    print(f"\n🤖 STAGE 3: VLM SUPERVISOR FALLBACK")
                    print("-" * 70)
                
                vlm_path = output_path / f"{output_name}_vlm_extracted.json"
                self.vlm_supervisor.extract_with_vlm(
                    image_path=image_path,
                    output_json_path=str(vlm_path),
                    verbose=self.config.verbose
                )
                files['vlm_extracted_json'] = str(vlm_path)
                final_extract_path = vlm_path
                extraction_method = 'vlm_supervisor'
            
            # ===== STAGE 4: VLM VERIFICATION & CSV =====
            csv_path = output_path / f"{output_name}_final.csv"
            
            if self.vlm_csv_extractor:
                if self.config.verbose:
                    print(f"\n✅ STAGE 4: VLM VERIFICATION & CSV")
                    print("-" * 70)
                
                verified_json_path = output_path / f"{output_name}_verified.json"
                vlm_result = self.vlm_csv_extractor.process_complete(
                    extracted_json_path=str(final_extract_path),
                    image_path=image_path,
                    output_csv_path=str(csv_path),
                    output_verified_json_path=str(verified_json_path),
                    include_metadata=self.config.include_csv_metadata,
                    verbose=self.config.verbose
                )
                
                files['csv'] = str(csv_path)
                files['verified_json'] = str(verified_json_path)
                
                verification_status = vlm_result.get('verification_status', 'verified')
                corrections_count = vlm_result.get('corrections_count', 0)
                confidence_score = vlm_result.get('confidence_score', getattr(scan_result, 'average_confidence', 0.0))
                total_sections = vlm_result.get('total_sections', 0)
                total_items = vlm_result.get('total_items', 0)
            else:
                # No VLM verification - simple CSV export
                if self.config.verbose:
                    print(f"\n⚠️  Exporting CSV without verification")
                
                with open(final_extract_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self._simple_csv_export(data, str(csv_path))
                files['csv'] = str(csv_path)
                
                verification_status = 'no_verification'
                corrections_count = 0
                confidence_score = getattr(scan_result, 'average_confidence', 0.0)
                total_sections = len(data.get('sections', []))
                # ✅ Fixed: Proper calculation
                total_items = sum(len(s.get('items', [])) for s in data.get('sections', []))
            
            # ===== CLEANUP =====
            if not self.config.keep_intermediate_files:
                for key in ['scan_json', 'extracted_json', 'vlm_extracted_json']:
                    if key in files and files[key]:
                        try:
                            Path(files[key]).unlink()
                            files[key] = None
                        except:
                            pass
            
            # ===== RESULT =====
            processing_time = time.time() - start_time
            
            result = PipelineResult(
                success=True,
                image_path=image_path,
                extraction_method=extraction_method,
                verification_status=verification_status,
                corrections_count=corrections_count,
                confidence_score=confidence_score,
                total_sections=total_sections,
                total_items=total_items,
                files={k: v for k, v in files.items() if v is not None},
                processing_time=processing_time
            )
            
            if self.config.verbose:
                print(f"\n{'='*70}")
                print("✅ PROCESSING COMPLETED")
                print(f"{'='*70}")
                print(f"Time: {processing_time:.2f}s")
                print(f"Method: {extraction_method}")
                print(f"Verification: {verification_status}")
                print(f"Corrections: {corrections_count}")
                print(f"Confidence: {confidence_score:.2%}")
                print(f"Sections: {total_sections}")
                print(f"Items: {total_items}")
                print(f"\n📁 Output CSV: {files.get('csv')}")
                print(f"{'='*70}\n")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            if self.config.verbose:
                print(f"\n{'='*70}")
                print(f"❌ ERROR: {e}")
                import traceback
                traceback.print_exc()
                print(f"{'='*70}\n")
            
            return PipelineResult(
                success=False,
                image_path=image_path,
                extraction_method='error',
                verification_status='error',
                corrections_count=0,
                confidence_score=0.0,
                total_sections=0,
                total_items=0,
                files=files,
                processing_time=processing_time,
                error=str(e)
            )
    
    def process_single_scan_result(self,
                                  scan_result,
                                  output_dir: str) -> PipelineResult:
        """
        Process from scan result (for batch mode)
        
        Args:
            scan_result: OCRScanResult from scanner
            output_dir: Output directory
            
        Returns:
            PipelineResult
        """
        start_time = time.time()
        
        image_path = scan_result.image_path
        image_name = Path(image_path).stem
        output_path = Path(output_dir)
        files = {}
        
        try:
            # Save scan result
            scan_json_path = output_path / f"{image_name}_scan.json"
            self.ocr_scanner.save_scan_result(scan_result, str(scan_json_path))
            files['scan_json'] = str(scan_json_path)
            
            # Extract
            extract_path = output_path / f"{image_name}_extracted.json"
            ocr_report = self.ocr_extractor.extract(
                scan_json_path=str(scan_json_path),
                output_json_path=str(extract_path),
                verbose=False  # Quiet in batch
            )
            files['extracted_json'] = str(extract_path)
            
            # VLM Supervisor if needed
            extraction_method = 'ocr'
            final_extract_path = extract_path
            
            has_low_conf = getattr(scan_result, 'has_low_confidence', False)
            
            if has_low_conf and self.vlm_supervisor:
                vlm_path = output_path / f"{image_name}_vlm_extracted.json"
                self.vlm_supervisor.extract_with_vlm(
                    image_path,
                    str(vlm_path),
                    verbose=False
                )
                files['vlm_extracted_json'] = str(vlm_path)
                final_extract_path = vlm_path
                extraction_method = 'vlm_supervisor'
            
            # CSV Export
            csv_path = output_path / f"{image_name}_final.csv"
            
            if self.vlm_csv_extractor:
                verified_json = output_path / f"{image_name}_verified.json"
                vlm_res = self.vlm_csv_extractor.process_complete(
                    extracted_json_path=str(final_extract_path),
                    image_path=image_path,
                    output_csv_path=str(csv_path),
                    output_verified_json_path=str(verified_json),
                    verbose=False
                )
                files['csv'] = str(csv_path)
                v_status = vlm_res.get('verification_status', 'verified')
                corr = vlm_res.get('corrections_count', 0)
                conf = vlm_res.get('confidence_score', getattr(scan_result, 'average_confidence', 0.0))
            else:
                with open(final_extract_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._simple_csv_export(data, str(csv_path))
                files['csv'] = str(csv_path)
                v_status = 'none'
                corr = 0
                conf = getattr(scan_result, 'average_confidence', 0.0)
            
            # ✅ Fixed: Proper total_items calculation
            total_sections = len(ocr_report.sections)
            total_items = sum(len(s.items) for s in ocr_report.sections)
            
            return PipelineResult(
                success=True,
                image_path=image_path,
                extraction_method=extraction_method,
                verification_status=v_status,
                corrections_count=corr,
                confidence_score=conf,
                total_sections=total_sections,
                total_items=total_items,
                files=files,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            print(f"  ❌ Error processing {image_name}: {e}")
            return PipelineResult(
                success=False,
                image_path=image_path,
                extraction_method='error',
                verification_status='error',
                corrections_count=0,
                confidence_score=0.0,
                total_sections=0,
                total_items=0,
                files=files,
                processing_time=time.time() - start_time,
                error=str(e)
            )
    
    def process_batch(self,
                     image_folder: str,
                     output_folder: str = './output') -> List[PipelineResult]:
        """
        Batch processing with multi-threading
        
        Args:
            image_folder: Folder containing images
            output_folder: Output folder
            
        Returns:
            List of PipelineResult
        """
        # Validate
        if not os.path.exists(image_folder):
            raise FileNotFoundError(f"Image folder not found: {image_folder}")
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Stage 1: Parallel scanning
        print(f"\n🚀 STAGE 1: Multi-thread scanning {image_folder}...")
        scan_results = self.ocr_scanner.scan_batch(
            image_folder,
            verbose=self.config.verbose
        )
        
        if not scan_results:
            print("⚠️  No images scanned")
            return []
        
        # Stage 2: Extract and export
        print(f"\n🔍 STAGE 2: Extracting and exporting...")
        final_results = []
        
        for i, scan_res in enumerate(scan_results, 1):
            img_name = Path(scan_res.image_path).name
            print(f"  [{i}/{len(scan_results)}] Processing {img_name}...")
            
            res = self.process_single_scan_result(scan_res, output_folder)
            final_results.append(res)
            
            status = "✅" if res.success else "❌"
            csv_file = Path(res.files.get('csv', '')).name if res.files.get('csv') else 'N/A'
            print(f"    {status} {res.extraction_method} | CSV: {csv_file}")
        
        # Summary
        success = sum(1 for r in final_results if r.success)
        print(f"\n{'='*70}")
        print("📊 BATCH SUMMARY")
        print(f"{'='*70}")
        print(f"Total images: {len(final_results)}")
        print(f"  ✅ Success: {success}")
        print(f"  ❌ Failed: {len(final_results) - success}")
        print(f"\n📁 Output folder: {output_folder}")
        print(f"{'='*70}\n")
        
        return final_results
    
    def _simple_csv_export(self, data: Dict, csv_path: str):
        """Simple CSV export without VLM verification"""
        import csv
        
        dates = data.get('report_dates', [])
        sections = data.get('sections', [])
        
        with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Section', 'Code', 'Item Name'] + dates)
            
            for sec in sections:
                section_name = sec.get('section_name', '')
                for item in sec.get('items', []):
                    row = [
                        section_name,
                        item.get('code', ''),
                        item.get('name', '')
                    ]
                    values = item.get('values', {})
                    for date in dates:
                        row.append(values.get(date, ''))
                    writer.writerow(row)


# Example usage
if __name__ == "__main__":
    # ===== CONFIGURATION =====
    config = PipelineConfig(
        # OCR settings
        ocr_use_gpu=False,
        ocr_num_workers=2,
        ocr_confidence_threshold=0.6,
        
        # VLM settings (tắt nếu chưa có key)
        vlm_api_key='',  # Paste API key vào đây nếu muốn dùng VLM
        enable_vlm_supervisor=False,
        enable_vlm_verification=False,
        
        # Output settings
        keep_intermediate_files=True,
        verbose=True
    )
    
    # ===== INITIALIZE PIPELINE =====
    pipeline = MainPipeline(config)
    
    # ===== BATCH PROCESSING =====
    results = pipeline.process_batch(
        image_folder=r'C:\Users\lengu\Desktop\KLTN\core\files\images',  
        output_folder='./output'
    )
    
    # ===== SUMMARY =====
    print("\n" + "="*70)
    print("PROCESSING RESULTS")
    print("="*70)
    for i, res in enumerate(results, 1):
        status = "✅" if res.success else "❌"
        img_name = Path(res.image_path).name
        csv_file = Path(res.files.get('csv', '')).name if res.files.get('csv') else 'N/A'
        print(f"{i}. {status} {img_name}")
        print(f"   Method: {res.extraction_method}")
        print(f"   CSV: {csv_file}")
        if not res.success:
            print(f"   Error: {res.error}")
        print()
    
    print("✅ All done! Check ./output folder for CSV files")