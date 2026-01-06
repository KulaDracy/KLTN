"""
=============================================================================
Download và Test Vietnamese OCR Model
- Force download PaddleOCR Vietnamese model
- Test với sample text
- Verify model hoạt động đúng
=============================================================================
"""

import os
import sys
from pathlib import Path

try:
    from paddleocr import PaddleOCR
    import cv2
    import numpy as np
    print("✓ Dependencies imported")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("\nInstall required packages:")
    print("  pip install paddleocr opencv-python")
    sys.exit(1)


def download_vietnamese_model():
    """Download Vietnamese OCR model"""
    print("\n" + "="*70)
    print("DOWNLOADING VIETNAMESE OCR MODEL")
    print("="*70)
    
    print("\n📥 Initializing PaddleOCR with Vietnamese model...")
    print("   This will download models (~100MB) on first run.")
    print("   Please wait...\n")
    
    try:
        # Force download Vietnamese model
        ocr = PaddleOCR(
            lang='vi',  # Vietnamese
            use_angle_cls=True,
            show_log=True,  # Show download progress
            det_model_dir=None,  # Auto download
            rec_model_dir=None,  # Auto download
            cls_model_dir=None,  # Auto download
        )
        
        print("\n✓ Vietnamese model downloaded successfully!")
        print(f"   Model location: ~/.paddleocr/")
        
        return ocr
        
    except Exception as e:
        print(f"\n❌ Failed to download model: {e}")
        return None


def create_test_image():
    """Create test image với tiếng Việt"""
    print("\n📝 Creating test image...")
    
    # Create white image
    img = np.ones((200, 800, 3), dtype=np.uint8) * 255
    
    # Add Vietnamese text
    font = cv2.FONT_HERSHEY_SIMPLEX
    texts = [
        "CÔNG TY CỔ PHẦN VIỆT NAM",
        "Tài sản ngắn hạn: 100,000,000",
        "Phải thu khách hàng: 50,000,000"
    ]
    
    y_pos = 50
    for text in texts:
        cv2.putText(img, text, (20, y_pos), font, 0.8, (0, 0, 0), 2)
        y_pos += 50
    
    # Save
    test_img_path = "test_vietnamese.png"
    cv2.imwrite(test_img_path, img)
    
    print(f"✓ Test image saved: {test_img_path}")
    return test_img_path


def test_vietnamese_ocr(ocr, image_path):
    """Test OCR với Vietnamese text"""
    print(f"\n🔍 Testing OCR on: {image_path}")
    
    try:
        # Read image
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"❌ Cannot read image: {image_path}")
            return False
        
        # Perform OCR
        print("   Running OCR...")
        result = ocr.ocr(img, cls=True)
        
        if not result or not result[0]:
            print("❌ No text detected")
            return False
        
        # Display results
        print("\n" + "="*70)
        print("OCR RESULTS")
        print("="*70)
        
        all_text = []
        for idx, line in enumerate(result[0]):
            bbox, (text, confidence) = line
            all_text.append(text)
            print(f"{idx+1}. Text: {text}")
            print(f"   Confidence: {confidence:.3f}")
            print()
        
        # Check if Vietnamese characters detected
        vietnamese_chars = ['ă', 'â', 'đ', 'ê', 'ô', 'ơ', 'ư', 
                           'á', 'à', 'ả', 'ã', 'ạ',
                           'Ă', 'Â', 'Đ', 'Ê', 'Ô', 'Ơ', 'Ư']
        
        full_text = " ".join(all_text)
        has_vietnamese = any(char in full_text for char in vietnamese_chars)
        
        if has_vietnamese:
            print("✓ Vietnamese characters detected correctly!")
            return True
        else:
            print("⚠️  Warning: No Vietnamese diacritics detected")
            print("   Model might not be recognizing Vietnamese properly")
            return False
        
    except Exception as e:
        print(f"❌ OCR test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_model_location():
    """Check where models are stored"""
    print("\n📂 Checking model locations...")
    
    home = Path.home()
    paddle_dir = home / ".paddleocr"
    
    if paddle_dir.exists():
        print(f"✓ PaddleOCR directory found: {paddle_dir}")
        
        # List model directories
        for item in paddle_dir.iterdir():
            if item.is_dir():
                print(f"  • {item.name}")
    else:
        print(f"⚠️  PaddleOCR directory not found: {paddle_dir}")
        print("   Models will be downloaded on first use")


def verify_vietnamese_support():
    """Verify Vietnamese language support"""
    print("\n🔍 Verifying Vietnamese support...")
    
    # Try to find Vietnamese config
    try:
        ocr = PaddleOCR(lang='vi', show_log=False)
        print("✓ Vietnamese language (vi) is supported")
        return True
    except Exception as e:
        print(f"❌ Vietnamese not supported: {e}")
        print("\nAvailable languages: en, ch, fr, german, korean, japan, etc.")
        return False


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download and test Vietnamese OCR model'
    )
    parser.add_argument('--test-image', help='Path to test image')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip model download (use existing)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("VIETNAMESE OCR MODEL SETUP")
    print("="*70)
    
    # Step 1: Verify Vietnamese support
    if not verify_vietnamese_support():
        print("\n❌ Cannot proceed without Vietnamese support")
        return
    
    # Step 2: Check existing models
    check_model_location()
    
    # Step 3: Download/Initialize model
    if args.skip_download:
        print("\n⏭️  Skipping download, using existing model")
        ocr = PaddleOCR(lang='vi', show_log=False)
    else:
        ocr = download_vietnamese_model()
        
        if ocr is None:
            print("\n❌ Failed to initialize OCR")
            return
    
    # Step 4: Test OCR
    if args.test_image:
        test_image_path = args.test_image
    else:
        test_image_path = create_test_image()
    
    success = test_vietnamese_ocr(ocr, test_image_path)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if success:
        print("✓ Vietnamese OCR model is working correctly!")
        print("\nYou can now run the OCR scanner:")
        print("  python ocr_scan.py /path/to/images")
    else:
        print("⚠️  Vietnamese OCR might not be working properly")
        print("\nTroubleshooting:")
        print("1. Make sure PaddleOCR is latest version:")
        print("   pip install --upgrade paddleocr")
        print("\n2. Clear cache and re-download:")
        print("   rm -rf ~/.paddleocr")
        print("   python download_vi_model.py")
        print("\n3. Check internet connection for model download")
    
    print("="*70)


if __name__ == "__main__":
    main()