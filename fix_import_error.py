"""
Script kiểm tra và sửa lỗi import
"""
from pathlib import Path

def check_and_fix():
    print("="*70)
    print("Checking project structure...")
    print("="*70)
    
    # Check directories
    dirs_to_check = [
        'app',
        'app/services',
        'config',
        'PaddleOCR'
    ]
    
    missing_dirs = []
    for dir_path in dirs_to_check:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
            print(f"❌ Missing directory: {dir_path}")
        else:
            print(f"✓ Found directory: {dir_path}")
    
    # Check __init__.py files
    init_files = [
        'app/__init__.py',
        'app/services/__init__.py',
        'config/__init__.py'
    ]
    
    missing_inits = []
    for init_file in init_files:
        if not Path(init_file).exists():
            missing_inits.append(init_file)
            print(f"❌ Missing file: {init_file}")
        else:
            print(f"✓ Found file: {init_file}")
    
    # Check key files
    key_files = [
        'app/services/paddleocr_local_service.py',
        'app/services/ocr_service_wrapper.py',
        'config/paddleocr_config.py'
    ]
    
    missing_files = []
    for file_path in key_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
            print(f"❌ Missing file: {file_path}")
        else:
            print(f"✓ Found file: {file_path}")
    
    print("\n" + "="*70)
    
    # Fix missing __init__.py files
    if missing_inits:
        print("\n🔧 Fixing missing __init__.py files...")
        for init_file in missing_inits:
            Path(init_file).parent.mkdir(parents=True, exist_ok=True)
            Path(init_file).touch()
            print(f"✓ Created: {init_file}")
        print("✅ All __init__.py files created!")
    
    # Report missing files
    if missing_dirs:
        print("\n❌ Missing directories:")
        for dir_path in missing_dirs:
            print(f"   - {dir_path}")
        print("\nPlease create these directories first!")
    
    if missing_files:
        print("\n❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease create these files using the setup script!")
    
    if not missing_dirs and not missing_files and not missing_inits:
        print("\n✅ All files and directories are in place!")
        print("You can now run: python ocr_scan.py imgs/")

if __name__ == "__main__":
    check_and_fix()