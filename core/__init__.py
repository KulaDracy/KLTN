"""
Core module cho OCR Document Processing System
"""
import sys
from pathlib import Path

# Thêm root directory vào Python path
ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

__version__ = "1.0.0"