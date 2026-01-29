#!/usr/bin/env python3
"""
REMOVE STAMP & SIGNATURE V3

Phiên bản 3.0 - Hỗ trợ:
- Dấu mộc màu đỏ/hồng (kể cả nhạt)
- Chữ ký màu XANH (blue ink) - FEATURE MỚI
- Chữ ký màu đen
- Multiple detection methods
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict


# ============================
# BLUE SIGNATURE DETECTION
# ============================

def create_blue_signature_mask(img, debug=False):
    """
    Detect chữ ký màu XANH (blue ink).
    
    Đây là điểm mới so với V2:
    - Detect màu xanh dương trong HSV
    - Loại trừ text đen và background
    - Filter theo shape characteristics của chữ ký
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ===== DETECT BLUE COLOR =====
    # Hue cho màu xanh dương: 85-150 độ trong HSV
    # Giảm saturation để catch cả màu xanh rất nhạt
    
    lower_blue = np.array([85, 1, 100])   # Hue: 85-150, Sat: 1+ (rất nhạt), Value: 100+
    upper_blue = np.array([150, 255, 255])
    
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # ===== LOẠI TRỪ BACKGROUND TRẮNG =====
    # Background trắng HOÀN TOÀN có value rất cao và saturation gần 0
    # Cẩn thận không loại trừ chữ ký xanh nhạt
    _, mask_not_white = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    mask_blue = cv2.bitwise_and(mask_blue, mask_not_white)
    
    # ===== LOẠI TRỪ TEXT ĐEN =====
    # Text đen có value rất thấp
    _, mask_black = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    mask_blue = cv2.bitwise_and(mask_blue, cv2.bitwise_not(mask_black))
    
    # ===== MORPHOLOGICAL OPERATIONS =====
    # Dilate để kết nối các nét chữ ký
    kernel_dilate = np.ones((2, 2), np.uint8)
    mask_blue = cv2.dilate(mask_blue, kernel_dilate, iterations=2)
    
    # Close để đóng lỗ
    kernel_close = np.ones((3, 3), np.uint8)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel_close)
    
    # Open để loại nhiễu
    kernel_open = np.ones((2, 2), np.uint8)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel_open)
    
    # ===== FILTER BY CONNECTED COMPONENTS =====
    # Chữ ký thường có area 200-15000 pixels (tăng min để loại nhiễu)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_blue, connectivity=8
    )
    
    mask_filtered = np.zeros_like(mask_blue)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 200 < area < 15000:  # Tăng min từ 50 lên 200
            mask_filtered[labels == i] = 255
    
    if debug:
        return mask_filtered, {
            'mask_blue_raw': mask_blue,
            'mask_not_white': mask_not_white,
            'mask_black': mask_black,
            'mask_filtered': mask_filtered
        }
    
    return mask_filtered


def create_black_signature_mask(img, debug=False):
    """
    Detect chữ ký màu ĐEN/TỐI.
    
    Giữ lại từ V2, nhưng có cải tiến.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # ===== DETECT DARK PIXELS =====
    # Chữ ký đen có value thấp
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 100])  # Value < 100
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
    
    # ===== EDGE DETECTION =====
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((2, 2), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Combine dark + edges
    mask_sig = cv2.bitwise_and(mask_dark, edges_dilated)
    
    # ===== MORPHOLOGY =====
    kernel = np.ones((2, 2), np.uint8)
    mask_sig = cv2.morphologyEx(mask_sig, cv2.MORPH_CLOSE, kernel)
    mask_sig = cv2.morphologyEx(mask_sig, cv2.MORPH_OPEN, kernel)
    
    # ===== FILTER BY AREA =====
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_sig, connectivity=8
    )
    
    mask_filtered = np.zeros_like(mask_sig)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 200 < area < 15000:  # Tăng min từ 50 lên 200
            mask_filtered[labels == i] = 255
    
    if debug:
        return mask_filtered, {
            'mask_dark': mask_dark,
            'edges': edges,
            'mask_filtered': mask_filtered
        }
    
    return mask_filtered


def create_signature_mask_v3(img, color='auto', debug=False):
    """
    V3: Detect chữ ký với hỗ trợ NHIỀU MÀU.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input image (BGR)
    color : str
        'auto': Tự động detect cả blue và black
        'blue': Chỉ detect blue
        'black': Chỉ detect black
    debug : bool
        Return debug info
    
    Returns:
    --------
    mask : numpy.ndarray
        Combined signature mask
    """
    if color == 'auto':
        # Detect cả hai loại
        mask_blue = create_blue_signature_mask(img, debug=False)
        mask_black = create_black_signature_mask(img, debug=False)
        
        # Combine
        mask_combined = cv2.bitwise_or(mask_blue, mask_black)
        
        if debug:
            return mask_combined, {
                'mask_blue': mask_blue,
                'mask_black': mask_black,
                'mask_combined': mask_combined
            }
        
        return mask_combined
    
    elif color == 'blue':
        return create_blue_signature_mask(img, debug=debug)
    
    elif color == 'black':
        return create_black_signature_mask(img, debug=debug)
    
    else:
        raise ValueError(f"Unknown color: {color}")


# ============================
# STAMP DETECTION (từ V2)
# ============================

def create_adaptive_stamp_mask(img, debug=False):
    """
    Detect dấu mộc màu đỏ/hồng (từ V2).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Red ranges
    lower_red1 = np.array([0, 20, 50])
    upper_red1 = np.array([20, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    
    lower_red2 = np.array([150, 20, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    mask_color = cv2.bitwise_or(mask_red1, mask_red2)
    
    # Loại trừ text đen
    _, mask_dark = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    mask_color = cv2.bitwise_and(mask_color, cv2.bitwise_not(mask_dark))
    
    # Morphology
    kernel_dilate = np.ones((3, 3), np.uint8)
    mask_color = cv2.dilate(mask_color, kernel_dilate, iterations=2)
    
    kernel_close = np.ones((5, 5), np.uint8)
    mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_CLOSE, kernel_close)
    
    kernel_open = np.ones((3, 3), np.uint8)
    mask_color = cv2.morphologyEx(mask_color, cv2.MORPH_OPEN, kernel_open)
    
    if debug:
        return mask_color, {
            'mask_red1': mask_red1,
            'mask_red2': mask_red2,
            'mask_dark': mask_dark,
            'mask_final': mask_color
        }
    
    return mask_color


# ============================
# REMOVAL FUNCTIONS V3
# ============================

def remove_stamp_v3(img, fill_method='inpaint', debug=False):
    """Remove stamp with V3."""
    mask = create_adaptive_stamp_mask(img, debug=False)
    result = apply_fill(img, mask, fill_method)
    
    if debug:
        return result, mask
    return result


def remove_signature_v3(img, color='auto', fill_method='inpaint', debug=False):
    """Remove signature with V3 - supports blue ink!"""
    mask = create_signature_mask_v3(img, color=color, debug=False)
    result = apply_fill(img, mask, fill_method)
    
    if debug:
        return result, mask
    return result


def remove_both_v3(img, sig_color='auto', fill_method='inpaint', debug=False):
    """Remove both stamp and signature with V3."""
    # Remove stamp first
    result = remove_stamp_v3(img, fill_method=fill_method)
    
    # Then remove signature
    result = remove_signature_v3(result, color=sig_color, fill_method=fill_method)
    
    if debug:
        stamp_mask = create_adaptive_stamp_mask(img)
        sig_mask = create_signature_mask_v3(result, color=sig_color)
        return result, {
            'stamp_mask': stamp_mask,
            'signature_mask': sig_mask
        }
    
    return result


def apply_fill(img, mask, method='inpaint'):
    """Apply fill method to masked regions."""
    result = img.copy()
    
    if method == 'white':
        result[mask > 0] = (255, 255, 255)
    
    elif method == 'inpaint':
        result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    
    elif method == 'blur':
        blurred = cv2.GaussianBlur(img, (15, 15), 0)
        result[mask > 0] = blurred[mask > 0]
    
    else:
        raise ValueError(f"Unknown fill method: {method}")
    
    return result


# ============================
# FILE PROCESSING V3
# ============================

def process_image_v3(
    input_path,
    output_path=None,
    remove_stamp=True,
    remove_sig=True,
    sig_color='auto',
    fill_method='inpaint',
    save_debug=False
):
    """
    Process image with V3.
    
    Parameters:
    -----------
    input_path : str or Path
    output_path : str or Path, optional
    remove_stamp : bool
    remove_sig : bool
    sig_color : str
        'auto', 'blue', 'black'
    fill_method : str
        'white', 'inpaint', 'blur'
    save_debug : bool
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")
    
    # Read
    img = cv2.imread(str(input_path))
    if img is None:
        raise ValueError(f"Cannot read: {input_path}")
    
    # Process
    if remove_stamp and remove_sig:
        result = remove_both_v3(img, sig_color=sig_color, fill_method=fill_method)
        suffix = "_clean"
    elif remove_stamp:
        result = remove_stamp_v3(img, fill_method=fill_method)
        suffix = "_no_stamp"
    elif remove_sig:
        result = remove_signature_v3(img, color=sig_color, fill_method=fill_method)
        suffix = "_no_sig"
    else:
        result = img.copy()
        suffix = "_copy"
    
    # Output path
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}{suffix}{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)
    
    # Debug
    if save_debug:
        debug_dir = output_path.parent / f"{output_path.stem}_debug"
        debug_dir.mkdir(exist_ok=True)
        
        if remove_stamp:
            stamp_mask = create_adaptive_stamp_mask(img)
            cv2.imwrite(str(debug_dir / "stamp_mask.png"), stamp_mask)
        
        if remove_sig:
            sig_mask = create_signature_mask_v3(
                result if remove_stamp else img, 
                color=sig_color
            )
            cv2.imwrite(str(debug_dir / "signature_mask.png"), sig_mask)
            
            # Also save individual color masks
            blue_mask = create_blue_signature_mask(result if remove_stamp else img)
            black_mask = create_black_signature_mask(result if remove_stamp else img)
            cv2.imwrite(str(debug_dir / "blue_signature_mask.png"), blue_mask)
            cv2.imwrite(str(debug_dir / "black_signature_mask.png"), black_mask)
        
        print(f"Debug images saved to: {debug_dir}")
    
    return output_path


# ============================
# CLI
# ============================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Remove stamp & signature V3 - Blue ink support!"
    )
    parser.add_argument("input", help="Input image")
    parser.add_argument("-o", "--output", help="Output path")
    parser.add_argument("--remove-stamp", action="store_true", default=True)
    parser.add_argument("--remove-signature", action="store_true", default=True)
    parser.add_argument(
        "--sig-color",
        choices=["auto", "blue", "black"],
        default="auto",
        help="Signature color to detect (default: auto - both)"
    )
    parser.add_argument(
        "--fill-method",
        choices=["white", "inpaint", "blur"],
        default="inpaint"
    )
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()
    
    try:
        result = process_image_v3(
            args.input,
            args.output,
            remove_stamp=args.remove_stamp,
            remove_sig=args.remove_signature,
            sig_color=args.sig_color,
            fill_method=args.fill_method,
            save_debug=args.debug
        )
        
        print(f"\n✓ V3 Success!")
        print(f"Output: {result}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import sys
        sys.exit(1)