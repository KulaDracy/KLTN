# PaddleOCR PDF Scanner

Hệ thống xử lý và xoay ảnh PDF tự động sử dụng PaddleOCR.

## Yêu cầu hệ thống

- **Python**: 3.8, 3.9, hoặc 3.10
- **Hệ điều hành**: Windows 10/11
- **RAM**: Tối thiểu 4GB (khuyên dùng 8GB)
- **Dung lượng**: ~10GB cho môi trường đầy đủ

## Cài đặt nhanh

```cmd
# 1. Tạo virtual environment
python -m venv venv

# 2. Activate
.\venv\Scripts\Activate.ps1

# 3. Cài packages
pip install -c constraints.txt -r requirements.txt

# 4. Verify
python -c "import numpy; print(numpy.__version__)"
```

## Sử dụng

### Activate môi trường

```cmd
.\venv\Scripts\Activate.ps1
```

### Chạy scanner

```cmd
python scan.py manifest.json output_dir/
```

```cmd
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "from paddleocr import PaddleOCR; print('PaddleOCR: OK')"
```

**Kết quả mong đợi:**
```
NumPy: 1.26.4
OpenCV: 4.6.0.66
PaddleOCR: OK
```

## Cấu trúc project

```
KLTN/
├── constraints.txt          # NumPy version lock
├── requirements.txt         # All packages
├── scan.py                  # Main scanner
├── SETUP.bat               # Auto setup script
├── activate.bat            # Quick activate
├── check_env.ps1           # Environment check
└── venv/                   # Virtual environment (tạo tự động)
```

## Troubleshooting

### Lỗi: NumPy version sai

```cmd
pip uninstall numpy -y
pip install -c constraints.txt numpy==1.26.4
```

### Lỗi: OpenCV import failed

```cmd
pip uninstall opencv-python opencv-python-headless -y
pip install -c constraints.txt opencv-python==4.6.0.66
```

### Lỗi: PaddleOCR không khởi tạo được

```cmd
pip uninstall paddleocr -y
pip install -c constraints.txt --no-deps paddleocr==2.7.0.3
pip install -c constraints.txt shapely scikit-image imgaug pyclipper lmdb tqdm pillow
```

## Liên hệ

Nếu gặp vấn đề, tạo issue hoặc liên hệ qua email.

## License

MIT License1.python -m venv venv
2.venv\Scripts\activate hoặc double click activate.bat ở ngoài
3.pip install --upgrade pip
4.pip install -r requirements.txt -c constraints.txt
5.Tải https://github.com/PaddlePaddle/PaddleOCR và giải nén ra và bỏ vào thư mục đang sử dung
6. copy constraint.txt và bỏ vào folder PaddleOCR
7. cd PaddleOCR
8. pip install -r requirements.txt -c constraints.txt
always run "pip check" after every install command
