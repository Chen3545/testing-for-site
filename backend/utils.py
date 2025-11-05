import os
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
import mimetypes
import hashlib
from PIL import Image
import cv2
import numpy as np
import json

def save_uploaded_file(file, upload_folder):
    """
    儲存上傳的檔案

    Args:
        file: Flask 檔案物件
        upload_folder: 上傳資料夾路徑

    Returns:
        str: 儲存的檔案路徑
    """
    try:
        # 建立上傳資料夾
        Path(upload_folder).mkdir(parents=True, exist_ok=True)

        # 產生安全的檔名
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name, ext = os.path.splitext(filename)
        safe_filename = f"{timestamp}_{name}{ext}"

        # 儲存檔案
        file_path = Path(upload_folder) / safe_filename
        file.save(str(file_path))

        return str(file_path)

    except Exception as e:
        raise Exception(f"檔案儲存失敗: {str(e)}")

def secure_filename(filename):
    """
    產生安全的檔名，移除危險字元

    Args:
        filename: 原始檔名

    Returns:
        str: 安全的檔名
    """
    # 移除危險字元
    import re
    filename = re.sub(r'[^\w\s.-]', '', filename)
    filename = re.sub(r'[-\s]+', '-', filename)
    return filename.strip('.-')

def create_temp_directory(base_folder):
    """創建臨時目錄"""
    # 確保使用絕對路徑
    if not os.path.isabs(base_folder):
        # 如果是相對路徑，相對於專案根目錄
        project_root = Path(__file__).parent.parent.absolute()
        base_folder = project_root / base_folder

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    temp_dir = Path(base_folder) / f"temp_{timestamp}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 創建臨時目錄: {temp_dir}")
    return str(temp_dir)

def cleanup_temp_files(file_paths):
    """
    清理臨時檔案

    Args:
        file_paths: 檔案路徑列表
    """
    for path in file_paths:
        try:
            if os.path.exists(path):
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
        except Exception as e:
            print(f"警告：清理檔案失敗 {path}: {e}")

def validate_file_type(filename, expected_type):
    """
    驗證檔案類型

    Args:
        filename: 檔案名稱
        expected_type: 預期類型 ('image', 'video', 'any')

    Returns:
        bool: 檔案類型是否有效
    """
    if not filename:
        return False

    mime_type, _ = mimetypes.guess_type(filename)
    if not mime_type:
        return False

    if expected_type == 'image':
        return mime_type.startswith('image/')
    elif expected_type == 'video':
        return mime_type.startswith('video/')
    elif expected_type == 'any':
        return mime_type.startswith(('image/', 'video/'))

    return False

def get_file_info(file_path):
    """
    獲取檔案資訊

    Args:
        file_path: 檔案路徑

    Returns:
        dict: 檔案資訊
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            return {'error': '檔案不存在'}

        stat = file_path.stat()
        mime_type, _ = mimetypes.guess_type(str(file_path))

        info = {
            'filename': file_path.name,
            'size': stat.st_size,
            'size_human': format_file_size(stat.st_size),
            'mime_type': mime_type,
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'extension': file_path.suffix.lower()
        }

        # 如果是圖像檔案，獲取額外資訊
        if mime_type and mime_type.startswith('image/'):
            image_info = get_image_info(file_path)
            info.update(image_info)

        return info

    except Exception as e:
        return {'error': f'獲取檔案資訊失敗: {str(e)}'}

def get_image_info(image_path):
    """
    獲取圖像檔案詳細資訊

    Args:
        image_path: 圖像檔案路徑

    Returns:
        dict: 圖像資訊
    """
    try:
        with Image.open(image_path) as img:
            info = {
                'dimensions': f"{img.width}x{img.height}",
                'width': img.width,
                'height': img.height,
                'mode': img.mode,
                'format': img.format
            }

            # 檢查 EXIF 資訊
            if hasattr(img, '_getexif') and img._getexif():
                info['has_exif'] = True
            else:
                info['has_exif'] = False

            return info

    except Exception as e:
        return {'image_error': f'讀取圖像資訊失敗: {str(e)}'}

def format_file_size(size_bytes):
    """
    格式化檔案大小

    Args:
        size_bytes: 檔案大小（位元組）

    Returns:
        str: 格式化的檔案大小
    """
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def create_error_response(message, status_code=400, details=None):
    """
    建立錯誤回應

    Args:
        message: 錯誤訊息
        status_code: HTTP 狀態碼
        details: 額外詳細資訊

    Returns:
        tuple: (response, status_code)
    """
    response = {
        'status': 'error',
        'message': message,
        'timestamp': datetime.now().isoformat()
    }

    if details:
        response['details'] = details

    return response, status_code

def create_success_response(data, message=None):
    """
    建立成功回應

    Args:
        data: 回應資料
        message: 成功訊息

    Returns:
        dict: 成功回應
    """
    response = {
        'status': 'success',
        'timestamp': datetime.now().isoformat()
    }

    if message:
        response['message'] = message

    if isinstance(data, dict):
        response.update(data)
    else:
        response['data'] = data

    return response

def calculate_file_hash(file_path):
    """
    計算檔案 MD5 雜湊值

    Args:
        file_path: 檔案路徑

    Returns:
        str: MD5 雜湊值
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def resize_image(input_path, output_path, max_size=(1920, 1080), quality=85):
    """
    調整圖像大小

    Args:
        input_path: 輸入圖像路徑
        output_path: 輸出圖像路徑
        max_size: 最大尺寸 (width, height)
        quality: 壓縮品質 (1-100)

    Returns:
        bool: 是否成功
    """
    try:
        with Image.open(input_path) as img:
            # 計算新尺寸
            img.thumbnail(max_size, Image.Resampling.LANCZOS)

            # 儲存圖像
            save_kwargs = {'optimize': True}
            if img.format == 'JPEG':
                save_kwargs['quality'] = quality

            img.save(output_path, **save_kwargs)
            return True

    except Exception as e:
        print(f"圖像調整大小失敗: {e}")
        return False

def validate_image_pair(image1_path, image2_path):
    """
    驗證兩張圖像是否適合進行比較

    Args:
        image1_path: 第一張圖像路徑
        image2_path: 第二張圖像路徑

    Returns:
        tuple: (is_valid, message)
    """
    try:
        # 檢查檔案是否存在
        if not os.path.exists(image1_path):
            return False, f"找不到圖像檔案: {image1_path}"
        if not os.path.exists(image2_path):
            return False, f"找不到圖像檔案: {image2_path}"

        # 讀取圖像資訊
        img1_info = get_image_info(image1_path)
        img2_info = get_image_info(image2_path)

        if 'error' in img1_info or 'error' in img2_info:
            return False, "無法讀取圖像檔案"

        # 檢查尺寸差異
        size_diff_ratio = abs(img1_info['width'] * img1_info['height'] -
                             img2_info['width'] * img2_info['height']) / \
                         (img1_info['width'] * img1_info['height'])

        if size_diff_ratio > 0.5:  # 50% 差異
            return False, f"圖像尺寸差異過大: {img1_info['dimensions']} vs {img2_info['dimensions']}"

        return True, "圖像驗證通過"

    except Exception as e:
        return False, f"圖像驗證失敗: {str(e)}"

def extract_video_frames(video_path, output_dir, interval=1.0, max_frames=None):
    """
    從影片中提取影格

    Args:
        video_path: 影片檔案路徑
        output_dir: 輸出目錄
        interval: 提取間隔（秒）
        max_frames: 最大影格數量

    Returns:
        list: 提取的影格檔案路徑列表
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise Exception("無法開啟影片檔案")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval)

        frame_paths = []
        frame_count = 0
        extracted_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_filename = f"frame_{extracted_count:06d}.jpg"
                frame_path = output_dir / frame_filename

                cv2.imwrite(str(frame_path), frame)
                frame_paths.append(str(frame_path))

                extracted_count += 1
                if max_frames and extracted_count >= max_frames:
                    break

            frame_count += 1

        cap.release()
        return frame_paths

    except Exception as e:
        raise Exception(f"影片影格提取失敗: {str(e)}")

def log_operation(operation_name, parameters=None, result=None, error=None):
    """
    記錄操作日誌

    Args:
        operation_name: 操作名稱
        parameters: 操作參數
        result: 操作結果
        error: 錯誤資訊
    """
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'operation': operation_name,
        'parameters': parameters,
        'success': error is None
    }

    if result:
        log_entry['result'] = result

    if error:
        log_entry['error'] = str(error)

    # 這裡可以實作日誌儲存到檔案或資料庫
    print(f"LOG: {json.dumps(log_entry, indent=2, default=str)}")

def get_system_info():
    """
    獲取系統資訊

    Returns:
        dict: 系統資訊
    """
    import platform
    import psutil

    try:
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'disk_usage': {
                'total': psutil.disk_usage('/').total,
                'free': psutil.disk_usage('/').free
            }
        }
    except Exception as e:
        return {'error': f'獲取系統資訊失敗: {str(e)}'}
