from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pathlib import Path
import os
import tempfile
import shutil
from datetime import datetime
import traceback
import numpy as np
import sys
import json
import cv2

# 🔧 關鍵修正：確保專案根目錄在 Python 路徑中
WEBSITE_ROOT = Path(__file__).parent.parent  # 0809-3 專案根目錄
BACKEND_ROOT = Path(__file__).parent         # backend 目錄

# 🔧 將專案目錄加入 Python 路徑
sys.path.insert(0, str(WEBSITE_ROOT))  # 加入專案根目錄
sys.path.insert(0, str(BACKEND_ROOT))  # 加入 backend 目錄

print(f"🏠 專案根目錄: {WEBSITE_ROOT}")
print(f"📁 後端目錄: {BACKEND_ROOT}")
print(f"🔧 Python 路徑已更新")

# 現在導入模組（確保這些檔案存在於 backend/modules/ 中）
try:
    from modules.adjust_image import align_images_api, validate_alignment_parameters
    from modules.sam2_segmenter import (
        segment_image_api, segment_multiple_images_api
    )
    from modules.mask_matching import (
        match_masks_with_images_api, load_masks_from_pickle, load_masks_from_individual_files,
        validate_mask_matching_parameters
    )
    from modules.detect_change import detect_changes_with_texture_analysis
    from modules.video_processor import VideoProcessor, extract_video_frames_api

    from utils import (
        save_uploaded_file, create_temp_directory, cleanup_temp_files,
        validate_file_type, get_file_info, create_error_response,
        create_success_response
    )
    print("✅ 所有模組導入成功")
except ImportError as e:
    print(f"❌ 模組導入失敗: {e}")
    print("請確認以下檔案是否存在：")
    print("  - backend/modules/adjust_image.py")
    print("  - backend/modules/sam2_segmenter.py")
    print("  - backend/modules/mask_matching.py")
    print("  - backend/modules/detect_change.py")
    print("  - backend/utils.py")

app = Flask(__name__)
CORS(app)

# 配置 - 統一路徑管理
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['UPLOAD_FOLDER'] = str(WEBSITE_ROOT / 'results' / 'uploads')
app.config['TEMP_FOLDER'] = str(WEBSITE_ROOT / 'results' / 'temp')
app.config['RESULTS_FOLDER'] = str(WEBSITE_ROOT / 'results')
app.config['CHECKPOINT_FOLDER'] = str(WEBSITE_ROOT / 'checkpoint')  # 修正：checkpoint 不是 checkpoints
app.config['CONFIG_FOLDER'] = str(WEBSITE_ROOT / 'configs')

print(f"📁 結果目錄: {app.config['RESULTS_FOLDER']}")
print(f"📤 上傳目錄: {app.config['UPLOAD_FOLDER']}")
print(f"🤖 模型目錄: {app.config['CHECKPOINT_FOLDER']}")
print(f"⚙️ 配置目錄: {app.config['CONFIG_FOLDER']}")

# 創建必要的資料夾
for folder_path in [app.config['UPLOAD_FOLDER'], app.config['TEMP_FOLDER'], app.config['RESULTS_FOLDER']]:
    Path(folder_path).mkdir(exist_ok=True, parents=True)
    print(f"✅ 確保目錄存在: {folder_path}")

def create_success_response(data, message="操作成功"):
    """創建成功回應"""
    return jsonify({
        'status': 'success',
        'message': message,
        'data': data,
        'timestamp': datetime.now().isoformat()
    })

def create_error_response(message, status_code=400, error_details=None):
    """創建錯誤回應"""
    response_data = {
        'status': 'error',
        'message': message,
        'timestamp': datetime.now().isoformat()
    }

    if error_details:
        response_data['error_details'] = error_details

    return jsonify(response_data), status_code

def generate_objects_data(detection_result, detection_dir):
    """生成物件檢視所需的資料 - 重新設計為局部區域檢視"""
    print("🔍 生成物件檢視資料...")

    disappeared_objects = []
    appeared_objects = []

    try:
        detection_path = Path(detection_dir)

        # 檢查消失物件資料夾
        disappear_folder = detection_path / 'Disappear'
        if disappear_folder.exists():
            print(f"📁 檢查消失物件資料夾: {disappear_folder}")

            # 獲取原始圖片路徑
            image1_original = detection_path / 'image1_original.jpg'
            image2_original = detection_path / 'image2_original.jpg'

            if image1_original.exists() and image2_original.exists():
                # 獲取所有消失遮罩並按名稱排序
                mask_files = sorted(list(disappear_folder.glob('disappeared_mask_*.png')))

                # 為每個消失遮罩生成局部檢視，使用順序編號
                for idx, mask_file in enumerate(mask_files, 1):
                    # 🔧 修改：使用順序編號而不是原始檔案編號
                    sequential_number = f'{idx:03d}'  # 001, 002, 003...

                    # 生成該遮罩的局部圖像，使用順序編號
                    crop_result = generate_mask_crop_images(
                        str(image1_original),
                        str(image2_original),
                        str(mask_file),
                        detection_dir,
                        sequential_number,  # 使用順序編號
                        'disappeared'
                    )

                    if crop_result:
                        # 🔧 計算實際的遮罩統計數據
                        mask_stats = calculate_mask_statistics(str(mask_file), str(image1_original), str(image2_original))

                        disappeared_objects.append({
                            'name': f'消失物件 {sequential_number}',  # 使用順序編號顯示
                            'before_path': crop_result['before_crop'],
                            'after_path': crop_result['after_crop'],
                            'mask_path': crop_result['mask_overlay'],
                            'original_mask': f'Disappear/{mask_file.name}',
                            'confidence': mask_stats['confidence'],
                            'changeRatio': mask_stats['change_ratio'],
                            'changedPixels': mask_stats['changed_pixels'],
                            'maskArea': mask_stats['mask_area'],
                            'bbox': crop_result['bbox']
                        })

        # 檢查新增物件資料夾
        newadded_folder = detection_path / 'NewAdded'
        if newadded_folder.exists():
            print(f"📁 檢查新增物件資料夾: {newadded_folder}")

            # 獲取原始圖片路徑
            image1_original = detection_path / 'image1_original.jpg'
            image2_original = detection_path / 'image2_original.jpg'

            if image1_original.exists() and image2_original.exists():
                # 獲取所有新增遮罩並按名稱排序
                mask_files = sorted(list(newadded_folder.glob('new_mask_*.png')))

                # 為每個新增遮罩生成局部檢視，使用不同的編號範圍（從消失物件數量+1開始）
                disappeared_count = len(disappeared_objects)  # 獲取已處理的消失物件數量
                for idx, mask_file in enumerate(mask_files, disappeared_count + 1):
                    # 🔧 修改：使用接續編號，避免與消失物件衝突
                    sequential_number = f'{idx:03d}'  # 從消失物件數量+1開始編號

                    # 生成該遮罩的局部圖像，使用順序編號
                    crop_result = generate_mask_crop_images(
                        str(image1_original),
                        str(image2_original),
                        str(mask_file),
                        detection_dir,
                        sequential_number,  # 使用不衝突的順序編號
                        'appeared'
                    )

                    if crop_result:
                        # 🔧 計算實際的遮罩統計數據
                        mask_stats = calculate_mask_statistics(str(mask_file), str(image1_original), str(image2_original))

                        appeared_objects.append({
                            'name': f'新增物件 {sequential_number}',  # 使用順序編號顯示
                            'before_path': crop_result['before_crop'],
                            'after_path': crop_result['after_crop'],
                            'mask_path': crop_result['mask_overlay'],
                            'original_mask': f'NewAdded/{mask_file.name}',
                            'confidence': mask_stats['confidence'],
                            'changeRatio': mask_stats['change_ratio'],
                            'changedPixels': mask_stats['changed_pixels'],
                            'maskArea': mask_stats['mask_area'],
                            'bbox': crop_result['bbox']
                        })

        print(f"✅ 物件檢視資料生成完成:")
        print(f"   📉 消失物件: {len(disappeared_objects)} 個")
        print(f"   📈 新增物件: {len(appeared_objects)} 個")

        return {
            'disappeared': disappeared_objects,
            'appeared': appeared_objects
        }

    except Exception as e:
        print(f"❌ 生成物件檢視資料時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return {'disappeared': [], 'appeared': []}

def generate_mask_crop_images(image1_path, image2_path, mask_path, output_dir, object_id, object_type):
    """為單個遮罩生成裁切的局部圖像"""
    try:
        import cv2
        import numpy as np

        print(f"🎯 處理 {object_type} 物件: {object_id}")

        # 載入圖片和遮罩
        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image1 is None or image2 is None or mask is None:
            print(f"❌ 無法載入檔案: {image1_path}, {image2_path}, {mask_path}")
            return None

        # 找到遮罩的邊界框
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"❌ 在遮罩中找不到輪廓: {mask_path}")
            return None

        # 計算所有輪廓的總邊界框
        x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        # 🔧 動態調整裁切區域：根據遮罩大小自適應，加上適當的邊距
        original_mask_w, original_mask_h = x_max - x_min, y_max - y_min

        # 根據遮罩尺寸動態調整邊距
        # 小物件用較大比例的邊距，大物件用較小比例的邊距
        margin_ratio = max(0.3, min(0.8, 100 / max(original_mask_w, original_mask_h)))
        dynamic_margin = int(max(original_mask_w, original_mask_h) * margin_ratio)

        # 應用動態邊距到邊界框
        h_img, w_img = image1.shape[:2]
        x_min = max(0, x_min - dynamic_margin)
        y_min = max(0, y_min - dynamic_margin)
        x_max = min(w_img, x_max + dynamic_margin)
        y_max = min(h_img, y_max + dynamic_margin)

        crop_w, crop_h = x_max - x_min, y_max - y_min

        print(f"🎯 動態裁切: 遮罩尺寸({original_mask_w}x{original_mask_h}) -> 邊距({dynamic_margin}px) -> 最終裁切({crop_w}x{crop_h})")

        # 裁切圖像
        crop1 = image1[y_min:y_max, x_min:x_max]
        crop2 = image2[y_min:y_max, x_min:x_max]
        crop_mask = mask[y_min:y_max, x_min:x_max]

        # 創建輸出目錄
        crops_dir = Path(output_dir) / 'crops'
        crops_dir.mkdir(exist_ok=True)

        # 保存裁切圖像
        before_crop_path = crops_dir / f'{object_id}_before.jpg'
        after_crop_path = crops_dir / f'{object_id}_after.jpg'

        cv2.imwrite(str(before_crop_path), crop1)
        cv2.imwrite(str(after_crop_path), crop2)

        # 創建純遮罩圖像（用於疊加）
        mask_overlay_path = crops_dir / f'{object_id}_mask.png'

        # 創建RGB遮罩圖像
        mask_rgb = np.zeros((crop_mask.shape[0], crop_mask.shape[1], 3), dtype=np.uint8)

        if object_type == 'disappeared':
            # 消失物件：紅色遮罩
            mask_rgb[crop_mask > 0] = [0, 0, 255]  # BGR格式的紅色
        else:
            # 新增物件：綠色遮罩
            mask_rgb[crop_mask > 0] = [0, 255, 0]  # BGR格式的綠色

        # 保存遮罩圖像（PNG格式支援透明度）
        # 創建帶透明度的遮罩
        mask_rgba = np.zeros((crop_mask.shape[0], crop_mask.shape[1], 4), dtype=np.uint8)
        if object_type == 'disappeared':
            mask_rgba[crop_mask > 0] = [0, 0, 255, 160]  # 紅色，透明度160
        else:
            mask_rgba[crop_mask > 0] = [0, 255, 0, 160]  # 綠色，透明度160

        cv2.imwrite(str(mask_overlay_path), mask_rgba)

        print(f"✅ 成功生成局部檢視: {object_id}")
        print(f"  裁切區域: ({x_min}, {y_min}) - ({x_max}, {y_max})")
        print(f"  尺寸: {crop_w} x {crop_h}")

        return {
            'before_crop': f'crops/{object_id}_before.jpg',
            'after_crop': f'crops/{object_id}_after.jpg',
            'mask_overlay': f'crops/{object_id}_mask.png',
            'bbox': {
                'x': int(x_min),
                'y': int(y_min),
                'width': int(x_max - x_min),
                'height': int(y_max - y_min)
            }
        }

    except Exception as e:
        print(f"❌ 生成局部檢視失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_mask_statistics(mask_path, image1_path, image2_path):
    """計算遮罩的實際統計數據"""
    try:
        import cv2
        import numpy as np
        from pathlib import Path

        # 載入遮罩和圖片
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)

        if mask is None:
            print(f"❌ 無法載入遮罩: {mask_path}")
            return get_default_mask_stats()

        # 計算遮罩面積
        mask_area = np.sum(mask > 0)

        # 🔧 改進變化程度計算：基於遮罩區域內的實際像素差異
        if image1 is not None and image2 is not None:
            # 確保所有圖片尺寸一致
            if image1.shape != image2.shape or image1.shape[:2] != mask.shape:
                # 調整圖片大小以匹配遮罩
                h, w = mask.shape
                image1 = cv2.resize(image1, (w, h))
                image2 = cv2.resize(image2, (w, h))

            # 將圖片轉換為灰階以便比較
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

            # 計算兩張圖片在遮罩區域內的差異
            diff = cv2.absdiff(gray1, gray2)

            # 只考慮遮罩區域內的差異
            masked_diff = diff * (mask > 0).astype(np.uint8)

            # 計算遮罩區域內的平均差異強度
            mask_pixels = mask > 0
            if np.sum(mask_pixels) > 0:
                avg_diff = np.mean(masked_diff[mask_pixels])
                max_diff = np.max(masked_diff[mask_pixels])

                # 基於平均差異和最大差異計算變化程度
                # avg_diff 範圍通常是 0-255，我們將其轉換為 0-100%
                intensity_score = (avg_diff / 255.0) * 100
                peak_score = (max_diff / 255.0) * 100

                # 綜合評分：70% 平均差異 + 30% 峰值差異
                change_ratio = (intensity_score * 0.7 + peak_score * 0.3)

                # 根據變化強度調整範圍
                if change_ratio > 40:
                    # 強烈變化
                    change_ratio = 65 + (change_ratio - 40) * 0.5
                elif change_ratio > 20:
                    # 中等變化
                    change_ratio = 25 + (change_ratio - 20) * 2.0
                elif change_ratio > 5:
                    # 輕微變化
                    change_ratio = 8 + (change_ratio - 5) * 1.13
                else:
                    # 極輕微變化
                    change_ratio = max(1, change_ratio * 1.6)

                print(f"🔍 像素差異分析:")
                print(f"   平均差異強度: {avg_diff:.1f}/255 ({intensity_score:.1f}%)")
                print(f"   最大差異強度: {max_diff:.1f}/255 ({peak_score:.1f}%)")
                print(f"   遮罩區域像素: {np.sum(mask_pixels)}")
            else:
                change_ratio = 1  # 沒有遮罩區域的情況

        else:
            # 備用計算方法：基於遮罩密度
            total_pixels = mask.shape[0] * mask.shape[1]
            change_density = (mask_area / total_pixels) * 100
            change_ratio = min(50, max(5, change_density * 1.5))

        # 🔧 改進信心度計算：基於遮罩的複雜度和大小
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # 基於最大輪廓的面積和形狀計算信心度
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)

            # 計算輪廓的周長和面積比（衡量形狀的複雜度）
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                compactness = 4 * np.pi * contour_area / (perimeter * perimeter)
            else:
                compactness = 0

            # 基於面積大小和形狀複雜度計算信心度
            area_score = min(60, (contour_area / 1000) * 20)  # 面積越大信心度越高
            shape_score = compactness * 25  # 形狀越規則信心度越高
            base_confidence = 50 + area_score + shape_score

            confidence = max(45, min(98, base_confidence))
        else:
            confidence = 45  # 沒有明顯輪廓的情況

        # 🔧 動態變化程度範圍控制
        change_ratio = max(1, min(95, change_ratio))  # 確保在 1-95% 範圍內

        print(f"📊 遮罩統計 {Path(mask_path).name}:")
        print(f"   面積: {mask_area} 像素")
        print(f"   變化程度: {round(change_ratio, 1)}%")
        print(f"   信心度: {round(confidence)}%")

        return {
            'confidence': round(confidence),
            'change_ratio': round(change_ratio, 1),
            'changed_pixels': int(mask_area),
            'mask_area': int(mask_area)
        }

    except Exception as e:
        print(f"❌ 計算遮罩統計數據失敗: {e}")
        import traceback
        traceback.print_exc()
        return get_default_mask_stats()

def get_default_mask_stats():
    """返回預設的遮罩統計數據"""
    return {
        'confidence': 65,
        'change_ratio': 25,
        'changed_pixels': 800,
        'mask_area': 800
    }

# 🔧 直接定義：遮罩載入函數
def load_masks_from_pickle(masks_1_path, masks_2_path):
    """
    🔧 修改版：同時支援 pickle 檔案和個別遮罩檔案載入
    """
    import pickle
    import numpy as np
    import cv2
    from pathlib import Path

    try:
        print(f"🔄 載入遮罩資料...")
        print(f"  - 檔案1: {Path(masks_1_path).name}")
        print(f"  - 檔案2: {Path(masks_2_path).name}")

        # 🆕 檢查是否為 pickle 檔案或目錄
        path1 = Path(masks_1_path)
        path2 = Path(masks_2_path)

        masks_data_1 = None
        masks_data_2 = None

        # 處理第一個輸入
        if path1.is_file() and path1.suffix == '.pkl':
            # 從 pickle 檔案載入
            with open(path1, 'rb') as f:
                pickle_data_1 = pickle.load(f)

            # 🆕 檢查是否為新的 2 次分割格式
            if 'all_masks_results' in pickle_data_1:
                # 新格式：從 all_masks 目錄載入
                all_masks_dir = pickle_data_1['all_masks_results'].get('masks_directory')
                if all_masks_dir and Path(all_masks_dir).exists():
                    masks_data_1 = load_masks_from_directory(all_masks_dir)
                else:
                    print(f"⚠️ all_masks 目錄不存在，嘗試從 pickle 資料載入")
                    masks_data_1 = extract_masks_from_pickle(pickle_data_1)
            else:
                # 舊格式：直接從 pickle 載入
                masks_data_1 = extract_masks_from_pickle(pickle_data_1)

        elif path1.is_dir():
            # 直接從目錄載入
            masks_data_1 = load_masks_from_directory(str(path1))
        else:
            print(f"❌ 不支援的檔案格式: {path1}")
            return None, None

        # 處理第二個輸入（同樣邏輯）
        if path2.is_file() and path2.suffix == '.pkl':
            with open(path2, 'rb') as f:
                pickle_data_2 = pickle.load(f)

            if 'all_masks_results' in pickle_data_2:
                all_masks_dir = pickle_data_2['all_masks_results'].get('masks_directory')
                if all_masks_dir and Path(all_masks_dir).exists():
                    masks_data_2 = load_masks_from_directory(all_masks_dir)
                else:
                    masks_data_2 = extract_masks_from_pickle(pickle_data_2)
            else:
                masks_data_2 = extract_masks_from_pickle(pickle_data_2)

        elif path2.is_dir():
            masks_data_2 = load_masks_from_directory(str(path2))
        else:
            print(f"❌ 不支援的檔案格式: {path2}")
            return None, None

        if masks_data_1 is None or masks_data_2 is None:
            print(f"❌ 載入遮罩資料失敗")
            return None, None

        print(f"✅ 遮罩資料載入成功:")
        print(f"  - 檔案1 遮罩數量: {len(masks_data_1.get('masks', []))}")
        print(f"  - 檔案2 遮罩數量: {len(masks_data_2.get('masks', []))}")

        return masks_data_1, masks_data_2

    except FileNotFoundError as e:
        print(f"❌ 錯誤：找不到遮罩檔案 - {e}")
        return None, None
    except Exception as e:
        print(f"❌ 載入遮罩檔案時發生錯誤: {e}")
        return None, None

# 🔧 也需要這些輔助函數
def load_masks_from_directory(masks_dir):
    """從指定目錄載入所有遮罩檔案並計算元資料"""
    try:
        masks_path = Path(masks_dir)
        if not masks_path.exists():
            print(f"❌ 遮罩目錄不存在: {masks_dir}")
            return None

        # 找出所有遮罩檔案
        mask_files = sorted(list(masks_path.glob("mask_*.png")))

        if len(mask_files) == 0:
            print(f"⚠️ 在目錄 {masks_dir} 中找不到遮罩檔案")
            return None

        print(f"📁 在 {masks_path.name} 中找到 {len(mask_files)} 個遮罩檔案")

        masks = []
        centroids = []
        areas = []
        bboxes = []

        for i, mask_file in enumerate(mask_files):
            try:
                # 載入遮罩圖像
                mask_img = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if mask_img is None:
                    print(f"⚠️ 無法載入遮罩檔案: {mask_file.name}")
                    continue

                # 轉換為二值化遮罩
                mask = (mask_img > 127).astype(np.float32)
                masks.append(mask)

                # 計算質心
                y_coords, x_coords = np.where(mask > 0.5)
                if len(x_coords) > 0:
                    centroid_x = float(np.mean(x_coords))
                    centroid_y = float(np.mean(y_coords))
                    centroids.append((centroid_x, centroid_y))
                else:
                    centroids.append((0.0, 0.0))

                # 計算面積
                area = int(np.sum(mask > 0.5))
                areas.append(area)

                # 計算邊界框
                if len(x_coords) > 0:
                    bbox = [float(np.min(x_coords)), float(np.min(y_coords)),
                           float(np.max(x_coords)), float(np.max(y_coords))]
                else:
                    bbox = [0.0, 0.0, 0.0, 0.0]
                bboxes.append(bbox)

            except Exception as e:
                print(f"⚠️ 處理遮罩檔案 {mask_file.name} 時發生錯誤: {e}")
                continue

        if len(masks) == 0:
            print(f"❌ 沒有成功載入任何遮罩")
            return None

        # 組織資料結構
        masks_data = {
            'masks': masks,
            'centroids': centroids,
            'areas': areas,
            'bboxes': bboxes,
            'num_masks': len(masks),
            'mask_files': [str(f) for f in mask_files[:len(masks)]]
        }

        print(f"✅ 成功處理 {len(masks)} 個遮罩檔案")
        return masks_data

    except Exception as e:
        print(f"❌ 載入遮罩目錄時發生錯誤: {e}")
        return None

def extract_masks_from_pickle(pickle_data):
    """從 pickle 資料中提取遮罩資訊（支援新舊格式）"""
    try:
        # 新格式（2次分割）
        if 'all_masks_results' in pickle_data:
            # 優先使用 all_masks 目錄
            all_masks_dir = pickle_data['all_masks_results'].get('masks_directory')
            if all_masks_dir and Path(all_masks_dir).exists():
                return load_masks_from_directory(all_masks_dir)

            # 備用：從 pickle 中的 masks 陣列載入
            if 'masks' in pickle_data:
                return {
                    'masks': pickle_data['masks'],
                    'centroids': pickle_data.get('centroids', []),
                    'areas': pickle_data.get('areas', []),
                    'bboxes': pickle_data.get('bboxes', []),
                    'num_masks': len(pickle_data['masks'])
                }

        # 舊格式（直接載入）
        if 'masks' in pickle_data:
            return {
                'masks': pickle_data['masks'],
                'centroids': pickle_data.get('centroids', []),
                'areas': pickle_data.get('areas', []),
                'bboxes': pickle_data.get('bboxes', []),
                'num_masks': len(pickle_data['masks'])
            }

        print(f"⚠️ pickle 資料中找不到有效的遮罩資訊")
        return None

    except Exception as e:
        print(f"❌ 從 pickle 資料提取遮罩時發生錯誤: {e}")
        return None

# ===== 🔧 新增：運行管理系統 =====

class RunManager:
    """運行次數管理器"""

    def __init__(self, results_root):
        self.results_root = Path(results_root)
        self.runs_dir = self.results_root / "runs"
        self.current_run_file = self.results_root / "current_run.txt"
        self.history_file = self.results_root / "run_history.json"

        # 確保必要目錄存在
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ 運行管理器初始化完成: {self.runs_dir}")

    def get_next_run_number(self):
        """獲取下一個運行編號"""
        try:
            if self.current_run_file.exists():
                with open(self.current_run_file, 'r') as f:
                    current_run = int(f.read().strip())
            else:
                current_run = 0

            next_run = current_run + 1

            # 更新當前運行次數
            with open(self.current_run_file, 'w') as f:
                f.write(str(next_run))

            return next_run

        except Exception as e:
            print(f"⚠️ 獲取運行編號失敗，使用預設值: {e}")
            return 1

    def create_run_directory(self, run_number=None):
        """建立新的運行目錄"""
        if run_number is None:
            run_number = self.get_next_run_number()

        run_dir = self.runs_dir / f"run_{run_number:03d}"

        # 🔧 修改：建立6個標準子目錄
        subdirs = ['upload', 'alignment', 'sky_removal', 'segmentation', 'matching', 'detection']

        for subdir in subdirs:
            (run_dir / subdir).mkdir(parents=True, exist_ok=True)

        print(f"✅ 建立運行目錄: {run_dir}")
        print(f"📁 包含子目錄: {', '.join(subdirs)}")

        # 記錄到歷史
        self._record_run_history(run_number, run_dir)

        return str(run_dir), run_number

    def _record_run_history(self, run_number, run_dir):
        """記錄運行歷史"""
        try:
            history = []

            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)

            run_record = {
                'run_number': run_number,
                'run_directory': str(run_dir),
                'start_time': datetime.now().isoformat(),
                'status': 'started',
                'steps_completed': [],
                'files_generated': {}
            }

            history.append(run_record)

            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"⚠️ 記錄運行歷史失敗: {e}")

    def update_run_status(self, run_number, step_name, status, files_info=None):
        """更新運行狀態"""
        try:
            if not self.history_file.exists():
                return

            with open(self.history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)

            # 找到對應的運行記錄
            for record in reversed(history):
                if record['run_number'] == run_number:
                    if step_name not in record['steps_completed']:
                        record['steps_completed'].append(step_name)

                    record['last_update'] = datetime.now().isoformat()
                    record['status'] = status

                    if files_info:
                        record['files_generated'][step_name] = files_info

                    break

            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"⚠️ 更新運行狀態失敗: {e}")

    def get_current_run_info(self):
        """獲取當前運行資訊"""
        try:
            if not self.current_run_file.exists():
                return None

            with open(self.current_run_file, 'r') as f:
                current_run = int(f.read().strip())

            run_dir = self.runs_dir / f"run_{current_run:03d}"

            return {
                'run_number': current_run,
                'run_directory': str(run_dir),
                'exists': run_dir.exists()
            }

        except Exception as e:
            print(f"⚠️ 獲取當前運行資訊失敗: {e}")
            return None

    def is_valid_run(self, session_id):
        """檢查會話ID是否有效"""
        try:
            if not session_id or not session_id.startswith('run_'):
                return False

            # 直接使用 session_id 作為目錄名，不進行數字轉換
            run_dir = self.runs_dir / session_id

            return run_dir.exists()
        except Exception as e:
            print(f"⚠️ 檢查會話有效性失敗: {e}")
            return False

    def get_run_directory(self, run_number):
        """獲取指定運行的目錄路徑"""
        # 處理 run_number 可能是字符串（如 "run_77"）或整數的情況
        if isinstance(run_number, str):
            if run_number.startswith('run_'):
                # 如果已經是 "run_XXX" 格式，直接使用
                run_dir = self.runs_dir / run_number
            else:
                # 如果是純數字字符串，轉換為整數後格式化
                try:
                    num = int(run_number)
                    run_dir = self.runs_dir / f"run_{num:03d}"
                except ValueError:
                    # 如果無法轉換，直接使用原字符串
                    run_dir = self.runs_dir / f"run_{run_number}"
        else:
            # 如果是整數，正常格式化
            run_dir = self.runs_dir / f"run_{run_number:03d}"
        return str(run_dir)

    def list_all_runs(self):
        """列出所有運行"""
        runs = []

        try:
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)

                for record in history:
                    run_dir = Path(record['run_directory'])
                    record['directory_exists'] = run_dir.exists()

                    if run_dir.exists():
                        # 統計檔案數量
                        record['file_counts'] = {}
                        for subdir in ['upload', 'alignment', 'segmentation', 'matching', 'detection']:
                            subdir_path = run_dir / subdir
                            if subdir_path.exists():
                                file_count = len([f for f in subdir_path.rglob('*') if f.is_file()])
                                record['file_counts'][subdir] = file_count

                    runs.append(record)

        except Exception as e:
            print(f"⚠️ 列出運行失敗: {e}")

        return runs

# 🔧 初始化運行管理器
run_manager = RunManager(app.config['RESULTS_FOLDER'])

# 全域變數儲存工作階段資料
session_data = {}

# 🔧 新增：NumPy 數據類型轉換函式
def convert_numpy_types(obj):
    """遞歸轉換 NumPy 數據類型為 Python 原生類型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif hasattr(obj, 'item'):  # 處理其他 NumPy 標量類型
        return obj.item()
    else:
        return obj

@app.route('/api/health', methods=['GET'])
def health_check():
    """系統健康檢查"""
    return create_success_response({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

# ===== 🔧 新增：運行管理相關 API =====

@app.route('/api/start_new_run', methods=['POST'])
def start_new_run():
    """開始新的運行"""
    try:
        run_dir, run_number = run_manager.create_run_directory()

        return create_success_response({
            'run_number': run_number,
            'run_directory': run_dir,
            'message': f'開始第 {run_number} 次運行'
        })

    except Exception as e:
        return create_error_response(f'開始新運行失敗: {str(e)}', 500)

@app.route('/api/runs', methods=['GET'])
def list_runs():
    """列出所有運行"""
    try:
        runs = run_manager.list_all_runs()
        return create_success_response({
            'runs': runs,
            'total_runs': len(runs)
        })
    except Exception as e:
        return create_error_response(f'列出運行失敗: {str(e)}', 500)

# ===== 🕒 歷史回顧 API =====
@app.route('/api/history/runs', methods=['GET'])
def get_history_runs():
    """取得歷史分析縮圖列表"""
    try:
        runs_dir = Path(app.config['RESULTS_FOLDER']) / 'runs'
        if not runs_dir.exists():
            return jsonify({'runs': []})

        runs = []
        for run_name in sorted(os.listdir(runs_dir)):
            run_path = runs_dir / run_name
            if not run_path.is_dir():
                continue

            # 檢查 upload 目錄中的 image1 和 image2
            upload_path = run_path / 'upload'
            image1_path = None
            image2_path = None

            if upload_path.exists():
                # 尋找 image1 和 image2 檔案
                for file in upload_path.glob('*'):
                    if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        filename_lower = file.name.lower()
                        if 'image1' in filename_lower or filename_lower.startswith('1_') or filename_lower.endswith('_1.jpg') or filename_lower.endswith('_1.jpeg') or filename_lower.endswith('_1.png'):
                            image1_path = f'/api/files/{run_name}/upload/{file.name}'
                        elif 'image2' in filename_lower or filename_lower.startswith('2_') or filename_lower.endswith('_2.jpg') or filename_lower.endswith('_2.jpeg') or filename_lower.endswith('_2.png'):
                            image2_path = f'/api/files/{run_name}/upload/{file.name}'

                # 如果沒找到特定命名，就取前兩個圖片檔案
                if not image1_path and not image2_path:
                    image_files = sorted([f for f in upload_path.glob('*') if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
                    if len(image_files) >= 1:
                        image1_path = f'/api/files/{run_name}/upload/{image_files[0].name}'
                    if len(image_files) >= 2:
                        image2_path = f'/api/files/{run_name}/upload/{image_files[1].name}'

            # 如果找到圖片，加入列表
            if image1_path or image2_path:
                runs.append({
                    'run_id': run_name,
                    'image1_url': image1_path,
                    'image2_url': image2_path
                })

        return jsonify({'runs': runs})
    except Exception as e:
        return create_error_response(f'取得歷史運行失敗: {str(e)}', 500)

@app.route('/api/history/run/<run_id>', methods=['GET'])
def get_history_run_detail(run_id):
    """取得指定歷史分析詳細結果"""
    try:
        run_path = Path(app.config['RESULTS_FOLDER']) / 'runs' / run_id
        result_file = run_path / 'result.json'

        # 如果 result.json 存在，直接讀取
        if result_file.exists():
            with open(result_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
            return jsonify(result)

        # 如果沒有 result.json，嘗試從現有檔案重建結果
        print(f"🔍 重建 {run_id} 的分析結果...")

        # 檢查必要的目錄
        upload_dir = run_path / 'upload'
        detection_dir = run_path / 'detection'

        if not upload_dir.exists():
            return create_error_response('找不到上傳檔案', 404)

        # 重建基本結果結構
        rebuild_result = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'generated_images': [],
            'separated_images': [],
            'output_directory': str(detection_dir) if detection_dir.exists() else '',
            'mask_folders': [],
            'statistics': {},
            'report_path': '',
            'disappeared_objects': [],
            'appeared_objects': [],
            'same_objects': [],
            'visualization_images': []
        }

        # 如果有檢測結果，嘗試重建物件資料
        if detection_dir.exists():
            try:
                objects_data = generate_objects_data({'success': True}, detection_dir)
                rebuild_result['disappeared_objects'] = objects_data['disappeared']
                rebuild_result['appeared_objects'] = objects_data['appeared']

                # 查找生成的圖片
                for img_file in detection_dir.glob('*.jpg'):
                    img_path = f'/api/files/{run_id}/detection/{img_file.name}'
                    rebuild_result['generated_images'].append(img_path)
                    rebuild_result['visualization_images'].append({
                        'title': img_file.stem.replace('_', ' ').title(),
                        'description': f'檢測結果圖片: {img_file.name}',
                        'path': img_path
                    })

            except Exception as e:
                print(f"⚠️ 重建物件資料失敗: {e}")

        # 保存重建的結果供下次使用
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(rebuild_result, f, indent=2, ensure_ascii=False)
        print(f"💾 重建結果已保存到: {result_file}")

        return jsonify(rebuild_result)

    except Exception as e:
        return create_error_response(f'取得分析結果失敗: {str(e)}', 500)

@app.route('/api/run/<int:run_number>', methods=['GET'])
def get_run_details(run_number):
    """獲取特定運行的詳細資訊"""
    try:
        runs = run_manager.list_all_runs()
        run_info = next((r for r in runs if r['run_number'] == run_number), None)

        if not run_info:
            return create_error_response('找不到指定的運行', 404)

        return create_success_response(run_info)

    except Exception as e:
        return create_error_response(f'獲取運行詳細資訊失敗: {str(e)}', 500)

@app.route('/api/current_run', methods=['GET'])
def get_current_run():
    """獲取當前運行編號"""
    try:
        current_run = run_manager.get_current_run_info()

        if not current_run or not current_run.get('exists', False):
            return create_success_response({
                'run_number': None,
                'message': '目前沒有活躍的運行'
            })

        return create_success_response({
            'run_number': current_run['run_number'],
            'run_directory': current_run['run_directory'],
            'exists': current_run['exists']
        })

    except Exception as e:
        return create_error_response(f'獲取當前運行失敗: {str(e)}', 500)

# ===== 🎥 新增：影片處理相關 API =====

@app.route('/api/extract_frames', methods=['POST'])
def extract_frames():
    """從影片提取影格"""
    try:
        if 'video' not in request.files:
            return create_error_response('未提供影片檔案', 400)

        video_file = request.files['video']
        if video_file.filename == '':
            return create_error_response('未選擇檔案', 400)

        # 獲取參數
        interval_seconds = float(request.form.get('interval', 1.0))
        max_frames = int(request.form.get('max_frames', 50))
        session_id = request.form.get('session_id')  # 🔧 新增：檢查是否有現有會話

        print(f"📹 開始處理影片: {video_file.filename}")
        print(f"⏰ 提取間隔: {interval_seconds}秒, 最大影格數: {max_frames}")
        print(f"🔍 檢查現有會話: {session_id}")

        # 🔧 修正：優先使用現有會話，否則創建新會話
        if session_id and run_manager.is_valid_run(session_id):
            run_dir = run_manager.get_run_directory(session_id)
            run_number = int(session_id.replace('run_', ''))
            print(f"🔄 重用現有會話: {session_id}")
        else:
            run_dir, run_number = run_manager.create_run_directory()
            session_id = f'run_{run_number:03d}'  # 🔧 使用3位數格式保持一致
            print(f"🆕 創建新會話: {session_id}")

        # 創建影片處理目錄
        video_dir = Path(run_dir) / 'video_processing'
        video_dir.mkdir(exist_ok=True)

        # 儲存影片檔案
        video_filename = f"input_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{video_file.filename.split('.')[-1]}"
        video_path = video_dir / video_filename
        video_file.save(str(video_path))

        print(f"💾 影片已儲存: {video_path}")

        # 處理影片
        result = extract_video_frames_api(
            str(video_path),
            str(video_dir),
            interval_seconds,
            max_frames
        )

        if result['success']:
            # 更新運行狀態
            run_manager.update_run_status(run_number, 'video_processing', 'completed')

            # 🎯 簡化回應：專注於影格提取結果
            response_data = {
                'session_id': session_id,  # 🔧 新增：返回會話ID
                'run_number': run_number,
                'video_info': result['video_info'],
                'extracted_frames': len(result['frames']),
                'frames_directory': result['output_dir'],
                'frames': result['frames'],  # 完整的影格列表
                'message': f'成功提取 {len(result["frames"])} 個影格，現在可以選擇任意2個影格進行變化檢測'
            }

            print(f"✅ 影片處理完成: 提取了 {len(result['frames'])} 個影格")
            print(f"📁 影格存放位置: {result['output_dir']}")

            return create_success_response(response_data, '影格提取完成，可開始選擇影格進行分析')
        else:
            return create_error_response(result['error'], 500)

    except Exception as e:
        print(f"❌ 影片處理錯誤: {str(e)}")
        traceback.print_exc()
        return create_error_response(f'影片處理失敗: {str(e)}', 500)

@app.route('/api/video_frames/<int:run_number>', methods=['GET'])
def get_video_frames(run_number):
    """獲取指定運行的影片影格列表"""
    try:
        run_dir = Path(app.config['RESULTS_FOLDER']) / 'runs' / f'run_{run_number:03d}'
        frames_dir = run_dir / 'video_processing' / 'frames'

        if not frames_dir.exists():
            return create_error_response('找不到影片影格目錄', 404)

        # 獲取所有影格檔案
        frame_files = sorted([
            f for f in frames_dir.iterdir()
            if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ])

        frames_info = []
        for i, frame_file in enumerate(frame_files):
            # 從檔名解析時間戳
            filename = frame_file.name
            timestamp = 0.0
            if '_t' in filename and 's.' in filename:
                try:
                    timestamp_str = filename.split('_t')[1].split('s.')[0]
                    timestamp = float(timestamp_str)
                except:
                    pass

            frames_info.append({
                'index': i,
                'filename': filename,
                'path': str(frame_file),
                'timestamp': timestamp,
                'url': f'/api/files/run_{run_number:03d}/video_processing/frames/{filename}'
            })

        return create_success_response({
            'run_number': run_number,
            'total_frames': len(frames_info),
            'frames': frames_info
        })

    except Exception as e:
        return create_error_response(f'獲取影格列表失敗: {str(e)}', 500)

@app.route('/api/upload', methods=['POST'])
def upload_files():
    try:
        print("🚀 開始處理檔案上傳...")

        # 🔧 新增：檢查是否有傳入現有的session_id
        existing_session_id = request.form.get('session_id')
        print(f"🔍 收到的 session_id: {existing_session_id}")

        if existing_session_id:
            print(f"🔍 檢查會話有效性: {run_manager.is_valid_run(existing_session_id)}")

        if existing_session_id and run_manager.is_valid_run(existing_session_id):
            # 重用現有的run
            run_dir = run_manager.get_run_directory(existing_session_id)
            # 🔧 直接使用原始的session_id，不重新格式化
            session_id = existing_session_id
            # 從session_id提取數字用於update_run_status
            run_number = int(existing_session_id.replace('run_', ''))
            print(f"♻️ 重用現有運行: {session_id}")
            print(f"📁 運行目錄: {run_dir}")
        else:
            # 創建新的run
            run_dir, run_number = run_manager.create_run_directory()
            session_id = f"run_{run_number:03d}"
            print(f"✅ 創建新運行: {session_id}")
            print(f"📁 運行目錄: {run_dir}")

        # 獲取上傳目錄
        upload_dir = os.path.join(run_dir, "upload")

        # 處理檔案上傳
        ref_image = request.files.get('ref_image')
        input_image = request.files.get('input_image')

        if not ref_image or not input_image:
            raise ValueError('需要兩個圖片檔案')

        # 🔧 關鍵修改：簡化檔名為 image1.jpg 和 image2.jpg
        ref_filename = "image1.jpg"
        input_filename = "image2.jpg"

        # 儲存檔案
        ref_path = os.path.join(upload_dir, ref_filename)
        input_path = os.path.join(upload_dir, input_filename)

        ref_image.save(ref_path)
        input_image.save(input_path)

        # 🔧 更新或創建 session_data
        if session_id not in session_data:
            session_data[session_id] = {}

        session_data[session_id].update({
            'run_number': run_number,
            'run_directory': run_dir,
            'ref_image': {
                'filename': ref_filename,
                'path': ref_path
            },
            'input_image': {
                'filename': input_filename,
                'path': input_path
            }
        })

        # 更新運行狀態
        run_manager.update_run_status(run_number, 'upload', 'completed')

        print(f"📤 檔案儲存至: {upload_dir}")
        print(f"  - {ref_filename}")
        print(f"  - {input_filename}")
        print(f"🔑 session_id: {session_id}")

        return jsonify({
            'status': 'success',
            'message': '檔案上傳成功',
            'session_id': session_id,
            'run_id': session_id,
            'upload_directory': upload_dir,
            'files': {
                'ref_image': ref_filename,
                'input_image': input_filename,
                'ref_path': ref_path,
                'input_path': input_path
            }
        })

    except Exception as e:
        print(f"❌ 上傳失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/align', methods=['POST'])
def align_images():
    """圖像對齊端點 - 整合運行管理"""
    try:
        print("=" * 60)
        print("📐 圖像對齊 API 被調用")
        print(f"⏰ 時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        data = request.get_json()
        print(f"📝 接收參數: {data}")

        session_id = data.get('session_id')
        print(f"🔑 查詢工作階段 ID: {session_id}")

        # 🔧 修正：檢查 session_data 中是否有該 session_id
        if session_id not in session_data:
            print(f"❌ 可用的 session_id: {list(session_data.keys())}")
            return create_error_response(f'無效的工作階段ID: {session_id}', 400)

        session = session_data[session_id]
        run_number = session.get('run_number')
        run_dir = session.get('run_directory')

        print(f"🔢 運行編號: {run_number}")
        print(f"📁 運行目錄: {run_dir}")

        # 🔧 設定輸出到當前運行的 alignment 目錄
        output_dir = Path(run_dir) / 'alignment'
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 對齊輸出目錄: {output_dir}")

        # 🔧 修正：從 session 中獲取檔案路徑
        ref_path = session['ref_image']['path']
        input_path = session['input_image']['path']

        print(f"📁 參考圖像: {ref_path}")
        print(f"📁 輸入圖像: {input_path}")

        if not os.path.exists(ref_path) or not os.path.exists(input_path):
            return create_error_response('圖像檔案不存在於伺服器', 400)

        # 執行對齊
        print("🚀 開始執行圖像對齊...")
        result = align_images_api(
            ref_path, input_path, str(output_dir),
            data.get('pyramid_levels', 4),
            data.get('motion_type', 'EUCLIDEAN')
        )

        # 🔧 更新運行狀態
        if result.get('status') == 'success':
            files_info = [f"aligned_image: {Path(output_dir).name}"]
            run_manager.update_run_status(run_number, 'alignment', 'completed', files_info)
            print(f"✅ 運行狀態已更新 - 對齊完成")

        print(f"📊 對齊結果狀態: {result.get('status')}")
        print("=" * 60)

        serializable_result = convert_numpy_types(result)
        return jsonify(serializable_result)

    except Exception as e:
        print("=" * 60)
        print(f"💥 圖像對齊發生錯誤: {str(e)}")
        traceback.print_exc()
        print("=" * 60)
        return create_error_response(f'圖像對齊失敗: {str(e)}', 500)

@app.route('/api/remove_sky', methods=['POST'])
def remove_sky():
    """天空遮罩去除端點 - 修正版：正確使用圖片1和圖片2"""
    try:
        print("=" * 60)
        print("🌤️ 天空遮罩去除 API 被調用")
        print(f"⏰ 時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        data = request.get_json()
        session_id = data.get('session_id')
        enable_sky_removal = data.get('enable_sky_removal', True)  # 🔧 新增：天空遮罩去除開關

        print(f"🔧 天空遮罩去除設定: {enable_sky_removal}")

        if session_id not in session_data:
            return create_error_response('無效的工作階段ID', 400)

        session = session_data[session_id]
        run_number = session.get('run_number')
        run_dir = session.get('run_directory')

        # 設定輸出目錄
        output_dir = Path(run_dir) / 'sky_removal'
        output_dir.mkdir(parents=True, exist_ok=True)

        # 🔧 關鍵修正：分別取得圖片1和圖片2，不要混用
        input_image_paths = []

        # 優先使用對齊後的圖片，但要確保是正確的對應關係
        alignment_dir = Path(run_dir) / 'alignment'

        if alignment_dir.exists():
            # 🔧 修正：尋找對齊後的圖片，確保與原始圖片對應
            ref_image_name = Path(session['ref_image']['filename']).stem  # 圖片1檔名
            input_image_name = Path(session['input_image']['filename']).stem  # 圖片2檔名

            print(f"📋 尋找對應的對齊圖片:")
            print(f"  - 圖片1原檔名: {ref_image_name}")
            print(f"  - 圖片2原檔名: {input_image_name}")

            # 尋找對應的對齊結果
            aligned_image1_path = None
            aligned_image2_path = None

            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for aligned_file in alignment_dir.glob(ext):
                    file_name = aligned_file.stem
                    # 🔧 更精確的檔案匹配邏輯
                    if file_name == 'aligned_image':  # 精確匹配對齊後的圖片
                        aligned_image2_path = str(aligned_file)
                        print(f"  ✅ 找到圖片2對齊結果: {aligned_file.name}")
                    elif ref_image_name in file_name and 'aligned' not in file_name.lower():
                        # 參考圖片的對齊版本（通常參考圖片不會被對齊）
                        aligned_image1_path = str(aligned_file)
                        print(f"  ✅ 找到圖片1對齊結果: {aligned_file.name}")

            # 🔧 如果找到對齊結果，使用對齊後的圖片
            if aligned_image1_path and aligned_image2_path:
                input_image_paths = [aligned_image1_path, aligned_image2_path]
                print("📐 ✅ 使用對齊後的圖片進行天空去除")
            else:
                # 🔧 回退：尋找 aligned_image 文件
                aligned_image_file = None
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    potential_files = list(alignment_dir.glob(f'aligned_image.{ext[2:]}'))
                    if potential_files:
                        aligned_image_file = potential_files[0]
                        break

                if aligned_image_file:
                    # 使用原始ref_image + 對齊後的input_image
                    input_image_paths = [
                        session['ref_image']['path'],      # 圖片1：使用原始檔
                        str(aligned_image_file)            # 圖片2：使用正確的對齊檔
                    ]
                    print(f"📐 ⚠️ 部分對齊：圖片1使用原檔，圖片2使用對齊版本 ({aligned_image_file.name})")
                else:
                    input_image_paths = []
                    print("📐 ❌ 未找到有效的對齊文件")

        # 🔧 最終回退：使用原始上傳圖片
        if not input_image_paths:
            input_image_paths = [
                session['ref_image']['path'],    # 圖片1
                session['input_image']['path']   # 圖片2
            ]
            print("📤 ⚠️ 回退使用原始上傳圖片")

        print(f"🌤️ 最終使用的圖片:")
        print(f"  - 圖片1 (image1): {Path(input_image_paths[0]).name}")
        print(f"  - 圖片2 (image2): {Path(input_image_paths[1]).name}")

        # 🔧 驗證兩張圖片確實不同
        if Path(input_image_paths[0]).name == Path(input_image_paths[1]).name:
            print("⚠️ 警告：兩張圖片檔名相同，可能存在重複使用問題")

        # 執行天空遮罩處理
        from modules.sky_removal import remove_sky_masks_api

        result = remove_sky_masks_api(
            image1_path=input_image_paths[0],  # 🔧 確保是圖片1
            image2_path=input_image_paths[1],  # 🔧 確保是圖片2
            output_dir=str(output_dir),
            device=data.get('device', 'auto'),
            enable_sky_removal=enable_sky_removal  # 🔧 新增：傳遞天空遮罩設定
        )

        # 🔧 更新 session 中的天空去除圖片路徑
        if result.get('status') == 'success':
            result_data = result.get('data', {})

            if 'sam2_ready_files' in result_data:
                sam2_files = result_data['sam2_ready_files']

                # 🔧 修正：確保正確對應關係
                if 'image1' in sam2_files and sam2_files['image1']:
                    session['sky_removed_image1'] = {
                        'filename': Path(sam2_files['image1']).name,
                        'path': sam2_files['image1']
                    }
                    print(f"✅ 已儲存天空去除圖片1: {session['sky_removed_image1']['filename']}")

                if 'image2' in sam2_files and sam2_files['image2']:
                    session['sky_removed_image2'] = {
                        'filename': Path(sam2_files['image2']).name,
                        'path': sam2_files['image2']
                    }
                    print(f"✅ 已儲存天空去除圖片2: {session['sky_removed_image2']['filename']}")

                # 🔧 最終驗證：確保兩張天空去除圖片不同
                if (session.get('sky_removed_image1', {}).get('filename') ==
                    session.get('sky_removed_image2', {}).get('filename')):
                    print("❌ 錯誤：兩張天空去除圖片檔名相同！")
                else:
                    print("✅ 確認：兩張天空去除圖片檔名不同")

            # 更新運行狀態
            run_manager.update_run_status(run_number, 'sky_removal', 'completed')

        return jsonify(convert_numpy_types(result))

    except Exception as e:
        print(f"💥 天空遮罩去除錯誤: {str(e)}")
        traceback.print_exc()
        return create_error_response(f'天空遮罩去除失敗: {str(e)}', 500)

@app.route('/api/segment', methods=['POST'])
def segment_images():
    """SAM2 語意分割端點 - 使用天空去除後的圖片並回傳使用的圖片"""
    try:
        print("=" * 60)
        print("🤖 SAM2 分割 API 被調用")
        print(f"⏰ 時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        data = request.get_json()
        session_id = data.get('session_id')

        if session_id not in session_data:
            return create_error_response('無效的工作階段ID', 400)

        session = session_data[session_id]
        run_number = session.get('run_number')
        run_dir = session.get('run_directory')

        print(f"🔢 運行編號: {run_number}")

        # 🔧 設定輸出到當前運行的 segmentation 目錄
        output_dir = Path(run_dir) / 'segmentation'
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 分割輸出目錄: {output_dir}")

        # 🔧 關鍵修正：優先使用 session 中保存的天空去除圖片路徑
        image_paths = []
        current_image_info = []

        # 🔧 優先使用 session 中保存的天空去除圖片
        sky_removed_image1 = session.get('sky_removed_image1', {})
        sky_removed_image2 = session.get('sky_removed_image2', {})

        if sky_removed_image1.get('path') and sky_removed_image2.get('path'):
            # 使用 session 中保存的天空處理後圖片
            image_paths = [sky_removed_image1['path'], sky_removed_image2['path']]
            current_image_info = [
                {
                    'source': 'sky_processed',
                    'filename': sky_removed_image1['filename'],
                    'path': sky_removed_image1['path']
                },
                {
                    'source': 'sky_processed',
                    'filename': sky_removed_image2['filename'],
                    'path': sky_removed_image2['path']
                }
            ]
            print("🌤️ ✅ 使用 session 中的天空處理圖片進行分割")
            print(f"  - 圖片1: {sky_removed_image1['filename']}")
            print(f"  - 圖片2: {sky_removed_image2['filename']}")
        else:
            # 🔧 回退：從 sky_removal 目錄中尋找 SAM2 專用的圖片
            sky_removal_dir = Path(run_dir) / 'sky_removal'
            image1_sam2_path = sky_removal_dir / 'image1_sam2_ready.png'
            image2_sam2_path = sky_removal_dir / 'aligned_image_sam2_ready.png'

            print(f"🔍 檢查天空去除目錄圖片:")
            print(f"  - 圖片1: {image1_sam2_path}")
            print(f"  - 圖片2: {image2_sam2_path}")

            if image1_sam2_path.exists() and image2_sam2_path.exists():
                image_paths = [str(image1_sam2_path), str(image2_sam2_path)]
                current_image_info = [
                    {
                        'source': 'sky_removed_directory',
                        'filename': image1_sam2_path.name,
                        'path': str(image1_sam2_path)
                    },
                    {
                        'source': 'sky_removed_directory',
                        'filename': image2_sam2_path.name,
                        'path': str(image2_sam2_path)
                    }
                ]
                print("🌤️ ⚠️ 使用天空去除目錄的圖片")
            else:
                # 🔧 如果沒有天空去除版本，回退到其他版本
                print("⚠️ 未找到天空去除圖片，檢查其他版本...")

            # 檢查是否有對齊後的圖片
            alignment_dir = Path(run_dir) / 'alignment'
            aligned_files = []

            if alignment_dir.exists():
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    aligned_files.extend(list(alignment_dir.glob(ext)))

            if len(aligned_files) >= 2:
                aligned_files.sort()
                image_paths = [str(aligned_files[0]), str(aligned_files[1])]
                current_image_info = [
                    {
                        'source': 'aligned',
                        'filename': aligned_files[0].name,
                        'path': str(aligned_files[0])
                    },
                    {
                        'source': 'aligned',
                        'filename': aligned_files[1].name,
                        'path': str(aligned_files[1])
                    }
                ]
                print("📐 ⚠️ 使用對齊後的圖片")
            else:
                # 最後回退到原始圖片
                image_paths = [
                    session['ref_image']['path'],
                    session['input_image']['path']
                ]
                current_image_info = [
                    {
                        'source': 'original',
                        'filename': session['ref_image']['filename'],
                        'path': session['ref_image']['path']
                    },
                    {
                        'source': 'original',
                        'filename': session['input_image']['filename'],
                        'path': session['input_image']['path']
                    }
                ]
                print("📤 ⚠️ 回退到原始上傳圖片")

        # 🆕 分割前回傳當前使用的圖片資訊並複製到 segmentation 目錄
        print(f"\n📸 當前分割使用的圖片資訊:")
        for i, img_info in enumerate(current_image_info, 1):
            print(f"  圖片{i}: {img_info['filename']} (來源: {img_info['source']})")
            print(f"         路徑: {img_info['path']}")

            # 檢查檔案是否存在
            if os.path.exists(img_info['path']):
                file_size = os.path.getsize(img_info['path'])
                print(f"         狀態: ✅ 存在 ({file_size:,} bytes)")
            else:
                print(f"         狀態: ❌ 檔案不存在")

        # 🆕 複製使用的圖片到 segmentation 目錄中
        copied_images = []
        for i, img_info in enumerate(current_image_info, 1):
            try:
                source_path = Path(img_info['path'])
                if source_path.exists():
                    # 創建 image1_results 和 image2_results 目錄
                    result_dir = output_dir / f"image{i}_results"
                    result_dir.mkdir(parents=True, exist_ok=True)

                    # 複製圖片到對應目錄
                    dest_path = result_dir / f"segmentation_input_{source_path.name}"
                    shutil.copy2(source_path, dest_path)

                    copied_images.append({
                        'image_index': i,
                        'source_path': str(source_path),
                        'copied_path': str(dest_path),
                        'source_type': img_info['source']
                    })

                    print(f"📋 已複製圖片{i}到: {dest_path}")
                else:
                    print(f"❌ 無法複製圖片{i}：源檔案不存在")
            except Exception as copy_error:
                print(f"⚠️ 複製圖片{i}失敗: {copy_error}")

        # 驗證圖片檔案存在性
        valid_images = []
        for img_path in image_paths:
            if os.path.exists(img_path):
                valid_images.append(img_path)
            else:
                print(f"❌ 警告：圖片不存在 {img_path}")

        if len(valid_images) < 2:
            return create_error_response(f'分割需要至少2張有效圖片，目前只找到{len(valid_images)}張', 400)

        print(f"\n📊 最終使用的圖片數量: {len(valid_images)}")

        # 執行分割
        print("\n🚀 開始執行 SAM2 分割...")
        start_time = datetime.now()

        try:
            if len(valid_images) == 1:
                result = segment_image_api(valid_images[0], str(output_dir))
            else:
                # 🔧 使用前端傳來的參數進行分割
                api_params = {
                    'checkpoint_path': None,  # 使用預設模型
                    'device': data.get('device', 'auto'),
                    'save_individual_masks': True,
                    'enable_quality_enhancement': True
                }

                # 🆕 加入前端傳來的分割參數
                segmentation_params = {}
                if 'points_per_side' in data:
                    segmentation_params['points_per_side'] = data['points_per_side']
                if 'points_per_batch' in data:
                    segmentation_params['points_per_batch'] = data['points_per_batch']
                if 'pred_iou_thresh' in data:
                    segmentation_params['pred_iou_thresh'] = data['pred_iou_thresh']
                if 'stability_score_thresh' in data:
                    segmentation_params['stability_score_thresh'] = data['stability_score_thresh']
                if 'stability_score_offset' in data:
                    segmentation_params['stability_score_offset'] = data['stability_score_offset']
                if 'min_mask_region_area' in data:
                    segmentation_params['min_mask_region_area'] = data['min_mask_region_area']

                print(f"📋 使用前端參數: {api_params}")
                print(f"🎯 分割參數: {segmentation_params}")

                result = segment_multiple_images_api(
                    valid_images,
                    str(output_dir),
                    **api_params,
                    **segmentation_params
                )

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            # 🔧 分割完成後，將圖片來源資訊和複製資訊加入結果
            if result.get('status') == 'success' and result.get('data'):
                result['data']['images_used'] = current_image_info
                result['data']['processing_time'] = processing_time
                result['data']['copied_images'] = copied_images  # 🆕 加入複製的圖片資訊

            # 更新運行狀態
            if result.get('status') == 'success' or result.get('status') == 'partial_failure':
                total_masks = result.get('data', {}).get('total_masks_generated', 0)
                files_info = [
                    f"masks_generated: {total_masks}",
                    f"image_source: {current_image_info[0]['source']}",
                    f"copied_images: {len(copied_images)}"
                ]
                run_manager.update_run_status(run_number, 'segmentation', 'completed', files_info)
                print(f"✅ 運行狀態已更新 - 分割完成")

            print(f"\n📊 SAM2 分割完成:")
            print(f"  結果狀態: {result.get('status')}")
            print(f"  處理時間: {processing_time:.2f} 秒")
            print(f"  生成遮罩: {result.get('data', {}).get('total_masks_generated', 0)} 個")
            print(f"  圖片來源: {current_image_info[0]['source']}")
            print(f"  複製圖片: {len(copied_images)} 張")

        except Exception as segment_error:
            print(f"💥 SAM2 分割執行錯誤: {segment_error}")
            raise segment_error

        print("=" * 60)

        serializable_result = convert_numpy_types(result)
        return jsonify(serializable_result)

    except Exception as e:
        print("=" * 60)
        print(f"💥 SAM2 分割發生嚴重錯誤: {str(e)}")
        traceback.print_exc()
        print("=" * 60)
        return create_error_response(f'SAM2 分割失敗: {str(e)}', 500)

@app.route('/api/match_masks', methods=['POST'])
def match_masks():
    """遮罩匹配端點 - 支援單次分割和兩次分割的統一處理"""
    try:
        print("=" * 60)
        print("🎭 遮罩匹配 API 被調用")
        print(f"⏰ 時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        data = request.get_json()
        session_id = data.get('session_id')

        if session_id not in session_data:
            return create_error_response('無效的工作階段ID', 400)

        session = session_data[session_id]
        run_number = session.get('run_number')
        run_dir = session.get('run_directory')

        print(f"🔢 運行編號: {run_number}")

        # 設定輸出到當前運行的 matching 目錄
        output_dir = Path(run_dir) / 'matching'
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 匹配輸出目錄: {output_dir}")

        # 🔧 新增函數：統一的遮罩目錄查找邏輯
        def find_mask_directories(segmentation_dir):
            """查找並驗證遮罩目錄，支援多種格式"""

            # 查找圖像結果目錄
            image_dirs = [d for d in segmentation_dir.iterdir()
                         if d.is_dir() and d.name.startswith('image')]

            if len(image_dirs) < 2:
                print(f"❌ 圖像目錄數量不足: {len(image_dirs)} < 2")
                return None, None, None

            image_dirs.sort()  # 確保順序一致

            # 按優先順序尋找遮罩目錄類型
            mask_dir_types = [
                'single_pass_masks',  # 優先：單次分割目錄
                'all_masks'           # 備用：兩次分割目錄
            ]

            for mask_dir_type in mask_dir_types:
                masks_1_candidate = image_dirs[0] / mask_dir_type
                masks_2_candidate = image_dirs[1] / mask_dir_type

                print(f"🔍 檢查 {mask_dir_type} 目錄:")
                print(f"  - 目錄1: {masks_1_candidate} (存在: {masks_1_candidate.exists()})")
                print(f"  - 目錄2: {masks_2_candidate} (存在: {masks_2_candidate.exists()})")

                if masks_1_candidate.exists() and masks_2_candidate.exists():
                    # 驗證是否有足夠的遮罩檔案
                    mask_files_1 = list(masks_1_candidate.glob("mask_*.png"))
                    mask_files_2 = list(masks_2_candidate.glob("mask_*.png"))

                    print(f"  - 目錄1遮罩檔案: {len(mask_files_1)} 個")
                    print(f"  - 目錄2遮罩檔案: {len(mask_files_2)} 個")

                    if len(mask_files_1) >= 1 and len(mask_files_2) >= 1:
                        print(f"✅ 找到有效遮罩目錄: {mask_dir_type}")
                        return str(masks_1_candidate), str(masks_2_candidate), mask_dir_type
                    else:
                        print(f"⚠️ {mask_dir_type} 目錄存在但遮罩檔案不足")
                else:
                    print(f"⚠️ {mask_dir_type} 目錄不存在或不完整")

            print(f"❌ 找不到任何有效的遮罩目錄")
            return None, None, None

        # 查找分割結果
        segmentation_dir = Path(run_dir) / 'segmentation'

        if not segmentation_dir.exists():
            print(f"❌ 分割目錄不存在: {segmentation_dir}")
            return create_error_response('分割目錄不存在', 400)

        print(f"📁 分割目錄存在: {segmentation_dir}")

        # 使用統一的目錄查找邏輯
        masks_1_path, masks_2_path, found_mask_type = find_mask_directories(segmentation_dir)

        if masks_1_path is None or masks_2_path is None:
            # 🔧 備用方案：查找 pickle 檔案
            print("⚠️ 遮罩目錄查找失敗，嘗試尋找 pickle 檔案...")

            pickle_files = []
            for root, dirs, files in os.walk(segmentation_dir):
                for file in files:
                    if (file.endswith('_single_pass_complete.pkl') or
                        file.endswith('_two_pass_complete.pkl') or
                        file.endswith('_masks_complete.pkl')):
                        pickle_files.append(os.path.join(root, file))

            if len(pickle_files) >= 2:
                pickle_files.sort()
                masks_1_path = pickle_files[0]
                masks_2_path = pickle_files[1]
                found_mask_type = 'pickle_files'
                print(f"✅ 使用 pickle 檔案:")
                print(f"  - 檔案1: {Path(masks_1_path).name}")
                print(f"  - 檔案2: {Path(masks_2_path).name}")
            else:
                print(f"❌ 找不到足夠的 pickle 檔案: {len(pickle_files)} 個")
                return create_error_response('無法找到足夠的遮罩檔案進行匹配', 400)

        # 最終路徑驗證
        path1_exists = Path(masks_1_path).exists()
        path2_exists = Path(masks_2_path).exists()

        print(f"📁 最終路徑驗證:")
        print(f"  - 路徑1存在: {path1_exists} ({masks_1_path})")
        print(f"  - 路徑2存在: {path2_exists} ({masks_2_path})")

        if not path1_exists or not path2_exists:
            return create_error_response('遮罩檔案路徑不存在', 400)

        # 載入遮罩資料
        print(f"🔄 載入遮罩資料 (類型: {found_mask_type})...")
        masks_data_1, masks_data_2 = load_masks_from_pickle(masks_1_path, masks_2_path)

        if masks_data_1 is None or masks_data_2 is None:
            return create_error_response('載入遮罩資料失敗', 500)

        print(f"✅ 遮罩資料載入成功:")
        print(f"  - 遮罩1數量: {len(masks_data_1.get('masks', []))}")
        print(f"  - 遮罩2數量: {len(masks_data_2.get('masks', []))}")

        # 獲取圖像路徑
        image1_path = session.get('ref_image', {}).get('path')
        image2_path = session.get('input_image', {}).get('path')

        print(f"📸 圖像路徑:")
        print(f"  - 圖像1: {image1_path}")
        print(f"  - 圖像2: {image2_path}")

        # 執行匹配
        print("🚀 開始執行遮罩匹配...")

        result = match_masks_with_images_api(
            masks_data_1, masks_data_2,
            image1_path, image2_path,
            str(output_dir),
            data.get('iou_threshold', 0.2),
            data.get('distance_threshold', 50),
            data.get('similarity_threshold', 0.25)
        )

        # 更新運行狀態
        if result.get('status') == 'success':
            stats = result.get('data', {}).get('statistics', {})
            files_info = [
                f"matched: {stats.get('matched_objects', 0)}",
                f"disappeared: {stats.get('disappeared_objects', 0)}",
                f"new: {stats.get('new_objects', 0)}",
                f"mask_type: {found_mask_type}"
            ]

            run_manager.update_run_status(run_number, 'matching', 'completed', files_info)
            print(f"✅ 運行狀態已更新 - 匹配完成")

        print(f"📊 遮罩匹配結果: {result.get('status')}")
        print("=" * 60)

        serializable_result = convert_numpy_types(result)
        return jsonify(serializable_result)

    except Exception as e:
        print("=" * 60)
        print(f"💥 遮罩匹配發生錯誤: {str(e)}")
        traceback.print_exc()
        print("=" * 60)
        return create_error_response(f'遮罩匹配失敗: {str(e)}', 500)


@app.route('/api/detect_change', methods=['POST'])
def detect_change():
    try:
        print("🔍 變化檢測 API 被調用")
        print(f"⏰ 時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        data = request.get_json()
        session_id = data.get('session_id')

        # 🔧 修正：使用正確的 session_data 檢查
        if session_id not in session_data:
            return jsonify({
                'success': False,
                'error': '沒有活動的運行階段',
                'details': '請先執行上傳和分割步驟'
            }), 400

        session = session_data[session_id]
        run_number = session.get('run_number')
        run_dir = session.get('run_directory')

        print(f"🔢 運行編號: {run_number}")

        # 設定路徑
        run_dir_path = Path(run_dir)
        upload_dir = run_dir_path / 'upload'
        matching_dir = run_dir_path / 'matching'
        detection_dir = run_dir_path / 'detection'

        print(f"📁 檢測輸出目錄: {detection_dir}")

        # 確保detection目錄存在
        detection_dir.mkdir(parents=True, exist_ok=True)

        # 🔧 修正：優先使用天空去除處理後的圖片，回退到原始圖片
        image1_path = session.get('sky_removed_image1', {}).get('path') or session['ref_image']['path']
        image2_path = session.get('sky_removed_image2', {}).get('path') or session['input_image']['path']

        print(f"📁 圖片1: {Path(image1_path).name} ({'天空處理後' if 'sky_removed_image1' in session else '原始圖片'})")
        print(f"📁 圖片2: {Path(image2_path).name} ({'天空處理後' if 'sky_removed_image2' in session else '原始圖片'})")

        print("🚀 開始執行變化檢測...")

        # 呼叫新的紋理檢測函數
        result = detect_changes_with_texture_analysis(
            image1_path=image1_path,
            image2_path=image2_path,
            matching_results_path=str(matching_dir),
            detection_output_path=str(detection_dir)
        )

        if result['success']:
            # 更新運行狀態
            run_manager.update_run_status(run_number, 'detection', 'completed')

            print("✅ 運行狀態已更新 - 檢測完成")
            print(f"📊 變化檢測結果: success")

            # 生成物件檢視資料
            objects_data = generate_objects_data(result, detection_dir)

            # 📁 保存完整結果到 result.json 供歷史回顧使用
            complete_result = {
                'run_id': f'run_{run_number:03d}',
                'timestamp': datetime.now().isoformat(),
                'generated_images': result['generated_images'],
                'separated_images': result['generated_images'],
                'output_directory': str(result['output_path']),
                'mask_folders': result['mask_folders'],
                'statistics': result['statistics'],
                'report_path': str(result['report_path']),
                'disappeared_objects': objects_data['disappeared'],
                'appeared_objects': objects_data['appeared'],
                'same_objects': [],  # 如果有相同物件資料可以在這裡添加
                'visualization_images': [
                    {
                        'title': '變化檢測結果',
                        'description': '顯示圖片間的變化區域',
                        'path': f'/api/files/run_{run_number:03d}/detection/detection_result.jpg'
                    }
                ] if 'detection_result.jpg' in [Path(img).name for img in result.get('generated_images', [])] else []
            }

            result_file = run_dir_path / 'result.json'
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(complete_result, f, indent=2, ensure_ascii=False)
            print(f"💾 結果已保存到: {result_file}")

            return jsonify({
                'success': True,
                'message': '變化檢測完成',
                'data': {
                    'generated_images': result['generated_images'],
                    'separated_images': result['generated_images'],  # 🔧 向下相容
                    'output_directory': result['output_path'],
                    'mask_folders': result['mask_folders'],
                    'statistics': result['statistics'],
                    'report_path': result['report_path'],
                    'analysis_results': {
                        'disappeared_objects': objects_data['disappeared'],
                        'appeared_objects': objects_data['appeared']
                    }
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', '檢測過程發生未知錯誤'),
                'details': result
            }), 500

    except Exception as e:
        print("=" * 60)
        print(f"💥 變化檢測發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        print("=" * 60)

        return jsonify({
            'success': False,
            'error': f'變化檢測失敗: {str(e)}',
            'details': traceback.format_exc()
        }), 500

@app.route('/api/process_pipeline', methods=['POST'])
def process_pipeline():
    """完整處理流程端點 - 整合運行管理"""
    try:
        print("=" * 60)
        print("🔄 完整處理流程 API 被調用")
        print(f"⏰ 時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        data = request.get_json()
        session_id = data.get('session_id')

        if not session_id or session_id not in session_data:
            return create_error_response('無效的工作階段ID', 400)

        session = session_data[session_id]
        run_number = session.get('run_number')
        run_dir = session.get('run_directory')

        print(f"🔢 運行編號: {run_number}")
        print(f"📁 運行目錄: {run_dir}")

        results = {'pipeline_steps': []}

        try:
            # 使用運行管理的各個步驟
            # 由於每個步驟都已經整合了運行管理，這裡主要是協調流程

            # 步驟 1: 圖像對齊
            if 'ref_image' in session and 'input_image' in session:
                # 模擬調用對齊 API（實際上會通過 HTTP 調用）
                pass

            # 其他步驟類似...

            results.update({
                'status': 'success',
                'message': '完整流程執行完成',
                'run_number': run_number,
                'run_directory': run_dir,
                'total_steps': len(results['pipeline_steps'])
            })

            serializable_results = convert_numpy_types(results)
            return jsonify(serializable_results)

        except Exception as pipeline_error:
            results.update({
                'status': 'error',
                'message': '流程執行失敗',
                'error': str(pipeline_error),
                'run_number': run_number,
                'completed_steps': len(results['pipeline_steps'])
            })

            serializable_results = convert_numpy_types(results)
            return jsonify(serializable_results), 500

    except Exception as e:
        print("=" * 60)
        print(f"💥 完整流程發生嚴重錯誤: {str(e)}")
        traceback.print_exc()
        print("=" * 60)
        return create_error_response(f'流程執行失敗: {str(e)}', 500)

# 🔧 修復：檔案服務路由
@app.route('/api/files/<path:filename>')
def serve_file(filename):
    """提供檔案服務 - 支援運行目錄結構"""
    try:
        print(f"📁 檔案服務請求: {filename}")

        # 安全檢查
        if '..' in filename or filename.startswith('/'):
            return create_error_response('無效的檔案路徑', 400)

        project_root = Path(__file__).parent.parent.absolute()

        # 🔧 處理運行特定檔案路徑 (如: run_008/upload/image1.jpg)
        if '/' in filename:
            parts = filename.split('/')
            if len(parts) >= 3 and parts[0].startswith('run_'):
                run_id = parts[0]
                subdir = parts[1]
                file_name = '/'.join(parts[2:])  # 支援多層嵌套

                target_path = project_root / 'results' / 'runs' / run_id / subdir / file_name
                print(f"🔍 運行特定檔案: {target_path}")

                if target_path.exists() and target_path.is_file():
                    print(f"✅ 找到運行檔案: {target_path}")

                    # 設定 MIME 類型
                    mimetype = None
                    if file_name.lower().endswith(('.jpg', '.jpeg')):
                        mimetype = 'image/jpeg'
                    elif file_name.lower().endswith('.png'):
                        mimetype = 'image/png'

                    return send_file(str(target_path), mimetype=mimetype)
        if filename.startswith('results/'):
            full_path = project_root / filename
            print(f"🔍 嘗試載入檔案: {full_path}")

            if full_path.exists() and full_path.is_file():
                print(f"✅ 找到檔案: {full_path}")

                # 設定 MIME 類型
                mimetype = None
                if filename.lower().endswith(('.jpg', '.jpeg')):
                    mimetype = 'image/jpeg'
                elif filename.lower().endswith('.png'):
                    mimetype = 'image/png'

                return send_file(str(full_path), mimetype=mimetype)

        # 🔧 備用：搜尋運行目錄（適用於相對路徑）
        runs_dir = project_root / 'results' / 'runs'
        if runs_dir.exists():
            # 按時間排序，搜尋最新的運行目錄
            run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith('run_')],
                            key=lambda x: x.stat().st_ctime, reverse=True)

            for run_dir in run_dirs[:5]:  # 搜尋最新的5個運行
                for subdir in ['upload', 'alignment', 'segmentation', 'matching', 'detection', 'video_processing']:
                    search_path = run_dir / subdir
                    if search_path.exists():
                        target_file = search_path / filename
                        if target_file.exists() and target_file.is_file():
                            print(f"✅ 在運行目錄中找到檔案: {target_file}")

                            # 設定 MIME 類型
                            mimetype = None
                            if filename.lower().endswith(('.jpg', '.jpeg')):
                                mimetype = 'image/jpeg'
                            elif filename.lower().endswith('.png'):
                                mimetype = 'image/png'

                            return send_file(str(target_file), mimetype=mimetype)

                        # 🎥 新增：檢查video_processing/frames子目錄
                        if subdir == 'video_processing':
                            frames_path = search_path / 'frames'
                            if frames_path.exists():
                                target_file = frames_path / filename
                                if target_file.exists() and target_file.is_file():
                                    print(f"✅ 在影格目錄中找到檔案: {target_file}")

                                    # 設定 MIME 類型
                                    mimetype = None
                                    if filename.lower().endswith(('.jpg', '.jpeg')):
                                        mimetype = 'image/jpeg'
                                    elif filename.lower().endswith('.png'):
                                        mimetype = 'image/png'

                                    return send_file(str(target_file), mimetype=mimetype)

        # 備用：搜尋原有的 temp 目錄結構
        results_dir = project_root / 'results'
        if results_dir.exists():
            temp_dirs = sorted([d for d in results_dir.iterdir()
                              if d.is_dir() and d.name.startswith('temp_')],
                             key=lambda x: x.stat().st_ctime, reverse=True)

            for temp_dir in temp_dirs[:5]:
                target_file = temp_dir / filename
                if target_file.exists() and target_file.is_file():
                    print(f"✅ 在臨時目錄中找到檔案: {target_file}")
                    return send_file(str(target_file), mimetype='image/jpeg')

        return create_error_response('檔案不存在或無法存取', 404)

    except Exception as e:
        print(f"💥 檔案服務發生嚴重錯誤: {str(e)}")
        return create_error_response(f'檔案服務失敗: {str(e)}', 500)

@app.route('/api/cleanup', methods=['POST'])
def cleanup_session():
    """清理工作階段"""
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id')

        if session_id and session_id in session_data:
            session = session_data[session_id]
            run_number = session.get('run_number')

            # 更新運行狀態為已清理
            if run_number:
                run_manager.update_run_status(run_number, 'cleanup', 'completed')

            # 清理檔案
            for key in session:
                if isinstance(session[key], dict) and 'path' in session[key]:
                    cleanup_temp_files([session[key]['path']])
                elif isinstance(session[key], list):
                    paths = [item['path'] for item in session[key] if 'path' in item]
                    cleanup_temp_files(paths)

            # 移除工作階段
            del session_data[session_id]

        return create_success_response({'message': '工作階段清理完成'})

    except Exception as e:
        return create_error_response(f'清理失敗: {str(e)}', 500)

@app.errorhandler(413)
def too_large(e):
    return create_error_response('檔案太大，請上傳小於100MB的檔案', 413)

@app.errorhandler(404)
def not_found(e):
    return create_error_response('API端點不存在', 404)

@app.errorhandler(500)
def internal_error(e):
    return create_error_response('伺服器內部錯誤', 500)

def validate_alignment_parameters_enhanced(data, session_data):
    """增強版參數驗證，優先檢查 session_id"""
    session_id = data.get('session_id')

    if session_id:
        if session_id not in session_data:
            return False, f"無效的工作階段ID: {session_id}"

        session = session_data[session_id]

        if 'ref_image' not in session:
            return False, "工作階段資料不完整：缺少參考圖像"
        if 'input_image' not in session:
            return False, "工作階段資料不完整：缺少輸入圖像"

        ref_path = session['ref_image'].get('path')
        input_path = session['input_image'].get('path')

        if not ref_path or not input_path:
            return False, "工作階段資料不完整：圖像路徑為空"

        return True, None

    if not data.get('ref_path') or not data.get('input_path'):
        return False, "缺少必要參數: 需要 session_id 或 ref_path/input_path"

    return True, None

if __name__ == '__main__':
    print("🚀 照片變化檢測系統後端啟動 (整合運行管理)")
    print("📍 API 基礎 URL: http://127.0.0.1:5000/api")
    print("🔧 支援功能:")
    print("  - 開始新運行: /api/start_new_run")
    print("  - 列出運行: /api/runs")
    print("  - 運行詳情: /api/run/<run_number>")
    print("  - 檔案上傳: /api/upload")
    print("  - 圖像對齊: /api/align")
    print("  - SAM2 分割: /api/segment")
    print("  - 遮罩匹配: /api/match_masks")
    print("  - 變化檢測: /api/detect_change")
    print("  - 完整流程: /api/process_pipeline")
    print("  - 影片處理: /api/extract_frames")
    print("  - 檔案服務: /api/files/<filename>")
    print("📊 系統健康: /api/health")
    print("🗂️ 運行管理已啟用 - 系統化管理所有處理結果")
    print("📁 運行目錄結構: results/runs/run_XXX/[upload|alignment|segmentation|matching|detection]")

    app.run(debug=True, host='0.0.0.0', port=5000)
