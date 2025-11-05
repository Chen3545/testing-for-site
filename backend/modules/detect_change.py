import cv2
import numpy as np
import os
from pathlib import Path
import json
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# 導入 ViT 相關模組
try:
    from transformers import ViTModel, ViTConfig
    VIT_AVAILABLE = True
except ImportError:
    print("⚠️ 警告：無法導入 transformers，將使用備用特徵提取方法")
    VIT_AVAILABLE = False

class ViTTextureExtractor:
    """基於 Vision Transformer 的紋理特徵提取器"""

    def __init__(self, device='auto'):
        """初始化 ViT 紋理特徵提取器"""
        self.device = self._setup_device(device)
        self.model = None
        self.transform = None
        self._load_model()
        print(f"🤖 ViT 紋理提取器已初始化 (設備: {self.device})")

    def _setup_device(self, device):
        """設定計算設備"""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)

    def _load_model(self):
        """載入 ViT 預訓練模型"""
        try:
            if VIT_AVAILABLE:
                # 使用 ViT-Base 模型
                model_name = 'google/vit-base-patch16-224'
                self.model = ViTModel.from_pretrained(model_name)
                self.model.to(self.device)
                self.model.eval()

                # 設定圖像預處理
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
                print("✅ ViT 模型載入成功")
            else:
                # 備用方案：使用傳統 CNN 特徵
                self._load_fallback_model()
        except Exception as e:
            print(f"⚠️ ViT 模型載入失敗，使用備用方案: {e}")
            self._load_fallback_model()

    def _load_fallback_model(self):
        """備用方案：使用傳統特徵提取"""
        print("🔧 使用備用特徵提取方法")
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def extract_region_features(self, image, mask):
        """提取遮罩區域的紋理特徵"""
        try:
            # 提取遮罩區域
            masked_region = self._extract_masked_region(image, mask)
            if masked_region is None:
                return np.zeros(768)  # ViT-Base 的特徵維度

            # 轉換為 PIL 圖像
            if len(masked_region.shape) == 3:
                masked_region_rgb = cv2.cvtColor(masked_region, cv2.COLOR_BGR2RGB)
            else:
                masked_region_rgb = cv2.cvtColor(masked_region, cv2.COLOR_GRAY2RGB)

            pil_image = Image.fromarray(masked_region_rgb)

            # 預處理
            input_tensor = self.transform(pil_image).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)

            # 特徵提取
            with torch.no_grad():
                if self.model is not None and VIT_AVAILABLE:
                    # 使用 ViT 提取特徵
                    outputs = self.model(input_tensor)
                    features = outputs.last_hidden_state.mean(dim=1).squeeze()  # 全局平均池化
                    features = features.cpu().numpy()
                else:
                    # 備用：使用傳統方法
                    features = self._extract_traditional_features(masked_region)

            return features

        except Exception as e:
            print(f"⚠️ 特徵提取失敗: {e}")
            return np.zeros(768)

    def _extract_masked_region(self, image, mask):
        """提取遮罩區域並生成有效的圖像塊"""
        # 找到遮罩邊界
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0:
            return None

        # 計算邊界框
        padding = 20
        y_min = max(0, y_indices.min() - padding)
        y_max = min(image.shape[0], y_indices.max() + padding)
        x_min = max(0, x_indices.min() - padding)
        x_max = min(image.shape[1], x_indices.max() + padding)

        # 確保最小尺寸
        min_size = 32
        if (y_max - y_min) < min_size or (x_max - x_min) < min_size:
            # 擴展到最小尺寸
            y_center = (y_min + y_max) // 2
            x_center = (x_min + x_max) // 2
            y_min = max(0, y_center - min_size // 2)
            y_max = min(image.shape[0], y_center + min_size // 2)
            x_min = max(0, x_center - min_size // 2)
            x_max = min(image.shape[1], x_center + min_size // 2)

        # 裁切區域
        region = image[y_min:y_max, x_min:x_max]
        return region

    def _extract_traditional_features(self, region):
        """備用：傳統紋理特徵提取"""
        try:
            # 轉換為灰階
            if len(region.shape) == 3:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                gray = region

            # 調整大小
            gray = cv2.resize(gray, (64, 64))

            # 計算多種紋理特徵
            features = []

            # 1. LBP (Local Binary Pattern)
            lbp = self._calculate_lbp(gray)
            features.extend(lbp.flatten()[:256])  # 限制長度

            # 2. GLCM 特徵
            glcm_features = self._calculate_glcm_features(gray)
            features.extend(glcm_features)

            # 3. Gabor 濾波器響應
            gabor_features = self._calculate_gabor_features(gray)
            features.extend(gabor_features)

            # 填充到 768 維
            while len(features) < 768:
                features.append(0.0)

            return np.array(features[:768])

        except Exception as e:
            print(f"⚠️ 傳統特徵提取失敗: {e}")
            return np.zeros(768)

    def _calculate_lbp(self, gray):
        """計算 Local Binary Pattern"""
        try:
            # 簡化的 LBP 實現
            rows, cols = gray.shape
            lbp = np.zeros_like(gray)

            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    center = gray[i, j]
                    code = 0
                    # 8-neighborhood
                    neighbors = [
                        gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                        gray[i, j+1], gray[i+1, j+1], gray[i+1, j],
                        gray[i+1, j-1], gray[i, j-1]
                    ]

                    for k, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            code |= (1 << k)

                    lbp[i, j] = code

            return lbp
        except Exception:
            return np.zeros((8, 8))

    def _calculate_glcm_features(self, gray):
        """計算 GLCM 特徵"""
        try:
            # 簡化的 GLCM 特徵
            features = []

            # 計算基本統計特徵
            features.append(float(np.mean(gray)))
            features.append(float(np.std(gray)))
            features.append(float(np.var(gray)))

            # 計算梯度特徵
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            features.append(float(np.mean(np.abs(grad_x))))
            features.append(float(np.mean(np.abs(grad_y))))

            return features[:20]  # 限制特徵數量
        except Exception:
            return [0.0] * 20

    def _calculate_gabor_features(self, gray):
        """計算 Gabor 濾波器特徵"""
        try:
            features = []

            # 多個方向的 Gabor 濾波器
            for theta in [0, 45, 90, 135]:
                kernel = cv2.getGaborKernel((21, 21), 5, np.radians(theta),
                                          2*np.pi*0.5, 0.5, 0, ktype=cv2.CV_32F)
                filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                features.append(float(np.mean(filtered)))
                features.append(float(np.std(filtered)))

            return features[:16]  # 限制特徵數量
        except Exception:
            return [0.0] * 16

    def calculate_similarity(self, features1, features2):
        """計算兩個特徵向量的相似度"""
        try:
            # 確保特徵向量有效
            if np.allclose(features1, 0) or np.allclose(features2, 0):
                return 0.0

            # 正規化特徵向量
            features1_norm = features1 / (np.linalg.norm(features1) + 1e-8)
            features2_norm = features2 / (np.linalg.norm(features2) + 1e-8)

            # 計算餘弦相似度
            similarity = np.dot(features1_norm, features2_norm)

            # 確保在 [0, 1] 範圍內
            similarity = max(0.0, min(1.0, (similarity + 1) / 2))

            return float(similarity)

        except Exception as e:
            print(f"⚠️ 相似度計算失敗: {e}")
            return 0.0

class MaskReclassifier:
    """遮罩二次重新分類器 - 整合視覺差異驗證"""

    def __init__(self, feature_extractor):
        """初始化重新分類器"""
        self.feature_extractor = feature_extractor

        # 🔧 修改為更嚴格的閾值
        self.reclassification_params = {
            'similarity_threshold_for_same': 0.60,      # 🔧 降低閾值
            'similarity_threshold_for_different': 0.50,  # 🔧 降低閾值
            'brightness_tolerance': 0.15,               # 增加亮度容忍度
            'visual_difference_threshold': 0.12,        # 🆕 視覺差異閾值
            'brightness_penalty_threshold': 0.15,       # 🆕 亮度差異懲罰閾值
            'max_brightness_penalty': 0.8,              # 🆕 最大亮度懲罰
        }

        print("🔄 遮罩二次重新分類器已初始化（整合視覺差異驗證）")

    def calculate_mask_similarity(self, img_old, img_new, mask):
        """🆕 改進版：整合紋理、亮度差異和視覺差異驗證"""
        try:
            # 原有的紋理特徵相似度
            old_features = self.feature_extractor.extract_region_features(img_old, mask)
            new_features = self.feature_extractor.extract_region_features(img_new, mask)
            texture_similarity = self.feature_extractor.calculate_similarity(old_features, new_features)

            # 🆕 亮度差異懲罰
            brightness_penalty = self._calculate_brightness_difference_penalty(img_old, img_new, mask)

            # 🆕 視覺差異檢測
            visual_difference_penalty = self._calculate_visual_difference_penalty(img_old, img_new, mask)

            # 🆕 邊緣一致性檢測
            edge_consistency = self._calculate_edge_consistency(img_old, img_new, mask)

            # 🆕 綜合計算最終相似度
            final_similarity = texture_similarity * (1.0 - brightness_penalty) * (1.0 - visual_difference_penalty) * edge_consistency

            # 確保在 [0, 1] 範圍內
            final_similarity = max(0.0, min(1.0, final_similarity))

            print(f"    📊 紋理: {texture_similarity:.3f}")
            print(f"    🔆 亮度懲罰: {brightness_penalty:.3f}")
            print(f"    👁️ 視覺差異懲罰: {visual_difference_penalty:.3f}")
            print(f"    🔲 邊緣一致性: {edge_consistency:.3f}")
            print(f"    ➡️ 最終相似度: {final_similarity:.3f}")

            return final_similarity

        except Exception as e:
            print(f"⚠️ 計算遮罩相似度失敗: {e}")
            return texture_similarity if 'texture_similarity' in locals() else 0.0

    def _calculate_brightness_difference_penalty(self, img_old, img_new, mask):
        """計算亮度差異懲罰值"""
        try:
            # 轉換為灰階
            old_gray = cv2.cvtColor(img_old, cv2.COLOR_BGR2GRAY) if len(img_old.shape) == 3 else img_old
            new_gray = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY) if len(img_new.shape) == 3 else img_new

            # 提取遮罩區域的亮度
            old_brightness = old_gray[mask > 0]
            new_brightness = new_gray[mask > 0]

            if len(old_brightness) == 0 or len(new_brightness) == 0:
                return 0.0

            # 計算平均亮度差異
            old_mean = np.mean(old_brightness)
            new_mean = np.mean(new_brightness)
            brightness_diff = abs(old_mean - new_mean) / 255.0

            brightness_threshold = self.reclassification_params['brightness_penalty_threshold']
            max_penalty = self.reclassification_params['max_brightness_penalty']

            if brightness_diff > brightness_threshold:
                # 線性懲罰：差異越大，懲罰越重
                penalty = min(max_penalty, (brightness_diff - brightness_threshold) * 2.0)
                return penalty

            return 0.0

        except Exception as e:
            print(f"⚠️ 亮度差異計算失敗: {e}")
            return 0.0

    def _calculate_visual_difference_penalty(self, img_old, img_new, mask):
        """🆕 計算視覺差異懲罰值"""
        try:
            # 提取遮罩區域
            old_region = img_old.copy()
            new_region = img_new.copy()

            # 只保留遮罩區域，其他設為黑色
            old_region[mask == 0] = 0
            new_region[mask == 0] = 0

            # 計算直接的像素差異
            diff = cv2.absdiff(old_region, new_region)

            # 只看遮罩區域的差異
            mask_diff = diff[mask > 0]

            if len(mask_diff) == 0:
                return 0.0

            # 計算平均差異（歸一化到 0-1）
            mean_diff = np.mean(mask_diff) / 255.0

            visual_threshold = self.reclassification_params['visual_difference_threshold']

            if mean_diff > visual_threshold:
                # 線性懲罰：視覺差異越大，懲罰越重
                penalty = min(0.9, (mean_diff - visual_threshold) * 3.0)  # 最大懲罰90%
                return penalty

            return 0.0

        except Exception as e:
            print(f"⚠️ 視覺差異計算失敗: {e}")
            return 0.0

    def _calculate_edge_consistency(self, img_old, img_new, mask):
        """🆕 計算邊緣一致性分數"""
        try:
            # 轉換為灰階
            old_gray = cv2.cvtColor(img_old, cv2.COLOR_BGR2GRAY) if len(img_old.shape) == 3 else img_old
            new_gray = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY) if len(img_new.shape) == 3 else img_new

            # Canny邊緣檢測
            old_edges = cv2.Canny(old_gray, 50, 150)
            new_edges = cv2.Canny(new_gray, 50, 150)

            # 只比較遮罩區域內的邊緣
            old_edges_masked = old_edges[mask > 0]
            new_edges_masked = new_edges[mask > 0]

            if len(old_edges_masked) == 0:
                return 1.0  # 如果沒有邊緣，認為一致

            # 計算邊緣相似度（邊緣結構應該保持一致）
            edge_diff = np.mean(np.abs(old_edges_masked.astype(float) - new_edges_masked.astype(float)))
            edge_consistency = max(0.3, 1.0 - edge_diff / 255.0)  # 最低保持30%的權重

            return edge_consistency

        except Exception as e:
            print(f"⚠️ 邊緣一致性計算失敗: {e}")
            return 1.0

    def reclassify_masks(self, img_old, img_new, original_same_masks,
                        original_disappeared_masks, original_appeared_masks):
        """根據多重驗證重新分類所有遮罩"""

        print(f"\n🔄 開始執行遮罩二次重新分類（視覺差異驗證版）...")
        print(f"📊 原始分類統計:")
        print(f"  相同遮罩: {len(original_same_masks)} 個")
        print(f"  消失遮罩: {len(original_disappeared_masks)} 個")
        print(f"  新增遮罩: {len(original_appeared_masks)} 個")

        # 重新分類後的結果容器
        reclassified_same_masks = {}
        reclassified_disappeared_masks = {}
        reclassified_appeared_masks = {}

        similarity_threshold_same = self.reclassification_params['similarity_threshold_for_same']
        similarity_threshold_different = self.reclassification_params['similarity_threshold_for_different']

        # 統計資訊
        stats = {
            'appeared_to_same': 0,
            'disappeared_to_same': 0,
            'same_kept': 0,
            'appeared_kept': 0,
            'disappeared_kept': 0,
            'visual_diff_detected': 0,
            'brightness_penalty_applied': 0
        }

        # Step 1: 重新分類「新增遮罩」
        print(f"\n🟢 Step 1: 重新分類新增遮罩 ({len(original_appeared_masks)} 個)")
        for mask_name, mask in original_appeared_masks.items():
            try:
                mask_area = np.count_nonzero(mask)
                if mask_area == 0:
                    continue

                print(f"  🔍 新增遮罩: {mask_name[:20]}...")

                # 🆕 使用改進的相似度計算
                similarity = self.calculate_mask_similarity(img_old, img_new, mask)

                if similarity >= similarity_threshold_same:
                    # 重新分類為相同遮罩
                    reclassified_same_masks[mask_name] = mask
                    stats['appeared_to_same'] += 1
                    print(f"  🔄 新增→相同 (相似度 {similarity:.3f} >= {similarity_threshold_same})")
                else:
                    # 保持為新增遮罩
                    reclassified_appeared_masks[mask_name] = mask
                    stats['appeared_kept'] += 1
                    print(f"  ✅ 保持新增 (相似度 {similarity:.3f} < {similarity_threshold_same})")

            except Exception as e:
                print(f"  ❌ 處理新增遮罩 {mask_name} 時發生錯誤: {e}")
                reclassified_appeared_masks[mask_name] = mask
                continue

        # Step 2: 重新分類「消失遮罩」
        print(f"\n🔴 Step 2: 重新分類消失遮罩 ({len(original_disappeared_masks)} 個)")
        for mask_name, mask in original_disappeared_masks.items():
            try:
                mask_area = np.count_nonzero(mask)
                if mask_area == 0:
                    continue

                print(f"  🔍 消失遮罩: {mask_name[:20]}...")

                # 🆕 使用改進的相似度計算
                similarity = self.calculate_mask_similarity(img_old, img_new, mask)

                if similarity >= similarity_threshold_same:
                    # 重新分類為相同遮罩
                    reclassified_same_masks[mask_name] = mask
                    stats['disappeared_to_same'] += 1
                    print(f"  🔄 消失→相同 (相似度 {similarity:.3f} >= {similarity_threshold_same})")
                else:
                    # 保持為消失遮罩
                    reclassified_disappeared_masks[mask_name] = mask
                    stats['disappeared_kept'] += 1
                    print(f"  ✅ 保持消失 (相似度 {similarity:.3f} < {similarity_threshold_same})")

            except Exception as e:
                print(f"  ❌ 處理消失遮罩 {mask_name} 時發生錯誤: {e}")
                reclassified_disappeared_masks[mask_name] = mask
                continue

        # Step 3: 處理「相同遮罩」（保持相同）
        print(f"\n🔵 Step 3: 處理相同遮罩 ({len(original_same_masks)} 個)")
        for mask_name, mask in original_same_masks.items():
            try:
                mask_area = np.count_nonzero(mask)
                if mask_area == 0:
                    continue

                # 保持為相同遮罩
                reclassified_same_masks[mask_name] = mask
                stats['same_kept'] += 1
                print(f"  ✅ 保持相同: {mask_name[:20]}...")

            except Exception as e:
                print(f"  ❌ 處理相同遮罩 {mask_name} 時發生錯誤: {e}")
                reclassified_same_masks[mask_name] = mask
                continue

        # 統計結果
        print(f"\n📊 重新分類結果統計:")
        print(f"  新增→相同: {stats['appeared_to_same']} 個")
        print(f"  消失→相同: {stats['disappeared_to_same']} 個")
        print(f"  相同遮罩保留: {stats['same_kept']} 個")
        print(f"  新增遮罩保留: {stats['appeared_kept']} 個")
        print(f"  消失遮罩保留: {stats['disappeared_kept']} 個")

        print(f"\n📈 重新分類後數量:")
        print(f"  相同遮罩: {len(reclassified_same_masks)} 個")
        print(f"  消失遮罩: {len(reclassified_disappeared_masks)} 個")
        print(f"  新增遮罩: {len(reclassified_appeared_masks)} 個")

        return {
            'same_masks': reclassified_same_masks,
            'disappeared_masks': reclassified_disappeared_masks,
            'appeared_masks': reclassified_appeared_masks,
            'stats': stats
        }

def load_mask_images(mask_folder_path):
    """載入指定資料夾內的所有遮罩影像"""
    mask_folder = Path(mask_folder_path)
    mask_images = {}

    print(f"正在載入遮罩檔案從: {mask_folder_path}")

    if not mask_folder.exists():
        print(f"⚠️ 警告：遮罩資料夾不存在: {mask_folder_path}")
        return mask_images

    for mask_file in mask_folder.glob("*.png"):
        mask_img = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask_img is not None:
            # 二值化，確保為0或255
            _, binary_mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
            mask_images[mask_file.name] = binary_mask
            print(f"  載入遮罩: {mask_file.name}, 尺寸: {binary_mask.shape}")

    print(f"總共載入 {len(mask_images)} 個遮罩檔案")
    return mask_images

def save_masks_to_folders(masks_dict, output_base_path, category_name):
    """將遮罩字典儲存到指定資料夾"""
    category_path = Path(output_base_path) / category_name
    category_path.mkdir(parents=True, exist_ok=True)

    print(f"💾 儲存 {len(masks_dict)} 個{category_name}遮罩到: {category_path}")

    for mask_name, mask in masks_dict.items():
        mask_file_path = category_path / mask_name
        cv2.imwrite(str(mask_file_path), mask)
        print(f"  ✅ 儲存: {mask_name}")

    return str(category_path)

def create_mask_only_image(masks_dict, image_shape, color=(0, 255, 0)):
    """🆕 創建只包含遮罩的圖像（純遮罩，背景透明）"""
    if not masks_dict:
        # 如果沒有遮罩，返回完全透明的圖像
        transparent_image = np.zeros((*image_shape[:2], 4), dtype=np.uint8)
        return transparent_image

    # 🔧 關鍵修正：創建4通道RGBA圖像（支援透明度）
    mask_image = np.zeros((*image_shape[:2], 4), dtype=np.uint8)

    # 將所有遮罩合併
    combined_mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for mask_name, mask in masks_dict.items():
        if isinstance(mask, np.ndarray) and mask.size > 0:
            # 確保遮罩尺寸與圖像匹配
            if mask.shape != image_shape[:2]:
                mask = cv2.resize(mask, (image_shape[1], image_shape[0]))

            # 合併遮罩
            combined_mask = cv2.bitwise_or(combined_mask, mask)

    # 將合併的遮罩區域設為指定顏色，背景保持透明
    mask_positions = combined_mask > 0
    if np.any(mask_positions):
        # 🔧 關鍵修正：只有遮罩區域有顏色，其他區域保持透明
        mask_image[mask_positions, 0] = color[0]  # B
        mask_image[mask_positions, 1] = color[1]  # G
        mask_image[mask_positions, 2] = color[2]  # R
        mask_image[mask_positions, 3] = 255       # A（不透明）

    # 背景區域的 Alpha 通道已經是 0（透明）

    print(f"🎨 創建透明背景遮罩圖像: {len(masks_dict)} 個遮罩, 顏色: {color}")

    return mask_image

def detect_changes_with_texture_analysis(image1_path, image2_path, matching_results_path,
                                       detection_output_path, params=None):
    """
    🆕 主要功能：使用紋理分析進行變化檢測（整合視覺差異驗證）
    """

    try:
        print(f"\n🔍 開始執行基於紋理分析的變化檢測（視覺差異驗證版）...")
        print(f"📁 Matching結果: {matching_results_path}")
        print(f"📁 Detection輸出: {detection_output_path}")

        # 🔧 修正：配合簡化檔名策略，使用 upload 目錄中的標準檔名
        detection_path = Path(detection_output_path)
        upload_dir = detection_path.parent / 'upload'

        print(f"📂 搜尋上傳目錄: {upload_dir}")

        # 🔧 關鍵修正：直接尋找簡化檔名的圖片
        image1_file = upload_dir / "image1.jpg"
        image2_file = upload_dir / "image2.jpg"

        if image1_file.exists() and image2_file.exists():
            # 使用簡化檔名
            corrected_image1_path = str(image1_file)
            corrected_image2_path = str(image2_file)
            print(f"✅ 找到簡化檔名的圖片:")
            print(f"  圖片1: {image1_file.name}")
            print(f"  圖片2: {image2_file.name}")
        else:
            # 🔧 備用方案：如果沒有簡化檔名，按檔名排序取前兩張
            print(f"⚠️ 未找到簡化檔名圖片，使用備用搜尋...")

            all_images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                all_images.extend(list(upload_dir.glob(ext)))

            all_images.sort()
            if len(all_images) < 2:
                raise ValueError(f"Upload目錄中只找到 {len(all_images)} 張圖片，需要至少2張")

            corrected_image1_path = str(all_images[0])
            corrected_image2_path = str(all_images[1])

            print(f"📂 使用備用圖片:")
            print(f"  圖片1: {Path(corrected_image1_path).name}")
            print(f"  圖片2: {Path(corrected_image2_path).name}")

        print(f"\n🔧 最終使用的圖片路徑:")
        print(f"  圖片1: {corrected_image1_path}")
        print(f"  圖片2: {corrected_image2_path}")

        # 載入圖片
        img1 = cv2.imread(corrected_image1_path)
        img2 = cv2.imread(corrected_image2_path)

        if img1 is None or img2 is None:
            raise ValueError(f"無法載入圖片: {corrected_image1_path} 或 {corrected_image2_path}")

        # 檢查圖片是否真的不同
        if np.array_equal(img1, img2):
            print("⚠️ 警告：載入的兩張圖片完全相同！")
        else:
            print("✅ 確認：載入的兩張圖片不同")

        # 確保圖片大小一致
        if img1.shape != img2.shape:
            print(f"⚠️ 調整圖片2尺寸: {img2.shape} -> {img1.shape}")
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # 建立輸出目錄
        detection_path.mkdir(parents=True, exist_ok=True)

        # 載入 matching 結果
        matching_path = Path(matching_results_path)
        same_masks_path = matching_path / "Same"
        disappeared_masks_path = matching_path / "Disappear"
        appeared_masks_path = matching_path / "NewAdded"

        print(f"\n📂 載入 matching 結果...")
        original_same_masks = load_mask_images(same_masks_path)
        original_disappeared_masks = load_mask_images(disappeared_masks_path)
        original_appeared_masks = load_mask_images(appeared_masks_path)

        # 初始化紋理分析器和重新分類器
        print(f"\n🤖 初始化紋理分析系統（視覺差異驗證版）...")
        feature_extractor = ViTTextureExtractor()
        reclassifier = MaskReclassifier(feature_extractor)

        # 執行重新分類
        print(f"\n🔄 執行紋理重新分類...")
        reclassification_result = reclassifier.reclassify_masks(
            img1, img2,
            original_same_masks,
            original_disappeared_masks,
            original_appeared_masks
        )

        # 取得重新分類後的遮罩
        final_same_masks = reclassification_result['same_masks']
        final_disappeared_masks = reclassification_result['disappeared_masks']
        final_appeared_masks = reclassification_result['appeared_masks']

        # 💾 Step 1: 儲存重新分類後的遮罩到資料夾
        print(f"\n💾 Step 1: 儲存重新分類遮罩到 detection 資料夾...")

        same_folder_path = save_masks_to_folders(final_same_masks, detection_path, "Same")
        disappeared_folder_path = save_masks_to_folders(final_disappeared_masks, detection_path, "Disappear")
        appeared_folder_path = save_masks_to_folders(final_appeared_masks, detection_path, "NewAdded")

        # 🖼️ Step 2: 生成透明背景結果圖片
        print(f"\n🖼️ Step 2: 生成透明背景結果圖片...")

        # 定義顏色 (BGR格式)
        same_color = (255, 150, 100)      # 藍色 - 相同遮罩
        disappeared_color = (0, 0, 255)   # 綠色 - 消失遮罩
        appeared_color = (0, 255, 0)      # 紅色 - 新增遮罩

        # 生成結果圖
        results = {}

        # 儲存正確的原始圖片（JPG格式）
        img1_original_path = detection_path / "image1_original.jpg"
        img2_original_path = detection_path / "image2_original.jpg"
        cv2.imwrite(str(img1_original_path), img1)
        cv2.imwrite(str(img2_original_path), img2)
        results['image1_original'] = str(img1_original_path)
        results['image2_original'] = str(img2_original_path)
        print(f"✅ 儲存正確的原始圖片:")
        print(f"  {img1_original_path.name} <- {Path(corrected_image1_path).name}")
        print(f"  {img2_original_path.name} <- {Path(corrected_image2_path).name}")

        # 🔧 關鍵修正：生成透明背景的遮罩圖片（PNG格式）
        # image1_same_masks (透明背景)
        img1_same_masks = create_mask_only_image(final_same_masks, img1.shape, same_color)
        img1_same_path = detection_path / "image1_same_masks.png"  # 改為PNG
        cv2.imwrite(str(img1_same_path), img1_same_masks)
        results['image1_same_masks'] = str(img1_same_path)
        print(f"✅ 生成透明背景遮罩: {img1_same_path.name} ({len(final_same_masks)} 個相同遮罩)")

        # image2_same_masks (透明背景)
        img2_same_masks = create_mask_only_image(final_same_masks, img2.shape, same_color)
        img2_same_path = detection_path / "image2_same_masks.png"  # 改為PNG
        cv2.imwrite(str(img2_same_path), img2_same_masks)
        results['image2_same_masks'] = str(img2_same_path)
        print(f"✅ 生成透明背景遮罩: {img2_same_path.name} ({len(final_same_masks)} 個相同遮罩)")

        # image1_disappeared_masks (透明背景 - 綠色)
        img1_disappeared_masks = create_mask_only_image(final_disappeared_masks, img1.shape, disappeared_color)
        img1_disappeared_path = detection_path / "image1_disappeared_masks.png"  # 改為PNG
        cv2.imwrite(str(img1_disappeared_path), img1_disappeared_masks)
        results['image1_disappeared_masks'] = str(img1_disappeared_path)
        print(f"✅ 生成透明背景遮罩: {img1_disappeared_path.name} ({len(final_disappeared_masks)} 個消失遮罩 - 綠色)")

        # image2_appeared_masks (透明背景 - 紅色)
        img2_appeared_masks = create_mask_only_image(final_appeared_masks, img2.shape, appeared_color)
        img2_appeared_path = detection_path / "image2_appeared_masks.png"  # 改為PNG
        cv2.imwrite(str(img2_appeared_path), img2_appeared_masks)
        results['image2_appeared_masks'] = str(img2_appeared_path)
        print(f"✅ 生成透明背景遮罩: {img2_appeared_path.name} ({len(final_appeared_masks)} 個新增遮罩 - 紅色)")

        # 📊 Step 3: 生成檢測報告
        detection_stats = {
            'original_classification': {
                'same_masks': len(original_same_masks),
                'disappeared_masks': len(original_disappeared_masks),
                'appeared_masks': len(original_appeared_masks),
                'total': len(original_same_masks) + len(original_disappeared_masks) + len(original_appeared_masks)
            },
            'final_classification': {
                'same_masks': len(final_same_masks),
                'disappeared_masks': len(final_disappeared_masks),
                'appeared_masks': len(final_appeared_masks),
                'total': len(final_same_masks) + len(final_disappeared_masks) + len(final_appeared_masks)
            },
            'reclassification_stats': reclassification_result['stats']
        }

        # 計算改善效果
        original_changes = len(original_disappeared_masks) + len(original_appeared_masks)
        final_changes = len(final_disappeared_masks) + len(final_appeared_masks)
        change_reduction = original_changes - final_changes
        improvement_percentage = (change_reduction / original_changes * 100) if original_changes > 0 else 0

        detection_stats['improvement'] = {
            'original_changes': original_changes,
            'final_changes': final_changes,
            'change_reduction': change_reduction,
            'improvement_percentage': improvement_percentage
        }

        print(f"\n📈 檢測結果統計:")
        print(f"  原始變化數: {original_changes} 個")
        print(f"  最終變化數: {final_changes} 個")
        print(f"  減少誤判: {change_reduction} 個 ({improvement_percentage:.1f}% 改善)")

        # 儲存檢測報告
        report_path = detection_path / "detection_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(detection_stats, f, ensure_ascii=False, indent=2)

        print(f"\n✅ 紋理檢測完成（視覺差異驗證版）！")
        print(f"📁 輸出資料夾: {detection_path}")
        print(f"📊 檢測報告: {report_path}")

        return {
            'success': True,
            'output_path': str(detection_path),
            'generated_images': results,
            'mask_folders': {
                'same': same_folder_path,
                'disappeared': disappeared_folder_path,
                'appeared': appeared_folder_path
            },
            'statistics': detection_stats,
            'report_path': str(report_path)
        }

    except Exception as e:
        print(f"❌ 紋理檢測過程發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'output_path': detection_output_path,
            'generated_images': {},
            'mask_folders': {},
            'statistics': {},
            'report_path': ''
        }

if __name__ == "__main__":
    # 測試用的主函數
    print("🧪 測試紋理檢測模組（視覺差異驗證版）...")

    # 測試路徑（請根據實際情況修改）
    image1_path = r"C:\Users\my544\Desktop\0808\results\runs\run_001\upload\image1.jpg"  # 🔧 修正為簡化檔名
    image2_path = r"C:\Users\my544\Desktop\0808\results\runs\run_001\upload\image2.jpg"  # 🔧 修正為簡化檔名
    matching_results_path = r"C:\Users\my544\Desktop\0808\results\runs\run_001\matching"
    detection_output_path = r"C:\Users\my544\Desktop\0808\results\runs\run_001\detection"

    # 執行檢測
    result = detect_changes_with_texture_analysis(
        image1_path, image2_path, matching_results_path, detection_output_path
    )

    if result['success']:
        print("🎉 測試成功！")
    else:
        print(f"❌ 測試失敗: {result['error']}")
