import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import json
import time
from pathlib import Path
from datetime import datetime

# 🔧 修復：設定matplotlib使用非GUI後端
plt.switch_backend('Agg')
import matplotlib
matplotlib.use('Agg')

class AdvancedREBNCONV(nn.Module):
    """增強版 REBNCONV 模組，支援注意力機制"""
    def __init__(self, in_ch=3, out_ch=3, dirate=1, use_attention=True):
        super(AdvancedREBNCONV, self).__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)
        self.use_attention = use_attention

        if use_attention:
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_ch, out_ch//4, 1),
                nn.ReLU(),
                nn.Conv2d(out_ch//4, out_ch, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))
        if self.use_attention:
            att = self.attention(xout)
            xout = xout * att
        return xout

class UltraU2NET(nn.Module):
    """終極版 U2NET 模型 - 簡化版"""
    def __init__(self, in_ch=3, out_ch=1, use_attention=True):
        super(UltraU2NET, self).__init__()
        self.encoder = nn.Sequential(
            AdvancedREBNCONV(in_ch, 64, use_attention=use_attention),
            nn.MaxPool2d(2),
            AdvancedREBNCONV(64, 128, use_attention=use_attention),
            nn.MaxPool2d(2),
            AdvancedREBNCONV(128, 256, use_attention=use_attention)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            AdvancedREBNCONV(128, 64, use_attention=use_attention),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            AdvancedREBNCONV(32, out_ch, use_attention=use_attention)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return torch.sigmoid(decoded)

class SkyRemovalProcessor:
    """天空遮罩分離處理器 - 整合到網頁系統"""

    def __init__(self, device="auto"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        self.model = None
        self.config = self._load_default_config()

        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        print(f"✅ 天空遮罩處理器初始化完成，設備: {self.device}")

    def _load_default_config(self):
        """載入預設配置"""
        return {
            "sky_threshold": 0.25,
            "use_traditional_backup": True,
            "morphology_kernel_size": 5,
            "gaussian_blur_kernel": 3,
            "save_visualization": True
        }

    def initialize_model(self, model_path=None):
        """初始化深度學習模型（可選）"""
        try:
            self.model = UltraU2NET(3, 1, use_attention=True)
            if model_path and os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"✅ 成功載入預訓練模型: {model_path}")
            else:
                print("⚠️ 未找到預訓練模型，使用隨機初始化權重")

            self.model.to(self.device)
            self.model.eval()
            return True
        except Exception as e:
            print(f"⚠️ 模型初始化失敗: {e}")
            self.model = None
            return False

    def traditional_sky_detection(self, image_np):
        """傳統天空檢測方法"""
        h, w = image_np.shape[:2]

        # HSV色彩空間檢測
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

        # 藍天檢測
        sky_mask1 = cv2.inRange(hsv, np.array([100, 20, 100]), np.array([130, 255, 255]))

        # 白雲檢測
        sky_mask2 = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 30, 255]))

        # 合併遮罩
        color_mask = cv2.bitwise_or(sky_mask1, sky_mask2) / 255.0

        # 位置先驗（天空通常在上方）
        y_coords = np.arange(h).reshape(-1, 1)
        position_prior = np.exp(-y_coords / (h * 0.3))
        position_mask = np.tile(position_prior, (1, w))

        # 結合色彩和位置信息
        combined_mask = color_mask * 0.6 + position_mask * 0.4

        return combined_mask

    def deep_learning_prediction(self, image):
        """深度學習預測天空遮罩"""
        if not self.model:
            return None

        try:
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                prediction = self.model(input_tensor)
                mask = F.interpolate(prediction, size=image.size[::-1],
                                   mode='bilinear', align_corners=False)

            return mask.squeeze().cpu().numpy()
        except Exception as e:
            print(f"⚠️ 深度學習預測失敗: {e}")
            return None

    def post_process_mask(self, mask):
        """後處理遮罩"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (self.config['morphology_kernel_size'],
                                          self.config['morphology_kernel_size']))

        # 開運算去除小噪點
        mask_processed = cv2.morphologyEx(mask.astype(np.float32), cv2.MORPH_OPEN, kernel)

        # 閉運算填補小洞
        mask_processed = cv2.morphologyEx(mask_processed, cv2.MORPH_CLOSE, kernel)

        # 高斯模糊柔化邊緣
        mask_processed = cv2.GaussianBlur(mask_processed,
                                        (self.config['gaussian_blur_kernel'],
                                         self.config['gaussian_blur_kernel']), 0)

        return mask_processed

    def create_sam2_optimized_image(self, original_image_path, sky_mask_path, output_path):
        """
        🔧 修改版：創建黑色天空填充的 SAM2 優化圖片
        """
        try:
            print(f"🎯 生成 SAM2 黑色填充優化圖片: {Path(output_path).name}")

            # 載入原始圖片和天空遮罩
            original_img = cv2.imread(original_image_path)
            sky_mask = cv2.imread(sky_mask_path, cv2.IMREAD_GRAYSCALE)

            if original_img is None or sky_mask is None:
                raise ValueError("無法載入圖片或遮罩")

            # 確保尺寸一致
            if original_img.shape[:2] != sky_mask.shape:
                sky_mask = cv2.resize(sky_mask, (original_img.shape[1], original_img.shape[0]))

            sam2_optimized = original_img.copy()
            sky_pixels = sky_mask > 127

            # 🔧 關鍵修改：將天空區域設為純黑色 (0, 0, 0)
            sam2_optimized[sky_pixels] = [0, 0, 0]

            # 儲存結果
            success = cv2.imwrite(output_path, sam2_optimized)
            if not success:
                raise RuntimeError(f"無法儲存 SAM2 優化圖片")

            # 統計天空區域比例
            sky_pixel_count = np.sum(sky_pixels)
            total_pixels = sky_mask.shape[0] * sky_mask.shape[1]
            sky_percentage = (sky_pixel_count / total_pixels) * 100

            print(f"✅ SAM2 黑色填充優化圖片已生成")
            print(f"   - 輸出檔案: {Path(output_path).name}")
            print(f"   - 天空區域: {sky_percentage:.1f}% (已填充為黑色)")
            print(f"   - 地面區域: {100-sky_percentage:.1f}% (保持原色彩)")
            print(f"   - 處理方式: 純黑色填充")

            return output_path

        except Exception as e:
            print(f"❌ 生成 SAM2 黑色填充優化圖片失敗: {e}")
            return None

    def make_sky_transparent(self, image_path, sky_mask_path, output_path=None):
        """
        🔧 修復版：將天空區域設為透明
        """
        try:
            print(f"🌤️ 開始將天空區域透明化：{Path(image_path).name}")

            # 載入原始圖片
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise FileNotFoundError(f"無法載入圖片: {image_path}")

            # 載入天空遮罩 (灰階)
            sky_mask = cv2.imread(sky_mask_path, cv2.IMREAD_GRAYSCALE)
            if sky_mask is None:
                raise FileNotFoundError(f"無法載入天空遮罩: {sky_mask_path}")

            # 確保圖片和遮罩尺寸相同
            if image.shape[:2] != sky_mask.shape[:2]:
                sky_mask = cv2.resize(sky_mask, (image.shape[1], image.shape[0]))

            # 如果原圖只有3個通道(BGR)，新增Alpha通道
            if len(image.shape) == 3 and image.shape[2] == 3:
                # 分離BGR通道
                b, g, r = cv2.split(image)
                # 創建Alpha通道：天空區域=0(透明)，其他區域=255(不透明)
                alpha = np.where(sky_mask > 128, 0, 255).astype(np.uint8)
                # 合併為BGRA格式
                image_rgba = cv2.merge([b, g, r, alpha])
            elif len(image.shape) == 3 and image.shape[2] == 4:
                # 如果已有Alpha通道，直接修改
                b, g, r, a = cv2.split(image)
                alpha = np.where(sky_mask > 128, 0, 255).astype(np.uint8)
                image_rgba = cv2.merge([b, g, r, alpha])
            else:
                raise ValueError("圖片格式不支援，必須是3通道(BGR)或4通道(BGRA)")

            # 生成輸出路徑
            if output_path is None:
                input_path = Path(image_path)
                output_path = input_path.parent / f"{input_path.stem}_sky_transparent.png"
            else:
                output_path = Path(output_path)

            # 確保輸出目錄存在
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 保存透明化圖片 (必須用PNG格式支援透明度)
            success = cv2.imwrite(str(output_path), image_rgba)
            if not success:
                raise RuntimeError(f"無法保存透明化圖片到: {output_path}")

            # 統計透明像素數量
            transparent_pixels = np.sum(image_rgba[:, :, 3] == 0)
            total_pixels = image_rgba.shape[0] * image_rgba.shape[1]
            transparent_percentage = (transparent_pixels / total_pixels) * 100

            print(f"✅ 天空透明化完成")
            print(f"   - 輸出檔案: {output_path}")
            print(f"   - 透明像素比例: {transparent_percentage:.1f}%")

            return str(output_path)

        except Exception as e:
            print(f"❌ 天空透明化失敗: {e}")
            raise e

    def process_sky_removal(self, image_path, output_dir):
        """🔧 主要處理函數 - 使用黑色填充天空優化"""
        try:
            print(f"🌤️ 開始處理天空遮罩分離: {Path(image_path).name}")

            # 載入圖像
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_np = np.array(image)

            # 深度學習預測（如果模型可用）
            dl_mask = self.deep_learning_prediction(image)

            # 傳統方法備份
            traditional_mask = self.traditional_sky_detection(image_np)

            # 融合結果
            if dl_mask is not None:
                final_mask = 0.6 * dl_mask + 0.4 * traditional_mask
                method_used = "深度學習+傳統融合"
            else:
                final_mask = traditional_mask
                method_used = "傳統演算法"

            # 後處理
            processed_mask = self.post_process_mask(final_mask)

            # 創建輸出
            outputs = self._create_sky_removal_outputs(image_np, processed_mask)

            # 保存結果到運行目錄
            saved_files = self._save_results_to_run_directory(
                image_path, outputs, processed_mask, output_dir, method_used
            )

            # 🔧 關鍵修改：使用黑色填充版本替代白色填充
            sky_mask_path = saved_files.get('sky_mask')
            if sky_mask_path:
                try:
                    # 🆕 改用黑色填充方法生成 SAM2 專用圖片
                    optimized_path = self.create_sam2_optimized_image(
                        image_path,
                        sky_mask_path,
                        os.path.join(output_dir, f"{Path(image_path).stem}_sam2_ready.png")
                    )

                    if optimized_path:
                        saved_files['sam2_ready_optimized'] = optimized_path
                        print(f"🎯 已生成SAM2專用黑色填充圖片: {Path(optimized_path).name}")

                except Exception as e:
                    print(f"⚠️ 生成SAM2優化圖片失敗: {e}")

            # 統計信息
            sky_percentage = np.mean(processed_mask > self.config['sky_threshold']) * 100

            result_data = {
                'sky_percentage': float(sky_percentage),
                'method_used': method_used,
                'processing_successful': True,
                'saved_files': saved_files,
                'image_shape': image_np.shape
            }

            print(f"✅ 天空遮罩分離完成 - 方法: {method_used}, 天空佔比: {sky_percentage:.1f}%")
            return result_data

        except Exception as e:
            print(f"❌ 天空遮罩分離失敗: {e}")
            return None

    def _create_sky_removal_outputs(self, image_np, mask):
        """🔧 修改版：創建黑色填充的天空去除輸出"""
        threshold = self.config['sky_threshold']
        hard_mask = (mask > threshold).astype(np.uint8)

        outputs = {}

        # 1. 原圖（保持不變）
        outputs['original'] = image_np.copy()

        # 2. 🔧 修改：天空區域移除（黑色填充）
        sky_removed = image_np.copy()
        sky_removed[hard_mask == 1] = [0, 0, 0]  # 改為黑色填充
        outputs['sky_removed_black'] = sky_removed

        # 3. 透明版本（RGBA）
        rgba_output = np.zeros((image_np.shape[0], image_np.shape[1], 4), dtype=np.uint8)
        rgba_output[:, :, :3] = image_np
        rgba_output[:, :, 3] = ((1 - mask) * 255).astype(np.uint8)
        outputs['sky_removed_transparent'] = rgba_output

        # 4. 天空遮罩視覺化
        mask_visual = (mask * 255).astype(np.uint8)
        outputs['sky_mask'] = mask_visual

        return outputs

    def _save_results_to_run_directory(self, image_path, outputs, mask, output_dir, method_used):
        """保存結果到運行目錄"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        image_name = Path(image_path).stem
        saved_files = {}

        # 保存各種輸出
        for output_type, output_data in outputs.items():
            if output_type == 'sky_removed_transparent':
                file_path = output_path / f"{image_name}_sky_removed_transparent.png"
                Image.fromarray(output_data).save(file_path)
            elif output_type == 'sky_mask':
                file_path = output_path / f"{image_name}_sky_mask.png"
                Image.fromarray(output_data).save(file_path)
            else:
                file_path = output_path / f"{image_name}_{output_type}.jpg"
                Image.fromarray(output_data).save(file_path, quality=95)

            saved_files[output_type] = str(file_path)

        # 保存處理報告
        report = {
            'input_image': str(image_path),
            'processing_time': datetime.now().isoformat(),
            'method_used': method_used,
            'sky_percentage': float(np.mean(mask > self.config['sky_threshold']) * 100),
            'output_files': saved_files,
            'configuration': self.config
        }

        report_path = output_path / f"{image_name}_sky_removal_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        saved_files['report'] = str(report_path)
        print(f"📁 天空遮罩分離結果已保存至: {output_path}")

        return saved_files

# ===== API 封裝函式 =====

def remove_sky_masks_api(image1_path, image2_path, output_dir, device="auto", enable_sky_removal=True):
    """
    🔧 修改版：生成黑色填充的天空區域供 SAM2 使用

    Args:
        image1_path: 第一張圖片路徑
        image2_path: 第二張圖片路徑
        output_dir: 輸出目錄
        device: 設備設定 ("auto", "cpu", "cuda")
        enable_sky_removal: 是否啟用天空遮罩去除功能
    """
    try:
        print(f"🌤️ 開始執行天空遮罩分離與黑色填充優化... (啟用: {enable_sky_removal})")

        # 🔧 新增：如果禁用天空遮罩去除，直接複製原圖
        if not enable_sky_removal:
            print("🔧 天空遮罩去除已禁用，直接複製原圖...")

            # 確保輸出目錄存在
            os.makedirs(output_dir, exist_ok=True)

            # 複製原圖到輸出目錄
            import shutil
            output1_path = os.path.join(output_dir, 'image1_sky_removed.jpg')
            output2_path = os.path.join(output_dir, 'image2_sky_removed.jpg')

            shutil.copy2(image1_path, output1_path)
            shutil.copy2(image2_path, output2_path)

            print(f"✅ 原圖已複製到: {output1_path}, {output2_path}")

            return {
                'status': 'success',
                'message': '天空遮罩去除已禁用，使用原圖',
                'data': {
                    'processed_images': 2,
                    'output_dir': output_dir,
                    'image1_sky_removed': output1_path,
                    'image2_sky_removed': output2_path,
                    'sky_removal_enabled': False
                }
            }

        # 驗證輸入檔案
        for img_path in [image1_path, image2_path]:
            if not os.path.exists(img_path):
                return {
                    'status': 'error',
                    'message': f'找不到圖像檔案: {img_path}',
                    'error': f'File not found: {img_path}'
                }

        # 初始化處理器
        processor = SkyRemovalProcessor(device=device)
        processor.initialize_model()

        # 確保輸出目錄存在
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {'image1': None, 'image2': None}
        sam2_ready_files = {}

        # 🔧 處理兩張圖片，使用簡化檔名
        image_configs = [
            {
                'input_path': image1_path,
                'output_subdir': 'image1_sky_removal',
                'result_key': 'image1',
                'sam2_filename': 'image1_sam2_ready.png',
                'sam2_key': 'image1'
            },
            {
                'input_path': image2_path,
                'output_subdir': 'image2_sky_removal',
                'result_key': 'image2',
                'sam2_filename': 'aligned_image_sam2_ready.png',
                'sam2_key': 'image2'
            }
        ]

        # 處理每張圖像
        for config in image_configs:
            input_path = config['input_path']
            result_key = config['result_key']

            print(f"\n🔄 處理 {result_key}: {Path(input_path).name}")

            # 執行天空遮罩分離
            image_output_dir = output_path / config['output_subdir']
            image_output_dir.mkdir(parents=True, exist_ok=True)
            result = processor.process_sky_removal(input_path, str(image_output_dir))

            if result and result.get('processing_successful'):
                results[result_key] = result

                # 🔧 修改：獲取黑色填充圖片路徑
                sam2_optimized_path = result['saved_files'].get('sam2_ready_optimized')
                if sam2_optimized_path:
                    # 複製到主目錄使用統一檔名
                    unified_sam2_path = output_path / config['sam2_filename']
                    import shutil
                    shutil.copy2(sam2_optimized_path, unified_sam2_path)
                    sam2_ready_files[config['sam2_key']] = str(unified_sam2_path)

                    print(f"🎯 SAM2 黑色填充圖片已準備: {config['sam2_filename']}")
                else:
                    print(f"⚠️ {result_key} SAM2 優化圖片生成失敗")
                    sam2_ready_files[config['sam2_key']] = None
            else:
                print(f"❌ {result_key} 處理失敗")
                results[result_key] = None
                sam2_ready_files[config['sam2_key']] = None

        # 統計處理結果
        successful_count = sum(1 for r in results.values() if r and r.get('processing_successful'))

        if successful_count > 0:
            # 🔧 驗證生成的 SAM2 專用檔案
            print(f"\n📸 生成的 SAM2 專用檔案:")
            for key, path in sam2_ready_files.items():
                if path and os.path.exists(path):
                    file_size = os.path.getsize(path)
                    print(f"  {key}: {Path(path).name} ({file_size:,} bytes) ✅")
                else:
                    print(f"  {key}: 檔案生成失敗 ❌")

            response_data = {
                'processed_images': successful_count,
                'total_images': 2,
                'results': results,
                'output_directory': str(output_path),
                'processing_summary': {
                    'image1_sky_percentage': results['image1'].get('sky_percentage', 0) if results['image1'] else 0,
                    'image2_sky_percentage': results['image2'].get('sky_percentage', 0) if results['image2'] else 0,
                    'average_sky_percentage': (
                        (results['image1'].get('sky_percentage', 0) if results['image1'] else 0) +
                        (results['image2'].get('sky_percentage', 0) if results['image2'] else 0)
                    ) / 2
                },
                # 🆕 SAM2 專用黑色填充圖片路徑
                'sam2_ready_files': sam2_ready_files,
                'optimization_method': 'black_sky_filling'  # 🔧 更新方法標記
            }

            return {
                'status': 'success',
                'message': f'天空遮罩分離與黑色填充優化完成，成功處理 {successful_count}/2 張圖像',
                'data': response_data
            }
        else:
            return {
                'status': 'error',
                'message': '所有圖像處理都失敗'
            }

    except Exception as e:
        import traceback
        return {
            'status': 'error',
            'message': '天空遮罩分離與黑色填充優化過程發生錯誤',
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def validate_sky_removal_parameters(params):
    """驗證天空遮罩分離參數"""
    try:
        # 驗證必要參數
        required_params = ['image1_path', 'image2_path', 'output_dir']
        for param in required_params:
            if param not in params or not params[param]:
                return False, f"缺少必要參數: {param}"

        # 驗證設備參數
        device = params.get('device', 'auto')
        valid_devices = ['auto', 'cuda', 'cpu']
        if device not in valid_devices:
            return False, f"device 必須是 {valid_devices} 之一"

        return True, ""

    except Exception as e:
        return False, f"參數驗證錯誤: {str(e)}"

# 主要執行程式
if __name__ == "__main__":
    print("🌤️ 天空遮罩分離處理器（含黑色填充優化功能）已載入")
    print("📋 主要功能:")
    print("   - remove_sky_masks_api() - 天空遮罩分離與黑色填充優化 API")
    print("   - validate_sky_removal_parameters() - 參數驗證")
    print("🎯 生成的 SAM2 優化圖片特色:")
    print("   - 天空區域純黑色填充 (0, 0, 0)")
    print("   - 地面區域保持原色彩")
    print("   - SAM2 更容易忽略黑色天空區域")
    print("   - 檔名: image1_sam2_ready.png, aligned_image_sam2_ready.png")
