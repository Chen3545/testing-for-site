import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import json
import pickle
from PIL import Image
import os
import sys
from datetime import datetime

# 🔧 修改：使用統一的專案路徑架構
WEBSITE_ROOT = Path(__file__).parent.parent.parent  # Git Version/
CHECKPOINT_PATH = WEBSITE_ROOT / "checkpoint" / "sam2.1_hiera_large.pt"  # 注意：checkpoint 不是 checkpoints
CONFIG_PATH = WEBSITE_ROOT / "configs" / "sam2.1_hiera_l.yaml"
RESULTS_ROOT = WEBSITE_ROOT / "results"

print(f"🏠 專案根目錄: {WEBSITE_ROOT}")
print(f"📁 結果目錄: {RESULTS_ROOT}")
print(f"⚖️ 模型權重: {CHECKPOINT_PATH}")
print(f"⚙️ 模型配置: {CONFIG_PATH}")

try:
    # 從已安裝的 sam2 套件導入
    from sam2.build_sam import build_sam2_video_predictor, build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    print("✅ SAM2 核心模組載入成功")
except ImportError as e:
    print(f"❌ SAM2 導入錯誤: {e}")
    print("請檢查 SAM2 是否已安裝: pip install git+https://github.com/facebookresearch/sam2.git")
    sys.exit(1)

def validate_sam2_parameters(params):
    """
    驗證 SAM2 分割參數

    Args:
        params: 參數字典

    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # 檢查必要參數
        required_params = [
            'points_per_side',
            'pred_iou_thresh',
            'stability_score_thresh',
            'min_mask_region_area'
        ]

        for param in required_params:
            if param not in params:
                return False, f"缺少必要參數: {param}"

        # 驗證數值範圍
        if not isinstance(params.get('points_per_side'), int) or params['points_per_side'] < 1:
            return False, "points_per_side 必須是正整數"

        if not (0.0 <= params.get('pred_iou_thresh', 0) <= 1.0):
            return False, "pred_iou_thresh 必須在 0.0-1.0 之間"

        if not (0.0 <= params.get('stability_score_thresh', 0) <= 1.0):
            return False, "stability_score_thresh 必須在 0.0-1.0 之間"

        if not isinstance(params.get('min_mask_region_area'), int) or params['min_mask_region_area'] < 0:
            return False, "min_mask_region_area 必須是非負整數"

        return True, ""

    except Exception as e:
        return False, f"參數驗證錯誤: {str(e)}"

# ===== 🆕 黑色區域過濾模組 =====

def detect_black_regions(image, black_threshold=20):
    """檢測圖像中的黑色區域"""
    try:
        # 轉換為灰階
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # 創建黑色區域遮罩
        black_mask = gray <= black_threshold

        # 計算黑色區域比例
        black_ratio = np.sum(black_mask) / black_mask.size
        print(f"🖤 檢測到黑色區域比例: {black_ratio:.1%} (閾值: {black_threshold})")

        return black_mask

    except Exception as e:
        print(f"❌ 黑色區域檢測失敗: {e}")
        return np.zeros(image.shape[:2], dtype=bool)

# ===== 🆕 像素大小過濾模組 =====

def filter_masks_by_min_area(masks, min_area, stage_name=""):
    """
    根據像素面積過濾遮罩

    Args:
        masks: 遮罩列表（numpy陣列）
        min_area: 最小面積閾值（像素數）
        stage_name: 階段名稱（用於日誌輸出）

    Returns:
        filtered_masks: 過濾後的遮罩列表
        filtered_count: 被過濾掉的遮罩數量
    """
    try:
        filtered_masks = []
        filtered_count = 0

        print(f"🔍 {stage_name}像素面積過濾 (最小面積: {min_area} 像素)...")

        for i, mask in enumerate(masks):
            # 計算遮罩面積
            area = int(np.sum(mask > 0.5))

            if area >= min_area:
                filtered_masks.append(mask)
            else:
                filtered_count += 1
                print(f"  ❌ 過濾遮罩 {i}: 面積 {area} < {min_area} 像素")

        print(f"📊 {stage_name}面積過濾完成: 保留 {len(filtered_masks)} 個，過濾 {filtered_count} 個")

        return filtered_masks, filtered_count

    except Exception as e:
        print(f"❌ {stage_name}面積過濾失敗: {e}")
        return masks, 0

def filter_black_region_masks(masks, black_mask, overlap_threshold=0.5):
    """過濾與黑色區域重疊過多的遮罩"""
    try:
        filtered_masks = []
        black_filtered_count = 0

        print(f"🖤 開始過濾黑色區域遮罩...")

        for i, mask in enumerate(masks):
            # 確保尺寸匹配
            if mask.shape != black_mask.shape:
                mask = cv2.resize(mask.astype(np.uint8),
                                (black_mask.shape[1], black_mask.shape[0]))

            # 計算與黑色區域的重疊
            mask_binary = mask > 0
            overlap_pixels = np.sum(mask_binary & black_mask)
            mask_pixels = np.sum(mask_binary)

            if mask_pixels > 0:
                overlap_ratio = overlap_pixels / mask_pixels

                # 🔧 過濾條件：與黑色區域重疊超過閾值的遮罩
                if overlap_ratio > overlap_threshold:
                    black_filtered_count += 1
                    print(f"  🖤 過濾黑色遮罩 {i}: 黑色重疊 {overlap_ratio:.1%}")
                else:
                    filtered_masks.append(mask)
                    print(f"  ✅ 保留遮罩 {i}: 黑色重疊 {overlap_ratio:.1%}")
            else:
                filtered_masks.append(mask)

        print(f"🖤 黑色區域過濾完成: 移除 {black_filtered_count} 個黑色遮罩，保留 {len(filtered_masks)} 個")

        return filtered_masks, black_filtered_count

    except Exception as e:
        print(f"❌ 黑色區域過濾失敗: {e}")
        return masks, 0

# ===== 🆕 遮罩檔案生成模組 =====

def save_masks_to_files(masks, output_dir, prefix="mask"):
    """
    保存每個遮罩為單獨的PNG檔案

    Args:
        masks: 遮罩列表
        output_dir: 輸出目錄
        prefix: 檔案前綴

    Returns:
        mask_files: 保存的遮罩檔案路徑列表
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        mask_files = []

        print(f"💾 保存 {len(masks)} 個個別遮罩檔案到: {output_path}")

        for i, mask in enumerate(masks):
            # 生成檔名
            filename = f"{prefix}_{i:04d}.png"
            file_path = output_path / filename

            # 確保遮罩是二值化的 uint8 格式
            if mask.dtype == bool:
                mask_uint8 = mask.astype(np.uint8) * 255
            elif mask.dtype == np.float32 or mask.dtype == np.float64:
                mask_uint8 = (mask * 255).astype(np.uint8)
            else:
                mask_uint8 = mask.astype(np.uint8)

            # 保存遮罩檔案
            success = cv2.imwrite(str(file_path), mask_uint8)

            if success:
                mask_files.append(str(file_path))
                print(f"  ✅ 保存遮罩: {filename}")
            else:
                print(f"  ❌ 保存失敗: {filename}")

        print(f"📁 所有遮罩檔案已保存完成")
        return mask_files

    except Exception as e:
        print(f"❌ 保存個別遮罩檔案失敗: {e}")
        return []

# ===== 🆕 結果圖像生成模組 =====

def create_overview_image(image, masks, unsegmented_color=(255, 20, 147), image_name="overview"):
    """創建總覽圖：未分割區域用指定顏色，已分割區域用原圖"""
    try:
        print(f"🎨 創建{image_name}圖像...")

        # 創建總遮罩（所有分割到的區域）
        total_mask = np.zeros(image.shape[:2], dtype=bool)

        for mask in masks:
            # 確保尺寸匹配
            if mask.shape != image.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8),
                                (image.shape[1], image.shape[0]))

            total_mask |= (mask > 0)

        # 創建結果圖像
        result_image = image.copy()

        # 未分割區域用指定顏色填充
        result_image[~total_mask] = unsegmented_color

        # 統計
        segmented_ratio = np.sum(total_mask) / total_mask.size
        print(f"  📊 已分割區域比例: {segmented_ratio:.1%}")
        print(f"  🎨 未分割區域顏色: RGB{unsegmented_color}")

        return result_image

    except Exception as e:
        print(f"❌ 創建{image_name}圖像失敗: {e}")
        return image.copy()

# 🆕 新增：創建未分割成功展示圖
def create_unsegmented_display_image(image, masks, image_name="未分割展示"):
    """
    創建未分割成功展示圖：未分割區域用原始圖像，已分割區域用黑色

    Args:
        image: 原始圖像
        masks: 分割遮罩列表
        image_name: 圖像名稱

    Returns:
        result_image: 處理後的圖像
    """
    try:
        print(f"🎨 創建{image_name}圖像...")

        # 創建總遮罩（所有分割到的區域）
        total_mask = np.zeros(image.shape[:2], dtype=bool)

        for mask in masks:
            # 確保尺寸匹配
            if mask.shape != image.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8),
                                (image.shape[1], image.shape[0]))

            total_mask |= (mask > 0)

        # 創建結果圖像（從原始圖像開始）
        result_image = image.copy()

        # 已分割區域用黑色填充
        black_color = [0, 0, 0]  # 黑色
        result_image[total_mask] = black_color

        # 統計
        segmented_ratio = np.sum(total_mask) / total_mask.size
        unsegmented_ratio = 1 - segmented_ratio

        print(f"  📊 未分割區域比例: {unsegmented_ratio:.1%}")
        print(f"  📊 已分割區域比例: {segmented_ratio:.1%}")
        print(f"  🖤 已分割區域顏色: 黑色")
        print(f"  🌈 未分割區域: 保持原始圖像")

        return result_image

    except Exception as e:
        print(f"❌ 創建{image_name}圖像失敗: {e}")
        return image.copy()

# ===== 🆕 單次分割參數配置 =====

class SinglePassSegmentationConfig:
    """單次分割參數配置類"""

    def __init__(self, custom_params=None):
        # 🔧 單次分割參數 (精細分割優化版) - 預設值
        default_params = {
            'points_per_side': 48,           # 從32增加到48 - 大幅提升採樣點密度48
            'points_per_batch': 64,         # 保持128 - 維持良好的批次處理效率64
            'pred_iou_thresh': 0.75,         # 從0.75降到0.60 - 接受更多候選遮罩0.75
            'stability_score_thresh': 0.9,  # 從0.85降到0.70 - 降低穩定性門檻0.9
            'stability_score_offset': 0.9,  # 從0.9降到0.70 - 更寬鬆的偏移量0.9
            'crop_n_layers': 2,              # 從1增加到2 - 增加裁剪層次提升小區域精度 2
            'box_nms_thresh': 0.70,          # 從0.7降到0.50 - 更嚴格抑制重疊遮罩0.7
            'crop_n_points_downscale_factor': 2,  # 從2改為1 - 取消裁剪降採樣2
            'min_mask_region_area': 10000,    # 從5000降到1000 - 捕捉更小的物件10000
            'use_m2m': True,                 # 保持開啟 - 使用遮罩精煉功能
        }

        # 🆕 使用自定義參數覆蓋預設值
        self.segmentation_params = default_params.copy()
        if custom_params:
            print("🔧 使用自定義分割參數:")
            for key, value in custom_params.items():
                if key in self.segmentation_params:
                    old_value = self.segmentation_params[key]
                    self.segmentation_params[key] = value
                    print(f"  - {key}: {old_value} → {value}")
                else:
                    print(f"  - 跳過未知參數: {key} = {value}")

        # 黑色區域檢測參數
        self.black_detection_params = {
            'black_threshold': 20,
            'black_overlap_threshold': 0.5
        }

        # 🆕 像素面積過濾參數 - 設為 500 像素
        self.area_filter_params = {
            'min_area_threshold': 800,  # 🔧 您要求的 500 像素閾值
        }

        print("✅ 單次分割參數配置已初始化")
        print(f"🚀 執行單次分割...")
        print("📋 當前分割參數:")
        for param_name, param_value in self.segmentation_params.items():
            print(f"  - {param_name}: {param_value}")
        print(f"  🎯 像素面積過濾: {self.area_filter_params['min_area_threshold']} 像素")

class SAM2Segmenter:
    """SAM2 圖像分割器 - 單次分割策略，完整結果生成 + 500像素過濾"""

    def __init__(self, checkpoint_path=None, device="auto", enable_quality_enhancement=True, custom_params=None):
        """初始化 SAM2 模型"""
        if checkpoint_path is None:
            checkpoint_path = str(CHECKPOINT_PATH)

        self.checkpoint_path = checkpoint_path
        self.enable_quality_enhancement = enable_quality_enhancement

        # 設定設備
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"正在載入 SAM2 模型...")
        print(f"權重檔案: {checkpoint_path}")
        print(f"使用設備: {self.device}")
        print(f"分割策略: 單次分割 + 500像素過濾")

        # 🆕 初始化單次分割配置（可選自定義參數）
        self.config = SinglePassSegmentationConfig(custom_params=custom_params)

        # 載入 SAM2 模型
        self._load_sam2_model_with_correct_path()

    def _load_sam2_model_with_correct_path(self):
        """使用專案內的配置檔案路徑載入 SAM2 模型"""
        # 使用專案根目錄的配置檔案
        config_path = CONFIG_PATH
        print(f"使用專案配置檔案路徑: {config_path}")

        # 驗證配置檔案是否存在
        if not config_path.exists():
            raise FileNotFoundError(f"配置檔案不存在: {config_path}\n請確認 configs/sam2.1_hiera_l.yaml 檔案是否存在")

        # 驗證模型權重是否存在
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"模型權重不存在: {self.checkpoint_path}\n請確認 checkpoint/sam2.1_hiera_large.pt 檔案是否存在")

        try:
            # 載入 SAM2 模型
            self.sam2_model = build_sam2(str(config_path), self.checkpoint_path, device=self.device)
            print("✅ SAM2 模型載入成功！")

        except Exception as e:
            print(f"❌ SAM2 模型載入失敗: {e}")
            raise Exception(f"SAM2 模型載入失敗: {e}")

    def _create_mask_generator(self, pass_params):
        """根據指定參數創建遮罩生成器"""
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

            # 將模型添加到參數中
            generator_params = pass_params.copy()
            generator_params['model'] = self.sam2_model

            print("📋 當前分割參數:")
            for param_name, param_value in generator_params.items():
                if param_name != 'model':
                    print(f"  - {param_name}: {param_value}")

            mask_generator = SAM2AutomaticMaskGenerator(**generator_params)

            # 驗證 generate 方法
            if hasattr(mask_generator, 'generate'):
                print("✅ 遮罩生成器創建成功")
            else:
                raise Exception("遮罩生成器缺少必要的 generate 方法")

            return mask_generator

        except Exception as e:
            print(f"❌ 創建遮罩生成器失敗: {e}")
            raise Exception(f"創建遮罩生成器失敗: {e}")

    def _load_and_validate_image(self, image_input):
        """🔧 正確處理透明圖片的載入"""
        try:
            # 處理不同類型的輸入
            if isinstance(image_input, str):
                # 字符串路徑
                image_path = Path(image_input)

                if not image_path.exists():
                    raise FileNotFoundError(f"圖像檔案不存在: {image_input}")

                # 檢查檔案副檔名
                valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
                if image_path.suffix.lower() not in valid_extensions:
                    raise ValueError(f"不支援的圖像格式: {image_path.suffix}")

                # 🔧 關鍵修正：正確處理透明圖片
                if image_path.suffix.lower() == '.png':
                    # 使用 PIL 載入 PNG 以確保正確處理透明度
                    pil_image = Image.open(str(image_path))

                    if pil_image.mode == 'RGBA':
                        print(f"🌤️ 檢測到透明圖片: {image_path.name}")

                        # 🔧 將透明區域設為純白色背景
                        background = Image.new('RGB', pil_image.size, (255, 255, 255))
                        rgb_image = Image.alpha_composite(background.convert('RGBA'), pil_image)
                        final_image = rgb_image.convert('RGB')

                        print("✅ 透明區域已轉換為白色背景")

                        # 轉換為 numpy 陣列
                        image = np.array(final_image)

                    else:
                        # 非透明 PNG，正常處理
                        image = np.array(pil_image.convert('RGB'))

                else:
                    # 非 PNG 檔案，使用 OpenCV 載入
                    image = cv2.imread(str(image_path))
                    if image is None:
                        raise ValueError(f"OpenCV 無法載入圖像: {image_input}")

                    # 轉換為 RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            elif isinstance(image_input, np.ndarray):
                # NumPy 陣列
                image = image_input
            else:
                raise TypeError(f"不支援的圖像輸入類型: {type(image_input)}")

            # 驗證圖像格式
            if not hasattr(image, 'shape'):
                raise AttributeError(f"圖像對象缺少 shape 屬性")

            if len(image.shape) != 3:
                raise ValueError(f"圖像必須是3維陣列 (H, W, C)，實際: {image.shape}")

            if image.shape[2] != 3:
                raise ValueError(f"圖像必須是3通道 RGB，實際通道數: {image.shape[2]}")

            # 檢查圖像尺寸
            height, width = image.shape[:2]
            if height < 10 or width < 10:
                raise ValueError(f"圖像尺寸太小: {width}x{height}")

            if height > 10000 or width > 10000:
                print(f"⚠️ 警告：圖像尺寸很大 ({width}x{height})，可能影響處理速度")

            return image

        except Exception as e:
            print(f"❌ 圖像載入失敗: {e}")
            raise e

    def segment_image(self, image, save_individual_masks=True, output_dir=None):
        """
        🆕 單次分割主函數 - 完整結果生成 + 500像素過濾 + 未分割展示圖
        """
        try:
            # 🔧 載入和驗證圖像
            if isinstance(image, str):
                print(f"🔄 載入圖像從路徑: {image}")
                original_image_path = image
                image = self._load_and_validate_image(image)
            elif isinstance(image, np.ndarray):
                print(f"🔄 驗證圖像陣列格式...")
                original_image_path = None
                image = self._load_and_validate_image(image)
            else:
                raise TypeError(f"不支援的圖像類型: {type(image)}")

            print(f"✅ 圖像驗證成功，尺寸: {image.shape}")

            # 🆕 檢測黑色區域
            black_mask = detect_black_regions(image, self.config.black_detection_params['black_threshold'])

            # 🆕 單次分割
            print(f"\n🚀 執行單次分割...")
            mask_generator = self._create_mask_generator(self.config.segmentation_params)
            masks_raw = mask_generator.generate(image)
            print(f"✅ 單次分割完成，生成 {len(masks_raw)} 個原始遮罩")

            # 處理分割結果
            masks = []
            for mask_data in masks_raw:
                masks.append(mask_data['segmentation'])

            # 黑色區域過濾
            masks_filtered, black_filtered = filter_black_region_masks(
                masks, black_mask, self.config.black_detection_params['black_overlap_threshold']
            )

            # 🆕 像素面積過濾（500像素）
            final_masks, area_filtered = filter_masks_by_min_area(
                masks_filtered,
                self.config.area_filter_params['min_area_threshold'],
                "單次分割"
            )

            print(f"📊 單次分割結果: 保留 {len(final_masks)} 個有效遮罩")

            # 保存個別遮罩檔案
            mask_results = {}
            if save_individual_masks and output_dir:
                masks_dir = Path(output_dir) / "single_pass_masks"
                mask_files = save_masks_to_files(final_masks, masks_dir, "mask")
                mask_results['mask_files'] = mask_files
                mask_results['masks_directory'] = str(masks_dir)

            # 🆕 生成結果圖像
            print(f"\n🎨 生成單次分割結果圖像...")
            overview_image = create_overview_image(image, final_masks, (0, 255, 0), "單次分割總覽")

            # 🆕 生成未分割成功展示圖
            unsegmented_display_image = create_unsegmented_display_image(image, final_masks, "未分割成功展示")

            # 🆕 保存結果圖像
            result_images = {}
            if output_dir:
                saved_files = self._save_single_pass_results(
                    image, final_masks, overview_image, unsegmented_display_image, output_dir, original_image_path
                )
                result_images = saved_files

            # 🆕 準備返回結果
            result = {
                'masks': final_masks,
                'num_masks': len(final_masks),
                'image_shape': image.shape,
                'status': 'success',
                'message': f'單次分割完成，總共生成 {len(final_masks)} 個遮罩（已過濾<500像素）',

                'segmentation_results': {
                    'num_masks': len(final_masks),
                    'black_filtered_count': black_filtered,
                    'area_filtered_count': area_filtered,
                    'mask_files': mask_results.get('mask_files', []),
                    'masks_directory': mask_results.get('masks_directory', ''),
                },

                'black_region_ratio': np.sum(black_mask) / black_mask.size,
                'area_filtering_stats': {
                    'min_area_threshold': self.config.area_filter_params['min_area_threshold'],
                    'total_filtered': area_filtered
                },
                'result_images': result_images,
                'original_image_path': original_image_path,
                'processing_timestamp': datetime.now().isoformat()
            }

            # 🆕 計算額外的元資料
            if len(final_masks) > 0:
                centroids, areas, bboxes, confidences, mask_arrays = self._calculate_metadata(final_masks)
                result.update({
                    'centroids': centroids,
                    'areas': areas,
                    'bboxes': bboxes,
                    'confidences': confidences,
                    'mask_arrays': mask_arrays
                })

            return result

        except Exception as e:
            print(f"💥 單次分割失敗: {e}")
            import traceback
            traceback.print_exc()

            return {
                'masks': [],
                'num_masks': 0,
                'image_shape': image.shape if 'image' in locals() else None,
                'status': 'error',
                'error': str(e),
                'message': f'單次分割失敗: {e}',
                'result_images': {}
            }

    def _calculate_metadata(self, masks):
        """計算遮罩元資料（保持相容性）"""
        centroids = []
        areas = []
        bboxes = []
        confidences = []
        mask_arrays = []

        for mask in masks:
            # 質心
            y_coords, x_coords = np.where(mask > 0.5)
            if len(x_coords) > 0:
                centroid_x = float(np.mean(x_coords))
                centroid_y = float(np.mean(y_coords))
                centroids.append((centroid_x, centroid_y))
            else:
                centroids.append((0.0, 0.0))

            # 面積
            area = int(np.sum(mask > 0.5))
            areas.append(area)

            # 邊界框
            if len(x_coords) > 0:
                bbox = [float(np.min(x_coords)), float(np.min(y_coords)),
                       float(np.max(x_coords)), float(np.max(y_coords))]
            else:
                bbox = [0.0, 0.0, 0.0, 0.0]
            bboxes.append(bbox)

            # 信心分數
            confidences.append(1.0)

            # 遮罩陣列
            mask_array = (mask > 0.5).astype(np.uint8) * 255
            mask_arrays.append(mask_array)

        return centroids, areas, bboxes, confidences, mask_arrays

    def _save_single_pass_results(self, original_image, masks, overview_image, unsegmented_display_image,
                                 output_dir, original_image_path):
        """保存單次分割結果 + 未分割展示圖"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            saved_files = {}

            # 生成檔案名稱前綴
            if original_image_path:
                image_name = Path(original_image_path).stem
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_name = f"{timestamp}_image"

            # 保存原始圖像
            image1_filename = "image_1.jpg"
            image1_path = output_path / image1_filename
            original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(image1_path), original_bgr)
            saved_files['image_1'] = str(image1_path)
            print(f"✅ 保存原始圖像: {image1_filename}")

            # 單次分割總覽圖
            overview_filename = f"{image_name}_single_pass_overview.jpg"
            overview_path = output_path / overview_filename
            overview_bgr = cv2.cvtColor(overview_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(overview_path), overview_bgr)
            saved_files['single_pass_overview'] = str(overview_path)
            print(f"✅ 保存單次分割總覽圖: {overview_filename}")

            # 🆕 保存未分割展示圖
            unsegmented_filename = f"{image_name}_unsegmented_display.jpg"
            unsegmented_path = output_path / unsegmented_filename
            unsegmented_bgr = cv2.cvtColor(unsegmented_display_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(unsegmented_path), unsegmented_bgr)
            saved_files['unsegmented_display'] = str(unsegmented_path)
            print(f"✅ 保存未分割展示圖: {unsegmented_filename}")

            # 保存完整的 pickle 檔案
            pickle_filename = f"{image_name}_single_pass_complete.pkl"
            pickle_path = output_path / pickle_filename

            complete_data = {
                'original_image_path': original_image_path,
                'processing_timestamp': datetime.now().isoformat(),
                'segmentation_strategy': 'single_pass_segmentation',
                'min_area_threshold': self.config.area_filter_params['min_area_threshold'],
                'masks': masks,
                'saved_files': saved_files,
                'parameters': {
                    'segmentation_pass': self.config.segmentation_params,
                    'black_detection': self.config.black_detection_params,
                    'area_filter': self.config.area_filter_params
                }
            }

            with open(pickle_path, 'wb') as f:
                pickle.dump(complete_data, f)

            saved_files['complete_pickle'] = str(pickle_path)
            print(f"✅ 完整單次分割結果已保存: {pickle_path}")

            return saved_files

        except Exception as e:
            print(f"⚠️ 保存結果時發生錯誤: {e}")
            return {}

# ===== 🆕 API 封裝函式 =====

def segment_image_api(image_path, output_dir=None, checkpoint_path=None, device="auto",
                     save_individual_masks=True, enable_quality_enhancement=True):
    """
    🆕 API 封裝函式：執行單次 SAM2 圖像分割 + 500像素過濾 + 未分割展示圖
    """
    try:
        # 驗證圖像檔案是否存在
        if not os.path.exists(image_path):
            return {
                'status': 'error',
                'message': '找不到圖像檔案',
                'error': f'File not found: {image_path}'
            }

        # 🆕 初始化單次分割 SAM2 分割器
        segmenter = SAM2Segmenter(
            checkpoint_path=checkpoint_path,
            device=device,
            enable_quality_enhancement=enable_quality_enhancement
        )

        # 執行分割
        result = segmenter.segment_image(
            image_path,
            save_individual_masks=save_individual_masks,
            output_dir=output_dir
        )

        if result['status'] == 'error':
            return {
                'status': 'error',
                'message': result['message'],
                'error': result.get('error', 'Unknown error')
            }

        # 準備回應資料
        response_data = {
            'num_masks': result['num_masks'],
            'image_shape': result['image_shape'],
            'segmentation_strategy': '單次分割 + 500像素過濾',
            'area_filtering_stats': result.get('area_filtering_stats', {}),
            'segmentation_results': result.get('segmentation_results', {}),
            'black_region_ratio': result.get('black_region_ratio', 0),
            'result_images_info': {
                'image_1': '原始圖像',
                'single_pass_overview': '單次分割總覽（未分割=亮綠色）',
                'unsegmented_display': '未分割展示圖（未分割=原圖，已分割=黑色）'  # 🆕 新增
            },
            'masks_metadata': {
                'areas': result.get('areas', []),
                'bboxes': result.get('bboxes', []),
                'centroids': result.get('centroids', []),
                'confidences': result.get('confidences', [])
            }
        }

        # 如果有輸出目錄，加入檔案路徑
        if output_dir and result.get('result_images'):
            response_data.update({
                'output_directory': str(output_dir),
                'saved_files': result['result_images'],
                'original_image_path': str(image_path)
            })

        return {
            'status': 'success',
            'message': f'單次SAM2圖像分割完成（已過濾<500像素遮罩，含未分割展示圖）',
            'data': response_data
        }

    except Exception as e:
        import traceback
        return {
            'status': 'error',
            'message': '單次SAM2分割過程發生錯誤',
            'error': str(e),
            'traceback': traceback.format_exc()
        }

def segment_multiple_images_api(image_paths, output_base_dir=None, checkpoint_path=None,
                               device="auto", save_individual_masks=True, enable_quality_enhancement=True,
                               points_per_side=None, points_per_batch=None, pred_iou_thresh=None,
                               stability_score_thresh=None, stability_score_offset=None, min_mask_region_area=None):
    """批次處理多張圖像的單次 SAM2 分割 + 500像素過濾 + 未分割展示圖"""
    try:
        print(f"🔄 開始批次單次分割處理 {len(image_paths)} 張圖像")

        results = {
            'status': 'success',
            'data': {
                'results': [],
                'total_images': len(image_paths),
                'processed_images': 0,
                'total_masks_generated': 0,
                'failed_images': 0,
                'failures': [],
                'output_base_directory': str(output_base_dir) if output_base_dir else None,
                'segmentation_info': {
                    'strategy': '單次分割 + 500像素過濾 + 未分割展示圖',
                    'quality_enhancement': enable_quality_enhancement,
                    'average_masks_per_image': 0
                }
            }
        }

        # 🆕 創建自定義參數字典
        custom_params = {}
        if points_per_side is not None:
            custom_params['points_per_side'] = points_per_side
        if points_per_batch is not None:
            custom_params['points_per_batch'] = points_per_batch
        if pred_iou_thresh is not None:
            custom_params['pred_iou_thresh'] = pred_iou_thresh
        if stability_score_thresh is not None:
            custom_params['stability_score_thresh'] = stability_score_thresh
        if stability_score_offset is not None:
            custom_params['stability_score_offset'] = stability_score_offset
        if min_mask_region_area is not None:
            custom_params['min_mask_region_area'] = min_mask_region_area

        # 🆕 創建單次分割器實例（傳遞自定義參數）
        segmenter = SAM2Segmenter(
            checkpoint_path=checkpoint_path,
            device=device,
            enable_quality_enhancement=enable_quality_enhancement,
            custom_params=custom_params if custom_params else None
        )

        for i, image_path in enumerate(image_paths, 1):
            try:
                print(f"🖼️ 處理圖像 {i}/{len(image_paths)}: {Path(image_path).name}")

                # 驗證圖像路徑
                if not isinstance(image_path, str):
                    raise TypeError(f"圖像路徑必須是字符串，實際類型: {type(image_path)}")

                if not Path(image_path).exists():
                    raise FileNotFoundError(f"圖像檔案不存在: {image_path}")

                # 創建個別輸出目錄
                if output_base_dir:
                    image_name = Path(image_path).stem
                    image_output_dir = Path(output_base_dir) / f"image{i}_results"
                    image_output_dir.mkdir(parents=True, exist_ok=True)
                else:
                    image_output_dir = None

                # 🆕 執行單次分割
                print(f"  🚀 開始單次分割...")
                segment_result = segmenter.segment_image(
                    image_path,
                    save_individual_masks=save_individual_masks,
                    output_dir=str(image_output_dir) if image_output_dir else None
                )

                if segment_result and segment_result.get('status') == 'success':
                    num_masks = segment_result.get('num_masks', 0)

                    if num_masks > 0:
                        # 添加到結果
                        image_result = {
                            'image_index': i,
                            'image_path': str(image_path),
                            'result': {
                                'status': 'success',
                                'num_masks': num_masks,
                                'black_region_ratio': segment_result.get('black_region_ratio', 0),
                                'result_images_generated': len(segment_result.get('result_images', {})),
                                'original_image_path': str(image_path),
                                'output_directory': str(image_output_dir) if image_output_dir else None
                            }
                        }

                        results['data']['results'].append(image_result)
                        results['data']['processed_images'] += 1
                        results['data']['total_masks_generated'] += num_masks

                        print(f"  ✅ 圖像 {i} 單次分割成功，生成 {num_masks} 個遮罩（已過濾<500像素，含未分割展示圖）")
                    else:
                        print(f"  ⚠️ 圖像 {i} 沒有生成遮罩")
                        raise Exception("沒有生成任何遮罩")

                else:
                    error_msg = segment_result.get('error', '單次分割失敗') if segment_result else '未知錯誤'
                    raise Exception(error_msg)

            except Exception as e:
                print(f"  ❌ 圖像 {i} 單次分割失敗: {e}")

                failure_info = {
                    'image_index': i,
                    'image_path': str(image_path),
                    'error': str(e)
                }

                results['data']['failures'].append(failure_info)
                results['data']['failed_images'] += 1

        # 更新最終狀態
        if results['data']['processed_images'] > 0:
            # 計算平均遮罩數
            avg_masks = results['data']['total_masks_generated'] / results['data']['processed_images']
            results['data']['segmentation_info']['average_masks_per_image'] = round(avg_masks, 1)

            results['message'] = f"批次單次分割完成：{results['data']['processed_images']}/{results['data']['total_images']} 張圖像成功處理"
            if results['data']['failed_images'] == 0:
                results['status'] = 'success'
            else:
                results['status'] = 'partial_success'
        else:
            results['message'] = f"批次單次分割失敗：所有 {results['data']['total_images']} 張圖像都處理失敗"
            results['status'] = 'error'

        print(f"📊 批次單次分割完成:")
        print(f"  - 成功: {results['data']['processed_images']} 張")
        print(f"  - 失敗: {results['data']['failed_images']} 張")
        print(f"  - 總遮罩數: {results['data']['total_masks_generated']}")

        return results

    except Exception as e:
        print(f"💥 批次單次分割發生嚴重錯誤: {e}")
        import traceback
        traceback.print_exc()

        return {
            'status': 'error',
            'message': f'批次單次分割失敗: {str(e)}',
            'error': str(e),
            'data': {
                'results': [],
                'total_images': len(image_paths) if 'image_paths' in locals() else 0,
                'processed_images': 0,
                'failed_images': len(image_paths) if 'image_paths' in locals() else 0
            }
        }

# ===== 保持原有相容性的函式 =====

def load_masks_from_pickle(pickle_path_1, pickle_path_2):
    """從兩個 pickle 檔案載入遮罩資料（用於遮罩匹配）"""
    try:
        print(f"🔄 載入單次分割遮罩資料...")
        print(f"  檔案1: {pickle_path_1}")
        print(f"  檔案2: {pickle_path_2}")

        if not os.path.exists(pickle_path_1):
            print(f"❌ 找不到檔案1: {pickle_path_1}")
            return None, None

        if not os.path.exists(pickle_path_2):
            print(f"❌ 找不到檔案2: {pickle_path_2}")
            return None, None

        with open(pickle_path_1, 'rb') as f:
            masks_data_1 = pickle.load(f)

        with open(pickle_path_2, 'rb') as f:
            masks_data_2 = pickle.load(f)

        print(f"✅ 成功載入單次分割遮罩資料:")
        print(f"  檔案1 遮罩數量: {len(masks_data_1.get('masks', []))}")
        print(f"  檔案1 分割策略: {masks_data_1.get('segmentation_strategy', 'N/A')}")
        print(f"  檔案2 遮罩數量: {len(masks_data_2.get('masks', []))}")
        print(f"  檔案2 分割策略: {masks_data_2.get('segmentation_strategy', 'N/A')}")

        return masks_data_1, masks_data_2

    except Exception as e:
        print(f"❌ 載入遮罩資料失敗: {e}")
        return None, None

def run_sam2_segmentation():
    """🆕 執行單次 SAM2 分割（保持原有介面相容性）"""
    # 🔧 修改：使用統一的專案路徑
    website_root = WEBSITE_ROOT
    results_root = RESULTS_ROOT

    # 圖像路徑 - 從 uploads 目錄讀取
    image1_path = results_root / "uploads" / "frame_00008.jpg"
    image2_path = results_root / "uploads" / "frame_00007.jpg"
    output_dir = results_root / "segmentation"

    # 確保必要目錄存在
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🏠 使用專案根目錄: {website_root}")
    print(f"📁 圖像1路徑: {image1_path}")
    print(f"📁 圖像2路徑: {image2_path}")
    print(f"📁 輸出目錄: {output_dir}")

    try:
        # 🆕 初始化單次分割 SAM2 分割器
        segmenter = SAM2Segmenter(
            checkpoint_path=str(CHECKPOINT_PATH),
            device="auto",
            enable_quality_enhancement=True
        )

        print("=== 單次 SAM2 圖像分割處理 + 500像素過濾 + 未分割展示圖 ===")
        print("處理第一張圖像...")
        result_1 = segmenter.segment_image(
            str(image1_path),
            output_dir=str(output_dir / "image1_results"),
            save_individual_masks=True
        )

        print("\n處理第二張圖像...")
        result_2 = segmenter.segment_image(
            str(image2_path),
            output_dir=str(output_dir / "image2_results"),
            save_individual_masks=True
        )

        print(f"\n=== 單次 SAM2 分割完成 ===")
        print(f"圖像1 總遮罩數量: {result_1['num_masks']}（已過濾<500像素）")
        print(f"  過濾統計: {result_1.get('area_filtering_stats', {})}")

        print(f"圖像2 總遮罩數量: {result_2['num_masks']}（已過濾<500像素）")
        print(f"  過濾統計: {result_2.get('area_filtering_stats', {})}")

        print("✅ 單次分割、黑色區域排除、500像素過濾、完整結果生成功能已啟用！")
        print("📁 生成了個別遮罩檔案")
        print("🎨 生成了結果圖像（包含未分割展示圖）")
        print("🖤 黑色區域已自動排除分割")
        print("🔍 已過濾所有小於500像素的遮罩")
        print("🆕 新增：未分割展示圖（未分割=原圖，已分割=黑色）")
        print("可以開始進行遮罩對應分析了！")

        return result_1, result_2

    except Exception as e:
        print(f"❌ 單次 SAM2 分割失敗: {e}")
        print(f"\n故障排除建議：")
        print("1. 確認圖像檔案存在於 results/uploads/ 目錄")
        print("2. 驗證 SAM2 模型檔案完整性")
        print(f"3. 確認模型權重路徑: {CHECKPOINT_PATH}")
        print(f"4. 確認配置檔案路徑: {CONFIG_PATH}")
        import traceback
        traceback.print_exc()
        return None, None

# 主要執行程式
if __name__ == "__main__":
    results = run_sam2_segmentation()

    if results[0] is not None:
        print("✅ 單次 SAM2 分割處理完成！")
        print("🎨 功能: 單次分割、個別遮罩檔案、結果圖像、黑色區域排除、未分割展示圖")
        print("🔍 像素過濾: 已過濾所有小於500像素的遮罩")
        print("🆕 新增圖像: 未分割成功展示圖（未分割=原圖，已分割=黑色）")
        print("📁 所有遮罩檔案和結果圖像已生成，可進行下一步的遮罩匹配分析")
    else:
        print("❌ 單次 SAM2 分割處理失敗，請檢查配置")
