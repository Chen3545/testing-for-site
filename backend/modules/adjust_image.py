import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path
import json

class ECCImageAligner:
    """
    專為定點攝影機設計的ECC圖像對齊器
    針對5K解析度室外圖像優化
    """

    def __init__(self, pyramid_levels=4, motion_type='EUCLIDEAN'):
        """
        初始化ECC對齊器
        Args:
            pyramid_levels: 金字塔層數（5K圖像建議4層）
            motion_type: 運動模型 ('TRANSLATION', 'EUCLIDEAN', 'AFFINE', 'HOMOGRAPHY')
        """
        self.pyramid_levels = pyramid_levels

        self.motion_models = {
            'TRANSLATION': cv2.MOTION_TRANSLATION,
            'EUCLIDEAN': cv2.MOTION_EUCLIDEAN,
            'AFFINE': cv2.MOTION_AFFINE,
            'HOMOGRAPHY': cv2.MOTION_HOMOGRAPHY
        }

        self.motion_type = self.motion_models.get(motion_type, cv2.MOTION_EUCLIDEAN)

        # 針對5K圖像的優化參數
        self.pyramid_configs = [
            {'iterations': 50, 'threshold': 1e-4},   # 粗糙層
            {'iterations': 100, 'threshold': 1e-5},  # 中等層
            {'iterations': 200, 'threshold': 1e-6},  # 精細層
            {'iterations': 200, 'threshold': 1e-7}   # 原始層
        ]

    def create_image_pyramid(self, image, levels):
        """創建圖像金字塔"""
        pyramid = [image]
        current = image.copy()

        for i in range(levels - 1):
            current = cv2.pyrDown(current)
            pyramid.insert(0, current)  # 從粗糙到精細排列

        return pyramid

    def initialize_warp_matrix(self, motion_type):
        """初始化變換矩陣"""
        if motion_type == cv2.MOTION_TRANSLATION:
            return np.eye(2, 3, dtype=np.float32)
        elif motion_type == cv2.MOTION_EUCLIDEAN:
            return np.eye(2, 3, dtype=np.float32)
        elif motion_type == cv2.MOTION_AFFINE:
            return np.eye(2, 3, dtype=np.float32)
        else:  # HOMOGRAPHY
            return np.eye(3, 3, dtype=np.float32)

    def scale_warp_matrix(self, warp_matrix, scale_factor):
        """縮放變換矩陣"""
        if warp_matrix.shape[0] == 2:  # 2x3 matrix
            scaled_matrix = warp_matrix.copy()
            scaled_matrix[0, 2] *= scale_factor  # x translation
            scaled_matrix[1, 2] *= scale_factor  # y translation
            return scaled_matrix
        else:  # 3x3 matrix (homography)
            scaled_matrix = warp_matrix.copy()
            scaled_matrix[0, 2] *= scale_factor
            scaled_matrix[1, 2] *= scale_factor
            return scaled_matrix

    def analyze_alignment(self, warp_matrix, image_shape):
        """分析對齊結果 - JSON 相容版本"""
        analysis = {}

        if warp_matrix.shape[0] == 2:  # Affine transformation
            # 提取平移量並轉換為 Python 原生類型
            dx = float(warp_matrix[0, 2])
            dy = float(warp_matrix[1, 2])
            analysis['translation_x'] = dx
            analysis['translation_y'] = dy
            analysis['total_displacement'] = float(np.sqrt(dx**2 + dy**2))

            # 提取旋轉角度（對於歐式變換）
            if self.motion_type == cv2.MOTION_EUCLIDEAN:
                rotation_angle = float(np.arctan2(warp_matrix[1, 0], warp_matrix[0, 0]))
                analysis['rotation_degrees'] = float(np.degrees(rotation_angle))

            # 計算位移百分比
            h, w = image_shape
            analysis['displacement_percent_x'] = float(abs(dx) / w * 100)
            analysis['displacement_percent_y'] = float(abs(dy) / h * 100)

        return analysis

    def align_images(self, ref_image_path, input_image_path, output_dir=None):
        """
        執行圖像對齊
        Args:
            ref_image_path: 參考圖像路徑
            input_image_path: 待對齊圖像路徑
            output_dir: 輸出目錄
        Returns:
            tuple: (success, results_dict)
        """
        try:
            # 讀取圖像
            ref_image = cv2.imread(str(ref_image_path))
            input_image = cv2.imread(str(input_image_path))

            if ref_image is None or input_image is None:
                print("錯誤: 無法讀取圖像文件")
                return False, {"error": "無法讀取圖像文件"}

            print(f"圖像尺寸: {ref_image.shape[:2]}")

            # 轉換為灰度圖
            ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
            input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

            # 創建圖像金字塔
            ref_pyramid = self.create_image_pyramid(ref_gray, self.pyramid_levels)
            input_pyramid = self.create_image_pyramid(input_gray, self.pyramid_levels)

            # 初始化變換矩陣
            warp_matrix = self.initialize_warp_matrix(self.motion_type)

            print("開始多層ECC對齊...")

            # 從粗糙到精細逐層對齊
            for level in range(self.pyramid_levels):
                print(f"處理金字塔層級 {level + 1}/{self.pyramid_levels}")

                ref_level = ref_pyramid[level]
                input_level = input_pyramid[level]

                config = self.pyramid_configs[min(level, len(self.pyramid_configs) - 1)]

                # 設定ECC算法參數
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                           config['iterations'], config['threshold'])

                try:
                    # 執行ECC對齊
                    correlation, warp_matrix = cv2.findTransformECC(
                        ref_level, input_level, warp_matrix,
                        self.motion_type, criteria
                    )

                    print(f"層級 {level + 1} 相關係數: {correlation:.6f}")

                    # 如果不是最後一層，放大變換矩陣用於下一層
                    if level < self.pyramid_levels - 1:
                        warp_matrix = self.scale_warp_matrix(warp_matrix, 2.0)

                except cv2.error as e:
                    print(f"警告: 層級 {level + 1} ECC失敗: {e}")
                    if level == 0:  # 如果第一層就失敗
                        return False, {"error": "ECC對齊完全失敗"}
                    break

            # 應用變換到原始圖像
            if self.motion_type == cv2.MOTION_HOMOGRAPHY:
                aligned_image = cv2.warpPerspective(input_image, warp_matrix,
                                                  (ref_image.shape[1], ref_image.shape[0]))
            else:
                aligned_image = cv2.warpAffine(input_image, warp_matrix,
                                             (ref_image.shape[1], ref_image.shape[0]))

            # 分析對齊結果
            analysis = self.analyze_alignment(warp_matrix, ref_image.shape[:2])

            # 準備結果
            results = {
                'aligned_image': aligned_image,
                'warp_matrix': warp_matrix,
                'analysis': analysis,
                'correlation': correlation if 'correlation' in locals() else 0
            }

            # 保存結果
            if output_dir:
                self.save_results(ref_image, input_image, aligned_image,
                                warp_matrix, analysis, output_dir)

            return True, results

        except Exception as e:
            print(f"對齊過程發生錯誤: {e}")
            return False, {"error": str(e)}

    def save_results(self, ref_image, input_image, aligned_image, warp_matrix, analysis, output_dir):
        """保存對齊結果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存對齊後的圖像
        cv2.imwrite(str(output_path / 'aligned_image.jpg'), aligned_image)

        # 保存變換矩陣
        np.save(str(output_path / 'warp_matrix.npy'), warp_matrix)

        # 創建對比圖像
        self.create_comparison_images(ref_image, input_image, aligned_image, output_path)

        # 保存分析報告
        self.save_analysis_report(analysis, warp_matrix, output_path)

        print(f"結果已保存至: {output_path}")

    def create_comparison_images(self, ref_image, input_image, aligned_image, output_path):
        """創建對比圖像"""
        # 並排對比
        h, w = ref_image.shape[:2]
        comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
        comparison[:, :w] = ref_image
        comparison[:, w:2*w] = input_image
        comparison[:, 2*w:] = aligned_image

        # 添加標籤
        cv2.putText(comparison, 'Reference', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (0, 255, 0), 2)
        cv2.putText(comparison, 'Original', (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (0, 0, 255), 2)
        cv2.putText(comparison, 'Aligned', (2*w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (255, 0, 0), 2)

        cv2.imwrite(str(output_path / 'comparison.jpg'), comparison)

        # 差異圖
        diff_before = cv2.absdiff(ref_image, input_image)
        diff_after = cv2.absdiff(ref_image, aligned_image)

        cv2.imwrite(str(output_path / 'diff_before.jpg'), diff_before)
        cv2.imwrite(str(output_path / 'diff_after.jpg'), diff_after)

    def save_analysis_report(self, analysis, warp_matrix, output_path):
        """保存分析報告"""
        with open(output_path / 'alignment_info.txt', 'w', encoding='utf-8') as f:
            f.write("=== ECC圖像對齊分析報告 ===\n\n")
            f.write(f"變換矩陣:\n{warp_matrix}\n\n")

            if 'translation_x' in analysis:
                f.write(f"水平位移: {analysis['translation_x']:.2f} 像素\n")
                f.write(f"垂直位移: {analysis['translation_y']:.2f} 像素\n")
                f.write(f"總位移: {analysis['total_displacement']:.2f} 像素\n\n")

                f.write(f"水平位移百分比: {analysis['displacement_percent_x']:.3f}%\n")
                f.write(f"垂直位移百分比: {analysis['displacement_percent_y']:.3f}%\n\n")

                if 'rotation_degrees' in analysis:
                    f.write(f"旋轉角度: {analysis['rotation_degrees']:.4f}°\n\n")

                # 穩定性評估
                if analysis.get('total_displacement', 0) < 5:
                    f.write("相機穩定性: 優秀 (<5像素位移)\n")
                elif analysis.get('total_displacement', 0) < 20:
                    f.write("相機穩定性: 良好 (5-20像素位移)\n")
                else:
                    f.write("相機穩定性: 需要檢查 (>20像素位移)\n")


# ===== 新增的 API 封裝函式 =====

def align_images_api(ref_image_path, input_image_path, output_dir=None,
                    pyramid_levels=4, motion_type='EUCLIDEAN'):
    """
    🔧 修改版：API 封裝函式：執行圖像對齊並回傳 JSON 格式結果，使用簡化檔名

    Args:
        ref_image_path: 參考圖像路徑
        input_image_path: 待對齊圖像路徑
        output_dir: 輸出目錄（可選）
        pyramid_levels: 金字塔層數
        motion_type: 運動模型類型

    Returns:
        dict: JSON 格式的回應字典
    """
    try:
        print(f"📐 開始圖像對齊處理...")
        print(f"  參考圖像: {Path(ref_image_path).name}")
        print(f"  待對齊圖像: {Path(input_image_path).name}")

        # 驗證輸入檔案
        if not os.path.exists(ref_image_path):
            return {
                'status': 'error',
                'message': '找不到參考圖像檔案',
                'error': f'File not found: {ref_image_path}'
            }

        if not os.path.exists(input_image_path):
            return {
                'status': 'error',
                'message': '找不到待對齊圖像檔案',
                'error': f'File not found: {input_image_path}'
            }

        # 初始化對齊器
        aligner = ECCImageAligner(pyramid_levels=pyramid_levels, motion_type=motion_type)

        # 執行對齊
        success, results = aligner.align_images(ref_image_path, input_image_path, output_dir)

        if success:
            # 準備成功回應的資料
            response_data = {
                'correlation': float(results.get('correlation', 0.0)),
                'analysis': results.get('analysis', {}),
                'warp_matrix': results.get('warp_matrix', []).tolist() if hasattr(results.get('warp_matrix', []), 'tolist') else results.get('warp_matrix', [])
            }

            # 🔧 關鍵修改：如果有輸出目錄，使用簡化的檔名
            if output_dir:
                output_path = Path(output_dir)

                # 🆕 使用簡化的檔名策略
                aligned_files = {
                    'image1_aligned': 'image1_aligned.jpg',     # 對齊後的參考圖像（通常不變）
                    'image2_aligned': 'image2_aligned.jpg',     # 對齊後的輸入圖像
                    'comparison': 'alignment_comparison.jpg',    # 對齊比較圖
                    'analysis_report': 'alignment_info.txt',     # 分析報告
                    'warp_matrix': 'warp_matrix.npy',           # 變形矩陣
                    'diff_before': 'diff_before.jpg',           # 對齊前差異圖
                    'diff_after': 'diff_after.jpg'              # 對齊後差異圖
                }

                response_data.update({
                    'aligned_ref_image_path': str(output_path / aligned_files['image1_aligned']),
                    'aligned_input_image_path': str(output_path / aligned_files['image2_aligned']),
                    'comparison_image_path': str(output_path / aligned_files['comparison']),
                    'analysis_report_path': str(output_path / aligned_files['analysis_report']),
                    'warp_matrix_path': str(output_path / aligned_files['warp_matrix']),
                    'diff_before_path': str(output_path / aligned_files['diff_before']),
                    'diff_after_path': str(output_path / aligned_files['diff_after']),
                    'output_directory': str(output_path),
                    # 🆕 新增：檔案對應資訊
                    'file_mapping': {
                        'reference_image': 'image1_aligned.jpg',  # 參考圖像（通常是複製原檔）
                        'aligned_image': 'image2_aligned.jpg'     # 對齊後的圖像
                    }
                })

            # 🔧 新增：記錄處理結果
            print(f"✅ 圖像對齊成功完成")
            print(f"  相關性係數: {response_data['correlation']:.4f}")
            print(f"  輸出目錄: {output_dir}")

            return {
                'status': 'success',
                'message': '圖像對齊完成',
                'data': response_data
            }
        else:
            # 對齊失敗的回應
            print(f"❌ 圖像對齊失敗: {results.get('error', '未知錯誤')}")
            return {
                'status': 'error',
                'message': '圖像對齊失敗',
                'error': results.get('error', '未知錯誤'),
                'error_details': results
            }

    except FileNotFoundError as e:
        print(f"❌ 檔案未找到: {str(e)}")
        return {
            'status': 'error',
            'message': '找不到指定的圖像檔案',
            'error': str(e)
        }
    except Exception as e:
        print(f"❌ 對齊過程發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'message': '對齊過程發生錯誤',
            'error': str(e)
        }


def create_alignment_response(success, results, output_dir=None):
    """
    輔助函式：創建標準化的回應格式
    """
    if success:
        # 確保 warp_matrix 可以被序列化為 JSON
        warp_matrix = results.get('warp_matrix')
        if warp_matrix is not None and hasattr(warp_matrix, 'tolist'):
            warp_matrix = warp_matrix.tolist()

        response = {
            'status': 'success',
            'data': {
                'correlation': float(results.get('correlation', 0.0)),
                'analysis': results.get('analysis', {}),
                'warp_matrix': warp_matrix
            }
        }

        # 如果有輸出目錄，加入檔案路徑
        if output_dir:
            output_path = Path(output_dir)
            response['data'].update({
                'aligned_image_path': str(output_path / 'aligned_image.jpg'),
                'comparison_image_path': str(output_path / 'comparison.jpg'),
                'analysis_report_path': str(output_path / 'alignment_info.txt'),
                'output_directory': str(output_path)
            })

        return response
    else:
        return {
            'status': 'error',
            'message': results.get('error', '對齊失敗'),
            'error_details': results
        }


def validate_alignment_parameters(params):
    """
    驗證對齊參數

    Args:
        params: 參數字典

    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # 驗證必要參數
        required_params = ['ref_path', 'input_path']
        for param in required_params:
            if param not in params or not params[param]:
                return False, f"缺少必要參數: {param}"

        # 驗證可選參數
        pyramid_levels = params.get('pyramid_levels', 4)
        if not isinstance(pyramid_levels, int) or pyramid_levels < 1 or pyramid_levels > 6:
            return False, "pyramid_levels 必須是 1-6 之間的整數"

        motion_type = params.get('motion_type', 'EUCLIDEAN')
        valid_motion_types = ['TRANSLATION', 'EUCLIDEAN', 'AFFINE', 'HOMOGRAPHY']
        if motion_type not in valid_motion_types:
            return False, f"motion_type 必須是 {valid_motion_types} 之一"

        return True, ""

    except Exception as e:
        return False, f"參數驗證錯誤: {str(e)}"


# ===== 相容性函式 =====

def run_alignment_with_paths(ref_image_path, input_image_path, output_dir,
                           pyramid_levels=4, motion_type='EUCLIDEAN'):
    """
    使用指定路徑執行圖像對齊（保持原有介面相容性）
    """
    print("=== ECC圖像對齊工具 ===")
    print(f"參考圖像: {ref_image_path}")
    print(f"輸入圖像: {input_image_path}")
    print(f"輸出目錄: {output_dir}")
    print(f"運動模型: {motion_type}")
    print(f"金字塔層數: {pyramid_levels}")
    print("-" * 50)

    # 創建對齊器
    aligner = ECCImageAligner(pyramid_levels=pyramid_levels, motion_type=motion_type)

    # 執行對齊
    success, results = aligner.align_images(ref_image_path, input_image_path, output_dir)

    if success:
        analysis = results['analysis']
        print("\n=== 對齊完成 ===")
        print(f"總位移: {analysis.get('total_displacement', 0):.2f} 像素")
        if 'rotation_degrees' in analysis:
            print(f"旋轉角度: {analysis['rotation_degrees']:.4f}°")
        print(f"相關係數: {results.get('correlation', 0):.6f}")

        print("\n=== 下一步建議 ===")
        print("1. 檢查 comparison.jpg 查看對齊效果")
        print("2. 查看 alignment_info.txt 了解詳細分析")
        print("3. 使用 aligned_image.jpg 進行後續SAM2分割")
        print("4. 變換矩陣已保存為 warp_matrix.npy，可用於批次處理")

        return 0
    else:
        print("對齊失敗:", results.get('error', '未知錯誤'))
        return 1


# 主要執行區域 - 保持原有功能
if __name__ == "__main__":
    # 您的圖像路徑
    ref_path = r"C:\Users\my544\Desktop\Project\image\frame_00007.jpg"
    input_path = r"C:\Users\my544\Desktop\Project\image\frame_00008.jpg"
    output_dir = r"C:\Users\my544\Desktop\Project\correlation_image\output"

    # 執行圖像對齊
    run_alignment_with_paths(ref_path, input_path, output_dir)
