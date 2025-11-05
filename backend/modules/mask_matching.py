import numpy as np
from scipy.optimize import linear_sum_assignment
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import shutil
import os
import json
from datetime import datetime


class MaskMatcher:
    """整合運行管理的優化版遮罩匹配器"""

    def __init__(self, iou_threshold=0.3, distance_threshold=20, similarity_threshold=0.35):
        """
        初始化匹配器參數
        Args:
            iou_threshold: IoU閾值，提高到0.3確保形狀相似
            distance_threshold: 距離閾值，降低到20像素避免遠距離誤匹配
            similarity_threshold: 綜合相似度閾值，降低到0.35平衡嚴格要求
        """
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.similarity_threshold = similarity_threshold

        print(f"🔧 匹配器參數設定:")
        print(f"  - IoU閾值: {self.iou_threshold}")
        print(f"  - 距離閾值: {self.distance_threshold} 像素")
        print(f"  - 相似度閾值: {self.similarity_threshold}")

    def calculate_iou(self, mask1, mask2):
        """計算兩個遮罩的IoU"""
        intersection = np.logical_and(mask1 > 0.5, mask2 > 0.5)
        union = np.logical_or(mask1 > 0.5, mask2 > 0.5)
        intersection_area = np.sum(intersection)
        union_area = np.sum(union)

        if union_area == 0:
            return 0.0
        return intersection_area / union_area

    def calculate_centroid_distance(self, centroid1, centroid2):
        """計算質心間的歐式距離"""
        return np.sqrt((centroid1[0] - centroid2[0])**2 +
                      (centroid1[1] - centroid2[1])**2)

    def build_similarity_matrix(self, masks_data_1, masks_data_2):
        """建立相似度矩陣"""
        num_masks_1 = len(masks_data_1['masks'])
        num_masks_2 = len(masks_data_2['masks'])

        iou_matrix = np.zeros((num_masks_1, num_masks_2))
        distance_matrix = np.zeros((num_masks_1, num_masks_2))
        similarity_matrix = np.zeros((num_masks_1, num_masks_2))

        print(f"🔄 計算 {num_masks_1} x {num_masks_2} 的相似度矩陣...")

        for i in range(num_masks_1):
            for j in range(num_masks_2):
                # 計算IoU
                iou = self.calculate_iou(
                    masks_data_1['masks'][i],
                    masks_data_2['masks'][j]
                )
                iou_matrix[i, j] = iou

                # 計算距離
                distance = self.calculate_centroid_distance(
                    masks_data_1['centroids'][i],
                    masks_data_2['centroids'][j]
                )
                distance_matrix[i, j] = distance

                # 計算綜合相似度（更嚴格的計算方式）
                if distance < self.distance_threshold and iou > 0:
                    # 距離因子：距離越近分數越高
                    distance_factor = 1 - (distance / self.distance_threshold)
                    # 綜合分數：IoU和距離因子各佔一定權重
                    similarity_score = 0.6 * iou + 0.4 * distance_factor
                else:
                    # 距離過遠或IoU為0時，相似度設為0
                    similarity_score = 0.0

                similarity_matrix[i, j] = similarity_score

        return {
            'iou_matrix': iou_matrix,
            'distance_matrix': distance_matrix,
            'similarity_matrix': similarity_matrix
        }

    def find_matches(self, masks_data_1, masks_data_2):
        """使用優化的匹配演算法"""
        matrices = self.build_similarity_matrix(masks_data_1, masks_data_2)
        similarity_matrix = matrices['similarity_matrix']

        # 使用匈牙利演算法
        cost_matrix = -similarity_matrix
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_1 = list(range(len(masks_data_1['masks'])))
        unmatched_2 = list(range(len(masks_data_2['masks'])))

        for i, j in zip(row_indices, col_indices):
            similarity_score = similarity_matrix[i, j]
            iou_score = matrices['iou_matrix'][i, j]
            distance = matrices['distance_matrix'][i, j]

            # 更嚴格的匹配條件
            if (similarity_score >= self.similarity_threshold and
                iou_score >= self.iou_threshold and
                distance <= self.distance_threshold):

                matches.append({
                    'mask1_idx': i,
                    'mask2_idx': j,
                    'similarity_score': float(similarity_score),
                    'iou_score': float(iou_score),
                    'centroid_distance': float(distance),
                    'centroid_1': [float(masks_data_1['centroids'][i][0]), float(masks_data_1['centroids'][i][1])],
                    'centroid_2': [float(masks_data_2['centroids'][j][0]), float(masks_data_2['centroids'][j][1])],
                    'area_1': int(masks_data_1['areas'][i]),
                    'area_2': int(masks_data_2['areas'][j])
                })
                unmatched_1.remove(i)
                unmatched_2.remove(j)

        print(f"✅ 匹配結果: {len(matches)} 對成功匹配")
        print(f"📊 未匹配: 圖像1有 {len(unmatched_1)} 個, 圖像2有 {len(unmatched_2)} 個")

        return {
            'matches': matches,
            'unmatched_1': unmatched_1,
            'unmatched_2': unmatched_2,
            'matrices': matrices
        }

    def analyze_changes(self, match_results, masks_data_1, masks_data_2):
        """分析物件變化"""
        analysis = {
            'matched_objects': len(match_results['matches']),
            'disappeared_objects': len(match_results['unmatched_1']),
            'new_objects': len(match_results['unmatched_2']),
            'total_objects_1': len(masks_data_1['masks']),
            'total_objects_2': len(masks_data_2['masks']),
            'movement_analysis': [],
            'size_changes': []
        }

        for match in match_results['matches']:
            movement_distance = match['centroid_distance']
            analysis['movement_analysis'].append({
                'mask_pair': (match['mask1_idx'], match['mask2_idx']),
                'movement_distance': float(movement_distance),
                'movement_vector': [
                    float(match['centroid_2'][0] - match['centroid_1'][0]),
                    float(match['centroid_2'][1] - match['centroid_1'][1])
                ]
            })

            area_change = match['area_2'] - match['area_1']
            area_change_percent = (area_change / match['area_1']) * 100 if match['area_1'] > 0 else 0
            analysis['size_changes'].append({
                'mask_pair': (match['mask1_idx'], match['mask2_idx']),
                'area_change': int(area_change),
                'area_change_percent': float(area_change_percent)
            })

        return analysis

    def clean_output_directories(self, save_base_path):
        """
        🔧 整合運行管理：清空輸出目錄中的所有檔案
        Args:
            save_base_path: 基礎儲存路徑（現在會是 runs/run_XXX/matching/）
        """
        save_base = Path(save_base_path)
        # 定義需要清空的子目錄
        subdirs = ["Disappear", "NewAdded", "Same"]

        print(f"\n🧹 開始清理運行目錄: {save_base_path}")
        total_deleted = 0

        for subdir in subdirs:
            dir_path = save_base / subdir
            if dir_path.exists():
                # 計算該目錄中的檔案數量
                files = list(dir_path.glob("*"))
                file_count = len([f for f in files if f.is_file()])

                if file_count > 0:
                    print(f"  🗑️ 清理 {subdir} 目錄: 刪除 {file_count} 個檔案")
                    # 刪除目錄中的所有檔案
                    for file_path in files:
                        if file_path.is_file():
                            try:
                                file_path.unlink()
                                total_deleted += 1
                            except Exception as e:
                                print(f"  ⚠️ 無法刪除檔案 {file_path}: {e}")
                else:
                    print(f"  ✅ {subdir} 目錄已經是空的")
            else:
                print(f"  📁 {subdir} 目錄不存在，將會自動創建")
                dir_path.mkdir(parents=True, exist_ok=True)

        print(f"✅ 清理完成: 總共刪除 {total_deleted} 個檔案")
        return total_deleted

    def visualize_four_panel_results(self, masks_data_1, masks_data_2, match_results,
                                   image1_path, image2_path, output_dir):
        """🔧 整合運行管理：完整的四面板視覺化"""
        # 載入原始圖像
        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)

        if image1 is None or image2 is None:
            print("⚠️ 警告：無法載入原始圖像，跳過視覺化")
            return

        image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        # 創建 2x2 布局
        fig, axes = plt.subplots(2, 2, figsize=(24, 18))

        # 顯示所有圖像
        axes[0, 0].imshow(image1_rgb)
        axes[0, 1].imshow(image2_rgb)
        axes[1, 0].imshow(image1_rgb)
        axes[1, 1].imshow(image2_rgb)

        # 上排：顯示匹配成功的物件
        num_matches = len(match_results['matches'])
        print(f"🎨 顯示 {num_matches} 個匹配物件")

        # 為匹配物件分配顏色
        if num_matches > 0:
            colors = plt.cm.Set3(np.linspace(0, 1, num_matches))

            for i, match in enumerate(match_results['matches']):
                mask1_idx = match['mask1_idx']
                mask2_idx = match['mask2_idx']
                color = colors[i] if i < len(colors) else plt.cm.tab20(i % 20)[:3]

                # 左上：圖像1的匹配物件
                try:
                    mask1 = masks_data_1['masks'][mask1_idx]
                    contours1, _ = cv2.findContours((mask1 > 0.5).astype(np.uint8),
                                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours1:
                        if len(contour) > 2:
                            contour_points = contour.reshape(-1, 2)
                            axes[0, 0].plot(contour_points[:, 0], contour_points[:, 1],
                                          color=color, linewidth=2, alpha=0.8)

                    # 標記質心和編號
                    centroid1 = match['centroid_1']
                    axes[0, 0].plot(centroid1[0], centroid1[1], 'o',
                                  color=color, markersize=6, markeredgecolor='white', markeredgewidth=1)
                    if i < 30:  # 避免標籤過多
                        axes[0, 0].text(centroid1[0]+8, centroid1[1]+8, str(i),
                                      fontsize=8, color='white', weight='bold',
                                      bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8))
                except:
                    continue

                # 右上：圖像2的匹配物件
                try:
                    mask2 = masks_data_2['masks'][mask2_idx]
                    contours2, _ = cv2.findContours((mask2 > 0.5).astype(np.uint8),
                                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours2:
                        if len(contour) > 2:
                            contour_points = contour.reshape(-1, 2)
                            axes[0, 1].plot(contour_points[:, 0], contour_points[:, 1],
                                          color=color, linewidth=2, alpha=0.8)

                    centroid2 = match['centroid_2']
                    axes[0, 1].plot(centroid2[0], centroid2[1], 'o',
                                  color=color, markersize=6, markeredgecolor='white', markeredgewidth=1)
                    if i < 30:
                        axes[0, 1].text(centroid2[0]+8, centroid2[1]+8, str(i),
                                      fontsize=8, color='white', weight='bold',
                                      bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8))
                except:
                    continue

        # 下排：顯示真正的差異物件
        print(f"🔍 顯示差異物件:")
        print(f"  - 圖像1獨有: {len(match_results['unmatched_1'])} 個")
        print(f"  - 圖像2獨有: {len(match_results['unmatched_2'])} 個")

        # 左下：圖像1獨有的物件（消失的物件）
        disappeared_count = 0
        for mask_idx in match_results['unmatched_1']:
            try:
                mask = masks_data_1['masks'][mask_idx]
                contours, _ = cv2.findContours((mask > 0.5).astype(np.uint8),
                                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if len(contour) > 2:
                        contour_points = contour.reshape(-1, 2)
                        # 使用紅色表示消失的物件
                        axes[1, 0].plot(contour_points[:, 0], contour_points[:, 1],
                                      'r-', linewidth=2, alpha=0.9)
                disappeared_count += 1
            except:
                continue

        # 右下：圖像2獨有的物件（新增的物件）
        new_count = 0
        for mask_idx in match_results['unmatched_2']:
            try:
                mask = masks_data_2['masks'][mask_idx]
                contours, _ = cv2.findContours((mask > 0.5).astype(np.uint8),
                                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if len(contour) > 2:
                        contour_points = contour.reshape(-1, 2)
                        # 使用綠色表示新增的物件
                        axes[1, 1].plot(contour_points[:, 0], contour_points[:, 1],
                                      'g-', linewidth=2, alpha=0.9)
                new_count += 1
            except:
                continue

        # 設定標題
        axes[0, 0].set_title(f'圖像1 - 配對成功物件 ({num_matches} 個)',
                           fontsize=14, weight='bold', pad=10)
        axes[0, 1].set_title(f'圖像2 - 配對成功物件 ({num_matches} 個)',
                           fontsize=14, weight='bold', pad=10)
        axes[1, 0].set_title(f'圖像1獨有物件 ({disappeared_count} 個)\n(圖像1有但圖像2沒有)',
                           fontsize=14, weight='bold', pad=10)
        axes[1, 1].set_title(f'圖像2獨有物件 ({new_count} 個)\n(圖像2有但圖像1沒有)',
                           fontsize=14, weight='bold', pad=10)

        # 移除坐標軸
        for ax in axes.flat:
            ax.axis('off')

        plt.tight_layout()

        # 🔧 整合運行管理：儲存到指定的運行目錄
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        visualization_file = output_path / "optimized_mask_matching_results.jpg"
        plt.savefig(str(visualization_file), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 優化版四面板視覺化結果已保存至: {visualization_file}")

        # 詳細統計資訊
        if num_matches > 0:
            movement_distances = [match['centroid_distance'] for match in match_results['matches']]
            quality_scores = [match['similarity_score'] for match in match_results['matches']]

            print(f"\n📊 優化版匹配統計:")
            print(f"  - 配對成功物件: {num_matches} 對")
            print(f"  - 圖像1獨有物件: {disappeared_count} 個")
            print(f"  - 圖像2獨有物件: {new_count} 個")
            print(f"  - 平均移動距離: {np.mean(movement_distances):.1f} 像素")
            print(f"  - 最大移動距離: {max(movement_distances):.1f} 像素")
            print(f"  - 平均匹配品質: {np.mean(quality_scores):.3f}")

        return str(visualization_file)

    def save_mask_images_by_category(self, masks_data_1, masks_data_2, match_results, save_base_path):
        """
        🔧 整合運行管理：根據匹配結果，將遮罩分為消失、新增和相同三類，分別保存為影像檔
        Args:
            masks_data_1: 圖像1的遮罩資料
            masks_data_2: 圖像2的遮罩資料
            match_results: 匹配結果字典，含匹配及未匹配列表
            save_base_path: 運行目錄下的matching子目錄路徑
        """
        save_base = Path(save_base_path)
        disappeared_path = save_base / "Disappear"
        new_path = save_base / "NewAdded"
        same_path = save_base / "Same"

        # 建立三個資料夾
        disappeared_path.mkdir(parents=True, exist_ok=True)
        new_path.mkdir(parents=True, exist_ok=True)
        same_path.mkdir(parents=True, exist_ok=True)

        print(f"\n📁 建立分類資料夾於運行目錄: {save_base_path}")

        # 1. 儲存消失的遮罩（圖像1獨有的物件）
        disappeared_count = 0
        for idx in match_results['unmatched_1']:
            try:
                mask = masks_data_1['masks'][idx]
                mask_img = (mask > 0.5).astype(np.uint8) * 255
                file_path = disappeared_path / f"disappeared_mask_{idx:03d}.png"
                cv2.imwrite(str(file_path), mask_img)
                disappeared_count += 1
            except Exception as e:
                print(f"⚠️ 儲存消失遮罩 {idx} 時發生錯誤: {e}")

        # 2. 儲存新增的遮罩（圖像2獨有的物件）
        new_count = 0
        for idx in match_results['unmatched_2']:
            try:
                mask = masks_data_2['masks'][idx]
                mask_img = (mask > 0.5).astype(np.uint8) * 255
                file_path = new_path / f"new_mask_{idx:03d}.png"
                cv2.imwrite(str(file_path), mask_img)
                new_count += 1
            except Exception as e:
                print(f"⚠️ 儲存新增遮罩 {idx} 時發生錯誤: {e}")

        # 3. 儲存相同的遮罩（匹配成功的物件對）
        same_count = 0
        for i, match in enumerate(match_results['matches']):
            try:
                mask1_idx = match['mask1_idx']
                mask2_idx = match['mask2_idx']

                # 儲存圖像1的遮罩
                mask1 = masks_data_1['masks'][mask1_idx]
                mask1_img = (mask1 > 0.5).astype(np.uint8) * 255
                file_path1 = same_path / f"same_pair_{i:03d}_image1_mask_{mask1_idx:03d}.png"
                cv2.imwrite(str(file_path1), mask1_img)

                # 儲存圖像2的遮罩
                mask2 = masks_data_2['masks'][mask2_idx]
                mask2_img = (mask2 > 0.5).astype(np.uint8) * 255
                file_path2 = same_path / f"same_pair_{i:03d}_image2_mask_{mask2_idx:03d}.png"
                cv2.imwrite(str(file_path2), mask2_img)

                same_count += 1
            except Exception as e:
                print(f"⚠️ 儲存匹配遮罩對 {i} 時發生錯誤: {e}")

        # 輸出統計資訊
        print(f"\n📊 遮罩分類儲存完成:")
        print(f"  - 儲存路徑: {save_base_path}")
        print(f"  - 消失物件: {disappeared_count} 個 -> {disappeared_path}")
        print(f"  - 新增物件: {new_count} 個 -> {new_path}")
        print(f"  - 相同物件: {same_count} 對 ({same_count * 2} 個檔案) -> {same_path}")
        print(f"  - 總計儲存: {disappeared_count + new_count + same_count * 2} 個遮罩檔案")

        return {
            'disappeared_count': disappeared_count,
            'new_count': new_count,
            'same_count': same_count,
            'total_files': disappeared_count + new_count + same_count * 2,
            'disappeared_path': str(disappeared_path),
            'new_added_path': str(new_path),
            'same_path': str(same_path)
        }


# ===== 🆕 新增：2次分割結果載入模組 =====

def load_masks_from_directory(masks_dir):
    """
    🆕 從指定目錄載入所有遮罩檔案並計算元資料

    Args:
        masks_dir: 包含個別遮罩檔案的目錄路徑

    Returns:
        dict: 包含 masks, centroids, areas, bboxes 的字典
    """
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
    """
    🆕 從 pickle 資料中提取遮罩資訊（支援新舊格式）
    """
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

def load_masks_from_individual_files(masks_dir_1, masks_dir_2):
    """
    🆕 從 all_masks 目錄中載入個別遮罩檔案

    Args:
        masks_dir_1: 圖像1的 all_masks 目錄路徑
        masks_dir_2: 圖像2的 all_masks 目錄路徑

    Returns:
        tuple: (masks_data_1, masks_data_2) 或 (None, None) 如果載入失敗
    """
    try:
        print(f"🔄 從個別遮罩檔案載入資料...")
        print(f"  - 目錄1: {Path(masks_dir_1).name}")
        print(f"  - 目錄2: {Path(masks_dir_2).name}")

        masks_data_1 = load_masks_from_directory(masks_dir_1)
        masks_data_2 = load_masks_from_directory(masks_dir_2)

        if masks_data_1 is None or masks_data_2 is None:
            return None, None

        print(f"✅ 個別遮罩檔案載入成功:")
        print(f"  - 目錄1 遮罩數量: {len(masks_data_1.get('masks', []))}")
        print(f"  - 目錄2 遮罩數量: {len(masks_data_2.get('masks', []))}")

        return masks_data_1, masks_data_2

    except Exception as e:
        print(f"❌ 載入個別遮罩檔案時發生錯誤: {e}")
        return None, None


# ===== 🔧 修改後的 API 封裝函式 =====

def match_masks_api(masks_data_1, masks_data_2, output_dir=None,
                   iou_threshold=0.3, distance_threshold=20, similarity_threshold=0.35):
    """
    🔧 整合運行管理：API 封裝函式執行遮罩匹配並回傳 JSON 格式結果

    Args:
        masks_data_1: 第一張圖像的遮罩資料
        masks_data_2: 第二張圖像的遮罩資料
        output_dir: 運行目錄下的 matching 子目錄（可選）
        iou_threshold: IoU 閾值
        distance_threshold: 距離閾值
        similarity_threshold: 相似度閾值

    Returns:
        dict: JSON 格式的回應字典
    """
    try:
        print(f"\n🚀 開始執行遮罩匹配 API...")
        print(f"📊 輸入資料: 圖像1有 {len(masks_data_1.get('masks', []))} 個遮罩, 圖像2有 {len(masks_data_2.get('masks', []))} 個遮罩")

        # 初始化匹配器
        matcher = MaskMatcher(
            iou_threshold=iou_threshold,
            distance_threshold=distance_threshold,
            similarity_threshold=similarity_threshold
        )

        # 執行匹配
        match_results = matcher.find_matches(masks_data_1, masks_data_2)

        # 分析變化
        analysis = matcher.analyze_changes(match_results, masks_data_1, masks_data_2)

        # 準備回應資料
        response_data = {
            'matches': match_results['matches'],
            'unmatched_masks_1': match_results['unmatched_1'],
            'unmatched_masks_2': match_results['unmatched_2'],
            'analysis': analysis,
            'statistics': {
                'matched_objects': analysis['matched_objects'],
                'disappeared_objects': analysis['disappeared_objects'],
                'new_objects': analysis['new_objects'],
                'total_objects_1': analysis['total_objects_1'],
                'total_objects_2': analysis['total_objects_2'],
                'match_rate': float(analysis['matched_objects'] / max(analysis['total_objects_1'], analysis['total_objects_2']) * 100) if max(analysis['total_objects_1'], analysis['total_objects_2']) > 0 else 0.0
            }
        }

        # 🔧 整合運行管理：如果有輸出目錄，儲存分類檔案到運行目錄
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # 清理輸出目錄
            matcher.clean_output_directories(output_path)

            # 儲存分類遮罩檔案
            save_stats = matcher.save_mask_images_by_category(
                masks_data_1, masks_data_2, match_results, output_path
            )

            response_data.update({
                'output_directory': str(output_path),
                'saved_files': save_stats,
                'disappear_folder': save_stats['disappeared_path'],
                'newadded_folder': save_stats['new_added_path'],
                'same_folder': save_stats['same_path']
            })

        return {
            'status': 'success',
            'message': '遮罩匹配完成',
            'data': response_data
        }

    except Exception as e:
        import traceback
        print(f"💥 遮罩匹配過程發生錯誤: {e}")
        traceback.print_exc()
        return {
            'status': 'error',
            'message': '遮罩匹配過程發生錯誤',
            'error': str(e)
        }

def match_masks_with_images_api(masks_data_1, masks_data_2, image1_path, image2_path,
                               output_dir=None, iou_threshold=0.3, distance_threshold=20,
                               similarity_threshold=0.35):
    """
    🔧 整合運行管理：執行遮罩匹配並生成視覺化結果到運行目錄

    Args:
        masks_data_1: 第一張圖像的遮罩資料
        masks_data_2: 第二張圖像的遮罩資料
        image1_path: 第一張圖像路徑
        image2_path: 第二張圖像路徑
        output_dir: 運行目錄下的 matching 子目錄（可選）
        iou_threshold: IoU 閾值
        distance_threshold: 距離閾值
        similarity_threshold: 相似度閾值

    Returns:
        dict: JSON 格式的回應字典
    """
    try:
        print(f"\n🎨 開始執行遮罩匹配與視覺化 API...")

        # 驗證圖像檔案是否存在
        if not os.path.exists(image1_path):
            return {
                'status': 'error',
                'message': '找不到第一張圖像檔案',
                'error': f'File not found: {image1_path}'
            }

        if not os.path.exists(image2_path):
            return {
                'status': 'error',
                'message': '找不到第二張圖像檔案',
                'error': f'File not found: {image2_path}'
            }

        # 初始化匹配器
        matcher = MaskMatcher(
            iou_threshold=iou_threshold,
            distance_threshold=distance_threshold,
            similarity_threshold=similarity_threshold
        )

        # 執行匹配
        match_results = matcher.find_matches(masks_data_1, masks_data_2)

        # 分析變化
        analysis = matcher.analyze_changes(match_results, masks_data_1, masks_data_2)

        # 準備回應資料
        response_data = {
            'matches': match_results['matches'],
            'unmatched_masks_1': match_results['unmatched_1'],
            'unmatched_masks_2': match_results['unmatched_2'],
            'analysis': analysis,
            'statistics': {
                'matched_objects': analysis['matched_objects'],
                'disappeared_objects': analysis['disappeared_objects'],
                'new_objects': analysis['new_objects'],
                'total_objects_1': analysis['total_objects_1'],
                'total_objects_2': analysis['total_objects_2'],
                'match_rate': float(analysis['matched_objects'] / max(analysis['total_objects_1'], analysis['total_objects_2']) * 100) if max(analysis['total_objects_1'], analysis['total_objects_2']) > 0 else 0.0
            }
        }

        # 🔧 整合運行管理：如果有輸出目錄，生成視覺化和儲存檔案到運行目錄
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # 清理輸出目錄
            matcher.clean_output_directories(output_path)

            # 生成視覺化結果
            visualization_path = matcher.visualize_four_panel_results(
                masks_data_1, masks_data_2, match_results,
                image1_path, image2_path, output_path
            )

            # 儲存分類遮罩檔案
            save_stats = matcher.save_mask_images_by_category(
                masks_data_1, masks_data_2, match_results, output_path
            )

            response_data.update({
                'output_directory': str(output_path),
                'visualization_path': visualization_path,
                'saved_files': save_stats,
                'disappear_folder': save_stats['disappeared_path'],
                'newadded_folder': save_stats['new_added_path'],
                'same_folder': save_stats['same_path']
            })

        return {
            'status': 'success',
            'message': '遮罩匹配和視覺化完成',
            'data': response_data
        }

    except Exception as e:
        import traceback
        print(f"💥 遮罩匹配過程發生錯誤: {e}")
        traceback.print_exc()
        return {
            'status': 'error',
            'message': '遮罩匹配過程發生錯誤',
            'error': str(e)
        }

def load_masks_from_pickle(masks_1_path, masks_2_path):
    """
    🔧 修改版：同時支援 pickle 檔案和個別遮罩檔案載入
    """
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
            masks_data_1 = load_masks_from_directory(path1)
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
            masks_data_2 = load_masks_from_directory(path2)
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

def validate_mask_matching_parameters(params):
    """
    驗證遮罩匹配參數

    Args:
        params: 參數字典

    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # 驗證數值參數
        iou_threshold = params.get('iou_threshold', 0.3)
        if not isinstance(iou_threshold, (int, float)) or not (0.0 <= iou_threshold <= 1.0):
            return False, "iou_threshold 必須是 0.0-1.0 之間的數值"

        distance_threshold = params.get('distance_threshold', 20)
        if not isinstance(distance_threshold, (int, float)) or distance_threshold < 0:
            return False, "distance_threshold 必須是非負數"

        similarity_threshold = params.get('similarity_threshold', 0.35)
        if not isinstance(similarity_threshold, (int, float)) or not (0.0 <= similarity_threshold <= 1.0):
            return False, "similarity_threshold 必須是 0.0-1.0 之間的數值"

        return True, ""

    except Exception as e:
        return False, f"參數驗證錯誤: {str(e)}"

def create_mask_matching_response(match_results, analysis, save_stats=None, output_dir=None):
    """
    🔧 整合運行管理：輔助函式創建標準化的遮罩匹配回應格式
    """
    response_data = {
        'matches': match_results['matches'],
        'unmatched_masks_1': match_results['unmatched_1'],
        'unmatched_masks_2': match_results['unmatched_2'],
        'analysis': analysis,
        'statistics': {
            'matched_objects': analysis['matched_objects'],
            'disappeared_objects': analysis['disappeared_objects'],
            'new_objects': analysis['new_objects'],
            'total_objects_1': analysis['total_objects_1'],
            'total_objects_2': analysis['total_objects_2'],
            'match_rate': float(analysis['matched_objects'] / max(analysis['total_objects_1'], analysis['total_objects_2']) * 100) if max(analysis['total_objects_1'], analysis['total_objects_2']) > 0 else 0.0
        }
    }

    if output_dir and save_stats:
        response_data.update({
            'output_directory': str(output_dir),
            'saved_files': save_stats,
            'disappear_folder': save_stats.get('disappeared_path', str(Path(output_dir) / "Disappear")),
            'newadded_folder': save_stats.get('new_added_path', str(Path(output_dir) / "NewAdded")),
            'same_folder': save_stats.get('same_path', str(Path(output_dir) / "Same"))
        })

    return {
        'status': 'success',
        'message': '遮罩匹配完成',
        'data': response_data
    }

# ===== 🔧 修改後的相容性函式 =====

def run_mask_matching_with_run_management(run_directory=None, run_number=None):
    """
    🔧 整合運行管理：執行優化版遮罩對應分析，支援運行目錄結構和新的2次分割格式

    Args:
        run_directory: 指定的運行目錄路徑
        run_number: 運行編號
    """
    print("🚀 優化版遮罩匹配分析系統 (整合運行管理 + 2次分割支援)")

    # 🔧 設定運行目錄結構
    if run_directory:
        website_root = Path(run_directory).parent.parent.parent  # 從 runs/run_XXX 推導出 website0731
    else:
        website_root = Path(__file__).parent.parent.parent  # website0731/

    results_root = website_root / "results"

    # 🔧 如果指定了運行目錄，使用該目錄；否則查找最新的運行
    if run_directory and Path(run_directory).exists():
        current_run_dir = Path(run_directory)
        print(f"📁 使用指定的運行目錄: {current_run_dir}")
    else:
        # 查找最新的運行目錄
        runs_dir = results_root / "runs"
        if runs_dir.exists():
            run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith('run_')],
                            key=lambda x: x.stat().st_ctime, reverse=True)
            if run_dirs:
                current_run_dir = run_dirs[0]
                print(f"📁 使用最新的運行目錄: {current_run_dir}")
            else:
                print("❌ 沒有找到任何運行目錄")
                return None, None
        else:
            print("❌ runs 目錄不存在")
            return None, None

    # 🔧 設定路徑到當前運行目錄結構
    segmentation_dir = current_run_dir / "segmentation"
    matching_dir = current_run_dir / "matching"
    upload_dir = current_run_dir / "upload"

    # 🔧 修正：查找分割結果中的遮罩檔案（支援新的 all_masks 目錄）
    masks_1_path = None
    masks_2_path = None

    if segmentation_dir.exists():
        print(f"📁 分割目錄存在: {segmentation_dir}")

        # 列出所有子目錄
        all_subdirs = [d for d in segmentation_dir.iterdir() if d.is_dir()]
        print(f"📂 找到子目錄: {[d.name for d in all_subdirs]}")

        # 檢查 image 目錄
        image_dirs = [d for d in all_subdirs if d.name.startswith('image')]
        print(f"🖼️ 圖像目錄: {[d.name for d in image_dirs]}")

        if len(image_dirs) >= 2:
            image_dirs.sort()  # 確保順序一致

            # 🔧 修正：實際檢查並設定路徑
            all_masks_1 = image_dirs[0] / "all_masks"
            all_masks_2 = image_dirs[1] / "all_masks"

            print(f"🔍 檢查 all_masks 目錄:")
            print(f"  - 目錄1: {all_masks_1} (存在: {all_masks_1.exists()})")
            print(f"  - 目錄2: {all_masks_2} (存在: {all_masks_2.exists()})")

            if all_masks_1.exists() and all_masks_2.exists():
                # ✅ 關鍵修正：實際賦值路徑
                masks_1_path = all_masks_1
                masks_2_path = all_masks_2
                print(f"✅ 找到 all_masks 目錄:")
                print(f"  - 目錄1: {all_masks_1}")
                print(f"  - 目錄2: {all_masks_2}")

                # 驗證遮罩檔案
                mask_files_1 = list(all_masks_1.glob("mask_*.png"))
                mask_files_2 = list(all_masks_2.glob("mask_*.png"))
                print(f"  - 目錄1遮罩檔案: {len(mask_files_1)} 個")
                print(f"  - 目錄2遮罩檔案: {len(mask_files_2)} 個")

            else:
                print(f"⚠️ all_masks 目錄不存在，尋找備用 pickle 檔案...")
                # 🔧 備用：查找 pickle 檔案
                pickle_files = []
                for root, dirs, files in os.walk(segmentation_dir):
                    for file in files:
                        if file.endswith('_two_pass_complete.pkl') or file.endswith('_masks_complete.pkl'):
                            pickle_files.append(Path(root) / file)

                if len(pickle_files) >= 2:
                    pickle_files.sort()
                    masks_1_path = pickle_files[0]
                    masks_2_path = pickle_files[1]
                    print(f"✅ 使用 pickle 檔案:")
                    print(f"  - 檔案1: {pickle_files[0]}")
                    print(f"  - 檔案2: {pickle_files[1]}")
                else:
                    print(f"❌ 找不到足夠的遮罩檔案或目錄")
                    return None, None
        else:
            print(f"❌ 圖像目錄數量不足: {len(image_dirs)} < 2")
            return None, None
    else:
        print(f"❌ 分割目錄不存在: {segmentation_dir}")
        return None, None

    # 🔧 重要：檢查路徑是否已設定
    if masks_1_path is None or masks_2_path is None:
        print(f"❌ 無法從分割結果中提取完整的遮罩檔案路徑")
        print(f"  - masks_1_path: {masks_1_path}")
        print(f"  - masks_2_path: {masks_2_path}")
        return None, None

    # 🔧 查找原始圖像
    image1_path = None
    image2_path = None

    if upload_dir.exists():
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(upload_dir.glob(ext))

        if len(image_files) >= 2:
            image_files.sort()
            image1_path = image_files[0]
            image2_path = image_files[1]
        else:
            print(f"⚠️ 在 {upload_dir} 中找不到足夠的圖像檔案")

    print(f"📊 使用檔案:")
    print(f"  - 遮罩1: {masks_1_path}")
    print(f"  - 遮罩2: {masks_2_path}")
    print(f"  - 圖像1: {image1_path}")
    print(f"  - 圖像2: {image2_path}")
    print(f"  - 輸出目錄: {matching_dir}")

    # 確保匹配輸出目錄存在
    matching_dir.mkdir(parents=True, exist_ok=True)

    # 🆕 載入遮罩資料前的驗證
    print(f"\n📁 驗證遮罩路徑...")
    print(f"  - 路徑1: {masks_1_path} (存在: {Path(masks_1_path).exists()})")
    print(f"  - 路徑2: {masks_2_path} (存在: {Path(masks_2_path).exists()})")

    if not Path(masks_1_path).exists() or not Path(masks_2_path).exists():
        print("❌ 遮罩路徑驗證失敗")
        return None, None

    # 🆕 載入遮罩資料（支援新格式）
    print(f"\n📁 載入遮罩資料...")
    masks_data_1, masks_data_2 = load_masks_from_pickle(str(masks_1_path), str(masks_2_path))

    if masks_data_1 is None or masks_data_2 is None:
        print("❌ 載入遮罩資料失敗")
        return None, None

    # 初始化優化版匹配器
    matcher = MaskMatcher(
        iou_threshold=0.3,        # 保持IoU要求
        distance_threshold=20,    # 降低距離閾值（已校正圖片）
        similarity_threshold=0.35 # 降低相似度閾值以平衡嚴格的距離要求
    )

    # 清理輸出目錄
    matcher.clean_output_directories(matching_dir)

    # 執行匹配
    print(f"\n🚀 執行優化版遮罩匹配...")
    match_results = matcher.find_matches(masks_data_1, masks_data_2)

    # 分析變化
    analysis = matcher.analyze_changes(match_results, masks_data_1, masks_data_2)

    # 顯示結果
    print(f"\n📊 優化後遮罩對應結果:")
    print(f"  - 匹配成功: {analysis['matched_objects']} 對")
    print(f"  - 消失物件: {analysis['disappeared_objects']} 個")
    print(f"  - 新增物件: {analysis['new_objects']} 個")
    print(f"  - 匹配率: {analysis['matched_objects']/max(analysis['total_objects_1'], analysis['total_objects_2'])*100:.1f}%")

    # 顯示移動分析（只顯示前10個）
    if analysis['movement_analysis']:
        print(f"\n🔄 物件移動分析（前10個）:")
        for i, movement in enumerate(analysis['movement_analysis'][:10]):
            distance = movement['movement_distance']
            vector = movement['movement_vector']
            print(f"  - 物件 {i+1}: 移動距離 {distance:.1f} 像素, 向量 ({vector[0]:.1f}, {vector[1]:.1f})")

    # 生成視覺化結果（如果有圖像）
    if image1_path and image2_path:
        print(f"\n🎨 生成優化版四面板視覺化結果...")
        visualization_path = matcher.visualize_four_panel_results(
            masks_data_1, masks_data_2, match_results,
            image1_path=str(image1_path),
            image2_path=str(image2_path),
            output_dir=str(matching_dir)
        )

    # 分類儲存遮罩檔案
    print(f"\n💾 開始分類儲存遮罩檔案...")
    save_stats = matcher.save_mask_images_by_category(
        masks_data_1,
        masks_data_2,
        match_results,
        save_base_path=str(matching_dir)
    )

    print(f"\n🎯 === 優化版遮罩匹配分析完成！ ===")
    print(f"📁 結果已統一儲存於運行目錄: {current_run_dir}")
    print(f"  - 📸 視覺化結果: {matching_dir}/optimized_mask_matching_results.jpg")
    print(f"  - 📤 消失遮罩: {save_stats['disappeared_path']} ({save_stats['disappeared_count']} 個)")
    print(f"  - 📥 新增遮罩: {save_stats['new_added_path']} ({save_stats['new_count']} 個)")
    print(f"  - 🔄 相同遮罩: {save_stats['same_path']} ({save_stats['same_count']} 對)")
    print(f"✅ 運行管理已整合: 所有檔案都儲存在系統化的運行目錄中")
    print(f"🆕 支援2次分割格式: 已從 all_masks 目錄載入個別遮罩檔案")

    return match_results, analysis, masks_data_1, masks_data_2


# 主要執行程式
if __name__ == "__main__":
    print("🚀 === 整合運行管理的優化版遮罩匹配分析系統 (2次分割支援) ===")

    # 🔧 支援兩種執行模式
    import sys

    if len(sys.argv) > 1:
        # 指定運行目錄模式
        run_directory = sys.argv[1]
        print(f"📁 指定運行目錄模式: {run_directory}")
        results = run_mask_matching_with_run_management(run_directory=run_directory)
    else:
        # 自動查找最新運行模式
        print(f"🔍 自動查找最新運行模式")
        results = run_mask_matching_with_run_management()

    if results[0] is not None:
        print("🎉 優化版遮罩匹配分析完成！(已整合運行管理 + 2次分割支援)")
        print("📊 所有結果已系統化組織在運行目錄中")
        print("🔗 可直接用於後續的變化檢測步驟")
        print("🆕 完美支援新的2次分割all_masks目錄結構")
    else:
        print("❌ 遮罩匹配分析失敗，請檢查運行目錄和檔案")
        print("💡 提示: 請確保先執行 SAM2 分割生成遮罩檔案")
