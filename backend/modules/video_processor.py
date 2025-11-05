"""
影片處理模組
功能：從影片提取影格，進行時間序列變化檢測
"""
import cv2
import numpy as np
from pathlib import Path
import tempfile
import shutil
import os
from typing import List, Tuple, Dict, Optional

class VideoProcessor:
    def __init__(self, output_dir: str):
        """
        初始化影片處理器
        Args:
            output_dir: 輸出目錄路徑
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 創建子目錄
        self.frames_dir = self.output_dir / 'frames'
        self.frames_dir.mkdir(exist_ok=True)

    def extract_frames(self, video_path: str, interval_seconds: float = 1.0,
                      max_frames: int = 100) -> Dict:
        """
        從影片提取影格
        Args:
            video_path: 影片檔案路徑
            interval_seconds: 提取間隔（秒）
            max_frames: 最大提取影格數
        Returns:
            包含提取結果的字典
        """
        try:
            # 打開影片
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {'success': False, 'error': '無法打開影片檔案'}

            # 獲取影片資訊
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps

            print(f"影片資訊: FPS={fps}, 總影格數={frame_count}, 長度={duration:.2f}秒")

            # 計算提取間隔（以影格為單位）
            frame_interval = int(fps * interval_seconds)

            extracted_frames = []
            frame_number = 0
            extracted_count = 0

            # 清空之前的影格
            if self.frames_dir.exists():
                shutil.rmtree(self.frames_dir)
            self.frames_dir.mkdir(exist_ok=True)

            while extracted_count < max_frames:
                # 設定到指定影格位置
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

                ret, frame = cap.read()
                if not ret:
                    break

                # 儲存影格
                timestamp = frame_number / fps
                filename = f"frame_{extracted_count:05d}_t{timestamp:.2f}s.jpg"
                frame_path = self.frames_dir / filename

                cv2.imwrite(str(frame_path), frame)

                extracted_frames.append({
                    'filename': filename,
                    'path': str(frame_path),
                    'frame_number': frame_number,
                    'timestamp': timestamp
                })

                print(f"提取影格 {extracted_count + 1}: {filename} (時間: {timestamp:.2f}s)")

                extracted_count += 1
                frame_number += frame_interval

            cap.release()

            return {
                'success': True,
                'frames': extracted_frames,
                'video_info': {
                    'fps': fps,
                    'duration': duration,
                    'total_frames': frame_count,
                    'extracted_count': extracted_count
                },
                'output_dir': str(self.frames_dir)
            }

        except Exception as e:
            return {'success': False, 'error': f'提取影格時發生錯誤: {str(e)}'}

    def detect_temporal_changes(self, frames_data: List[Dict],
                               sensitivity: float = 0.1) -> Dict:
        """
        檢測時間序列變化
        Args:
            frames_data: 影格資料列表
            sensitivity: 敏感度閾值
        Returns:
            變化檢測結果
        """
        try:
            changes = []

            for i in range(1, len(frames_data)):
                prev_frame_path = frames_data[i-1]['path']
                curr_frame_path = frames_data[i]['path']

                # 讀取影格
                prev_img = cv2.imread(prev_frame_path)
                curr_img = cv2.imread(curr_frame_path)

                if prev_img is None or curr_img is None:
                    continue

                # 計算差異
                change_score = self._calculate_frame_difference(prev_img, curr_img)

                if change_score > sensitivity:
                    changes.append({
                        'from_frame': i-1,
                        'to_frame': i,
                        'from_timestamp': frames_data[i-1]['timestamp'],
                        'to_timestamp': frames_data[i]['timestamp'],
                        'change_score': change_score,
                        'from_filename': frames_data[i-1]['filename'],
                        'to_filename': frames_data[i]['filename']
                    })

            return {
                'success': True,
                'changes': changes,
                'total_comparisons': len(frames_data) - 1,
                'significant_changes': len(changes)
            }

        except Exception as e:
            return {'success': False, 'error': f'時間序列變化檢測錯誤: {str(e)}'}

    def _calculate_frame_difference(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        計算兩個影格之間的差異分數
        Args:
            img1, img2: 待比較的影格
        Returns:
            差異分數 (0-1之間)
        """
        # 轉換為灰階
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # 計算絕對差異
        diff = cv2.absdiff(gray1, gray2)

        # 計算差異比例
        total_pixels = diff.shape[0] * diff.shape[1]
        changed_pixels = np.count_nonzero(diff > 30)  # 閾值30

        change_ratio = changed_pixels / total_pixels

        return change_ratio

    def get_frame_pairs_for_analysis(self, frames_data: List[Dict],
                                   pair_strategy: str = 'consecutive') -> List[Tuple[str, str]]:
        """
        獲取用於分析的影格對
        Args:
            frames_data: 影格資料
            pair_strategy: 配對策略 ('consecutive', 'first_last', 'interval')
        Returns:
            影格對路徑列表
        """
        pairs = []

        if pair_strategy == 'consecutive':
            # 連續影格配對
            for i in range(len(frames_data) - 1):
                pairs.append((frames_data[i]['path'], frames_data[i+1]['path']))

        elif pair_strategy == 'first_last':
            # 第一個和最後一個影格
            if len(frames_data) >= 2:
                pairs.append((frames_data[0]['path'], frames_data[-1]['path']))

        elif pair_strategy == 'interval':
            # 間隔配對（每n個影格配對一次）
            interval = max(1, len(frames_data) // 10)  # 最多10對
            for i in range(0, len(frames_data) - interval, interval):
                pairs.append((frames_data[i]['path'], frames_data[i + interval]['path']))

        return pairs

    def cleanup(self):
        """清理臨時檔案"""
        try:
            if self.frames_dir.exists():
                shutil.rmtree(self.frames_dir)
            print("已清理影片處理臨時檔案")
        except Exception as e:
            print(f"清理檔案時發生錯誤: {e}")

def extract_video_frames_api(video_file_path: str, output_dir: str,
                           interval_seconds: float = 1.0, max_frames: int = 100) -> Dict:
    """
    API介面：從影片提取影格 - 簡化版本，專注於影格提取
    """
    processor = VideoProcessor(output_dir)
    result = processor.extract_frames(video_file_path, interval_seconds, max_frames)

    # 🎯 簡化：只提取影格，不進行時間序列分析
    # 提取的影格將用於後續的照片分析流程

    return result
