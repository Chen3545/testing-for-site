import os
import json
import shutil
from datetime import datetime
from pathlib import Path

class RunManager:
    def __init__(self, base_results_dir="results"):
        self.base_results_dir = Path(base_results_dir)
        self.runs_dir = self.base_results_dir / "runs"
        self.current_run_id = None
        self.current_run_dir = None

        # 確保基礎目錄存在
        self.base_results_dir.mkdir(exist_ok=True)
        self.runs_dir.mkdir(exist_ok=True)

        print(f"🏗️ RunManager 初始化: {self.base_results_dir}")

    def get_next_run_number(self):
        """🔧 修正：自動獲取下一個運行編號"""
        existing_runs = []

        # 掃描現有的 run 資料夾
        if self.runs_dir.exists():
            for item in self.runs_dir.iterdir():
                if item.is_dir() and item.name.startswith('run_'):
                    try:
                        # 提取運行編號
                        run_number = int(item.name.split('_')[1])
                        existing_runs.append(run_number)
                    except (ValueError, IndexError):
                        # 忽略格式不正確的資料夾
                        continue

        # 🔧 關鍵修正：找到下一個可用的編號
        if not existing_runs:
            next_run = 1
        else:
            next_run = max(existing_runs) + 1

        print(f"📊 現有運行: {sorted(existing_runs)}")
        print(f"🆕 下一個運行編號: {next_run:03d}")

        return next_run

    def create_new_run(self, description=""):
        """🆕 創建新的運行資料夾"""
        run_number = self.get_next_run_number()
        self.current_run_id = f"run_{run_number:03d}"
        self.current_run_dir = self.runs_dir / self.current_run_id

        # 🔧 關鍵修正：確保創建全新的資料夾
        if self.current_run_dir.exists():
            print(f"⚠️ 警告：{self.current_run_id} 已存在，嘗試下一個編號...")
            return self.create_new_run(description)  # 遞歸尋找下一個可用編號

        # 創建運行資料夾結構
        self.current_run_dir.mkdir(parents=True)
        (self.current_run_dir / "upload").mkdir()
        (self.current_run_dir / "aligned").mkdir()
        (self.current_run_dir / "segmentation").mkdir()
        (self.current_run_dir / "matching").mkdir()
        (self.current_run_dir / "detection").mkdir()

        # 🆕 創建運行資訊記錄
        run_info = {
            "run_id": self.current_run_id,
            "run_number": run_number,
            "created_at": datetime.now().isoformat(),
            "description": description or f"運行 {run_number}",
            "status": "created",
            "steps_completed": [],
            "total_files": 0,
            "processing_start": None,
            "processing_end": None
        }

        # 儲存運行資訊
        info_file = self.current_run_dir / "run_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(run_info, f, ensure_ascii=False, indent=2)

        print(f"✅ 創建新運行: {self.current_run_id}")
        print(f"📁 運行目錄: {self.current_run_dir}")

        return self.current_run_id, str(self.current_run_dir)

    def get_current_run_info(self):
        """獲取當前運行資訊"""
        if not self.current_run_dir or not self.current_run_id:
            return None

        return {
            "run_id": self.current_run_id,
            "run_dir": str(self.current_run_dir),
            "upload_dir": str(self.current_run_dir / "upload"),
            "aligned_dir": str(self.current_run_dir / "aligned"),
            "segmentation_dir": str(self.current_run_dir / "segmentation"),
            "matching_dir": str(self.current_run_dir / "matching"),
            "detection_dir": str(self.current_run_dir / "detection")
        }

    def update_run_status(self, status, step_name=None):
        """🔧 修正：更新運行狀態"""
        if not self.current_run_dir:
            return False

        info_file = self.current_run_dir / "run_info.json"
        if not info_file.exists():
            return False

        try:
            # 讀取現有資訊
            with open(info_file, 'r', encoding='utf-8') as f:
                run_info = json.load(f)

            # 更新狀態
            run_info["status"] = status
            run_info["last_updated"] = datetime.now().isoformat()

            if step_name and step_name not in run_info["steps_completed"]:
                run_info["steps_completed"].append(step_name)

            if status == "processing" and not run_info.get("processing_start"):
                run_info["processing_start"] = datetime.now().isoformat()

            if status in ["completed", "failed"]:
                run_info["processing_end"] = datetime.now().isoformat()

            # 儲存更新後的資訊
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(run_info, f, ensure_ascii=False, indent=2)

            print(f"📝 更新運行狀態: {self.current_run_id} -> {status}")
            if step_name:
                print(f"✅ 完成步驟: {step_name}")

            return True

        except Exception as e:
            print(f"❌ 更新運行狀態失敗: {e}")
            return False

    def list_all_runs(self):
        """🆕 列出所有運行"""
        runs = []

        if not self.runs_dir.exists():
            return runs

        for run_dir in sorted(self.runs_dir.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith('run_'):
                continue

            info_file = run_dir / "run_info.json"
            if info_file.exists():
                try:
                    with open(info_file, 'r', encoding='utf-8') as f:
                        run_info = json.load(f)
                    runs.append(run_info)
                except:
                    # 如果無法讀取資訊檔案，創建基本資訊
                    runs.append({
                        "run_id": run_dir.name,
                        "run_dir": str(run_dir),
                        "created_at": "未知",
                        "status": "未知"
                    })

        return runs

    def delete_run(self, run_id):
        """🗑️ 刪除指定運行"""
        run_dir = self.runs_dir / run_id

        if not run_dir.exists():
            print(f"❌ 運行不存在: {run_id}")
            return False

        try:
            shutil.rmtree(run_dir)
            print(f"🗑️ 已刪除運行: {run_id}")

            # 如果刪除的是當前運行，清空當前運行資訊
            if self.current_run_id == run_id:
                self.current_run_id = None
                self.current_run_dir = None

            return True

        except Exception as e:
            print(f"❌ 刪除運行失敗: {e}")
            return False

    def cleanup_old_runs(self, keep_recent=10):
        """🧹 清理舊運行（保留最近的N個）"""
        runs = self.list_all_runs()

        if len(runs) <= keep_recent:
            print(f"📊 運行數量 {len(runs)} <= {keep_recent}，無需清理")
            return

        # 按創建時間排序，刪除最舊的
        runs.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        runs_to_delete = runs[keep_recent:]

        deleted_count = 0
        for run_info in runs_to_delete:
            run_id = run_info.get('run_id')
            if run_id and self.delete_run(run_id):
                deleted_count += 1

        print(f"🧹 清理完成：刪除了 {deleted_count} 個舊運行")

    def get_run_summary(self):
        """📊 獲取運行摘要"""
        runs = self.list_all_runs()

        summary = {
            "total_runs": len(runs),
            "current_run": self.current_run_id,
            "recent_runs": []
        }

        # 獲取最近5個運行的狀態
        recent_runs = sorted(runs, key=lambda x: x.get('created_at', ''), reverse=True)[:5]

        for run in recent_runs:
            summary["recent_runs"].append({
                "run_id": run.get('run_id'),
                "status": run.get('status'),
                "created_at": run.get('created_at'),
                "steps_completed": len(run.get('steps_completed', []))
            })

        return summary

    def ensure_run_exists(self):
        """🔧 確保有可用的運行目錄"""
        if not self.current_run_dir or not self.current_run_dir.exists():
            print("📁 當前無可用運行，創建新運行...")
            self.create_new_run()

        return self.get_current_run_info()

# 🆕 新增：全域運行管理器實例
_global_run_manager = None

def get_run_manager(base_results_dir="results"):
    """🔧 獲取全域運行管理器實例"""
    global _global_run_manager

    if _global_run_manager is None:
        _global_run_manager = RunManager(base_results_dir)

    return _global_run_manager

def create_new_run(description=""):
    """🚀 便捷函數：創建新運行"""
    manager = get_run_manager()
    return manager.create_new_run(description)

def get_current_run():
    """📁 便捷函數：獲取當前運行資訊"""
    manager = get_run_manager()
    return manager.get_current_run_info()

def update_run_status(status, step_name=None):
    """📝 便捷函數：更新運行狀態"""
    manager = get_run_manager()
    return manager.update_run_status(status, step_name)


# 🧪 測試功能
if __name__ == "__main__":
    print("🧪 測試 RunManager...")

    # 創建管理器
    manager = RunManager("test_results")

    # 創建幾個測試運行
    for i in range(3):
        run_id, run_dir = manager.create_new_run(f"測試運行 {i+1}")
        print(f"創建運行: {run_id} -> {run_dir}")

        # 模擬步驟完成
        manager.update_run_status("processing", "upload")
        manager.update_run_status("processing", "alignment")
        manager.update_run_status("completed")

    # 顯示摘要
    summary = manager.get_run_summary()
    print(f"\n📊 運行摘要:")
    print(f"總運行數: {summary['total_runs']}")
    print(f"當前運行: {summary['current_run']}")
    print(f"最近運行: {[r['run_id'] for r in summary['recent_runs']]}")

    print("\n✅ 測試完成！")
