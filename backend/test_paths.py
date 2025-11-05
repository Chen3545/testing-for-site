"""
測試 SAM2 路徑配置
驗證所有必要檔案是否存在且路徑正確
"""
from pathlib import Path
import sys

# 設定專案根目錄
WEBSITE_ROOT = Path(__file__).parent.parent
CHECKPOINT_PATH = WEBSITE_ROOT / "checkpoint" / "sam2.1_hiera_large.pt"
CONFIG_PATH = WEBSITE_ROOT / "configs" / "sam2.1_hiera_l.yaml"
RESULTS_ROOT = WEBSITE_ROOT / "results"

print("=" * 60)
print("🔍 SAM2 路徑配置檢查")
print("=" * 60)

# 檢查專案根目錄
print(f"\n🏠 專案根目錄: {WEBSITE_ROOT}")
print(f"   存在: {'✅' if WEBSITE_ROOT.exists() else '❌'}")

# 檢查模型權重
print(f"\n⚖️ 模型權重: {CHECKPOINT_PATH}")
print(f"   存在: {'✅' if CHECKPOINT_PATH.exists() else '❌'}")
if CHECKPOINT_PATH.exists():
    size_mb = CHECKPOINT_PATH.stat().st_size / (1024 * 1024)
    print(f"   大小: {size_mb:.2f} MB")

# 檢查配置檔案
print(f"\n⚙️ 模型配置: {CONFIG_PATH}")
print(f"   存在: {'✅' if CONFIG_PATH.exists() else '❌'}")

# 檢查結果目錄
print(f"\n📁 結果目錄: {RESULTS_ROOT}")
print(f"   存在: {'✅' if RESULTS_ROOT.exists() else '❌'}")

# 檢查 SAM2 套件是否已安裝
print("\n" + "=" * 60)
print("📦 檢查 SAM2 套件安裝")
print("=" * 60)
try:
    import sam2
    print("✅ sam2 套件已安裝")
    print(f"   路徑: {sam2.__file__}")

    # 嘗試導入核心模組
    try:
        from sam2.build_sam import build_sam2
        print("✅ build_sam2 可以導入")
    except ImportError as e:
        print(f"❌ build_sam2 導入失敗: {e}")

    try:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        print("✅ SAM2AutomaticMaskGenerator 可以導入")
    except ImportError as e:
        print(f"❌ SAM2AutomaticMaskGenerator 導入失敗: {e}")

except ImportError:
    print("❌ sam2 套件未安裝")
    print("   請執行: pip install git+https://github.com/facebookresearch/sam2.git")

# 總結
print("\n" + "=" * 60)
print("📊 檢查總結")
print("=" * 60)

all_ok = (
    WEBSITE_ROOT.exists() and
    CHECKPOINT_PATH.exists() and
    CONFIG_PATH.exists() and
    RESULTS_ROOT.exists()
)

if all_ok:
    print("✅ 所有路徑配置正確！")
    print("   可以開始使用 SAM2 模型")
else:
    print("❌ 發現問題，請檢查缺失的檔案或目錄")

print("=" * 60)
