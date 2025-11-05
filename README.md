# testing-for-site
在這裡面包含了所有關於網頁的版本
然後目前這是最初版
我原先是直接用sam2的github的整個資料夾去做路徑設定
但是現在為了縮減檔案大小 我就將所有的大型檔案全部先移除了

# how to use
首先先補齊所有缺失的檔案
現在有2個檔案需要補齊
1.checkpoint/sam2.1_hiera_large.pt
2.configs/sam2.1_hiera_l.yaml
請按照以下資料結構擺放
project/
├── checkpoints/
│   └── sam2.1_hiera_large.pt      # 模型權重
├── configs/
│   └── sam2.1_hiera_l.yaml        # 模型配置
├── backend/
│   ├── app.py                     # 你的主程式
│   └── modules/
│       └── sam2_segmenter.py      # 你的 SAM2 包裝器
├──frontend/
|   ├── photo_viewer_backup.html
|   ├── photo_viewer.html          # 網頁主要程式
│   ├── js/
│       ├── app.js
│       └── object_viewer_new.js
└── results/                       # 結果儲存位置 (資料夾會自動創建)

以下是相關檔案連結
https://github.com/facebookresearch/sam2
