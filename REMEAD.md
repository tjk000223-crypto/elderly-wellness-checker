🏥 遠端高齡者視覺健康檢查系統 (Elderly Wellness Checker)
這是一個結合 Python 與 AI 電腦視覺 技術的健康監控工具，專為高齡者居家健康管理設計。系統透過分析臉部細微特徵與中醫舌象狀態，即時提供潛在健康風險的預警。

✨ 核心功能
🔍 臉部狀態分析 (face_analyzer.py)
系統利用 MediaPipe 468 點特徵追蹤，進行多維度分析：

疲勞偵測：計算眼睛縱橫比 (EAR) 監控眨眼頻率與開啟程度。

色澤診斷：自動偵測臉頰與嘴唇顏色，辨識蒼白、發紅或發紺等異常循環徵兆。

情緒張力：分析眉毛與口周肌肉特徵，識別潛在的不適感或壓力。

風險篩查：透過臉部對稱性評分，輔助識別中風前兆等突發性病徵。

👅 舌象中醫輔助分析 (tongue_analyzer.py)
結合傳統中醫診斷邏輯與數位影像處理：


舌質辨色：識別舌頭呈現之淡白、紅絳或青紫顏色 。


苔層分析：量化舌苔覆蓋程度，區分薄苔、中厚苔或厚苔 。


濕潤度監測：分析舌面反光紋理，評估體內水分狀態（乾燥或濕潤） 。

📊 自動化報價系統 (wellness_checker.py)
即時標註：動態在影像上疊加分析數據與健康標籤。

數位摘要：生成包含「警報 (Alerts)」與「警告 (Warnings)」的專業 JSON 報告。

靈活模式：支援 單張靜態照片 分析或 Webcam 即時監控。

技術元件,主要職責
MediaPipe,高精度臉部特徵點定位與追蹤
OpenCV,影像擷取、色彩空間轉換 (RGB/HSV/LAB) 及繪圖標註
NumPy,負責對稱性數學運算、顏色比率分析及矩陣處理

🚀 快速上手
1. 環境需求
Python 3.9 或以上版本

電腦需具備鏡頭 (Webcam)

# 克隆專案
git clone <你的 GitHub 專案網址>
cd <專案資料夾>

# 安裝依賴套件
pip install -r requirements.txt

3. 執行分析
分析單張圖片：
python wellness_checker.py --image test_photo.jpg
系統將產出標註後的影像 test_photo_annotated.jpg。

啟動即時監控 (預設 60 秒)：
python wellness_checker.py --camera --duration 60

├── src/
│   ├── face_analyzer.py      # 臉部特徵與疲勞分析
│   ├── tongue_analyzer.py    # 舌象特徵分析
│   └── wellness_checker.py   # 主程式與匯報系統
├── requirements.txt          # 必要套件清單
└── README.md                 # 專案說明文件

⚠️ 免責聲明 (Disclaimer)
本系統僅作為居家健康輔助參考，其分析結果不能代替專業醫療診斷、建議或治療。若使用者感到任何身體不適，請務必諮詢專業醫療專業人員。
