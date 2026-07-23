# Fall Detection Monitoring System

Hệ thống phát hiện té ngã thời gian thực từ camera/video. Project dùng `YOLOv11-Pose` để trích xuất keypoints cơ thể, chuyển mỗi frame thành vector đặc trưng 52 chiều, gom thành chuỗi frame và dùng mô hình `LSTM` để phân loại `fall` / `normal`. Bản demo realtime có theo dõi nhiều người, lưu sự kiện vào SQLite, chụp ảnh bằng chứng, gửi cảnh báo Telegram và xem lại qua dashboard Streamlit.

## Kiến trúc hệ thống

```text
Camera / Video
    |
    v
YOLOv11-Pose
    |
    v
17 keypoints + bbox ratio
    |
    v
Feature vector 52-D
    |
    v
Sequence window
    |
    v
LSTM classifier
    |
    v
Decision engine
    |
    +--> SQLite event database
    +--> Telegram alert
    +--> Streamlit dashboard
    +--> Demo recording
```

## Công nghệ sử dụng

- Python 3.10+
- OpenCV
- NumPy, Pandas
- TensorFlow / Keras
- Ultralytics YOLO
- PyTorch
- scikit-learn
- Streamlit
- Matplotlib
- SQLite

## Cấu trúc chính

```text
.
|-- fall_features.py                         # Trích xuất đặc trưng pose 52 chiều
|-- extract_frame_feature_cache.py           # Trích feature frame-level từ dataset
|-- build_sequence_dataset_from_frame_cache.py
|                                             # Tạo train/val/test sequence dataset
|-- compare_lstm_transformer.py              # So sánh model thử nghiệm
|-- main_telegram_database.py                # Realtime detection + SQLite + Telegram
|-- dashboard_streamlit.py                   # Dashboard giám sát sự kiện
|-- record.py                                # Record video demo có overlay
|-- requirements.txt                         # Dependency cài đặt nhanh
|-- .env.example                             # Mẫu biến môi trường
|-- telegram_config.example.json             # Mẫu cấu hình Telegram
|-- models_lstm/
|   |-- lstm_runtime_config.json             # Runtime config local
|   `-- lstm_runtime_config.example.json     # Runtime config mẫu
|-- models_4_clean/
|   |-- sequence_comparison_30_35_40.csv     # Bảng so sánh sequence length
|   |-- seq30/
|   |-- seq35/
|   `-- seq40/
|-- captures/                                # Snapshot sự kiện local
|-- recordings/                              # Video demo local
`-- fall_events.db                           # SQLite database local
```

## Cài đặt

Tạo môi trường ảo:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Cài thư viện:

```powershell
pip install -r requirements.txt
```

Nếu dùng GPU, nên cài PyTorch/TensorFlow theo đúng phiên bản CUDA của máy trước khi chạy extract/train.

## Cấu hình

Runtime hiện tại ưu tiên file:

```text
models_lstm/lstm_runtime_config.json
```

File mẫu nằm ở:

```text
models_lstm/lstm_runtime_config.example.json
```

Nếu muốn dùng biến môi trường qua file `.env`, copy từ file mẫu:

```powershell
Copy-Item .env.example .env
```

Các biến môi trường quan trọng:

```powershell
$env:LSTM_MODEL_PATH="models_4_clean/seq30/lstm_best_seq30.h5"
$env:SEQUENCE_LEN="30"
$env:FALL_THRESHOLD="0.50"
$env:FALL_LOW_THRESHOLD="0.55"
$env:FALL_HIGH_THRESHOLD="0.60"
$env:TARGET_FPS="10"
```

## Cấu hình Telegram

Không commit token thật lên GitHub. Dùng một trong hai cách sau.

Cách 1: đặt biến môi trường:

```powershell
$env:TELEGRAM_BOT_TOKEN="your_bot_token"
$env:TELEGRAM_CHAT_ID="your_chat_id"
```

Cách 2: copy file mẫu:

```powershell
Copy-Item telegram_config.example.json telegram_config.json
```

Sau đó sửa `telegram_config.json` trên máy local.

## Chạy realtime

Chạy bằng webcam mặc định:

```powershell
python main_telegram_database.py
```

Chạy bằng video:

```powershell
python main_telegram_database.py --camera "recordings/fall_demo_20260704_202814.mp4" --no-telegram
```

Chỉ định model:

```powershell
python main_telegram_database.py --model "models_4_clean/seq30/lstm_best_seq30.h5" --sequence-len 30
```

Một số phím trong cửa sổ OpenCV:

- `Q`: thoát
- `S`: lưu snapshot

## Chạy dashboard

Dashboard đọc dữ liệu từ `fall_events.db`:

```powershell
python scripts/migrate_database.py
streamlit run dashboard_streamlit.py
```

Dashboard hiển thị:

- Tổng số sự kiện phát hiện té ngã
- Sự kiện mới nhất
- Timeline theo thời gian
- Bảng event và ảnh capture
- Quản lý event trong tab `Manage`: đổi status, ghi review note và export CSV
- Báo cáo trong tab `Reports`: event theo ngày, theo status và tổng hợp theo camera
- Trạng thái event: `new`, `alert_sent`, `alert_failed`, `confirmed`, `false_alarm`, `resolved`
- Trạng thái cấu hình model/Telegram
- Kết quả đánh giá model

## Record video demo

Record demo 60 giây, không gửi Telegram:

```powershell
python record.py --duration 60 --no-alerts
```

Record ra file cụ thể:

```powershell
python record.py --output recordings/demo.mp4 --duration 60 --fps 20 --no-alerts
```

## Smoke test

Chạy kiểm tra nhanh cấu hình, metric CSV, feature dimension và SQLite event schema:

```powershell
python tests/smoke_test.py
```

Log runtime được ghi vào:

```text
logs/fall_detection.log
```

## Dataset và training

Project đã hỗ trợ các nguồn dữ liệu:

- URFD
- Lei2Fall
- GMDCSA24
- MCFD

Theo `frame_feature_cache_v3/summary.json`, cache hiện có:

- 744 video
- Target FPS: 10
- Image size: 640
- Các split chính: train, validation, test và demo

Trích feature frame-level:

```powershell
python extract_frame_feature_cache.py --datasets urfd,leifall,gmdcsa24,mcfd --output-dir frame_feature_cache_v3 --skip-existing
```

Tạo sequence dataset:

```powershell
python build_sequence_dataset_from_frame_cache.py `
  --cache-dir frame_feature_cache_v3 `
  --output-dir extracted_merged_seq30 `
  --sequence-len 30 `
  --stride 15 `
  --fall-ratio-threshold 0.3
```

## Kết quả model hiện tại

Bảng dưới lấy từ `models_4_clean/sequence_comparison_30_35_40.csv`.

| Sequence length | Test accuracy | Test precision | Test recall | Test F1 | Best val threshold |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | 87.40% | 65.22% | 75.76% | 70.09% | 0.55 |
| 35 | 87.47% | 66.00% | 71.74% | 68.75% | 0.65 |
| 40 | 85.68% | 61.17% | 70.79% | 65.63% | 0.60 |

Runtime config hiện đang trỏ về model seq30:

```text
models_4_clean/seq30/lstm_best_seq30.h5
```

Với bài toán té ngã, recall thường quan trọng hơn precision vì bỏ sót té ngã nguy hiểm hơn báo nhầm. Khi trình bày CV, nên ghi rõ chiến lược threshold/smoothing được tối ưu theo mục tiêu an toàn.
