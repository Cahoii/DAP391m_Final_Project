# 🚗 Báo Cáo Dự Án Car Price Prediction

> DAP · SS4 SU26 — FPT University

---

## 📋 Tổng Quan

Dự án **Car Price Prediction** là một hệ thống Machine Learning dự đoán giá xe ô tô dựa trên 9 đặc trưng đầu vào (Brand, Model, Year, Engine_Size, Fuel_Type, Transmission, Mileage, Doors, Owner_Count). Dự án bao gồm pipeline xử lý dữ liệu hoàn chỉnh, huấn luyện 3 mô hình ML và triển khai ứng dụng web bằng Flask.

---

## 🔄 Luồng Chạy & Thứ Tự Khi Thuyết Trình

### Sơ đồ luồng dữ liệu

```
┌─────────────────────────────────────────────────────────────┐
│                      TRAINING PIPELINE                      │
│                                                             │
│  config.py → train.py                                       │
│                │                                            │
│    ┌───────────┼───────────┐                                │
│    ▼           ▼           ▼                                │
│  data_loader  pipeline    model                             │
│  (load CSV)   │           (train 3 models)                  │
│               ├─ cleaning         ├─ LinearRegression       │
│               ├─ feature_eng      ├─ RandomForest           │
│               ├─ eda (grab_cols)  └─ XGBoost                │
│               └─ preprocessing    → evaluation → best.pkl   │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                      PREDICTION PIPELINE                    │
│                                                             │
│  app.py (Flask) → predict.py                                │
│    user input      ├─ feature_engineering                   │
│                    ├─ grab_col_names                        │
│                    ├─ preprocess_data (training=False)      │
│                    └─ model.predict → return price          │
└─────────────────────────────────────────────────────────────┘
```

### Thứ tự chạy file khi thuyết trình

| STT | File | Mục đích | Thời lượng đề xuất |
|-----|------|----------|-------------------|
| **1** | `config.py` | Cấu hình đường dẫn, hằng số | 1 phút |
| **2** | `data/car_price_dataset.csv` | Giới thiệu dataset (10 cột, 10,000 dòng) | 2 phút |
| **3** | `notebooks/EDA.ipynb` | Trình bày kết quả phân tích khám phá | 4 phút |
| **4** | `utils/cleaning.py` | Pipeline làm sạch dữ liệu | 3 phút |
| **5** | `utils/outlier.py` | Phương pháp xử lý outlier (IQR Capping) | 2 phút |
| **6** | `utils/feature_engineering.py` | Tạo 6 feature mới từ dữ liệu gốc | 3 phút |
| **7** | `utils/preprocessing.py` | One-Hot Encoding + RobustScaler | 2 phút |
| **8** | `utils/pipeline.py` | Tổng hợp toàn bộ pipeline `prepare_dataset()` | 1 phút |
| **9** | `utils/model.py` | Train 3 mô hình, so sánh kết quả | 3 phút |
| **10** | `utils/evaluation.py` + `results/model_metrics.csv` | Đánh giá RMSE, MAE, R² | 2 phút |
| **11** | `predict.py` | Module dự đoán tích hợp vào Flask | 2 phút |
| **12** | `app.py` + `templates/index.html` | Demo ứng dụng web | 3 phút |
| **13** | `PRESENTATION.md` | Tổng kết & hướng phát triển | 2 phút |

---

## ✅ Điểm Mạnh

### 1. Cấu trúc module hóa rõ ràng
Code được chia thành các module độc lập trong `utils/`, mỗi module đảm nhiệm một nhiệm vụ riêng:
- `cleaning.py` → Làm sạch dữ liệu
- `feature_engineering.py` → Tạo đặc trưng mới
- `preprocessing.py` → Encoding & Scaling
- `model.py` → Huấn luyện & lưu mô hình
- `evaluation.py` → Đánh giá
- `pipeline.py` → Tổng hợp
- `outlier.py` → Xử lý ngoại lai
- `eda.py` → Phân tích khám phá

### 2. Pipeline có thể tái sử dụng
Hàm `preprocess_data(training=True/False)` dùng chung cho cả train và predict, đảm bảo tính nhất quán giữa 2 môi trường.

### 3. Feature Engineering có ý nghĩa thực tiễn
Các feature mới được tạo ra phản ánh đúng logic kinh doanh:
- `Car_Age` = 2026 − Year (tuổi xe)
- `Mileage_Per_Year` = Mileage / Car_Age (cường độ sử dụng)
- `Engine_Efficiency` = Mileage / Engine_Size
- `Engine_Size_Group`, `Mileage_Group`, `Year_Group` (phân nhóm)

### 4. So sánh đa mô hình
Train 3 mô hình (Linear Regression, Random Forest, XGBoost), so sánh bằng RMSE, MAE, R² và tự động chọn mô hình tốt nhất.

### 5. Cơ chế cache model
`model_exists()` kiểm tra sự tồn tại của `best_model.pkl`, `scaler.pkl`, `feature_columns.pkl` để tránh train lại không cần thiết.

### 6. Giao diện web thân thiện
Ứng dụng Flask có giao diện Bootstrap 5, dropdown Model được lọc theo Brand, form giữ lại giá trị đã nhập sau khi submit.

### 7. Alignment khi predict
Hàm `align_columns()` đảm bảo dữ liệu người dùng nhập (1 dòng) khớp chính xác với cấu trúc feature lúc train.

### 8. Tài liệu đầy đủ
Có `PRESENTATION.md` bằng tiếng Việt với đầy đủ mục lục, sơ đồ, bảng biểu.

---

## ⚠️ Điểm Yếu

### 1. R² = 0.9995 — Dấu hiệu bất thường
Giá trị R² cực cao này có thể là dấu hiệu của:
- **Data Leakage**: Có feature chứa thông tin từ biến mục tiêu (Price)
- **Dữ liệu synthetic quá sạch**: Dữ liệu được tạo nhân tạo với quan hệ tuyến tính hoàn hảo
- **Thiếu Cross-Validation**: Chỉ đánh giá trên 1 lần split, chưa kiểm tra tính tổng quát

→ **Cần kiểm tra**: Chạy cross-validation, kiểm tra feature importance, xác nhận nguồn gốc dữ liệu.

### 2. Không có Cross-Validation
Chỉ dùng `train_test_split` 80/20 một lần duy nhất. Nên dùng `cross_val_score` với k-fold để đánh giá khách quan hơn.

### 3. Thiếu Error Handling
`app.py` không có try-except khi gọi `predict_price()`. Nếu model file bị thiếu hoặc input không hợp lệ, người dùng sẽ thấy lỗi 500 thay vì thông báo thân thiện.

### 4. Thiếu Hyperparameter Tuning
Các mô hình dùng tham số mặc định hoặc cố định, chưa được tối ưu bằng GridSearchCV.

### 5. Không có logging
Toàn bộ project dùng `print()` thay vì module `logging` chuẩn, khó debug khi deploy production.

### 6. Không có Unit Tests
Chưa có bất kỳ file test nào để kiểm tra tính đúng đắn của các hàm.

---

## 🔧 Các Điểm Cần Cải Thiện

| # | Vấn đề | Đề xuất | Độ ưu tiên |
|---|--------|---------|------------|
| 1 | R² bất thường | Chạy cross-validation, kiểm tra data leakage | 🔴 Cao |
| 2 | Thiếu Cross-Validation | Thêm `cross_val_score` vào `evaluation.py` | 🔴 Cao |
| 3 | Thiếu error handling | Bọc `predict_price` trong try-except ở `app.py` | 🟡 Trung bình |
| 4 | Thiếu Hyperparameter Tuning | Thêm GridSearchCV cho RandomForest và XGBoost | 🟡 Trung bình |
| 5 | Dùng `print()` thay vì `logging` | Thay bằng `logging` module | 🟡 Trung bình |
| 6 | `preprocessing.py` hardcode path | Dùng `config.SCALER_PATH` thay vì `"models/scaler.pkl"` | 🟢 Thấp |
| 7 | Thiếu `.gitignore` | Tạo `.gitignore` loại trừ `models/`, `__pycache__/`, `*.pyc` | 🟢 Thấp |
| 8 | Model dropdown load toàn bộ | Dùng AJAX để load model theo brand đã chọn | 🟢 Thấp |
| 9 | Thiếu SHAP/Feature Importance | Thêm biểu đồ feature importance để giải thích mô hình | 🟢 Thấp |
| 10 | Chưa có CI/CD | Tự động hóa re-train khi có dữ liệu mới | 🟢 Thấp |

---

## 🐛 Lỗi Đã Phát Hiện & Sửa

| # | Mô tả | File | Mức độ | Trạng thái |
|---|-------|------|--------|-----------|
| 1 | `requirements.txt` chứa thư viện không liên quan (Django, gym, pygame-ce...) và **thiếu các thư viện chính**: flask, scikit-learn, xgboost, joblib, seaborn | `requirements.txt` | 🔴 Nghiêm trọng | ✅ Đã sửa |
| 2 | `predict.py` load `scaler.pkl` và `feature_columns.pkl` 2 lần không cần thiết (dòng 38-40) — vừa tốn I/O vừa không dùng kết quả | `predict.py` | 🟡 Trung bình | ✅ Đã sửa |
| 3 | `app.py`: `round(predict_price(...), 0)` trả về `float` (vd: `12345.0`) thay vì `int` | `app.py` | 🟡 Trung bình | ✅ Đã sửa |
| 4 | `clean_data()` trong `cleaning.py` **không gọi outlier capping** mặc dù module `outlier.py` đã được viết sẵn với đầy đủ hàm `cap_outliers()` | `utils/cleaning.py` | 🟡 Trung bình | ✅ Đã sửa |

---

## 📊 Kết Quả Mô Hình

| Mô hình | RMSE | MAE | R² |
|----------|------|-----|----|
| **Linear Regression** ⭐ | **64.37** | **21.83** | **0.9995** |
| XGBoost | 128.32 | 101.32 | 0.9982 |
| Random Forest | 249.09 | 182.69 | 0.9932 |

> Mô hình được chọn: **Linear Regression** — RMSE thấp nhất, R² cao nhất.

---

## 🏗️ Cấu Trúc Dự Án

```
CarPricePrediction/
├── config.py                 # Cấu hình đường dẫn, hằng số
├── train.py                  # Script huấn luyện chính
├── predict.py                # Module dự đoán (dùng bởi Flask)
├── app.py                    # Flask Web Application
├── requirements.txt          # Thư viện Python (đã sửa)
├── Procfile                  # Deploy Heroku
├── PRESENTATION.md           # Tài liệu thuyết trình
├── BAO_CAO.md                # File báo cáo này
│
├── data/
│   └── car_price_dataset.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   └── Model_Training_annotated.ipynb
│
├── utils/
│   ├── __init__.py
│   ├── cleaning.py           # Làm sạch dữ liệu
│   ├── outlier.py            # Xử lý outlier (IQR)
│   ├── feature_engineering.py # Tạo feature mới
│   ├── eda.py                # Hàm phân tích EDA
│   ├── preprocessing.py      # Encoding & Scaling
│   ├── model.py              # Train & lưu mô hình
│   ├── evaluation.py         # Đánh giá model
│   ├── pipeline.py           # Pipeline tổng hợp
│   └── data_loader.py        # Load CSV
│
├── models/                   # Artifacts đã train
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── results/
│   ├── model_metrics.csv
│   └── predictions.csv
│
├── templates/
│   └── index.html
│
└── static/
    └── css/
        └── style.css
```

---

## 🎯 Kết Luận

Dự án **Car Price Prediction** có cấu trúc code tốt, module hóa rõ ràng, pipeline hoàn chỉnh từ đọc dữ liệu → làm sạch → feature engineering → huấn luyện → deploy. Tất cả các lỗi đã được phát hiện và sửa chữa.

**Lưu ý khi thuyết trình:** Nên chủ động đề cập đến việc R² = 0.9995 có thể là dấu hiệu của data leakage hoặc dữ liệu synthetic — đây là điểm thể hiện tư duy phản biện và hiểu biết sâu về ML, sẽ được đánh giá cao.

---

> *Báo cáo được tạo ngày 13/07/2026*
