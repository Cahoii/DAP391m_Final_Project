# 🚗 Car Price Prediction
### Dự án Machine Learning — Nhóm Sinh Viên | DAP · SS4 SU26

---

## 📋 Mục Lục

1. [Giới thiệu dự án](#1-giới-thiệu-dự-án)
2. [Bộ dữ liệu](#2-bộ-dữ-liệu)
3. [Phân tích khám phá dữ liệu (EDA)](#3-phân-tích-khám-phá-dữ-liệu-eda)
4. [Tiền xử lý dữ liệu](#4-tiền-xử-lý-dữ-liệu)
5. [Feature Engineering](#5-feature-engineering)
6. [Mô hình Machine Learning](#6-mô-hình-machine-learning)
7. [Kết quả & Đánh giá](#7-kết-quả--đánh-giá)
8. [Ứng dụng Web](#8-ứng-dụng-web)
9. [Cấu trúc dự án](#9-cấu-trúc-dự-án)
10. [Kết luận & Hướng phát triển](#10-kết-luận--hướng-phát-triển)

---

## 1. Giới Thiệu Dự Án

### 🎯 Mục tiêu
> Xây dựng mô hình Machine Learning có khả năng **dự đoán giá xe ô tô** dựa trên các đặc điểm của xe như hãng xe, năm sản xuất, dung tích động cơ, số km đã đi, v.v.

### ❓ Vấn đề đặt ra
Thị trường xe ô tô cũ có sự biến động giá lớn và thiếu minh bạch. Người mua và người bán thường khó xác định được mức giá hợp lý cho một chiếc xe. Dự án này nhằm:

- **Hỗ trợ người mua** tránh mua xe với giá quá cao
- **Hỗ trợ người bán** định giá xe hợp lý, cạnh tranh
- **Tăng tính minh bạch** trong thị trường xe cũ

### 💡 Giải pháp
Xây dựng pipeline hoàn chỉnh gồm: thu thập dữ liệu → EDA → tiền xử lý → feature engineering → huấn luyện mô hình → triển khai ứng dụng web.

---

## 2. Bộ Dữ Liệu

### 📂 Nguồn dữ liệu
| Thuộc tính | Chi tiết |
|---|---|
| **File** | `data/car_price_dataset.csv` |
| **Kích thước** | ~574 KB |
| **Định dạng** | CSV |

### 📊 Các đặc trưng (Features)

| Tên cột | Loại | Mô tả |
|---|---|---|
| `Brand` | Categorical | Hãng xe (Toyota, BMW, Ford, ...) |
| `Model` | Categorical | Tên mẫu xe |
| `Year` | Numerical | Năm sản xuất |
| `Engine_Size` | Numerical | Dung tích động cơ (lít) |
| `Fuel_Type` | Categorical | Loại nhiên liệu (Petrol, Diesel, Electric, ...) |
| `Transmission` | Categorical | Hộp số (Manual, Automatic) |
| `Mileage` | Numerical | Số km đã đi |
| `Doors` | Numerical | Số cửa xe |
| `Owner_Count` | Numerical | Số lần đổi chủ |
| **`Price`** | **Numerical** | **🎯 Biến mục tiêu — Giá xe (USD)** |

---

## 3. Phân Tích Khám Phá Dữ Liệu (EDA)

> 📓 Chi tiết trong notebook: `notebooks/EDA.ipynb`

### 🔍 Các bước thực hiện

```
Tổng quan Dataset
      ↓
Phân phối biến mục tiêu (Price)
      ↓
Phân tích biến số (Numerical)
      ↓
Phân tích biến phân loại (Categorical)
      ↓
Ma trận tương quan (Correlation Matrix)
      ↓
Phát hiện Outlier
```

### 📌 Những phát hiện chính từ EDA

- **Phân phối giá xe** có dạng lệch phải (right-skewed) — một số xe có giá rất cao
- **Năm sản xuất** và **số km đã đi** có tương quan mạnh với giá xe
- **Hãng xe** ảnh hưởng đáng kể đến giá — xe hạng sang cao hơn rõ rệt
- **Dung tích động cơ** lớn hơn thường đi kèm với mức giá cao hơn
- Phát hiện các **outlier** ở cột `Mileage` và `Price` cần xử lý

---

## 4. Tiền Xử Lý Dữ Liệu

> 📓 Chi tiết trong notebook: `notebooks/Model_Training_annotated.ipynb`

### Pipeline làm sạch dữ liệu

```
┌─────────────────────────────────────────────────┐
│              DATA CLEANING PIPELINE             │
│                                                 │
│  1. Xóa dữ liệu trùng lặp (Deduplication)      │
│         ↓                                       │
│  2. Xử lý giá trị thiếu (Missing Values)        │
│     • Số: điền bằng Median                      │
│     • Category: điền bằng Mode                  │
│         ↓                                       │
│  3. Gom nhóm hiếm (Rare Encoding, 1%)           │
│     Ví dụ: Lamborghini → "Rare"                 │
│         ↓                                       │
│  4. Xử lý Outlier (IQR Capping / Winsorization) │
│     • Tính ngưỡng Q1, Q3, IQR                   │
│     • Capping về Lower / Upper bound            │
│         ↓                                       │
│  5. Kiểm tra & xác nhận dataset                 │
└─────────────────────────────────────────────────┘
```

### Tiền xử lý mô hình

| Bước | Kỹ thuật | Công cụ |
|---|---|---|
| Mã hóa biến phân loại | One-Hot Encoding | `pd.get_dummies()` |
| Chuẩn hóa số liệu | Robust Scaler | `sklearn.RobustScaler` |
| Lưu cột features | Pickle | `joblib.dump()` |
| Căn chỉnh feature lúc predict | Column Alignment | Custom function |

> **Lý do chọn RobustScaler:** Ít bị ảnh hưởng bởi outlier hơn StandardScaler, phù hợp với dữ liệu giá xe có nhiều giá trị cực đoan.

---

## 5. Feature Engineering

> Tạo thêm các đặc trưng mới từ dữ liệu gốc để giúp mô hình học hiệu quả hơn.

### 🔧 Các feature được tạo ra

| Feature Mới | Công thức / Phương pháp | Ý nghĩa |
|---|---|---|
| `Engine_Size_Group` | Chia 3 nhóm: Small / Medium / Large | Học theo nhóm thay vì số liên tục |
| `Mileage_Group` | Chia 3 nhóm: Low / Medium / High | Mức độ sử dụng xe |
| `Year_Group` | Chia 3 nhóm: 2000s / 2010s / 2020s | Thế hệ xe |
| `Car_Age` | `2026 − Year` | Tuổi xe (có ý nghĩa hơn năm SX) |
| `Mileage_Per_Year` | `Mileage / Car_Age` | Cường độ sử dụng xe mỗi năm |
| `Engine_Efficiency` | `Mileage / Engine_Size` | Hiệu suất động cơ |

### 💡 Ví dụ minh hoạ

```
Xe A: Mileage = 100,000 km, Tuổi = 20 năm  →  Mileage_Per_Year = 5,000 km/năm
Xe B: Mileage = 100,000 km, Tuổi =  5 năm  →  Mileage_Per_Year = 20,000 km/năm

→ Xe B có cường độ sử dụng cao hơn → thường có giá thấp hơn
```

---

## 6. Mô Hình Machine Learning

### 🤖 Các mô hình được thử nghiệm

```
         Dataset (80% Train | 20% Test)
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
  Linear Regression  Random Forest  XGBoost
                     (200 trees)    (300 trees,
                                     lr=0.05)
         │             │             │
         └─────────────┼─────────────┘
                       ▼
              So sánh theo RMSE
                       ▼
               Chọn Best Model
                       ▼
             Lưu → best_model.pkl
```

### ⚙️ Siêu tham số

| Mô hình | Tham số chính |
|---|---|
| **Linear Regression** | Mặc định |
| **Random Forest** | `n_estimators=200`, `random_state=42`, `n_jobs=-1` |
| **XGBoost** | `n_estimators=300`, `learning_rate=0.05`, `max_depth=5`, `objective=reg:squarederror` |

### 📐 Tiêu chí đánh giá

| Metric | Ý nghĩa |
|---|---|
| **RMSE** | Root Mean Squared Error — tiêu chí chính để chọn model |
| **MAE** | Mean Absolute Error — sai số tuyệt đối trung bình |
| **R²** | R-squared — tỷ lệ phương sai được giải thích bởi mô hình |

---

## 7. Kết Quả & Đánh Giá

### 📈 Kết quả so sánh mô hình

| Mô hình | RMSE | MAE | R² |
|---|---|---|---|
| **Linear Regression** | **64.37** | **21.83** | **0.9995** ⭐ |
| XGBoost | 128.32 | 101.32 | 0.9982 |
| Random Forest | 249.09 | 182.69 | 0.9932 |

> **✅ Mô hình được chọn: Linear Regression**
> Đạt R² = **0.9995** — mô hình giải thích được **99.95%** sự biến động của giá xe trên tập test, với RMSE thấp nhất.

### 🏆 Nhận xét kết quả

- **Linear Regression** bất ngờ vượt trội — cho thấy mối quan hệ giữa các đặc trưng và giá xe có tính **tuyến tính cao** sau khi đã qua feature engineering và scaling tốt
- **XGBoost** đứng thứ hai với R² = 0.9982, vẫn là mô hình rất tốt
- **Random Forest** có RMSE cao nhất, nhưng R² = 0.9932 vẫn đạt chất lượng tốt
- Cả 3 mô hình đều cho kết quả xuất sắc (R² > 0.99), khẳng định chất lượng của toàn bộ pipeline

---

## 8. Ứng Dụng Web

### 🌐 Triển khai với Flask

Dự án được triển khai thành **web application** cho phép người dùng nhập thông tin xe và nhận dự đoán giá ngay lập tức.

```
Người dùng nhập thông tin xe
           ↓
       Flask App (app.py)
           ↓
  Feature Engineering (predict.py)
           ↓
  Preprocessing (Scaler + Alignment)
           ↓
     best_model.pkl → Dự đoán
           ↓
    Hiển thị giá xe dự đoán
```

### 📝 Thông tin người dùng nhập

| Trường | Kiểu dữ liệu |
|---|---|
| Hãng xe (Brand) | Dropdown |
| Mẫu xe (Model) | Dropdown (lọc theo hãng) |
| Năm sản xuất (Year) | Số nguyên |
| Dung tích động cơ (Engine Size) | Số thực |
| Loại nhiên liệu (Fuel Type) | Dropdown |
| Hộp số (Transmission) | Dropdown |
| Số km đã đi (Mileage) | Số thực |
| Số cửa (Doors) | Số nguyên |
| Số lần đổi chủ (Owner Count) | Số nguyên |

### 🔧 Công nghệ sử dụng

```
Backend  : Python · Flask
ML Libs  : scikit-learn · XGBoost · pandas · numpy · joblib
Frontend : HTML · CSS · Jinja2 Templates
Deploy   : Procfile (Heroku-compatible)
```

---

## 9. Cấu Trúc Dự Án

```
CarPricePrediction/
│
├── 📁 data/
│   └── car_price_dataset.csv           # Bộ dữ liệu gốc
│
├── 📁 notebooks/
│   ├── EDA.ipynb                       # Phân tích khám phá dữ liệu
│   └── Model_Training_annotated.ipynb  # Huấn luyện & đánh giá mô hình
│
├── 📁 utils/                           # Thư viện xử lý dữ liệu
│   ├── data_loader.py                  # Load dữ liệu
│   ├── cleaning.py                     # Làm sạch dữ liệu
│   ├── outlier.py                      # Xử lý Outlier (IQR)
│   ├── eda.py                          # Hàm phân tích EDA
│   ├── feature_engineering.py          # Tạo feature mới
│   ├── preprocessing.py                # Encoding & Scaling
│   ├── model.py                        # Train & lưu mô hình
│   ├── evaluation.py                   # Đánh giá model
│   └── pipeline.py                     # Pipeline tổng hợp
│
├── 📁 models/                          # Artifacts đã train
│   ├── best_model.pkl                  # Mô hình tốt nhất
│   ├── scaler.pkl                      # RobustScaler đã fit
│   └── feature_columns.pkl             # Danh sách cột features
│
├── 📁 results/
│   ├── model_metrics.csv               # Kết quả so sánh các model
│   └── predictions.csv                 # Dự đoán trên tập test
│
├── 📁 templates/                       # HTML templates (Flask)
├── 📁 static/                          # CSS, JS, Assets
│
├── app.py                              # Flask Web Application
├── train.py                            # Script huấn luyện mô hình
├── predict.py                          # Module dự đoán
├── config.py                           # Cấu hình đường dẫn
├── Procfile                            # Cấu hình deploy
└── requirements.txt                    # Thư viện cần thiết
```

---

## 10. Kết Luận & Hướng Phát Triển

### ✅ Những gì đã đạt được

- [x] Xây dựng pipeline tiền xử lý dữ liệu hoàn chỉnh và có thể tái sử dụng
- [x] Thực hiện EDA bài bản, phát hiện được các pattern quan trọng
- [x] Tạo các feature có ý nghĩa thực tiễn (Car Age, Mileage/Year, ...)
- [x] So sánh 3 mô hình ML, chọn ra mô hình tốt nhất
- [x] Đạt độ chính xác rất cao: **R² = 0.9995**
- [x] Triển khai thành công ứng dụng web với Flask

### 🚀 Hướng phát triển tiếp theo

| Hướng | Mô tả |
|---|---|
| **Thêm dữ liệu** | Thu thập thêm dữ liệu thực tế từ các sàn giao dịch xe |
| **Tối ưu mô hình** | Hyperparameter tuning với GridSearchCV / Optuna |
| **Feature thêm** | Màu sắc, điều kiện xe, tình trạng bảo dưỡng |
| **Giải thích mô hình** | Sử dụng SHAP values để giải thích kết quả dự đoán |
| **CI/CD Pipeline** | Tự động hoá việc re-train khi có dữ liệu mới |
| **Cloud Deploy** | Deploy lên Heroku / Render / AWS |

---

## 📚 Công Nghệ & Thư Viện

| Nhóm | Thư viện |
|---|---|
| **Data Processing** | `pandas`, `numpy` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Machine Learning** | `scikit-learn`, `xgboost` |
| **Serialization** | `joblib` |
| **Web Framework** | `Flask` |
| **Notebook** | `Jupyter` |

---

> *Dự án được thực hiện trong khuôn khổ môn học Data Analysis & Processing (DAP) — FPT University, Summer 2026.*
