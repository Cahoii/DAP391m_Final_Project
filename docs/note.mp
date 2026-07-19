1. Mở đầu 

Đề tài: Car Price Prediction — dự đoán giá xe ô tô đã qua sử dụng bằng Machine Learning.
Bài toán thực tế: người mua/bán xe khó định giá chính xác, cần công cụ hỗ trợ dựa trên dữ liệu (Year, Mileage, Engine_Size, Brand...).
Mục tiêu: xây dựng pipeline hoàn chỉnh từ EDA → tiền xử lý → train nhiều model → chọn model tốt nhất → deploy thành web app.

2. Tổng quan Dataset

10,000 mẫu, 10 thuộc tính gốc (Brand, Model, Year, Engine_Size, Fuel_Type, Transmission, Mileage, Doors, Owner_Count, Price).
Không có missing values, không duplicate — dữ liệu sạch ngay từ đầu.
đây là dataset dùng cho mục đích học tập/nghiên cứu.

3. EDA (Exploratory Data Analysis)

Phân phối Price: gần chuẩn, mean ≈ median ≈ 8,850 → dữ liệu cân đối, không cần transform log.
Các biến phân loại (Brand, Fuel_Type, Transmission) phân bố đồng đều → không bias theo nhóm.
Tương quan với Price: Year (r≈0.66), Mileage (r≈ -0.55), Engine_Size (r≈0.36) — đây là 3 biến ảnh hưởng mạnh nhất.
Brand/Fuel_Type/Transmission gần như không ảnh hưởng rõ rệt đến Price (boxplot median tương đồng).
1 slide heatmap correlation, 1 slide scatter Year/Mileage vs Price là đủ, không cần show hết mọi biểu đồ.

4. Tiền xử lý & Feature Engineering

Feature Engineering: tạo thêm Car_Age, Mileage_Per_Year, Engine_Efficiency, và các nhóm (Engine_Size_Group, Mileage_Group, Year_Group).
Preprocessing: One-Hot Encoding cho biến phân loại, RobustScaler cho biến số → 72 cột đặc trưng.
Đóng gói thành module tái sử dụng (utils/feature_engineering.py, utils/preprocessing.py) — dùng chung cho cả training và predict trên Flask (align_columns đảm bảo khớp cột lúc predict).

5. Huấn luyện & So sánh mô hình

3 mô hình: Linear Regression, Random Forest, XGBoost.
Bảng kết quả RMSE/MAE/R²:

Linear Regression: RMSE 64.4, R² 0.9995 (tốt nhất)
XGBoost: RMSE 128.3, R² 0.9982
Random Forest: RMSE 249.1, R² 0.9932


Chọn Linear Regression làm best model theo RMSE thấp nhất, lưu lại bằng joblib.

6. Phân tích kết quả 

trong file noteboooks

R² cao bất thường (>0.99 cho cả 3 model, Linear Regression còn thắng cả model phức tạp hơn) — điều này không điển hình với dữ liệu giá xe thực tế ngoài đời.
feature_engineering.py, preprocessing.py không leakage (dùng trực tiếp Price để tính feature khác) → không leakage.
Đối chiếu với tương quan thô ở bước EDA (chỉ 0.36–0.66) → kết luận: khả năng cao do dataset là dữ liệu tổng hợp (synthetic), Price được sinh từ công thức gần tuyến tính của các biến, nên khi kết hợp nhiều đặc trưng trong Linear Regression, R² tổng hợp cao dù từng tương quan đơn lẻ chỉ ở mức trung bình.

7. Ứng dụng (Demo)

Giới thiệu ngắn gọn kiến trúc Flask (app.py, predict.py, templates/, models/*.pkl).
Demo trực tiếp: nhập thông tin xe → app load best_model.pkl, scaler.pkl, feature_columns.pkl → trả về giá dự đoán.
Nếu deploy (thấy có Procfile) → có thể demo bản online thay vì chạy local, ấn tượng hơn.

8. Hạn chế & hướng phát triển

Dataset có khả năng là synthetic → cần thử nghiệm với dữ liệu thực tế (crawl từ các web bán xe...) để kiểm chứng mô hình.
Chưa dùng cross-validation, chỉ 1 lần train/test split.
Hướng mở rộng: thêm cross-validation, hyperparameter tuning cho Random Forest/XGBoost, thử thêm feature (tình trạng xe, khu vực, thời gian thu thập data... để mô hình có ý thức về không gian và thời gian).

9. Kết luận

xây dựng thành công pipeline end-to-end, đạt độ chính xác cao trên dataset hiện có, đã tự đánh giá và xác định rõ giới hạn của kết quả.