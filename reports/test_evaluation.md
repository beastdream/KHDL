# Đánh giá Test chính thức

Thời điểm: 2026-10-19T13:12:44.605907+07:00. 219 mẫu Test nội bộ có nhãn; không phải test.csv Kaggle.

Model được đóng băng trước truy cập Test; không fit, tuning, feature selection hoặc early stopping trên Test.

| Model | MAE (USD) | RMSE (USD) | R² |
|---|---:|---:|---:|
| Dummy Regressor | 55352.09 | 87732.91 | -0.011398 |
| XGBoost | 17171.10 | 26740.50 | 0.906042 |

Dummy dùng median đã fit chỉ trên Train. Pipeline nhận đúng 8 features, Id và SalePrice không đi vào X.
Chỉ 1 lần đánh giá chính thức. Chạy lại CLI chỉ xác minh checksum kết quả đã xuất; không mở lại Test.
SHA-256 model: `133f798a66927bf9b6be827bf733c6007a3b0779b82b2f43f6c881f648205c71`. Chi tiết: test_metrics.json, test_predictions.json.