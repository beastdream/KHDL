# Mô hình dự đoán giá nhà TP.HCM

Dữ liệu: 6,622 Train / 1,419 Validation / 1,419 Test (xem vn_data_funnel.md). Đơn vị giá: tỷ VNĐ. Mọi mô hình (trừ Dummy) học trên log(giá).

## So sánh trên Validation

| Mô hình                   |   MAE |   RMSE |     R2 |   MAPE_% |   MedAPE_% |
|:--------------------------|------:|-------:|-------:|---------:|-----------:|
| Dummy (trung vị)          | 4.165 |  9.271 | -0.052 |   69.089 |     37.895 |
| Linear Regression (Ridge) | 2.763 | 16.402 | -2.292 |   28.942 |     19.563 |
| Random Forest             | 1.831 |  4.986 |  0.696 |   21.349 |     14.468 |
| XGBoost (tinh chỉnh CV)   | 1.800 |  4.923 |  0.703 |   20.548 |     14.965 |

**Mô hình được chọn:** XGBoost (tinh chỉnh CV) (MAE Validation thấp nhất).

XGBoost tốt nhất qua 5-fold CV trên Train: MAE CV = 1.768 tỷ, tham số {'subsample': 0.8, 'reg_lambda': 5, 'n_estimators': 1000, 'min_child_weight': 1, 'max_depth': 4, 'learning_rate': 0.03, 'colsample_bytree': 0.7}.

## Đánh giá cuối trên Test (chạy một lần)

| Chỉ số | Mô hình chọn | Dummy |
|---|---:|---:|
| MAE | 1.741 | 4.213 |
| RMSE | 4.600 | 9.460 |
| R2 | 0.753 | -0.046 |
| MAPE_% | 20.004 | 67.566 |
| MedAPE_% | 14.656 | 37.895 |

## Sai số theo khoảng giá (Test)

| nhóm     |      n |   MAE (tỷ) |   MAPE % |
|:---------|-------:|-----------:|---------:|
| <3 tỷ    | 220.00 |       0.55 |    28.62 |
| 3–5 tỷ   | 366.00 |       0.72 |    17.63 |
| 5–8 tỷ   | 457.00 |       1.00 |    15.91 |
| 8–12 tỷ  | 172.00 |       1.71 |    18.04 |
| 12–20 tỷ | 118.00 |       3.40 |    22.80 |
| ≥20 tỷ   |  86.00 |      10.81 |    29.93 |

## Permutation importance trên Validation (mức tăng MAE, tỷ VNĐ)

|                |   Tăng MAE |
|:---------------|-----------:|
| size           |      2.167 |
| house_type     |      0.530 |
| latitude       |      0.461 |
| longitude      |      0.414 |
| district       |      0.331 |
| rooms          |      0.210 |
| floors         |      0.198 |
| legal          |      0.164 |
| car_alley      |      0.083 |
| toilets        |      0.058 |
| width          |      0.036 |
| length         |      0.021 |
| furnishing     |      0.013 |
| direction      |      0.011 |
| expanding_back |      0.003 |

## Nhận xét

- Linear Regression có MAPE chấp nhận được nhưng RMSE rất lớn và R² âm: mô hình tuyến tính trên log(giá) cho vài căn diện tích lớn dự đoán quá cao, và khi đổi ngược exp() thì sai số bị khuếch đại. Mô hình cây (Random Forest, XGBoost) không ngoại suy nên ổn định hơn.
- Sai số phần trăm cao nhất ở nhóm <3 tỷ và ≥20 tỷ: hai đầu phân phối có ít mẫu và nhiều tin nhiễu hơn.
- Diện tích, loại nhà và vị trí (toạ độ, quận) quyết định phần lớn giá; hướng nhà và nội thất gần như không ảnh hưởng.

## Giới hạn

- Giá là **giá rao bán**, không phải giá giao dịch thật; thường cao hơn giá chốt.
- Dữ liệu là ảnh chụp tại một thời điểm thu thập; không phản ánh biến động theo thời gian.
- Tin đăng do người dùng tự nhập nên vẫn còn nhiễu sau khi làm sạch.
