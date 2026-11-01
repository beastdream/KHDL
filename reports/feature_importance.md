# Permutation Importance trên Validation

Pipeline đã đóng băng, scoring neg_mean_absolute_error, 20 lần hoán vị, seed 42. Giá trị là mức tăng MAE (USD), error bar là độ lệch chuẩn qua các hoán vị, không phải khoảng tin cậy.

| Hạng | Thuộc tính | Tăng MAE USD | Std USD |
|---:|---|---:|---:|
| 1 | OverallQual | 18030.03 | 1046.71 |
| 2 | GrLivArea | 16438.91 | 1154.67 |
| 3 | TotalBsmtSF | 7045.78 | 838.99 |
| 4 | YearBuilt | 4525.26 | 803.47 |
| 5 | Neighborhood | 2159.55 | 487.99 |
| 6 | GarageCars | 2010.95 | 451.51 |
| 7 | BedroomAbvGr | 16.21 | 197.13 |
| 8 | FullBath | -261.09 | 66.51 |

Quan trọng nhất theo phép đo này: OverallQual; thấp nhất: FullBath. Hoán vị Neighborhood ở cột gốc trước preprocessing, nên toàn bộ nhóm dummy one-hot thay đổi cùng nhau; không cộng tùy tiện importance của từng dummy.

EDA Train cho thấy quan hệ giá với chất lượng, diện tích, khu vực và năm xây dựng (eda_analysis.md). Xếp hạng ở đây đo đóng góp dự đoán có điều kiện trong model, không bằng thứ tự tương quan Pearson. Các biến như diện tích, tầng hầm, garage và chất lượng có thể chia sẻ thông tin; biến tương quan có thể bù nhau và làm giảm importance riêng lẻ. Hoán vị cũng có thể tạo tổ hợp ít gặp. Importance thấp không chứng minh biến vô ích, importance cao không chứng minh nhân quả. Không loại feature, tuning hoặc chọn lại model từ kết quả này; không dùng Test.