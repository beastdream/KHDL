# Phân tích EDA — chỉ Train

Phân tích 1022 mẫu Train; không đọc Validation hoặc Test để EDA.

## Xu hướng chính

SalePrice có skewness 1.762, trung vị 165,000 USD. Phân phối lệch phải: một số nhà giá cao kéo dài đuôi phân phối.
Diện tích sinh hoạt, chất lượng tổng thể và năm xây dựng có liên hệ với giá; biểu đồ cho thấy độ phân tán đáng kể giữa các căn nhà.

## Tương quan với giá

| Biến | Pearson r với SalePrice |
|---|---:|
| OverallQual | 0.7847 |
| GrLivArea | 0.6892 |
| GarageCars | 0.6427 |
| TotalBsmtSF | 0.5900 |
| FullBath | 0.5492 |
| YearBuilt | 0.5122 |
| BedroomAbvGr | 0.1599 |

Biến có tương quan tuyến tính dương lớn nhất trong 7 biến số là OverallQual (0.785).
Số phòng ngủ đơn lẻ không đại diện đầy đủ cho diện tích/chất lượng; không suy ra tăng phòng ngủ sẽ làm tăng giá tương ứng.

## Neighborhood

| Neighborhood | Trung vị USD | Số mẫu Train |
|---|---:|---:|
| MeadowV | 89,500 | 10 |
| BrDale | 103,000 | 12 |
| IDOTRR | 107,250 | 24 |
| BrkSide | 119,000 | 41 |
| OldTown | 123,250 | 82 |
| Blueste | 124,000 | 1 |
| Edwards | 125,000 | 79 |
| Sawyer | 134,000 | 48 |
| SWISU | 139,750 | 16 |
| NAmes | 142,000 | 155 |
| NPkVill | 147,750 | 6 |
| Mitchel | 158,950 | 34 |
| SawyerW | 179,100 | 38 |
| Blmngtn | 179,870 | 14 |
| NWAmes | 183,950 | 54 |
| Gilbert | 184,050 | 52 |
| Crawfor | 195,550 | 38 |
| CollgCr | 200,500 | 109 |
| ClearCr | 211,000 | 19 |
| Somerst | 227,680 | 57 |
| Timber | 228,000 | 27 |
| Veenker | 231,750 | 8 |
| StoneBr | 282,000 | 20 |
| NoRidge | 290,000 | 28 |
| NridgHt | 312,000 | 50 |

Giá trung vị theo khu vực là so sánh thô, chưa kiểm soát diện tích, chất lượng và năm xây dựng. Nhóm ít mẫu có ước lượng kém ổn định.

## Outlier

Có 41 mẫu ngoài ngưỡng 1.5 IQR của giá và 3 mẫu GrLivArea > 4.000 sq ft. Đây là cờ xem xét, không phải bằng chứng dữ liệu sai.
Các mẫu diện tích lớn: [{"Id": 1183, "GrLivArea": 4476, "SalePrice": 745000}, {"Id": 524, "GrLivArea": 4676, "SalePrice": 184750}, {"Id": 1299, "GrLivArea": 5642, "SalePrice": 160000}]
Danh sách ID outlier giá: [1183, 1143, 179, 679, 225, 322, 1354, 337, 350, 1389, 988, 645, 390, 441, 1170, 804, 1182, 186, 279, 689, 826, 12, 516, 474, 1269, 314, 321, 497, 1374, 878, 54, 152, 609, 113, 643, 379, 162, 703, 1268, 1438, 770]
Không xóa hoặc winsorize bất kỳ mẫu nào.

## Hạn chế diễn giải

Tương quan không chứng minh quan hệ nhân quả. Pearson chỉ đo liên hệ tuyến tính, nhạy với outlier; các features có thể tương quan lẫn nhau.
OverallQual là thang thứ bậc nên hệ số Pearson chỉ mang tính mô tả. Neighborhood là biến phân loại, không mã hóa thành số thứ tự để tính Pearson.
Ames là một thị trường trong một giai đoạn lịch sử; kết quả không tự động khái quát sang địa phương, thời điểm hoặc thị trường Việt Nam.
EDA chỉ dựa trên Train. Missing values chưa được điền trong biểu đồ; preprocessing về sau chỉ học thống kê Train. Không dùng EDA để loại mẫu test hoặc chỉnh split.

## Biểu đồ

![Train plot](figures\week2_01_price_histogram.png)

![Train plot](figures\week2_02_area_price.png)

![Train plot](figures\week2_03_quality_price.png)

![Train plot](figures\week2_04_neighborhood_median.png)

![Train plot](figures\week2_05_correlation.png)

![Train plot](figures\week2_06_year_price.png)

![Train plot](figures\week2_07_bedrooms_price.png)