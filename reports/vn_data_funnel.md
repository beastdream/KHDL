# Phễu dữ liệu nhà ở TP.HCM (Nhà Tốt)

Nguồn: tin bán nhà ở công khai trên nhatot.com, thu thập cho mục đích học tập (xem reports/vn_collection.json). Đã bỏ thông tin người đăng.

| Bước | Còn lại | Bị loại |
|---|---:|---:|
| Dữ liệu thu thập (tin không trùng mã) | 9,995 | 0 |
| Chỉ giữ tin BÁN tại TP.HCM | 9,995 | 0 |
| Bỏ tin thiếu giá, diện tích hoặc quận | 9,974 | 21 |
| Bỏ giá ngoài 0,3–300 tỷ hoặc diện tích ngoài 10–1.000 m² | 9,953 | 21 |
| Bỏ giá/m² ngoài 15–1.500 triệu (nhập sai đơn vị) | 9,797 | 156 |
| Bỏ tin đăng trùng (cùng phường, diện tích, giá, tầng, phòng) | 9,460 | 337 |

Đổi chỗ mặt tiền/chiều dài bị nhập ngược: 201 tin.
Giá trị vô lý được chuyển thành "thiếu" (không bỏ dòng): floors: 26, rooms: 0, toilets: 0, width: 4, length: 6, latitude/longitude: 5.

## Tỷ lệ thiếu theo cột (%)

| Cột | Thiếu % |
|---|---:|
| size | 0.0 |
| floors | 28.2 |
| rooms | 0.2 |
| toilets | 28.8 |
| width | 22.9 |
| length | 25.7 |
| latitude | 0.1 |
| longitude | 0.1 |
| district | 0.0 |
| house_type | 0.0 |
| legal | 25.5 |
| direction | 75.0 |
| furnishing | 55.1 |
| car_alley | 0.0 |
| expanding_back | 0.0 |

## Chia dữ liệu (seed 42)

| Tập | Số dòng |
|---|---:|
| train | 6,622 |
| valid | 1,419 |
| test | 1,419 |

## Giá bán (tỷ VNĐ)

Trung vị 5.90; trung bình 7.96; nhỏ nhất 0.35; lớn nhất 300.00; độ lệch (skew) 10.06.
