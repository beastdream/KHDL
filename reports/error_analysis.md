# Phân tích sai số Test

Phần dư = giá thực tế − dự đoán; dương là dự đoán thấp. Chỉ đọc test_predictions.json đã xuất.

Trung bình phần dư 1,226.71 USD; trung vị 578.83 USD; tỷ lệ dự đoán thấp 51.6%; skew 0.816. Trung bình phản ánh thiên lệch tổng thể, không có nghĩa mọi căn nhà bị sai cùng hướng.

| Khoảng giá USD | n | MAE USD | MAPE % | Phần dư TB USD |
|---|---:|---:|---:|---:|
| <100k | 20 | 11935.63 | 16.71 | -8907.17 |
| 100–200k | 148 | 12810.12 | 8.90 | -2233.27 |
| 200–300k | 35 | 21067.25 | 8.91 | 5207.10 |
| 300–400k | 10 | 47337.75 | 14.07 | 17993.78 |
| >=400k | 6 | 69188.20 | 12.55 | 69188.20 |

Nhóm MAE cao nhất: >=400k, n=6. Đây là phân nhóm theo giá thực tế sau đánh giá, không phải tiêu chí chọn lại model. So sánh cả MAPE để tránh kết luận chỉ do đơn vị giá; nhóm ít mẫu có độ bất định lớn.

MAE Test 17,171.10 USD, RMSE 26,740.50 USD. RMSE/MAE = 1.56: sai số lớn ảnh hưởng RMSE mạnh hơn. Sai số tuyệt đối lớn nhất 121,419.38 USD.

So với Validation, MAE đổi -1.49%, RMSE đổi +7.90%, R² đổi +0.00383. MAE gần nhau trong khi RMSE Test tăng, phù hợp ảnh hưởng một số sai số lớn. Không có suy giảm đồng loạt cho thấy overfitting nghiêm trọng; cũng không đủ cơ sở loại trừ overfitting hoặc gọi khác biệt có ý nghĩa thống kê khi chưa định lượng độ bất định. Validation đã qua lựa chọn model.

219 mẫu holdout là nhỏ; vài nhà đắt có thể làm thay đổi RMSE đáng kể. Một split ngẫu nhiên không đại diện cho thay đổi thời gian, địa lý hoặc thị trường Việt Nam. Không tối ưu sau Test. Có 0 giá thực tế bằng 0; MAPE loại mẫu giá 0 thay vì chia cho 0.

Biểu đồ: figures/final/actual_vs_predicted.png, residual_distribution.png, residual_vs_predicted.png, absolute_error_distribution.png, error_by_price_range.png, percentage_error_by_price_range.png.