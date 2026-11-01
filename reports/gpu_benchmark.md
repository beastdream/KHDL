# Benchmark CPU/GPU

Ngày: 2026-10-19T13:14:02.384697+07:00; XGBoost 3.2.0.

Cùng Train, 8 features, preprocessing, hyperparameters, seed và n_jobs=1. Một warmup không tính giờ + 3 lần fit tuần tự/thiết bị. Thời gian gồm Pipeline.fit và preprocessing. Không tuning, không dùng Test; model benchmark chỉ trong RAM, không thay model chính.

| Thiết bị | Fit median s | Std s | 1 mẫu ms | 100 mẫu ms | Bytes |
|---|---:|---:|---:|---:|---:|
| cpu | 0.2315 | 0.0049 | 9.028 | 10.175 | 1140366 |
| cuda | 1.1770 | 0.0736 | 12.660 | 11.593 | 1161001 |

Tỷ số CPU/GPU = 0.197. Tỷ số >1 là GPU nhanh hơn; <1 là GPU chậm hơn. Dataset nhỏ, overhead truyền dữ liệu và khởi tạo có thể chi phối. Không ngoại suy cho dữ liệu lớn hoặc phần cứng khác.

Latency gồm transform + tạo DMatrix + Booster.predict, median 100 lần sau warmup. Đây là đường host-input có thể so sánh, không phải latency HTTP/Streamlit. Inference triển khai vẫn CPU. Booster mỗi fit xác minh actual_device và GPU updater; không coi GPU READY lịch sử là bằng chứng benchmark mới.

Model chính: 1160913 bytes. Peak VRAM: không đo được đáng tin cậy; không thay bằng snapshot nvidia-smi. GPU inventory, driver, compatibility CUDA và cấu hình chi tiết lưu trong gpu_benchmark.json.