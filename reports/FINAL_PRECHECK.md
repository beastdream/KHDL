# Kiểm tra trước nghiệm thu cuối

Ngày: 2026-10-19T13:12:32.667612+07:00. Gate: PASS. Kiểm tra hoàn tất trước khi mở Test.

Đã quét cấu trúc và đọc README Tuần 1–3, requirements/lock, pytest, app/theme, notebook, báo cáo Tuần 1–3/UI, manifest, metrics, model selection, GPU audit, artifact metadata và các module dữ liệu/model/inference/tests. Không thiếu file đầu vào bắt buộc. Các deliverable Tuần 4 chưa tồn tại tại thời điểm bắt đầu.

Kế thừa: audit Ames, split cố định, 7 biểu đồ EDA + notebook đã thực thi, baseline, 20 cấu hình CV 5 fold, 103 GPU fit, Pipeline cuối, 5 trang Streamlit và UI navy/sáng đã refactor. Không xây lại giao diện.

Kiểm thử mới: {'week2': {'passed': 27, 'skipped': 14}, 'week3': {'passed': 27, 'skipped': 0}}. Không có failure/error. Preservation PASS; xem final_preservation_evidence.json. src/week1.py khớp hash gốc 9ff7049430ee114f9e2c2b0eec4909a49a4b760203c4da8bd56085f16159180c. Lỗi cũ thực tế là 3 LF thừa ở requirements.txt, đã khôi phục về phiên bản được duyệt; không sửa checksum gốc/test. Bằng chứng: week3_repair_notes.md và week3_repair_evidence/. git status/log xác nhận không có repository; không thể quy nguồn thay đổi qua lịch sử Git.

Model: XGBoost tuned CV#3; SHA-256 133f798a66927bf9b6be827bf733c6007a3b0779b82b2f43f6c881f648205c71. Validation CPU tính lại: {'MAE': 17431.26953125, 'RMSE': 24782.89087253543, 'R2': 0.902214765548706}; phù hợp metrics chọn model trong rtol=1e-6. validation_metrics.json thuộc baseline Tuần 2, không phải model cuối; dùng model_metadata.json cho model cuối.

GPU lịch sử: GTX 1650 4096 MiB, driver 566.07, compatibility CUDA 12.7, XGBoost 3.2.0. 103 fit có cuda:0/grow_gpu_hist; CPU inference là chủ động. Benchmark mới sẽ xác minh thiết bị thực tế.

Train/Valid xác minh hash, schema, membership, imputer Train-only, encoder Train-only; manifest IDs/indices ba tập không giao nhau, seed 42, không chia lại. Test 15% nội bộ có nhãn khác test.csv Kaggle không nhãn. Chưa mở/rehash holdout.

Không còn lỗi nghiêm trọng chặn freeze. Hạng mục tiếp theo: kiểm soát truy cập Test, đánh giá chính thức, phân tích sai số/importance, benchmark, trang hiệu năng và kiểm thử nghiệm thu. 14 tests cũ cần raw/holdout vẫn SKIP theo policy cũ, không gọi là PASS.
