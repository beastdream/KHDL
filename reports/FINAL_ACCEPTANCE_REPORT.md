# Nghiệm thu cuối HousePrice AI

Ngày: 2026-10-19T13:32:25.680014+07:00

**PROJECT STATUS: READY**

**GPU STATUS: GPU READY**

- [x] Dataset Audit hoàn thành (bằng chứng Tuần 1 bảo toàn)
- [x] Split 70/15/15 chính xác, seed 42
- [x] Data Leakage được kiểm soát theo protocol
- [x] Jupyter Notebook đã thực thi và audit PASS
- [x] Có ít nhất 7 biểu đồ EDA
- [x] Interactive EDA và bốn bộ lọc hoạt động
- [x] XGBoost Training có bằng chứng fit
- [x] GPU status được xác minh thực tế
- [x] Hyperparameter Tuning hoàn thành trước Test
- [x] Validation Metrics model cuối đầy đủ
- [x] Model Artifact được đóng băng trước Test
- [x] Final Test Evaluation chính thức hoàn thành
- [x] MAE, RMSE, R² đầy đủ, Dummy so sánh Train-only
- [x] Error Analysis và sáu hình hoàn thành
- [x] Permutation Importance trên Validation hoàn thành
- [x] GPU Benchmark thực hiện, không thay model chính
- [x] Website Streamlit và HTTP health hoạt động
- [x] UI/UX không có lỗi nghiêm trọng trong phạm vi QA
- [x] Các bài test bắt buộc trong phạm vi policy PASS; 14 SKIP khai báo riêng
- [x] FINAL_README hướng dẫn bản cuối đầy đủ; README lịch sử giữ nguyên
- [x] Báo cáo học thuật tám chương hoàn chỉnh
- [x] Có hướng dẫn chạy và demo

## Căn cứ và phạm vi nghiệm thu

| Tập / Model | MAE USD | RMSE USD | R² |
|---|---:|---:|---:|
| Validation / XGBoost cuối | 17,431.28 | 24,782.91 | 0.902215 |
| Test / Dummy Regressor | 55,352.09 | 87,732.91 | -0.011398 |
| Test / XGBoost cuối | 17,171.10 | 26,740.50 | 0.906042 |


Pytest: {'PASS': 73, 'FAIL': 0, 'SKIPPED': 14}. JUnit tổng hợp final_pytest.xml; skip từng test ghi final_test_results.md. Preservation: PASS. Model SHA-256 `133f798a66927bf9b6be827bf733c6007a3b0779b82b2f43f6c881f648205c71` giữ nguyên. Test đánh giá chính thức một lần, không thay model sau Test. Byte/IDs holdout được xác minh trong evaluator; tests hậu kiểm dùng export. Không tuyên bố đã chạy lại mọi test Tuần 1 bị policy cấm.

Streamlit: HTTP root/health 200, Chrome kiểm tra desktop/mobile, 7 hình mới tải được, đủ 5 trang và form thật; server QA đã dừng. Bằng chứng final_streamlit_verification.json, final_ui_browser_verification.json và final_ui_screenshots/.

GPU status: GPU READY; CPU fit median 0.2315s, GPU 1.1770s. GPU chậm hơn CPU khoảng 5.09 lần. Peak VRAM chưa đo; không tạo số liệu thay thế.

## Hạn chế được chấp nhận và công việc chưa xác minh

- 14 tests raw/holdout cũ SKIP theo phạm vi dữ liệu, không tính PASS; audit nguồn toàn bộ kế thừa bằng chứng đã bảo toàn.
- Chưa xác minh cài từ đầu trên máy Windows sạch, mọi trình duyệt, screen reader, zoom 200% hoặc thiết bị thật.
- Còn cảnh báo deprecated notebook; chưa có prediction intervals hoặc đánh giá ngoài Ames/theo thời gian.
- Benchmark trên workstation có thể chịu tải nền; không đo peak VRAM và không suy rộng hiệu năng.
- Dataset Test đã dùng, không được tái sử dụng để tuning. Audit hook là chính sách chống truy cập nhầm, không chống mã ác ý.

Những giới hạn trên được công bố, không phải lỗi nghiêm trọng còn tồn tại về dữ liệu/model/kết quả trong phạm vi nghiệm thu. READY áp dụng cho đồ án demo/nghiên cứu này, không phải chứng nhận triển khai thương mại. GPU READY chỉ xác nhận chạy GPU thật, không đồng nghĩa tăng tốc.

## Hướng dẫn nộp và demo

Xem FINAL_README.md ở gốc. Lệnh tests: `.\.venv\Scripts\python.exe -m src.utils.validate_final`. Lệnh website: `.\.venv\Scripts\python.exe -m streamlit run app.py`, mở http://localhost:8501. Demo theo thứ tự Tổng quan → EDA → Dự đoán → Validation/Test → benchmark/giới hạn. Không cần training hoặc đánh giá Test lại.
