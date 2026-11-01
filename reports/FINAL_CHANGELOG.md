# Danh sách bàn giao và thay đổi cuối dự án

## File tạo mới

- Gốc: `FINAL_README.md`, `requirements-final-lock.txt`.
- Data Guard: `src/data/final_inputs.py`.
- Model/evaluation: `src/models/evaluate_test.py`, `error_analysis.py`, `feature_importance.py`, `benchmark.py`.
- Utilities: `src/utils/final_precheck.py`, `validate_final.py`, `verify_final_ui.py`, `final_report.py`.
- Tests/QA: `tests/final_policy.py`, `test_end_to_end.py`, `final_ui_browser_check.py`.
- Precheck/provenance: `reports/FINAL_PRECHECK.md`, `final_model_freeze.json`, `final_preservation_evidence.json`, `final_integrity_verification.json`, `final_evaluation_attempt.json`, `final_evaluation_verifications.json`.
- Đánh giá: `reports/test_metrics.json`, `test_evaluation.md`, `test_predictions.json`.
- Phân tích: `reports/error_analysis.json`, `error_analysis.md`, `feature_importance.json`, `feature_importance.md`, `gpu_benchmark.json`, `gpu_benchmark.md`.
- Hình: `reports/figures/final/actual_vs_predicted.png`, `residual_distribution.png`, `residual_vs_predicted.png`, `absolute_error_distribution.png`, `error_by_price_range.png`, `percentage_error_by_price_range.png`, `feature_importance.png`.
- Kết quả test: `reports/final_pytest.xml`, `final_legacy_pytest.xml`, `final_week3_pytest.xml`, `final_e2e_pytest.xml`, `final_test_counts.json`, `final_test_results.md`.
- QA browser/server: `reports/final_streamlit_verification.json`, `final_ui_browser_verification.json`, `final_streamlit.log`, `final_browser.log`, `final_ui_screenshots/` (ảnh desktop/mobile và DOM slider).
- Báo cáo: `reports/FINAL_PROJECT_REPORT.md`, `FINAL_ACCEPTANCE_REPORT.md`, `FINAL_CHANGELOG.md`, `final_summary.json`, `FINAL_SUMMARY.txt`.

## File chỉnh sửa trực tiếp

- `app.py`: bổ sung Test metrics/comparison, sáu hình Test, importance Validation, benchmark và hướng phát triển. Giữ UI refactor và form/model inference cũ.
- `src/data/app_views.py`: đọc dự đoán Validation đã xuất thay vì gọi lại model khi mở trang performance; bỏ import inference không còn cần thiết.

## Hồ sơ kế thừa tái sinh bởi lệnh kiểm tra có sẵn

`src.utils.validate_week3` chạy hai suite và hai report generators: `reports/week2_pytest.xml`, `week3_pytest.xml`, `week2_notebook_audit.json`, `week2_preservation_check.json`, `week2_status.json`, `week3_status.json`, `WEEK_2_REPORT.md`, `WEEK_3_REPORT.md`. Notebook audit còn tái sinh `eda_summary.json`, `eda_analysis.md` và bảy PNG week2 từ Train. `validate_final` có thể tái sinh cùng EDA output khi kiểm tra notebook; JUnit của bản cuối dùng tên mới riêng.

Không chỉnh source Tuần 1, notebook gốc, split/data, artifact/metadata, original checksum, approvals hoặc hook Tuần 2/3. Bằng chứng preservation PASS và freeze checksum vẫn khớp. README/requirements-lock lịch sử giữ nguyên; hướng dẫn cuối nằm trong FINAL_README.md và lock đầy đủ mới. Không có Git commit/push trong thư mục không có repository.

Đối chiếu bổ sung 34 source/artifact/notebook Tuần 1–3 với snapshot UI có sẵn: 33 file UNCHANGED; `src/data/app_views.py` là thay đổi chức năng được yêu cầu ở Phase 6 để đọc báo cáo đã xuất. Ghi hash trước/sau và lý do trong final_integrity_verification.json; không sửa snapshot tham chiếu.

## Vấn đề đã xử lý và hợp đồng kiểm thử

- Xác minh lỗi preservation đã được sửa từ trước: src/week1.py không lệch; requirements.txt từng có ba LF thừa. Không khôi phục/ghi đè thêm trong lần hoàn thiện này.
- Phân biệt metrics baseline Tuần 2 với model cuối trong metadata, tránh báo sai Validation.
- Tạo freeze trước Test, capability chỉ evaluator và read-only dataset, kiểm tra checksum/membership. Hook là chính sách quy trình chống truy cập nhầm, không phải sandbox chống mã độc.
- Chỉ một đánh giá Test chính thức; lần CLI kiểm chứng sau đó dùng cache và ghi log không mở lại Test.
- E2E mới ban đầu gọi nhầm `app.get('imgs')` (tên protobuf); đổi sang API `app.image` theo thư viện hiện cài và vẫn yêu cầu đúng bảy hình. Không xóa test cũ hoặc hạ yêu cầu.
- So sánh CPU/GPU thật, nhận diện CUDA qua Booster; không giả số GPU hoặc peak VRAM. Model benchmark chỉ trong RAM.
- Chạy các suite legacy/final trong process riêng để giữ guard cũ; không thay đổi skip policy. 14 SKIP có lý do từng trường hợp trong final_test_results.md.

## Giới hạn và lệnh thực tế

Chưa đo peak VRAM, chưa cài lại trên máy sạch, chưa kiểm tra mọi browser/thiết bị/screen reader, còn warning notebook deprecated. 14 raw/holdout tests cũ SKIP; audit nguồn kế thừa bằng chứng trước đó. Chưa có uncertainty intervals hay đánh giá ngoài miền. GPU chạy thật nhưng chậm hơn CPU trên dataset này; benchmark có thể chịu tải nền workstation. Test đã dùng không được trở thành tập tuning.

```powershell
.\.venv\Scripts\python.exe -m src.utils.validate_week3
.\.venv\Scripts\python.exe -m src.utils.final_precheck
.\.venv\Scripts\python.exe -m src.models.evaluate_test
.\.venv\Scripts\python.exe -m src.models.error_analysis
.\.venv\Scripts\python.exe -m src.models.feature_importance
.\.venv\Scripts\python.exe -m src.models.benchmark
.\.venv\Scripts\python.exe -m src.utils.validate_final
.\.venv\Scripts\python.exe -m src.utils.verify_final_ui
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m compileall -q app.py src tests/test_end_to_end.py tests/final_policy.py tests/final_ui_browser_check.py
.\.venv\Scripts\python.exe -m src.utils.final_report
```

Kiểm thử cuối: 73 PASS / 0 FAIL / 14 SKIP; xem JUnit. Máy chủ QA tạm đã dừng. Website để demo:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Mở http://localhost:8501. Hướng dẫn demo chi tiết ở FINAL_README.md: Tổng quan → lọc EDA → nhập tám features/dự đoán → Validation/Test → importance/GPU và giới hạn; không chạy lại training/Test trong demo.
