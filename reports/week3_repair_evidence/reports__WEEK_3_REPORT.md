# Báo cáo Tuần 3 — HousePrice AI

Ngày lập: 2026-10-18T16:21:35.866547+07:00

## 1. Tóm tắt công việc

Tiếp tục dự án hiện có; tuning XGBoost, chọn mô hình trên Validation, lưu Pipeline hoàn chỉnh và triển khai Streamlit tiếng Việt 5 trang.

## 2. Trạng thái kế thừa Tuần 1–2

Gate Tuần 2: NOT READY; core: NOT READY. Hai regression notebook và preservation PASS.
Đã đọc đủ các tài liệu/module bắt buộc trong Phase 0; không thiếu file trong danh sách. Giữ nguyên source Tuần 1–2, requirements, notebook, baseline model và manifest.
Train=1022, Validation=219; Test=219 theo metadata đã lưu, không mở/rehash Test. Split 70/15/15, seed 42.
test.csv ở thư mục gốc là dữ liệu Kaggle không nhãn; data/processed/test.csv là holdout nội bộ có nhãn. Cả hai đều bị khóa. Hash Train/Valid được kiểm tra qua loader cũ; không chia lại dữ liệu.
Data Guard Tuần 2 giữ nguyên. Policy Tuần 3 giữ hạn chế dữ liệu và chỉ thêm ba đường dẫn CSV báo cáo cụ thể. Hai bộ tests chạy trong hai process riêng để policy Tuần 2 không chặn CSV báo cáo mới.

## 3. Hyperparameter tuning

RandomizedSearchCV: 20 / 288 cấu hình, KFold 5-fold shuffle=True, seed 42, scoring neg_mean_absolute_error, n_jobs=1, refit=False. Tổng 100 fit CV + 3 fit ứng viên trên toàn Train.
Preprocessing nằm trong Pipeline và fit độc lập mỗi fold. Nhật ký lưu chỉ số dòng và median thực tế từng lần fit, được tests đối chiếu với đúng fold Train. Không early stopping; không fit Validation.
Best CV MAE: 19,916.31 USD; thời gian search: 112.35 giây. Best CV parameters:
```json
{
  "model__subsample": 0.8,
  "model__reg_lambda": 1,
  "model__n_estimators": 300,
  "model__max_depth": 3,
  "model__learning_rate": 0.05,
  "model__colsample_bytree": 0.8
}
```
Ba ứng viên CV tốt nhất được so sánh với baseline trên Validation; best CV và model cuối có thể khác nhau.

## 4. GPU training verification

XGBoost 3.2.0; scikit-learn 1.9.1. 103 fit được ghi nhận; thiết bị thực tế: cuda:0.
Booster của từng fit xác nhận cuda:0 và grow_gpu_hist. NVIDIA GeForce GTX 1650, 4 GB, driver 566.07; audit Tuần 1 ghi driver CUDA compatibility 12.7. Không thay driver/CUDA.
Có cảnh báo prediction fallback sang DMatrix do input CPU và Booster CUDA; đây là đường dự đoán, không phải training fallback CPU. Toàn bộ cảnh báo giữ trong week3_fit_audit.json.
VRAM chỉ là snapshot sau training (xem audit), không phải peak hoặc đo riêng process. Không kết luận GPU nhanh hơn CPU vì chưa benchmark đối chứng.
Model cuối: GPU TRAINING SUCCESS, fit 1.59 giây; inference CPU đã kiểm tra kết quả khớp.

## 5. Model comparison

| Model | MAE (USD) | RMSE (USD) | R² | Device |
|---|---:|---:|---:|---|
| Dummy Baseline | 58,741.92 | 81,912.06 | -0.0682 | cpu |
| XGBoost baseline | 17,792.26 | 25,752.38 | 0.8944 | cuda:0 |
| XGBoost tuned CV#1 | 17,762.19 | 25,499.05 | 0.8965 | cuda:0 |
| XGBoost tuned CV#2 | 17,763.78 | 25,815.06 | 0.8939 | cuda:0 |
| XGBoost tuned CV#3 | 17,431.28 | 24,782.91 | 0.9022 | cuda:0 |

## 6. Best model selection

Chọn **XGBoost tuned CV#3** theo MAE Validation thấp nhất; RMSE dùng khi bằng MAE. Không chọn chỉ theo R². Xem model_selection.md và model_comparison.csv.
Artifact artifacts/house_price_model.joblib là Pipeline nhận DataFrame 8 features trực tiếp. Metadata lưu schema, dtypes, dataset/split hashes, seed, số mẫu, IDs nguồn, metrics, versions, tham số, thời gian và checksum.
Training device được lưu riêng; hyperparameters.device=cpu mô tả cấu hình inference sau khi đóng băng. Không fit lại Train+Validation. CLI từ chối ghi đè artifact đã đóng băng; nghiên cứu mới phải dùng bản sao dự án.
SHA-256 artifact: `133f798a66927bf9b6be827bf733c6007a3b0779b82b2f43f6c881f648205c71`.

## 7. Streamlit application

app.py: Tổng quan, Khám phá dữ liệu, Dự đoán giá nhà, Hiệu năng mô hình, Về dự án. Giao diện tiếng Việt, bố cục co giãn và màu xanh nhất quán.

## 8. Các chức năng đã triển khai

- Dashboard: thống kê và biểu đồ chỉ từ Train.
- EDA: 7 biểu đồ Plotly, bộ lọc khu vực/chất lượng/giá/diện tích, xử lý kết quả lọc rỗng.
- Dự đoán: 8 trường bắt buộc, kiểm tra số nguyên/hữu hạn/phạm vi/khu vực; gọi Pipeline.predict, hiển thị USD và cảnh báo ngoài khoảng Train.
- Performance: Dummy, baseline, 3 ứng viên tuned; 4 biểu đồ so sánh/actual-predicted/residual/absolute error trên Validation.
- About: dữ liệu lịch sử Ames, nguồn Kaggle, giới hạn áp dụng; không tuyên bố dự đoán thị trường Việt Nam.

## 9. Test results

Regression Tuần 1–2: 26 PASS / 1 FAIL / 14 SKIP.
Tuần 3: 24 PASS / 0 FAIL / 0 SKIP.
14 SKIP là các tests cũ cần raw/holdout: schema nguồn, tái lập split, dữ liệu không hợp lệ, missing, tamper/freeze, persisted artifacts, chống ghi đè holdout và audit toàn nguồn. Không coi SKIP là PASS; chưa xác minh lại toàn bộ dữ liệu nguồn/holdout. Giữ bằng chứng lịch sử 21 PASS của Tuần 1.
AppTest thực thi cả 5 trang, 7 biểu đồ EDA, lọc rỗng, nút dự đoán và 4 biểu đồ performance. Đây không phải kiểm thử trình duyệt thực về hover/zoom, ảnh chụp hoặc breakpoint mobile.
Kiểm tra máy chủ cục bộ ngày 2026-10-18: GET / và GET /_stcore/health đều trả HTTP 200; health trả ok. Máy chủ kiểm tra được dừng sau xác minh; dùng lệnh bên dưới để mở lại.
JUnit: week2_pytest.xml và week3_pytest.xml. Safe validation runner trả mã lỗi nếu bất kỳ suite nào lỗi.

## 10. Lỗi còn tồn tại và giới hạn

Không có lỗi chặn trong bộ kiểm thử đã chạy. Cảnh báo seaborn/Matplotlib deprecated của notebook cũ được giữ lại; không sửa notebook để loại warning.
Validation đã dùng chọn mô hình nên có selection bias; không coi là kết quả tổng quát cuối. Không có khoảng dự đoán, đánh giá drift, kiểm định ngoài Ames hoặc benchmark GPU/CPU. Giới hạn form là kiểm tra hợp lý đầu vào, không bảo đảm độ chính xác ngoài dữ liệu Train.

## 11. Hướng dẫn chạy trên Windows PowerShell

```powershell
Set-Location D:\KHDL
# Chỉ khi thiếu dependency; môi trường hiện tại đã có đủ:
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Tuning đã hoàn tất; lệnh này từ chối ghi đè artifact đã đóng băng:
.\.venv\Scripts\python.exe -m src.models.tune_model --n-iter 20
# Chạy hai suite dưới policy riêng và cập nhật báo cáo:
.\.venv\Scripts\python.exe -m src.utils.validate_week3
# Chạy website:
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Mở http://localhost:8501. Không chạy src.week1, src.week2 hoặc pytest không có policy trong giai đoạn này.

## 12. Kết luận

WEEK 3 STATUS = **NOT READY**
Mô hình đã đóng băng cho đánh giá Tuần 4. Chưa chạy Tuần 4 và chưa truy cập Test.

## Tham chiếu API

- [XGBoost GPU](https://xgboost.readthedocs.io/en/stable/gpu/)
- [scikit-learn Pipeline](https://scikit-learn.org/stable/modules/compose)

## Files created / modified

Tạo app.py, WEEK_3_README.md, src/data/week3_inputs.py, src/data/app_views.py, src/models/week3_pipeline.py, src/models/tune_model.py, src/models/inference.py, src/utils/week3_report.py, src/utils/validate_week3.py, tests/week3_policy.py và 4 file tests Tuần 3.
Tạo artifacts/house_price_model.joblib, artifacts/model_metadata.json; reports/model_comparison.csv, reports/tuning_results.csv, reports/validation_predictions.csv, reports/model_selection.md, reports/week3_fit_audit.json, reports/week3_pytest.xml, reports/week3_status.json và báo cáo này.
Lệnh nghiệm thu hiện có cập nhật reports/week2_pytest.xml, reports/week2_notebook_audit.json, reports/week2_preservation_check.json, reports/week2_status.json, reports/WEEK_2_REPORT.md; notebook audit tái tạo EDA summary/analysis và 7 PNG Tuần 2. Không sửa notebook/source cũ/checksum gốc.
