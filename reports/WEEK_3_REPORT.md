# Báo cáo Tuần 3 — HousePrice AI

Ngày lập: 2026-10-19T13:35:38.033246+07:00

## 1. Tóm tắt công việc

Tiếp tục dự án hiện có; tuning XGBoost, chọn mô hình trên Validation, lưu Pipeline hoàn chỉnh và triển khai Streamlit tiếng Việt 5 trang.

## 2. Trạng thái kế thừa Tuần 1–2

Gate Tuần 2: READY; core: PASS. Notebook: PASS; preservation: PASS.
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

Regression Tuần 1–2: 27 PASS / 0 FAIL / 14 SKIP.
Tuần 3: 27 PASS / 0 FAIL / 0 SKIP.
14 SKIP là các tests cũ cần raw/holdout: schema nguồn, tái lập split, dữ liệu không hợp lệ, missing, tamper/freeze, persisted artifacts, chống ghi đè holdout và audit toàn nguồn. Không coi SKIP là PASS; chưa xác minh lại toàn bộ dữ liệu nguồn/holdout. Giữ bằng chứng lịch sử 21 PASS của Tuần 1.
AppTest thực thi cả 5 trang, 7 biểu đồ EDA, lọc rỗng, nút dự đoán và 4 biểu đồ performance. Đây không phải kiểm thử trình duyệt thực về hover/zoom, ảnh chụp hoặc breakpoint mobile.
Kiểm tra máy chủ cục bộ ngày 2026-10-18: GET / và GET /_stcore/health đều trả HTTP 200; health trả ok. Máy chủ kiểm tra được dừng sau xác minh; dùng lệnh bên dưới để mở lại.
JUnit: week2_pytest.xml và week3_pytest.xml. Safe validation runner trả mã lỗi nếu bất kỳ suite nào lỗi.

## 10. Lỗi còn tồn tại và giới hạn

Còn điều kiện nghiệm thu chưa đạt; xem JUnit và trạng thái gate, không tuyên bố PASS. Cảnh báo seaborn/Matplotlib deprecated của notebook cũ được giữ lại; không sửa notebook để loại warning.
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

## Điều tra preservation và GPU/CPU — 2026-10-18

### Nguyên nhân và thời điểm

`src/week1.py` **không thay đổi so với baseline**: SHA-256 hiện tại và gốc đều là `9ff7049430ee114f9e2c2b0eec4909a49a4b760203c4da8bd56085f16159180c`. File có BOM và kiểu xuống dòng hiện hữu; không chuẩn hóa chúng vì chính các byte này khớp baseline. Không có diff nội dung để quy cho Tuần 2 hay Tuần 3. Không sửa file này.

JUnit thất bại in `result['entries']` là toàn bộ danh sách audit, bắt đầu bằng src/week1.py và bị rút gọn. Entry đầu thực tế là UNCHANGED; đây không phải bằng chứng nó lệch checksum. Entry lỗi duy nhất trong audit là **requirements.txt**.

File requirements trước sửa có 243 byte, kết thúc bằng b'\n\n\n\n'. Bản đã duyệt có 240 byte, kết thúc bằng một LF. Diff chính xác: thêm 3 dòng trống cuối file, không đổi bất kỳ dependency hay constraint nào.

- Hash trước sửa: `7b9a3c61708810ef3fe2e467905c746b1a1c4b43e1df03f15bbeca4b42328278`.
- Hash sau loại đúng 3 LF: `ac22392cc5f19ea3de826ba902d9d4b9ccbc12288bb6e555856a423c4260f220`, khớp **approved_sha256** đã lưu từ Tuần 2.
- Original SHA-256 Tuần 1 của requirements vẫn là `b1fd0913615710a0b56bbed45a23fad635cbed473bc78e5d943c55cc42788ada`; không thay thế bằng hash khác.

Bằng chứng của lần triển khai Tuần 3 trước đó có 240 byte và preservation PASS; mtime hiện tại trước sửa là 2026-10-18 11:57 +07:00. Điều này phù hợp một lần lưu sau triển khai ban đầu. Không có thư mục Git: git status/log đều báo “not a git repository”; không thể xác định người, công cụ hay thời điểm chính xác tạo diff chỉ từ mtime. Không suy đoán rằng training hoặc Streamlit đã sửa file.

### Phân loại, ảnh hưởng và khắc phục

Ba LF là thay đổi định dạng không có tác dụng dependency, không cần thiết cho chức năng và chưa thuộc phiên bản được duyệt. Không ảnh hưởng logic Dataset Audit, Data Split, Data Guard hay nội dung Test; ảnh hưởng trực tiếp đến kiểm tra tính toàn vẹn byte và khiến preservation NOT READY. Test phát hiện đúng thay đổi chưa được duyệt.

Đã lưu bản trước sửa, JUnit lỗi, audit, baseline/approvals và hashes trong `reports/week3_repair_evidence/`. Sau khi chứng minh bỏ đúng ba byte cho hash đã duyệt và danh sách token dependency giống hệt, khôi phục file requirements về phiên bản này. Không gỡ dependency notebook, không thay hash lịch sử/approval, không xóa hoặc skip preservation test, không sửa src/week1.py hoặc Data Guard.

Với thay đổi chức năng hợp lệ trong tương lai: lưu diff, lý do, căn cứ ủy quyền và approved_sha256 mới theo cơ chế ghim phiên bản đang có; giữ nguyên original_sha256. Không cần thêm ngoại lệ cho lần sửa này.

Phát hiện thêm báo cáo Tuần 3 dùng câu cố định “preservation PASS” và “không có lỗi chặn” ngay cả khi gate NOT READY. Đã sửa `src/utils/week3_report.py` lấy trạng thái test từ JUnit và điều kiện nghiệm thu; đồng thời tự nhúng ghi chú điều tra này khi chạy lại để không mất kết luận.

### Xác minh GPU training và CPU inference

Audit môi trường Tuần 1 ghi GTX 1650 / GPU READY, nhưng bằng chứng training là 103 bản ghi fit Tuần 3: tất cả success, device=cuda:0 và updater grow_gpu_hist, không có training failure. Artifact cuối còn chứa fit_evidence_ của ứng viên CV#3; đối chiếu với audit và metadata. Metadata: training_device=cuda:0, training_status=GPU TRAINING SUCCESS, inference_device=cpu.

CPU inference là cấu hình **chủ động**: tune_model.py gọi set_params(device='cpu') trước lưu và so sánh lại metrics; inference.py tiếp tục đặt CPU khi load artifact tin cậy. Việc này phục vụ ứng dụng CPU không phụ thuộc GPU cho mỗi yêu cầu. Không thay cây đã học và không chứng minh training dùng CPU. Cảnh báo DMatrix khi prediction nhận dữ liệu CPU trong tuning không phải fallback training.

Model vẫn là XGBoost tuned CV#3, checksum `133f798a66927bf9b6be827bf733c6007a3b0779b82b2f43f6c881f648205c71`. Kiểm tra lại 219 dự đoán Validation với CSV đã lưu và metrics metadata; không chạy lại tuning hay ghi lại artifact. Bộ validate có test fit mô hình nhỏ độc lập 3 cây trên 30 dòng Train trong bộ nhớ; đó là unit test cũ, không retrain artifact cuối.

Kết quả đối chiếu lưu tại `week3_repair_evidence/verification.json`: 219 dự đoán CPU **khớp chính xác khi chuyển CSV về float32** như output model. Đọc CSV thành float64 cho chênh lệch biểu diễn thập phân tối đa 0,025 USD. Metrics CPU tính lại: MAE 17.431,26953125; RMSE 24.782,89087253543; R² 0,902214765548706. So với metrics được ghi khi chọn ứng viên, chênh MAE −0,005859375 USD, RMSE −0,014203343 USD, R² +1,19209e−7. Chúng nằm trong rtol=1e−6 đã có từ pipeline/tests; không sửa tolerance để qua kiểm tra, không tuyên bố metrics CPU/GPU bằng nhau từng bit. Metadata lịch sử được giữ nguyên.

Lần chạy `python.exe -m src.utils.validate_week3` sau khắc phục trả **exit code 0**: suite Tuần 1–2 **27 PASS / 0 FAIL / 14 SKIP**, suite Tuần 3 **24 PASS / 0 FAIL / 0 SKIP**; tổng **51 PASS / 0 FAIL / 14 SKIP**. Test preservation chạy và PASS; không bị bỏ qua. 14 SKIP vẫn là phạm vi raw/holdout đã được chấp nhận, không phải kết quả PASS. Bằng chứng sau kiểm tra xác nhận cả artifact cuối, metadata, baseline, manifest, notebook, source audit/split, hai Data Guard, preservation module và preservation test đều giữ nguyên hash trước sửa.

Kết quả pytest hiện tại nằm ở mục 9 và trạng thái nghiệm thu ở mục 12, được sinh từ lần validate mới. AppTest kiểm tra nút dự đoán và đủ 5 trang. Train/Valid được xác minh bằng hash và IDs; manifest và hash Test đã lưu được giữ nguyên. Không mở Test để rehash nên không tuyên bố đã xác minh lại byte nội dung holdout; split 1022/219/219 (70/15/15), seed 42 được xác nhận qua manifest gốc và các tập được phép đọc.

### File thay đổi trong lần khắc phục

- requirements.txt: khôi phục đúng 3 LF thừa.
- src/utils/week3_report.py: sửa mô tả trạng thái và giữ ghi chú RCA khi tái sinh báo cáo.
- reports/week3_repair_notes.md, reports/week3_repair_evidence/: ghi chú, snapshot trước sửa và bằng chứng đối chiếu.
- Các báo cáo JUnit, gate, preservation, notebook audit, báo cáo Tuần 2–3 và EDA Train được tái sinh bởi validate_week3. Source Tuần 1, test preservation, original hashes, approvals, model artifact, metadata và notebook không sửa.

