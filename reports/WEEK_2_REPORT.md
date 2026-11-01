# Báo cáo Tuần 2 — House Price Prediction

Ngày lập: 2026-10-19T13:35:36.693711+07:00

## 1. Phạm vi và bảo toàn Tuần 1

Chỉ sử dụng 1.022 mẫu Train và 219 mẫu Validation. Không đọc raw hoặc test.csv, kể cả để hash; không chạy lại pipeline Tuần 1.
Preservation: PASS. 25 tệp khớp baseline; requirements.txt là ngoại lệ A đã xác minh và ghim hash. Không sửa checksum gốc.
Lớp audit hook chặn mở CSV ngoài hai tập được phép và chặn ghi lên hai tập này. Các phép thử truy cập bị chặn trước khi đọc byte dữ liệu.

## 2. EDA trên Train

Đủ 7 biểu đồ PNG. SalePrice: trung vị 165,000 USD, skewness 1.762.
Tương quan Pearson với giá: OverallQual=0.785, GrLivArea=0.689, GarageCars=0.643, TotalBsmtSF=0.590, FullBath=0.549, YearBuilt=0.512, BedroomAbvGr=0.160.
Có 41 outlier giá theo 1.5 IQR và 3 mẫu diện tích >4.000 sq ft; giữ nguyên tất cả.
[EDA tiếng Việt](eda_analysis.md). Notebook có 10 code cells chạy lại không lỗi và 16 display_data lưu sẵn. Giữ nguyên mọi cell và outputs.

## 3. Preprocessing và huấn luyện

ColumnTransformer: 7 biến số dùng SimpleImputer(median); Neighborhood dùng SimpleImputer(most_frequent) rồi OneHotEncoder(handle_unknown=ignore).
Chỉ fit Train. Validation chỉ transform/predict; không eval_set, early stopping, tuning hoặc lựa chọn mô hình dựa trên Test. Giữ đúng 8 features; loại Id và SalePrice khỏi X.
DummyRegressor(median) fit Train. XGBoost dùng đúng cấu hình khởi đầu:

```json
{
  "objective": "reg:squarederror",
  "n_estimators": 300,
  "max_depth": 5,
  "learning_rate": 0.05,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "tree_method": "hist",
  "device": "cuda",
  "random_state": 42
}
```

## 4. Kết quả Validation

| Model | MAE | RMSE | R2 | Device |
|---|---:|---:|---:|---|
| Dummy Baseline | 58,741.92 | 81,912.06 | -0.0682 | CPU |
| XGBoost | 17,792.26 | 25,752.38 | 0.8944 | cuda:0 |

MAE/RMSE: USD. XGBoost giảm RMSE 68.56% so với baseline. R² baseline âm nghĩa là baseline trung vị Train kém hơn dự đoán trung bình Validation theo định nghĩa R², không phải lỗi tính metric.
[Metrics JSON](validation_metrics.json), [bảng metrics](validation_metrics.md). Không báo cáo metric Test.

## 5. GPU và thời gian

GPU READY
Thiết bị Booster sau fit: cuda:0. Đã xác nhận device CUDA và updater grow_gpu_hist sau huấn luyện trên dữ liệu nhà thực.
Thời gian fit XGBoost: 2.2512 giây; preprocessing: 0.1289 giây (wall-clock, không phải benchmark tốc độ GPU so với CPU).
```text
name, memory.total [MiB], driver_version
NVIDIA GeForce GTX 1650, 4096 MiB, 566.07

```
Preprocessing chạy trên CPU; dự đoán dùng Booster.predict(DMatrix) với Booster đã xác nhận CUDA. Không ngụ ý mọi thao tác đều chạy trên GPU.
Các lần thử, warnings, cấu hình Booster, phiên bản, nvidia-smi, thông số và hash artifact: artifacts/xgboost_baseline.metadata.json.

## 6. Artifact

`artifacts/xgboost_baseline.joblib` chứa preprocessor đã fit, XGBRegressor đã fit, Dummy baseline, schema và metadata. Đã load lại và so sánh prediction khớp tuyệt đối.
Dùng src.models.preprocessing.predict_artifact với DataFrame đúng 8 features. Hàm từ chối thiếu/thừa cột và giá trị sai kiểu; cho phép missing và category chưa thấy.

## 7. Kiểm thử

Kết quả hiện tại: 27 PASS / 0 FAIL / 14 SKIP.
Toàn bộ test Tuần 2 và 7 test Tuần 1 an toàn đã chạy. 14 test cũ SKIP theo lựa chọn đã xác nhận của người dùng; không coi SKIP là PASS.
Báo cáo Tuần 1 lưu kết quả lịch sử 21 PASS và được giữ nguyên. Kết quả lịch sử không chứng minh toàn bộ hồi quy vừa chạy lại.

| Test | Trạng thái |
|---|---|
| tests.test_week1.test_csv_schema_and_target | SKIP |
| tests.test_week1.test_split_counts_disjoint_reproducible[20] | SKIP |
| tests.test_week1.test_split_counts_disjoint_reproducible[101] | SKIP |
| tests.test_week1.test_split_counts_disjoint_reproducible[1460] | SKIP |
| tests.test_week1.test_invalid_data_rejected[target] | SKIP |
| tests.test_week1.test_invalid_data_rejected[id] | SKIP |
| tests.test_week1.test_invalid_data_rejected[area] | SKIP |
| tests.test_week1.test_invalid_data_rejected[missing_column] | SKIP |
| tests.test_week1.test_invalid_data_rejected[duplicate_content] | SKIP |
| tests.test_week1.test_missing_predictors_preserved | SKIP |
| tests.test_week1.test_freeze_and_tamper_detection | SKIP |
| tests.test_week1.test_persisted_artifacts | SKIP |
| tests.test_week1.test_gpu_configuration_detection[cpu-grow_quantile_histmaker-False] | PASS |
| tests.test_week1.test_gpu_configuration_detection[cuda:0-grow_gpu_hist-True] | PASS |
| tests.test_week1.test_gpu_configuration_detection[cuda:0-grow_quantile_histmaker-False] | PASS |
| tests.test_week1.test_gpu_silent_cpu_fallback_is_failure | PASS |
| tests.test_week1.test_gpu_exception_reported | PASS |
| tests.test_week1.test_actual_gpu_audit_truthful | PASS |
| tests.test_week1.test_missing_source_instructions | PASS |
| tests.test_week1.test_untracked_holdout_never_overwritten | SKIP |
| tests.test_week1.test_audit_scope_and_no_removal | SKIP |
| tests.test_model.test_artifact_fit_and_schema | PASS |
| tests.test_model.test_predictions_and_reload | PASS |
| tests.test_model.test_bad_schema_rejected[missing] | PASS |
| tests.test_model.test_bad_schema_rejected[target] | PASS |
| tests.test_model.test_bad_schema_rejected[id] | PASS |
| tests.test_model.test_bad_schema_rejected[string] | PASS |
| tests.test_model.test_bad_schema_rejected[infinity] | PASS |
| tests.test_model.test_bad_schema_rejected[category] | PASS |
| tests.test_model.test_missing_and_unseen_categories | PASS |
| tests.test_model.test_preprocessor_train_only_and_transform_does_not_fit | PASS |
| tests.test_model.test_preprocessor_imputes_from_fit_rows_only | PASS |
| tests.test_model.test_metrics_independently_recomputed | PASS |
| tests.test_model.test_input_access_restriction | PASS |
| tests.test_model.test_eda_train_only_and_notebook_executed | PASS |
| tests.test_model.test_week1_preservation | PASS |
| tests.test_gpu_training.test_actual_training_gpu_evidence | PASS |
| tests.test_gpu_training.test_gpu_failure_explicit_cpu_fallback[exception] | PASS |
| tests.test_gpu_training.test_gpu_failure_explicit_cpu_fallback[silent_cpu] | PASS |
| tests.test_preservation_policy.test_preservation_approved_change_is_version_pinned | PASS |
| tests.test_preservation_policy.test_preservation_forbidden_paths_never_hashed | PASS |

[JUnit](week2_pytest.xml).

## 8. Hạn chế và việc còn lại

Một lần chia holdout nhỏ và chỉ một cấu hình; chưa cross-validation, tuning, khoảng tin cậy, phân tích công bằng theo nhóm, kiểm tra drift hoặc đánh giá ngoài Ames.
Outlier được giữ lại; RMSE nhạy với nhà giá cao. Correlation không chứng minh quan hệ nhân quả. Chỉ dùng 8 features nên bỏ qua nhiều đặc điểm có thể quan trọng.
14 test Tuần 1 cần raw/holdout không chạy lại theo lựa chọn đã xác nhận. Giới hạn này không được che bằng kết quả lịch sử.

## 9. Kết luận

Yêu cầu cốt lõi EDA/preprocessing/model/validation/artifact: PASS.
GPU: GPU READY.
Toàn bộ mục tiêu: PASS — theo phạm vi test an toàn đã được chấp nhận.
Chưa chuyển sang Tuần 3.

## Tài liệu API

- [XGBoost GPU](https://xgboost.readthedocs.io/en/stable/gpu/index.html)
- [scikit-learn ColumnTransformer](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html)

## 10. Khắc phục regression notebook và preservation

**Nguyên nhân lỗi notebook:** test giả định chính xác 7 `display_data`. Notebook thực tế có 16: 7 ảnh PNG bắt buộc, 4 biểu đồ Plotly, 1 Markdown, 3 bảng HTML và 1 widget. Một output có thể chứa nhiều MIME; display_data không đồng nghĩa với một biểu đồ bắt buộc.

Đã sửa `test_eda_train_only_and_notebook_executed` để chấp nhận tối thiểu 7 outputs, đồng thời kiểm tra đủ từng nội dung: histogram SalePrice; scatter GrLivArea/SalePrice; boxplot OverallQual/SalePrice; trung vị SalePrice theo Neighborhood; Pearson heatmap; scatter YearBuilt/SalePrice; boxplot BedroomAbvGr/SalePrice. Audit kiểm tra loại biểu đồ, cột X/Y và giá trị truyền vào hàm vẽ so với dữ liệu Train; kiểm tra notebook đọc/hiển thị đủ 7 PNG. Không chỉ kiểm tra tên file hay số outputs.

Notebook gốc không có error output. Cả 10 code cells được chạy lại trong bộ nhớ dưới Data Guard; callback widget cũng được kiểm tra. Cell heatmap nâng cao trước đó lưu bảng và cảnh báo backend Agg, không có ảnh riêng; heatmap bắt buộc vẫn có trong 7 PNG đầu tiên. Audit render ảnh bổ sung trong bộ nhớ, không sửa notebook. Widget được kiểm tra ở mức thực thi callback, không tuyên bố đã kiểm thử thao tác chuột trong frontend VS Code.

Kiểm tra source chỉ thấy đọc processed/train.csv. Lần thực thi mới được Data Guard bảo vệ. Không thể chứng minh toàn bộ lịch sử truy cập của kernel cũ chỉ từ notebook lưu sẵn.

**Nguyên nhân lỗi preservation:** chỉ `requirements.txt` thay đổi, nhưng test cũ không có cơ chế chấp nhận thay đổi hợp lệ. File thêm ipykernel, nbformat, ipywidgets và một dòng Streamlit trùng constraint. SHA-256 của 10 dòng đầu khớp chính xác hash requirements.txt gốc: không có constraint cũ bị sửa.

- **A — hợp lệ:** requirements.txt hỗ trợ các cell notebook tương tác. Giữ nguyên nội dung người dùng. `week2_preservation_approvals.json` ghi hash gốc, hash phiên bản đã duyệt, lý do, bằng chứng và căn cứ ủy quyền trong yêu cầu sửa lỗi này. Không ghi lại baseline. Bất kỳ thay đổi tiếp theo ngoài hash đã duyệt vẫn FAIL.
- **B — không mong muốn:** không phát hiện trong 26 đường dẫn theo dõi. Nếu có thay đổi mới: giữ nguyên file, so sánh diff với backup và đề xuất phục hồi sang bản sao để review, không tự ghi đè.
- **C — chưa thể kiểm tra:** raw và Test 15% là **NOT VERIFIED**; không mở hoặc hash. Chúng không nằm trong 26 đường dẫn snapshot hiện tại. Helper mới kiểm tra chính sách trước khi mở file; nếu baseline sau này chứa đường dẫn bị cấm, nó trả NOT VERIFIED, không né Data Guard.

**Các file sửa:** `tests/test_model.py`, `src/utils/week2_report.py`. **Các file thêm:** `src/utils/notebook_audit.py`, `src/utils/preservation.py`, `tests/test_preservation_policy.py`, ngoại lệ đã duyệt và các báo cáo kiểm chứng. Các báo cáo EDA Train được tạo lại khi kiểm tra nội dung biểu đồ.

`test_week1_preservation` dùng audit với ngoại lệ ghim hash chính xác thay vì yêu cầu mọi tệp bất biến. Hai test policy mới kiểm tra thay đổi sau phê duyệt vẫn thất bại và đường dẫn bị cấm không được truyền vào hàm đọc checksum.

Regression có 4 warnings deprecation từ Seaborn/Matplotlib (`vert`, `set_bad`), không phải lỗi thực thi notebook. Không nâng thư viện hoặc đổi model để che cảnh báo. Tổng PASS/FAIL/SKIP và điều kiện nghiệm thu nằm ở mục 7–9.

Không sửa hoặc huấn luyện lại model; không đổi split 70/15/15. Notebook, requirements.txt, Data Guard, artifact, metadata model, manifest chia tập và checksum baseline được đối chiếu trước/sau và giữ nguyên byte. Chưa thực hiện Tuần 3.


### Danh sách preservation

| Đường dẫn | Phân loại | Trạng thái |
|---|---|---|
| src/week1.py | UNCHANGED | VERIFIED |
| src/__init__.py | UNCHANGED | VERIFIED |
| src/data/audit_dataset.py | UNCHANGED | VERIFIED |
| src/data/schema.py | UNCHANGED | VERIFIED |
| src/data/split_dataset.py | UNCHANGED | VERIFIED |
| src/data/__init__.py | UNCHANGED | VERIFIED |
| src/models/__init__.py | UNCHANGED | VERIFIED |
| src/utils/check_gpu.py | UNCHANGED | VERIFIED |
| src/utils/paths.py | UNCHANGED | VERIFIED |
| src/utils/__init__.py | UNCHANGED | VERIFIED |
| tests/test_week1.py | UNCHANGED | VERIFIED |
| reports/dataset_audit.json | UNCHANGED | VERIFIED |
| reports/dataset_audit.md | UNCHANGED | VERIFIED |
| reports/data_split.json | UNCHANGED | VERIFIED |
| reports/gpu_audit.json | UNCHANGED | VERIFIED |
| reports/gpu_audit.md | UNCHANGED | VERIFIED |
| reports/pytest.xml | UNCHANGED | VERIFIED |
| reports/pytest_results.txt | UNCHANGED | VERIFIED |
| reports/WEEK_1_REPORT.md | UNCHANGED | VERIFIED |
| reports/figures/train_distribution.png | UNCHANGED | VERIFIED |
| notebooks/.gitkeep | UNCHANGED | VERIFIED |
| README.md | UNCHANGED | VERIFIED |
| requirements.txt | A | APPROVED CHANGE |
| requirements-lock.txt | UNCHANGED | VERIFIED |
| pytest.ini | UNCHANGED | VERIFIED |
| .gitignore | UNCHANGED | VERIFIED |

Chi tiết: [notebook audit](week2_notebook_audit.json), [preservation audit](week2_preservation_check.json), [ngoại lệ đã duyệt](week2_preservation_approvals.json).