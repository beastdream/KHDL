# HousePrice AI — Bản nghiệm thu cuối

Tài liệu chạy và demo cuối dự án. `README.md`, `WEEK_2_README.md`, `WEEK_3_README.md` được giữ làm hồ sơ lịch sử; hướng dẫn dưới đây dùng cho bản đã đóng băng và đã đánh giá Test. Không chạy lại tuning để cải thiện kết quả Test.

## Môi trường Windows / VS Code

Đã kiểm chứng Python 3.11.9, XGBoost 3.2.0, scikit-learn 1.9.1, Streamlit 1.66.0. Trong VS Code chọn interpreter `.venv\Scripts\python.exe` và kernel tương ứng khi mở notebook.

```powershell
Set-Location D:\KHDL
# Chỉ tạo venv nếu chưa có:
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-final-lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m src.utils.validate_final
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Mở http://localhost:8501. Không cần activate venv. `requirements.txt` khai báo đủ dependency trực tiếp; môi trường đang cài thỏa mọi constraint. Dòng Streamlit lặp là dư thừa nhưng không xung đột. `requirements-lock.txt` cũ là snapshot Tuần 1, thiếu ba dependency notebook được thêm về sau; giữ nguyên để bảo toàn lịch sử. **Dùng `requirements-final-lock.txt` cho bản cuối**, được xuất từ venv đã chạy kiểm thử, gồm notebook/widget dependencies. `pip check` PASS. Chưa cài lại trên máy Windows sạch; snapshot không bảo đảm mọi wheel tiếp tục có sẵn trên mọi Python/OS.

GPU kiểm chứng: NVIDIA GTX 1650 4096 MiB, driver 566.07, nvidia-smi compatibility CUDA 12.7. Đây là khả năng driver báo cáo, không chứng minh cài CUDA Toolkit riêng. Wheel XGBoost hiện tại đã fit thành công với `tree_method='hist', device='cuda'`; kiểm tra Booster thực tế, không chỉ tên thiết bị. Không cần PyTorch. Không nâng driver/thư viện trong bản nghiệm thu. CPU inference hoạt động dù model được train bằng GPU; CUDA không bắt buộc để demo dự đoán.

## Dữ liệu và artifact

Dataset thật từ [Kaggle House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data). Khi tạo **bản nghiên cứu mới**, tải `train.csv` sau khi chấp nhận điều khoản và đặt ở `data/raw/train.csv`. Bản hiện tại đã có split 70/15/15, seed 42, manifest/checksum trong `reports/data_split.json`; không chia lại.

`test.csv` ở gốc là tập Kaggle không nhãn. `data/processed/test.csv` là holdout nội bộ có SalePrice, đã dùng đúng một lần cho đánh giá cuối sau freeze. Không dùng nó để tuning, EDA hoặc cập nhật preprocessing. Dashboard chỉ đọc Train; trang performance chỉ đọc báo cáo xuất sẵn.

Model chính: `artifacts/house_price_model.joblib`, metadata đi kèm. Pipeline nhận đúng tám features, giá USD/diện tích ft². Chỉ load joblib từ nguồn tin cậy và đúng phiên bản môi trường. Model ~1,16 MB đã bị `.gitignore` loại khỏi Git theo policy hiện có. Khi bàn giao, cung cấp artifact/metadata và baseline trong gói nộp riêng hoặc kho artifact có quyền truy cập; kiểm tra SHA-256. Không cam kết retrain cho cùng byte joblib giữa các môi trường.

## Các lệnh hợp lệ

```powershell
# Kiểm thử an toàn trong bản đã có artifact:
.\.venv\Scripts\python.exe -m src.utils.validate_final
# Đọc/xác minh lại kết quả Test đã xuất, không mở lại holdout:
.\.venv\Scripts\python.exe -m src.models.evaluate_test
# Phân tích lại từ kết quả xuất sẵn:
.\.venv\Scripts\python.exe -m src.models.error_analysis
# Giải thích trên Validation, không fit model:
.\.venv\Scripts\python.exe -m src.models.feature_importance
# Thí nghiệm Train-only, model tạm trong RAM, không ghi artifact chính:
.\.venv\Scripts\python.exe -m src.models.benchmark
# QA Chrome cục bộ; cần Chrome ở đường dẫn cài Windows thông thường:
.\.venv\Scripts\python.exe -m src.utils.verify_final_ui
# Tổng hợp báo cáo từ bằng chứng kiểm chứng:
.\.venv\Scripts\python.exe -m src.utils.final_report
```

Benchmark chạy lại sẽ thay báo cáo timing, không đổi model hay kết quả Test. Để bảo toàn báo cáo nghiệm thu đang nộp, thực hiện thử nghiệm mới trong bản sao và ghi nhận điều kiện đo.

Notebook `notebooks/house_price_eda.ipynb` có sẵn output 7 hình và interactive Plotly/widgets. Có thể mở bằng VS Code/Jupyter; lệnh `.\.venv\Scripts\python.exe -m src.utils.run_eda_notebook` thực thi EDA Train. Các tests notebook còn kiểm tra lại output thật; cảnh báo deprecated Seaborn/Matplotlib được báo cáo.

Chỉ trong **bản sao nghiên cứu mới chưa có model cuối**, quy trình tái tạo cũ là:

```powershell
.\.venv\Scripts\python.exe -m src.week1
.\.venv\Scripts\python.exe -m src.week2
.\.venv\Scripts\python.exe -m src.models.tune_model --n-iter 20
.\.venv\Scripts\python.exe -m src.utils.validate_week3
.\.venv\Scripts\python.exe -m src.utils.final_precheck
.\.venv\Scripts\python.exe -m src.models.evaluate_test
```

Các lệnh Tuần 1–2 có phạm vi ghi báo cáo riêng; không chạy trên hồ sơ nghiệm thu hiện tại. Tuning từ chối ghi đè artifact cuối. `final_precheck` tạo freeze một lần trước đánh giá; các lần sau chỉ kiểm tra freeze cũ. Evaluator từ chối tự động đánh giá lại nếu phát hiện lần chạy dở dang; giữ bằng chứng để điều tra, không xóa marker nhằm chạy lại tùy ý.

## Demo trước giảng viên (5–7 phút)

1. Mở `reports/FINAL_ACCEPTANCE_REPORT.md`, trình bày split 1022/219/219, nguyên tắc chống leakage và model checksum.
2. Tổng quan: chỉ rõ 1022 bản ghi Train, 8 features, giá trung vị/giá trung bình.
3. Khám phá: thử khu vực NAmes, quality 4–6; thay khoảng giá/diện tích và giải thích 7 biểu đồ; thử lọc rỗng.
4. Dự đoán: nhập 1800 ft², CollgCr, 4 phòng ngủ, 2 phòng tắm, garage 2 xe, năm 2005, quality 7, tầng hầm 1100 ft²; nhấn Dự đoán và giải thích giá USD từ model thật.
5. Hiệu năng: phân biệt Validation và Test; so sánh Dummy/XGBoost, chỉ ra residual/error theo giá và importance trên Validation.
6. Trình bày GPU hoạt động nhưng chậm hơn CPU với dữ liệu nhỏ; inference CPU chủ động. Nêu giới hạn Ames lịch sử, nhóm nhà đắt ít mẫu và chưa có khoảng dự đoán.
7. Mở `reports/final_test_results.md` để giải thích PASS/FAIL/SKIP, không coi 14 SKIP là PASS.

Không cần chạy lại Test/training trong buổi demo. Không commit venv/cache, dữ liệu theo điều khoản Kaggle hoặc secrets. `.gitignore` đã loại dữ liệu/artifact/cache; thư mục hiện tại không có repository Git và không có commit/push trong lần hoàn thiện này.

## Hồ sơ nộp

- `reports/FINAL_PROJECT_REPORT.md`: báo cáo học thuật tám chương.
- `reports/FINAL_ACCEPTANCE_REPORT.md`: checklist và giới hạn nghiệm thu.
- `reports/FINAL_PRECHECK.md`, `final_model_freeze.json`, `test_metrics.json`, `test_evaluation.md`: provenance và đánh giá.
- `reports/error_analysis.md`, `feature_importance.md`, `gpu_benchmark.md`, `figures/final/`: phân tích và hình thật.
- `reports/final_pytest.xml`, `final_test_results.md`, `final_streamlit_verification.json`, `final_ui_browser_verification.json`: bằng chứng QA.
