# House Price Prediction — Tuần 1

Dự án Python 3.11 trên Windows: kiểm tra dữ liệu Ames/Kaggle, chia tập và xác minh XGBoost CUDA. Chưa huấn luyện mô hình dự đoán giá nhà.

## Chạy bằng PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.week1
```

Không cần activate venv hoặc cài PyTorch. `requirements-lock.txt` ghi lại phiên bản đã kiểm chứng; có thể cài bằng `-r requirements-lock.txt` để tái lập môi trường.

## Dữ liệu thật

File sử dụng: `data/raw/train.csv`, sao chép nguyên byte từ `train.csv` người dùng cung cấp. Không sinh dữ liệu nhà giả. Nếu thiếu: đăng nhập [Kaggle House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data), chấp nhận điều khoản cuộc thi, tải và giải nén `train.csv` vào `data/raw/`. Dataset không được commit.

`test.csv` gốc là tập Kaggle không có SalePrice, không dùng trong Tuần 1. Tập holdout nội bộ là `data/processed/test.csv` có nhãn, khóa đến Tuần 4.

## Schema và nguyên tắc

8 features: GrLivArea, Neighborhood, BedroomAbvGr, FullBath, GarageCars, YearBuilt, OverallQual, TotalBsmtSF. Target: SalePrice. `Id` chỉ dùng truy vết, không thuộc X. CSV sau chia giữ các cột gốc; dùng `src.data.schema.xy` để lấy đúng X/y.

Hai lần train_test_split, seed 42: 70/15/15 với sai số làm tròn. Không impute, encode hay scale trước chia. Missing predictor được giữ nguyên để xử lý bằng pipeline chỉ fit train ở tuần sau. Giá không dương, ID trùng, schema sai hoặc nội dung bản ghi trùng sẽ chặn split, không âm thầm xóa. Outlier chỉ được gắn cờ để xem xét.

Audit chất lượng kiểm tra toàn nguồn (kiểu, missing, duplicate, hợp lệ). Để tuân thủ cấm EDA trên test, phân phối target, thống kê mô tả và biểu đồ chỉ tính trên train sau khi chia. Không thống kê phân phối valid/test hoặc tuning. Metadata test chỉ phục vụ kiểm tra định danh và toàn vẹn.

Manifest `reports/data_split.json` giữ seed, source hash, index dòng 0-based, Id, số mẫu, tỷ lệ và SHA-256 từng CSV. Chạy lại không ghi đè các tập hiện có; nếu hash khác sẽ dừng. Đây là kiểm tra tính toàn vẹn, không phải cơ chế bảo mật chống sửa cả manifest. Khi nguồn thay đổi, tạo phiên bản dataset/thư mục dự án mới; không ghi đè holdout cũ.

## Lệnh riêng

```powershell
.\.venv\Scripts\python.exe -m src.data.audit_dataset
.\.venv\Scripts\python.exe -m src.data.split_dataset
.\.venv\Scripts\python.exe -m src.utils.check_gpu
.\.venv\Scripts\python.exe -m pytest -q
```

Chạy pipeline trước pytest tích hợp để tạo artifacts. Tests sử dụng dữ liệu thật và bản sao nhỏ trong thư mục tạm; không tạo dataset thay thế. GPU smoke dùng ma trận ngẫu nhiên nhỏ độc lập đúng yêu cầu kiểm tra môi trường, không dùng giá nhà.

## Kết quả và cấu trúc

- `src/data/`: schema, audit, split; `src/utils/`: GPU và đường dẫn tuyệt đối dựa trên project root.
- `src/models/`, `notebooks/`, `artifacts/`: dành cho các tuần tiếp theo.
- `reports/WEEK_1_REPORT.md`: kết luận PASS/NOT READY của dữ liệu và tests, GPU READY/GPU NOT READY riêng.
- `reports/dataset_audit.{md,json}`: audit và thống kê train; `reports/figures/`: biểu đồ train.
- `reports/gpu_audit.{md,json}`: nvidia-smi, driver, VRAM, CUDA compatibility, phiên bản Python/XGBoost, warnings/lỗi đầy đủ, cấu hình Booster.
- `reports/pytest_results.txt`, `reports/pytest.xml`: bằng chứng kiểm thử.

GPU chỉ READY khi fit thành công và Booster xác nhận CUDA cùng grow_gpu_hist. CPU fallback được báo cáo riêng. Khi lỗi, xem remediation trong báo cáo: kiểm tra driver, wheel XGBoost có CUDA và tính tương thích phần cứng. [Tài liệu chính thức XGBoost GPU](https://xgboost.readthedocs.io/en/stable/gpu/index.html).
