# Chạy Tuần 2

Giữ nguyên README và code/dữ liệu Tuần 1. Dùng môi trường .venv đã có.

```powershell
.\.venv\Scripts\python.exe -m src.week2
```

Pipeline chỉ mở data/processed/train.csv và valid.csv. Nó tạo notebook đã thực thi, 7 PNG, EDA tiếng Việt, mô hình, metrics và báo cáo. Không đọc test.csv hoặc data/raw/train.csv. Không chạy `src.week1` hoặc pytest không có policy trong Tuần 2.

Lệnh riêng:

```powershell
.\.venv\Scripts\python.exe -m src.utils.run_eda_notebook
.\.venv\Scripts\python.exe -m src.models.train_model
.\.venv\Scripts\python.exe -m src.models.evaluate_validation
.\.venv\Scripts\python.exe -m pytest -p tests.week2_policy -q --junitxml=reports/week2_pytest.xml
.\.venv\Scripts\python.exe -m src.utils.week2_report
```

Notebook chứa Python thuần, chạy lại được bằng VS Code/Jupyter với kernel .venv. Script run_eda_notebook thực thi các code cells và nhúng stdout/7 ảnh mà không cần cài thêm kernel server.

```python
import joblib
from src.data.week2_inputs import load_split
from src.data.schema import FEATURES
from src.models.preprocessing import predict_artifact
artifact = joblib.load('artifacts/xgboost_baseline.joblib')
pred = predict_artifact(artifact, load_split('valid')[FEATURES])
```

Artifact gồm preprocessor, model, baseline, feature_schema, metadata. File metadata JSON kèm hash SHA-256, thiết bị xác minh sau fit, thời gian, phiên bản, tham số, Id nguồn và hashes train/valid. Chỉ load artifact do bạn tin cậy.

Các test Tuần 1 đọc raw/holdout được policy đánh SKIP, giữ nguyên mã test và bằng chứng 21 PASS lịch sử. SKIP không đồng nghĩa PASS. Exit code pipeline là 1 và tổng trạng thái NOT READY nếu yêu cầu chạy lại toàn bộ test Tuần 1 chưa xác minh được dưới lệnh cấm hiện tại; EDA/model/validation có trạng thái riêng.

Xem reports/WEEK_2_REPORT.md và reports/validation_metrics.md. Chưa thực hiện Tuần 3.
