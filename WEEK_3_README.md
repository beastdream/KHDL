# HousePrice AI — Tuần 3

Ứng dụng dùng mô hình XGBoost đã đóng băng và Pipeline tiền xử lý đầy đủ. Dữ liệu chỉ gồm Train/Validation, không mở raw hoặc Test.

```powershell
Set-Location D:\KHDL
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Mở http://localhost:8501. Các trang: Tổng quan, Khám phá dữ liệu, Dự đoán giá nhà, Hiệu năng mô hình, Về dự án. Đơn vị diện tích ft², giá USD; dữ liệu lịch sử Ames, Iowa.

Kiểm thử an toàn và lập lại báo cáo:

```powershell
.\.venv\Scripts\python.exe -m src.utils.validate_week3
```

Runner chạy riêng tests Tuần 1–2 với policy cũ và tests Tuần 3 với policy mới. Không chạy pytest không có policy. Policy cũ chặn mọi CSV ngoài Train/Valid, kể cả CSV báo cáo mới; vì vậy không chạy tất cả tests trong cùng process với policy Tuần 2.

Nếu môi trường thiếu dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Tuning đã thực thi. Để tái lập trong bản sao dự án chưa có final artifact:

```powershell
.\.venv\Scripts\python.exe -m src.models.tune_model --n-iter 20
```

Lệnh từ chối ghi đè `artifacts/house_price_model.joblib` đã đóng băng. Không xóa artifact để tuning tiếp trên cùng Validation; tạo phiên bản nghiên cứu riêng nếu cần. Không chạy lại `src.week1` hoặc `src.week2` vì các lệnh đó có phạm vi ghi/đọc khác.

Xem [báo cáo Tuần 3](reports/WEEK_3_REPORT.md), [lựa chọn mô hình](reports/model_selection.md), [metadata](artifacts/model_metadata.json), [audit GPU từng fit](reports/week3_fit_audit.json). Chỉ load joblib tin cậy của dự án; checksum phát hiện thay đổi nhưng không thay thế nguồn artifact đáng tin cậy.
