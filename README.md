# HousePrice AI — Hệ Thống Dự Đoán Giá Nhà Ames Housing

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-FF6600?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.1-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66.0-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Dự án đồ án Khoa học Dữ liệu (KHDL) xây dựng giải pháp học máy hoàn chỉnh từ phân tích dữ liệu khám phá (EDA), tiền xử lý, huấn luyện mô hình dự đoán giá nhà ở (Ames Housing Dataset) bằng thuật toán **XGBoost Regressor**, đến triển khai ứng dụng web tương tác trực quan với **Streamlit**.

---



## 1. Giới thiệu dự án

### 1.1. Mục tiêu
Dự đoán giá bán thực tế (`SalePrice`, đơn vị USD) của các bất động sản tại thành phố Ames, bang Iowa (Mỹ) với sai số trung bình tuyệt đối (MAE) thấp, hỗ trợ người mua, người bán và môi giới bất động sản ước tính giá trị căn nhà nhanh chóng dựa trên các thông số kiến trúc và vị trí cốt lõi.

### 1.2. Bộ dữ liệu Ames Housing
Bộ dữ liệu chuẩn từ cuộc thi Kaggle [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques). Dự án chọn lọc **8 đặc trưng quan trọng nhất** mang tính thực tế cao khi người dùng nhập liệu:
- **GrLivArea** *(Diện tích sinh hoạt trên mặt đất - sq ft)*
- **Neighborhood** *(Khu dân cư / Vị trí địa lý)*
- **OverallQual** *(Đánh giá chất lượng vật liệu & hoàn thiện tổng thể, thang điểm 1–10)*
- **YearBuilt** *(Năm xây dựng)*
- **TotalBsmtSF** *(Tổng diện tích tầng hầm - sq ft)*
- **FullBath** *(Số phòng tắm đầy đủ trên mặt đất)*
- **BedroomAbvGr** *(Số phòng ngủ trên mặt đất)*
- **GarageCars** *(Sức chứa garage theo số lượng xe)*

### 1.3. Công nghệ và kỹ thuật
- **Nền tảng & Ngôn ngữ**: Python 3.11 trên hệ điều hành Windows.
- **Xử lý dữ liệu**: Pandas 2.3.3, NumPy 2.4.6.
- **Học máy & Pipeline**: Scikit-learn 1.9.1 (`ColumnTransformer`, `SimpleImputer`, `OneHotEncoder`), XGBoost 3.2.0 (hỗ trợ tăng tốc cả CPU và GPU CUDA).
- **Trực quan hóa**: Matplotlib, Seaborn, Plotly.
- **Giao diện Web**: Streamlit 1.66.0 hiện đại với 5 phân hệ chuyên sâu.
- **Quy trình chống rò rỉ dữ liệu (Data Leakage Guard)**:
  - Chia tập 70% Train (1,022 dòng), 15% Validation (219 dòng), 15% Holdout Test (219 dòng) với seed 42 cố định.
  - Imputer và Encoder chỉ `fit` duy nhất trên tập Train, sau đó `transform` trên Validation và Test.
  - Tập Holdout Test được đóng băng (freeze) và chỉ đánh giá đúng 1 lần sau khi hoàn thiện mô hình.
  - Mô hình đạt **MAE ~\$15,000 USD** và hệ số xác định **$R^2 \approx 0.89$** trên tập Test độc lập.




## 2. Hướng dẫn cài đặt trên Windows

Dự án sử dụng **Python 3.11** trên Windows. Mọi câu lệnh bên dưới được thiết kế để thực thi trực tiếp, **không yêu cầu kích hoạt môi trường ảo thủ công**.

### Bước 1: Kiểm tra phiên bản Python 3.11
```powershell
py -3.11 --version
```
*(Nếu chưa có Python 3.11, tải và cài đặt từ [python.org](https://www.python.org/downloads/release/python-3119/), nhớ tích chọn **"Add python.exe to PATH"**).*

### Bước 2: Tạo môi trường ảo `.venv`
```powershell
py -3.11 -m venv .venv
```

### Bước 3: Cài đặt các thư viện cần thiết
Sử dụng file snapshot chuẩn `requirements-final-lock.txt` để đảm bảo tương thích 100% các phiên bản thư viện đã kiểm thử:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-final-lock.txt
```

### Bước 4: Kiểm tra tính toàn vẹn phụ thuộc
```powershell
.\.venv\Scripts\python.exe -m pip check
```
*Kết quả thông báo `No broken requirements found.` là cài đặt thành công hoàn hảo.*

---

## 3. Hướng dẫn chạy Jupyter Notebook EDA

File notebook: `notebooks/house_price_eda.ipynb` chứa toàn bộ quá trình phân tích dữ liệu khám phá (EDA) chuyên sâu với 7 biểu đồ trực quan hóa dữ liệu tập Train.

### Các bước thực hiện trong VS Code:
1. Mở thư mục dự án trong VS Code:
   ```powershell
   code .
   ```
2. Cài đặt các extension bắt buộc (nếu chưa có):
   - **Python** (`ms-python.python`)
   - **Jupyter** (`ms-toolsai.jupyter`)
3. Mở file `notebooks/house_price_eda.ipynb`.
4. Chọn **Kernel**: Nhấp vào nút **Select Kernel** ở góc trên cùng bên phải cửa sổ notebook -> Chọn **Python Environments...** -> Trỏ đến interpreter của môi trường ảo: `.\.venv\Scripts\python.exe`.
5. **Xem kết quả đã lưu**: Notebook đã lưu sẵn toàn bộ kết quả chạy và hình ảnh biểu đồ trực quan.
6. **Thực thi lại**: Bạn có thể chạy từng ô (cell) hoặc nhấn **Run All** để tính toán lại.
7. **Lưu ý nghiên cứu**: Để giữ nguyên hồ sơ nghiệm thu ban đầu, khuyến nghị bạn tạo một bản sao notebook (`house_price_eda_copy.ipynb`) nếu muốn thử nghiệm thêm các biểu đồ mới.

---

## 4. Hướng dẫn chạy website Streamlit

Dự án cung cấp giao diện web hiện đại, dễ sử dụng, được xây dựng bằng Streamlit.

### Chạy ứng dụng:
Tại thư mục gốc của dự án, mở PowerShell và chạy lệnh:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Ứng dụng sẽ tự động mở trên trình duyệt mặc định tại địa chỉ:
👉 **`http://localhost:8501`**

> [!NOTE]
> Website sử dụng trực tiếp mô hình đã huấn luyện trong thư mục `artifacts/`. Bạn **không cần phải huấn luyện lại mô hình hay chạy notebook trước khi mở website**.

### Các chức năng chính trong ứng dụng:

1. **Tổng quan (`Overview`)**:
   - Thống kê kích thước tập huấn luyện (1,022 bản ghi) và phân bố 8 đặc trưng.
   - Thống kê các chỉ số mô tả giá bán: Giá thấp nhất, cao nhất, trung vị và trung bình.
2. **Khám phá dữ liệu (`Exploration`)**:
   - Bộ lọc tương tác đa chiều: Theo khu vực (`Neighborhood`), phân khúc chất lượng (`OverallQual`), và khoảng giá bán.
   - 7 biểu đồ phân tích EDA tương tác trực quan: Phân phối giá, tương quan diện tích - giá, tương quan chất lượng - giá, biến thiên theo năm xây dựng,...
3. **Dự đoán giá nhà (`Prediction`)**:
   - Form nhập liệu trực quan với các thanh trượt và menu chọn 8 thông số của ngôi nhà.
   - Nút **"Dự đoán giá nhà"**: Tính toán và hiển thị giá dự đoán thời gian thực bằng USD cùng các thẻ thông tin chi tiết.
4. **Hiệu năng mô hình (`Model Performance`)**:
   - So sánh định lượng giữa mô hình Baseline (Dummy Regressor) và mô hình **XGBoost**.
   - Biểu đồ phân tích sai số (Residual Plot), sai số tuyệt đối theo từng phân khúc giá và xếp hạng độ quan trọng đặc trưng (Feature Importance).
5. **Về dự án (`About`)**:
   - Giới thiệu nhóm phát triển, thông tin học thuật, giải trình quy trình xây dựng mô hình và tuyên bố miễn trừ trách nhiệm.

---

## 5. Kiểm thử hệ thống tự động (Test Suite)

Dự án đi kèm bộ kiểm thử tự động toàn diện gồm **87 bài test** kiểm tra tính toàn vẹn của mã nguồn, mô hình, dữ liệu, bảo mật chống rò rỉ dữ liệu và kiểm thử giao diện qua 4 giai đoạn độc lập:
- **Suite 1 (Tuần 1 - EDA & Data Audit):** 21/21 bài PASS
- **Suite 2 (Tuần 2 - Baseline & Preservation):** 20/20 bài PASS
- **Suite 3 (Tuần 3 - Hyperparameter Tuning & UI Views):** 27/27 bài PASS
- **Suite 4 (Final E2E - Holdout Evaluation & App):** 19/19 bài PASS

Chạy lệnh kiểm thử tổng hợp bằng script điều phối chính sách:

```powershell
.\.venv\Scripts\python.exe -m src.utils.validate_final
```

**Kết quả kiểm thử chuẩn:**
```text
{'PASS': 87, 'FAIL': 0, 'SKIPPED': 0}
```
*(100% bài kiểm thử ĐẠT, 0 lỗi, 0 bài bị bỏ qua).*



## 6. Hướng dẫn tải dữ liệu thô và huấn luyện lại

Dự án **đã bao gồm sẵn mô hình đã huấn luyện** (`artifacts/`) và **dữ liệu đã chia tập** (`data/processed/`) để bạn có thể clone về và chạy ngay lập tức.

Trong trường hợp bạn muốn tái lập quá trình huấn luyện từ đầu trong một môi trường nghiên cứu mới:

1. Đăng nhập vào [Kaggle House Prices Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data), chấp thuận điều khoản cuộc thi.
2. Tải file `train.csv` và đặt vào thư mục `data/raw/train.csv`.
3. Chạy lệnh kiểm toán và phân chia dữ liệu:
   ```powershell
   .\.venv\Scripts\python.exe -m src.data.split_dataset
   ```
4. Chạy quá trình tìm kiếm siêu tham số và huấn luyện mô hình XGBoost:
   ```powershell
   .\.venv\Scripts\python.exe -m src.models.tune_model --n-iter 20
   ```
5. Đánh giá mô hình trên tập Holdout Test:
   ```powershell
   .\.venv\Scripts\python.exe -m src.models.evaluate_test
   ```

---

## 7. Hướng dẫn khắc phục lỗi thường gặp

| Lỗi gặp phải | Nguyên nhân | Cách khắc phục |
|---|---|---|
| `py : The term 'py' is not recognized` | Chưa cài đặt Python hoặc chưa thêm Python vào biến môi trường PATH. | Cài đặt Python 3.11 từ python.org và nhớ tích chọn ô **"Add python.exe to PATH"**. |
| `Execution of scripts is disabled on this system` | Chính sách bảo mật PowerShell của Windows chặn chạy script kích hoạt venv. | Không cần kích hoạt môi trường! Chỉ cần chạy trực tiếp qua `.\.venv\Scripts\python.exe` hoặc mở PowerShell quyền Admin và gõ: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`. |
| `No module named '...'` | Chưa cài đặt đầy đủ các thư viện từ requirements. | Chạy lại lệnh: `.\.venv\Scripts\python.exe -m pip install -r requirements-final-lock.txt`. |
| Không chọn được Kernel trong VS Code | Chưa cài đặt extension Jupyter hoặc chưa trỏ đúng venv. | Cài đặt extension **Jupyter** và **Python** trong VS Code, sau đó chọn Kernel trỏ đến file `.\.venv\Scripts\python.exe`. |
| `Port 8501 is already in use` | Cổng 8501 đang bị chiếm dụng bởi một ứng dụng khác. | Chỉ định cổng chạy khác: `.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502`. |
| `FileNotFoundError: data/processed/train.csv` | Chạy lệnh sai thư mục làm việc (working directory). | Luôn đảm bảo đứng ở thư mục gốc của dự án (`cd KHDL`). |
| Mô hình dự đoán chậm trên GPU | Kích thước dữ liệu nhỏ khiến chi phí truyền bộ nhớ sang VRAM lớn hơn thời gian tính toán. | Mặc định mô hình dự đoán (inference) đã được tối ưu chạy bằng CPU để đạt độ trễ thấp nhất (~vài mili-giây). |

---

## Tác giả & Giấy phép
- Dự án được phát triển phục vụ mục đích nghiên cứu và học tập môn Khoa học Dữ liệu.
- Giấy phép: MIT License.
