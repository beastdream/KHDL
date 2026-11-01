# HousePrice AI — UI/UX refactoring

Ngày hoàn tất kiểm tra: 2026-10-19 (+07:00). Phạm vi: giao diện Streamlit hiện có, không training/tuning.

**UI REFACTOR: PASS trong phạm vi AppTest và Chrome headless tại các viewport đã kiểm tra.** Giới hạn kiểm tra được ghi ở mục 7; không tuyên bố tương thích mọi trình duyệt/thiết bị.

## 1. Vấn đề phát hiện

- CSS cũ đặt sidebar nền sáng #e8f1f5, không đặt màu chữ tương ứng; không có config theme. Theme tối của Streamlit có thể cấp chữ sáng trên nền CSS sáng. Chưa quan sát trực tiếp trình duyệt để xác định theme người dùng đang chọn.
- `.stApp` và `[data-testid="stMetric"]` chỉ ghi đè nền, dễ xung đột màu chữ kế thừa. Nhãn form và kết quả dùng màu mặc định. Hero gradient và biểu đồ teal không đồng nhất với bảng màu mới.
- CSS inline trong app.py; nút chỉ có border-radius, thiếu thiết kế focus/hover riêng. Chưa có card kết quả độc lập và breakpoint rõ cho các cột.
- Performance thiếu metric cards và trường Inference device riêng. Các biểu đồ performance và EDA chưa dùng cùng font, background, màu và đơn vị.
- Kiểm tra tương phản phát hiện #64748B trên #F5F7FB chỉ 4,44:1. Đã dùng #172033 cho chú thích trên nền chính, giữ #64748B trên card trắng (khoảng 4,76:1).

## 2. Toàn bộ file tạo/sửa trong lần này

Sửa:

- `app.py`: bỏ CSS inline cũ, nạp CSS riêng; cập nhật bố cục, card kết quả, spinner, metric cards và trình bày 5 trang.
- `tests/test_app.py`: giữ tests hiện có, thêm test form đối chiếu model thật, kiểm tra presentation không đổi dữ liệu Plotly và tương phản bảng màu/config.

Tạo:

- `.streamlit/config.toml`: theme sáng và theme sidebar. Trước sửa chưa có config nên không có cấu hình cũ bị xóa.
- `assets/styles.css`: CSS có phạm vi main/sidebar/result card, các trạng thái điều khiển và breakpoint.
- `src/ui/__init__.py`, `src/ui/presentation.py`: module trình bày, `load_custom_css()`, `style_chart()`, `show_chart()`.
- `reports/UI_REFACTOR_REPORT.md`: báo cáo này.
- `reports/ui_refactor_before.json`: snapshot 83 hashes và dự đoán trước thay đổi; đây là bằng chứng UI, không thay preservation baseline.
- `reports/ui_before_pytest.xml`: kết quả AppTest trước sửa.
- `reports/ui_refactor_pytest.xml`: kết quả kiểm thử sau sửa.
- `reports/ui_refactor_verification.json`: đối chiếu tính toàn vẹn và startup.
- `tests/ui_browser_check.py`: kiểm tra UI bằng Chrome DevTools, không phụ thuộc Playwright hoặc cài thêm thư viện.
- `reports/ui_browser_verification.json`: computed styles, số biểu đồ, trạng thái nút và kích thước trang thực tế.
- `reports/ui_screenshots/`: ảnh 5 trang desktop, menu chọn, prediction 1024/768/360px, 4 trang còn lại ở 360px và ảnh biểu đồ mobile; `slider_dom.html` là snapshot DOM phục vụ kiểm tra selector.

Không sửa module ML, `src/data/app_views.py`, requirements, Data Guard, model, metadata, baseline checksum, notebook hoặc bất kỳ báo cáo Tuần 1–2/3 cũ nào. Không chạy validate_week3 vì lệnh đó tái sinh báo cáo cũ và chạy unit test fit; dùng bộ tests chọn lọc không training phù hợp nhiệm vụ UI.

## 3. Theme và CSS

Streamlit cài sẵn: **1.66.0**. Đã kiểm tra mọi khóa theme với registry config thực tế, đọc lại giá trị hiệu lực và khởi động server không có lỗi config.

| Vai trò | Màu |
|---|---|
| Nền chính / surface | #F5F7FB / #FFFFFF |
| Sidebar / chữ sidebar | #152238 / #FFFFFF |
| Chữ chính / phụ trên trắng | #172033 / #64748B |
| Accent / hover | #FF5252 / #E63E45 |
| Border | #DCE3ED |
| Success nền / chữ | #E4F5EA / #166534 |

Theme native xử lý màu input, selectbox/popover, slider và success alert. CSS chỉ bổ sung spacing, card, navigation và focus. Không dùng `!important`, class CSS băm, gradient hoặc animation. Các selector `data-testid` vẫn là chi tiết DOM cần kiểm tra khi nâng Streamlit; class `.st-key-prediction_result` dựa trên container key do ứng dụng định nghĩa.

Sidebar giữ nguyên 5 mục, chữ trắng, active border Coral, nền hover #243650, focus trắng; không ẩn nút collapse native. `initial_sidebar_state='auto'` giữ cách thu gọn native trên màn hình hẹp.

Nút Coral dùng chữ trắng **20px bold**: tỷ lệ khoảng 3,19:1 đáp ứng ngưỡng chữ lớn 3:1, không được coi là đạt 4,5:1 cho chữ nhỏ. Hover #E63E45, focus outline Navy 3px, disabled nền border/chữ tối. Các cặp chữ thường đang dùng được kiểm tra >=4,5:1. Chrome xác nhận chữ nút 20px/700, sidebar trắng trên Navy, labels tối; các thẻ khu vực đã chọn đổi sang Navy/chữ trắng thay vì chữ nhỏ trắng trên Coral. Giá trị slider dùng màu tối, có focus outline quanh slider.

## 4. Thay đổi trên từng trang

| Trang | Thay đổi UI | Nguồn dữ liệu giữ nguyên |
|---|---|---|
| Tổng quan | Intro gọn, 6 metric cards trắng, mô tả ngắn, thông tin model/training/inference | Train |
| Khám phá dữ liệu | Giữ 7 biểu đồ và 4 bộ lọc; cùng Plotly white/font/legend; nhãn số nhà tiếng Việt | Train và phép lọc cũ |
| Dự đoán giá nhà | Header/subtitle theo yêu cầu, 8 trường và bounds cũ, 2 cột desktop, spinner, card giá Navy nổi bật với viền Coral | Hàm predict và artifact cũ |
| Hiệu năng mô hình | Cards MAE/RMSE/R² lấy metadata, bảng kết quả nguyên giá trị, 4 biểu đồ cùng theme, biểu đồ so sánh ngang, training/inference tách riêng | Validation, không Test |
| Về dự án | Phân mục tên đề tài, nguồn Ames/Kaggle, quy trình, công nghệ/GPU và giới hạn | Nội dung nghiên cứu hiện có |

Plotly dùng `template='plotly_white'`, `theme=None` để không bị Streamlit ghi đè palette. Style helper chỉ thay màu marker/line dạng đơn sắc và layout, không đổi dữ liệu/encoding; tests đối chiếu cả 7 biểu đồ sau khi loại riêng thuộc tính màu trình bày. Màu số theo chất lượng và correlation vẫn giữ encoding. Tiêu đề dài xuống dòng để đọc ở màn hình nhỏ. Nhãn trục có USD/ft² cho đại lượng tương ứng; correlation vẫn giữ thang phân kỳ −1..1.

Responsive: không đặt chiều rộng pixel cố định; dùng `width='stretch'`, autosize Plotly, font `clamp()`, wrap giá và các cột chuyển sang dọc dưới 700px. Chrome đo không có tràn ngang toàn trang/main ở 1440px và 360px trên cả 5 trang; form/card prediction còn được kiểm tra ở 1024px và 768px. Bảng performance có thể cuộn ngang bên trong.

## 5. Chức năng và dữ liệu bảo toàn

- Lần đối chiếu đầu sau kiểm thử: 83/83 file bảo vệ giữ nguyên hash. Lần chốt ngày 19/10: 78/83 giữ nguyên; 5 file báo cáo đã được tái sinh ngoài các lệnh UI của phiên này lúc 07:21–07:22: `week2_pytest.xml`, `week3_pytest.xml`, `week3_status.json`, `WEEK_2_REPORT.md`, `WEEK_3_REPORT.md`. Không có lệnh validate_week3/report generator trong công việc UI này; không xác định người/process thực hiện từ mtime. Giữ nguyên phiên bản mới, không khôi phục đè hoặc sửa baseline. Các source ML, Data Guard, artifact/metadata, manifest, Train/Valid, notebook và báo cáo Tuần 1 vẫn khớp snapshot.
- Artifact cuối: `133f798a66927bf9b6be827bf733c6007a3b0779b82b2f43f6c881f648205c71`.
- 219/219 dự đoán Validation bằng nhau chính xác trước/sau; form với giá trị nhập khác mặc định cho kết quả đúng bằng hàm inference hiện có.
- Không đổi preprocessing, schema 8 features, hyperparameters, seed 42, model selection, metrics hoặc training configuration. Không fit/retrain model trong lần refactor.
- Các báo cáo mới hiện ghi Tuần 2–3 PASS; đó là bằng chứng được quan sát, không được tính vào kết quả 24 tests UI/inference/selection do phiên này chạy.
- Split 70/15/15 giữ nguyên manifest; Train/Valid vẫn qua loader kiểm tra hash/IDs. Không mở Test hoặc raw, kể cả để hash. Bằng chứng holdout chỉ là metadata được giữ nguyên.
- Preservation audit chạy read-only và PASS; baseline/approval không sửa. CPU inference vẫn là cấu hình chủ động, không phải GPU training failure.

## 6. Kết quả kiểm thử

- Trước sửa: `tests/test_app.py`: **2 PASS**.
- Sau sửa: UI + inference + selection: **24 PASS / 0 FAIL / 0 SKIP**, exit code 0. Bộ này không thay thế toàn bộ regression Tuần 1–3.
- AppTest thực thi đủ 5 trang, 7 EDA charts, 4 performance charts, bộ lọc rỗng, 6 numeric inputs + 1 selectbox + 1 slider, submit và hiển thị giá, đối chiếu kết quả model thực.
- Tests inference kiểm tra giá trị sai, thiếu/thừa field, batch/single prediction, CPU và preprocessing không bị fit lại.
- Python compileall: PASS. Streamlit startup: PASS; HTTP `/` và `/_stcore/health` tại cổng kiểm tra 8513 đều **200**, health `ok`. Server kiểm tra đã dừng.
- Một lần chạy trung gian phát hiện palette secondary/main 4,44:1; đã sửa cặp màu sử dụng, không hạ ngưỡng test. Kết quả cuối nằm trong `ui_refactor_pytest.xml`.
- Bổ sung Chrome headless qua DevTools với profile tạm và port debug ngẫu nhiên, chỉ mở localhost; không cài thư viện. Browser được đóng sau kiểm tra. Phiên Chromium thông thường không hoạt động trong môi trường nên script dùng `--no-sandbox` cho phiên QA localhost riêng, không dùng để duyệt Internet.
- Browser test kiểm tra hover bằng pointer thật, focus bằng keyboard, và disabled bằng thuộc tính DOM tạm thời rồi khôi phục (app không vô hiệu hóa nút theo nghiệp vụ). Kết quả: hover rgb(230,62,69), focus solid, disabled rgb(220,227,237). Selectbox mở bằng bàn phím, option có chữ tối. Không có Streamlit exception trên các trang kiểm tra.
- Render thực tế phát hiện selector `kind=primary` không khớp submit DOM ở phiên bản này và Plotly Express giữ marker color mặc định. Đã sửa selector có scope form submit và palette marker; ảnh sau sửa xác nhận chữ lớn và màu thống nhất.

## 7. Phạm vi nghiệm thu trực quan và giới hạn

Ban đầu không có connector browser/Playwright. Sau đó đã tìm thấy Chrome cài trên máy và dùng DevTools qua thư viện websockets có sẵn; ảnh được chụp từ ứng dụng thực, không tạo ảnh mô phỏng.

Đã đọc ảnh desktop, dropdown, card dự đoán mobile, bộ lọc và performance mobile; đối chiếu computed styles với bảng màu. Nền sidebar và card, nhãn input, thẻ khu vực, giá kết quả và nút đều có màu chữ phân biệt rõ. Script đi qua 5 mục navigation, submit model thật, thu gọn sidebar và đo overflow. Bằng chứng nằm trong `ui_browser_verification.json` và `ui_screenshots/`.

Giới hạn: không có screenshot nguyên bản trước refactor vì kết nối trình duyệt chỉ thiết lập được sau khi sửa; không giả lập ảnh “before”. Trạng thái trước được ghi bằng kiểm tra source và AppTest 2 PASS. Chưa kiểm tra Safari/Firefox, thiết bị cảm ứng thật, screen reader, zoom trình duyệt 200% hoặc mọi thao tác hover/zoom của Plotly. Nên kiểm tra các trường hợp này khi phát hành cho người dùng rộng hơn; đây không phải bằng chứng thay đổi model hay dữ liệu.

Checklist bổ sung khi mở bằng trình duyệt thường: Tab qua toàn bộ control; mở/đóng sidebar bằng chuột; mở menu multiselect trên điện thoại; zoom 200%; thử hover/zoom/reset Plotly. Không phát hiện lỗi chặn trong phạm vi đã kiểm tra.

## 8. Chạy website và kiểm thử trên Windows

Không cần cài thêm dependency. Nếu server cũ còn chạy, dừng và mở lại để nhận theme config mới.

```powershell
Set-Location D:\KHDL
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Mở http://localhost:8501.

```powershell
.\.venv\Scripts\python.exe -m compileall -q app.py src/ui tests/test_app.py
.\.venv\Scripts\python.exe -m pytest -p tests.week3_policy -q tests/test_app.py tests/test_inference.py tests/test_model_selection.py --junitxml=reports/ui_refactor_pytest.xml
```

Tham chiếu cấu hình chính thức: [Streamlit config.toml](https://docs.streamlit.io/develop/api-reference/configuration/config.toml), [màu và borders](https://docs.streamlit.io/develop/concepts/configuration/theming-customize-colors-and-borders). Khóa sidebar được hỗ trợ bởi phiên bản 1.66.0 đã kiểm tra tại máy.
