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
