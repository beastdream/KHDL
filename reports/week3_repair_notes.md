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
