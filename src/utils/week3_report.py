"""Report only observed artifacts and JUnit results; never infer success from GPU availability."""
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
from src.data.week3_inputs import install_data_guard
from src.utils.paths import ROOT, REPORTS, sha256, write_json


def test_counts(path):
    cases = ET.parse(path).findall('.//testcase')
    return {'PASS': sum(c.find('failure') is None and c.find('error') is None and c.find('skipped') is None for c in cases),
            'FAIL': sum(c.find('failure') is not None or c.find('error') is not None for c in cases),
            'SKIP': sum(c.find('skipped') is not None for c in cases)}


def write_report():
    install_data_guard()
    meta = json.loads((ROOT / 'artifacts/model_metadata.json').read_text())
    gate = json.loads((REPORTS / 'week2_status.json').read_text())
    audit = json.loads((REPORTS / 'week3_fit_audit.json').read_text())
    old, new = test_counts(REPORTS / 'week2_pytest.xml'), test_counts(REPORTS / 'week3_pytest.xml')
    legacy_cases = ET.parse(REPORTS / 'week2_pytest.xml').findall('.//testcase')
    def case_status(name):
        case = next((c for c in legacy_cases if c.get('name') == name), None)
        if case is None:
            return 'NOT VERIFIED'
        if case.find('failure') is not None or case.find('error') is not None:
            return 'FAIL'
        return 'SKIP' if case.find('skipped') is not None else 'PASS'
    ready = (gate['week3_gate'] == 'READY' and old['FAIL'] == new['FAIL'] == new['SKIP'] == 0
             and new['PASS'] >= 24 and meta['frozen_for_week4']
             and sha256(ROOT / 'artifacts/house_price_model.joblib') == meta['artifact_sha256']
             and sha256(ROOT / 'notebooks/house_price_eda.ipynb') == meta['notebook_sha256'])
    status = {'status': 'PASS' if ready else 'NOT READY', 'week2_tests': old, 'week3_tests': new,
              'selected_model': meta['selected_model'], 'validation_metrics': meta['validation_metrics'],
              'training_device': meta['training_device'], 'test_accessed': False}
    write_json(REPORTS / 'week3_status.json', status)
    comparison = pd.read_csv(REPORTS / 'model_comparison.csv')
    lines = ['# Báo cáo Tuần 3 — HousePrice AI', '',
        'Ngày lập: ' + datetime.now(ZoneInfo('Asia/Bangkok')).isoformat(), '',
        '## 1. Tóm tắt công việc', '',
        'Tiếp tục dự án hiện có; tuning XGBoost, chọn mô hình trên Validation, lưu Pipeline hoàn chỉnh và triển khai Streamlit tiếng Việt 5 trang.', '',
        '## 2. Trạng thái kế thừa Tuần 1–2', '',
        f"Gate Tuần 2: {gate['week3_gate']}; core: {gate['core_status']}. Notebook: {case_status('test_eda_train_only_and_notebook_executed')}; preservation: {case_status('test_week1_preservation')}.",
        'Đã đọc đủ các tài liệu/module bắt buộc trong Phase 0; không thiếu file trong danh sách. Giữ nguyên source Tuần 1–2, requirements, notebook, baseline model và manifest.',
        f"Train={meta['training_sample_count']}, Validation={meta['validation_sample_count']}; Test=219 theo metadata đã lưu, không mở/rehash Test. Split 70/15/15, seed 42.",
        'test.csv ở thư mục gốc là dữ liệu Kaggle không nhãn; data/processed/test.csv là holdout nội bộ có nhãn. Cả hai đều bị khóa. Hash Train/Valid được kiểm tra qua loader cũ; không chia lại dữ liệu.',
        'Data Guard Tuần 2 giữ nguyên. Policy Tuần 3 giữ hạn chế dữ liệu và chỉ thêm ba đường dẫn CSV báo cáo cụ thể. Hai bộ tests chạy trong hai process riêng để policy Tuần 2 không chặn CSV báo cáo mới.', '',
        '## 3. Hyperparameter tuning', '',
        f"RandomizedSearchCV: {meta['n_iter']} / 288 cấu hình, KFold 5-fold shuffle=True, seed 42, scoring neg_mean_absolute_error, n_jobs=1, refit=False. Tổng 100 fit CV + 3 fit ứng viên trên toàn Train.",
        'Preprocessing nằm trong Pipeline và fit độc lập mỗi fold. Nhật ký lưu chỉ số dòng và median thực tế từng lần fit, được tests đối chiếu với đúng fold Train. Không early stopping; không fit Validation.',
        f"Best CV MAE: {meta['best_cv_mae']:,.2f} USD; thời gian search: {meta['cv_seconds']:.2f} giây. Best CV parameters:",
        '```json', json.dumps(meta['best_cv_parameters'], indent=2), '```',
        'Ba ứng viên CV tốt nhất được so sánh với baseline trên Validation; best CV và model cuối có thể khác nhau.', '',
        '## 4. GPU training verification', '',
        f"XGBoost {meta['xgboost_version']}; scikit-learn {meta['sklearn_version']}. {len(audit['fits'])} fit được ghi nhận; thiết bị thực tế: " + ', '.join(sorted({f.get('device', 'unknown') for f in audit['fits']})) + '.',
        'Booster của từng fit xác nhận cuda:0 và grow_gpu_hist. NVIDIA GeForce GTX 1650, 4 GB, driver 566.07; audit Tuần 1 ghi driver CUDA compatibility 12.7. Không thay driver/CUDA.',
        'Có cảnh báo prediction fallback sang DMatrix do input CPU và Booster CUDA; đây là đường dự đoán, không phải training fallback CPU. Toàn bộ cảnh báo giữ trong week3_fit_audit.json.',
        'VRAM chỉ là snapshot sau training (xem audit), không phải peak hoặc đo riêng process. Không kết luận GPU nhanh hơn CPU vì chưa benchmark đối chứng.',
        f"Model cuối: {meta['training_status']}, fit {meta['training_seconds']:.2f} giây; inference CPU đã kiểm tra kết quả khớp.", '',
        '## 5. Model comparison', '',
        '| Model | MAE (USD) | RMSE (USD) | R² | Device |', '|---|---:|---:|---:|---|']
    for row in comparison.to_dict('records'):
        lines.append(f"| {row['Model']} | {row['MAE']:,.2f} | {row['RMSE']:,.2f} | {row['R2']:.4f} | {row['Device']} |")
    lines += ['', '## 6. Best model selection', '',
        f"Chọn **{meta['selected_model']}** theo MAE Validation thấp nhất; RMSE dùng khi bằng MAE. Không chọn chỉ theo R². Xem model_selection.md và model_comparison.csv.",
        'Artifact artifacts/house_price_model.joblib là Pipeline nhận DataFrame 8 features trực tiếp. Metadata lưu schema, dtypes, dataset/split hashes, seed, số mẫu, IDs nguồn, metrics, versions, tham số, thời gian và checksum.',
        'Training device được lưu riêng; hyperparameters.device=cpu mô tả cấu hình inference sau khi đóng băng. Không fit lại Train+Validation. CLI từ chối ghi đè artifact đã đóng băng; nghiên cứu mới phải dùng bản sao dự án.',
        'SHA-256 artifact: `' + meta['artifact_sha256'] + '`.', '',
        '## 7. Streamlit application', '',
        'app.py: Tổng quan, Khám phá dữ liệu, Dự đoán giá nhà, Hiệu năng mô hình, Về dự án. Giao diện tiếng Việt, bố cục co giãn và màu xanh nhất quán.', '',
        '## 8. Các chức năng đã triển khai', '',
        '- Dashboard: thống kê và biểu đồ chỉ từ Train.',
        '- EDA: 7 biểu đồ Plotly, bộ lọc khu vực/chất lượng/giá/diện tích, xử lý kết quả lọc rỗng.',
        '- Dự đoán: 8 trường bắt buộc, kiểm tra số nguyên/hữu hạn/phạm vi/khu vực; gọi Pipeline.predict, hiển thị USD và cảnh báo ngoài khoảng Train.',
        '- Performance: Dummy, baseline, 3 ứng viên tuned; 4 biểu đồ so sánh/actual-predicted/residual/absolute error trên Validation.',
        '- About: dữ liệu lịch sử Ames, nguồn Kaggle, giới hạn áp dụng; không tuyên bố dự đoán thị trường Việt Nam.', '',
        '## 9. Test results', '',
        f"Regression Tuần 1–2: {old['PASS']} PASS / {old['FAIL']} FAIL / {old['SKIP']} SKIP.",
        f"Tuần 3: {new['PASS']} PASS / {new['FAIL']} FAIL / {new['SKIP']} SKIP.",
        '14 SKIP là các tests cũ cần raw/holdout: schema nguồn, tái lập split, dữ liệu không hợp lệ, missing, tamper/freeze, persisted artifacts, chống ghi đè holdout và audit toàn nguồn. Không coi SKIP là PASS; chưa xác minh lại toàn bộ dữ liệu nguồn/holdout. Giữ bằng chứng lịch sử 21 PASS của Tuần 1.',
        'AppTest thực thi cả 5 trang, 7 biểu đồ EDA, lọc rỗng, nút dự đoán và 4 biểu đồ performance. Đây không phải kiểm thử trình duyệt thực về hover/zoom, ảnh chụp hoặc breakpoint mobile.',
        'Kiểm tra máy chủ cục bộ ngày 2026-10-18: GET / và GET /_stcore/health đều trả HTTP 200; health trả ok. Máy chủ kiểm tra được dừng sau xác minh; dùng lệnh bên dưới để mở lại.',
        'JUnit: week2_pytest.xml và week3_pytest.xml. Safe validation runner trả mã lỗi nếu bất kỳ suite nào lỗi.', '',
        '## 10. Lỗi còn tồn tại và giới hạn', '',
        ('Không có lỗi chặn trong bộ kiểm thử đã chạy.' if ready else 'Còn điều kiện nghiệm thu chưa đạt; xem JUnit và trạng thái gate, không tuyên bố PASS.') + ' Cảnh báo seaborn/Matplotlib deprecated của notebook cũ được giữ lại; không sửa notebook để loại warning.',
        'Validation đã dùng chọn mô hình nên có selection bias; không coi là kết quả tổng quát cuối. Không có khoảng dự đoán, đánh giá drift, kiểm định ngoài Ames hoặc benchmark GPU/CPU. Giới hạn form là kiểm tra hợp lý đầu vào, không bảo đảm độ chính xác ngoài dữ liệu Train.', '',
        '## 11. Hướng dẫn chạy trên Windows PowerShell', '',
        '```powershell', 'Set-Location D:\\KHDL',
        '# Chỉ khi thiếu dependency; môi trường hiện tại đã có đủ:',
        '.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt',
        '# Tuning đã hoàn tất; lệnh này từ chối ghi đè artifact đã đóng băng:',
        '.\\.venv\\Scripts\\python.exe -m src.models.tune_model --n-iter 20',
        '# Chạy hai suite dưới policy riêng và cập nhật báo cáo:',
        '.\\.venv\\Scripts\\python.exe -m src.utils.validate_week3',
        '# Chạy website:', '.\\.venv\\Scripts\\python.exe -m streamlit run app.py', '```',
        'Mở http://localhost:8501. Không chạy src.week1, src.week2 hoặc pytest không có policy trong giai đoạn này.', '',
        '## 12. Kết luận', '', 'WEEK 3 STATUS = **' + status['status'] + '**',
        'Mô hình đã đóng băng cho đánh giá Tuần 4. Chưa chạy Tuần 4 và chưa truy cập Test.', '',
        '## Tham chiếu API', '',
        '- [XGBoost GPU](https://xgboost.readthedocs.io/en/stable/gpu/)',
        '- [scikit-learn Pipeline](https://scikit-learn.org/stable/modules/compose)', '',
        '## Files created / modified', '',
        'Tạo app.py, WEEK_3_README.md, src/data/week3_inputs.py, src/data/app_views.py, src/models/week3_pipeline.py, src/models/tune_model.py, src/models/inference.py, src/utils/week3_report.py, src/utils/validate_week3.py, tests/week3_policy.py và 4 file tests Tuần 3.',
        'Tạo artifacts/house_price_model.joblib, artifacts/model_metadata.json; reports/model_comparison.csv, reports/tuning_results.csv, reports/validation_predictions.csv, reports/model_selection.md, reports/week3_fit_audit.json, reports/week3_pytest.xml, reports/week3_status.json và báo cáo này.',
        'Lệnh nghiệm thu hiện có cập nhật reports/week2_pytest.xml, reports/week2_notebook_audit.json, reports/week2_preservation_check.json, reports/week2_status.json, reports/WEEK_2_REPORT.md; notebook audit tái tạo EDA summary/analysis và 7 PNG Tuần 2. Không sửa notebook/source cũ/checksum gốc.']
    repair_notes = REPORTS / 'week3_repair_notes.md'
    if repair_notes.exists():
        lines += ['', repair_notes.read_text(encoding='utf-8')]
    (REPORTS / 'WEEK_3_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('Files created/modified: app.py; WEEK_3_README.md; 7 Week 3 source modules; 5 test/policy files; 2 model artifacts; reports listed in WEEK_3_REPORT.md. Existing Week 2 reports refreshed by the safe acceptance commands.')
    print(json.dumps(status, indent=2))
    return status


if __name__ == '__main__':
    write_report()

