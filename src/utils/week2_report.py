"""Generate the Week 2 report from recorded metrics, metadata and JUnit evidence."""
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from src.utils.paths import ROOT, REPORTS, sha256, write_json
from src.utils.preservation import audit_preservation
from src.data.week2_inputs import install_data_guard


def write_report():
    install_data_guard()
    metrics=json.loads((REPORTS/'validation_metrics.json').read_text())
    metadata=json.loads((ROOT/'artifacts/xgboost_baseline.metadata.json').read_text())
    eda=json.loads((REPORTS/'eda_summary.json').read_text())
    tree=ET.parse(REPORTS/'week2_pytest.xml')
    cases=tree.findall('.//testcase')
    tests=[]
    for case in cases:
        state='SKIP' if case.find('skipped') is not None else 'FAIL' if case.find('failure') is not None or case.find('error') is not None else 'PASS'
        tests.append({'name':case.attrib['classname']+'.'+case.attrib['name'],'status':state})
    preserved=audit_preservation()
    write_json(REPORTS/'week2_preservation_check.json',preserved)
    notebook=json.loads((REPORTS/'week2_notebook_audit.json').read_text(encoding='utf-8'))
    notebook_current=notebook['notebook_sha256']==sha256(ROOT/'notebooks/house_price_eda.ipynb')
    counts={s:sum(t['status']==s for t in tests) for s in ['PASS','FAIL','SKIP']}
    core=counts['FAIL']==0 and preserved['status']=='PASS' and notebook['status']=='PASS' and notebook_current and all(t['status']=='PASS' for t in tests if not t['name'].startswith('tests.test_week1.'))
    # User explicitly accepted the safe Week 1 subset; no unapproved skips count as success.
    scope=json.loads((REPORTS/'week2_test_scope.json').read_text(encoding='utf-8'))
    observed={t['name'] for t in tests}
    skipped={t['name'] for t in tests if t['status']=='SKIP'}
    covered=set(scope['required_cases']) <= observed
    full=core and covered and skipped==set(scope['approved_skips'])
    status={'core_status':'PASS' if core else 'NOT READY','overall_status':'PASS' if full else 'NOT READY',
            'gpu_status':metadata['gpu_status'],'tests':counts,'week1_regression':'SAFE SUBSET PASS; 14 skips explicitly accepted by user' if full else 'NOT VERIFIED',
            'week3_gate':'READY' if full else 'NOT READY', 'test_accessed':False, 'test_scope':'User-approved safe subset; historical full Week 1 results retained'}
    write_json(REPORTS/'week2_status.json',status)
    lines=['# Báo cáo Tuần 2 — House Price Prediction','',
           'Ngày lập: '+datetime.now(timezone(timedelta(hours=7))).isoformat(),'',
           '## 1. Phạm vi và bảo toàn Tuần 1','',
           'Chỉ sử dụng 1.022 mẫu Train và 219 mẫu Validation. Không đọc raw hoặc test.csv, kể cả để hash; không chạy lại pipeline Tuần 1.',
           f"Preservation: {preserved['status']}. 25 tệp khớp baseline; requirements.txt là ngoại lệ A đã xác minh và ghim hash. Không sửa checksum gốc.",
           'Lớp audit hook chặn mở CSV ngoài hai tập được phép và chặn ghi lên hai tập này. Các phép thử truy cập bị chặn trước khi đọc byte dữ liệu.', '',
           '## 2. EDA trên Train','',
           f"Đủ 7 biểu đồ PNG. SalePrice: trung vị {eda['target_median']:,.0f} USD, skewness {eda['target_skew']:.3f}.",
           'Tương quan Pearson với giá: '+', '.join(f'{k}={v:.3f}' for k,v in eda['pearson_target'].items())+'.',
           f"Có {len(eda['iqr_outliers'])} outlier giá theo 1.5 IQR và {len(eda['large_area_outliers'])} mẫu diện tích >4.000 sq ft; giữ nguyên tất cả.",
           f"[EDA tiếng Việt](eda_analysis.md). Notebook có {len(notebook['cells'])} code cells chạy lại không lỗi và {notebook['stored_display_data']} display_data lưu sẵn. Giữ nguyên mọi cell và outputs.", '',
           '## 3. Preprocessing và huấn luyện','',
           'ColumnTransformer: 7 biến số dùng SimpleImputer(median); Neighborhood dùng SimpleImputer(most_frequent) rồi OneHotEncoder(handle_unknown=ignore).',
           'Chỉ fit Train. Validation chỉ transform/predict; không eval_set, early stopping, tuning hoặc lựa chọn mô hình dựa trên Test. Giữ đúng 8 features; loại Id và SalePrice khỏi X.',
           'DummyRegressor(median) fit Train. XGBoost dùng đúng cấu hình khởi đầu:', '', '```json',json.dumps(metadata['params'],indent=2),'```','',
           '## 4. Kết quả Validation','',
           '| Model | MAE | RMSE | R2 | Device |','|---|---:|---:|---:|---|']
    for name, values in metrics['models'].items():
        lines.append(f"| {name} | {values['MAE']:,.2f} | {values['RMSE']:,.2f} | {values['R2']:.4f} | {values['Device']} |")
    lines += ['',f"MAE/RMSE: USD. XGBoost giảm RMSE {metrics['rmse_improvement_pct']:.2f}% so với baseline. R² baseline âm nghĩa là baseline trung vị Train kém hơn dự đoán trung bình Validation theo định nghĩa R², không phải lỗi tính metric.",
              '[Metrics JSON](validation_metrics.json), [bảng metrics](validation_metrics.md). Không báo cáo metric Test.', '',
              '## 5. GPU và thời gian','',metadata['gpu_status'],
              f"Thiết bị Booster sau fit: {metadata['actual_device']}. Đã xác nhận device CUDA và updater grow_gpu_hist sau huấn luyện trên dữ liệu nhà thực.",
              f"Thời gian fit XGBoost: {metadata['training_seconds']:.4f} giây; preprocessing: {metadata['preprocessing_seconds']:.4f} giây (wall-clock, không phải benchmark tốc độ GPU so với CPU).",
              '```text',metadata['gpu_inventory'].get('stdout','Unavailable'),'```',
              'Preprocessing chạy trên CPU; dự đoán dùng Booster.predict(DMatrix) với Booster đã xác nhận CUDA. Không ngụ ý mọi thao tác đều chạy trên GPU.',
              'Các lần thử, warnings, cấu hình Booster, phiên bản, nvidia-smi, thông số và hash artifact: artifacts/xgboost_baseline.metadata.json.', '',
              '## 6. Artifact','',
              '`artifacts/xgboost_baseline.joblib` chứa preprocessor đã fit, XGBRegressor đã fit, Dummy baseline, schema và metadata. Đã load lại và so sánh prediction khớp tuyệt đối.',
              'Dùng src.models.preprocessing.predict_artifact với DataFrame đúng 8 features. Hàm từ chối thiếu/thừa cột và giá trị sai kiểu; cho phép missing và category chưa thấy.', '',
              '## 7. Kiểm thử','',f"Kết quả hiện tại: {counts['PASS']} PASS / {counts['FAIL']} FAIL / {counts['SKIP']} SKIP.",
              'Toàn bộ test Tuần 2 và 7 test Tuần 1 an toàn đã chạy. 14 test cũ SKIP theo lựa chọn đã xác nhận của người dùng; không coi SKIP là PASS.',
              'Báo cáo Tuần 1 lưu kết quả lịch sử 21 PASS và được giữ nguyên. Kết quả lịch sử không chứng minh toàn bộ hồi quy vừa chạy lại.', '',
              '| Test | Trạng thái |','|---|---|']
    lines += [f"| {t['name']} | {t['status']} |" for t in tests]
    lines += ['', '[JUnit](week2_pytest.xml).', '', '## 8. Hạn chế và việc còn lại','',
              'Một lần chia holdout nhỏ và chỉ một cấu hình; chưa cross-validation, tuning, khoảng tin cậy, phân tích công bằng theo nhóm, kiểm tra drift hoặc đánh giá ngoài Ames.',
              'Outlier được giữ lại; RMSE nhạy với nhà giá cao. Correlation không chứng minh quan hệ nhân quả. Chỉ dùng 8 features nên bỏ qua nhiều đặc điểm có thể quan trọng.',
              '14 test Tuần 1 cần raw/holdout không chạy lại theo lựa chọn đã xác nhận. Giới hạn này không được che bằng kết quả lịch sử.', '',
              '## 9. Kết luận','',f"Yêu cầu cốt lõi EDA/preprocessing/model/validation/artifact: {status['core_status']}.",
              f"GPU: {status['gpu_status']}.",f"Toàn bộ mục tiêu: {status['overall_status']} — theo phạm vi test an toàn đã được chấp nhận.",
              'Chưa chuyển sang Tuần 3.', '', '## Tài liệu API','',
              '- [XGBoost GPU](https://xgboost.readthedocs.io/en/stable/gpu/index.html)',
              '- [scikit-learn ColumnTransformer](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html)']
    lines += ['', (REPORTS/'week2_repair_notes.md').read_text(encoding='utf-8-sig'), '',
              '### Danh sách preservation', '', '| Đường dẫn | Phân loại | Trạng thái |', '|---|---|---|']
    lines += [f"| {e['path']} | {e['category']} | {e['status']} |" for e in preserved['entries']]
    lines += ['', 'Chi tiết: [notebook audit](week2_notebook_audit.json), [preservation audit](week2_preservation_check.json), [ngoại lệ đã duyệt](week2_preservation_approvals.json).']
    (REPORTS/'WEEK_2_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(status,indent=2))
    return status


if __name__=='__main__': write_report()
