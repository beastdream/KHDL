"""Read-only preflight and append-only model freeze; never opens holdout."""
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import xml.etree.ElementTree as ET
import joblib
import numpy as np
from src.data.schema import FEATURES, NUMERIC_FEATURES
from src.data.week3_inputs import install_data_guard, load_split
from src.models.inference import load_model
from src.models.evaluate_validation import metrics
from src.utils.preservation import audit_preservation
from src.utils.paths import ROOT, REPORTS, sha256, write_json


def now():
    return datetime.now(ZoneInfo('Asia/Bangkok')).isoformat()


def freeze():
    install_data_guard()
    preservation = audit_preservation()
    assert preservation['status'] == 'PASS', preservation
    suites = {}
    for name in ('week2', 'week3'):
        cases = ET.parse(REPORTS / f'{name}_pytest.xml').findall('.//testcase')
        assert cases and not any(c.find('failure') is not None or c.find('error') is not None for c in cases)
        suites[name] = {'passed': sum(c.find('skipped') is None for c in cases),
                        'skipped': sum(c.find('skipped') is not None for c in cases)}
    assert json.loads((REPORTS / 'week3_status.json').read_text())['status'] == 'PASS'
    model, meta = load_model()
    manifest = json.loads((REPORTS / 'data_split.json').read_text())
    assert sha256(REPORTS / 'data_split.json') == meta['split_manifest_sha256']
    assert manifest['seed'] == meta['random_seed'] == 42
    splits = manifest['splits']
    assert sum(s['count'] for s in splits.values()) == manifest['rows']
    for field in ('source_ids', 'source_indices'):
        members = [set(s[field]) for s in splits.values()]
        assert sum(map(len, members)) == len(set.union(*members)) == manifest['rows']
    train, valid = load_split('train'), load_split('valid')
    assert meta['fit_source_ids'] == train.Id.tolist()
    assert meta['validation_source_ids'] == valid.Id.tolist()
    prep = model.named_steps['preprocessor']
    assert prep.feature_names_in_.tolist() == FEATURES == meta['features']
    np.testing.assert_allclose(prep.named_transformers_['numeric'].statistics_, train[NUMERIC_FEATURES].median())
    cat = prep.named_transformers_['categorical']
    assert set(cat.named_steps['encoder'].categories_[0]) == set(train.Neighborhood.dropna())
    actual = metrics(valid.SalePrice, model.predict(valid[FEATURES]))
    for k, value in actual.items():
        assert np.isclose(value, meta['validation_metrics'][k], rtol=1e-6)
    assert sha256(ROOT / 'artifacts/xgboost_baseline.joblib') == meta['baseline_sha256']
    base = joblib.load(ROOT / 'artifacts/xgboost_baseline.joblib')
    assert base['baseline'].constant_[0, 0] == train.SalePrice.median()
    assert base['metadata']['fit_source_ids'] == train.Id.tolist()
    result = dict(created_at=now(), artifact='artifacts/house_price_model.joblib',
        artifact_sha256=meta['artifact_sha256'], metadata_sha256=sha256(ROOT / 'artifacts/model_metadata.json'),
        baseline_sha256=meta['baseline_sha256'], manifest_sha256=meta['split_manifest_sha256'],
        selected_model=meta['selected_model'], features=FEATURES, hyperparameters=meta['hyperparameters'],
        random_seed=42, split_counts={k: s['count'] for k, s in splits.items()},
        split_hashes=meta['split_hashes'], validation_metrics_recorded=meta['validation_metrics'],
        validation_metrics_cpu_verified=actual, training_device=meta['training_device'], inference_device='cpu',
        preprocessing={'numeric': 'median fitted only on Train', 'categorical': 'most frequent + one-hot; ignore unknown',
                       'transformed_features': prep.get_feature_names_out().tolist(),
                       'missing_train': train[FEATURES].isna().sum().to_dict(),
                       'missing_valid': valid[FEATURES].isna().sum().to_dict()},
        test_accessed=False, tuning_complete=True, selection_complete=True,
        no_test_based_changes=True, prerequisite_suites=suites)
    path = REPORTS / 'final_model_freeze.json'
    if path.exists():
        old = json.loads(path.read_text(encoding='utf-8'))
        for key in ('artifact_sha256', 'metadata_sha256', 'baseline_sha256', 'manifest_sha256', 'features', 'hyperparameters'):
            assert old[key] == result[key], f'Freeze mismatch: {key}'
        return old
    write_json(REPORTS / 'final_preservation_evidence.json', preservation)
    (REPORTS / 'FINAL_PRECHECK.md').write_text(
        '# Kiểm tra trước nghiệm thu cuối\n\n'
        f'Ngày: {now()}. Gate: PASS. Kiểm tra hoàn tất trước khi mở Test.\n\n'
        'Đã quét cấu trúc và đọc README Tuần 1–3, requirements/lock, pytest, app/theme, notebook, báo cáo Tuần 1–3/UI, '
        'manifest, metrics, model selection, GPU audit, artifact metadata và các module dữ liệu/model/inference/tests. '
        'Không thiếu file đầu vào bắt buộc. Các deliverable Tuần 4 chưa tồn tại tại thời điểm bắt đầu.\n\n'
        'Kế thừa: audit Ames, split cố định, 7 biểu đồ EDA + notebook đã thực thi, baseline, 20 cấu hình CV 5 fold, '
        '103 GPU fit, Pipeline cuối, 5 trang Streamlit và UI navy/sáng đã refactor. Không xây lại giao diện.\n\n'
        f'Kiểm thử mới: {suites}. Không có failure/error. Preservation PASS; xem final_preservation_evidence.json. '
        'src/week1.py khớp hash gốc 9ff7049430ee114f9e2c2b0eec4909a49a4b760203c4da8bd56085f16159180c. '
        'Lỗi cũ thực tế là 3 LF thừa ở requirements.txt, đã khôi phục về phiên bản được duyệt; '
        'không sửa checksum gốc/test. Bằng chứng: week3_repair_notes.md và week3_repair_evidence/. '
        'git status/log xác nhận không có repository; không thể quy nguồn thay đổi qua lịch sử Git.\n\n'
        f'Model: {meta["selected_model"]}; SHA-256 {meta["artifact_sha256"]}. '
        f'Validation CPU tính lại: {actual}; phù hợp metrics chọn model trong rtol=1e-6. '
        'validation_metrics.json thuộc baseline Tuần 2, không phải model cuối; dùng model_metadata.json cho model cuối.\n\n'
        'GPU lịch sử: GTX 1650 4096 MiB, driver 566.07, compatibility CUDA 12.7, XGBoost 3.2.0. '
        '103 fit có cuda:0/grow_gpu_hist; CPU inference là chủ động. Benchmark mới sẽ xác minh thiết bị thực tế.\n\n'
        'Train/Valid xác minh hash, schema, membership, imputer Train-only, encoder Train-only; '
        'manifest IDs/indices ba tập không giao nhau, seed 42, không chia lại. '
        'Test 15% nội bộ có nhãn khác test.csv Kaggle không nhãn. Chưa mở/rehash holdout.\n\n'
        'Không còn lỗi nghiêm trọng chặn freeze. Hạng mục tiếp theo: kiểm soát truy cập Test, đánh giá chính thức, '
        'phân tích sai số/importance, benchmark, trang hiệu năng và kiểm thử nghiệm thu. '
        '14 tests cũ cần raw/holdout vẫn SKIP theo policy cũ, không gọi là PASS.\n', encoding='utf-8')
    write_json(path, result)
    return result


if __name__ == '__main__':
    print(json.dumps(freeze(), ensure_ascii=False, indent=2))
