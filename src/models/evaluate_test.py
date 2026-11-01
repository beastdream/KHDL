"""One official evaluation of the frozen model; reruns verify cached outputs only."""
import json
import joblib
import numpy as np
import pandas as pd
from src.data.final_inputs import install_data_guard, evaluation_access
from src.data.week2_inputs import load_split
from src.data.schema import FEATURES, validate
from src.models.inference import load_model
from src.models.evaluate_validation import metrics
from src.utils.paths import ROOT, REPORTS, PROCESSED, sha256, write_json
from src.utils.final_precheck import now


def verify_freeze():
    frozen = json.loads((REPORTS / 'final_model_freeze.json').read_text(encoding='utf-8'))
    for path, key in [('artifacts/house_price_model.joblib', 'artifact_sha256'),
                      ('artifacts/model_metadata.json', 'metadata_sha256'),
                      ('artifacts/xgboost_baseline.joblib', 'baseline_sha256'),
                      ('reports/data_split.json', 'manifest_sha256')]:
        if sha256(ROOT / path) != frozen[key]:
            raise ValueError(f'Frozen checksum mismatch: {path}')
    return frozen


def evaluate():
    install_data_guard()
    frozen = verify_freeze()
    result_path = REPORTS / 'test_metrics.json'
    prediction_path = REPORTS / 'test_predictions.json'
    attempts = REPORTS / 'final_evaluation_attempt.json'
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding='utf-8'))
        if result['artifact_sha256'] != frozen['artifact_sha256'] or result['predictions_sha256'] != sha256(prediction_path):
            raise ValueError('Cached evaluation provenance mismatch')
        log = REPORTS / 'final_evaluation_verifications.json'
        history = json.loads(log.read_text()) if log.exists() else []
        history.append({'at': now(), 'action': 'cached checksum verification only', 'test_reopened': False})
        write_json(log, history)
        return result
    if attempts.exists() or prediction_path.exists():
        raise RuntimeError('Prior evaluation attempt found; review evidence before any technical rerun')
    model, meta = load_model()
    train = load_split('train')
    baseline = joblib.load(ROOT / 'artifacts/xgboost_baseline.joblib')['baseline']
    assert baseline.constant_[0, 0] == train.SalePrice.median()
    manifest = json.loads((REPORTS / 'data_split.json').read_text())
    write_json(attempts, {'started_at': now(), 'freeze_sha256': sha256(REPORTS / 'final_model_freeze.json'),
                          'purpose': 'official final evaluation', 'status': 'started'})
    # Read only the exact frozen holdout and only after all prerequisites/checksums.
    with evaluation_access():
        path = PROCESSED / 'test.csv'
        if sha256(path) != manifest['splits']['test']['sha256']:
            raise ValueError('Holdout checksum mismatch')
        test = pd.read_csv(path)
    validate(test)
    expected = manifest['splits']['test']
    assert len(test) == expected['count'] and test.Id.tolist() == expected['source_ids']
    assert not set(test.Id) & set(train.Id)
    assert not set(test.Id) & set(manifest['splits']['valid']['source_ids'])
    x, y = test[FEATURES], test.SalePrice
    selected = model.predict(x)
    dummy = baseline.predict(x)
    for pred in (selected, dummy):
        assert pred.shape == (len(test),) and np.isfinite(pred).all()
    records = [{'Id': int(i), 'Actual': float(a), 'Predicted': float(p), 'Dummy': float(d)}
               for i, a, p, d in zip(test.Id, y, selected, dummy)]
    write_json(prediction_path, {'evaluation_split': 'test', 'artifact_sha256': frozen['artifact_sha256'], 'records': records})
    result = {'evaluated_at': now(), 'evaluation_split': 'test', 'rows': len(test), 'unit': 'USD',
              'artifact_sha256': frozen['artifact_sha256'], 'freeze_sha256': sha256(REPORTS / 'final_model_freeze.json'),
              'test_sha256': expected['sha256'], 'predictions_sha256': sha256(prediction_path),
              'models': {'Dummy Regressor': metrics(y, dummy), 'XGBoost': metrics(y, selected)},
              'validation_selected_model': meta['validation_metrics'], 'official_evaluation_count': 1,
              'baseline_strategy': 'median fitted on Train only; original baseline artifact',
              'model_changed': False, 'tuning_after_test': False}
    verify_freeze()
    write_json(result_path, result)
    lines = ['# Đánh giá Test chính thức', '', f'Thời điểm: {result["evaluated_at"]}. {len(test)} mẫu Test nội bộ có nhãn; không phải test.csv Kaggle.',
             '', 'Model được đóng băng trước truy cập Test; không fit, tuning, feature selection hoặc early stopping trên Test.',
             '', '| Model | MAE (USD) | RMSE (USD) | R² |', '|---|---:|---:|---:|']
    for name, values in result['models'].items():
        lines.append(f'| {name} | {values["MAE"]:.2f} | {values["RMSE"]:.2f} | {values["R2"]:.6f} |')
    lines += ['', 'Dummy dùng median đã fit chỉ trên Train. Pipeline nhận đúng 8 features, Id và SalePrice không đi vào X.',
              'Chỉ 1 lần đánh giá chính thức. Chạy lại CLI chỉ xác minh checksum kết quả đã xuất; không mở lại Test.',
              f'SHA-256 model: `{frozen["artifact_sha256"]}`. Chi tiết: test_metrics.json, test_predictions.json.']
    (REPORTS / 'test_evaluation.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(attempts, {'completed_at': now(), 'status': 'completed', 'official_evaluation_count': 1,
                          'freeze_sha256': result['freeze_sha256']})
    return result


if __name__ == '__main__':
    print(json.dumps(evaluate(), indent=2, ensure_ascii=False))
