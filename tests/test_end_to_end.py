"""Acceptance contract verifies the actual frozen research and deployed inference."""
import json
import subprocess
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from src.data.schema import FEATURES, NUMERIC_FEATURES
from src.data.week2_inputs import load_split
from src.data.final_inputs import evaluation_access
from src.models.inference import load_model, predict
from src.models.evaluate_test import verify_freeze
from src.models.evaluate_validation import metrics
from src.utils.paths import ROOT, REPORTS, sha256
from src.utils.preservation import audit_preservation


def read(name):
    return json.loads((REPORTS / name).read_text(encoding='utf-8'))


def test_split_schema_membership_and_seed():
    manifest = read('data_split.json')
    assert manifest['seed'] == 42 and manifest['preprocessing_fit'] is False
    from sklearn.model_selection import train_test_split
    indices = np.arange(manifest['rows'])
    train, rest = train_test_split(indices, test_size=.30, random_state=42)
    valid, test = train_test_split(rest, test_size=.50, random_state=42)
    for name, expected in zip(('train', 'valid', 'test'), (train, valid, test)):
        saved = manifest['splits'][name]
        assert saved['source_indices'] == expected.tolist()
        assert saved['count'] == len(expected)
        assert saved['fraction'] == pytest.approx(len(expected) / len(indices))
    for field in ('source_ids', 'source_indices'):
        sets = [set(s[field]) for s in manifest['splits'].values()]
        assert sum(map(len, sets)) == len(set.union(*sets)) == len(indices)
    for name in ('train', 'valid'):
        frame = load_split(name)
        assert set(FEATURES).issubset(frame.columns)
        assert 'Id' not in FEATURES and 'SalePrice' not in FEATURES


def test_frozen_model_and_preprocessing_train_only():
    frozen = verify_freeze()
    model, meta = load_model()
    assert frozen['features'] == meta['features'] == FEATURES
    assert meta['frozen_for_week4'] and not meta['early_stopping']
    train = load_split('train')
    assert train.Id.tolist() == meta['fit_source_ids']
    prep = model.named_steps['preprocessor']
    assert prep.feature_names_in_.tolist() == FEATURES
    np.testing.assert_allclose(prep.named_transformers_['numeric'].statistics_, train[NUMERIC_FEATURES].median())
    assert set(prep.named_transformers_['categorical'].named_steps['encoder'].categories_[0]) == set(train.Neighborhood)
    before = joblib.hash(prep)
    probe = train[FEATURES].head(3).copy()
    probe['GarageCars'] = np.nan
    probe['Neighborhood'] = 'UNSEEN'
    assert np.isfinite(model.predict(probe)).all()
    assert before == joblib.hash(prep)


def test_real_single_batch_and_cpu_inference():
    model, metadata = load_model()
    frame = load_split('train')[FEATURES].head(100)
    assert model.named_steps['model'].get_params()['device'] == 'cpu'
    values = predict(frame, model, metadata)
    assert len(values) == 100 and np.isfinite(values).all()
    np.testing.assert_array_equal(values[:1], predict(frame.head(1), model, metadata))


@pytest.mark.parametrize('change', ['missing', 'extra_target', 'negative', 'nan', 'infinite', 'category'])
def test_invalid_inference(change):
    model, metadata = load_model()
    frame = load_split('train')[FEATURES].head(1).copy()
    if change == 'missing': frame = frame.drop(columns='FullBath')
    if change == 'extra_target': frame['SalePrice'] = 1
    if change == 'negative': frame['GrLivArea'] = -1
    if change == 'nan': frame['GarageCars'] = np.nan
    if change == 'infinite': frame['GrLivArea'] = np.inf
    if change == 'category': frame['Neighborhood'] = 'NOT_SUPPORTED'
    with pytest.raises(ValueError): predict(frame, model, metadata)


def test_missing_model_without_touching_artifact(monkeypatch, tmp_path):
    import src.models.inference as inference
    (tmp_path / 'artifacts').mkdir()
    (tmp_path / 'artifacts/model_metadata.json').write_text((ROOT / 'artifacts/model_metadata.json').read_text())
    monkeypatch.setattr(inference, 'ROOT', tmp_path)
    with pytest.raises(FileNotFoundError): inference.load_model()


@pytest.mark.parametrize('name', ['data/processed/test.csv', 'data/raw/train.csv', 'test.csv'])
def test_unauthorized_dataset_reads_and_writes_blocked(name):
    path = ROOT / name
    for mode in ('rb', 'wb'):
        with pytest.raises(PermissionError): path.open(mode)


def test_holdout_capability_rejects_non_evaluator():
    with pytest.raises(PermissionError):
        with evaluation_access():
            pytest.fail('Capability leaked')


def test_guard_revokes_capability_even_after_exception():
    # Compile a tiny trusted evaluator-frame probe in a separate process. It tests
    # access-policy transitions without opening holdout bytes or running evaluation.
    code = '''
from src.data.final_inputs import install_data_guard, evaluation_access, _access
from src.utils.paths import ROOT
install_data_guard()
namespace = {}
exec(compile("from src.data.final_inputs import evaluation_access, _access\\ndef probe():\\n with evaluation_access():\\n  assert _access.get()\\n  raise RuntimeError('probe')", str(ROOT / 'src/models/evaluate_test.py'), 'exec'), namespace)
try: namespace['probe']()
except RuntimeError: pass
assert not _access.get()
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_official_evaluation_export_and_no_model_change():
    frozen = verify_freeze()
    report = read('test_metrics.json')
    assert report['official_evaluation_count'] == 1 and not report['model_changed'] and not report['tuning_after_test']
    assert report['artifact_sha256'] == frozen['artifact_sha256']
    assert report['freeze_sha256'] == sha256(REPORTS / 'final_model_freeze.json')
    assert report['test_sha256'] == frozen['split_hashes']['test']
    assert report['predictions_sha256'] == sha256(REPORTS / 'test_predictions.json')
    records = pd.DataFrame(read('test_predictions.json')['records'])
    manifest = read('data_split.json')
    assert records.Id.tolist() == manifest['splits']['test']['source_ids']
    assert len(records) == report['rows'] == manifest['splits']['test']['count']
    for name, column in [('XGBoost', 'Predicted'), ('Dummy Regressor', 'Dummy')]:
        # Native predictions were float32; exported JSON preserves exact values.
        values = records[column].to_numpy(dtype=np.float32 if name == 'XGBoost' else np.float64)
        actual = metrics(records.Actual, values)
        for key in actual:
            assert actual[key] == pytest.approx(report['models'][name][key], rel=1e-6)
    train = load_split('train')
    assert records.Dummy.eq(train.SalePrice.median()).all()


def test_selection_gpu_fit_and_validation_evidence():
    model, meta = load_model()
    actual = metrics(load_split('valid').SalePrice, model.predict(load_split('valid')[FEATURES]))
    for key in actual: assert actual[key] == pytest.approx(meta['validation_metrics'][key], rel=1e-6)
    audit = read('week3_fit_audit.json')
    assert len(audit['fits']) == meta['n_iter'] * meta['cv_folds'] + 3
    assert all(f['success'] and f['device'].startswith('cuda') for f in audit['fits'])
    importance = read('feature_importance.json')
    assert importance['split'] == 'valid' and {r['feature'] for r in importance['ranking']} == set(FEATURES)
    assert all(len(r['repeats']) == 20 for r in importance['ranking'])
    benchmark = read('gpu_benchmark.json')
    assert benchmark['devices']['cpu']['status'] == 'SUCCESS'
    assert not benchmark['test_accessed'] and not benchmark['benchmark_models_replace_final']
    if benchmark['devices']['cuda']['status'] == 'SUCCESS':
        assert all(r['actual_device'].startswith('cuda') and r['updater'] == 'grow_gpu_hist'
                   for r in benchmark['devices']['cuda']['fits'])
    else:
        assert benchmark['devices']['cuda']['reason']


def test_preservation_and_seven_exported_charts():
    assert audit_preservation()['status'] == 'PASS'
    expected = ['actual_vs_predicted', 'residual_distribution', 'residual_vs_predicted',
                'absolute_error_distribution', 'error_by_price_range', 'percentage_error_by_price_range', 'feature_importance']
    for name in expected:
        assert (REPORTS / f'figures/final/{name}.png').read_bytes().startswith(b'\x89PNG')


def test_streamlit_five_pages_and_final_performance():
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
    assert not app.exception
    assert len(app.sidebar.radio[0].options) == 5
    for name in app.sidebar.radio[0].options:
        app.sidebar.radio[0].set_value(name).run()
        assert not app.exception, name
        if name == 'Dự đoán giá nhà':
            app.button[0].click().run()
            assert not app.exception and any('USD' in m.value for m in app.metric)
        if name == 'Hiệu năng mô hình':
            assert any(m.label == 'Test R²' for m in app.metric)
            assert len(app.image) == 7
