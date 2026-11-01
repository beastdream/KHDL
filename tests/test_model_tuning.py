import json
import numpy as np
import pandas as pd
import pytest
from src.data.week3_inputs import load_split
from src.data.schema import FEATURES, NUMERIC_FEATURES
from src.models.week3_pipeline import build_pipeline
from src.models.tune_model import SPACE
from src.utils.paths import ROOT, REPORTS, sha256


def test_cv_fold_preprocessing_and_device_evidence():
    audit = json.loads((REPORTS / 'week3_fit_audit.json').read_text())
    meta = json.loads((ROOT / 'artifacts/model_metadata.json').read_text())
    train = load_split('train')
    successful = [f for f in audit['fits'] if f['success']]
    # A failed CUDA search can leave successful partial fits; check the completed search suffix.
    completed = successful[-(meta['n_iter'] * 5 + 3):]
    assert len(completed) == meta['n_iter'] * 5 + 3
    for index, fit in enumerate(completed):
        if index < meta['n_iter'] * 5:
            fold = audit['folds'][index % 5]
            assert fit['row_indices'] == fold['train_indices']
            assert not set(fold['train_indices']) & set(fold['heldout_indices'])
            assert set(fold['train_indices']) | set(fold['heldout_indices']) == set(range(len(train)))
        else:
            assert fit['row_indices'] == list(range(len(train)))
        expected = train.iloc[fit['row_indices']][NUMERIC_FEATURES].median().to_numpy()
        np.testing.assert_allclose(fit['numeric_medians'], expected)
        assert fit['seconds'] > 0
        assert fit['device'] == fit['booster_config']['learner']['generic_param']['device']
        if audit['cv_device'] == 'cuda':
            from src.utils.check_gpu import cuda_config_ready
            assert cuda_config_ready(fit['booster_config'])
    results = pd.read_csv(REPORTS / 'tuning_results.csv')
    assert len(results) == meta['n_iter'] == 20
    assert np.isfinite(results.CV_MAE).all()
    assert results.CV_MAE.min() == pytest.approx(meta['best_cv_mae'])
    for key, choices in SPACE.items():
        assert set(results['param_' + key]) <= set(choices)


def test_pipeline_can_train_and_imputes_from_fit_rows():
    x = load_split('train')[FEATURES].head(30).copy()
    y = load_split('train').SalePrice.head(30)
    x.loc[x.index[0], 'GarageCars'] = np.nan
    pipe = build_pipeline('cpu').set_params(model__n_estimators=3, model__max_depth=2)
    pipe.fit(x, y)
    assert np.isfinite(pipe.predict(x)).all()
    expected = x[NUMERIC_FEATURES].median().to_numpy()
    np.testing.assert_allclose(pipe.named_steps['preprocessor'].named_transformers_['numeric'].statistics_, expected)


def test_data_guard_and_preservation():
    for relative in ['test.csv', 'train.csv', 'data/raw/train.csv', 'data/processed/test.csv']:
        with pytest.raises(PermissionError):
            (ROOT / relative).open('rb')
    with pytest.raises(PermissionError):
        (ROOT / 'data/processed/train.csv').open('wb')
    meta = json.loads((ROOT / 'artifacts/model_metadata.json').read_text())
    from src.utils.plotting_preservation import verify_notebook_presentation
    assert verify_notebook_presentation()
    assert sha256(REPORTS / 'data_split.json') == meta['split_manifest_sha256']
    assert sha256(ROOT / 'artifacts/xgboost_baseline.joblib') == meta['baseline_sha256']
    manifest = json.loads((REPORTS / 'data_split.json').read_text())
    assert manifest['seed'] == 42
    assert {k: v['sha256'] for k,v in manifest['splits'].items()} == meta['split_hashes']
    assert not set(meta['fit_source_ids']) & set(meta['validation_source_ids'])
    assert not set(meta['fit_source_ids']) & set(manifest['splits']['test']['source_ids'])
    assert not set(meta['validation_source_ids']) & set(manifest['splits']['test']['source_ids'])
    for name in ('train', 'valid'):
        assert len(load_split(name)) == manifest['splits'][name]['count']

