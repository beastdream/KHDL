"""Train-only randomized CV; compare three CV finalists on Validation."""
import argparse
import copy
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from src.data.schema import FEATURES, TARGET
from src.data.week3_inputs import install_data_guard, load_split
from src.models.preprocessing import predict_artifact
from src.models.evaluate_validation import metrics
from src.models.week3_pipeline import build_pipeline, FIT_AUDIT
from src.utils.paths import ROOT, REPORTS, sha256, write_json
from src.utils.check_gpu import command

SPACE = {f'model__{k}': v for k, v in {
    'n_estimators': [100, 200, 300, 500], 'max_depth': [3, 5, 7],
    'learning_rate': [.03, .05, .1], 'subsample': [.8, 1.],
    'colsample_bytree': [.8, 1.], 'reg_lambda': [1, 5]}.items()}


def select_model(rows):
    """Primary MAE, RMSE tie-break; R2 remains visible for assessment."""
    eligible = [r for r in rows if r['Model'] != 'Dummy Baseline']
    if not eligible or any(not np.isfinite([r[k] for k in ('MAE', 'RMSE', 'R2')]).all() for r in eligible):
        raise ValueError('Model selection requires finite metrics')
    return min(eligible, key=lambda r: (r['MAE'], r['RMSE']))['Model']


def tune(n_iter=20):
    install_data_guard()
    status = json.loads((REPORTS / 'week2_status.json').read_text())
    if status['core_status'] != 'PASS' or status['week3_gate'] != 'READY':
        raise RuntimeError('Week 2 gate is NOT READY')
    if int(xgboost.__version__.split('.')[0]) < 2:
        raise RuntimeError('XGBoost >= 2 is required for device=cuda')
    destination = ROOT / 'artifacts/house_price_model.joblib'
    if destination.exists():
        raise FileExistsError('Final artifact is frozen. Use a separate project copy for a new experiment.')
    started = time.perf_counter()
    train, valid = load_split('train'), load_split('valid')
    x, y = train[FEATURES], train[TARGET]
    folds = list(KFold(5, shuffle=True, random_state=42).split(x))
    FIT_AUDIT.clear()
    search_warnings, failures = [], []
    for device in ('cuda', 'cpu'):
        try:
            search = RandomizedSearchCV(build_pipeline(device), SPACE, n_iter=n_iter,
                scoring='neg_mean_absolute_error', cv=folds, random_state=42,
                n_jobs=1, refit=False, error_score='raise', verbose=1)
            cv_start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                search.fit(x, y)
            search_warnings.extend(str(w.message) for w in caught)
            cv_seconds = time.perf_counter() - cv_start
            break
        except Exception as exc:
            failures.append({'device': device, 'error': repr(exc)})
            write_json(REPORTS / 'week3_fit_audit.json', {'fits': FIT_AUDIT, 'failures': failures})
            if device == 'cpu':
                raise
    cv_results = pd.DataFrame(search.cv_results_)
    cv_results['CV_MAE'] = -cv_results.mean_test_score
    cv_results.to_csv(REPORTS / 'tuning_results.csv', index=False)
    candidates, candidate_rows = {}, []
    for rank, index in enumerate(np.argsort(search.cv_results_['rank_test_score'])[:3], 1):
        pipe = build_pipeline(device).set_params(**search.cv_results_['params'][index])
        pipe.fit(x, y)
        with warnings.catch_warnings(record=True) as caught:
            prediction = pipe.predict(valid[FEATURES])
        search_warnings.extend(str(w.message) for w in caught)
        name = f'XGBoost tuned CV#{rank}'
        candidates[name] = pipe
        candidate_rows.append({'Model': name, **metrics(valid[TARGET], prediction),
            'CV_MAE': float(-search.cv_results_['mean_test_score'][index]),
            'Device': pipe.fit_evidence_['device'], 'TrainingSeconds': pipe.fit_evidence_['seconds']})
    old_path = ROOT / 'artifacts/xgboost_baseline.joblib'
    old_meta = json.loads((ROOT / 'artifacts/xgboost_baseline.metadata.json').read_text())
    if sha256(old_path) != old_meta['artifact_sha256']:
        raise ValueError('Baseline artifact checksum mismatch')
    baseline = joblib.load(old_path)
    rows = [
        {'Model': 'Dummy Baseline', **metrics(valid[TARGET], baseline['baseline'].predict(valid[FEATURES])), 'Device': 'cpu', 'TrainingSeconds': None},
        {'Model': 'XGBoost baseline', **metrics(valid[TARGET], predict_artifact(baseline, valid[FEATURES])), 'Device': old_meta['actual_device'], 'TrainingSeconds': old_meta['training_seconds']},
        *candidate_rows]
    selected = select_model(rows)
    if selected == 'XGBoost baseline':
        final = Pipeline([('preprocessor', copy.deepcopy(baseline['preprocessor'])), ('model', copy.deepcopy(baseline['model']))])
    else:
        final = candidates[selected]
    chosen = next(r for r in rows if r['Model'] == selected)
    # Portable CPU inference does not change trees or the recorded training device.
    final.named_steps['model'].set_params(device='cpu')
    predictions = final.predict(valid[FEATURES])
    for key, value in metrics(valid[TARGET], predictions).items():
        if not np.isclose(value, chosen[key], rtol=1e-6):
            raise ValueError('CPU inference changed validation metrics')
    joblib.dump(final, destination)
    np.testing.assert_allclose(joblib.load(destination).predict(valid[FEATURES]), predictions, rtol=0, atol=0)
    manifest = json.loads((REPORTS / 'data_split.json').read_text())
    metadata = {'algorithm': 'XGBRegressor', 'selected_model': selected,
        'features': FEATURES, 'feature_dtypes': {c: str(train[c].dtype) for c in FEATURES},
        'dataset_version': manifest['source_sha256'], 'split_manifest_sha256': sha256(REPORTS / 'data_split.json'),
        'split_hashes': {k: v['sha256'] for k, v in manifest['splits'].items()},
        'random_seed': 42, 'training_sample_count': len(train), 'validation_sample_count': len(valid),
        'fit_source_ids': train.Id.tolist(), 'validation_source_ids': valid.Id.tolist(),
        'validation_metrics': {k: chosen[k] for k in ('MAE', 'RMSE', 'R2')},
        'hyperparameters': final.named_steps['model'].get_params(),
        'training_device': chosen['Device'], 'inference_device': 'cpu',
        'training_status': 'GPU TRAINING SUCCESS' if chosen['Device'].startswith('cuda') else 'CPU TRAINING',
        'training_seconds': chosen['TrainingSeconds'], 'cv_seconds': cv_seconds,
        'total_seconds': time.perf_counter() - started, 'n_iter': n_iter, 'cv_folds': 5,
        'best_cv_parameters': search.best_params_, 'best_cv_mae': float(-search.best_score_),
        'xgboost_version': xgboost.__version__, 'sklearn_version': sklearn.__version__,
        'artifact_sha256': sha256(destination), 'neighborhoods': sorted(train.Neighborhood.dropna().unique()),
        'notebook_sha256': sha256(ROOT / 'notebooks/house_price_eda.ipynb'),
        'baseline_sha256': sha256(old_path), 'test_accessed': False, 'frozen_for_week4': True,
        'selection_rule': 'Lowest Validation MAE among baseline and top three CV candidates; RMSE tie-break.',
        'early_stopping': False}
    # JSON does not encode NaN (XGBoost's missing-value default).
    metadata['hyperparameters']['missing'] = 'NaN (XGBoost default)'
    write_json(ROOT / 'artifacts/model_metadata.json', metadata)
    pd.DataFrame(rows).to_csv(REPORTS / 'model_comparison.csv', index=False)
    pd.DataFrame({'Id': valid.Id, 'Actual': valid[TARGET], 'Predicted': predictions,
                  'Residual': valid[TARGET] - predictions, 'AbsoluteError': abs(valid[TARGET] - predictions)}).to_csv(REPORTS / 'validation_predictions.csv', index=False)
    write_json(REPORTS / 'week3_fit_audit.json', {'fits': FIT_AUDIT, 'failures': failures,
        'warnings': sorted(set(search_warnings)), 'cv_device': device,
        'folds': [{'train_indices': a.tolist(), 'heldout_indices': b.tolist()} for a,b in folds],
        'gpu_inventory': command(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used', '--format=csv']),
        'memory_note': 'Post-training snapshot; not peak allocation or per-process VRAM.'})
    (REPORTS / 'model_selection.md').write_text(
        '# Model selection\n\n' + selected + '\n\n' + metadata['selection_rule'] +
        '\n\nValidation is used for selection, so these metrics are not an unbiased final evaluation. Test remains locked until Week 4. No refit on Train+Validation.\n\n' +
        json.dumps(chosen, indent=2), encoding='utf-8')
    print(json.dumps({k: metadata[k] for k in ('selected_model', 'validation_metrics', 'training_device', 'best_cv_mae', 'total_seconds')}, indent=2))
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-iter', type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.n_iter <= 288:
        parser.error('--n-iter must be between 1 and 288')
    tune(args.n_iter)
