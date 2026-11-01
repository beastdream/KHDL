import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data.schema import FEATURES, TARGET
from src.data.week2_inputs import load_split, install_data_guard
from src.models.preprocessing import predict_artifact
from src.utils.paths import ROOT, REPORTS, write_json


def metrics(y, pred):
    pred = np.asarray(pred)
    if pred.shape != np.asarray(y).shape or not np.isfinite(pred).all():
        raise ValueError('Prediction shape or values invalid')
    return {'MAE': float(mean_absolute_error(y,pred)), 'RMSE': float(np.sqrt(mean_squared_error(y,pred))), 'R2': float(r2_score(y,pred))}


def evaluate(artifact, valid):
    x, y = valid[FEATURES], valid[TARGET]
    results = {'evaluation_split':'valid', 'rows':len(valid), 'target': TARGET, 'unit':'USD', 'models':{}}
    results['models']['Dummy Baseline'] = {**metrics(y, artifact['baseline'].predict(x)), 'Device':'CPU'}
    results['models']['XGBoost'] = {**metrics(y, predict_artifact(artifact,x)), 'Device':artifact['metadata']['actual_device']}
    baseline, model = results['models'].values()
    results['rmse_improvement_pct'] = 100*(baseline['RMSE']-model['RMSE'])/baseline['RMSE']
    write_json(REPORTS/'validation_metrics.json',results)
    lines=['# Validation metrics', '', 'Chỉ đánh giá Validation; không dùng Test.', '', '| Model | MAE | RMSE | R2 | Device |', '|---|---:|---:|---:|---|']
    for name, values in results['models'].items():
        lines.append(f"| {name} | {values['MAE']:.2f} | {values['RMSE']:.2f} | {values['R2']:.4f} | {values['Device']} |")
    lines += ['', f"RMSE giảm {results['rmse_improvement_pct']:.2f}% so với giá trung vị Train. MAE/RMSE tính bằng USD."]
    (REPORTS/'validation_metrics.md').write_text('\n'.join(lines),encoding='utf-8')
    return results


if __name__=='__main__':
    install_data_guard()
    artifact=joblib.load(ROOT/'artifacts/xgboost_baseline.joblib')
    print(json.dumps(evaluate(artifact,load_split('valid')),indent=2))
