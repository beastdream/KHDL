"""Huấn luyện và so sánh mô hình dự đoán giá nhà TP.HCM.

- So sánh: Dummy (trung vị), Linear Regression, Random Forest, XGBoost.
- Mọi mô hình học trên log(giá) vì giá nhà lệch phải mạnh; kết quả đổi lại về tỷ VNĐ.
- Tiền xử lý nằm trong Pipeline nên chỉ học từ tập Train (không rò rỉ dữ liệu).
- Tinh chỉnh XGBoost bằng 5-fold CV trên Train; chọn mô hình theo MAE trên Validation;
  Test chỉ dùng MỘT lần cho mô hình đã chọn.

Chạy: python -m src.models.vn_train
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.data.vn_clean import FEATURES, NUMERIC, CATEGORICAL, BINARY, TARGET
from src.utils.paths import ROOT, REPORTS, PROCESSED, write_json

SEED = 42
ARTIFACT = ROOT / 'artifacts/vn_house_price_model.joblib'


def load(name):
    return pd.read_csv(PROCESSED / f'vn_{name}.csv')


def preprocessor(scale):
    numeric = [('imputer', SimpleImputer(strategy='median', add_indicator=True))]
    if scale:
        numeric.append(('scaler', StandardScaler()))
    return ColumnTransformer([
        ('num', Pipeline(numeric), NUMERIC),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Không rõ')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore', min_frequency=5))]), CATEGORICAL),
        ('bin', 'passthrough', BINARY)])


def log_model(prep, model):
    return TransformedTargetRegressor(regressor=Pipeline([('prep', prep), ('model', model)]),
                                      func=np.log, inverse_func=np.exp)


def metrics(y, pred):
    return {'MAE': float(mean_absolute_error(y, pred)),
            'RMSE': float(np.sqrt(mean_squared_error(y, pred))),
            'R2': float(r2_score(y, pred)),
            'MAPE_%': float(np.mean(np.abs(y - pred) / y) * 100),
            'MedAPE_%': float(np.median(np.abs(y - pred) / y) * 100)}


def candidates(x, y):
    models = {
        'Dummy (trung vị)': DummyRegressor(strategy='median'),
        'Linear Regression (Ridge)': log_model(preprocessor(True), Ridge(alpha=1.0)),
        'Random Forest': log_model(preprocessor(False), RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=SEED)),
    }
    xgb = log_model(preprocessor(False), XGBRegressor(tree_method='hist', device='cpu',
                                                      random_state=SEED, n_jobs=-1))
    space = {f'regressor__model__{k}': v for k, v in {
        'n_estimators': [300, 600, 1000], 'max_depth': [4, 6, 8], 'learning_rate': [0.03, 0.05, 0.1],
        'subsample': [0.8, 1.0], 'colsample_bytree': [0.7, 0.9], 'min_child_weight': [1, 3, 5],
        'reg_lambda': [1, 5]}.items()}
    search = RandomizedSearchCV(xgb, space, n_iter=25, cv=KFold(5, shuffle=True, random_state=SEED),
                                scoring='neg_mean_absolute_error', random_state=SEED, n_jobs=1)
    search.fit(x, y)
    models['XGBoost (tinh chỉnh CV)'] = search.best_estimator_
    return models, search


def main():
    train, valid, test = load('train'), load('valid'), load('test')
    xtr, ytr = train[FEATURES], train[TARGET]
    xva, yva = valid[FEATURES], valid[TARGET]
    started = time.perf_counter()
    models, search = candidates(xtr, ytr)
    rows = []
    for name, model in models.items():
        if name != 'XGBoost (tinh chỉnh CV)':      # mô hình tinh chỉnh đã được fit khi search
            model.fit(xtr, ytr)
        rows.append({'Mô hình': name, **metrics(yva, model.predict(xva))})
    comparison = pd.DataFrame(rows)
    eligible = comparison[comparison['Mô hình'] != 'Dummy (trung vị)']
    selected = eligible.sort_values(['MAE', 'RMSE']).iloc[0]['Mô hình']
    final = models[selected]

    # Đánh giá Test đúng một lần, sau khi đã chốt mô hình.
    test_pred = final.predict(test[FEATURES])
    test_metrics = metrics(test[TARGET], test_pred)
    dummy_test = metrics(test[TARGET], models['Dummy (trung vị)'].predict(test[FEATURES]))

    bins = [0, 3, 5, 8, 12, 20, np.inf]
    labels = ['<3 tỷ', '3–5 tỷ', '5–8 tỷ', '8–12 tỷ', '12–20 tỷ', '≥20 tỷ']
    err = pd.DataFrame({'y': test[TARGET], 'p': test_pred})
    err['nhóm'] = pd.cut(err.y, bins, labels=labels, right=False)
    by_range = err.groupby('nhóm', observed=True).apply(
        lambda g: pd.Series({'n': len(g), 'MAE (tỷ)': (g.y - g.p).abs().mean(),
                             'MAPE %': ((g.y - g.p).abs() / g.y).mean() * 100}), include_groups=False)

    imp = permutation_importance(final, xva, yva, scoring='neg_mean_absolute_error',
                                 n_repeats=10, random_state=SEED)
    importance = pd.Series(imp.importances_mean, index=FEATURES).sort_values(ascending=False)

    joblib.dump(final, ARTIFACT)
    meta = {'selected_model': selected, 'features': FEATURES, 'numeric': NUMERIC,
            'categorical': CATEGORICAL, 'binary': BINARY, 'target': TARGET, 'unit': 'tỷ VNĐ',
            'train_rows': len(train), 'valid_rows': len(valid), 'test_rows': len(test),
            'validation': comparison.set_index('Mô hình').to_dict('index'),
            'test': test_metrics, 'test_dummy': dummy_test,
            'best_xgb_params': {k.replace('regressor__model__', ''): v for k, v in search.best_params_.items()},
            'best_xgb_cv_mae': float(-search.best_score_),
            'input_ranges': {c: [float(train[c].quantile(0.01)), float(train[c].quantile(0.99))]
                             for c in ['size', 'floors', 'rooms', 'toilets', 'width', 'length']},
            'districts': sorted(train['district'].dropna().unique().tolist()),
            'choices': {c: sorted(train[c].dropna().unique().tolist()) for c in CATEGORICAL if c != 'district'},
            'district_centers': train.groupby('district')[['latitude', 'longitude']].median().round(4)
                                     .dropna().to_dict('index'),
            'seconds': time.perf_counter() - started}
    write_json(ROOT / 'artifacts/vn_model_metadata.json', meta)
    comparison.to_csv(REPORTS / 'vn_model_comparison.csv', index=False, encoding='utf-8')

    fmt = lambda v: f'{v:,.3f}'
    lines = ['# Mô hình dự đoán giá nhà TP.HCM', '',
             f"Dữ liệu: {len(train):,} Train / {len(valid):,} Validation / {len(test):,} Test (xem vn_data_funnel.md). "
             'Đơn vị giá: tỷ VNĐ. Mọi mô hình (trừ Dummy) học trên log(giá).', '',
             '## So sánh trên Validation', '', comparison.to_markdown(index=False, floatfmt='.3f'), '',
             f'**Mô hình được chọn:** {selected} (MAE Validation thấp nhất).', '',
             f"XGBoost tốt nhất qua 5-fold CV trên Train: MAE CV = {fmt(-search.best_score_)} tỷ, tham số {meta['best_xgb_params']}.", '',
             '## Đánh giá cuối trên Test (chạy một lần)', '',
             '| Chỉ số | Mô hình chọn | Dummy |', '|---|---:|---:|']
    lines += [f'| {k} | {fmt(test_metrics[k])} | {fmt(dummy_test[k])} |' for k in test_metrics]
    lines += ['', '## Sai số theo khoảng giá (Test)', '', by_range.to_markdown(floatfmt='.2f'), '',
              '## Permutation importance trên Validation (mức tăng MAE, tỷ VNĐ)', '',
              importance.to_frame('Tăng MAE').to_markdown(floatfmt='.3f'), '',
              '## Nhận xét', '',
              '- Linear Regression có MAPE chấp nhận được nhưng RMSE rất lớn và R² âm: mô hình tuyến tính trên '
              'log(giá) cho vài căn diện tích lớn dự đoán quá cao, và khi đổi ngược exp() thì sai số bị khuếch đại. '
              'Mô hình cây (Random Forest, XGBoost) không ngoại suy nên ổn định hơn.',
              '- Sai số phần trăm cao nhất ở nhóm <3 tỷ và ≥20 tỷ: hai đầu phân phối có ít mẫu và nhiều tin nhiễu hơn.',
              '- Diện tích, loại nhà và vị trí (toạ độ, quận) quyết định phần lớn giá; hướng nhà và nội thất gần như không ảnh hưởng.', '',
              '## Giới hạn', '',
              '- Giá là **giá rao bán**, không phải giá giao dịch thật; thường cao hơn giá chốt.',
              '- Dữ liệu là ảnh chụp tại một thời điểm thu thập; không phản ánh biến động theo thời gian.',
              '- Tin đăng do người dùng tự nhập nên vẫn còn nhiễu sau khi làm sạch.']
    (REPORTS / 'vn_model_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(comparison.to_string(index=False))
    print('Selected:', selected, '| Test:', json.dumps(test_metrics))


if __name__ == '__main__':
    main()
