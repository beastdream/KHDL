import json
import numpy as np
import pandas as pd
import pytest
from src.models.tune_model import select_model
from src.models.inference import load_model
from src.models.evaluate_validation import metrics
from src.data.week3_inputs import load_split
from src.data.schema import FEATURES
from src.utils.paths import REPORTS, ROOT


def test_selection_uses_mae_and_can_keep_baseline():
    rows = [dict(Model='XGBoost baseline', MAE=1., RMSE=3., R2=.8),
            dict(Model='XGBoost tuned', MAE=2., RMSE=2., R2=.9)]
    assert select_model(rows) == 'XGBoost baseline'
    rows[1]['MAE'] = .5
    assert select_model(rows) == 'XGBoost tuned'
    rows[1]['MAE'] = np.nan
    with pytest.raises(ValueError):
        select_model(rows)


def test_real_selection_and_cpu_predictions():
    rows = pd.read_csv(REPORTS / 'model_comparison.csv')
    model, meta = load_model()
    assert select_model(rows.to_dict('records')) == meta['selected_model']
    assert len(rows) == 5
    valid = load_split('valid')
    measured = metrics(valid.SalePrice, model.predict(valid[FEATURES]))
    for key, value in measured.items():
        assert value == pytest.approx(meta['validation_metrics'][key], rel=1e-6)
    saved = pd.read_csv(REPORTS / 'validation_predictions.csv')
    np.testing.assert_allclose(saved.Predicted, model.predict(valid[FEATURES]), rtol=1e-6)
    assert saved.Id.tolist() == valid.Id.tolist()
    assert meta['frozen_for_week4'] and not meta['test_accessed']

