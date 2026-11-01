import joblib
import numpy as np
import pytest
from src.models.inference import load_model, predict
from src.data.week3_inputs import load_split
from src.data.schema import FEATURES


@pytest.fixture(scope='module')
def loaded():
    return load_model()


@pytest.mark.parametrize('count', [1, 7])
def test_inference_batch_cpu_and_preprocessor_immutable(loaded, count):
    model, meta = loaded
    frame = load_split('train')[FEATURES].head(count)
    before = joblib.hash(model.named_steps['preprocessor'])
    values = predict(frame, model, meta)
    assert values.shape == (count,) and np.isfinite(values).all()
    assert model.named_steps['model'].get_params()['device'] == 'cpu'
    assert joblib.hash(model.named_steps['preprocessor']) == before


@pytest.mark.parametrize('column,value', [('GrLivArea', -1), ('GrLivArea', 0), ('GrLivArea', 20001),
    ('BedroomAbvGr', -1), ('FullBath', 1.5), ('GarageCars', np.inf), ('YearBuilt', 1700),
    ('YearBuilt', 3000), ('OverallQual', 11), ('TotalBsmtSF', -1), ('Neighborhood', 'UNKNOWN'),
    ('Neighborhood', None), ('GrLivArea', np.nan), ('FullBath', True)])
def test_reject_invalid_inputs(loaded, column, value):
    model, meta = loaded
    frame = load_split('train')[FEATURES].head(1).copy()
    frame[column] = value
    with pytest.raises(ValueError):
        predict(frame, model, meta)


def test_reject_missing_and_extra_fields(loaded):
    model, meta = loaded
    frame = load_split('train')[FEATURES].head(1)
    with pytest.raises(ValueError):
        predict(frame.drop(columns='FullBath'), model, meta)
    with pytest.raises(ValueError):
        predict(frame.assign(SalePrice=100), model, meta)

