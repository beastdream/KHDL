"""Trusted local artifact loading and strict application input validation."""
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import joblib
import numpy as np
import pandas as pd
from src.data.schema import FEATURES, NUMERIC_FEATURES
from src.models.preprocessing import check_features
from src.utils.paths import ROOT, sha256


def load_model():
    path = ROOT / 'artifacts/house_price_model.joblib'
    metadata = json.loads((ROOT / 'artifacts/model_metadata.json').read_text(encoding='utf-8'))
    if sha256(path) != metadata['artifact_sha256']:
        raise ValueError('Artifact checksum mismatch')
    if metadata['features'] != FEATURES:
        raise ValueError('Artifact feature schema mismatch')
    model = joblib.load(path)
    model.named_steps['model'].set_params(device='cpu')
    return model, metadata


def validate_inputs(frame, metadata):
    frame = check_features(frame)
    if frame.empty or frame.isna().any().any():
        raise ValueError('Vui lòng nhập đủ 8 thuộc tính cho ít nhất một căn nhà.')
    for column in NUMERIC_FEATURES:
        values = frame[column].to_numpy(dtype=float)
        if pd.api.types.is_bool_dtype(frame[column]) or not np.isfinite(values).all() or (values % 1 != 0).any():
            raise ValueError(f'{column}: cần số nguyên hữu hạn.')
    year = datetime.now(ZoneInfo('Asia/Bangkok')).year
    bounds = {'GrLivArea': (1, 20000), 'TotalBsmtSF': (0, 20000),
              'BedroomAbvGr': (0, 50), 'FullBath': (0, 50), 'GarageCars': (0, 50),
              'YearBuilt': (1800, year), 'OverallQual': (1, 10)}
    for column, (low, high) in bounds.items():
        if not frame[column].between(low, high).all():
            raise ValueError(f'{column}: giá trị hợp lệ từ {low} đến {high}.')
    if not frame.Neighborhood.isin(metadata['neighborhoods']).all():
        raise ValueError('Khu vực không nằm trong danh sách được hỗ trợ.')
    return frame


def predict(frame, model=None, metadata=None):
    if model is None or metadata is None:
        model, metadata = load_model()
    frame = validate_inputs(frame, metadata)
    values = np.asarray(model.predict(frame))
    if values.shape != (len(frame),) or not np.isfinite(values).all():
        raise ValueError('Mô hình trả về kết quả không hợp lệ.')
    return values

