"""Reusable preprocessing and schema-checked artifact inference."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import DMatrix
from src.data.schema import FEATURES, NUMERIC_FEATURES


def build_preprocessor():
    return ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median'), NUMERIC_FEATURES),
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), ['Neighborhood'])
    ], remainder='drop', sparse_threshold=0)


def check_features(frame):
    if not isinstance(frame, pd.DataFrame) or frame.columns.duplicated().any() or set(frame.columns) != set(FEATURES):
        raise ValueError('Provide exactly the eight feature columns; no target or Id.')
    frame = frame.loc[:, FEATURES].copy()
    for c in NUMERIC_FEATURES:
        if not pd.api.types.is_numeric_dtype(frame[c]) or np.isinf(frame[c].to_numpy(dtype=float)).any():
            raise ValueError(f'Invalid numeric feature: {c}')
    if not frame.Neighborhood.dropna().map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError('Invalid Neighborhood')
    return frame


def predict_artifact(artifact, features):
    if artifact['feature_schema']['features'] != FEATURES:
        raise ValueError('Artifact schema mismatch')
    features = check_features(features)
    matrix = artifact['preprocessor'].transform(features)
    # Explicit DMatrix supports host preprocessing without inplace_predict device mismatch.
    pred = artifact['model'].get_booster().predict(DMatrix(matrix))
    if pred.shape != (len(features),) or not np.isfinite(pred).all():
        raise ValueError('Invalid predictions')
    return pred
