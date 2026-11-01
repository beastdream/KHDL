import numpy as np
import pandas as pd

FEATURES = ['GrLivArea', 'Neighborhood', 'BedroomAbvGr', 'FullBath',
            'GarageCars', 'YearBuilt', 'OverallQual', 'TotalBsmtSF']
TARGET = 'SalePrice'
SOURCE_ID = 'Id'
NUMERIC_FEATURES = [c for c in FEATURES if c != 'Neighborhood']


def validation_errors(df):
    errors = {}
    missing = sorted(set(FEATURES + [TARGET, SOURCE_ID]) - set(df.columns))
    if missing:
        return {'missing_columns': missing}
    if df.empty:
        errors['empty_dataset'] = True
    ids = pd.to_numeric(df[SOURCE_ID], errors='coerce')
    bad_ids = ids.isna() | ~np.isfinite(ids) | (ids <= 0) | (ids % 1 != 0) | ids.duplicated(keep=False)
    if bad_ids.any():
        errors['invalid_or_duplicate_ids'] = df.index[bad_ids].tolist()
    for col in NUMERIC_FEATURES + [TARGET]:
        s = pd.to_numeric(df[col], errors='coerce')
        bad = df[col].notna() & (s.isna() | ~np.isfinite(s))
        if col == TARGET:
            bad |= s.isna() | (s <= 0)
        elif col == 'GrLivArea':
            bad |= s <= 0
        elif col == 'OverallQual':
            bad |= (s < 1) | (s > 10)
        elif col == 'YearBuilt':
            bad |= (s < 1) | (s > 2026)
        else:
            bad |= s < 0
        if col != TARGET:
            bad |= s.notna() & (s % 1 != 0)
        if bad.any():
            errors[col] = df.index[bad].tolist()
    category = df['Neighborhood']
    bad = category.notna() & ~category.map(lambda v: isinstance(v, str) and bool(v.strip()))
    if bad.any():
        errors['Neighborhood'] = df.index[bad].tolist()
    return errors


def validate(df):
    errors = validation_errors(df)
    if errors:
        raise ValueError(f'Invalid dataset; no rows removed automatically: {errors}')


def xy(df):
    validate(df)
    return df[FEATURES].copy(), df[TARGET].copy()
