import json
import pandas as pd
import numpy as np
from src.data.schema import validation_errors, TARGET
from src.data.split_dataset import save_split
from src.utils.paths import RAW, PROCESSED, REPORTS, write_json


def audit(raw=RAW, output=PROCESSED, reports=REPORTS):
    reports.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        result = {'status': 'NOT READY', 'error': 'Missing data/raw/train.csv; see README download instructions.'}
        write_json(reports / 'dataset_audit.json', result)
        (reports / 'dataset_audit.md').write_text('# Dataset audit\n\n' + result['error'], encoding='utf-8')
        return result
    df = pd.read_csv(raw)
    errors = validation_errors(df)
    for col in df.select_dtypes(include='number'):
        bad = df[col].notna() & ~np.isfinite(df[col])
        if bad.any():
            errors[f'{col}_nonfinite'] = df.index[bad].tolist()
    for col, lower, upper in [('LotArea', 1, None), ('MoSold', 1, 12), ('OverallCond', 1, 10)]:
        if col in df:
            values = pd.to_numeric(df[col], errors='coerce')
            bad = df[col].notna() & (values.isna() | (values < lower))
            if upper is not None:
                bad |= values > upper
            if bad.any():
                errors[col] = df.index[bad].tolist()
    if {'YearBuilt', 'YrSold'} <= set(df):
        bad = df['YearBuilt'] > df['YrSold']
        if bad.any():
            errors['built_after_sale'] = df.index[bad].tolist()
    duplicates = int(df.duplicated().sum())
    content_duplicates = int(df.drop(columns=['Id'], errors='ignore').duplicated().sum())
    # Dataset-wide checks are quality gates only, never EDA/model selection.
    area_cols = [c for c in df if c.endswith('SF') or 'Area' in c]
    negative_areas = {}
    for c in area_cols:
        s = pd.to_numeric(df[c], errors='coerce')
        bad = s < 0
        if bad.any():
            negative_areas[c] = df.index[bad].tolist()
    result = {'status': 'NOT READY' if errors or duplicates or content_duplicates or negative_areas else 'PASS',
              'rows': len(df), 'columns': len(df.columns), 'predictor_columns_excluding_id_target': len(df.columns)-2,
              'dtypes': {c: str(t) for c, t in df.dtypes.items()},
              'missing': df.isna().sum().to_dict(), 'duplicate_rows': duplicates,
              'duplicate_content_excluding_id': content_duplicates, 'invalid_records': errors,
              'negative_areas': negative_areas, 'rows_removed': 0,
              'policy': 'Full-source quality checks only. Descriptive statistics/outliers/plots use TRAIN only; holdout excluded.'}
    if result['status'] == 'PASS':
        save_split(raw, output, reports / 'data_split.json')
        train = pd.read_csv(output / 'train.csv')
        stats = train.select_dtypes(include='number').agg(['min','max','mean','median','std']).T
        result['train_numeric_statistics'] = json.loads(stats.to_json(orient='index'))
        result['train_target_distribution'] = json.loads(train[TARGET].describe(percentiles=[.01,.25,.5,.75,.99]).to_json())
        result['train_target_skew'] = float(train[TARGET].skew())
        flagged = (train['GrLivArea'] > 4000) | (train['LotArea'] > 100000)
        result['train_area_review'] = train.loc[flagged, ['Id','GrLivArea','LotArea',TARGET]].to_dict('records')
        q1, q3 = train[TARGET].quantile([.25,.75])
        result['train_target_iqr_outlier_ids'] = train.loc[(train[TARGET] < q1-1.5*(q3-q1)) | (train[TARGET] > q3+1.5*(q3-q1)), 'Id'].tolist()
        result['outlier_policy'] = 'Review flags only (GrLivArea >4000, LotArea >100000 sq ft; target 1.5 IQR). No deletion.'
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11,4))
        axes[0].hist(train[TARGET], bins=35)
        axes[0].set(xlabel='SalePrice (USD)', ylabel='Train count', title='Train target distribution')
        axes[1].scatter(train['GrLivArea'], train[TARGET], alpha=.4, s=12)
        axes[1].set(xlabel='GrLivArea (sq ft)', ylabel='SalePrice (USD)', title='Train only: area and price')
        fig.tight_layout()
        (reports / 'figures').mkdir(exist_ok=True)
        fig.savefig(reports / 'figures/train_distribution.png', dpi=150)
        plt.close(fig)
    write_json(reports / 'dataset_audit.json', result)
    lines = ['# Dataset audit', '', f"Status: {result['status']}", '', result['policy'], '',
             f'Rows: {len(df)}; columns: {len(df.columns)}; exact duplicates: {duplicates}; content duplicates: {content_duplicates}.',
             f'Invalid records: {errors}. Negative areas: {negative_areas}. No rows removed.', '',
             '| Column | Dtype | Missing |', '|---|---|---:|']
    lines += [f"| {c} | {df[c].dtype} | {df[c].isna().sum()} |" for c in df]
    if 'train_numeric_statistics' in result:
        lines += ['', '## TRAIN-only statistics', '', '| Column | Min | Max | Mean | Median | Std |', '|---|---:|---:|---:|---:|---:|']
        for c, values in result['train_numeric_statistics'].items():
            lines.append('| ' + c + ' | ' + ' | '.join(str(values[k]) for k in ['min','max','mean','median','std']) + ' |')
        lines += ['', result['outlier_policy'], '', 'Area flags: ' + str(result['train_area_review']), '',
                  'Target IQR flags (source IDs): ' + str(result['train_target_iqr_outlier_ids']), '',
                  'Target summary: ' + str(result['train_target_distribution']), '',
                  '![Train distributions](figures/train_distribution.png)']
    (reports / 'dataset_audit.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


if __name__ == '__main__':
    print(audit()['status'])
