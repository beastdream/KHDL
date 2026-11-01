import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data.schema import validate, SOURCE_ID
from src.utils.paths import RAW, PROCESSED, REPORTS, sha256, write_json


def split_frames(df):
    validate(df)
    # Reject duplicate property records even if their IDs differ; require review.
    if df.drop(columns=[SOURCE_ID]).duplicated().any():
        raise ValueError('Duplicate content records require review before splitting.')
    train, rest = train_test_split(df, test_size=0.30, random_state=42)
    valid, test = train_test_split(rest, test_size=0.50, random_state=42)
    return {'train': train, 'valid': valid, 'test': test}


def save_split(raw=RAW, output=PROCESSED, manifest=REPORTS / 'data_split.json'):
    raw, output, manifest = map(Path, (raw, output, manifest))
    if not raw.exists():
        raise FileNotFoundError(f'{raw}: download Kaggle House Prices train.csv (see README.md).')
    frames = split_frames(pd.read_csv(raw))
    # Validate every existing artifact before writing anything. Never silently reset a holdout.
    existing = manifest.exists() or any((output / f'{n}.csv').exists() for n in frames)
    if existing:
        if not manifest.exists():
            raise ValueError('Untracked split files exist; refusing overwrite.')
        old = json.loads(manifest.read_text(encoding='utf-8'))
        if old['source_sha256'] != sha256(raw):
            raise ValueError('Source changed; frozen split cannot be overwritten.')
        for name, frame in frames.items():
            path = output / f'{name}.csv'
            if not path.exists() or sha256(path) != old['splits'][name]['sha256']:
                raise ValueError(f'Integrity failure: {name}.csv')
            if frame[SOURCE_ID].tolist() != old['splits'][name]['source_ids']:
                raise ValueError('Split membership changed.')
        return old
    output.mkdir(parents=True, exist_ok=True)
    info = {'seed': 42, 'source_sha256': sha256(raw), 'source_file': str(raw),
            'rows': sum(map(len, frames.values())), 'preprocessing_fit': False,
            'test_policy': 'Frozen until final Week 4 evaluation; no EDA or tuning.', 'splits': {}}
    for name, frame in frames.items():
        path = output / f'{name}.csv'
        frame.to_csv(path, index=False)
        info['splits'][name] = {'count': len(frame), 'fraction': len(frame) / info['rows'],
            'source_indices': frame.index.tolist(), 'source_ids': frame[SOURCE_ID].tolist(),
            'sha256': sha256(path)}
    write_json(manifest, info)
    return info


if __name__ == '__main__':
    print(json.dumps(save_split(), indent=2))

