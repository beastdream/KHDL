"""Week 2 data access: only immutable train and validation inputs."""
import json
import sys
from pathlib import Path
import pandas as pd
from src.data.schema import validate
from src.utils.paths import ROOT, PROCESSED, REPORTS, sha256


def install_data_guard():
    allowed = {(PROCESSED / f'{name}.csv').resolve() for name in ('train', 'valid')}
    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, Path)):
            return
        path = Path(args[0]).resolve()
        if path.suffix.lower() == '.csv' and path not in allowed:
            raise PermissionError(f'Week 2 prohibits CSV access: {path}')
        if path in allowed:
            mode, flags = args[1:3]
            if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags & 3):
                raise PermissionError('Week 2 inputs are read-only')
    sys.addaudithook(guard)


def load_split(name):
    if name not in ('train', 'valid'):
        raise ValueError('Week 2 allows only train and valid')
    path = PROCESSED / f'{name}.csv'
    manifest = json.loads((REPORTS / 'data_split.json').read_text(encoding='utf-8'))
    expected = manifest['splits'][name]
    if sha256(path) != expected['sha256']:
        raise ValueError(f'{name} input hash differs from Week 1')
    df = pd.read_csv(path)
    validate(df)
    if len(df) != expected['count'] or df.Id.tolist() != expected['source_ids']:
        raise ValueError('Split membership mismatch')
    return df
