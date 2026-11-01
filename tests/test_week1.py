import json
import math
from pathlib import Path
import pandas as pd
import pytest
from src.data.schema import FEATURES, TARGET, validate, xy
from src.data.split_dataset import split_frames, save_split
from src.utils.paths import RAW, PROCESSED, REPORTS, sha256
from src.utils.check_gpu import cuda_config_ready, smoke


@pytest.fixture
def raw():
    if not RAW.exists():
        pytest.skip('Real Kaggle CSV unavailable; data readiness remains NOT READY.')
    return pd.read_csv(RAW)


def test_csv_schema_and_target(raw):
    validate(raw)
    x, y = xy(raw)
    assert len(FEATURES) == len(set(FEATURES)) == 8
    assert x.columns.tolist() == FEATURES
    assert TARGET not in x and 'Id' not in x
    assert y.name == TARGET and len(y) == len(x)


@pytest.mark.parametrize('size', [20, 101, 1460])
def test_split_counts_disjoint_reproducible(raw, size):
    frame = raw.iloc[:size]
    first, second = split_frames(frame), split_frames(frame)
    n = len(frame)
    rest = math.ceil(n * .3)
    assert [len(first[k]) for k in ('train','valid','test')] == [n-rest, rest-math.ceil(rest*.5), math.ceil(rest*.5)]
    assert sum(map(len, first.values())) == n
    seen = set()
    contents = set()
    for name, part in first.items():
        assert abs(len(part) / n - {'train':.7,'valid':.15,'test':.15}[name]) <= 1 / n
        ids = set(part.Id)
        hashes = set(pd.util.hash_pandas_object(part.drop(columns='Id'), index=False))
        assert not seen & ids and not contents & hashes
        seen |= ids
        contents |= hashes
        pd.testing.assert_frame_equal(part, second[name])
    assert seen == set(frame.Id)


@pytest.mark.parametrize('kind', ['target', 'id', 'area', 'missing_column', 'duplicate_content'])
def test_invalid_data_rejected(raw, kind):
    df = raw.head(20).copy()
    if kind == 'target': df.loc[0, TARGET] = -1
    elif kind == 'id': df.loc[0, 'Id'] = df.loc[1, 'Id']
    elif kind == 'area': df.loc[0, 'GrLivArea'] = -1
    elif kind == 'missing_column': df = df.drop(columns='FullBath')
    else:
        df.iloc[0, 1:] = df.iloc[1, 1:]
    with pytest.raises(ValueError):
        split_frames(df)


def test_missing_predictors_preserved(raw):
    frame = raw.head(20).copy()
    frame.loc[0,'GarageCars'] = float('nan')
    parts = split_frames(frame)
    assert sum(p.GarageCars.isna().sum() for p in parts.values()) == 1


def test_freeze_and_tamper_detection(raw, tmp_path):
    source = tmp_path / 'raw.csv'
    raw.head(30).to_csv(source, index=False)
    out, manifest = tmp_path / 'processed', tmp_path / 'split.json'
    original = save_split(source, out, manifest)
    test = out / 'test.csv'
    before = test.read_bytes()
    assert save_split(source, out, manifest) == original
    assert test.read_bytes() == before
    test.write_bytes(before + b'\n')
    with pytest.raises(ValueError, match='Integrity'):
        save_split(source, out, manifest)
    assert test.read_bytes() == before + b'\n'
    test.write_bytes(before)
    source.write_bytes(source.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='Source changed'):
        save_split(source, out, manifest)


def test_persisted_artifacts(raw):
    manifest = json.loads((REPORTS / 'data_split.json').read_text(encoding='utf-8'))
    assert manifest['source_sha256'] == sha256(RAW)
    expected = split_frames(raw)
    for name, info in manifest['splits'].items():
        path = PROCESSED / f'{name}.csv'
        assert sha256(path) == info['sha256']
        stored = pd.read_csv(path)
        assert len(stored) == info['count']
        assert stored.Id.tolist() == info['source_ids'] == expected[name].Id.tolist()
        pd.testing.assert_frame_equal(stored, expected[name].reset_index(drop=True))


@pytest.mark.parametrize('device,updater,ready', [('cpu','grow_quantile_histmaker',False),('cuda:0','grow_gpu_hist',True),('cuda:0','grow_quantile_histmaker',False)])
def test_gpu_configuration_detection(device, updater, ready):
    config = {'learner': {'generic_param': {'device':device}, 'gradient_booster': {'updater':updater}}}
    assert cuda_config_ready(config) is ready


def test_gpu_silent_cpu_fallback_is_failure(monkeypatch):
    import xgboost
    class CPUModel:
        def __init__(self, **kwargs): pass
        def fit(self, x, y): return self
        def get_booster(self): return self
        def save_config(self):
            return json.dumps({'learner': {'generic_param': {'device':'cpu'}, 'gradient_booster': {}}})
    monkeypatch.setattr(xgboost, 'XGBRegressor', CPUModel)
    result = smoke('cuda')
    assert not result['success'] and result['actual_device'] == 'cpu'


def test_gpu_exception_reported(monkeypatch):
    import xgboost
    def fail(**kwargs):
        raise RuntimeError('deliberate test failure')
    monkeypatch.setattr(xgboost, 'XGBRegressor', fail)
    result = smoke('cuda')
    assert not result['success'] and 'deliberate test failure' in result['error']


def test_actual_gpu_audit_truthful():
    report = json.loads((REPORTS / 'gpu_audit.json').read_text(encoding='utf-8'))
    result = report['gpu_smoke']
    confirmed = result['success'] and cuda_config_ready(result.get('booster_config', {}))
    assert (report['status'] == 'GPU READY') == confirmed
    if not confirmed:
        assert result.get('error')


def test_missing_source_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match='download Kaggle'):
        save_split(tmp_path / 'missing.csv', tmp_path / 'out', tmp_path / 'manifest.json')


def test_untracked_holdout_never_overwritten(raw, tmp_path):
    source = tmp_path / 'raw.csv'
    raw.head(20).to_csv(source, index=False)
    out = tmp_path / 'out'
    out.mkdir()
    frozen = out / 'test.csv'
    frozen.write_bytes(b'original holdout')
    with pytest.raises(ValueError, match='Untracked'):
        save_split(source, out, tmp_path / 'manifest.json')
    assert frozen.read_bytes() == b'original holdout'


def test_audit_scope_and_no_removal(raw):
    report = json.loads((REPORTS / 'dataset_audit.json').read_text(encoding='utf-8'))
    assert report['status'] == 'PASS'
    assert report['rows'] == len(raw) and report['columns'] == len(raw.columns)
    assert report['rows_removed'] == 0
    train = pd.read_csv(PROCESSED / 'train.csv')
    assert report['train_target_distribution']['count'] == len(train)
    assert report['train_numeric_statistics']['SalePrice']['mean'] == pytest.approx(train.SalePrice.mean())
    assert (REPORTS / 'figures/train_distribution.png').is_file()
