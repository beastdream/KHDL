from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/raw/train.csv'
PROCESSED = ROOT / 'data/processed'
REPORTS = ROOT / 'reports'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
