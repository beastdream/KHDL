"""Read-only dataset access, with an exact allowlist for Week 3 report CSVs."""
import os
import sys
from pathlib import Path
from src.data.week2_inputs import load_split
from src.utils.paths import PROCESSED, REPORTS, ROOT

REPORT_CSVS = {'model_comparison.csv', 'tuning_results.csv', 'validation_predictions.csv'}


def install_data_guard():
    allowed = {(PROCESSED / f'{n}.csv').resolve() for n in ('train', 'valid')}
    outputs = {(REPORTS / n).resolve() for n in REPORT_CSVS}
    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, Path)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to((ROOT / 'data/raw').resolve()):
            raise PermissionError('Raw data is locked until Week 4')
        if path.suffix.lower() == '.csv' and path not in allowed | outputs:
            raise PermissionError(f'Week 3 prohibits CSV access: {path}')
        if path in allowed:
            mode, flags = args[1:3]
            if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_TRUNC | os.O_APPEND | os.O_CREAT)):
                raise PermissionError('Dataset inputs are read-only')
    sys.addaudithook(guard)

