"""Final phase guard: holdout readable only inside the evaluator's scoped capability.

Python audit hooks enforce accidental-access policy, not a hostile-code sandbox.
Legacy Week 2/3 hooks stay unchanged and run in separate processes.
"""
from contextvars import ContextVar
from contextlib import contextmanager
import inspect
import os
from pathlib import Path
import sys
from src.utils.paths import ROOT, PROCESSED, REPORTS

_access = ContextVar('final_holdout_read', default=False)
_installed = False


def install_data_guard():
    global _installed
    if _installed:
        return
    train_valid = {(PROCESSED / f'{n}.csv').resolve() for n in ('train', 'valid')}
    holdout = (PROCESSED / 'test.csv').resolve()
    report_csvs = {(REPORTS / n).resolve() for n in
                  ('model_comparison.csv', 'tuning_results.csv', 'validation_predictions.csv')}
    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, Path)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1:3]
        writing = ((isinstance(mode, str) and any(c in mode for c in 'wax+')) or
                   (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_TRUNC | os.O_APPEND | os.O_CREAT)))
        if path.is_relative_to((ROOT / 'data/raw').resolve()):
            raise PermissionError('Final phase does not reopen raw data')
        if path == holdout:
            if writing or not _access.get():
                raise PermissionError('Holdout requires scoped Final Evaluation access; writes forbidden')
        elif path.suffix.lower() == '.csv' and path not in train_valid | report_csvs:
            raise PermissionError(f'CSV outside final allowlist: {path}')
        if path in train_valid and writing:
            raise PermissionError('Frozen dataset is read-only')
    sys.addaudithook(guard)
    _installed = True


@contextmanager
def evaluation_access():
    caller = inspect.currentframe().f_back
    # contextlib enters generators through __enter__; examine the external caller.
    caller = caller.f_back
    expected = (ROOT / 'src/models/evaluate_test.py').resolve()
    if Path(caller.f_code.co_filename).resolve() != expected:
        raise PermissionError('Only src.models.evaluate_test may open holdout capability')
    token = _access.set(True)
    try:
        yield
    finally:
        _access.reset(token)
