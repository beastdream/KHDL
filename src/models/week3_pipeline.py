"""Sklearn-compatible pipeline with per-fit data and device evidence."""
import json
import time
import warnings
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from src.models.preprocessing import build_preprocessor
from src.utils.check_gpu import cuda_config_ready

FIT_AUDIT = []


class AuditedPipeline(Pipeline):
    def fit(self, X, y=None, **params):
        record = {'row_indices': X.index.tolist(), 'rows': len(X)}
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                super().fit(X, y, **params)
                model = self.named_steps['model']
                config = json.loads(model.get_booster().save_config())
                device = config['learner']['generic_param']['device']
                record.update(device=device, booster_config=config,
                              numeric_medians=self.named_steps['preprocessor'].named_transformers_['numeric'].statistics_.tolist())
                if model.get_params()['device'] == 'cuda' and not cuda_config_ready(config):
                    raise RuntimeError('Requested CUDA but GPU fit was not verified')
                record['success'] = True
            except Exception as exc:
                record.update(success=False, error=repr(exc))
                raise
            finally:
                record.update(seconds=time.perf_counter() - start, warnings=[str(w.message) for w in caught])
                FIT_AUDIT.append(record)
        self.fit_evidence_ = record
        return self


def build_pipeline(device='cuda'):
    return AuditedPipeline([
        ('preprocessor', build_preprocessor()),
        ('model', XGBRegressor(objective='reg:squarederror', tree_method='hist',
                               device=device, random_state=42, n_jobs=1))
    ])

