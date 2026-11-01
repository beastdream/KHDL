import json
import joblib
import numpy as np
import pytest
from src.models import train_model
from src.utils.check_gpu import cuda_config_ready
from src.utils.paths import ROOT


def test_actual_training_gpu_evidence():
    metadata=json.loads((ROOT/'artifacts/xgboost_baseline.metadata.json').read_text())
    attempt=metadata['attempts'][-1]
    assert attempt['success']
    confirmed=cuda_config_ready(attempt['booster_config'])
    assert (metadata['gpu_status']=='GPU READY')==confirmed
    assert metadata['actual_device']==attempt['actual_device']
    assert metadata['training_seconds']>0
    loaded=joblib.load(ROOT/'artifacts/xgboost_baseline.joblib')
    live=json.loads(loaded['model'].get_booster().save_config())
    assert live['learner']['generic_param']['device']==metadata['actual_device']
    assert cuda_config_ready(live)==confirmed
    if not confirmed:
        assert metadata['actual_device']=='cpu'
        assert metadata['attempts'][0]['error'] and metadata['remediation']


@pytest.mark.parametrize('failure',['exception','silent_cpu'])
def test_gpu_failure_explicit_cpu_fallback(monkeypatch,failure):
    class FakeModel:
        def __init__(self,**kwargs): self.device=kwargs['device']
        def fit(self,x,y):
            if self.device=='cuda' and failure=='exception': raise RuntimeError('CUDA unavailable test')
        def get_booster(self): return self
        def save_config(self): return json.dumps({'learner':{'generic_param':{'device':'cpu'},'gradient_booster':{}}})
    monkeypatch.setattr(train_model,'XGBRegressor',FakeModel)
    _,attempts=train_model.fit_xgboost(np.zeros((4,2)),np.ones(4))
    assert len(attempts)==2
    assert not attempts[0]['success'] and attempts[0]['error']
    assert attempts[1]['success'] and attempts[1]['actual_device']=='cpu'
