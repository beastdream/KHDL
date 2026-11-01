import json
import platform
import time
import traceback
import warnings
import joblib
import sklearn
import xgboost
from sklearn.dummy import DummyRegressor
from xgboost import XGBRegressor
from src.data.schema import FEATURES, NUMERIC_FEATURES, TARGET
from src.data.week2_inputs import load_split, install_data_guard
from src.models.preprocessing import build_preprocessor
from src.models.evaluate_validation import evaluate
from src.utils.check_gpu import cuda_config_ready, command
from src.utils.paths import ROOT, REPORTS, PROCESSED, sha256, write_json

PARAMS=dict(objective='reg:squarederror',n_estimators=300,max_depth=5,learning_rate=.05,
            subsample=.8,colsample_bytree=.8,tree_method='hist',device='cuda',random_state=42)


def fit_xgboost(x,y):
    attempts=[]
    for device in ['cuda','cpu']:
        record={'requested_device':device}
        start=time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                model=XGBRegressor(**{**PARAMS,'device':device})
                model.fit(x,y)
                config=json.loads(model.get_booster().save_config())
                record.update(booster_config=config,actual_device=config['learner']['generic_param']['device'])
                if device=='cuda' and not cuda_config_ready(config):
                    raise RuntimeError('GPU fit not verified: Booster silently fell back or lacks GPU updater')
                if device=='cpu' and record['actual_device']!='cpu':
                    raise RuntimeError('CPU fallback device not verified')
                record['success']=True
            except Exception:
                record.update(success=False,error=traceback.format_exc())
            record['training_seconds']=time.perf_counter()-start
            record['warnings']=[str(w.message) for w in caught]
        attempts.append(record)
        if record['success']:
            return model, attempts
    write_json(REPORTS/'week2_training_failure.json',attempts)
    raise RuntimeError('Both GPU and CPU training failed; see week2_training_failure.json')


def train():
    train_df, valid = load_split('train'), load_split('valid')
    if set(train_df.Id)&set(valid.Id):
        raise ValueError('Train/validation overlap')
    x,y=train_df[FEATURES],train_df[TARGET]
    prep=build_preprocessor()
    start=time.perf_counter()
    xt=prep.fit_transform(x)
    xv=prep.transform(valid[FEATURES])
    preprocessing_seconds=time.perf_counter()-start
    baseline=DummyRegressor(strategy='median').fit(x,y)
    model, attempts=fit_xgboost(xt,y)
    final=attempts[-1]
    metadata={'params':{**PARAMS,'device':final['requested_device']}, 'actual_device':final['actual_device'],
              'gpu_status':'GPU READY' if final['requested_device']=='cuda' else 'GPU NOT READY',
              'attempts':attempts,'training_seconds':final['training_seconds'],'preprocessing_seconds':preprocessing_seconds,
              'gpu_inventory':command(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv']),
              'nvidia_smi':command(['nvidia-smi']),
              'versions':{'python':platform.python_version(),'xgboost':xgboost.__version__,'sklearn':sklearn.__version__},
              'fit_split':'train','fit_source_ids':train_df.Id.tolist(),'validation_source_ids':valid.Id.tolist(),
              'source_hashes':{n:sha256(PROCESSED/f'{n}.csv') for n in ['train','valid']},
              'transformed_features':prep.get_feature_names_out().tolist(),
              'matrix_shapes':{'train':list(xt.shape),'valid':list(xv.shape)},
              'baseline_median':float(y.median()), 'test_accessed':False,
              'prediction_method':'Booster.predict(DMatrix); no inplace_predict device mismatch',
              'remediation':[] if len(attempts)==1 else ['Inspect CUDA exception/warnings; update compatible NVIDIA driver or CUDA-enabled XGBoost wheel. CPU fallback is labelled separately.']}
    artifact={'preprocessor':prep,'model':model,'baseline':baseline,'metadata':metadata,
              'feature_schema':{'features':FEATURES,'numeric':NUMERIC_FEATURES,'categorical':['Neighborhood'],'target':TARGET}}
    path=ROOT/'artifacts/xgboost_baseline.joblib'
    joblib.dump(artifact,path)
    metadata['artifact_sha256']=sha256(path)
    write_json(ROOT/'artifacts/xgboost_baseline.metadata.json',metadata)
    evaluate(artifact,valid)
    return artifact


if __name__=='__main__':
    install_data_guard()
    result=train()
    print(result['metadata']['gpu_status'])
