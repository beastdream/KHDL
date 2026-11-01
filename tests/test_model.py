import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data.schema import FEATURES, NUMERIC_FEATURES, TARGET
from src.data.week2_inputs import load_split
from src.models.preprocessing import predict_artifact, build_preprocessor
from src.utils.paths import ROOT, REPORTS, sha256


@pytest.fixture(scope='module')
def artifact():
    return joblib.load(ROOT/'artifacts/xgboost_baseline.joblib')


def test_artifact_fit_and_schema(artifact):
    assert artifact['feature_schema']['features']==FEATURES
    assert artifact['model'].get_booster().num_boosted_rounds()==300
    assert artifact['preprocessor'].feature_names_in_.tolist()==FEATURES
    assert artifact['model'].n_features_in_==len(artifact['preprocessor'].get_feature_names_out())
    metadata=json.loads((ROOT/'artifacts/xgboost_baseline.metadata.json').read_text())
    assert sha256(ROOT/'artifacts/xgboost_baseline.joblib')==metadata['artifact_sha256']


def test_predictions_and_reload(artifact):
    frame=load_split('valid')[FEATURES]
    pred=predict_artifact(artifact,frame)
    assert pred.shape==(len(frame),) and np.issubdtype(pred.dtype,np.number)
    assert np.isfinite(pred).all()
    reloaded=joblib.load(ROOT/'artifacts/xgboost_baseline.joblib')
    np.testing.assert_array_equal(pred,predict_artifact(reloaded,frame))
    np.testing.assert_array_equal(pred,predict_artifact(reloaded,frame[FEATURES[::-1]]))


@pytest.mark.parametrize('change',['missing','target','id','string','infinity','category'])
def test_bad_schema_rejected(artifact,change):
    x=load_split('valid')[FEATURES].head(2).copy()
    if change=='missing': x=x.drop(columns='FullBath')
    elif change=='target': x[TARGET]=1
    elif change=='id': x['Id']=1
    elif change=='string': x['GrLivArea']='wrong'
    elif change=='infinity': x['GrLivArea']=np.inf
    else: x['Neighborhood']=123
    with pytest.raises(ValueError): predict_artifact(artifact,x)


def test_missing_and_unseen_categories(artifact):
    x=load_split('valid')[FEATURES].head(2).copy()
    for c in NUMERIC_FEATURES: x[c]=np.nan
    x['Neighborhood']=[np.nan,'UNSEEN_VALIDATION_NEIGHBORHOOD']
    assert np.isfinite(predict_artifact(artifact,x)).all()


def test_preprocessor_train_only_and_transform_does_not_fit(artifact):
    train=load_split('train')
    valid=load_split('valid')
    prep=artifact['preprocessor']
    imp=prep.named_transformers_['numeric']
    np.testing.assert_allclose(imp.statistics_,train[NUMERIC_FEATURES].median().to_numpy())
    cat=prep.named_transformers_['categorical']
    assert cat.named_steps['imputer'].statistics_[0]==train.Neighborhood.mode().iloc[0]
    assert set(cat.named_steps['encoder'].categories_[0])==set(train.Neighborhood.dropna())
    before=joblib.hash(prep)
    probe=valid[FEATURES].copy()
    probe['GrLivArea']=999999999
    probe['Neighborhood']='NEVER_SEEN_IN_TRAIN'
    prep.transform(probe)
    assert joblib.hash(prep)==before
    assert 'NEVER_SEEN_IN_TRAIN' not in cat.named_steps['encoder'].categories_[0]
    assert set(artifact['metadata']['fit_source_ids'])==set(train.Id)
    assert not set(train.Id)&set(valid.Id)
    assert artifact['metadata']['matrix_shapes']['train'][0]==len(train)


def test_preprocessor_imputes_from_fit_rows_only():
    train=load_split('train')[FEATURES].head(10).copy()
    train.loc[train.index[0],'GarageCars']=np.nan
    prep=build_preprocessor().fit(train)
    probe=train.head(1).copy()
    probe['GarageCars']=np.nan
    result=prep.transform(probe)
    index=NUMERIC_FEATURES.index('GarageCars')
    assert result[0,index]==train.GarageCars.median()


def test_metrics_independently_recomputed(artifact):
    train,valid=load_split('train'),load_split('valid')
    report=json.loads((REPORTS/'validation_metrics.json').read_text())
    assert report['evaluation_split']=='valid' and report['rows']==len(valid)
    for name,pred in [('Dummy Baseline',np.full(len(valid),train.SalePrice.median())),('XGBoost',predict_artifact(artifact,valid[FEATURES]))]:
        actual=report['models'][name]
        assert actual['MAE']==pytest.approx(mean_absolute_error(valid.SalePrice,pred))
        assert actual['RMSE']==pytest.approx(np.sqrt(mean_squared_error(valid.SalePrice,pred)))
        assert actual['R2']==pytest.approx(r2_score(valid.SalePrice,pred))
    assert artifact['baseline'].constant_[0,0]==train.SalePrice.median()


def test_input_access_restriction():
    with pytest.raises(ValueError): load_split('test')
    # Hook blocks BEFORE any bytes can be read.
    with pytest.raises(PermissionError): (ROOT/'data/processed/test.csv').open('rb')
    with pytest.raises(PermissionError): (ROOT/'data/raw/train.csv').open('rb')


def test_eda_train_only_and_notebook_executed():
    from src.utils.notebook_audit import inspect_notebook, REQUIRED
    summary=json.loads((REPORTS/'eda_summary.json').read_text())
    assert summary['split']=='train' and summary['rows']==len(load_split('train'))
    assert len(summary['figures'])>=7 and summary['rows_removed']==0
    assert set(REQUIRED)<=set(Path(p).name for p in summary['figures'])
    result=inspect_notebook()
    assert result['status']=='PASS', result
    assert result['stored_display_data']>=7
    assert set(result['required_charts'])==set(REQUIRED)
    assert all(c['train_values_verified'] for c in result['required_charts'].values())
    assert not result['stored_errors'] and not result['fresh_execution_errors']
    assert not result['missing_displayed_pngs']


def test_week1_preservation():
    from src.utils.preservation import audit_preservation
    result=audit_preservation()
    assert result['status']=='PASS', result['entries']
