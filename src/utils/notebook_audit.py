"""Execute notebook in memory under the unchanged Data Guard; validate chart data, not output count."""
import base64
import io
import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Image
from IPython.utils.capture import capture_output
from src.data.week2_inputs import install_data_guard, load_split
from src.data.schema import NUMERIC_FEATURES, TARGET
from src.utils.paths import ROOT, REPORTS, sha256, write_json

REQUIRED={
 'week2_01_price_histogram.png':('histogram','SalePrice',None),
 'week2_02_area_price.png':('scatter','GrLivArea','SalePrice'),
 'week2_03_quality_price.png':('box','OverallQual','SalePrice'),
 'week2_04_neighborhood_median.png':('barh','Neighborhood','median(SalePrice)'),
 'week2_05_correlation.png':('heatmap','Pearson',None),
 'week2_06_year_price.png':('scatter','YearBuilt','SalePrice'),
 'week2_07_bedrooms_price.png':('box','BedroomAbvGr','SalePrice')}


def inspect_notebook():
    install_data_guard()
    path=ROOT/'notebooks/house_price_eda.ipynb'
    nb=json.loads(path.read_text(encoding='utf-8'))
    code=[(i,c) for i,c in enumerate(nb['cells']) if c['cell_type']=='code']
    stored_errors=[{'cell':i,'error':o.get('evalue')} for i,c in code for o in c.get('outputs',[]) if o['output_type']=='error']
    mime=Counter(k for _,c in code for o in c.get('outputs',[]) if o['output_type']=='display_data' for k in o.get('data',{}))
    train=load_split('train')
    expected_corr=train[NUMERIC_FEATURES+[TARGET]].corr().to_numpy()
    group=train.groupby('Neighborhood').SalePrice.median().sort_values()
    last=[None]
    charts={}
    displayed=[]
    errors=[]
    cells=[]
    def capture_plot(kind,original):
        def wrapper(*args,**kwargs):
            if kind in ('scatter','box'):
                frame=kwargs['data']; x=kwargs['x']; y=kwargs['y']
                valid=frame.equals(train) and y=='SalePrice'
                record=(kind,x,y,valid)
            elif kind=='histogram':
                values=args[0] if args else kwargs.get('x')
                record=(kind,'SalePrice',None,np.array_equal(np.asarray(values),train.SalePrice.to_numpy()))
            elif kind=='heatmap':
                values=args[0] if args else kwargs['data']
                record=(kind,'Pearson',None,np.asarray(values).shape==expected_corr.shape and np.allclose(values,expected_corr))
            else:
                record=(kind,'Neighborhood','median(SalePrice)',list(args[0])==group.index.tolist() and np.allclose(args[1],group.to_numpy()))
            result=original(*args,**kwargs)
            last[0]=record
            return result
        return wrapper
    save=plt.savefig
    def save_checked(filename,*args,**kwargs):
        name=Path(filename).name
        if name in REQUIRED:
            expected=REQUIRED[name]
            assert last[0] and last[0][:3]==expected and last[0][3], f'Wrong chart content: {name}, {last[0]}'
            charts[name]={'type':expected[0],'x':expected[1],'y':expected[2],'train_values_verified':True}
        return save(filename,*args,**kwargs)
    read=plt.imread
    def read_checked(filename,*args,**kwargs):
        displayed.append(Path(filename).name)
        return read(filename,*args,**kwargs)
    def show(*args,**kwargs):
        for number in plt.get_fignums():
            buffer=io.BytesIO()
            plt.figure(number).savefig(buffer,format='png',bbox_inches='tight')
            display(Image(data=buffer.getvalue()))
        plt.close('all')
    from IPython.core.interactiveshell import InteractiveShell
    InteractiveShell.instance()
    namespace={'__name__':'__main__'}
    import plotly.io as pio
    old_renderer=pio.renderers.default
    pio.renderers.default='plotly_mimetype'
    try:
        with patch.object(sns,'histplot',capture_plot('histogram',sns.histplot)), \
             patch.object(sns,'scatterplot',capture_plot('scatter',sns.scatterplot)), \
             patch.object(sns,'boxplot',capture_plot('box',sns.boxplot)), \
             patch.object(sns,'heatmap',capture_plot('heatmap',sns.heatmap)), \
             patch.object(plt,'barh',capture_plot('barh',plt.barh)), \
             patch.object(plt,'savefig',save_checked), patch.object(plt,'imread',read_checked), patch.object(plt,'show',show):
            for i,c in code:
                try:
                    with capture_output() as captured:
                        exec(compile(''.join(c['source']),f'notebook-cell-{i}','exec'),namespace)
                    cells.append({'cell':i,'status':'PASS','rich_outputs':len(captured.outputs),'stderr':captured.stderr})
                except Exception as e:
                    errors.append({'cell':i,'type':type(e).__name__,'error':str(e)})
                    break
    finally:
        pio.renderers.default=old_renderer
        plt.close('all')
    if not errors and 'interactive_analysis' in namespace:
        try:
            with capture_output():
                namespace['interactive_analysis']('Tất cả',(1,10))
                namespace['interactive_analysis']('NO_SUCH_AREA',(1,10))
        except Exception as e:
            errors.append({'cell':'widget callback','error':str(e)})
    missing=sorted(set(REQUIRED)-set(charts))
    missing_display=sorted(set(REQUIRED)-set(displayed))
    result={'status':'PASS' if not (errors or stored_errors or missing or missing_display) and len(cells)==len(code) else 'NOT READY',
            'notebook_sha256':sha256(path),'stored_display_data':sum(o['output_type']=='display_data' for _,c in code for o in c.get('outputs',[])),
            'stored_mime_counts':dict(mime),'stored_errors':stored_errors,'fresh_execution_errors':errors,'cells':cells,
            'required_charts':charts,'missing_charts':missing,'missing_displayed_pngs':missing_display,
            'source_policy':'All inspected cells read processed/train.csv only; fresh execution protected by unchanged Data Guard.',
            'historical_access':'Source and stored output inspection cannot prove past kernel access; this fresh execution is guarded.',
            'notebook_modified':False}
    write_json(REPORTS/'week2_notebook_audit.json',result)
    return result


if __name__=='__main__':
    result=inspect_notebook()
    print(json.dumps(result,indent=2,ensure_ascii=True))
    raise SystemExit(0 if result['status']=='PASS' else 1)
