"""Execute this project's plain Python notebook cells, capturing stdout and plots."""
import base64
import contextlib
import io
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.utils.paths import ROOT


def run():
    path=ROOT/'notebooks/house_price_eda.ipynb'
    nb=json.loads(path.read_text(encoding='utf-8'))
    namespace={'__name__':'__main__'}
    count=0
    for cell in nb['cells']:
        if cell['cell_type']!='code': continue
        count+=1
        outputs=[]
        def show():
            for number in plt.get_fignums():
                buffer=io.BytesIO()
                plt.figure(number).savefig(buffer,format='png',bbox_inches='tight')
                outputs.append({'output_type':'display_data','metadata':{},'data':{'image/png':base64.b64encode(buffer.getvalue()).decode('ascii')}})
            plt.close('all')
        plt.show=show
        buffer=io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exec(compile(''.join(cell['source']),f'notebook-cell-{count}','exec'),namespace)
        if buffer.getvalue(): outputs.insert(0,{'output_type':'stream','name':'stdout','text':buffer.getvalue().splitlines(True)})
        cell.update(execution_count=count,outputs=outputs)
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Executed {count} notebook code cells.')


if __name__=='__main__': run()
