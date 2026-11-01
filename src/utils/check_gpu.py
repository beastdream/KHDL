import json
import platform
import re
import subprocess
import traceback
import warnings
from src.utils.paths import REPORTS, write_json


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    except Exception:
        return {'returncode': None, 'error': traceback.format_exc()}


def cuda_config_ready(config):
    learner = config.get('learner', {})
    device = learner.get('generic_param', {}).get('device', '')
    updater = json.dumps(learner.get('gradient_booster', {}))
    return (device == 'cuda' or device.startswith('cuda:')) and 'grow_gpu_hist' in updater


def smoke(device):
    result = {'requested_device': device, 'success': False, 'warnings': []}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            import numpy as np
            import xgboost as xgb
            result['xgboost_version'] = xgb.__version__
            rng = np.random.default_rng(42)
            x = rng.normal(size=(128, 4)).astype('float32')
            y = 2*x[:,0] - x[:,1]
            model = xgb.XGBRegressor(n_estimators=4, max_depth=2, device=device, tree_method='hist', random_state=42)
            model.fit(x,y)
            config = json.loads(model.get_booster().save_config())
            result['booster_config'] = config
            result['actual_device'] = config['learner']['generic_param'].get('device', 'unknown')
            result['success'] = cuda_config_ready(config) if device == 'cuda' else result['actual_device'] == 'cpu'
            if not result['success']:
                result['error'] = 'Fit returned but actual Booster configuration does not confirm requested execution.'
        except Exception:
            result['error'] = traceback.format_exc()
        result['warnings'] = [str(w.message) for w in caught]
    return result


def audit_gpu():
    smi = command(['nvidia-smi'])
    query = command(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv'])
    match = re.search(r'CUDA Version:\s*([\d.]+)', smi.get('stdout',''))
    gpu = smoke('cuda')
    result = {'python_version': platform.python_version(), 'nvidia_smi': smi, 'gpu_inventory': query,
              'driver_cuda_compatibility': match.group(1) if match else None,
              'cuda_note': 'Driver-reported compatibility is not proof of installed CUDA toolkit or successful training.',
              'gpu_smoke': gpu, 'status': 'GPU READY' if gpu['success'] else 'GPU NOT READY',
              'cpu_fallback': None}
    if not gpu['success']:
        result['cpu_fallback'] = smoke('cpu')
        result['remediation'] = ['Inspect full exception and warnings below.', 'Install/update NVIDIA driver compatible with the XGBoost wheel and GPU.',
                                 'Install CUDA-enabled xgboost in this Python 3.11 venv; rerun this audit.',
                                 'CPU results are separate and never establish GPU readiness. PyTorch is not required.']
    write_json(REPORTS / 'gpu_audit.json', result)
    (REPORTS / 'gpu_audit.md').write_text('# GPU environment audit\n\n' + result['status'] + '\n\n```json\n' + json.dumps(result, indent=2) + '\n```\n\nReference: https://xgboost.readthedocs.io/en/stable/gpu/index.html\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    print(audit_gpu()['status'])
