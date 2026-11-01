"""Matched CPU/GPU Train-only experiment; benchmark models stay in memory."""
import io
import json
import platform
import statistics
import time
import warnings
import joblib
from sklearn.base import clone
from xgboost import DMatrix
import xgboost
from src.data.final_inputs import install_data_guard
from src.data.week2_inputs import load_split
from src.data.schema import FEATURES
from src.models.inference import load_model
from src.models.evaluate_test import verify_freeze
from src.utils.check_gpu import cuda_config_ready, command
from src.utils.paths import ROOT, REPORTS, write_json
from src.utils.final_precheck import now


def latency(model, x, repeats=100):
    # Warm prediction; native DMatrix handles host data for both CPU/GPU.
    def predict():
        matrix = model.named_steps['preprocessor'].transform(x)
        return model.named_steps['model'].get_booster().predict(DMatrix(matrix))
    predict()
    times = []
    for _ in range(repeats):
        start = time.perf_counter(); predict(); times.append(time.perf_counter() - start)
    return {'median_ms': statistics.median(times) * 1000, 'repeats': repeats}


def benchmark(repeats=3):
    install_data_guard(); frozen = verify_freeze()
    final, meta = load_model()
    train = load_split('train'); x, y = train[FEATURES], train.SalePrice
    results = {'created_at': now(), 'repeats': repeats, 'rows': len(train), 'seed': 42,
               'python': platform.python_version(), 'xgboost': xgboost.__version__,
               'hyperparameters': meta['hyperparameters'], 'frozen_artifact_sha256': frozen['artifact_sha256'],
               'frozen_artifact_bytes': (ROOT / frozen['artifact']).stat().st_size,
               'inventory': command(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv']),
               'nvidia_smi': command(['nvidia-smi']), 'devices': {},
               'timing_scope': 'full Pipeline.fit including preprocessing; one untimed warmup per device; sequential 3 fits; n_jobs=1',
               'prediction_scope': 'preprocessor.transform + DMatrix + Booster.predict; warm; host data for both devices',
               'peak_vram_mib': None,
               'peak_vram_note': 'Not measured: WDDM/per-process telemetry unavailable; snapshot is not peak VRAM',
               'test_accessed': False, 'benchmark_models_replace_final': False}
    for device in ('cpu', 'cuda'):
        records, caught_messages = [], []
        try:
            for run in range(repeats + 1):
                pipe = clone(final).set_params(model__device=device)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    start = time.perf_counter(); pipe.fit(x, y); seconds = time.perf_counter() - start
                caught_messages += [str(w.message) for w in caught]
                config = json.loads(pipe.named_steps['model'].get_booster().save_config())
                actual = config['learner']['generic_param']['device']
                if device == 'cuda' and not cuda_config_ready(config):
                    raise RuntimeError('CUDA silently fell back or GPU updater not confirmed')
                if device == 'cpu' and actual != 'cpu':
                    raise RuntimeError('CPU device not verified')
                if run:
                    records.append({'seconds': seconds, 'actual_device': actual,
                        'updater': config['learner']['gradient_booster']['gbtree_train_param']['updater']})
            buffer = io.BytesIO(); joblib.dump(pipe, buffer)
            results['devices'][device] = {'status': 'SUCCESS', 'fits': records,
                'training_median_seconds': statistics.median(r['seconds'] for r in records),
                'training_stdev_seconds': statistics.stdev(r['seconds'] for r in records),
                'latency_1': latency(pipe, x.head(1)), 'latency_100': latency(pipe, x.head(100)),
                'serialized_bytes': buffer.tell(), 'warnings': caught_messages}
        except Exception as exc:
            results['devices'][device] = {'status': 'UNAVAILABLE', 'reason': repr(exc), 'partial_fits': records,
                                          'warnings': caught_messages}
    cpu, gpu = results['devices']['cpu'], results['devices']['cuda']
    results['gpu_status'] = 'GPU READY' if gpu['status'] == 'SUCCESS' else 'GPU NOT READY'
    results['speedup_cpu_over_gpu'] = (cpu['training_median_seconds'] / gpu['training_median_seconds']
        if cpu['status'] == gpu['status'] == 'SUCCESS' else None)
    verify_freeze(); write_json(REPORTS / 'gpu_benchmark.json', results)
    lines = ['# Benchmark CPU/GPU', '', f'Ngày: {results["created_at"]}; XGBoost {xgboost.__version__}.', '',
             'Cùng Train, 8 features, preprocessing, hyperparameters, seed và n_jobs=1. '
             'Một warmup không tính giờ + 3 lần fit tuần tự/thiết bị. Thời gian gồm Pipeline.fit và preprocessing. '
             'Không tuning, không dùng Test; model benchmark chỉ trong RAM, không thay model chính.', '',
             '| Thiết bị | Fit median s | Std s | 1 mẫu ms | 100 mẫu ms | Bytes |', '|---|---:|---:|---:|---:|---:|']
    for name, values in results['devices'].items():
        if values['status'] == 'SUCCESS':
            lines.append(f'| {name} | {values["training_median_seconds"]:.4f} | {values["training_stdev_seconds"]:.4f} | '
                         f'{values["latency_1"]["median_ms"]:.3f} | {values["latency_100"]["median_ms"]:.3f} | {values["serialized_bytes"]} |')
        else:
            lines.append(f'| {name} | Không khả dụng: {values["reason"]} | — | — | — | — |')
    if results['speedup_cpu_over_gpu'] is not None:
        lines += ['', f'Tỷ số CPU/GPU = {results["speedup_cpu_over_gpu"]:.3f}. '
                  'Tỷ số >1 là GPU nhanh hơn; <1 là GPU chậm hơn. Dataset nhỏ, overhead truyền dữ liệu và khởi tạo '
                  'có thể chi phối. Không ngoại suy cho dữ liệu lớn hoặc phần cứng khác.']
    lines += ['', 'Latency gồm transform + tạo DMatrix + Booster.predict, median 100 lần sau warmup. '
              'Đây là đường host-input có thể so sánh, không phải latency HTTP/Streamlit. '
              'Inference triển khai vẫn CPU. Booster mỗi fit xác minh actual_device và GPU updater; '
              'không coi GPU READY lịch sử là bằng chứng benchmark mới.', '',
              f'Model chính: {results["frozen_artifact_bytes"]} bytes. '
              'Peak VRAM: không đo được đáng tin cậy; không thay bằng snapshot nvidia-smi. '
              'GPU inventory, driver, compatibility CUDA và cấu hình chi tiết lưu trong gpu_benchmark.json.']
    (REPORTS / 'gpu_benchmark.md').write_text('\n'.join(lines), encoding='utf-8')
    return results


if __name__ == '__main__':
    print(json.dumps(benchmark(), ensure_ascii=False, indent=2))
