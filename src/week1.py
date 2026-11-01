import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone, timedelta
from src.data.audit_dataset import audit
from src.utils.check_gpu import audit_gpu
from src.utils.paths import ROOT, REPORTS, write_json


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    failure = None
    try:
        data = audit()
    except Exception:
        failure = traceback.format_exc()
        data = {'status': 'NOT READY', 'error': failure}
        write_json(REPORTS / 'dataset_audit.json', data)
        (REPORTS / 'dataset_audit.md').write_text('# Dataset audit\n\nNOT READY\n\n```\n' + failure + '\n```', encoding='utf-8')
    gpu = audit_gpu()
    proc = subprocess.run([sys.executable, '-m', 'pytest', '-q', '--junitxml=reports/pytest.xml'], cwd=ROOT, capture_output=True, text=True)
    log = proc.stdout + proc.stderr
    (REPORTS / 'pytest_results.txt').write_text(log, encoding='utf-8')
    split_path = REPORTS / 'data_split.json'
    split = json.loads(split_path.read_text(encoding='utf-8')) if split_path.exists() else None
    passed = data['status'] == 'PASS' and split is not None and proc.returncode == 0
    lines = ['# Báo cáo Tuần 1 — House Price Prediction', '',
        'Ngày chạy: ' + datetime.now(timezone(timedelta(hours=7))).isoformat(), '',
        '## 1. Tổng quan dataset', '',
        f"Nguồn: Kaggle House Prices, data/raw/train.csv. Số dòng: {data.get('rows', 'N/A')}; số cột: {data.get('columns', 'N/A')}. Target: SalePrice.",
        'Giữ nguyên file gốc. test.csv ở thư mục gốc là dữ liệu thi Kaggle không có nhãn, không phải holdout nội bộ.', '',
        '## 2. Kết quả audit', '', f"Trạng thái: {data['status']}. Không tự xóa bản ghi/outlier.",
        f"Duplicate: {data.get('duplicate_rows', 'N/A')}; duplicate nội dung: {data.get('duplicate_content_excluding_id','N/A')}; lỗi schema: {data.get('invalid_records', 'N/A')}.",
        'Kiểm tra chất lượng trên toàn bộ nguồn. Thống kê min/max/mean/median/std, target và biểu đồ chỉ dùng TRAIN; không EDA trên test.',
        'Chi tiết: [dataset_audit.md](dataset_audit.md), [dataset_audit.json](dataset_audit.json).', '',
        '## 3. Bảng Train/Valid/Test', '', '| Tập | Số mẫu | Tỷ lệ |', '|---|---:|---:|']
    if split:
        for name, info in split['splits'].items():
            lines.append(f"| {name} | {info['count']} | {info['fraction']:.2%} |")
        lines += ['', 'Seed: 42; hai lần train_test_split; không fit preprocessing. Giữ Id và index nguồn trong manifest.',
                  'SHA-256 test.csv: `' + split['splits']['test']['sha256'] + '`.',
                  'Test được khóa đến đánh giá cuối Tuần 4; chạy lại kiểm tra hash trước khi ghi.']
    lines += ['', '## 4. Kiểm tra GPU', '', gpu['status'], '',
              '```text\n' + gpu['gpu_inventory'].get('stdout', str(gpu['gpu_inventory'])) + '\n```',
              f"Python: {gpu['python_version']}; XGBoost: {gpu['gpu_smoke'].get('xgboost_version','unavailable')}; driver CUDA compatibility: {gpu['driver_cuda_compatibility']}.",
              'Kiểm tra fit trên dữ liệu smoke độc lập, device=cuda/tree_method=hist, xác nhận device và GPU updater trong Booster.',
              'Chi tiết đầy đủ: [gpu_audit.md](gpu_audit.md).', '', '## 5. Các lỗi còn tồn tại', '',
              failure or data.get('error') or ('Không có lỗi chặn dữ liệu.' if passed else 'Kiểm tra audit và pytest bên dưới.'),
              'Missing predictors được giữ nguyên; chỉ fit imputer/encoder trên train trong tuần tiếp theo.',
              'GPU: ' + ('Không có lỗi chặn smoke test.' if gpu['status'] == 'GPU READY' else 'Chưa sẵn sàng; xem lỗi và remediation trong gpu_audit.json.'),
              '', '## 6. Kết quả pytest', '', '```text\n' + log + '\n```', '',
              '## 7. Kết luận', '', ('PASS' if passed else 'NOT READY'), '', gpu['status'], '',
              'Chỉ thực hiện Tuần 1. Không chuyển Tuần 2 nếu kiểm tra dữ liệu chưa PASS.']
    (REPORTS / 'WEEK_1_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(('PASS' if passed else 'NOT READY') + ' / ' + gpu['status'])
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
