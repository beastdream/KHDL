"""Read-only experimental invariants and real warning evidence for plotting QA."""
import json
import shutil
import subprocess
import sys
import warnings
import xml.etree.ElementTree as ET
from src.utils.paths import ROOT, REPORTS, sha256, write_json

EVIDENCE = REPORTS / 'warning_resolution'
PROTECTED = ['artifacts/house_price_model.joblib', 'artifacts/model_metadata.json',
    'artifacts/xgboost_baseline.joblib', 'src/models/preprocessing.py',
    'reports/data_split.json', 'reports/final_model_freeze.json', 'reports/test_metrics.json',
    'reports/test_predictions.json', 'reports/test_evaluation.md',
    'reports/final_evaluation_attempt.json', 'reports/final_evaluation_verifications.json',
    'reports/validation_predictions.csv', 'requirements.txt', 'requirements-lock.txt',
    'requirements-final-lock.txt', 'data/processed/train.csv', 'data/processed/valid.csv',
    '.venv/Lib/site-packages/seaborn/categorical.py', '.venv/Lib/site-packages/seaborn/matrix.py',
    '.venv/Lib/site-packages/matplotlib/axes/_axes.py']


def snapshot():
    EVIDENCE.mkdir(exist_ok=True)
    destination = EVIDENCE / 'before.json'
    if destination.exists():
        raise FileExistsError('Keep initial warning evidence immutable')
    for name in ['notebooks/house_price_eda.ipynb', 'src/data/eda.py', 'tests/test_model_tuning.py']:
        shutil.copyfile(ROOT / name, EVIDENCE / name.replace('/', '__'))
    write_json(destination, {'hashes': {p: sha256(ROOT / p) for p in PROTECTED},
        'eda_summary': json.loads((REPORTS / 'eda_summary.json').read_text(encoding='utf-8')),
        'holdout_policy': 'No Test bytes opened or rehashed; original manifest remains protected'})


def capture(name):
    import matplotlib
    import seaborn
    from src.utils.notebook_audit import inspect_notebook
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = inspect_notebook()
    assert result['status'] == 'PASS', result
    evidence = {'matplotlib': matplotlib.__version__, 'seaborn': seaborn.__version__,
        'notebook_status': result['status'], 'warnings': [
            {'category': w.category.__name__, 'message': str(w.message),
             'file': w.filename, 'line': w.lineno} for w in caught]}
    write_json(EVIDENCE / f'{name}_warnings.json', evidence)
    print(json.dumps(evidence, ensure_ascii=True, indent=2))


def suite(name):
    run = subprocess.run([sys.executable, '-m', 'src.utils.validate_final'], cwd=ROOT,
                         capture_output=True, text=True, encoding='utf-8', errors='replace')
    (EVIDENCE / f'{name}_pytest.log').write_text(run.stdout + '\n' + run.stderr, encoding='utf-8')
    shutil.copyfile(REPORTS / 'final_pytest.xml', EVIDENCE / f'{name}_pytest.xml')
    shutil.copyfile(REPORTS / 'final_test_counts.json', EVIDENCE / f'{name}_counts.json')
    print(run.stdout)
    return run.returncode


def finish():
    from src.utils.plotting_preservation import verify_notebook_presentation, verify_eda_presentation
    from src.utils.preservation import audit_preservation
    from src.models.evaluate_test import verify_freeze
    assert verify_notebook_presentation() and verify_eda_presentation()
    verify_freeze()  # Checks artifacts only: never calls evaluate() or opens Test.
    assert audit_preservation()['status'] == 'PASS'
    before = json.loads((EVIDENCE / 'before.json').read_text(encoding='utf-8'))
    checks = {p: sha256(ROOT / p) == h for p, h in before['hashes'].items()}
    assert all(checks.values()), checks
    assert json.loads((REPORTS / 'eda_summary.json').read_text(encoding='utf-8')) == before['eda_summary']
    import numpy as np
    from PIL import Image
    images = sorted(EVIDENCE.glob('week2_*.png'))
    assert len(images) == 7
    for image in images:
        assert np.array_equal(np.asarray(Image.open(image)),
                              np.asarray(Image.open(REPORTS / 'figures' / image.name))), image.name
    first = json.loads((EVIDENCE / 'before_warnings.json').read_text())
    last = json.loads((EVIDENCE / 'after_warnings.json').read_text())
    assert len(first['warnings']) == 4 and not last['warnings']
    before_counts = json.loads((EVIDENCE / 'before_counts.json').read_text())
    after_counts = json.loads((EVIDENCE / 'after_counts.json').read_text())
    assert before_counts == after_counts == {'PASS': 73, 'FAIL': 0, 'SKIPPED': 14}
    strict = ET.parse(EVIDENCE / 'strict_notebook_pytest.xml').findall('.//testcase')
    assert len(strict) == 1 and not strict[0].findall('failure') and not strict[0].findall('error')
    server = json.loads((REPORTS / 'final_streamlit_verification.json').read_text())
    assert server['status'] == 'PASS' and server['browser_exit_code'] == 0
    write_json(EVIDENCE / 'completion_verification.json', {'status': 'PASS',
        'before_counts': before_counts, 'after_counts': after_counts,
        'before_warning_count': 4, 'after_warning_count': 0, 'strict_warning_error_test': 'PASS',
        'protected_hashes_unchanged_after_all_tests': checks, 'notebook_preservation': 'EXACT REVIEWED DIFF',
        'eda_preservation': 'EXACT REVIEWED DIFF', 'streamlit': server,
        'dependency_upgrade': False, 'test_evaluated_again': False})
    text = '''# Xử lý cảnh báo thư viện — HousePrice AI

## 1. Môi trường và kết luận

Matplotlib **3.11.2**, Seaborn **0.13.2**, Python 3.11.9 trong `.venv` của dự án. Không cài/nâng/hạ dependency, không sửa file site-packages hoặc .venv. `pip check`: No broken requirements found. Các requirements/lock và source thư viện được đối chiếu hash trước/sau, giữ nguyên.

**Đã xử lý 4 warnings trong luồng Notebook/EDA kiểm thử: 4 → 0. Bộ kiểm thử trước và sau đều 73 PASS / 0 FAIL / 14 SKIPPED.** Không dùng ignore filter hoặc sửa cấu hình pytest để che warning. Chạy notebook với `-W error` còn PASS, nên warning tái xuất hiện sẽ làm test thất bại.

## 2. Truy vết chính xác

Tái hiện bằng thực thi đầy đủ notebook trong bộ nhớ với Data Guard, `warnings.simplefilter('always')` và ghi mọi warning vào JSON. Chế độ capture này dùng để thu thập bằng chứng, không được dùng để coi có warning là sạch. Baseline pytest thông thường cũng báo đúng bốn warning.

| Số lần | Loại | Vị trí phát ra | Nguyên nhân |
|---:|---|---|---|
| 2 | MatplotlibDeprecationWarning: `vert: bool` deprecated từ 3.11, bỏ ở 3.13 | `.venv/Lib/site-packages/seaborn/categorical.py:700` | Seaborn boxplot truyền `vert=True` vào `Axes.bxp`; hai boxplot chất lượng/phòng ngủ trong run_eda |
| 2 | PendingDeprecationWarning: `set_bad` | `.venv/Lib/site-packages/seaborn/matrix.py:243` | Seaborn tái căn giữa colormap khi `center=0`, rồi gọi `self.cmap.set_bad`; một heatmap run_eda và một heatmap notebook |

**Không tái hiện warning `get_cmap` trong môi trường này.** `seaborn/_compat.py:get_colormap` có nhánh fallback `mpl.cm.get_cmap`, nhưng Matplotlib hiện có `mpl.colormaps`, nên nhánh hiện đại được dùng. Không suy diễn một warning không quan sát được thành lỗi đã sửa. Các cảnh báo thực tế là `vert` và `set_bad`; xem before_warnings.json để biết nguyên văn category/message/file/line.

Source dự án kích hoạt warning qua API public Seaborn, nhưng lệnh deprecated nằm trong thư viện. Không phải lỗi model/preprocessing. Đã đọc source thư viện để xác minh; không thay source thư viện hay monkeypatch API lúc chạy ứng dụng.

## 3. Thay đổi tối thiểu và lý do

### Boxplot

Trong `src/data/eda.py`, hai lời gọi boxplot dùng `vert=None, orientation='vertical'` khi `Axes.bxp` hỗ trợ orientation. Seaborn 0.13.2 cho phép kwargs chuyển tới bxp; `vert=None` thay bool mà Seaborn tự truyền bằng giá trị trung lập, để orientation điều khiển hướng vẽ. Đây là cách tránh dùng bool deprecated, không phải lọc warning. Có feature detection để Matplotlib cũ chưa có orientation dùng đường gọi cũ; nhánh cũ chưa được kiểm chứng bằng cài nhiều phiên bản trong lần này.

Giữ nhóm dữ liệu, thứ tự, thống kê quartile/whisker, outlier và hướng dọc. So sánh pixel hai hình trước/sau cho kết quả giống hệt. Đây là giải pháp tương thích với cặp phiên bản hiện cài; khi lên Matplotlib có thể đã xóa tham số vert, phải kiểm tra lại Seaborn và bỏ giải pháp chuyển tiếp này khi API thư viện đã sửa.

### Heatmap

Trong run_eda, giữ `vmin=-1, vmax=1` và bỏ `center=0` dư thừa: thang đối xứng đã đặt 0 tại tâm, không cần chạy nhánh recenter gọi set_bad. Trong cell 15 notebook, thay đúng một dòng `center=0` bằng `vmin=-1, vmax=1` cho tương quan Pearson. Giữ ma trận, nhãn, annotations, cmap coolwarm, linewidths và các cell khác. Không đổi dữ liệu hay cách tính correlation.

Bảy PNG EDA có kích thước và từng pixel giống bản trước; eda_summary.json bằng nhau về toàn bộ nội dung JSON. Notebook được chạy lại trong bộ nhớ, gồm Plotly/widgets và callback có/không dữ liệu; không ghi lại output notebook để tránh thay đổi hồ sơ thí nghiệm không cần thiết.

### Bảo toàn notebook và cập nhật hợp đồng kiểm thử

Model metadata chứa hash notebook tại thời điểm nghiên cứu. Không sửa hash này. Bản notebook nguyên byte trước sửa được giữ ở `reports/warning_resolution/notebooks__house_price_eda.ipynb`, khớp chính xác `model_metadata.json:notebook_sha256`.

`test_data_guard_and_preservation` vẫn kiểm tra nguyên vẹn guard/split/baseline, đồng thời kiểm tra notebook qua `verify_notebook_presentation`: hash bản nghiên cứu gốc phải khớp metadata và toàn bộ notebook hiện tại phải bằng bản đó sau đúng một thay thế cấu hình heatmap ở cell 15. Mọi sửa cell khác, metadata hoặc stored output đều bị từ chối. Không chỉ thay golden checksum để qua test. `verify_eda_presentation` kiểm tra toàn bộ source EDA bằng diff được phép từ bản gốc đã ghim với snapshot UI lịch sử. Test không bị xóa/skip; tổng số test giữ nguyên.

`src/utils/final_report.py` được cập nhật để nhận diện các thay đổi trình bày đã xác minh, thay vì báo tamper cho cải tiến đã được yêu cầu; baseline và snapshot lịch sử vẫn nguyên. Ghi chú warning trong báo cáo sinh mới trỏ tới tài liệu này. Không chạy lại training, tuning hoặc Final Evaluation.

## 4. Kết quả kiểm thử thực tế

| Kiểm tra | Trước | Sau |
|---|---:|---:|
| Regression safe suite PASS | 73 | 73 |
| FAIL | 0 | 0 |
| SKIPPED | 14 | 14 |
| Warnings trong pytest | 4 | 0 |
| Notebook chạy đầy đủ, capture mọi warning | PASS, 4 | PASS, 0 |

14 SKIP giữ nguyên lý do hạn chế raw/holdout của policy cũ; không tính SKIP là PASS. Suite gồm legacy 27 PASS/14 SKIP, Tuần 3 27 PASS và E2E 19 PASS. Notebook test riêng với `-W error`: **1 PASS**, không warning. AppTest kiểm tra đủ 5 trang và form/model thật trong safe suite. Chrome headless và HTTP root/health đều PASS; bảy hình performance tải được, không phát hiện tràn ngang trong các viewport đã thử. Máy chủ QA đã dừng. Chưa tuyên bố kiểm chứng mọi browser hoặc mọi bộ phiên bản dependency.

Lệnh đã thực thi:

```powershell
.\\.venv\\Scripts\\python.exe -m src.utils.warning_audit snapshot
.\\.venv\\Scripts\\python.exe -m src.utils.warning_audit before
.\\.venv\\Scripts\\python.exe -m src.utils.warning_audit before-suite
.\\.venv\\Scripts\\python.exe -m src.utils.warning_audit after
.\\.venv\\Scripts\\python.exe -m pytest -p tests.week2_policy -q tests/test_model.py::test_eda_train_only_and_notebook_executed -W error --junitxml=reports/warning_resolution/strict_notebook_pytest.xml
.\\.venv\\Scripts\\python.exe -m src.utils.warning_audit after-suite
.\\.venv\\Scripts\\python.exe -m src.utils.verify_final_ui
.\\.venv\\Scripts\\python.exe -m pip check
.\\.venv\\Scripts\\python.exe -m src.utils.warning_audit finish
```

## 5. Bảo toàn thực nghiệm

Đối chiếu SHA-256 trước/sau sau toàn bộ kiểm thử: artifact chính và baseline, model metadata/hyperparameters, source preprocessing, manifest split, Train/Valid bytes, freeze, metrics/predictions/evaluation Test, Validation predictions, marker/log lần đánh giá Test và requirements/locks đều nguyên. Không mở lại/rehash holdout; không gọi evaluate_test.evaluate(), không dự đoán lại Test. Split 70/15/15 và seed 42 giữ nguyên.

Không huấn luyện lại model chính hay chạy lại tuning/benchmark để sửa warning. Các test legacy vốn có vẫn chứa một unit fit nhỏ 3 cây trên 30 dòng Train trong RAM; đây là kiểm tra Pipeline độc lập, không thay artifact hay tham số đã chọn. Các GPU fallback test dùng FakeModel. Không gọi pipeline training chính.

## 6. Warning còn tồn tại và giới hạn

**Không còn warning trong các luồng đã chạy ở trên.** Không có bằng chứng warning get_cmap trên cặp phiên bản này, vì vậy không ghi nhận nó là một warning đã sửa. Nội bộ Seaborn 0.13.2 vẫn có API deprecated cho các cách gọi khác; nhiệm vụ xử lý các lời gọi của dự án hiện tại, không sửa mọi chức năng của thư viện. Chưa kiểm chứng nâng Matplotlib/Seaborn hoặc cài sạch trên máy khác. Không nâng dependency để tránh thay môi trường thực nghiệm đã kiểm chứng.

## 7. Bằng chứng và file thay đổi

Sửa: `src/data/eda.py`, `notebooks/house_price_eda.ipynb` (một dòng source), `tests/test_model_tuning.py`, `src/utils/final_report.py`. Thêm: `src/utils/plotting_preservation.py`, `src/utils/warning_audit.py`, báo cáo này và thư mục bằng chứng `reports/warning_resolution/`. Các output EDA/notebook audit, JUnit và báo cáo QA được tái sinh bởi lệnh kiểm chứng, không phải số liệu được điền tay. Các báo cáo nghiệm thu trước đó là hồ sơ lịch sử; tài liệu này bổ sung trạng thái warning hiện tại.

Thư mục bằng chứng gồm bản gốc source/notebook/test, before.json, before/after_warnings.json, before/after_pytest.log và XML, before/after_counts.json, strict_notebook_pytest.xml, invariant_verification.json, completion_verification.json và bảy PNG trước sửa. Final JUnit hiện tại cũng nằm ở reports/final_pytest.xml.

Tham chiếu API chính thức: [Matplotlib Axes.bxp](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bxp.html), [Seaborn heatmap](https://seaborn.pydata.org/generated/seaborn.heatmap.html). Các phiên bản và vị trí phát warning được xác nhận trực tiếp từ môi trường cài tại máy.
'''
    (REPORTS / 'WARNING_RESOLUTION_REPORT.md').write_text(text, encoding='utf-8')
    print('Warning resolution VERIFIED: 73 PASS / 0 FAIL / 14 SKIPPED; warnings 4 -> 0; protected hashes unchanged.')


if __name__ == '__main__':
    action = sys.argv[1]
    if action == 'snapshot': snapshot()
    elif action in ('before', 'after'): capture(action)
    elif action in ('before-suite', 'after-suite'): sys.exit(suite(action.split('-')[0]))
    elif action == 'finish': finish()
