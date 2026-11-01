"""Run separate processes so both data policies remain strict and unchanged."""
import subprocess
import sys
from src.utils.paths import ROOT


def main():
    suites = [
        ('tests.week2_policy', ['tests/test_week1.py', 'tests/test_model.py', 'tests/test_gpu_training.py', 'tests/test_preservation_policy.py'], 'reports/week2_pytest.xml'),
        ('tests.week3_policy', ['tests/test_model_tuning.py', 'tests/test_model_selection.py', 'tests/test_inference.py', 'tests/test_app.py'], 'reports/week3_pytest.xml')]
    failed = False
    for policy, files, output in suites:
        result = subprocess.run([sys.executable, '-m', 'pytest', '-p', policy, '-q', *files, '--junitxml=' + output], cwd=ROOT)
        failed |= result.returncode != 0
    for module in ['src.utils.week2_report', 'src.utils.week3_report']:
        result = subprocess.run([sys.executable, '-m', module], cwd=ROOT)
        failed |= result.returncode != 0
    import json
    status = json.loads((ROOT / 'reports/week3_status.json').read_text())
    return 1 if failed or status['status'] != 'PASS' else 0


if __name__ == '__main__':
    sys.exit(main())

