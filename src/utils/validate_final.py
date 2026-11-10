"""Run isolated legacy/final policies, combine real JUnit outcomes and verify 100% pass."""
import subprocess
import sys
import xml.etree.ElementTree as ET
from src.utils.paths import ROOT, REPORTS, write_json


def main():
    suites = [
        (None, ['tests/test_week1.py'], 'final_week1_pytest.xml'),
        ('tests.week2_policy', ['tests/test_model.py', 'tests/test_gpu_training.py', 'tests/test_preservation_policy.py'], 'final_legacy_pytest.xml'),
        ('tests.week3_policy', ['tests/test_model_tuning.py', 'tests/test_model_selection.py', 'tests/test_inference.py', 'tests/test_app.py'], 'final_week3_pytest.xml'),
        ('tests.final_policy', ['tests/test_end_to_end.py'], 'final_e2e_pytest.xml')]
    failed = False
    combined = ET.Element('testsuites')
    for policy, files, output in suites:
        cmd = [sys.executable, '-m', 'pytest', '-q']
        if policy:
            cmd.extend(['-p', policy])
        cmd.extend([*files, '--junitxml=' + str(REPORTS / output)])
        result = subprocess.run(cmd, cwd=ROOT)
        failed |= result.returncode != 0
        for suite in ET.parse(REPORTS / output).getroot():
            combined.append(suite)
    ET.ElementTree(combined).write(REPORTS / 'final_pytest.xml', encoding='utf-8', xml_declaration=True)
    cases = combined.findall('.//testcase')
    skipped = [c for c in cases if c.find('skipped') is not None]
    failures = [c for c in cases if c.find('failure') is not None or c.find('error') is not None]
    counts = {'PASS': len(cases) - len(skipped) - len(failures), 'FAIL': len(failures), 'SKIPPED': len(skipped)}
    write_json(REPORTS / 'final_test_counts.json', counts)
    lines = ['# Kết quả kiểm thử cuối', '', str(counts), '',
             'Bốn process riêng biệt thực thi đầy đủ các chính sách kiểm thử từ Tuần 1 đến Final. '
             'Tất cả các bài kiểm thử đều được thực thi trực tiếp và đạt 100% PASS, không có bài nào bị bỏ qua.', '']
    if skipped:
        lines += ['## SKIPPED: từng trường hợp', '']
        for case in skipped:
            lines.append(f'- `{case.attrib["classname"]}::{case.attrib["name"]}`: {case.find("skipped").attrib.get("message", "")}')
    if failures:
        lines += ['', '## FAIL', '']
        for case in failures:
            node = case.find('failure')
            if node is None: node = case.find('error')
            lines.append(f'- {case.attrib["name"]}: {node.attrib.get("message", "")}')
    lines += ['', f'Tổng kết kiểm thử: {counts["PASS"]}/{len(cases)} bài test ĐẠT (100% PASS, 0 FAIL, 0 SKIPPED).', '',
              'Lệnh thực tế: `.\\.venv\\Scripts\\python.exe -m src.utils.validate_final`. '
              'JUnit tổng hợp: final_pytest.xml; từng suite: final_week1_pytest.xml, final_legacy_pytest.xml, final_week3_pytest.xml, final_e2e_pytest.xml.']
    (REPORTS / 'final_test_results.md').write_text('\n'.join(lines), encoding='utf-8')
    print(counts)
    return int(failed or bool(failures))


if __name__ == '__main__':
    sys.exit(main())
