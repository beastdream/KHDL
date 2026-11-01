"""Start a private local Streamlit instance, check health/browser, stop it."""
import json
import subprocess
import sys
import time
import urllib.request
from src.utils.paths import ROOT, REPORTS, write_json
from src.utils.final_precheck import now


def main():
    log = (REPORTS / 'final_streamlit.log').open('w', encoding='utf-8')
    process = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'app.py',
        '--server.address=127.0.0.1', '--server.port=8513', '--server.headless=true',
        '--browser.gatherUsageStats=false'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW)
    result = {'at': now(), 'address': 'http://127.0.0.1:8513'}
    try:
        for _ in range(100):
            try:
                response = urllib.request.urlopen('http://127.0.0.1:8513/_stcore/health', timeout=2)
                result.update(health_status=response.status, health_body=response.read().decode())
                break
            except OSError:
                if process.poll() is not None: raise RuntimeError('Streamlit exited before healthy')
                time.sleep(.2)
        else: raise RuntimeError('Streamlit health timeout')
        result['root_status'] = urllib.request.urlopen(result['address'], timeout=5).status
        browser = subprocess.run([sys.executable, str(ROOT / 'tests/final_ui_browser_check.py')],
                                 cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=240)
        (REPORTS / 'final_browser.log').write_text(browser.stdout + '\n' + browser.stderr, encoding='utf-8')
        result['browser_exit_code'] = browser.returncode
        result['status'] = 'PASS' if browser.returncode == 0 else 'FAIL'
    finally:
        process.terminate()
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired: process.kill(); process.wait()
        log.close()
        result['server_stopped'] = True
        write_json(REPORTS / 'final_streamlit_verification.json', result)
    print(json.dumps(result, indent=2))
    return int(result['status'] != 'PASS')


if __name__ == '__main__':
    sys.exit(main())
