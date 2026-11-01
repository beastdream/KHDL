"""Reproduce Week 2 without touching Week 1 artifacts or opening holdout."""
import subprocess
import sys
from src.data.week2_inputs import install_data_guard
from src.models.train_model import train
from src.utils.run_eda_notebook import run
from src.utils.week2_report import write_report
from src.utils.paths import ROOT, REPORTS


def main():
    install_data_guard()
    run()
    train()
    result=subprocess.run([sys.executable,'-m','pytest','-p','tests.week2_policy','-q',
                           '--junitxml=reports/week2_pytest.xml'],cwd=ROOT,capture_output=True,text=True)
    (REPORTS/'week2_pytest_results.txt').write_text(result.stdout+result.stderr,encoding='utf-8')
    print(result.stdout)
    status=write_report()
    return 0 if status['overall_status']=='PASS' and result.returncode==0 else 1


if __name__=='__main__': sys.exit(main())
