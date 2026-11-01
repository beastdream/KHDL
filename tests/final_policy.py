"""Final E2E tests read reports/Train/Valid, never reopen the official holdout."""
from src.data.final_inputs import install_data_guard


def pytest_sessionstart(session):
    install_data_guard()
