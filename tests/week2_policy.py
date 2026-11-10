"""Explicit Week 2 pytest mode: preserve Week 1 files and enforce data restriction."""
from src.data.week2_inputs import install_data_guard


def pytest_sessionstart(session):
    install_data_guard()
