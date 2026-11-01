"""Week 3-only suite; legacy tests run separately under their unchanged policy."""
from src.data.week3_inputs import install_data_guard


def pytest_sessionstart(session):
    install_data_guard()

