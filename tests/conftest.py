
import json
from pathlib import Path

import joblib
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture
def model():
    return joblib.load(
        PROJECT_DIR / "models" / "retailiq_model.joblib"
    )


@pytest.fixture
def defaults():
    with open(
        PROJECT_DIR / "models" / "feature_defaults.json"
    ) as file:
        return json.load(file)
