import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.preprocessing import get_preprocessor
from tests.test_preprocessing import make_sample_data


PROJECT_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture
def model():
    """Create a small deterministic model for automated tests."""
    X = make_sample_data()
    y = np.array([False, True, False])

    test_model = Pipeline([
        ("preprocessor", get_preprocessor(X.columns)),
        ("classifier", RandomForestClassifier(
            n_estimators=10,
            random_state=42
        ))
    ])

    test_model.fit(X, y)

    return test_model


@pytest.fixture
def defaults():
    """Load saved feature defaults used by the prediction pipeline."""
    with open(
        PROJECT_DIR / "models" / "feature_defaults.json"
    ) as file:
        return json.load(file)
