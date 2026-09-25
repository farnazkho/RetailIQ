
import sys
from pathlib import Path

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.preprocessing import get_preprocessor
from tests.test_preprocessing import make_sample_data


def make_test_model():
    """Train a small model for automated testing."""
    X = make_sample_data()
    y = np.array([False, True, False])

    model = Pipeline([
        ("preprocessor", get_preprocessor(X.columns)),
        ("classifier", RandomForestClassifier(
            n_estimators=10,
            random_state=42
        ))
    ])

    model.fit(X, y)
    return model, X


def test_model_predicts_one_result_per_session():
    model, X = make_test_model()

    predictions = model.predict(X)

    assert len(predictions) == len(X)
    assert set(predictions).issubset({False, True})


def test_model_returns_valid_probabilities():
    model, X = make_test_model()

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (len(X), 2)
    assert np.all(
        (probabilities >= 0) &
        (probabilities <= 1)
    )
    assert np.allclose(
        probabilities.sum(axis=1),
        1
    )
