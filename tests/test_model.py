import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)

from src.preprocessing import get_preprocessor
from tests.test_preprocessing import make_sample_data


def make_test_model():
    """Train a small deterministic model for automated testing."""
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
    return model, X, y


def test_model_predicts_one_result_per_session():
    """Model should return one valid prediction per session."""
    model, X, _ = make_test_model()

    predictions = model.predict(X)

    assert len(predictions) == len(X)
    assert set(predictions).issubset({False, True})


def test_model_returns_valid_probabilities():
    """Model probabilities should have the expected shape and range."""
    model, X, _ = make_test_model()

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


def test_model_meets_minimum_performance_threshold():
    """Model should meet a minimum accuracy threshold on a known sample."""
    model, X, y = make_test_model()

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    assert accuracy >= 0.80
