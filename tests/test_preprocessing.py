
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)

from src.preprocessing import get_preprocessor


def test_preprocessor_has_all_features():
    """Check that all 17 input features are assigned."""

    feature_columns = [
        "Administrative",
        "Administrative_Duration",
        "Informational",
        "Informational_Duration",
        "ProductRelated",
        "ProductRelated_Duration",
        "BounceRates",
        "ExitRates",
        "PageValues",
        "SpecialDay",
        "Month",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType",
        "VisitorType",
        "Weekend"
    ]

    preprocessor = get_preprocessor(feature_columns)

    numerical_cols = preprocessor.transformers[0][2]
    categorical_cols = preprocessor.transformers[1][2]

    assert len(numerical_cols) == 10
    assert len(categorical_cols) == 7

    assert set(numerical_cols + categorical_cols) == set(
        feature_columns
    )


import numpy as np
import pandas as pd


def make_sample_data():
    """Create a small dataset with all RetailIQ features."""
    return pd.DataFrame({
        "Administrative": [1, 3, 2],
        "Administrative_Duration": [10.0, 30.0, 20.0],
        "Informational": [0, 1, 2],
        "Informational_Duration": [0.0, 15.0, 25.0],
        "ProductRelated": [5, 20, 10],
        "ProductRelated_Duration": [100.0, 500.0, 300.0],
        "BounceRates": [0.01, 0.02, 0.03],
        "ExitRates": [0.03, 0.04, 0.05],
        "PageValues": [0.0, 10.0, 20.0],
        "SpecialDay": [0.0, 0.5, 1.0],
        "Month": ["Feb", "Mar", "May"],
        "OperatingSystems": [1, 2, 3],
        "Browser": [1, 2, 1],
        "Region": [1, 2, 3],
        "TrafficType": [1, 2, 3],
        "VisitorType": [
            "Returning_Visitor",
            "New_Visitor",
            "Returning_Visitor"
        ],
        "Weekend": [False, True, False]
    })


def test_preprocessor_output_has_no_nan():
    """Valid sample data should transform without NaN."""
    X = make_sample_data()
    preprocessor = get_preprocessor(X.columns)

    result = preprocessor.fit_transform(X)

    assert not np.isnan(result).any()


def test_unknown_category_is_handled():
    """An unfamiliar visitor category should not crash."""
    X = make_sample_data()
    preprocessor = get_preprocessor(X.columns)
    preprocessor.fit(X)

    new_data = X.iloc[[0]].copy()
    new_data["VisitorType"] = "Unknown_Visitor"

    result = preprocessor.transform(new_data)

    assert result.shape[0] == 1


def test_numerical_features_are_scaled():
    """Numerical features should have mean near zero."""
    X = make_sample_data()
    preprocessor = get_preprocessor(X.columns)

    result = preprocessor.fit_transform(X)
    numerical_result = result[:, :10]

    assert np.allclose(
        numerical_result.mean(axis=0),
        0,
        atol=1e-7
    )
