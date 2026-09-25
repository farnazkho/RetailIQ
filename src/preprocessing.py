
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_COLS = [
    "Month",
    "VisitorType",
    "Weekend",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType"
]

def get_preprocessor(feature_columns):
    """Build the RetailIQ preprocessing pipeline."""

    numerical_cols = [
        col for col in feature_columns
        if col not in CATEGORICAL_COLS
    ]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(
                handle_unknown="ignore"
            ), CATEGORICAL_COLS)
        ]
    )
