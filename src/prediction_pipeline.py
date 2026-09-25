
import json
import pandas as pd


def prepare_prediction_input(extracted, model, defaults):
    """Validate extracted values and prepare model input."""

    expected = list(model.feature_names_in_)

    if set(extracted) != set(expected):
        raise ValueError("The extracted feature names do not match the model.")

    # Validate the four features we have tested so far
    visitor = extracted.get("VisitorType")
    pages = extracted.get("ProductRelated")
    duration = extracted.get("ProductRelated_Duration")
    weekend = extracted.get("Weekend")

    if visitor not in (
        "Returning_Visitor", "New_Visitor", "Other"
    ):
        raise ValueError("Please tell us whether the visitor is new or returning, how many product pages they viewed, how long they browsed, and whether it was a weekend.")

    if type(pages) is not int or pages < 0:
        raise ValueError("Invalid product page count.")

    if type(duration) not in (int, float) or duration < 0:
        raise ValueError("Invalid product page duration.")

    if type(weekend) is not bool:
        raise ValueError("Missing or invalid weekend value.")

    # Fill missing values with saved training-data defaults
    complete = defaults.copy()
    defaulted = []

    for feature in expected:
        value = extracted[feature]

        if value is None:
            defaulted.append(feature)
        else:
            complete[feature] = value

    model_input = pd.DataFrame(
        [complete],
        columns=expected
    )

    return model_input, defaulted


def predict_purchase(model, model_input):
    """Generate a purchase prediction."""

    prediction = bool(model.predict(model_input)[0])
    probability = float(
        model.predict_proba(model_input)[0, 1]
    )

    return prediction, probability
