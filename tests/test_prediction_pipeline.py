
import pytest

from src.prediction_pipeline import prepare_prediction_input


def test_negative_product_pages(model, defaults):
    extracted = {
        feature: None
        for feature in model.feature_names_in_
    }

    extracted.update({
        "VisitorType": "Returning_Visitor",
        "ProductRelated": -5,
        "ProductRelated_Duration": 900,
        "Weekend": True
    })

    with pytest.raises(
        ValueError,
        match="Invalid product page count"
    ):
        prepare_prediction_input(
            extracted,
            model,
            defaults
        )



def test_missing_visitor_type(model, defaults):
    extracted = {
        feature: None
        for feature in model.feature_names_in_
    }

    extracted.update({
        "ProductRelated": 20,
        "ProductRelated_Duration": 900,
        "Weekend": True
    })

    with pytest.raises(
        ValueError,
        match="Please tell us whether the visitor"
    ):
        prepare_prediction_input(
            extracted,
            model,
            defaults
        )
