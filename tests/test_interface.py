
from src.interface import predict_purchase
from tests.test_model import make_test_model


def test_valid_session_prediction():
    """A valid session should return a prediction and probability."""

    model, X = make_test_model()

    result = predict_purchase(model, X.iloc[[0]])

    assert "prediction" in result
    assert "purchase_probability" in result

    assert isinstance(result["prediction"], bool)

    assert 0 <= result["purchase_probability"] <= 1


def test_empty_session_returns_error():
    """Empty input should return a helpful error."""

    model, X = make_test_model()

    result = predict_purchase(model, X.iloc[0:0])

    assert "error" in result
    assert "Please provide shopping session data" in result["error"]
