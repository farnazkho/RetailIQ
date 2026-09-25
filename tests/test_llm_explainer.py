
from unittest.mock import MagicMock, patch

from src.llm_explainer import explain_prediction


def test_llm_explainer():
    mock_response = MagicMock()
    mock_response.output_text = (
        "The estimated purchase probability is 22.4%. "
        "This result is illustrative because 13 features "
        "were filled with training-data defaults."
    )

    with patch("src.llm_explainer.OpenAI") as mock_openai:
        mock_openai.return_value.responses.create.return_value = (
            mock_response
        )

        result = explain_prediction(
            description=(
                "A returning visitor browsed 20 product pages "
                "for 15 minutes on Saturday."
            ),
            prediction=False,
            probability=0.224,
            defaulted_features=["Feature"] * 13,
            api_key="test-api-key"
        )

    assert "22.4%" in result
    assert "illustrative" in result
    assert "13 features" in result

    mock_openai.return_value.responses.create.assert_called_once()
