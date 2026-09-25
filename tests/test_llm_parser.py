
import json
from unittest.mock import MagicMock, patch

from src.llm_parser import extract_features


def test_llm_parser_extracts_features(model):
    feature_names = list(model.feature_names_in_)

    expected = {
        feature: None for feature in feature_names
    }
    expected.update({
        "VisitorType": "Returning_Visitor",
        "ProductRelated": 20,
        "ProductRelated_Duration": 900,
        "Weekend": True
    })

    mock_response = MagicMock()
    mock_response.output_text = json.dumps(expected)

    with patch("src.llm_parser.OpenAI") as mock_openai:
        mock_openai.return_value.responses.create.return_value = (
            mock_response
        )

        result = extract_features(
            "A returning visitor browsed 20 product pages "
            "for 15 minutes on Saturday.",
            feature_names,
            "test-api-key"
        )

    assert result == expected
    assert len(result) == 17
    assert result["ProductRelated_Duration"] == 900
    mock_openai.return_value.responses.create.assert_called_once()
