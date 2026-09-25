
import json
from openai import OpenAI


def extract_features(description, feature_names, api_key):
    """Extract retail features from natural language."""

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model="gpt-4.1-nano",
        instructions=f"""
        Extract shopping-session information into JSON.

        Return exactly these fields:
        {", ".join(feature_names)}

        ProductRelated is the number of product pages.
        ProductRelated_Duration is time spent on
        product pages, measured in seconds.

        Example:
        "20 product pages for 15 minutes" means
        ProductRelated = 20
        ProductRelated_Duration = 900.

        VisitorType must be Returning_Visitor,
        New_Visitor, Other, or null.

        Weekend must be true, false, or null.

        
Extract only explicitly provided information.

IMPORTANT: Preserve negative numbers exactly.
"minus 5 product pages" means ProductRelated = -5,
NOT 5.

Never correct, normalize, or silently change
an invalid numerical value.

For example:
"minus 5 product pages for 15 minutes"
means ProductRelated = -5 and
ProductRelated_Duration = 900.

        Use null for missing information.
        Never invent website analytics.
        Return valid JSON only.
        """,
        input="Extract all features as JSON:\n" + description,
        text={"format": {"type": "json_object"}},
        max_output_tokens=600
    )

    return json.loads(response.output_text)
