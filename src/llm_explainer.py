
from openai import OpenAI


def explain_prediction(
    description,
    prediction,
    probability,
    defaulted_features,
    api_key
):
    """Explain the model's prediction in plain English."""

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model="gpt-4.1-nano",
        instructions="""
        You are RetailIQ, a retail analytics assistant.

        Explain the supplied prediction in plain English.

        Rules:
        - Use the exact probability provided.
        - Never invent or change model results.
        - Explain that missing features were filled
          with training-data defaults.
        - Do not claim to know which features
          caused the prediction.
        - Make clear that the result is illustrative.
        - Keep your response under 100 words.
        """,
        input=f"""
        Shopping session: {description}
        Prediction: {"Purchase" if prediction else "No purchase"}
        Purchase probability: {probability:.1%}
        Number of defaulted features: {len(defaulted_features)}

        Explain this result.
        """,
        max_output_tokens=180
    )

    return response.output_text
