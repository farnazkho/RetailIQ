
import os
import json
from pathlib import Path

import joblib
import streamlit as st

from src.llm_explainer import explain_prediction
from src.llm_parser import extract_features
from src.prediction_pipeline import (
    prepare_prediction_input,
    predict_purchase
)


import re


def has_negative_page_count(description):
    """Detect explicitly negative product-page counts."""
    patterns = [
        r"\\bminus\\s+\\d+\\s+(?:product\\s+)?pages?\\b",
        r"(?<!\\w)-\\s*\\d+\\s+(?:product\\s+)?pages?\\b"
    ]
    return any(
        re.search(pattern, description, re.IGNORECASE)
        for pattern in patterns
    )

st.set_page_config(
    page_title="RetailIQ",
    page_icon="🛍️",
    layout="centered"
)

PROJECT_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    return joblib.load(
        PROJECT_DIR / "models" / "retailiq_model.joblib"
    )

@st.cache_data
def load_defaults():
    with open(
        PROJECT_DIR / "models" / "feature_defaults.json"
    ) as file:
        return json.load(file)

st.title("🛍️ RetailIQ")
st.subheader("AI-Powered Retail Purchase Predictor")

st.write(
    "Describe an online shopping session to estimate "
    "the likelihood of a purchase."
)

try:
    model = load_model()
    defaults = load_defaults()
    st.success("Prediction model loaded successfully!")
except Exception:
    st.error("The prediction model could not be loaded.")
    st.stop()

description = st.text_area(
    "Describe the shopping session",
    placeholder=(
        "A returning visitor browsed 20 product pages "
        "for 15 minutes on a Saturday."
    ),
    height=150
)

if st.button("Predict Purchase"):
    if not description.strip():
        st.warning("Please describe a shopping session.")
    elif has_negative_page_count(description):
        st.warning(
            "Product page count cannot be negative. "
            "Please enter a valid number of pages."
        )
    elif not os.getenv("OPENAI_API_KEY"):
        st.error("The OpenAI API key is not configured.")
    else:
        with st.spinner("Analyzing the shopping session..."):
            try:
                extracted = extract_features(
                    description,
                    list(model.feature_names_in_),
                    os.environ["OPENAI_API_KEY"]
                )

                model_input, defaulted = (
                    prepare_prediction_input(
                        extracted,
                        model,
                        defaults
                    )
                )

                prediction, probability = predict_purchase(
                    model,
                    model_input
                )

                explanation = explain_prediction(
                    description=description,
                    prediction=prediction,
                    probability=probability,
                    defaulted_features=defaulted,
                    api_key=os.environ["OPENAI_API_KEY"]
                )

                st.metric(
                    "Estimated purchase probability",
                    f"{probability:.1%}"
                )

                if prediction:
                    st.success("Prediction: Purchase")
                else:
                    st.info("Prediction: No purchase")

                st.warning(
                    f"{len(defaulted)} missing features were "
                    "filled with training-data defaults. "
                    "This prediction is illustrative."
                )

                st.subheader("AI Explanation")
                st.write(explanation)

                with st.expander("View defaulted features"):
                    st.write(defaulted)

            except ValueError as error:
                st.warning(str(error))
            except Exception:
                st.error(
                    "Unable to analyze this session. "
                    "Please try again."
                )
