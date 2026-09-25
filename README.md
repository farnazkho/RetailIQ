# RetailIQ: AI-Powered Retail Purchase Predictor

RetailIQ predicts whether an online shopping session
is likely to result in a purchase.

## How It Works

1. The user describes a shopping session in natural language.
2. OpenAI extracts structured shopping-session features.
3. The application validates the extracted information.
4. Missing features are filled with training-data defaults.
5. A trained Random Forest predicts purchase probability.
6. OpenAI explains the prediction in plain English.

Predictions based on incomplete information are illustrative.

## Dataset

UCI Online Shoppers Purchasing Intention Dataset

- 12,330 shopping sessions
- 17 input features
- Target: Revenue (purchase or no purchase)

## Machine Learning

Five model configurations were compared using MLflow.

Selected model: Tuned Random Forest

Final test-set performance:

| Metric | Score |
|---|---:|
| Accuracy | 0.8670 |
| Precision | 0.5509 |
| Recall | 0.7644 |
| F1 | 0.6404 |
| ROC-AUC | 0.9146 |

## Technologies

- Python
- pandas
- scikit-learn
- OpenAI API
- Streamlit
- MLflow
- pytest

## Testing

The project currently has 12 passing automated tests,
including tests for invalid inputs and simulated
OpenAI responses.

## Limitations

- Predictions depend on the information provided.
- Missing values are filled using training-data defaults.
- LLM extraction may occasionally be incorrect.
- The application is a demonstration, not a production
  decision-making system.

## Setup

Installation instructions and environment configuration
will be added before GitHub publication.
