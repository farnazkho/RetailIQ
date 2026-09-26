# RetailIQ: AI-Powered Purchase Prediction

RetailIQ is an end-to-end machine learning application that predicts whether an online shopper will make a purchase. Users describe a visitor's browsing session in natural language, and the application uses OpenAI to extract relevant features, a trained Random Forest model to predict purchase probability, and AI to explain the result.

**Live application:** https://retailiq-jpx84eqmuiyuunt8i87vo7.streamlit.app/

## Features

- Natural-language visitor descriptions
- LLM-powered extraction of 17 shopping-session features
- Random Forest purchase predictions and probabilities
- AI-generated explanations
- Validation of incomplete and invalid inputs
- Transparent disclosure of features filled with training-data defaults

## Dataset

UCI Online Shoppers Purchasing Intention dataset, containing 12,330 shopping sessions. The target variable, Revenue, indicates whether a session resulted in a purchase.

## Model and Evaluation

A tuned Random Forest classifier was selected using validation performance. Final evaluation was performed on a held-out test set.

| Metric | Test result |
|---|---:|
| Accuracy | 0.8670 |
| Precision | 0.5509 |
| Recall | 0.7644 |
| F1 score | 0.6404 |
| ROC-AUC | 0.9146 |

Five MLflow runs were logged during model development.

## Technology

- Python, pandas and NumPy
- scikit-learn and joblib
- OpenAI API
- Streamlit
- MLflow for experiment tracking
- pytest for automated testing

## Run Locally

1. Clone the repository:

```bash
git clone https://github.com/farnazkho/RetailIQ.git
cd RetailIQ
```

2. Create a virtual environment using Python 3.11, activate it and install dependencies:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the requirements:

```bash
pip install -r requirements.txt
```

3. Set the OPENAI_API_KEY environment variable using your own OpenAI API key. Never commit an actual key to GitHub.

4. Start the app:

```bash
streamlit run app.py
```

## Example

Input:

> A returning visitor browsed 20 product pages for 15 minutes on a Saturday.

The deployed application returned a 22.4% purchase probability and a no-purchase prediction. Thirteen unspecified features were filled using training-data defaults. Results depend on the supplied session information and are illustrative.

## Testing

The project has 12 passing automated tests. The deployed application was also manually checked using a complete visitor description, an incomplete description and an invalid negative product-page count.

Run automated tests with:

```bash
pytest -q
```

## Limitations

- The model was trained on a historical public dataset and has not been validated on live retail traffic.
- Natural-language descriptions may omit information, requiring default feature values.
- LLM extraction and explanations may contain errors.
- Purchase probabilities are estimates, not guarantees.
- The OpenAI API requires a separately configured key and may incur usage charges.

## Security

API keys must be supplied through environment variables or secure deployment secrets. Do not commit credentials to the repository.
