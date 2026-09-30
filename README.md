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
- YAML-based experiment configuration
- MLflow experiment tracking and model comparison
- Automated testing with pytest

## Dataset

RetailIQ uses the UCI Online Shoppers Purchasing Intention dataset.

The dataset contains 12,330 shopping sessions and 18 columns: 17 predictor features and the target variable `Revenue`, which indicates whether a session resulted in a purchase.

The dataset is not committed to Git. It can be obtained from the UCI Machine Learning Repository and placed at:

```text
data/online_shoppers_intention.csv
```

## Architecture

The application follows this workflow:

```text
Natural-language shopping session
            |
            v
OpenAI feature extraction
            |
            v
Validation + training-data defaults
            |
            v
Random Forest prediction
            |
            v
Purchase probability
            |
            v
OpenAI-generated explanation
            |
            v
Streamlit user interface
```

The main project components are:

- `app.py` — Streamlit application and user workflow
- `src/llm_parser.py` — natural-language feature extraction
- `src/prediction_pipeline.py` — validation, default handling and prediction
- `src/llm_explainer.py` — AI-generated prediction explanations
- `src/preprocessing.py` — preprocessing pipeline
- `src/train.py` — YAML-driven training and MLflow experiment tracking
- `src/evaluate.py` — reusable model evaluation utilities
- `configs/config.yaml` — experiment hyperparameters
- `tests/` — automated preprocessing, model and interface tests

## Model Training and MLflow

Training configurations are stored in `configs/config.yaml` rather than hardcoded in the training script.

Five meaningfully different Random Forest configurations are trained and logged as separate MLflow runs:

- `rf_baseline`
- `rf_balanced`
- `rf_depth_10`
- `rf_depth_15`
- `rf_final`

Each run logs model hyperparameters, dataset information, evaluation metrics and a trained model artifact.

The training workflow uses `mlflow.search_runs()` to compare experiment results programmatically.

In the reproducible five-run experiment, the highest ROC-AUC was produced by `rf_depth_15`:

| Metric | Result |
|---|---:|
| Accuracy | 0.8792 |
| Precision | 0.5864 |
| Recall | 0.7461 |
| F1 score | 0.6567 |
| ROC-AUC | 0.9205 |

This result was measured on a stratified held-out set of 2,466 sessions.

An earlier project evaluation of the previously selected Random Forest model produced:

| Metric | Result |
|---|---:|
| Accuracy | 0.8670 |
| Precision | 0.5509 |
| Recall | 0.7644 |
| F1 score | 0.6404 |
| ROC-AUC | 0.9146 |

## Model Artifact

The trained `.joblib` model is intentionally not committed to the Git repository.

The deployment model is stored as a GitHub Release asset. When the local model file is unavailable, `app.py` downloads the model from the RetailIQ v1.0.0 Release and loads it for prediction.

This keeps large model artifacts out of Git while allowing the deployed Streamlit application to run.

## Technology

- Python 3.11
- pandas and NumPy
- scikit-learn
- joblib
- OpenAI API
- Streamlit
- MLflow
- PyYAML
- pytest

## Run Locally

1. Clone the repository:

```bash
git clone https://github.com/farnazkho/RetailIQ.git
cd RetailIQ
```

2. Create a Python 3.11 virtual environment:

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

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set the `OPENAI_API_KEY` environment variable using your own OpenAI API key.

Never commit an actual API key or `.env` file to GitHub.

5. Start the application:

```bash
streamlit run app.py
```

The application downloads the released model automatically if the local model artifact is not present.

## Training

To reproduce the MLflow experiments, first place the UCI dataset at:

```text
data/online_shoppers_intention.csv
```

Then run the training workflow from the project root:

```bash
python -m src.train
```

The script reads the experiment hyperparameters from `configs/config.yaml`, trains the configured models, logs the runs to MLflow and uses `mlflow.search_runs()` to identify the run with the highest ROC-AUC.

## Example

Input:

> A returning visitor browsed 20 product pages for 15 minutes on a Saturday.

The application extracts the available shopping-session features, fills unspecified features with saved training-data defaults, generates a purchase probability and provides an AI-generated explanation.

## Testing

The project currently has **15 passing automated tests** covering preprocessing, model behavior, prediction validation and interface behavior.

The preprocessing tests cover:

- missing-value handling
- categorical encoding
- numerical scaling
- input-data immutability
- unknown-category handling
- feature assignment

The model tests cover prediction behavior, probability output and a minimum performance threshold.

The interface and prediction tests cover natural-language parsing, valid input, incomplete input and invalid values.

Run the complete test suite with:

```bash
pytest tests/ -v
```

The deployed application was also manually tested with a complete visitor description, an incomplete description and an invalid negative product-page count.

## Limitations

- The model was trained on a historical public dataset and has not been validated on live retail traffic.
- Natural-language descriptions may omit information, requiring default feature values.
- Defaulted features reduce the amount of session-specific information available to the model.
- LLM extraction and explanations may contain errors.
- Purchase probabilities are estimates, not guarantees.
- The OpenAI API requires a separately configured key and may incur usage charges.

## Security

API keys must be supplied through environment variables or secure deployment secrets. Credentials are not stored in the repository.

## Reflection

RetailIQ demonstrates how traditional machine learning and generative AI can be combined in a practical end-to-end application. One of the main challenges was converting free-form user descriptions into the structured 17-feature input expected by the machine learning model while still handling incomplete or invalid information safely.

Experiment tracking also became an important part of the project. Moving the model configurations into YAML and logging five reproducible runs with MLflow made the model-development process easier to compare and reproduce. Automated tests helped identify issues introduced when the model artifact was removed from Git, including dependencies between test fixtures and the production model file.

A key lesson from the project was that model performance is only one part of a deployable machine learning system. Reproducible training, testing, secure API-key handling, artifact management, input validation and a usable interface are also necessary parts of an end-to-end ML application.

Future improvements could include testing on newer retail data, expanding the natural-language interface, reducing reliance on default feature values, monitoring prediction quality over time and evaluating additional model families.
