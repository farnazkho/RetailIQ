"""Train and compare RetailIQ models with MLflow."""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import get_preprocessor


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_DIR / "configs" / "config.yaml"


def load_config():
    """Load project settings and experiments from YAML."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def calculate_metrics(model, X_test, y_test):
    """Calculate classification metrics."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def train_experiments(data_path):
    """Train all YAML configurations and log them to MLflow."""
    config = load_config()

    data = pd.read_csv(data_path)

    target = config["data"]["target"]
    X = data.drop(columns=[target])
    y = data[target].astype(bool)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["project"]["random_state"],
        stratify=y,
    )

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    for experiment_name, params in config["experiments"].items():

        classifier = RandomForestClassifier(**params)

        pipeline = Pipeline([
            ("preprocessor", get_preprocessor(X.columns)),
            ("classifier", classifier),
        ])

        with mlflow.start_run(run_name=experiment_name):

            mlflow.log_param(
                "dataset",
                config["data"]["dataset_name"]
            )
            mlflow.log_param(
                "target",
                target
            )
            mlflow.log_param(
                "test_size",
                config["data"]["test_size"]
            )
            mlflow.log_params(params)

            pipeline.fit(X_train, y_train)

            metrics = calculate_metrics(
                pipeline,
                X_test,
                y_test
            )

            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model"
            )

            print(
                experiment_name,
                "ROC-AUC:",
                round(metrics["roc_auc"], 4)
            )

    return X_test, y_test


def find_best_run():
    """Use MLflow to identify the highest ROC-AUC run."""
    config = load_config()

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    experiment = mlflow.get_experiment_by_name(
        config["mlflow"]["experiment_name"]
    )

    if experiment is None:
        raise ValueError("MLflow experiment not found.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.roc_auc DESC"],
    )

    if runs.empty:
        raise ValueError("No MLflow runs were found.")

    return runs.iloc[0]


def save_best_model(best_run):
    """Download and save the best MLflow model."""
    model_uri = f"runs:/{best_run['run_id']}/model"

    best_model = mlflow.sklearn.load_model(model_uri)

    model_dir = PROJECT_DIR / "models"
    model_dir.mkdir(exist_ok=True)

    output_path = model_dir / "retailiq_model.joblib"

    joblib.dump(best_model, output_path)

    print("Best model saved to:", output_path)


if __name__ == "__main__":

    data_path = (
        PROJECT_DIR
        / "data"
        / "online_shoppers_intention.csv"
    )

    train_experiments(data_path)

    best_run = find_best_run()

    print("\nBest MLflow run:")
    print("Run ID:", best_run["run_id"])
    print(
        "ROC-AUC:",
        round(best_run["metrics.roc_auc"], 4)
    )

    save_best_model(best_run)
