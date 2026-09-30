"""Evaluation utilities for RetailIQ classification models."""

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X, y):
    """Evaluate a trained model using classification metrics."""

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(
            y, predictions, zero_division=0
        ),
        "recall": recall_score(
            y, predictions, zero_division=0
        ),
        "f1": f1_score(
            y, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y, probabilities),
    }

    return metrics


def get_confusion_matrix(model, X, y):
    """Return the confusion matrix for a trained model."""

    predictions = model.predict(X)

    return confusion_matrix(y, predictions)
