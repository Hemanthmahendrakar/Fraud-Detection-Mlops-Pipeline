"""
Model Evaluation
"""

import pandas as pd
import matplotlib.pyplot as plt
import mlflow

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from config import (
    REPORTS_DIR,
    METRICS_CSV,
    CLASSIFICATION_REPORT_FILE,
    CONFUSION_MATRIX_IMAGE,
)

from utils import create_directory, print_header


def evaluate_model(model, X_test, y_test):
    """
    Evaluate trained model.
    Save reports.
    Log metrics and artifacts to MLflow.
    """

    print_header("MODEL EVALUATION")

    create_directory(REPORTS_DIR)

    # ---------------------------------------
    # Predictions
    # ---------------------------------------

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ---------------------------------------
    # Metrics
    # ---------------------------------------

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
    }

    # ---------------------------------------
    # Print Metrics
    # ---------------------------------------

    print("\nEvaluation Metrics")
    print("-" * 40)

    for key, value in metrics.items():
        print(f"{key:<12}: {value:.4f}")

    # ---------------------------------------
    # Log Metrics
    # ---------------------------------------

    mlflow.log_metrics(metrics)

    # ---------------------------------------
    # Save Metrics CSV
    # ---------------------------------------

    pd.DataFrame([metrics]).to_csv(
        METRICS_CSV,
        index=False,
    )

    # ---------------------------------------
    # Classification Report
    # ---------------------------------------

    report = classification_report(
        y_test,
        y_pred,
    )

    with open(CLASSIFICATION_REPORT_FILE, "w") as file:
        file.write(report)

    # ---------------------------------------
    # Confusion Matrix
    # ---------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
    )

    disp.plot()

    plt.savefig(CONFUSION_MATRIX_IMAGE)

    plt.close()

    # ---------------------------------------
    # Log Artifacts
    # ---------------------------------------

    mlflow.log_artifact(METRICS_CSV)
    mlflow.log_artifact(CLASSIFICATION_REPORT_FILE)
    mlflow.log_artifact(CONFUSION_MATRIX_IMAGE)

    print("\nReports Saved Successfully")

    return metrics
