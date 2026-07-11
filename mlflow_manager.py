"""
MLflow Manager
"""

import mlflow
from mlflow import MlflowClient

from config import (
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
)

# ---------------------------------------------------
# Configure MLflow
# ---------------------------------------------------

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# ---------------------------------------------------
# Run Management
# ---------------------------------------------------

def start_run(run_name="Fraud Detection Pipeline"):
    """
    Start an MLflow run.
    """
    return mlflow.start_run(run_name=run_name)


def end_run():
    """
    End the active MLflow run.
    """
    if mlflow.active_run():
        mlflow.end_run()


# ---------------------------------------------------
# Logging
# ---------------------------------------------------

def log_params(params: dict):
    """
    Log multiple parameters.
    """
    mlflow.log_params(params)


def log_metrics(metrics: dict):
    """
    Log evaluation metrics.
    """
    mlflow.log_metrics(metrics)


# ---------------------------------------------------
# Model Promotion
# ---------------------------------------------------

def promote_model():
    """
    Promote the latest registered model
    to Production stage.
    """

    client = MlflowClient()

    latest_versions = client.get_latest_versions(
        REGISTERED_MODEL_NAME,
        stages=["None"]
    )

    if not latest_versions:
        print("\nNo registered model found to promote.")
        return

    latest_version = latest_versions[0]

    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=latest_version.version,
        stage="Production",
        archive_existing_versions=True
    )

    print("\n======================================")
    print("MODEL PROMOTION")
    print("======================================")
    print(f"Model Name    : {REGISTERED_MODEL_NAME}")
    print(f"Version       : {latest_version.version}")
    print("Stage         : Production")
    print("Promotion Successful")
