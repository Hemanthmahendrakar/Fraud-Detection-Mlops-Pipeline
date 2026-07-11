"""
MLflow Manager
"""

import mlflow
import mlflow.sklearn

from mlflow import MlflowClient

from config import (
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
)

# ----------------------------------
# Configure MLflow
# ----------------------------------

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# ----------------------------------
# Run Management
# ----------------------------------

def start_run(run_name="Fraud Detection Pipeline"):
    """Start MLflow Run"""
    return mlflow.start_run(run_name=run_name)


def end_run():
    """End MLflow Run"""
    mlflow.end_run()


# ----------------------------------
# Logging
# ----------------------------------

def log_params(params):
    """Log Parameters"""
    mlflow.log_params(params)


def log_metrics(metrics):
    """Log Metrics"""
    mlflow.log_metrics(metrics)


# ----------------------------------
# Model Registration
# ----------------------------------

def register_model(model):
    """
    Register Model in MLflow Registry
    """

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=REGISTERED_MODEL_NAME,
    )

    print("\nModel Registered Successfully")
    print(f"Model URI : {model_info.model_uri}")

    return model_info


# ----------------------------------
# Model Promotion
# ----------------------------------

def promote_model():
    """
    Promote the latest registered model
    to Production.
    """

    client = MlflowClient()

    latest_versions = client.get_latest_versions(
        REGISTERED_MODEL_NAME,
        stages=["None"],
    )

    if len(latest_versions) == 0:

        print("No registered model available for promotion.")
        return

    latest_version = latest_versions[0]

    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=latest_version.version,
        stage="Production",
        archive_existing_versions=True,
    )

    print(
        f"\nModel Version {latest_version.version} promoted to Production."
    )
