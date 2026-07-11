"""
MLflow Manager
"""

import mlflow
import mlflow.sklearn

from config import (
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
)

# Configure MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def start_run(run_name="Fraud Detection Pipeline"):
    """Start MLflow Run"""
    return mlflow.start_run(run_name=run_name)


def end_run():
    """End MLflow Run"""
    mlflow.end_run()


def log_params(params: dict):
    """Log Parameters"""
    mlflow.log_params(params)


def log_metrics(metrics: dict):
    """Log Metrics"""
    mlflow.log_metrics(metrics)


def register_model(model):
    """
    Log and Register Model
    """

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=REGISTERED_MODEL_NAME,
    )

    print("\nModel Registered Successfully")
    print("Model URI :", model_info.model_uri)

    return model_info
