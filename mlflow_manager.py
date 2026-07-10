"""
MLflow Manager
"""

import mlflow

def setup_mlflow():
    # Use SQLite backend
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # Create / Use experiment
    mlflow.set_experiment("Fraud Detection MLOps")