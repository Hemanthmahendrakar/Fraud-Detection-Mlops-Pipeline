"""
MLflow Manager
"""

import mlflow
import mlflow
from config import *

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
def setup_mlflow():
    # Use SQLite backend
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # Create / Use experiment
    mlflow.set_experiment("Fraud Detection MLOps")
