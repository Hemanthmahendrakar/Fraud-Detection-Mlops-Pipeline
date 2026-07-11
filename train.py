"""
Model Training
"""

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

from config import (
    REGISTERED_MODEL_NAME,
    RANDOM_STATE,
    N_ESTIMATORS,
    MAX_DEPTH,
    MIN_SAMPLES_SPLIT,
    MIN_SAMPLES_LEAF,
    NEW_MODEL,
)

from utils import save_model, print_header


def train_model(X_train, y_train):
    """
    Train Random Forest Model.
    Save locally.
    Log parameters.
    Register model in MLflow.

    Returns
    -------
    model : RandomForestClassifier
    """

    print_header("MODEL TRAINING")

    print("Creating Random Forest Model...")

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        n_jobs=-1,
    )

    print("Training Model...")

    model.fit(X_train, y_train)

    print("Training Completed Successfully")

    # ---------------------------------
    # Save Model Locally
    # ---------------------------------

    print("\nSaving Model...")

    save_model(model, NEW_MODEL)

    print(f"Model Saved Successfully")
    print(f"Location : {NEW_MODEL}")

    # ---------------------------------
    # Log Parameters
    # ---------------------------------

    mlflow.log_param("algorithm", "RandomForest")
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("max_depth", MAX_DEPTH)
    mlflow.log_param("min_samples_split", MIN_SAMPLES_SPLIT)
    mlflow.log_param("min_samples_leaf", MIN_SAMPLES_LEAF)
    mlflow.log_param("random_state", RANDOM_STATE)

    # ---------------------------------
    # Log Model & Register
    # ---------------------------------

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=REGISTERED_MODEL_NAME,
    )

    print("\nModel Logged to MLflow")
    print("Model Registered Successfully")
    print(f"Model URI : {model_info.model_uri}")

    return model
