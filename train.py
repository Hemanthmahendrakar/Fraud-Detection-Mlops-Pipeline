"""
Model Training
"""

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
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

from preprocess import preprocess_data
from utils import save_model, print_header


def train_model(X_train, y_train):
    """Train a Random Forest model and register it with MLflow."""

    print_header("MODEL TRAINING")
    print("Training Random Forest Model...")

    # ----------------------------
    # Create Model
    # ----------------------------
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        n_jobs=-1,
    )

    with mlflow.start_run():

        # ----------------------------
        # Log Parameters
        # ----------------------------
        mlflow.log_param("algorithm", "RandomForest")
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("max_depth", MAX_DEPTH)
        mlflow.log_param("min_samples_split", MIN_SAMPLES_SPLIT)
        mlflow.log_param("min_samples_leaf", MIN_SAMPLES_LEAF)

        # ----------------------------
        # Train Model
        # ----------------------------
        model.fit(X_train, y_train)
        print("Training Completed Successfully")

        # ----------------------------
        # Save Model Locally
        # ----------------------------
        save_model(model, NEW_MODEL)

        print("Model Saved Successfully")
        print(f"Location: {NEW_MODEL}")

        # ----------------------------
        # Log & Register Model
        # ----------------------------
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random_forest_model",   # Use 'name' if using MLflow 3.x
            registered_model_name=REGISTERED_MODEL_NAME,
        )

    # ----------------------------
    # Promote Latest Version
    # ----------------------------
    client = MlflowClient()

    latest_version = client.get_latest_versions(
        REGISTERED_MODEL_NAME,
        stages=["None"],
    )[0]

    print(f"Latest Version: {latest_version.version}")

    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=latest_version.version,
        stage="Production",
        archive_existing_versions=True,
    )

    print("Model promoted to Production.")
    print("Model Registered Successfully")
    print(f"Model URI: {model_info.model_uri}")

    return model


if __name__ == "__main__":

    # Load and preprocess data
    X_train, X_test, y_train, y_test = preprocess_data()

    # Train model
    train_model(X_train, y_train)
