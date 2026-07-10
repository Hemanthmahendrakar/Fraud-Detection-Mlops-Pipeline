"""
Model Training
"""
import mlflow
from sklearn.ensemble import RandomForestClassifier
import mlflow.sklearn
from config import REGISTERED_MODEL_NAME
from preprocess import preprocess_data

from config import (
    RANDOM_STATE,
    N_ESTIMATORS,
    MAX_DEPTH,
    MIN_SAMPLES_SPLIT,
    MIN_SAMPLES_LEAF,
    NEW_MODEL
)

from utils import (
    save_model,
    print_header
)


def train_model(X_train,y_train):

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
        n_jobs=-1
    )
    # ----------------------------
    # Log Parameters to MLflow
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
    # Save Model
    # ----------------------------
    save_model(model, NEW_MODEL)

    print(f"Model Saved Successfully")
    print(f"Location : {NEW_MODEL}")
    # ----------------------------
    # Log Model to MLflow
    # ----------------------------


    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="random_forest_model",
        registered_model_name=REGISTERED_MODEL_NAME
    )

print("Model Registered Successfully")
print(model_info.model_uri)

    return (model)


if __name__ == "__main__":

    train_model()
