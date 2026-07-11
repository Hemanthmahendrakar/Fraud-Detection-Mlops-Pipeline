"""
model_loader.py
Production Model Loader
"""

import os
import joblib
import mlflow
import mlflow.pyfunc

from config import (
    REGISTERED_MODEL_NAME,
    NEW_MODEL,
)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
LOCAL_MODEL = os.path.join(BASE_DIR, NEW_MODEL)


def load_objects():
    """
    Load production model.

    Priority:
    1. MLflow Production Model
    2. Local Joblib Model
    """

    # -----------------------------
    # Try MLflow Production Model
    # -----------------------------
    try:

        model_uri = f"models:/{REGISTERED_MODEL_NAME}/Production"

        model = mlflow.pyfunc.load_model(model_uri)

        print("=" * 60)
        print("Production Model Loaded From MLflow")
        print(model_uri)
        print("=" * 60)

        return model, None, None

    except Exception as e:

        print("=" * 60)
        print("Unable to load MLflow Production Model")
        print(e)
        print("Trying Local Joblib Model...")
        print("=" * 60)

    # -----------------------------
    # Fallback
    # -----------------------------
    if not os.path.exists(LOCAL_MODEL):
        raise FileNotFoundError(
            f"Local model not found: {LOCAL_MODEL}"
        )

    loaded = joblib.load(LOCAL_MODEL)

    model = None
    scaler = None
    encoder = None

    if isinstance(loaded, dict):

        for key, value in loaded.items():

            key = key.lower()

            if hasattr(value, "predict"):
                model = value

            elif hasattr(value, "transform") and "scal" in key:
                scaler = value

            elif hasattr(value, "transform") and "enc" in key:
                encoder = value

    else:
        model = loaded

    print("=" * 60)
    print("Production Model Loaded From Local Joblib")
    print("=" * 60)

    return model, scaler, encoder
