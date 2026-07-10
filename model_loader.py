"""
model_loader.py
"""

import os
import joblib

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")


def load_objects():
    """
    Loads model, scaler and encoder.

    Today -> Local Joblib
    Future -> MLflow Registry
    """

    model = None
    scaler = None
    encoder = None

    loaded = joblib.load(MODEL_PATH)

    if isinstance(loaded, dict):

        for key, value in loaded.items():

            key = key.lower()

            if hasattr(value, "predict"):
                model = value

            elif hasattr(value, "transform") and (
                "scal" in key or "norm" in key
            ):
                scaler = value

            elif hasattr(value, "transform") and (
                "enc" in key or "label" in key
            ):
                encoder = value

    else:
        model = loaded

    print("Production Model Loaded Successfully")

    return model, scaler, encoder