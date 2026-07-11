"""
Model Comparison
"""

import mlflow.pyfunc
from sklearn.metrics import accuracy_score

from config import REGISTERED_MODEL_NAME
from utils import print_header


def compare_models(new_model, X_test, y_test):
    """
    Compare the newly trained model with the current Production model.

    Returns:
        True  -> New model is better
        False -> Production model is better
    """

    print_header("MODEL COMPARISON")

    # -----------------------------
    # New Model Accuracy
    # -----------------------------
    new_predictions = new_model.predict(X_test)
    new_accuracy = accuracy_score(y_test, new_predictions)

    print(f"New Model Accuracy : {new_accuracy:.6f}")

    # -----------------------------
    # Load Production Model
    # -----------------------------
    try:

        production_model = mlflow.pyfunc.load_model(
            f"models:/{REGISTERED_MODEL_NAME}/Production"
        )

        production_predictions = production_model.predict(X_test)
        production_accuracy = accuracy_score(
            y_test,
            production_predictions
        )

        print(f"Production Accuracy : {production_accuracy:.6f}")

    except Exception:

        print("No Production model found.")
        print("Treating new model as best model.")

        return True

    # -----------------------------
    # Compare
    # -----------------------------
    if new_accuracy > production_accuracy:

        print("\nNew model is BETTER than Production.")

        return True

    else:

        print("\nCurrent Production model is BETTER.")

        return False
