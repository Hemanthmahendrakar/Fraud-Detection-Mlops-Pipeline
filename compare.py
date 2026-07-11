"""
Model Comparison
"""

import mlflow.pyfunc
from sklearn.metrics import accuracy_score

from config import REGISTERED_MODEL_NAME
from utils import print_header


def compare_models(new_model, X_test, y_test):
    """
    Compare newly trained model with Production model.

    Returns
    -------
    True  -> New model should be promoted
    False -> Keep existing Production model
    """

    print_header("MODEL COMPARISON")

    # -----------------------------------
    # New Model Accuracy
    # -----------------------------------

    new_predictions = new_model.predict(X_test)
    new_accuracy = accuracy_score(y_test, new_predictions)

    print(f"New Model Accuracy : {new_accuracy:.6f}")

    # -----------------------------------
    # Load Production Model
    # -----------------------------------

    try:

        production_model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{REGISTERED_MODEL_NAME}/Production"
        )

        production_predictions = production_model.predict(X_test)

        production_accuracy = accuracy_score(
            y_test,
            production_predictions,
        )

        print(f"Production Model Accuracy : {production_accuracy:.6f}")

    except Exception as e:

        print("\nNo Production Model Found.")
        print(f"Reason : {e}")

        print("Promoting New Model.")

        return True

    # -----------------------------------
    # Compare Models
    # -----------------------------------

    print("\nComparing Models...")

    if new_accuracy >= production_accuracy:

        print("New Model is Better.")
        print("Promotion Required.")

        return True

    print("Current Production Model is Better.")
    print("Promotion Not Required.")

    return False


def validate_metrics(metrics):
    """
    Validate evaluation metrics before model promotion.
    """

    accuracy = metrics.get("accuracy", 0)

    print_header("METRIC VALIDATION")
    print(f"Accuracy : {accuracy:.4f}")

    MIN_ACCURACY = 0.80

    if accuracy >= MIN_ACCURACY:
        print("Metric Validation Passed.")
        return True

    print("Metric Validation Failed.")
    return False
