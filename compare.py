"""
Model Comparison
"""

import os
import shutil
import pandas as pd

from config import (
    METRICS_CSV,
    PRODUCTION_METRICS,
    NEW_MODEL,
    PRODUCTION_MODEL,
    MIN_ACCURACY,
    MIN_PRECISION,
    MIN_RECALL,
    MIN_F1_SCORE,
)
from utils import print_header


def compare_models():

    print_header("MODEL COMPARISON")
    def validate_metrics(metrics):

    print("\nChecking Model Quality...")

    if metrics["Accuracy"] < MIN_ACCURACY:
        print("Accuracy below threshold")
        return False

    if metrics["Precision"] < MIN_PRECISION:
        print("Precision below threshold")
        return False

    if metrics["Recall"] < MIN_RECALL:
        print("Recall below threshold")
        return False

    if metrics["F1 Score"] < MIN_F1_SCORE:
        print("F1 Score below threshold")
        return False

    print("Model passed all validation checks")
    return True

    # -----------------------------
    # First Model
    # -----------------------------
    if not os.path.exists(PRODUCTION_METRICS):

        print("No Production Model Found")
        print("Promoting First Model to Production...")

        shutil.copy(NEW_MODEL, PRODUCTION_MODEL)

        shutil.copy(METRICS_CSV, PRODUCTION_METRICS)

        print("Production Model Created")

        return True

    # -----------------------------
    # Load Metrics
    # -----------------------------
    new_metrics = pd.read_csv(METRICS_CSV)

    production_metrics = pd.read_csv(PRODUCTION_METRICS)

    new_f1 = new_metrics["F1 Score"][0]

    old_f1 = production_metrics["F1 Score"][0]

    print(f"Production F1 : {old_f1:.4f}")
    print(f"Candidate  F1 : {new_f1:.4f}")

    # -----------------------------
    # Compare
    # -----------------------------
    if new_f1 > old_f1:

        print("\nBetter Model Found")

        shutil.copy(NEW_MODEL, PRODUCTION_MODEL)

        shutil.copy(METRICS_CSV, PRODUCTION_METRICS)

        print("Production Model Updated")

        return True

    else:

        print("\nCurrent Production Model is Better")

        print("No Changes Made")

        return False


if __name__ == "__main__":

    compare_models()
