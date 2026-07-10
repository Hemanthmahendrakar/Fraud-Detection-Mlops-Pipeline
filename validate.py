"""
Dataset Validation
"""

import pandas as pd

from config import DATASET_PATH
from config import TARGET_COLUMN

from utils import print_header


def validate_dataset():

    print_header("DATASET VALIDATION")

    # Load dataset
    df = pd.read_csv(DATASET_PATH)

    print(f"Dataset Shape : {df.shape}")

    # -----------------------------
    # Empty Dataset
    # -----------------------------
    if df.empty:
        raise ValueError("Dataset is empty.")

    # -----------------------------
    # Missing Values
    # -----------------------------
    missing = df.isnull().sum().sum()

    print(f"Missing Values : {missing}")

    # -----------------------------
    # Duplicate Rows
    # -----------------------------
    duplicates = df.duplicated().sum()

    print(f"Duplicate Rows : {duplicates}")

    # -----------------------------
    # Target Column
    # -----------------------------
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"{TARGET_COLUMN} column not found."
        )

    print("Target Column Found")

    print("Dataset Validation Successful")

    return df


if __name__ == "__main__":
    validate_dataset()