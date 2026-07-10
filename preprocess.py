"""
Data Preprocessing
"""
import os

os.makedirs("models", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("mlruns", exist_ok=True)
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow

from config import (
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    LABEL_ENCODER
)

from utils import print_header


def preprocess_data(df):

    print_header("DATA PREPROCESSING")


    # ----------------------------
    # Encode Transaction Type
    # ----------------------------
    encoder = LabelEncoder()

    df["type"] = encoder.fit_transform(df["type"])

    # Save encoder
    joblib.dump(encoder, LABEL_ENCODER)
    # Log Label Encoder
    mlflow.log_artifact(LABEL_ENCODER)

    print("Label Encoder Saved")

    # ----------------------------
    # Drop Identifier Columns
    # ----------------------------
    columns_to_drop = ["nameOrig", "nameDest"]

    df.drop(columns=columns_to_drop, inplace=True)

    print("Dropped Identifier Columns")

    # ----------------------------
    # Features and Target
    # ----------------------------
    X = df.drop(columns=[TARGET_COLUMN])

    y = df[TARGET_COLUMN]

    # ----------------------------
    # Debug (temporary)
    # ----------------------------
    print("\nFeature Columns:")
    print(X.columns.tolist())

    print("\nFeature Data Types:")
    print(X.dtypes)

    # ----------------------------
    # Train-Test Split
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"\nTraining Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    preprocess_data()
