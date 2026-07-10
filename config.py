"""
Project Configuration File
Stores all paths and constants in one place.
"""

import os

# -------------------------------
# Base Directory
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------
# Dataset
# -------------------------------
DATASET_PATH = os.getenv("DATASET_PATH","dataset/training_data.csv")

# -------------------------------
# Models
# -------------------------------
MODELS_DIR = os.path.join(BASE_DIR, "models")

PRODUCTION_MODEL = os.path.join(BASE_DIR, "model.joblib")

NEW_MODEL = os.path.join(MODELS_DIR, "random_forest_model.joblib")

# -------------------------------
# Reports
# -------------------------------
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

METRICS_FILE = os.path.join(REPORTS_DIR, "metrics.csv")

CLASSIFICATION_REPORT = os.path.join(
    REPORTS_DIR,
    "classification_report.txt"
)

CONFUSION_MATRIX = os.path.join(
    REPORTS_DIR,
    "confusion_matrix.png"
)

# -------------------------------
# Target Column
# -------------------------------
TARGET_COLUMN = "isFraud"

# -------------------------------
# Train-Test Split
# -------------------------------
TEST_SIZE = 0.20

# -------------------------------
# Random Forest
# -------------------------------
RANDOM_STATE = 42

N_ESTIMATORS = 100


# -------------------------------
# Label Encoder
# -------------------------------
LABEL_ENCODER = os.path.join(
    MODELS_DIR,
    "type_label_encoder.pkl"
)


# -------------------------------
# Random Forest Hyperparameters
# -------------------------------
MAX_DEPTH = None
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1


# -------------------------------
# Evaluation Reports
# -------------------------------

CONFUSION_MATRIX_IMAGE = os.path.join(
    REPORTS_DIR,
    "confusion_matrix.png"
)

CLASSIFICATION_REPORT_FILE = os.path.join(
    REPORTS_DIR,
    "classification_report.txt"
)

METRICS_CSV = os.path.join(
    REPORTS_DIR,
    "metrics.csv"
)



# -------------------------------
# Production Metrics
# -------------------------------

PRODUCTION_METRICS = os.path.join(
    REPORTS_DIR,
    "production_metrics.csv"
)
