"""
Utility Functions
"""

import os
import joblib


def create_directory(path):
    """
    Create directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def save_model(model, path):
    """
    Save trained model.
    """
    joblib.dump(model, path)


def load_model(path):
    """
    Load saved model.
    """
    return joblib.load(path)


def print_header(title):
    """
    Print formatted title.
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)