"""
Main ML Pipeline
"""

from validate import validate_dataset
from preprocess import preprocess_data
from train import train_model
from evaluate import evaluate_model
from compare import compare_models, validate_metrics
from utils import print_header

from mlflow_manager import (
    start_run,
    end_run,
    log_metrics,
    register_model,
)


def run_pipeline():

    run = start_run()

    try:
        print_header("FRAUD DETECTION MLOPS PIPELINE")

        print("\nValidating dataset...")
        df = validate_dataset()

        print("Preprocessing data...")
        X_train, X_test, y_train, y_test = preprocess_data(df)

        print("Training model...")
        model = train_model(
            X_train,
            y_train,
        )

        print("Evaluating model...")
        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        print("Logging metrics...")
        log_metrics(metrics)

        print("Validating metrics...")
        if not validate_metrics(metrics):
            print("\nModel failed validation.")
            print("Stopping pipeline.")
            return None

        print("Comparing with production model...")
        if compare_models():
            register_model(model)
            print("\nNew model promoted to Production.")
        else:
            print("\nExisting Production model retained.")

        return metrics

    except Exception as e:
        print(f"\nPipeline failed: {e}")
        raise

    finally:
        end_run()


if __name__ == "__main__":
    run_pipeline()
