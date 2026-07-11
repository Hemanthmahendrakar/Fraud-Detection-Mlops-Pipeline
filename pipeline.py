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
    promote_model,
)


def run_pipeline():

    start_run()

    try:

        print_header("FRAUD DETECTION MLOPS PIPELINE")

        print("\nValidating dataset...")
        df = validate_dataset()

        print("\nPreprocessing data...")
        X_train, X_test, y_train, y_test = preprocess_data(df)

        print("\nTraining model...")
        model = train_model(
            X_train,
            y_train,
        )

        print("\nEvaluating model...")
        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        print("\nLogging metrics...")
        log_metrics(metrics)

        print("\nValidating metrics...")

        if not validate_metrics(metrics):

            print("Model failed validation.")
            return

        print("\nComparing with Production Model...")

        is_better = compare_models(
            model,
            X_test,
            y_test,
        )

        if is_better:

            print("\nPromoting New Model to Production...\n")

            promote_model()

        else:

            print("\nCurrent Production Model is Better.")
            print("Skipping Promotion.")

        print("\nPipeline Completed Successfully.")

        return metrics

    except Exception as e:

        print(f"\nPipeline Failed: {e}")
        raise

    finally:

        end_run()


if __name__ == "__main__":
    run_pipeline()
