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

        # -----------------------------
        # Dataset Validation
        # -----------------------------
        print("\nValidating Dataset...")
        df = validate_dataset()

        # -----------------------------
        # Data Preprocessing
        # -----------------------------
        print("\nPreprocessing Dataset...")
        X_train, X_test, y_train, y_test = preprocess_data(df)

        # -----------------------------
        # Model Training
        # -----------------------------
        print("\nTraining Model...")
        model = train_model(
            X_train,
            y_train,
        )

        # -----------------------------
        # Model Evaluation
        # -----------------------------
        print("\nEvaluating Model...")
        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        # -----------------------------
        # Log Metrics
        # -----------------------------
        print("\nLogging Metrics...")
        log_metrics(metrics)

        # -----------------------------
        # Validate Metrics
        # -----------------------------
        print("\nValidating Metrics...")

        if not validate_metrics(metrics):

            print("Model validation failed.")
            print("Stopping pipeline.")

            return

        # -----------------------------
        # Compare Models
        # -----------------------------
        print("\nComparing with Production Model...")

        is_better = compare_models(
            model,
            X_test,
            y_test,
        )

        # -----------------------------
        # Promote Model
        # -----------------------------
        if is_better:

            print("\nNew model is better.")
            print("Promoting to Production...")

            promote_model()

        else:

            print("\nCurrent Production model is better.")
            print("Skipping Promotion.")

        print_header("PIPELINE COMPLETED SUCCESSFULLY")

        return metrics

    except Exception as e:

        print(f"\nPipeline Failed : {e}")
        raise

    finally:

        end_run()


if __name__ == "__main__":
    run_pipeline()
