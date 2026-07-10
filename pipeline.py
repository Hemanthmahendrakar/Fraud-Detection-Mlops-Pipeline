"""
Main ML Pipeline
"""
import mlflow
from validate import validate_dataset
from preprocess import preprocess_data
from train import train_model
from evaluate import evaluate_model
from compare import compare_models
from utils import print_header
from mlflow_manager import setup_mlflow

def run_pipeline():

    setup_mlflow()

    with mlflow.start_run():

        print_header("FRAUD DETECTION MLOPS PIPELINE")

        # Step 1
        df = validate_dataset()

        # Step 2
        X_train, X_test, y_train, y_test = preprocess_data(df)

        # Step 3
        model = train_model(
            X_train,
            y_train
        )

        # Step 4
        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )
        compare_models()
        print("\nPipeline Completed Successfully")


    return metrics


if __name__ == "__main__":
    run_pipeline()