import mlflow
import mlflow.sklearn
from config import (
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# def start_run(run_name):
#     return mlflow.start_run(run_name=run_name)

# def end_run():
#     mlflow.end_run()

# def log_model(model):
#     mlflow.sklearn.log_model(
#         sk_model=model,
#         artifact_path="model",
#         registered_model_name=REGISTERED_MODEL_NAME,
#     )
