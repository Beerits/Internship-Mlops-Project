import os
import pandas as pd
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from mlProject.entity.config_entity import ModelEvaluationConfig
from mlProject.utils.common import save_json
from pathlib import Path

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score



class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        # Initialize ModelEvaluation with configuration parameters
        self.config = config

    def eval_metrics(self, actual, pred):
        # Calculate evaluation metrics (accuracy, precision, recall, F1 score)
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred, average='weighted')
        recall = recall_score(actual, pred, average='weighted')
        f1 = f1_score(actual, pred, average='weighted')

         # Round the metrics to two decimal places
        accuracy = round(accuracy, 2)
        precision = round(precision, 2)
        recall = round(recall, 2)
        f1 = round(f1, 2)
        
        # Return the computed metrics
        return accuracy, precision, recall, f1

    def log_into_mlflow(self):
        # Load test data from CSV file
        test_data = pd.read_csv(self.config.test_data_path)

        # Drop rows with missing target column values
        test_data = test_data.dropna(subset=[self.config.target_column])

        # Load the pre-trained model from a file
        model = joblib.load(self.config.model_path)

        # Split test data into input features (test_x) and target column (test_y)
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        # Set up environment variables for MLflow tracking server
        os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/Beerits/Internship-Mlops-Project.mlflow"
        os.environ["MLFLOW_TRACKING_USERNAME"] = "Beerits"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "3ff23dcadf9d40548cbc76b5ee56969248d509f2"

        # Print the configured MLflow URI for tracking
        print(f"Configured MLflow URI: {self.config.mlflow_uri}")

        # Set the MLflow tracking URI and registry URI
        mlflow.set_registry_uri(self.config.mlflow_uri)

        # Get the tracking URL type (e.g., file or server) to determine the logging method
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Start an MLflow run to track metrics and log the model
        with mlflow.start_run():
            print(f"Tracking URL scheme: 1 {tracking_url_type_store}")

            # Predict the target values using the model
            predicted_qualities = model.predict(test_x)

            # Calculate evaluation metrics for the predictions
            accuracy, precision, recall, f1 = self.eval_metrics(test_y, predicted_qualities)

            # Log the evaluation metrics into MLflow
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            # Save the metrics locally as a JSON file
            scores = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }
            save_json(path=Path(self.config.metric_file_name), data=scores)

            # Log hyperparameters and other relevant details into MLflow
            mlflow.log_params(self.config.all_params)

            print(f"Tracking URL scheme:2 {tracking_url_type_store}")

            # If the tracking URL is not a local file store, register the model in the MLflow model registry
            if tracking_url_type_store != "file":
                print("tracking_url_type_store working good")
                mlflow.sklearn.log_model(model, "model", registered_model_name="RandomForestClassifier")
            else:
                # Log the model without registering if using a local file store
                mlflow.sklearn.log_model(model, "model")
