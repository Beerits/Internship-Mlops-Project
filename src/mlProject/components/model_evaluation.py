import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss



class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def eval_metrics(self,actual, pred):
        # rmse = np.sqrt(mean_squared_error(actual, pred))
        # mae = mean_absolute_error(actual, pred)
        # r2 = r2_score(actual, pred)
        # return rmse, mae, r2
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred, average='weighted')
        recall = recall_score(actual, pred, average='weighted')
        f1 = f1_score(actual, pred, average='weighted')
        # auc_roc = roc_auc_score(actual, pred, average='weighted', multi_class='ovr')
        return accuracy, precision, recall, f1
    


    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path)

        test_data = test_data.dropna(subset=[self.config.target_column])
        
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        print(f"Configured MLflow URI: {self.config.mlflow_uri}")
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        print("tracking_url_type_store created")

        with mlflow.start_run():
            print(f"Tracking URL scheme: 1 {tracking_url_type_store}")


            predicted_qualities = model.predict(test_x)

            # (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)
            accuracy, precision, recall, f1 = self.eval_metrics(test_y, predicted_qualities)


            # Saving metrics as local
            # scores = {"rmse": rmse, "mae": mae, "r2": r2}
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            # mlflow.log_metric("auc_roc", auc_roc)

            # conf_matrix = confusion_matrix(test_y, predicted_qualities)
            # logloss = log_loss(test_y, model.predict_proba(test_x))

            # mlflow.log_metric("log_loss", logloss)

            scores = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                # "auc_roc": auc_roc,
                # "log_loss": logloss
            }

            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            # mlflow.log_metric("rmse", rmse)
            # mlflow.log_metric("r2", r2)
            # mlflow.log_metric("mae", mae)

            

            print(f"Tracking URL scheme:2 {tracking_url_type_store}")

            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                print("tracking_url_type_store working good")
                mlflow.sklearn.log_model(model, "model", registered_model_name="RandomForestClassifier")
            else:
                mlflow.sklearn.log_model(model, "model")

    












# import os
# import pandas as pd
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from urllib.parse import urlparse
# import mlflow
# import mlflow.sklearn
# import numpy as np
# import joblib
# from mlProject.entity.config_entity import ModelEvaluationConfig
# from mlProject.utils.common import save_json
# from pathlib import Path


# class ModelEvaluation:
#     def __init__(self, config: ModelEvaluationConfig):
#         self.config = config

    
#     def eval_metrics(self, actual, pred):
#         rmse = np.sqrt(mean_squared_error(actual, pred))
#         mae = mean_absolute_error(actual, pred)
#         r2 = r2_score(actual, pred)
#         return rmse, mae, r2

    
#     def log_into_mlflow(self):
#         test_data = pd.read_csv(self.config.test_data_path)
#         test_data = test_data.dropna(subset=[self.config.target_column])
        
#         model = joblib.load(self.config.model_path)

#         test_x = test_data.drop([self.config.target_column], axis=1)
#         test_y = test_data[[self.config.target_column]]

        

        
        
        
        

#         mlflow.set_registry_uri(self.config.mlflow_uri)
#         tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
#         print("tracking_url_type_store created")


#         with mlflow.start_run():
#             # Use a DataFrame for test_x to match the training data structure
#             predicted_qualities = model.predict(test_x)

#             (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)

#             scores = {"rmse": rmse, "mae": mae, "r2": r2}
#             # Saving metrics as local
#             save_json(path=Path(self.config.metric_file_name), data=scores)

#             mlflow.log_params(self.config.all_params)
#             mlflow.log_metric("rmse", rmse)
#             mlflow.log_metric("r2", r2)
#             mlflow.log_metric("mae", mae)

#             # Model registry does not work with file store
#             if tracking_url_type_store != "file":
#                 print("tracking_url_type_store working good")
#                 mlflow.sklearn.log_model(model, "model", registered_model_name="RandomForestClassifier")
#             else:
#                 mlflow.sklearn.log_model(model, "model")
