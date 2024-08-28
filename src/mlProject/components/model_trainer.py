# import pandas as pd
# import os
# from mlProject import logger
# import joblib
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from mlProject.entity.config_entity import ModelTrainerConfig
# from sklearn.preprocessing import MinMaxScaler
# from imblearn.under_sampling import NearMiss
# from sklearn.ensemble import RandomForestClassifier




# class ModelTrainer:
#     def __init__(self, config: ModelTrainerConfig):
#         self.config = config

    
#     def train(self):
#         train_data = pd.read_csv(self.config.train_data_path)
#         test_data = pd.read_csv(self.config.test_data_path)

#         train_data = train_data.dropna(subset=[self.config.target_column])
#         test_data = test_data.dropna(subset=[self.config.target_column])


#         # Separate features and target
#         train_x = train_data.drop([self.config.target_column], axis=1)
#         test_x = test_data.drop([self.config.target_column], axis=1)
#         train_y = train_data[self.config.target_column]
#         test_y = test_data[self.config.target_column]

#         scaler = MinMaxScaler()
#         train_x_scaled = scaler.fit_transform(train_x)
#         test_x_scaled = scaler.transform(test_x)


#         nearmiss = NearMiss()
#         train_x_balanced, train_y_balanced = nearmiss.fit_resample(train_x_scaled, train_y)
        


#         rf = RandomForestClassifier(n_estimators=self.config.n_estimators, class_weight=self.config.class_weight, random_state=self.config.random_state)
#         rf.fit(train_x_balanced, train_y_balanced)
#         predictions = rf.predict(test_x_scaled)


#         # Calculate metrics
#         accuracy = accuracy_score(test_y, predictions)
#         precision = precision_score(test_y, predictions)
#         recall = recall_score(test_y, predictions)
#         f1 = f1_score(test_y, predictions)

#         # Print metrics
#         print(f"Accuracy: {accuracy:.4f}")
#         print(f"Precision: {precision:.4f}")
#         print(f"Recall: {recall:.4f}")
#         print(f"F1 Score: {f1:.4f}")

#         # Save the model
#         joblib.dump(rf, os.path.join(self.config.root_dir, self.config.model_name))

#         # Logging
#         logger.info("Model training completed and model saved.")
#         logger.info(f"Accuracy: {accuracy:.4f}")
#         logger.info(f"Precision: {precision:.4f}")
#         logger.info(f"Recall: {recall:.4f}")
#         logger.info(f"F1 Score: {f1:.4f}")

#         print(f"Training and testing shapes: {train_x.shape}, {test_x.shape}")

        











import pandas as pd
import os
from mlProject import logger
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlProject.entity.config_entity import ModelTrainerConfig
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import NearMiss
from sklearn.ensemble import RandomForestClassifier


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_data = train_data.dropna(subset=[self.config.target_column])
        test_data = test_data.dropna(subset=[self.config.target_column])

        # Separate features and target
        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[self.config.target_column]
        test_y = test_data[self.config.target_column]

        scaler = MinMaxScaler()
        train_x_scaled = pd.DataFrame(scaler.fit_transform(train_x), columns=train_x.columns)  # Keep feature names
        test_x_scaled = pd.DataFrame(scaler.transform(test_x), columns=test_x.columns)  # Keep feature names

        nearmiss = NearMiss()
        train_x_balanced, train_y_balanced = nearmiss.fit_resample(train_x_scaled, train_y)

        rf = RandomForestClassifier(n_estimators=self.config.n_estimators, class_weight=self.config.class_weight, random_state=self.config.random_state)
        rf.fit(train_x_balanced, train_y_balanced)
        predictions = rf.predict(test_x_scaled)

        # Calculate metrics
        accuracy = accuracy_score(test_y, predictions)
        precision = precision_score(test_y, predictions)
        recall = recall_score(test_y, predictions)
        f1 = f1_score(test_y, predictions)

        # Print metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Save the model
        joblib.dump(rf, os.path.join(self.config.root_dir, self.config.model_name))

        # Logging
        logger.info("Model training completed and model saved.")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")

        print(f"Training and testing shapes: {train_x.shape}, {test_x.shape}")
