import pandas as pd
import os
import joblib
from mlProject.entity.config_entity import ModelTrainerConfig
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import NearMiss
from sklearn.ensemble import RandomForestClassifier

# Class to handle the training of the model
class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        # Initialize with configuration
        self.config = config

    # Method to handle model training
    def train(self):
        # Load training and testing data from CSV files
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        # Drop rows with missing target column values
        train_data = train_data.dropna(subset=[self.config.target_column])
        test_data = test_data.dropna(subset=[self.config.target_column])

        # Separate features (X) and target (y) for both training and testing sets
        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[self.config.target_column]
        test_y = test_data[self.config.target_column]

        # Scale the features using MinMaxScaler
        scaler = MinMaxScaler()
        train_x_scaled = pd.DataFrame(scaler.fit_transform(train_x), columns=train_x.columns)  # Scale train set
        test_x_scaled = pd.DataFrame(scaler.transform(test_x), columns=test_x.columns)  # Scale test set

        # Balance the training data using NearMiss (undersampling technique)
        nearmiss = NearMiss()
        train_x_balanced, train_y_balanced = nearmiss.fit_resample(train_x_scaled, train_y)

        # Initialize and train RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=self.config.n_estimators, 
                                    class_weight=self.config.class_weight, 
                                    random_state=self.config.random_state)
        rf.fit(train_x_balanced, train_y_balanced)

        # Save the trained model to the specified directory
        joblib.dump(rf, os.path.join(self.config.root_dir, self.config.model_name))

        # Print the shape of training and testing data
        print(f"Training and testing shapes: {train_x.shape}, {test_x.shape}")
