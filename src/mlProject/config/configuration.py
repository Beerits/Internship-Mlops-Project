# Importing constants and utility functions from project modules
from mlProject.constants import *
from mlProject.utils.common import read_yaml, create_directories
from mlProject.entity.config_entity import (DataIngestionConfig,
                                            DataValidationConfig,
                                            DataTransformationConfig,
                                            ModelTrainerConfig,
                                            ModelEvaluationConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,  # Default path to the config file
        params_filepath = PARAMS_FILE_PATH,  # Default path to the parameters file
        schema_filepath = SCHEMA_FILE_PATH):  # Default path to the schema file

        # Reading configuration, parameters, and schema from YAML files
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        # Creating directories for storing artifacts as per the config
        create_directories([self.config.artifacts_root])

    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        # Extracting data ingestion configuration from the config file
        config = self.config.data_ingestion

        # Creating directory for data ingestion
        create_directories([config.root_dir])

        # Creating and returning DataIngestionConfig object with required parameters
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    

    def get_data_validation_config(self) -> DataValidationConfig:
        # Extracting data validation configuration and schema information
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        # Creating directory for data validation artifacts
        create_directories([config.root_dir])

        # Creating and returning DataValidationConfig object with required parameters
        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir = config.unzip_data_dir,
            all_schema=schema,
        )

        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        # Extracting data transformation configuration from the config file
        config = self.config.data_transformation

        # Creating directory for data transformation artifacts
        create_directories([config.root_dir])

        # Creating and returning DataTransformationConfig object with required parameters
        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
        )

        return data_transformation_config
    
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        # Extracting model training configuration, parameters, and schema information
        config = self.config.model_trainer
        params = self.params.RandomForestClassifier
        schema =  self.schema.TARGET_COLUMN

        # Creating directory for model training artifacts
        create_directories([config.root_dir])

        # Creating and returning ModelTrainerConfig object with required parameters
        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path = config.train_data_path,
            test_data_path = config.test_data_path,
            model_name = config.model_name,
            n_estimators= params.n_estimators,
            class_weight= params.class_weight,
            random_state= params.random_state,
            target_column = schema.name
            
        )

        return model_trainer_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        # Extracting model evaluation configuration, parameters, and schema information
        config = self.config.model_evaluation
        params = self.params.RandomForestClassifier
        schema =  self.schema.TARGET_COLUMN

        # Creating directory for model evaluation artifacts
        create_directories([config.root_dir])

        # Creating and returning ModelEvaluationConfig object with required parameters
        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path = config.model_path,
            all_params=params,
            metric_file_name = config.metric_file_name,
            target_column = schema.name,
            mlflow_uri="https://dagshub.com/Beerits/Internship-Mlops-Project.mlflow",  # URI for tracking experiments on MLFlow
           
        )

        return model_evaluation_config
