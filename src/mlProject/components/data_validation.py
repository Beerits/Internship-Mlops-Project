import os
from mlProject import logger
from mlProject.entity.config_entity import DataValidationConfig
import pandas as pd


class DataValiadtion:
    def __init__(self, config: DataValidationConfig):
        # Initialize the DataValidation class with a configuration object
        self.config = config


    def validate_all_columns(self) -> bool:
        try:
            # Initialize a variable to store the validation status
            validation_status = None

            # Read the CSV file from the directory specified in the configuration
            data = pd.read_csv(self.config.unzip_data_dir)
            # Get a list of all columns in the CSV file
            all_cols = list(data.columns)

            # Get the list of valid schema columns from the configuration
            all_schema = self.config.all_schema.keys()

            # Iterate through all columns in the CSV file
            for col in all_cols:
                # Check if the column exists in the schema
                if col not in all_schema:
                    # If column is not in schema, set validation status to False
                    validation_status = False
                    # Write the validation status to the status file
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                else:
                    # If column is in schema, set validation status to True
                    validation_status = True
                    # Write the validation status to the status file
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")

            # Return the final validation status
            return validation_status
        
        except Exception as e:
            # Raise any exception that occurs during validation
            raise e
