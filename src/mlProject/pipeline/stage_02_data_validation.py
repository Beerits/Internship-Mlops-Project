# Import necessary modules and classes
from mlProject.config.configuration import ConfigurationManager  
from mlProject.components.data_validation import DataValiadtion  
from mlProject import logger  


STAGE_NAME = "Data Validation stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        
        pass

    def main(self):
       
        config = ConfigurationManager()  # Create an instance of ConfigurationManager
        data_validation_config = config.get_data_validation_config()  # Get data validation configuration
        data_validation = DataValiadtion(config=data_validation_config)  # Create an instance of DataValiadtion with the configuration
        data_validation.validate_all_columns()  # Call the method to validate all columns

if __name__ == '__main__':
    try:
        
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationTrainingPipeline()  # Create an instance of DataValidationTrainingPipeline
        obj.main()  
        
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        # Log any exceptions that occur during the execution
        logger.exception(e)
        raise e  
