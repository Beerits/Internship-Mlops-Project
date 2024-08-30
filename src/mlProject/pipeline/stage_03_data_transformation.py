from mlProject.config.configuration import ConfigurationManager
from mlProject.components.data_transformation import DataTransformation
from mlProject import logger
from pathlib import Path

# Define a constant for the stage name
STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            # Read the data validation status from a file
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            # Check if the data schema is valid
            if status == "True":
                # Load configuration for data transformation
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation_config()

                # Initialize the DataTransformation class with the configuration
                data_transformation = DataTransformation(config=data_transformation_config)

                # Perform the train-test split operation
                data_transformation.train_test_spliting()

            else:
                
                raise Exception("Your data schema is not valid")

        except Exception as e:
            
            print(e)

# Entry point of the script
if __name__ == '__main__':
    try:
       
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

        # Create an instance of the DataTransformationTrainingPipeline class and run the main method
        obj = DataTransformationTrainingPipeline()
        obj.main()

        
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        # Log any exceptions that occur and re-raise them
        logger.exception(e)
        raise e
