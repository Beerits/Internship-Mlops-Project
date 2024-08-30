# Import necessary modules and classes
from mlProject.config.configuration import ConfigurationManager  
from mlProject.components.data_ingestion import DataIngestion  
from mlProject import logger  

# Define the stage name for logging purposes
STAGE_NAME = "Data Ingestion stage"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass  

    def main(self):
        # Initialize the ConfigurationManager to load configuration
        config = ConfigurationManager()
        
        # Get the data ingestion configuration details
        data_ingestion_config = config.get_data_ingestion_config()
        
        # Initialize the DataIngestion component with the loaded configuration
        data_ingestion = DataIngestion(config=data_ingestion_config)
        
        # Start downloading the data file
        data_ingestion.download_file()
        
        # Extract the contents of the downloaded ZIP file
        data_ingestion.extract_zip_file()

# Entry point of the script
if __name__ == '__main__':
    try:
        
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        
        # Create an instance of the pipeline and start the main process
        obj = DataIngestionTrainingPipeline()
        obj.main()
        
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        # Log any exception that occurs during the process
        logger.exception(e)
        raise e
