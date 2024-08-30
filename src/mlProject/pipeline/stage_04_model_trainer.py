from mlProject.config.configuration import ConfigurationManager
from mlProject.components.model_trainer import ModelTrainer
from mlProject import logger


STAGE_NAME = "Model Trainer stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        
        pass

    def main(self):
        

        # Create a ConfigurationManager instance to manage configuration settings
        config = ConfigurationManager()
        
       
        model_trainer_config = config.get_model_trainer_config()
        
        # Create a ModelTrainer instance with the retrieved configuration
        model_trainer = ModelTrainer(config=model_trainer_config)
        
        
        model_trainer.train()

# Entry point of the script
if __name__ == '__main__':
    try:
        
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        
        # Create an instance of ModelTrainerTrainingPipeline and run the main method
        obj = ModelTrainerTrainingPipeline()
        obj.main()
        
        
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        # Log any exceptions that occur during execution
        logger.exception(e)
        
        raise e
