from mlProject.config.configuration import ConfigurationManager
from mlProject.components.model_evaluation import ModelEvaluation
from mlProject import logger


STAGE_NAME = "Model evaluation stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        

        # Initialize configuration manager to get configurations
        config = ConfigurationManager()

        # Retrieve model evaluation configuration
        model_evaluation_config = config.get_model_evaluation_config()

        # Initialize ModelEvaluation with the retrieved configuration
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)

        # Log evaluation metrics or results into MLflow
        model_evaluation_config.log_into_mlflow()

# Entry point for the script
if __name__ == '__main__':
    try:
        
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

        # Create an instance of ModelEvaluationTrainingPipeline and run the main method
        obj = ModelEvaluationTrainingPipeline()
        obj.main()

        
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        # Log any exceptions that occur during execution
        logger.exception(e)
        
        raise e
