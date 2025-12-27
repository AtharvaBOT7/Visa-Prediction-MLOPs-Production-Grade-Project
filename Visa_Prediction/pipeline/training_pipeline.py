import sys
from Visa_Prediction.exception import visaException
from Visa_Prediction.logger import logging

from Visa_Prediction.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from Visa_Prediction.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact

from Visa_Prediction.components.data_ingestion import DataIngestion
from Visa_Prediction.components.data_validation import DataValidation
from Visa_Prediction.components.data_transformation import DataTransformation
from Visa_Prediction.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        Responsible for starting the data ingestion step of the training pipeline
        """
        try:
            logging.info("Starting the data ingestion component of training pipeline")
            logging.info("Fetching the data from MongoDB to feature store")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Completed the data ingestion component of training pipeline")
            return data_ingestion_artifact
        except Exception as e:
            raise visaException(e, sys)
        
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """ 
        Responsible for starting the data validation step of the training pipeline
        """
        try:
            logging.info("Starting the data validation component of the training pipeline")
            data_validation = DataValidation(
                data_ingestion_artifact = data_ingestion_artifact,
                data_validation_config = self.data_validation_config
            )

            data_validation_artifact = data_validation.initiate_data_validation()

            logging.info("Data Validaiton step completed")
            return data_validation_artifact
        except Exception as e:
            raise visaException(e, sys) from e
        
    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        """ 
        This function starts the data transformation step of the training pipeline
        """
        try:
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config = self.data_transformation_config,
                data_validation_artifact = data_validation_artifact
            )
        
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise visaException(e, sys) from e
    
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        """
        This function starts the model trainer step of the training pipeline
        """
        try: 
            model_trainer = ModelTrainer(
                data_transformation_artifact = data_transformation_artifact,
                model_trainer_config = self.model_trainer_config
            )

            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact
        
        except Exception as e:
            raise visaException(e, sys) from e
        
    def run_pipeline(self):
        """
        Runs the entire training pipeline
        """
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact = data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact = data_ingestion_artifact, data_validation_artifact = data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact = data_transformation_artifact)
            
        except Exception as e:
            raise visaException(e, sys)