import sys
from Visa_Prediction.exception import visaException
from Visa_Prediction.logger import logging

from Visa_Prediction.entity.config_entity import DataIngestionConfig, DataValidaitonConfig
from Visa_Prediction.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact

from Visa_Prediction.components.data_ingestion import DataIngestion
from Visa_Prediction.components.data_validation import DataValidation

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidaitonConfig()

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
        
    def run_pipeline(self):
        """
        Runs the entire training pipeline
        """
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact = data_ingestion_artifact)
            
        except Exception as e:
            raise visaException(e, sys)