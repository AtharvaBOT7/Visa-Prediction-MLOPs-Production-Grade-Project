import sys
from Visa_Prediction.exception import visaException
from Visa_Prediction.logger import logging

from Visa_Prediction.entity.config_entity import DataIngestionConfig
from Visa_Prediction.entity.artifact_entity import DataIngestionArtifact

from Visa_Prediction.components.data_ingestion import DataIngestion

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

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
        
    def run_pipeline(self):
        """
        Runs the entire training pipeline
        """
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            
        except Exception as e:
            raise visaException(e, sys)