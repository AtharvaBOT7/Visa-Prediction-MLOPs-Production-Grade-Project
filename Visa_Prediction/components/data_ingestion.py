import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from Visa_Prediction.entity.config_entity import DataIngestionConfig
from Visa_Prediction.entity.artifact_entity  import DataIngestionArtifact
from Visa_Prediction.exception import visaException
from Visa_Prediction.logger import logging
from Visa_Prediction.data_access.visa_data import VisaData

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        """
        Configuration for data ingestion.
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise visaException(e, sys)
        
    def export_data_into_feature_store(self) -> DataFrame:
        """
        Export data from MongoDB to feature store.
        """
        try:
            logging.info("Exporting the data from MongoDB to feature store")
            visa_data = VisaData()
            dataframe = visa_data.export_collection_as_dataframe(collection_name = self.data_ingestion_config.collection_name)
            
            logging.info(f"Shape of the dataframe: {dataframe.shape}")
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path =os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok = True)
            logging.info(f"Saving the exported data into feature store file path: {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path, index =False, header =True)
            return dataframe
        
        except Exception as e:
            raise visaException(e, sys)
        
    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        """
        Splits the dataframe into training and testing sets.
        """
        logging.info("Starting to split the data into train and test sets")

        try:
            train_set, test_set = train_test_split(dataframe, test_size = self.data_ingestion_config.train_test_split_ratio)
            logging.info("Performed the train test split successfully")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok = True)

            logging.info(f"Exporting the train and test file path")
            train_set.to_csv(self.data_ingestion_config.training_file_path, index = False, header = True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index = False, header = True)
            logging.info("Exported the train and test file path successfully")

        except Exception as e:
            raise visaException(e, sys)
        
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Initiates the data ingestion components of training pipeline
        """
        logging.info("Starting the data ingestion component")
        try:
            dataframe = self.export_data_into_feature_store()

            logging.info("Data from MongoDB exported to feature store successfully")

            self.split_data_as_train_test(dataframe=dataframe)
            
            logging.info("Data split into train and test sets successfully")

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path = self.data_ingestion_config.training_file_path,
                test_file_path = self.data_ingestion_config.testing_file_path
            )

            logging.info(f"Data Ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        
        except Exception as e:
            raise visaException(e, sys)