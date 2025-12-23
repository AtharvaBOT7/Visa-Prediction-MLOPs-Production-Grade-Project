import sys
import os

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer

from Visa_Prediction.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from Visa_Prediction.entity.config_entity import DataTransformationConfig
from Visa_Prediction.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact, DataValidationArtifact
from Visa_Prediction.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, write_yaml_file, drop_columns
from Visa_Prediction.entity.estimator import TargetValueMapping

from Visa_Prediction.exception import visaException
from Visa_Prediction.logger import logging

class DataTransformation:
    def __init__(self, 
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        """
        This class will initialise the Data Ingestion artifact, Data Validation artifact and Data Transformation artifact.
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config 
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)

        except Exception as e:
            raise visaException(e, sys) from e
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise visaException(e, sys) from e
        
    def get_data_transformer_object(self) -> Pipeline:
        """
        This function will create and return a data transformer object for the data.
        """
        try:
            logging.info("Got numerical columns from the schema config")

            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()
            ordinal_encoder = OrdinalEncoder()

            oh_columns = self._schema_config['oh_columns']
            or_columns = self._schema_config['or_columns']
            transform_columns = self._schema_config['transform_columns']
            num_features = self._schema_config['num_features']

            transform_pipe = Pipeline(steps = [
                ('transformer', PowerTransformer(method = 'yeo-johnson'))
            ])

            preprocesssor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, oh_columns),
                    ("Ordinal_Encoder", ordinal_encoder, or_columns),
                    ("Transformer", transform_pipe, transform_columns),
                    ("StandardScaler", numeric_transformer, num_features)
                ]
            )

            logging.info("Created a preprocesssor object from Column Transformer")

            return preprocesssor
        
        except Exception as e:
            raise visaException(e, sys) from e
        
    def initiate_data_transformation(self, ) -> DataTransformationArtifact:
        """
        This function will initiate the data transformation process. 
        """
        try:
            if self.data_validation_artifact.validation_status:
                logging.info("Staarting the Data Transformation component")
                preprocessor = self.get_data_transformer_object()

                train_df = DataTransformation.read_data(file_path = self.data_ingestion_artifact.train_file_path)
                test_df = DataTransformation.read_data(file_path = self.data_ingestion_artifact.test_file_path)

                input_feature_train_df = train_df.drop(columns = [TARGET_COLUMN], axis = 1)
                target_feature_train_df = train_df[TARGET_COLUMN]

                logging.info("Got the preprocessed object, input features and target feature from the training dataframe")

                input_feature_train_df['company_age'] = CURRENT_YEAR - input_feature_train_df['yr_of_estab']
                
                drop_cols = self._schema_config['drop_columns']

                input_feature_train_df = drop_columns(df = input_feature_train_df, cols = drop_cols)

                target_feature_train_df = target_feature_train_df.replace(TargetValueMapping()._asdict())

                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_test_df = test_df[TARGET_COLUMN]

                input_feature_test_df['company_age'] = CURRENT_YEAR - input_feature_test_df['yr_of_estab']

                input_feature_test_df = drop_columns(df = input_feature_test_df, cols = drop_cols)

                target_feature_test_df = target_feature_test_df.replace(TargetValueMapping()._asdict())

                logging.info("Got train and test input features and target features")

                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)

                input_feature_test_arr = preprocessor.transform(input_feature_test_df)
                
                smt = SMOTEENN(sampling_strategy = "minority")
                input_feature_train_final, target_feature_train_final = smt.fit_resample(input_feature_train_arr, target_feature_train_df)
                input_feature_test_final, target_feature_test_final = smt.fit_resample(input_feature_test_arr, target_feature_test_df)

                logging.info("Applied SMOTEENN to the test dataset")

                train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
                test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]

                save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
                save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array = train_arr)
                save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array = test_arr)

                logging.info("Saved the transformed object, transformed train and test arrays")

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path = self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )
                return data_transformation_artifact
            else:
                raise Exception(self.data_validation_artifact.message)
        
        except Exception as e:
            raise visaException(e, sys) from e

