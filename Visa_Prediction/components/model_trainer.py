import os
import sys
from typing import Tuple

import numpy as np 
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from neuro_mf import ModelFactory

from Visa_Prediction.exception import VisaException
from Visa_Prediction.logger import logging
from Visa_Prediction.utils.main_utils import *
from Visa_Prediction.entity.config_entity import ModelTrainerConfig
from Visa_Prediction.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from Visa_Prediction.entity.estimator import VisaModel

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]: 
            """
            This function uses neuro_mf package to get the best model object and report of that model
            """
            try:
                logging.info("Getting the best model object and report")

                model_factory = ModelFactory(model_config_path = self.model_trainer_config.model_config_file_path)

                x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]

                best_model_detail = model_factory.get_best_model(
                    X = x_train,
                    y = y_train,
                    base_accuracy = self.model_trainer_config.expected_accuracy
                )

                model_obj = best_model_detail.best_model

                y_pred = model_obj.predict(x_test)

                accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
                f1score = f1_score(y_true = y_test, y_pred = y_pred) 
                precision = precision_score(y_true = y_test, y_pred = y_pred)
                recall = recall_score(y_true = y_test, y_pred = y_pred)

                metric_artifact = ClassificationMetricArtifact(
                    f1_score = f1score,
                    precision_score = precision,
                    recall_score = recall
                )

                return best_model_detail, metric_artifact
                                   
            except Exception as e:
                raise VisaException(e, sys) from e
    
    def initiate_model_trainer(self, ) -> ModelTrainerArtifact:
        """
        This function initiates the model training
        """
        try:
            train_arr = load_numpy_array_data(file_path = self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path = self.data_transformation_artifact.transformed_test_file_path)

            best_model_detail, metric_artifact = self.get_model_object_and_report(train = train_arr, test = test_arr)

            preprocessing_obj = load_object(file_path = self.data_transformation_artifact.transformed_object_file_path)

            if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
                logging.info("Best model with accuracy above expected accuracy is not found")
                raise Exception("The best model is not good as per the expected accuracy")              
            
            visa_model = VisaModel(preprocessing_object = preprocessing_obj, trained_model_object = best_model_detail.best_model)

            logging.info("Created the Visa Model object")

            save_object(self.model_trainer_config.trained_model_file_path, visa_model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path = self.model_trainer_config.trained_model_file_path,
                metric_artifact = metric_artifact
            )

            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")

            return model_trainer_artifact
        
        except Exception as e:
            raise VisaException(e, sys) from e