import sys
import os

from pandas import DataFrame
from sklearn.pipeline import Pipeline

from Visa_Prediction.exception import visaException
from Visa_Prediction.logger import logging

class TargetValueMapping:
    def __init__(self):
        self.Certified: int = 0
        self.Denied: int = 1

    def _asdict(self):
        return self.__dict__
    
    def reverse_mapping(self):
        mapping_response = self.asdict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))
    

class VisaModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: DataFrame) -> DataFrame:
        """
        This function will accept the raw inpits and then transform that using preprocessing object, this will make sure
        that same preprocessing steps are followed which were used during training.
        After this it will do the prediction using trained model object.
        """
        try:
            logging.info("Using trained model to get predictions")
            transformed_feature = self.preprocessing_object.transform(dataframe)

            logging.info("Used preprocessing object to get the predictions")
            return self.trained_model_object.predict(transformed_feature)

        except Exception as e:
            raise visaException(e, sys) from e
        
    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"
    
    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"

