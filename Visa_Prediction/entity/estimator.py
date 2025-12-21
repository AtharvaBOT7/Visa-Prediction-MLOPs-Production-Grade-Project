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