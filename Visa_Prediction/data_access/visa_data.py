from Visa_Prediction.configuration.mongo_db_conn import MongoDBConnection
from Visa_Prediction.constants import DB_NAME
from Visa_Prediction.exception import visaException
import pandas as pd
import os
import sys
from typing import Optional 
import numpy as np 

class VisaData:
    """
    This class helps to export the entire dataset from MongoDB as a pandas DataFrame.
    """
    def __init__(self):
        try:
            self.mongo_client = MongoDBConnection(database_name=DB_NAME)
        except Exception as e:
            raise visaException(e, sys)
        
    def export_collection_as_dataframe(self, collection_name: str, database_name: Optional[str]=None) -> pd.DataFrame:
        """
        This function exports the specified collection from MongoDB as a pandas DataFrame.
        """
        try:
            if database_name is None:
                collection_name = self.mongo_client.database[collection_name]
            else:
                collection_name = self.mongo_client[database_name][collection_name]

            df = pd.DataFrame(list(collection_name.find()))
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis = 1) # Dropping the id column from the DataFrame
            df.replace({"na":np.nan}, inplace=True)

            return df
        except Exception as e:
            raise visaException(e, sys)