import os 
from datetime import datetime
from dotenv import load_dotenv
from datetime import date

load_dotenv()

MONGO_CONNECTION_URL = os.environ.get("MONGO_CONNECTION_URL")
DB_NAME = os.environ.get("DB_NAME")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME")

PIPELINE_NAME: str = "visapred"
ARTIFACT_DIR: str = "artifact"

FILE_NAME = "visadata.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

MODEL_FILE_NAME = "model.pkl"

TARGET_COLUMN = "case_status"
CURRENT_YEAR = date.today().year
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

AWS_ACCESS_KEY_ID = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = os.environ.get("REGION_NAME")

""" 
These are the Data Ingestion related constant vairables and we save all these in this file because if we 
later want to change the values we can easily change it from here and it will reflect in the entire project 
"""

DATA_INGESTION_COLLECTION_NAME: str = os.environ.get("COLLECTION_NAME")
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

"""
These are the Data Validation related constants.
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

""" 
These are Data Transformation related constants.
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

"""
These are the Model Trainer related constants.
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_FILE_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")

""" 
These are the Model Evaluation related constants.
"""
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME = os.getenv("MODEL_BUCKET_NAME")
MODEL_PUSHER_S3_KEY = "model-registry"

APP_HOST = "0.0.0.0"
APP_PORT = 8081