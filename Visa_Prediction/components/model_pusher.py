from Visa_Prediction.cloud_storage.aws_storage import SimpleStorageService
from Visa_Prediction.exception import visaException
from Visa_Prediction.logger import logging
from Visa_Prediction.entity.config_entity import ModelPusherConfig
from Visa_Prediction.entity.artifact_entity import ModelPusherArtifact, ModelTrainerArtifact, ModelEvaluationArtifact
from Visa_Prediction.entity.s3_estimator import visaEstimator
import os, sys

class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact, model_pusher_config: ModelPusherConfig):
        self.s3 = SimpleStorageService()
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        
        self.visa_estimator = visaEstimator(
            bucket_name = model_pusher_config.bucket_name,
            model_path = model_pusher_config.s3_model_key_path
        )

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            logging.info("Uploading the Artifacts to S3 bucket")

            self.visa_estimator.save_model(
                from_file = self.model_evaluation_artifact.trained_model_path
            )

            model_pusher_artifact = ModelPusherArtifact(
                bucket_name = self.model_pusher_config.bucket_name,
                s3_model_path = self.model_pusher_config.s3_model_key_path
            )

            logging.info("Uploaded the Artifacts to S3 bucket")
            
            return model_pusher_artifact

        except Exception as e:
            raise visaException(e, sys) from e
    