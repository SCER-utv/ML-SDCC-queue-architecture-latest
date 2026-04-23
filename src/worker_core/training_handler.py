import os
import time
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from src.model.model_factory import ModelFactory


# handles the execution of partial training tasks on worker nodes
class TrainingHandler:

    # initializes the handler with the aws manager instance
    def __init__(self, aws_manager, config):
        self.aws = aws_manager

    # processes the assigned training chunk and uploads the resulting partial model to s3
    def process(self, task_data):
        job_id, task_id = task_data['job_id'], task_data['task_id']
        dataset_uri = task_data['dataset_s3_path']

        print(f" [TRAIN] Starting {task_id}. Downloading {task_data['num_rows']} rows...")
        print(f" Train set source: {dataset_uri}")

        # 1. perform zero-waste ram partial reading of the dataset from s3
        skip = task_data.get('skip_rows', 0)
        df = pd.read_csv(dataset_uri, skiprows=range(1, skip + 1) if skip > 0 else None, nrows=task_data['num_rows'])

        start_time = time.time()

        # 2. execute training logic based on dataset type
        if task_data.get("is_custom"):
            target_col = task_data.get("custom_target_col")
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found!")

            # separate features and target for custom datasets
            X = df.drop(columns=[target_col]).fillna(0)
            y = df[target_col]

            if task_data.get("task_type") == "classification":
                # initialize and fit random forest classifier
                rf = RandomForestClassifier(
                    n_estimators=task_data['trees'],
                    max_depth=task_data.get('max_depth'),
                    min_samples_split=task_data.get('min_samples_split', 2),
                    min_samples_leaf=task_data.get('min_samples_leaf', 1),
                    max_features=task_data.get('max_features', 'sqrt'),
                    max_samples=task_data.get('max_samples', 1.0),
                    criterion=task_data.get('criterion', 'gini'),
                    class_weight=task_data.get('class_weight', None)
                )
            else:
                # initialize and fit random forest regressor
                rf = RandomForestRegressor(
                    n_estimators=task_data['trees'],
                    max_depth=task_data.get('max_depth'),
                    min_samples_split=task_data.get('min_samples_split', 2),
                    min_samples_leaf=task_data.get('min_samples_leaf', 1),
                    max_features=task_data.get('max_features', 1.0),
                    max_samples=task_data.get('max_samples', 1.0),
                    criterion=task_data.get('criterion', 'squared_error'),
                    n_jobs=-1
                )
            rf.fit(X, y)
        else:
            # fallback to pre-configured handlers for golden standard datasets
            ml_handler = ModelFactory.get_model(dataset_name=task_data['dataset'])
            rf = ml_handler.process_and_train(df, task_data)

        print(f" [Job: {job_id} | Task: {task_id}] Training completed in {time.time() - start_time:.2f}s")

        # 3. serialize the model locally, upload to s3, and clean up
        local_path = f"/tmp/{task_id}_{job_id}.joblib"
        joblib.dump(rf, local_path)

        bucket, _ = self.aws.parse_s3_uri(dataset_uri)
        s3_key = f"models/{task_data['dataset']}/{job_id}/task_{task_id}.joblib"

        print(" Uploading model to S3...")
        self.aws.s3_client.upload_file(local_path, bucket, s3_key)

        os.remove(local_path)
        return f"s3://{bucket}/{s3_key}"