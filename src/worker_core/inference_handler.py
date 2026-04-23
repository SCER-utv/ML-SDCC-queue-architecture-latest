import os
import time
import gc
import joblib
import numpy as np
import pandas as pd
from src.model.model_factory import ModelFactory


# handles model inference execution including real-time single predictions and memory-efficient bulk evaluations
class InferenceHandler:

    # initializes the inference handler with aws manager and configured chunk size
    def __init__(self, aws_manager, config):
        self.aws = aws_manager
        self.chunk_size = config.get("inference_chunksize", 500000)

    # processes the inference task by downloading the model and routing to either real-time or bulk chunked prediction
    def process(self, task_data):
        job_id, task_id = task_data['job_id'], task_data['task_id']      
        bucket, model_key = self.aws.parse_s3_uri(task_data['model_s3_uri'])

        print(f"Test set source: {task_data.get('test_dataset_uri', 'N/A (Real time inference)')}")

        """
        # ==========================================================
        # TEST 2.1 (WORKER HARD CRASH DURING INFERENCE)
        # ==========================================================
        print(f" [TEST PHASE 2] Starting inference for {task_id}. 15-second pause...")
        time.sleep(15)
        """

        """
        # ==========================================================
        # TEST 2.2 (WORKER SOFT CRASH DURING INFERENCE)
        # ==========================================================
        # if task_id == "task_2":
        #     raise MemoryError("SOFT CRASH SIMULATION DURING INFERENCE - OUT OF MEMORY")
        # ==========================================================
        """

        # download the assigned partial model from s3 into temporary storage
        local_model_path = f"/tmp/model_{job_id}_{task_id}.joblib"
        self.aws.s3_client.download_file(bucket, model_key, local_model_path)
        rf = joblib.load(local_model_path)
        os.remove(local_model_path)

        # case a: real-time inference for a single data tuple
        if 'tuple_data' in task_data:
            print(f" [INFER] Single tuple real-time prediction in progress...")

            """
            # ==========================================================
            # TEST 3.1 (WORKER HARD CRASH IN SINGLE INFERENCE)
            # ==========================================================
            print("\n" + "!"*50)
            print(" [TEST 3.1] REAL-TIME INFERENCE STARTED!")
            print(" [TEST 3.1] You have 15 seconds to kill THIS Worker")
            print("!"*50 + "\n")
            time.sleep(15)
            # ==========================================================
            """

            """
            # ==========================================================
            # TEST 3.2 (WORKER SOFT CRASH IN SINGLE INFERENCE)
            # ==========================================================
            # If this is task 1, simulate a Python error
            if task_data['task_id'] == "task_infer_rt_1":
                print(" [TEST 3.2] Simulating Python exception...")
                raise ValueError("SIMULATED SOFT CRASH: Corrupted tuple data!")
            # ==========================================================
            """




            data_array = np.array(task_data['tuple_data']).reshape(1, -1)
            all_pred = [float(tree.predict(data_array)[0]) for tree in rf.estimators_]
            return {"tipo": "singolo", "valore": all_pred}

        # case b: bulk inference processing the full test dataset in memory-efficient chunks
        print(f" [INFER] Bulk inference started (Chunksize: {self.chunk_size})")
        start_time = time.time()
        all_predictions = []

        is_custom = task_data.get("is_custom", False) or task_data.get("dataset") == "custom"
        target_col = task_data.get("custom_target_col", "Label")
        ml_handler = None if is_custom else ModelFactory.get_model(dataset_name=task_data['dataset'])

        # process predictions chunk by chunk and aggressively free ram
        for chunk in pd.read_csv(task_data['test_dataset_uri'], chunksize=self.chunk_size, low_memory=False):
            if is_custom:
                X = chunk.drop(columns=[target_col]).fillna(0) if target_col in chunk.columns else chunk.fillna(0)
                preds = np.array([tree.predict(X.values) for tree in rf.estimators_])
                chunk_results = np.mean(preds, axis=0)
            else:
                chunk_results = ml_handler.process_and_predict(rf, chunk)

            all_predictions.append(chunk_results)
            del chunk, chunk_results
            gc.collect()

        numpy_results = np.concatenate(all_predictions)
        print(f" [INFER] {len(numpy_results)} predictions in {time.time() - start_time:.2f}s")

        # compress, save locally as a numpy array, and upload back to s3
        local_npy_path = f"/tmp/results_{job_id}_{task_id}.npy"
        np.save(local_npy_path, numpy_results)
        s3_key = f"results/{task_data['dataset']}/{task_data.get('dataset_variant', '1M')}/{job_id}/{task_id}.npy"
        self.aws.s3_client.upload_file(local_npy_path, bucket, s3_key)
        os.remove(local_npy_path)

        return {"tipo": "bulk", "valore": f"s3://{bucket}/{s3_key}"}