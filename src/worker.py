import gc
import json
import os
import threading
import time

import boto3
import joblib
import numpy as np
import pandas as pd

from src.model.model_factory import ModelFactory
from src.utils.config import load_config

# ==========================================
# DYNAMIC CONFIGURATION
# ==========================================
config = load_config()

AWS_REGION = config.get("aws_region")
CLIENT_QUEUE_URL = config["sqs_queues"]["client"]
TRAIN_TASK_QUEUE = config["sqs_queues"]["train_task"]
TRAIN_RESPONSE_QUEUE = config["sqs_queues"]["train_response"]
INFER_TASK_QUEUE = config["sqs_queues"]["infer_task"]
INFER_RESPONSE_QUEUE = config["sqs_queues"]["infer_response"]

sqs_client = boto3.client('sqs', region_name=AWS_REGION)
s3_client = boto3.client('s3', region_name=AWS_REGION)

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

# Background thread that periodically extends the SQS message visibility timeout. 
# Prevents other workers from picking up the message while it's still being processed.
def extend_sqs_visibility(queue_url, receipt_handle, stop_event):
    while not stop_event.is_set():
        # Sleep for 2 minutes. If, in the meantime, the training finishes (stop_event is set), the loop stops before making another call to AWS
        stop_event.wait(20) 
        if not stop_event.is_set():
            try:
                # Extends the timeout for another 5 minutes
                sqs_client.change_message_visibility(
                    QueueUrl=queue_url,
                    ReceiptHandle=receipt_handle,
                    VisibilityTimeout=60 
                )
                print(" [HEARTBEAT] SQS message visibility reset to 5 minutes.")
            except Exception as e:
                print(f" [HEARTBEAT] SQS extension error (likely already deleted): {e}")

# Parses an S3 URI into bucket and object key
def parse_s3_uri(s3_uri):
    parts = s3_uri.replace("s3://", "").split("/", 1)
    return parts[0], parts[1]

# ==========================================
# CORE LOGIC: TRAINING
# ==========================================

# Executes the distributed training task on a specific data chunk
def train(train_task_data, receipt_handle):
    job_id = train_task_data['job_id']
    task_id = train_task_data['task_id']
    dataset_uri = train_task_data['dataset_s3_path']
    
    # START HEATBEAT
    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(
        target=extend_sqs_visibility,
        args=(TRAIN_TASK_QUEUE, receipt_handle, stop_event)
    )
    heartbeat_thread.start()

    try:
        print(f" [TRAIN] Starting {task_id}. Fetching {train_task_data['num_rows']} rows from S3...")

        # 1. PARTIAL S3 READ (Zero-Waste RAM)
        skip_rows = train_task_data['skip_rows']
        if skip_rows > 0:
            rows_to_skip = range(1, skip_rows + 1)
        else:
            rows_to_skip = None

        df = pd.read_csv(
            dataset_uri,
            skiprows=rows_to_skip,
            nrows=train_task_data['num_rows']
        )

        ml_handler = ModelFactory.get_model(dataset_name=train_task_data['dataset'])

        print("Starting timer...")
        start_time = time.time()
        print("Timer started")

        # 2. Training
        rf = ml_handler.process_and_train(df, train_task_data)
        print(f" [Job: {job_id} | Task: {task_id}] Training completed in {time.time() - start_time:.2f}s")

        # 3. Save and upload
        local_model_path = f"/tmp/{task_id}_{job_id}.joblib"
        joblib.dump(rf, local_model_path)

        bucket, _ = parse_s3_uri(dataset_uri)
        dataset_name = train_task_data['dataset']
        s3_key = f"models/{dataset_name}/{job_id}/task_{task_id}.joblib"

        print(" Uploading model to S3...")
        s3_client.upload_file(local_model_path, bucket, s3_key)

        os.remove(local_model_path)  
        return f"s3://{bucket}/{s3_key}"

    finally:
        # END HEARTBEAT 
        stop_event.set()
        heartbeat_thread.join()


# ==========================================
# CORE LOGIC: INFERENCE
# ==========================================

# Executes inference using a previously trained local model chunk
def execute_inference(infer_task_data, receipt_handle):
    job_id = infer_task_data['job_id']
    task_id = infer_task_data['task_id']
    model_s3_uri = infer_task_data['model_s3_uri']

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
    
    # START HEATBEAT
    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(
        target=extend_sqs_visibility,
        args=(INFER_TASK_QUEUE, receipt_handle, stop_event)
    )
    heartbeat_thread.start()

    try:
        print(f" [INFER] Starting inference {task_id}. Downloading model...")
        bucket, model_key = parse_s3_uri(model_s3_uri)

        # 1. Download model from S3
        local_model_path = f"/tmp/model_{job_id}_{task_id}.joblib"
        s3_client.download_file(bucket, model_key, local_model_path)
        rf = joblib.load(local_model_path)

        # CASE 1: SINGLE TUPLE INFERENCE (Real-time)
        if 'tuple_data' in infer_task_data:
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
            
            # ==========================================================
            # TEST 3.2 (WORKER SOFT CRASH IN SINGLE INFERENCE)
            # ==========================================================
            # If this is task 1, simulate a Python error
            if infer_task_data['task_id'] == "task_infer_rt_1":
                print(" [TEST 3.2] Simulating Python exception...")
                raise ValueError("SIMULATED SOFT CRASH: Corrupted tuple data!")
            # ==========================================================


            data_array = np.array(infer_task_data['tuple_data']).reshape(1, -1)
            all_pred = [float(tree.predict(data_array)[0]) for tree in rf.estimators_]
            os.remove(local_model_path)
            
            # Note: Italian keys kept intact to preserve Master-Worker semantic contract
            return {"tipo": "singolo", "valore": all_pred}

        # CASE 2: BULK INFERENCE FROM S3 (Memory-efficient chunking)
        else:
            print(f" [INFER] Bulk inference on full dataset in progress (Chunked)...")
            test_dataset_uri = infer_task_data['test_dataset_uri']
            ml_handler = ModelFactory.get_model(dataset_name=infer_task_data['dataset'])

            chunk_size = config.get("inference_chunksize", 500000)
            print(f" Calculating predictions (Chunksize: {chunk_size})...")
            start_time = time.time()

            all_predictions = []


            for chunk in pd.read_csv(test_dataset_uri, chunksize=chunk_size, low_memory=False):
                # Predict on current chunk only
                chunk_results = ml_handler.process_and_predict(rf, chunk)
                all_predictions.append(chunk_results)

                # Aggressively free RAM
                del chunk, chunk_results
                gc.collect()

            # Merge partial results into a single Numpy array
            numpy_results = np.concatenate(all_predictions)
            print(f" Predictions completed in {time.time() - start_time:.2f}s. Generated {len(numpy_results)} rows.")

            # Compress, save and upload .npy
            local_npy_path = f"/tmp/results_{job_id}_{task_id}.npy"
            np.save(local_npy_path, numpy_results)

            dataset_name = infer_task_data['dataset']
            dataset_variant = infer_task_data.get('dataset_variant', '1M')
            s3_votes_key = f"results/{dataset_name}/{dataset_variant}/{job_id}/task_{task_id}.npy"
            s3_client.upload_file(local_npy_path, bucket, s3_votes_key)

            os.remove(local_model_path)
            os.remove(local_npy_path)
                                  
            # Note: Italian keys kept intact to preserve Master-Worker semantic contract
            return {"tipo": "bulk", "valore": f"s3://{bucket}/{s3_votes_key}"}

    finally:
        stop_event.set()
        heartbeat_thread.join()

# ==========================================
# EVENT LOOP: PRIORITY POLLING
# ==========================================
def main():
    print(" Worker Node initialized and waiting for tasks...")

    while True:
        # We keep track of the current message for error handling
        current_queue = None
        current_receipt = None
        
        try:
            # PRIORITY 1: TRAINING TASKS
            res_train = sqs_client.receive_message(
                QueueUrl=TRAIN_TASK_QUEUE, MaxNumberOfMessages=1, WaitTimeSeconds=5
            )

            if 'Messages' in res_train:
                msg = res_train['Messages'][0]
                current_queue = TRAIN_TASK_QUEUE
                current_receipt = msg['ReceiptHandle']
                
                train_task_data = json.loads(msg['Body'])

                # Execute heavy lifting
                s3_model_uri = train(train_task_data, current_receipt)

                # Respond to master
                train_resp = {
                    "job_id": train_task_data['job_id'],
                    "task_id": train_task_data['task_id'],
                    "s3_model_uri": s3_model_uri
                }
                sqs_client.send_message(QueueUrl=TRAIN_RESPONSE_QUEUE, MessageBody=json.dumps(train_resp))

                # Fault Tolerance: Delete message ONLY upon full success
                sqs_client.delete_message(QueueUrl=TRAIN_TASK_QUEUE, ReceiptHandle=current_receipt)
                print(f" Training {train_task_data['task_id']} completed successfully!\n")

                # CRITICAL: Loop back to check for more training tasks before inferencing
                continue

            # PRIORITY 2: INFERENCE TASKS. Reached ONLY if the training queue is empty.
            res_infer = sqs_client.receive_message(
                QueueUrl=INFER_TASK_QUEUE, MaxNumberOfMessages=1, WaitTimeSeconds=5
            )

            if 'Messages' in res_infer:
                msg = res_infer['Messages'][0]
                current_queue = INFER_TASK_QUEUE
                current_receipt = msg['ReceiptHandle']
                
                infer_task_data = json.loads(msg['Body'])

                # Execute heavy lifting
                s3_votes_uri = execute_inference(infer_task_data, current_receipt)

                # Risponde al Master
                infer_response = {
                    "job_id": infer_task_data['job_id'],
                    "task_id": infer_task_data['task_id'],
                    "s3_voti_uri": s3_votes_uri
                }
                sqs_client.send_message(QueueUrl=INFER_RESPONSE_QUEUE, MessageBody=json.dumps(infer_response))

                # Fault Tolerance: Delete message ONLY upon full success
                sqs_client.delete_message(QueueUrl=INFER_TASK_QUEUE, ReceiptHandle=current_receipt)
                print(f" Inference {infer_task_data['task_id']} completed successfully!\n")
                
                continue 

            # If both queues are empty, rest to avoid throttling AWS APIs
            time.sleep(2)

        except Exception as e:
            
            # SLOW DEATH / OOM MANAGEMENT
            print(f" \n[FAULT TOLERANCE] Critical error detected in Worker: {e}")
            
            # If holding a message, inject it back immediately via Immediate NACK
            if current_queue and current_receipt:
                try:
                    print(" [FAULT TOLERANCE] Executing Immediate Release (VisibilityTimeout=0) for rapid reassignment...")
                    sqs_client.change_message_visibility(
                        QueueUrl=current_queue,
                        ReceiptHandle=current_receipt,
                        VisibilityTimeout=0
                    )
                except Exception as inner_e:
                    print(f" [FAULT TOLERANCE] Could not release message (likely expired): {inner_e}")
            
            # Cooldown to prevent rapid crash looping on infrastructure errors
            time.sleep(10)


if __name__ == "__main__":
    main()
