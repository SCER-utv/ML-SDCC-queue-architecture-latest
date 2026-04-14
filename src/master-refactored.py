import json
import math
import os
import random
import threading
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score, mean_absolute_error, precision_score, recall_score, f1_score

from src.utils.config import load_config
from src.aws.aws_manager import AWSManager  # Il nostro gestore unico per AWS

config = load_config()

CLIENT_QUEUE_URL = config["sqs_queues"]["client"]
TRAIN_TASK_QUEUE = config["sqs_queues"]["train_task"]
TRAIN_RESPONSE_QUEUE = config["sqs_queues"]["train_response"]
INFER_TASK_QUEUE = config["sqs_queues"]["infer_task"]
INFER_RESPONSE_QUEUE = config["sqs_queues"]["infer_response"]

# Inizializziamo il manager AWS (Connection Pooling per S3, EC2, DynamoDB e SQS)
aws = AWSManager(config)

# ==========================================
# HEARTBEAT E UTILITY
# ==========================================
def extend_client_sqs_visibility(queue_url, receipt_handle, stop_event):
    while not stop_event.is_set():
        stop_event.wait(20)
        if not stop_event.is_set():
            try:
                # Usiamo il client centralizzato della classe AWSManager
                aws.sqs_client.change_message_visibility(QueueUrl=queue_url, ReceiptHandle=receipt_handle, VisibilityTimeout=60)
                print(" [HEARTBEAT] Master job timeout reset to 5 minutes.")
            except Exception:
                pass

# ==========================================
# GESTIONE TASK SQS
# ==========================================
def generate_initial_training_tasks(job_data, total_rows=None):
    num_workers = job_data['num_workers']
    num_trees_total = job_data['num_trees']
    dataset = job_data['dataset']
    job_id = job_data['job_id']
    strategy = job_data.get('strategy', 'homogeneous')

    train_s3_uri = job_data['train_s3_url']
    task_type = job_data['task_type']
    target_col = job_data['target_column']

    target_strategies = job_data.get('custom_hyperparams')

    if not target_strategies:
        root_dir = config.get('_root_dir', '.')
        strategies_path = os.path.join(root_dir, 'config', f'{strategy}_tasks.json')
        try:
            with open(strategies_path, 'r') as f:
                all_strategies = json.load(f)
            if strategy == "homogeneous":
                conf = all_strategies.get(dataset, {}).get(str(num_trees_total))
                target_strategies = [conf] * num_workers if conf else []
            else:
                target_strategies = all_strategies.get(dataset, {}).get(str(num_workers), [])
        except FileNotFoundError:
            pass

    if not target_strategies:
        target_strategies = [{"max_depth": None, "max_features": "sqrt", "criterion": "gini" if task_type == 'classification' else 'squared_error'}] * num_workers

    rows_per_worker = total_rows // num_workers
    remainder_rows = total_rows % num_workers
    trees_per_worker = math.floor(num_trees_total / num_workers)
    trees_remainder = num_trees_total % num_workers
    current_skip = 0

    print(f" [INFO] Distributing {num_trees_total} trees across {num_workers} training tasks...")
    for i in range(num_workers):
        trees = trees_per_worker + (1 if i < trees_remainder else 0)
        n_rows = rows_per_worker + (remainder_rows if i == num_workers - 1 else 0)
        conf = target_strategies[i % len(target_strategies)]

        raw_depth = conf.get('max_depth')
        max_depth = None if raw_depth in ["None", None] else (int(raw_depth) if str(raw_depth).lstrip('-').isdigit() else None)

        raw_features = conf.get('max_features', 'sqrt')
        if raw_features in ["sqrt", "log2", "None", None]:
            max_features = raw_features
        else:
            try:
                val_float = float(raw_features)
                max_features = int(val_float) if val_float.is_integer() else val_float
            except (ValueError, TypeError):
                max_features = "sqrt"

        raw_samples = conf.get('max_samples', 1.0)
        try:
            max_samples = float(raw_samples)
        except (ValueError, TypeError):
            max_samples = 1.0

        task_payload = {
            "job_id": job_id, "task_id": f"task_{i + 1}", "seed": i * 1000,
            "dataset": dataset, "dataset_s3_path": train_s3_uri, "trees": trees,
            "skip_rows": current_skip, "num_rows": n_rows,
            "max_depth": max_depth, "max_features": max_features,
            "criterion": conf.get('criterion'), "min_samples_split": conf.get('min_samples_split', 2),
            "min_samples_leaf": conf.get('min_samples_leaf', 1), "max_samples": max_samples,
            "class_weight": conf.get('class_weight', None), "n_jobs": conf.get('n_jobs', -1),
            "is_custom": True, "custom_target_col": target_col, "task_type": task_type
        }

        current_skip += n_rows
        # Utilizziamo aws.sqs_client
        aws.sqs_client.send_message(QueueUrl=TRAIN_TASK_QUEUE, MessageBody=json.dumps(task_payload))
        print(f" Enqueued {task_payload['task_id']} ({trees} trees).")

def generate_inference_tasks(job_data, train_resp):
    infer_task = {
        "job_id": job_data['job_id'], "task_id": train_resp['task_id'],
        "dataset": job_data['dataset'], "dataset_variant": job_data.get('dataset_variant', '1M'),
        "test_dataset_uri": job_data['test_s3_url'], "model_s3_uri": train_resp['s3_model_uri']
    }
    # Utilizziamo aws.sqs_client
    aws.sqs_client.send_message(QueueUrl=INFER_TASK_QUEUE, MessageBody=json.dumps(infer_task))
    print(f" [INFER DISPATCH] Task {train_resp['task_id']} sent to inference queue.")

# ==========================================
# AGGREGAZIONE E VALUTAZIONE
# ==========================================
def aggregate_and_evaluate(job_data, job_id, dataset_name, dataset_variant, s3_inference_results, num_workers, trees, weights, train_time, infer_time, strategy):
    print("\n" + "=" * 50)
    print(" FINAL AGGREGATION & EVALUATION PHASE")
    print("=" * 50)

    task_type = job_data['task_type']
    target_col = job_data['target_column']
    test_s3_uri = job_data['test_s3_url']

    predictions_list = []
    print(f" Downloading {len(s3_inference_results)} inference result files from S3...")

    for task_id, s3_uri in s3_inference_results.items():
        bucket, key = aws.parse_s3_uri(s3_uri)
        local_path = f"/tmp/res_{task_id}.npy"
        aws.s3_client.download_file(bucket, key, local_path)
        result_array = np.load(local_path)
        predictions_list.append(result_array)
        os.remove(local_path)

    if not predictions_list:
        print(" [CRITICAL ERROR] Nessun risultato scaricato. Impossibile aggregare.")
        return

    print(f" Reading Ground Truth from column '{target_col}'...")
    try:
        df_test = pd.read_csv(test_s3_uri, usecols=[target_col])
        y_true = df_test[target_col].values
    except Exception as e:
        print(f" [CRITICAL ERROR] Fallito il caricamento Test Set: {e}")
        return

    data_shape = predictions_list[0].shape

    if task_type == 'classification' or len(data_shape) == 2:
        print(" [EVALUATION] Classification task detected. Executing Majority Voting...")
        total_votes = np.sum(predictions_list, axis=0)

        if len(data_shape) == 2:
            votes_0, votes_1 = total_votes[:, 0], total_votes[:, 1]
            with np.errstate(divide='ignore', invalid='ignore'):
                y_prob = np.where((votes_0 + votes_1) == 0, 0, votes_1 / (votes_0 + votes_1))
        else:
            y_prob = total_votes / num_workers

        final_prediction = np.argmax(total_votes, axis=1) if len(data_shape) == 2 else np.round(total_votes / num_workers)

        try: auc = roc_auc_score(y_true, y_prob)
        except ValueError: auc = 0.0

        acc = accuracy_score(y_true, final_prediction)
        precision = precision_score(y_true, final_prediction, zero_division=0, average='macro')
        recall = recall_score(y_true, final_prediction, zero_division=0, average='macro')
        f1 = f1_score(y_true, final_prediction, zero_division=0, average='macro')

        print(f"\n GLOBAL DISTRIBUTED RESULTS:\n ROC-AUC: {auc:.4f}\n Accuracy: {acc:.4f}\n Precision: {precision:.4f}\n Recall: {recall:.4f}\n F1-Score: {f1:.4f}")
        metrics_dict = {'ROC-AUC': float(round(auc, 4)), 'Accuracy': float(round(acc, 4)), 'Precision': float(round(precision, 4)), 'Recall': float(round(recall, 4)), 'F1-Score': float(round(f1, 4))}
    else:
        print(" [EVALUATION] Regression task detected. Executing Weighted Averaging...")
        y_pred = np.average(predictions_list, axis=0, weights=weights)

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        print(f"\n GLOBAL DISTRIBUTED RESULTS:\n RMSE: {rmse:.4f}\n MAE: {mae:.4f}\n R2 Score: {r2:.4f}")
        metrics_dict = {'RMSE': float(round(rmse, 4)), 'MAE': float(round(mae, 4)), 'R2 Score': float(round(r2, 4))}

    print("=" * 50 + "\n")
    strategy_name = "Homogeneous" if strategy == "homogeneous" else "Heterogeneous"

    aws.save_metrics(test_s3_uri, dataset_name, dataset_variant, num_workers, trees, strategy_name, train_time, infer_time, metrics_dict)
    aws.cleanup_s3_inference_files(s3_inference_results)

# ==========================================
# PIPELINE PRINCIPALI
# ==========================================
def process_training(job_data, job_id):
    num_workers = job_data['num_workers']

    completed_train_tasks, s3_inference_results, db_train_start_time, tasks_dispatched, training_time, inference_time = aws.get_job_state(job_id)

    if db_train_start_time is None:
        total_start_time = start_train = time.time()
        aws.update_job_state(job_id, completed_train_tasks, s3_inference_results, start_train, False, training_time, inference_time)
    else:
        total_start_time = start_train = db_train_start_time
        print(f" [RECOVERY] Restored. Current state: {len(completed_train_tasks)} Train tasks complete.")

    aws.scale_worker_infrastructure(num_workers)
    time.sleep(10)

    calculated_train_rows = None
    needs_split = job_data.get('needs_split', True)
    train_source_url = job_data['train_s3_url']
    bucket, original_key = aws.parse_s3_uri(train_source_url)

    if not tasks_dispatched:
        if needs_split:
            print(f" [PIPELINE] Dataset split required for {original_key}.")
            file_name = original_key.split('/')[-1].replace('.csv', '')
            train_key = f"splits/{file_name}_train.csv"

            if not aws.check_s3_file_exists(bucket, train_key):
                try:
                    calculated_train_rows, new_train_url = aws.execute_streaming_split(train_source_url)
                    job_data['train_s3_url'] = new_train_url
                except Exception as e:
                    print(f" [CRITICAL] Split failed: {e}")
                    return
            else:
                print(" [PIPELINE] Dataset split already exists. Bypassing split.")
                calculated_train_rows = aws.get_total_rows_s3_select(bucket, train_key)
                job_data['train_s3_url'] = f"s3://{bucket}/{train_key}"
        else:
            print(" [PIPELINE] User requested NO SPLIT.")
            calculated_train_rows = aws.get_total_rows_s3_select(bucket, original_key)

    if not tasks_dispatched:
        generate_initial_training_tasks(job_data, total_rows=calculated_train_rows)
        tasks_dispatched = True
        aws.update_job_state(job_id, completed_train_tasks, s3_inference_results, start_train, tasks_dispatched, training_time, inference_time)

    print("\n [EVENT LOOP] Master listening actively for Worker responses...\n")
    while len(completed_train_tasks) < num_workers:
        # Utilizziamo aws.sqs_client
        res_train = aws.sqs_client.receive_message(QueueUrl=TRAIN_RESPONSE_QUEUE, MaxNumberOfMessages=10, WaitTimeSeconds=2)
        if 'Messages' in res_train:
            for msg in res_train['Messages']:
                train_resp = json.loads(msg['Body'])
                task_id = train_resp['task_id']
                if task_id not in completed_train_tasks:
                    completed_train_tasks.add(task_id)
                    print(f" [ACK] Worker completed training for {task_id}! ({len(completed_train_tasks)}/{num_workers})")
                    aws.update_job_state(job_id, completed_train_tasks, {}, start_train, tasks_dispatched, 0.0, 0.0)
                aws.sqs_client.delete_message(QueueUrl=TRAIN_RESPONSE_QUEUE, ReceiptHandle=msg['ReceiptHandle'])

    training_time = time.time() - start_train
    aws.update_job_state(job_id, completed_train_tasks, {}, start_train, tasks_dispatched, training_time, 0.0)

    print("\n [PIPELINE] All Workers completed their Training tasks!")
    total_run_time = time.time() - total_start_time
    print(f" [TIMERS] Distributed Training completed in {total_run_time:.2f}s")

    if job_data.get('mode') == 'train':
        client_response_queue = config.get("sqs_queues", {}).get("client_response")
        if client_response_queue:
            aws.sqs_client.send_message(QueueUrl=client_response_queue, MessageBody=json.dumps({"job_id": job_id, "target_model": job_id, "mode": "train", "total_time_sec": round(total_run_time, 2)}))

def process_bulk_inference(job_data, job_id):
    dataset, dataset_variant, target_model = job_data['dataset'], job_data.get('dataset_variant', '1M'), job_data['target_model']
    total_start_time = job_data.get('client_start_time', time.time())

    model_s3_uris = aws.count_model_parts(aws.bucket, dataset, target_model)
    num_workers = len(model_s3_uris)

    print(f" [BULK-INFER] Target model '{target_model}' chunked into {num_workers} parts. Scaling workers...")
    aws.scale_worker_infrastructure(num_workers)

    _, _, _, _, historical_train_time, _ = aws.get_job_state(target_model)
    _, s3_inference_results, db_infer_start, _, _, inference_time = aws.get_job_state(job_id)
    start_infer = db_infer_start if db_infer_start else time.time()

    for i, uri in enumerate(model_s3_uris):
        task_id = f"task_{i + 1}"
        if task_id not in s3_inference_results:
            generate_inference_tasks(job_data, {"task_id": task_id, "s3_model_uri": uri})

    aws.update_job_state(job_id, set(), s3_inference_results, start_infer, True, historical_train_time, inference_time)

    while len(s3_inference_results) < num_workers:
        # Utilizziamo aws.sqs_client
        res_infer = aws.sqs_client.receive_message(QueueUrl=INFER_RESPONSE_QUEUE, MaxNumberOfMessages=10, WaitTimeSeconds=2)
        if 'Messages' in res_infer:
            for msg in res_infer['Messages']:
                body = json.loads(msg['Body'])
                task_id = body['task_id']
                s3_votes_uri = body['s3_voti_uri']['valore'] if isinstance(body['s3_voti_uri'], dict) else body['s3_voti_uri']

                if task_id not in s3_inference_results:
                    s3_inference_results[task_id] = s3_votes_uri
                    print(f" [ACK] Worker completed Bulk Inference for {task_id}! ({len(s3_inference_results)}/{num_workers})")
                    aws.update_job_state(job_id, set(), s3_inference_results, start_infer, True, historical_train_time, time.time() - start_infer)
                aws.sqs_client.delete_message(QueueUrl=INFER_RESPONSE_QUEUE, ReceiptHandle=msg['ReceiptHandle'])

    inference_time = time.time() - start_infer
    try: num_trees = int(target_model.split('_')[3].replace('trees', ''))
    except: num_trees = num_workers * 10
    weights = [math.floor(num_trees / num_workers) + (1 if i < (num_trees % num_workers) else 0) for i in range(num_workers)]
    strat = "homogeneous" if "homogeneous" in target_model else "heterogeneous"

    aggregate_and_evaluate(job_data, job_id, dataset, dataset_variant, s3_inference_results, num_workers, num_trees, weights, historical_train_time, inference_time, strat)

    client_response_queue = config.get("sqs_queues", {}).get("client_response")
    if client_response_queue:
        aws.sqs_client.send_message(QueueUrl=client_response_queue, MessageBody=json.dumps({"job_id": job_id, "mode": job_data.get('mode', 'bulk_infer'), "total_time_sec": round(time.time() - total_start_time, 2)}))

def process_realtime_inference(job_data, job_id):
    dataset, target_model, tuple_data, task_type = job_data['dataset'], job_data['target_model'], job_data['tuple_data'], job_data['task_type']
    total_start_time = job_data.get('client_start_time', time.time())

    model_s3_uris = aws.count_model_parts(aws.bucket, dataset, target_model)
    num_workers = len(model_s3_uris)

    print(f" [REAL-TIME] Target model '{target_model}' chunked into {num_workers} parts. Scaling workers...")
    start_provisioning = time.time()
    aws.scale_worker_infrastructure(num_workers)
    provisioning_time = time.time() - start_provisioning

    inference_pure_start = time.time()
    for i, uri in enumerate(model_s3_uris):
        aws.sqs_client.send_message(QueueUrl=INFER_TASK_QUEUE, MessageBody=json.dumps({"job_id": job_id, "task_id": f"task_infer_rt_{i + 1}", "dataset": dataset, "model_s3_uri": uri, "tuple_data": tuple_data}))

    total_received_votes, read_messages = [], 0
    while read_messages < num_workers:
        # Utilizziamo aws.sqs_client
        res = aws.sqs_client.receive_message(QueueUrl=INFER_RESPONSE_QUEUE, WaitTimeSeconds=2)
        if 'Messages' in res:
            for msg in res['Messages']:
                body = json.loads(msg['Body'])
                res_data = body['s3_voti_uri']
                if isinstance(res_data, dict) and res_data.get("tipo") == "singolo":
                    total_received_votes.extend(res_data['valore'])
                    read_messages += 1
                    print(f"   -> Gathered {len(res_data['valore'])} votes from worker.")
                aws.sqs_client.delete_message(QueueUrl=INFER_RESPONSE_QUEUE, ReceiptHandle=msg['ReceiptHandle'])

    pure_inference_time = time.time() - inference_pure_start
    if task_type == 'classification':
        final_prediction = max(set(total_received_votes), key=total_received_votes.count)
        task_str = f"Classification (Majority Vote). Class 0: {total_received_votes.count(0)} | Class 1: {total_received_votes.count(1)}"
    else:
        final_prediction = sum(total_received_votes) / len(total_received_votes)
        task_str = "Regression (Mean)"

    total_run_time = time.time() - total_start_time
    print(f"\n{'='*60}\n REAL-TIME PREDICTION: {final_prediction:.2f}\n{'-'*60}\n AWS Prov. Time: {provisioning_time:.2f}s\n Pure Infer Time: {pure_inference_time:.2f}s\n Global Latency: {total_run_time:.2f}s\n{'='*60}\n")

    client_response_queue = config.get("sqs_queues", {}).get("client_response")
    if client_response_queue:
        aws.sqs_client.send_message(QueueUrl=client_response_queue, MessageBody=json.dumps({"job_id": job_id, "prediction": float(final_prediction), "task_type": task_str, "total_time_sec": round(total_run_time, 2)}))

# ==========================================
# MAIN LOOP
# ==========================================
def main():
    print(" Master Node initialized. Waiting for Client jobs...")

    while True:
        # Utilizziamo aws.sqs_client
        response = aws.sqs_client.receive_message(QueueUrl=CLIENT_QUEUE_URL, MaxNumberOfMessages=1, WaitTimeSeconds=20)
        if 'Messages' in response:
            client_msg = response['Messages'][0]
            receipt_handle = client_msg['ReceiptHandle']
            job_data = json.loads(client_msg['Body'])

            job_id = job_data.setdefault('job_id', client_msg['MessageId'])
            mode = job_data.get('mode', 'train')

            print(f"\n{'='*50}\n INITIALIZING ASYNC PIPELINE FOR JOB: {job_id}\n{'='*50}")

            stop_event_master = threading.Event()
            heartbeat_thread_master = threading.Thread(target=extend_client_sqs_visibility, args=(CLIENT_QUEUE_URL, receipt_handle, stop_event_master))
            heartbeat_thread_master.start()

            try:
                if mode == 'train': process_training(job_data, job_id)
                elif mode == 'bulk_infer': process_bulk_inference(job_data, job_id)
                elif mode == 'infer': process_realtime_inference(job_data, job_id)
                elif mode == 'train_and_infer':
                    process_training(job_data, job_id)
                    job_data['target_model'] = job_id
                    process_bulk_inference(job_data, job_id)
                else: print(f" [WARNING] Unknown mode requested: {mode}")
            except Exception as e:
                print(f" [CRITICAL ERROR] Pipeline execution failed: {e}")
            finally:
                stop_event_master.set()
                heartbeat_thread_master.join()
                aws.sqs_client.delete_message(QueueUrl=CLIENT_QUEUE_URL, ReceiptHandle=receipt_handle)
                print(f" JOB {job_id} PROCESSED AND REMOVED FROM QUEUE.\n")

if __name__ == "__main__":
    main()