import io
import json
import math
import os
import random
import threading
import time

import boto3
import botocore
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score, mean_absolute_error, \
    precision_score, recall_score, f1_score

from src.utils.config import load_config

config = load_config()

AWS_REGION = config.get("aws_region")
ASG_NAME = config.get("asg_name")
TARGET_BUCKET = config.get("s3_bucket")

CLIENT_QUEUE_URL = config["sqs_queues"]["client"]
TRAIN_TASK_QUEUE = config["sqs_queues"]["train_task"]
TRAIN_RESPONSE_QUEUE = config["sqs_queues"]["train_response"]
INFER_TASK_QUEUE = config["sqs_queues"]["infer_task"]
INFER_RESPONSE_QUEUE = config["sqs_queues"]["infer_response"]

sqs_client = boto3.client('sqs', region_name=AWS_REGION)
asg_client = boto3.client('autoscaling', region_name=AWS_REGION)


# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def parse_s3_uri(s3_uri):
    parts = s3_uri.replace("s3://", "").split("/", 1)
    return parts[0], parts[1]


# Counts and retrieves S3 URIs of distributed model chunks
def count_model_parts(bucket, dataset, target_model):
    # DEDUZIONE INTELLIGENTE: estraiamo il dataset originale direttamente dal nome del modello!
    try:
        model_dataset_folder = target_model.split('_')[1]
    except Exception:
        # Fallback di sicurezza se l'ID ha una formattazione strana
        model_dataset_folder = dataset

    s3 = boto3.client('s3', region_name=AWS_REGION)
    # Cerchiamo nella cartella giusta!
    prefix = f"models/{model_dataset_folder}/{target_model}/"
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

    chunks = [f"s3://{bucket}/{obj['Key']}" for obj in resp.get('Contents', []) if obj['Key'].endswith('.joblib')]

    if not chunks:
        print(f" [WARNING S3] Nessun file .joblib trovato in s3://{bucket}/{prefix}")

    return chunks


# Background thread to periodically extend SQS message visibility timeout
def extend_client_sqs_visibility(queue_url, receipt_handle, stop_event):
    while not stop_event.is_set():
        stop_event.wait(20)
        if not stop_event.is_set():
            try:
                sqs_client.change_message_visibility(
                    QueueUrl=queue_url,
                    ReceiptHandle=receipt_handle,
                    VisibilityTimeout=60
                )
                print(" [HEARTBEAT] Master job timeout reset to 5 minutes.")
            except Exception as e:
                pass


def scale_worker_infrastructure(num_workers):
    print(f" [ASG] Setting desired capacity to {num_workers} workers...")
    asg_client.update_auto_scaling_group(
        AutoScalingGroupName=ASG_NAME, MinSize=0, DesiredCapacity=num_workers, MaxSize=10
    )

    if num_workers == 0:
        return

    print(f" [ASG] Waiting for instances to start for tagging...")
    ec2_client = boto3.client('ec2', region_name=AWS_REGION)

    max_wait = 24  # 24 * 5 sec = 120 seconds max
    found_instances = []

    for _ in range(max_wait):
        time.sleep(5)
        response = ec2_client.describe_instances(
            Filters=[
                {'Name': 'tag:aws:autoscaling:groupName', 'Values': [ASG_NAME]},
                {'Name': 'instance-state-name', 'Values': ['pending', 'running']}
            ]
        )

        for reservation in response.get('Reservations', []):
            for inst in reservation.get('Instances', []):
                found_instances.append(inst['InstanceId'])

        if len(found_instances) >= num_workers:
            break

    if len(found_instances) > 0:
        if len(found_instances) < num_workers:
            print(
                f" [ASG WARN] Requested {num_workers} workers, but AWS provided {len(found_instances)}. Proceeding degraded.")
        else:
            print(f" [ASG] Found {len(found_instances)} instances. Applying name tags...")

        for i, instance_id in enumerate(found_instances):
            worker_name = f"DRF-worker{i + 1}"
            try:
                ec2_client.create_tags(
                    Resources=[instance_id],
                    Tags=[{'Key': 'Name', 'Value': worker_name}]
                )
            except Exception:
                pass
        print(" [ASG] Name tags applied successfully.")
    else:
        print(" [ASG CRITICAL] No instances provided by ASG within timeout!")


# Executes an S3 Select query to quickly count rows in a CSV without downloading it.
def get_total_rows_s3_select(bucket, key):
    print(f" [S3-SELECT] Executing 'SELECT count(*)' on s3://{bucket}/{key}...")
    s3 = boto3.client('s3')
    try:
        resp = s3.select_object_content(
            Bucket=bucket, Key=key,
            ExpressionType='SQL', Expression='SELECT count(*) FROM S3Object',
            InputSerialization={'CSV': {'FileHeaderInfo': 'USE', 'AllowQuotedRecordDelimiter': False}},
            OutputSerialization={'CSV': {}}
        )
        for event in resp['Payload']:
            if 'Records' in event:
                total_rows = int(event['Records']['Payload'].decode('utf-8').strip())
                print(f" [S3 Select] Found {total_rows} rows!")
                return total_rows
        return 0
    except Exception as e:
        print(f" [S3-SELECT ERROR] Failed query: {e}")
        raise e


# DATA-AGNOSTIC SPLIT: Prende un URL, lo splitta e salva i due nuovi file
def execute_streaming_split(source_url):
    config = load_config()
    ratios = config.get("split_ratios", {"train": 0.70})
    train_threshold = ratios.get("train", 0.70)
    s3 = boto3.client('s3', region_name=AWS_REGION)

    bucket, source_key = parse_s3_uri(source_url)
    file_name = source_key.split('/')[-1].replace('.csv', '')

    print(f" [SPLIT] Starting dynamic streaming split for '{file_name}'...")

    train_key = f"splits/{file_name}_train.csv"
    test_key = f"splits/{file_name}_test.csv"

    local_train = f"/tmp/{file_name}_train.csv"
    local_test = f"/tmp/{file_name}_test.csv"

    try:
        print(f" [SPLIT] Line-by-line streaming in progress...")
        response = s3.get_object(Bucket=bucket, Key=source_key)

        safe_streaming = io.TextIOWrapper(response['Body'], encoding='utf-8')

        with open(local_train, 'w', encoding='utf-8') as f_train, open(local_test, 'w', encoding='utf-8') as f_test:
            header = safe_streaming.readline()
            f_train.write(header)
            f_test.write(header)

            train_rows = 0
            test_rows = 0

            for line in safe_streaming:
                if line.strip():
                    if random.random() <= train_threshold:
                        f_train.write(line)
                        train_rows += 1
                    else:
                        f_test.write(line)
                        test_rows += 1

        print(f" [SPLIT] Finished. Train: {train_rows} rows | Test: {test_rows} rows.")
        print(" [SPLIT] Uploading to S3...")

        s3.upload_file(local_train, bucket, train_key)
        s3.upload_file(local_test, bucket, test_key)

    except Exception as e:
        print(f" [SPLIT ERROR] Failed during streaming split: {e}")
        raise e
    finally:
        if os.path.exists(local_train): os.remove(local_train)
        if os.path.exists(local_test): os.remove(local_test)

    print(f" [SPLIT] Operation completed successfully.")
    return train_rows, f"s3://{bucket}/{train_key}"


def check_s3_file_exists(bucket, key):
    s3 = boto3.client('s3', region_name=AWS_REGION)
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            return False
        raise e


# Calculates dataset splits based on total rows and generates SQS payloads for worker nodes.
def generate_initial_training_tasks(job_data, total_rows=None):
    num_workers = job_data['num_workers']
    num_trees_total = job_data['num_trees']
    dataset = job_data['dataset']
    job_id = job_data['job_id']
    strategy = job_data.get('strategy', 'homogeneous')

    # DATA-AGNOSTIC EXTRACTION
    train_s3_uri = job_data['train_s3_url']
    task_type = job_data['task_type']
    target_col = job_data['target_column']

    target_strategies = job_data.get('custom_hyperparams')

    # Se non abbiamo iperparametri custom dal Client, li carichiamo dai JSON di configurazione
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

    # Fallback definitivo se mancano strategie
    if not target_strategies:
        target_strategies = [{"max_depth": None, "max_features": "sqrt",
                              "criterion": "gini" if task_type == 'classification' else 'squared_error'}] * num_workers

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

        raw_depth = conf['max_depth']
        max_depth = None if raw_depth == "None" else (int(raw_depth) if str(raw_depth).isdigit() else None)

        raw_features = conf['max_features']
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
            "job_id": job_id,
            "task_id": f"task_{i + 1}",
            "seed": i * 1000,
            "dataset": dataset,
            "dataset_s3_path": train_s3_uri,
            "trees": trees,
            "skip_rows": current_skip,
            "num_rows": n_rows,

            "max_depth": max_depth,
            "max_features": max_features,
            "criterion": conf.get('criterion'),
            "min_samples_split": conf.get('min_samples_split', 2),
            "min_samples_leaf": conf.get('min_samples_leaf', 1),
            "max_samples": max_samples,
            "class_weight": conf.get('class_weight', None),
            "n_jobs": conf.get('n_jobs', -1),

            # Forziamo il worker a essere data-agnostic dicendogli di leggere sempre custom_target_col
            "is_custom": True,
            "custom_target_col": target_col,
            "task_type": task_type
        }

        current_skip += n_rows
        sqs_client.send_message(QueueUrl=TRAIN_TASK_QUEUE, MessageBody=json.dumps(task_payload))
        print(f" Enqueued {task_payload['task_id']} ({trees} trees).")


# Enqueues inference tasks mapped to completed training chunks
def generate_inference_tasks(job_data, train_resp):
    # DATA-AGNOSTIC: Usiamo l'URL del test set che il client ha preparato
    infer_task = {
        "job_id": job_data['job_id'],
        "task_id": train_resp['task_id'],
        "dataset": job_data['dataset'],
        "dataset_variant": job_data.get('dataset_variant', '1M'),
        "test_dataset_uri": job_data['test_s3_url'],
        "model_s3_uri": train_resp['s3_model_uri']
    }
    sqs_client.send_message(QueueUrl=INFER_TASK_QUEUE, MessageBody=json.dumps(infer_task))
    print(f" [INFER DISPATCH] Task {train_resp['task_id']} sent to inference queue.")


# Appends final job metrics to a persistent S3 CSV file
def save_metrics(test_set_url, dataset_name, dataset_variant, n_workers, n_trees, strategy_name, train_time, inf_time, metrics_dict,
                 config):
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    target_bucket = config.get("s3_bucket", "distributed-random-forest-bkt")
    s3_key = f"results/{dataset_name}/{dataset_name}_{dataset_variant}_distributed_results.csv"

    row_data = {
        'Test_set_url': test_set_url,
        'Workers': n_workers,
        'Trees': n_trees,
        'System_type': "Distributed",
        'Strategy': strategy_name,
        'Train_Time': round(train_time, 2),
        'Infer_Time': round(inf_time, 2)
    }
    row_data.update(metrics_dict)
    new_row_df = pd.DataFrame([row_data])

    try:
        obj = s3_client.get_object(Bucket=target_bucket, Key=s3_key)
        df_existing = pd.read_csv(io.BytesIO(obj['Body'].read()), sep=';', decimal=',')
        df_final = pd.concat([df_existing, new_row_df], ignore_index=True)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            df_final = new_row_df
        else:
            print(f" [METRICS ERROR] Unexpected S3 error: {e}")
            return

    csv_buffer = io.StringIO()
    df_final.to_csv(csv_buffer, index=False, sep=';', decimal=',')
    s3_client.put_object(Bucket=target_bucket, Key=s3_key, Body=csv_buffer.getvalue())
    print(f" [METRICS] Results securely appended to: s3://{target_bucket}/{s3_key}")


# Downloads partial inferences, executes majority vote/averaging, and calculates global metrics
def aggregate_and_evaluate(job_data, job_id, dataset_name, dataset_variant, s3_inference_results, num_workers, trees,
                           weights, train_time, infer_time, strategy):
    print("\n" + "=" * 50)
    print(" FINAL AGGREGATION & EVALUATION PHASE")
    print("=" * 50)

    s3 = boto3.client('s3')

    # DATA-AGNOSTIC: Leggiamo direttamente le info dal payload, zero branch logici!
    task_type = job_data['task_type']
    target_col = job_data['target_column']
    test_s3_uri = job_data['test_s3_url']

    predictions_list = []
    print(f" Downloading {len(s3_inference_results)} inference result files from S3...")

    for task_id, s3_uri in s3_inference_results.items():
        bucket, key = parse_s3_uri(s3_uri)
        local_path = f"/tmp/res_{task_id}.npy"
        s3.download_file(bucket, key, local_path)

        result_array = np.load(local_path)
        predictions_list.append(result_array)
        os.remove(local_path)

    if not predictions_list:
        print(" [CRITICAL ERROR] Nessun risultato di inferenza scaricato. Impossibile aggregare le metriche.")
        return

    print(f" Reading Ground Truth from column '{target_col}'...")
    try:
        df_test = pd.read_csv(test_s3_uri, usecols=[target_col])
        y_true = df_test[target_col].values
    except Exception as e:
        print(f" [CRITICAL ERROR] Fallito il caricamento del Test Set da {test_s3_uri}. Errore: {e}")
        return

    data_shape = predictions_list[0].shape

    if task_type == 'classification' or len(data_shape) == 2:
        print(" [EVALUATION] Classification task detected. Executing Majority Voting...")
        total_votes = np.sum(predictions_list, axis=0)

        if len(data_shape) == 2:
            votes_0 = total_votes[:, 0]
            votes_1 = total_votes[:, 1]
            with np.errstate(divide='ignore', invalid='ignore'):
                y_prob = np.where((votes_0 + votes_1) == 0, 0, votes_1 / (votes_0 + votes_1))
        else:
            y_prob = total_votes / num_workers

        final_prediction = np.argmax(total_votes, axis=1) if len(data_shape) == 2 else np.round(
            total_votes / num_workers)

        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.0

        acc = accuracy_score(y_true, final_prediction)
        precision = precision_score(y_true, final_prediction, zero_division=0, average='macro')
        recall = recall_score(y_true, final_prediction, zero_division=0, average='macro')
        f1 = f1_score(y_true, final_prediction, zero_division=0, average='macro')

        print(f"\n GLOBAL DISTRIBUTED RESULTS:")
        print(f" ROC-AUC:   {auc:.4f}")
        print(f" Accuracy:  {acc:.4f}")
        print(f" Precision: {precision:.4f}")
        print(f" Recall:    {recall:.4f}")
        print(f" F1-Score:  {f1:.4f}")

        metrics_dict = {
            'ROC-AUC': float(round(auc, 4)), 'Accuracy': float(round(acc, 4)),
            'Precision': float(round(precision, 4)), 'Recall': float(round(recall, 4)),
            'F1-Score': float(round(f1, 4))
        }

    else:
        print(" [EVALUATION] Regression task detected. Executing Weighted Averaging...")
        y_pred = np.average(predictions_list, axis=0, weights=weights)

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        print(f"\n GLOBAL DISTRIBUTED RESULTS:")
        print(f" RMSE: {rmse:.4f}")
        print(f" MAE: {mae:.4f}")
        print(f" R2 Score: {r2:.4f}")

        metrics_dict = {
            'RMSE': float(round(rmse, 4)), 'MAE': float(round(mae, 4)), 'R2 Score': float(round(r2, 4))
        }

    print("=" * 50 + "\n")

    strategy_name = "Homogeneous" if strategy == "homogeneous" else "Heterogeneous"

    save_metrics(
        test_set_url=test_s3_uri, dataset_name=dataset_name, dataset_variant=dataset_variant, n_workers=num_workers,
        n_trees=trees, strategy_name=strategy_name, train_time=train_time,
        inf_time=infer_time, metrics_dict=metrics_dict, config=load_config()
    )

    cleanup_s3_inference_files(s3_inference_results)


def get_job_state(job_id):
    table_name = load_config().get("dynamodb_table", "JobStatus")
    dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
    table = dynamodb.Table(table_name)
    try:
        response = table.get_item(Key={'job_id': job_id})
        if 'Item' in response:
            start_time = float(response['Item'].get('start_time'))
            tasks_dispatched = response['Item'].get('tasks_dispatched', False)
            training_time = float(response['Item'].get('tempo_training', 0.0))
            inference_time = float(response['Item'].get('tempo_inferenza', 0.0))

            return (set(response['Item'].get('completed_train', [])),
                    response['Item'].get('completed_infer', {}),
                    start_time, tasks_dispatched, training_time, inference_time)
    except Exception:
        pass
    return set(), {}, None, False, 0.0, 0.0


def update_job_state(job_id, completed_train_set, completed_infer_dict, start_time, tasks_dispatched, training_time=0.0,
                     inference_time=0.0):
    table_name = load_config().get("dynamodb_table", "JobStatus")
    dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
    table = dynamodb.Table(table_name)
    table.put_item(Item={
        'job_id': job_id,
        'completed_train': list(completed_train_set),
        'completed_infer': completed_infer_dict,
        'start_time': str(start_time),
        'tasks_dispatched': tasks_dispatched,
        'tempo_training': str(training_time),
        'tempo_inferenza': str(inference_time)
    })


def cleanup_s3_inference_files(s3_inference_results):
    s3 = boto3.client('s3', region_name=AWS_REGION)
    print(" [CLEANUP] Deleting temporary .npy from S3...")
    deleted_count = 0
    for task_id, s3_uri in s3_inference_results.items():
        try:
            bucket, key = parse_s3_uri(s3_uri)
            s3.delete_object(Bucket=bucket, Key=key)
            deleted_count += 1
        except Exception as e:
            print(f" [CLEANUP ERROR] Delete error of {s3_uri}: {e}")
    print(f" [CLEANUP] Removed {deleted_count} temporary files successfully.")


def process_training(job_data, job_id):
    """Gestisce l'intera logica di provisioning, split e addestramento distribuito in modo DATA-AGNOSTIC."""
    num_workers = job_data['num_workers']

    completed_train_tasks, s3_inference_results, db_train_start_time, tasks_dispatched, training_time, inference_time = get_job_state(
        job_id)

    if db_train_start_time is None:
        total_start_time = time.time()
        start_train = time.time()
        update_job_state(job_id, completed_train_tasks, s3_inference_results, start_train, False, training_time,
                         inference_time)
    else:
        total_start_time = db_train_start_time
        start_train = db_train_start_time
        print(f" [RECOVERY] Global timers restored from Database.")
        print(f" [RECOVERY] Current state: {len(completed_train_tasks)} Train tasks complete.")

    scale_worker_infrastructure(num_workers)
    time.sleep(10)

    # 3. Dynamic Data Split (DATA-AGNOSTIC)
    calculated_train_rows = None
    needs_split = job_data.get('needs_split', True)
    train_source_url = job_data['train_s3_url']
    bucket, original_key = parse_s3_uri(train_source_url)

    if not tasks_dispatched:
        if needs_split:
            print(f" [PIPELINE] Dataset split required for {original_key}.")
            file_name = original_key.split('/')[-1].replace('.csv', '')
            train_key = f"splits/{file_name}_train.csv"

            if not check_s3_file_exists(bucket, train_key):
                try:
                    calculated_train_rows, new_train_url = execute_streaming_split(train_source_url)
                    job_data['train_s3_url'] = new_train_url
                except Exception as e:
                    print(f" [CRITICAL] Split failed: {e}")
                    return
            else:
                print(" [PIPELINE] Dataset split already exists. Bypassing split.")
                calculated_train_rows = get_total_rows_s3_select(bucket, train_key)
                job_data['train_s3_url'] = f"s3://{bucket}/{train_key}"
        else:
            print(" [PIPELINE] User requested NO SPLIT. Using provided URL as raw Training Set.")
            try:
                calculated_train_rows = get_total_rows_s3_select(bucket, original_key)
            except Exception as e:
                print(f" [CRITICAL] Failed to read S3 URL: {e}")
                return
    else:
        print(" [RECOVERY] Split block skipped. Tasks already dispatched.")

    if not tasks_dispatched:
        generate_initial_training_tasks(job_data, total_rows=calculated_train_rows)
        tasks_dispatched = True
        update_job_state(job_id, completed_train_tasks, s3_inference_results, start_train, tasks_dispatched,
                         training_time, inference_time)
    else:
        print(" [RECOVERY] SQS Fan-Out skipped to prevent duplicates.")

    print("\n [EVENT LOOP] Master listening actively for Worker responses...\n")
    while len(completed_train_tasks) < num_workers:
        res_train = sqs_client.receive_message(QueueUrl=TRAIN_RESPONSE_QUEUE, MaxNumberOfMessages=10, WaitTimeSeconds=2)
        if 'Messages' in res_train:
            for msg in res_train['Messages']:
                train_resp = json.loads(msg['Body'])
                task_id = train_resp['task_id']

                if task_id not in completed_train_tasks:
                    completed_train_tasks.add(task_id)
                    print(
                        f" [ACK] Worker completed training for {task_id}! ({len(completed_train_tasks)}/{num_workers})")
                    update_job_state(job_id, completed_train_tasks, {}, start_train, tasks_dispatched, 0.0, 0.0)

                sqs_client.delete_message(QueueUrl=TRAIN_RESPONSE_QUEUE, ReceiptHandle=msg['ReceiptHandle'])

    training_time = time.time() - start_train
    update_job_state(job_id, completed_train_tasks, {}, start_train, tasks_dispatched, training_time, 0.0)

    print("\n [PIPELINE] All Workers completed their Training tasks!")
    # scale_worker_infrastructure(0)

    total_run_time = time.time() - total_start_time
    print(f" [TIMERS] Distributed Training completed in {total_run_time:.2f}s")

    if job_data.get('mode') == 'train':
        client_response_queue = config.get("sqs_queues", {}).get("client_response")
        if client_response_queue:
            response_payload = {
                "job_id": job_id, "target_model": job_id, "mode": "train", "total_time_sec": round(total_run_time, 2)
            }
            sqs_client.send_message(QueueUrl=client_response_queue, MessageBody=json.dumps(response_payload))


def process_bulk_inference(job_data, job_id):
    dataset = job_data['dataset']
    dataset_variant = job_data.get('dataset_variant', '1M')
    target_model = job_data['target_model']
    total_start_time = job_data.get('client_start_time', time.time())

    bucket = load_config().get("s3_bucket")
    model_s3_uris = count_model_parts(bucket, dataset, target_model)
    num_workers = len(model_s3_uris)

    print(f" [BULK-INFER] Target model '{target_model}' chunked into {num_workers} parts. Scaling workers...")
    scale_worker_infrastructure(num_workers)

    _, _, _, _, historical_train_time, _ = get_job_state(target_model)

    # FIX: Sostituiamo tasks_dispatched con un "_" per ignorarlo deliberatamente.
    # Quel 'True' apparteneva al Training, non deve bloccarci qui!
    _, s3_inference_results, db_infer_start, _, _, inference_time = get_job_state(job_id)

    start_infer = db_infer_start if db_infer_start else time.time()

    # FIX: Togliamo l'if not tasks_dispatched che bloccava il Fan-Out.
    # Inviamo i task in coda. Se crashiamo e ripartiamo, re-invieremo solo quelli mancanti.
    for i, uri in enumerate(model_s3_uris):
        task_id = f"task_{i + 1}"
        if task_id not in s3_inference_results:
            train_resp_mock = {"task_id": task_id, "s3_model_uri": uri}
            generate_inference_tasks(job_data, train_resp_mock)

    # Aggiorniamo il DB dicendo che stiamo lavorando
    update_job_state(job_id, set(), s3_inference_results, start_infer, True, historical_train_time, inference_time)

    while len(s3_inference_results) < num_workers:
        res_infer = sqs_client.receive_message(QueueUrl=INFER_RESPONSE_QUEUE, MaxNumberOfMessages=10, WaitTimeSeconds=2)
        if 'Messages' in res_infer:
            for msg in res_infer['Messages']:
                body = json.loads(msg['Body'])
                task_id = body['task_id']
                s3_votes_uri = body['s3_voti_uri']['valore'] if isinstance(body['s3_voti_uri'], dict) else body[
                    's3_voti_uri']

                if task_id not in s3_inference_results:
                    s3_inference_results[task_id] = s3_votes_uri
                    print(
                        f" [ACK] Worker completed Bulk Inference for {task_id}! ({len(s3_inference_results)}/{num_workers})")
                    update_job_state(job_id, set(), s3_inference_results, start_infer, True, historical_train_time,
                                     time.time() - start_infer)

                sqs_client.delete_message(QueueUrl=INFER_RESPONSE_QUEUE, ReceiptHandle=msg['ReceiptHandle'])

    inference_time = time.time() - start_infer
    total_run_time = time.time() - total_start_time

    try:
        num_trees = int(target_model.split('_')[3].replace('trees', ''))
    except:
        num_trees = num_workers * 10

    weights = [math.floor(num_trees / num_workers) + (1 if i < (num_trees % num_workers) else 0) for i in
               range(num_workers)]
    strat = "homogeneous" if "homogeneous" in target_model else "heterogeneous"

    aggregate_and_evaluate(job_data, job_id, dataset, dataset_variant, s3_inference_results, num_workers, num_trees,
                           weights, historical_train_time, inference_time, strat)

    client_response_queue = load_config().get("sqs_queues", {}).get("client_response")
    if client_response_queue:
        mode_to_return = job_data.get('mode', 'bulk_infer')
        sqs_client.send_message(QueueUrl=client_response_queue, MessageBody=json.dumps(
            {"job_id": job_id, "mode": mode_to_return, "total_time_sec": round(total_run_time, 2)}))


def process_realtime_inference(job_data, job_id):
    dataset = job_data['dataset']
    target_model = job_data['target_model']
    tuple_data = job_data['tuple_data']
    task_type = job_data['task_type']  # DATA-AGNOSTIC
    total_start_time = job_data.get('client_start_time', time.time())

    bucket = load_config().get("s3_bucket")
    model_s3_uris = count_model_parts(bucket, dataset, target_model)
    num_workers = len(model_s3_uris)

    print(f" [REAL-TIME] Target model '{target_model}' chunked into {num_workers} parts. Scaling workers...")

    start_provisioning = time.time()
    scale_worker_infrastructure(num_workers)
    provisioning_time = time.time() - start_provisioning

    inference_pure_start = time.time()
    for i, uri in enumerate(model_s3_uris):
        task_id = f"task_infer_rt_{i + 1}"
        infer_task = {
            "job_id": job_id, "task_id": task_id, "dataset": dataset,
            "model_s3_uri": uri, "tuple_data": tuple_data
        }
        sqs_client.send_message(QueueUrl=INFER_TASK_QUEUE, MessageBody=json.dumps(infer_task))

    total_received_votes = []
    read_messages = 0

    while read_messages < num_workers:
        res = sqs_client.receive_message(QueueUrl=INFER_RESPONSE_QUEUE, WaitTimeSeconds=2)
        if 'Messages' in res:
            for msg in res['Messages']:
                body = json.loads(msg['Body'])
                res_data = body['s3_voti_uri']

                if isinstance(res_data, dict) and res_data.get("tipo") == "singolo":
                    worker_predictions = res_data['valore']
                    total_received_votes.extend(worker_predictions)
                    read_messages += 1
                    print(f"   -> Gathered {len(worker_predictions)} votes from worker.")

                sqs_client.delete_message(QueueUrl=INFER_RESPONSE_QUEUE, ReceiptHandle=msg['ReceiptHandle'])

    pure_inference_time = time.time() - inference_pure_start
    # scale_worker_infrastructure(0)

    if task_type == 'classification':
        final_prediction = max(set(total_received_votes), key=total_received_votes.count)
        task_str = "Classification (Majority Vote)"
        votes_0 = total_received_votes.count(0)
        votes_1 = total_received_votes.count(1)
        print(f" [POLL] Class 0: {votes_0} votes | Class 1: {votes_1} votes")
    else:
        final_prediction = sum(total_received_votes) / len(total_received_votes)
        task_str = "Regression (Mean)"

    total_run_time = time.time() - total_start_time

    print("\n" + "=" * 60)
    print(f" REAL-TIME PREDICTION ({task_str}): {final_prediction:.2f}")
    print("-" * 60)
    print(f" AWS Provisioning Time (Cold Start):   {provisioning_time:.2f}s")
    print(f" Pure Inference Time (SQS + CPU):      {pure_inference_time:.2f}s")
    print(f" TOTAL Global System Latency:          {total_run_time:.2f}s")
    print("=" * 60 + "\n")

    client_response_queue = load_config().get("sqs_queues", {}).get("client_response")
    if client_response_queue:
        try:
            response_payload = {
                "job_id": job_id, "prediction": float(final_prediction),
                "task_type": task_str, "total_time_sec": round(total_run_time, 2)
            }
            sqs_client.send_message(QueueUrl=client_response_queue, MessageBody=json.dumps(response_payload))
            print(f" [SUCCESS] Real-Time Prediction sent back to Client via SQS.")
        except Exception as e:
            print(f" [ERROR] Failed to send response to client: {e}")


def main():
    print(" Master Node initialized. Waiting for Client jobs...")

    while True:
        response = sqs_client.receive_message(QueueUrl=CLIENT_QUEUE_URL, MaxNumberOfMessages=1, WaitTimeSeconds=20)

        if 'Messages' in response:
            client_msg = response['Messages'][0]
            receipt_handle = client_msg['ReceiptHandle']
            job_data = json.loads(client_msg['Body'])

            job_id = job_data.get('job_id', client_msg['MessageId'])
            job_data['job_id'] = job_id
            mode = job_data.get('mode', 'train')

            print(f"\n" + "=" * 50)
            print(f" INITIALIZING ASYNC PIPELINE FOR JOB: {job_id}")
            print("=" * 50)

            stop_event_master = threading.Event()
            heartbeat_thread_master = threading.Thread(
                target=extend_client_sqs_visibility,
                args=(CLIENT_QUEUE_URL, receipt_handle, stop_event_master)
            )
            heartbeat_thread_master.start()

            try:
                if mode == 'train':
                    process_training(job_data, job_id)
                elif mode == 'bulk_infer':
                    process_bulk_inference(job_data, job_id)
                elif mode == 'infer':
                    process_realtime_inference(job_data, job_id)
                elif mode == 'train_and_infer':
                    process_training(job_data, job_id)
                    job_data['target_model'] = job_id
                    process_bulk_inference(job_data, job_id)
                else:
                    print(f" [WARNING] Unknown mode requested: {mode}")
            except Exception as e:
                print(f" [CRITICAL ERROR] Pipeline execution failed: {e}")
            finally:
                stop_event_master.set()
                heartbeat_thread_master.join()
                sqs_client.delete_message(QueueUrl=CLIENT_QUEUE_URL, ReceiptHandle=receipt_handle)
                print(f" JOB {job_id} PROCESSED AND REMOVED FROM QUEUE.\n")


if __name__ == "__main__":
    main()