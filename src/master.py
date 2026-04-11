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
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score, mean_absolute_error, precision_score, recall_score, f1_score

from src.model.model_factory import ModelFactory
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

# Counts and retrieves S3 URIs of distributed model chunks
def count_model_parts(bucket, dataset, target_model):
    s3 = boto3.client('s3', region_name=AWS_REGION)
    prefix = f"models/{dataset}/{target_model}/"
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return [f"s3://{bucket}/{obj['Key']}" for obj in resp.get('Contents', []) if obj['Key'].endswith('.joblib')]
    
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


# Scales the Auto Scaling Group to the desired number of instances and updates their tags.
def scale_worker_infrastructure(num_workers):
    print(f" [ASG] Setting desired capacity to {num_workers} workers...")
    asg_client.update_auto_scaling_group(
        AutoScalingGroupName=ASG_NAME, MinSize=0, DesiredCapacity=num_workers, MaxSize=10
    )
    
    if num_workers == 0:
        return
        
    print(f" [ASG] Waiting for instances to start for tagging...")
    ec2_client = boto3.client('ec2', region_name=AWS_REGION)
    
    max_wait = 24 # 24 * 5 sec = 120 seconds max
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
                
        # If we have reached the target, we stop waiting
        if len(found_instances) >= num_workers:
            break
            
    # At the end of the time (or if we finish earlier), we rename everything we have found
    # This handles also the case where AWS gives us fewer machines than expected
    if len(found_instances) > 0:
        if len(found_instances) < num_workers:
            print(f" [ASG WARN] Requested {num_workers} workers, but AWS provided {len(found_instances)}. Proceeding degraded.")
        else:
            print(f" [ASG] Found {len(found_instances)} instances. Applying name tags...")
            
        for i, instance_id in enumerate(found_instances):
            worker_name = f"DRF-worker{i+1}"
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


# Executes an S3 Select query to quickly count rows in a CSV without downloading it. It will be executed when the split is not done.
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


# Uncomment this 3-way split block if a fixed Test set is required for Grid Search testing
""" 
def execute_streaming_split(dataset_name):
    ratios = config.get("split_ratios", {"train": 0.70, "val": 0.15})
    train_threshold = ratios.get("train", 0.70)
    val_threshold = train_threshold + ratios.get("val", 0.15)
    
    print(f" [SPLIT] Starting 3-way dynamic streaming split for '{dataset_name}'...")
    s3 = boto3.client('s3', region_name=AWS_REGION)
    bucket = config.get("s3_bucket")
    
    source_key = f"data/interim/{dataset_name}/{dataset_name}_optimized.csv"
    train_key = f"data/processed/{dataset_name}/{dataset_name}_train.csv"
    val_key = f"data/processed/{dataset_name}/{dataset_name}_val.csv"
    test_key = f"data/processed/{dataset_name}/{dataset_name}_test.csv"
    
    local_train = f"/tmp/{dataset_name}_train.csv"
    local_val = f"/tmp/{dataset_name}_val.csv"
    local_test = f"/tmp/{dataset_name}_test.csv"
    
    try:
        response = s3.get_object(Bucket=bucket, Key=source_key)
        safe_streaming = io.TextIOWrapper(response['Body'], encoding='utf-8')
        
        with open(local_train, 'w', encoding='utf-8') as f_train, \
             open(local_val, 'w', encoding='utf-8') as f_val, \
             open(local_test, 'w', encoding='utf-8') as f_test:
            
            header = safe_streaming.readline()
            f_train.write(header)
            f_val.write(header)
            f_test.write(header)
            
            rows = {'train': 0, 'val': 0, 'test': 0}
            
            for line in safe_streaming:
                if line.strip(): 
                    r = random.random()
                    if r <= train_threshold:         
                        f_train.write(line)
                        rows['train'] += 1
                    elif r <= val_threshold:        
                        f_val.write(line)
                        rows['val'] += 1
                    else:                            
                        f_test.write(line)
                        rows['test'] += 1
                        
        print(f" [SPLIT] Finished. Train: {rows['train']} | Val: {rows['val']} | Test: {rows['test']}")
        s3.upload_file(local_train, bucket, train_key)
        s3.upload_file(local_val, bucket, val_key)
        s3.upload_file(local_test, bucket, test_key)
        
    except Exception as e:
        print(f" [SPLIT ERROR] Failed during streaming split: {e}")
        raise e
    finally:
        if os.path.exists(local_train): os.remove(local_train)
        if os.path.exists(local_val): os.remove(local_val)
        if os.path.exists(local_test): os.remove(local_test) 

    return rows['train'] 
"""


# Downloads dataset via S3 and randomly splits it into Train and Test set, then uploads them back to S3.
def execute_streaming_split(dataset_name, dataset_variant):
    config = load_config()
    ratios = config.get("split_ratios", {"train": 0.70})
    train_threshold = ratios.get("train", 0.70)

    print(f" [SPLIT] Starting dynamic streaming split for '{dataset_name}' (Variant: {dataset_variant})...")
    s3 = boto3.client('s3', region_name=AWS_REGION)
    bucket = config.get("s3_bucket")

    # Recupero dinamico dal nuovo config annidato
    metadata = config['datasets_metadata'].get(dataset_name, {}).get(dataset_variant)
    if not metadata:
        raise ValueError(f"Metadata not found for {dataset_name}_{dataset_variant}")

    source_key = metadata['interim_path']
    train_key = metadata['train_path']
    test_key = metadata['test_path']

    # Aggiungiamo un identificativo robusto (es. il variant) ai file temporanei per evitare collisioni!
    local_train = f"/tmp/{dataset_name}_{dataset_variant}_train.csv"
    local_test = f"/tmp/{dataset_name}_{dataset_variant}_test.csv"
    
    try:
        print(f" [SPLIT] Line-by-line streaming in progress...")
        response = s3.get_object(Bucket=bucket, Key=source_key)
        
        # We wrap the raw stream (Body) in a TextIOWrapper that understands UTF-8 encoding and can correctly detect where a line actually ends (\n).
        safe_streaming = io.TextIOWrapper(response['Body'], encoding='utf-8')
        
        with open(local_train, 'w', encoding='utf-8') as f_train, open(local_test, 'w', encoding='utf-8') as f_test:
            
            # Read the header cleanly
            header = safe_streaming.readline()
            # Add the header on both files
            f_train.write(header)
            f_test.write(header)
            
            # Iterate
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
    return train_rows

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
    config = load_config()
    # Get job details
    num_workers = job_data['num_workers']
    num_trees_total = job_data['num_trees']
    dataset = job_data['dataset']
    dataset_variant = job_data.get('dataset_variant', '1M')
    job_id = job_data['job_id']
    strategy = job_data.get('strategy', 'homogeneous')

    target_bucket = config.get("s3_bucket", "distributed-random-forest-bkt")

    # Retrieving metadata for dataset variant
    metadata = config['datasets_metadata'].get(dataset, {}).get(dataset_variant)
    if not metadata:
        print(f" [INFO CRITICAL] Dataset variant '{dataset_variant}' not found for '{dataset}'. Aborting.")
        return

    train_s3_key = metadata['train_path']
    train_s3_uri = f"s3://{target_bucket}/{train_s3_key}"

    # Count the number of rows: Use the provided parameter or perform an S3 query.
    if total_rows is None:
        try:
            print(" [INFO] Total rows missing in cache. Triggering S3 Select fallback...")
            total_rows = get_total_rows_s3_select(target_bucket, train_s3_key)
        except Exception:
            print(" [INFO CRITICAL] S3 Select fallback failed. Aborting.")
            return


    # Calculating single worker task data
    rows_per_worker = total_rows // num_workers
    remainder_rows = total_rows % num_workers

    trees_per_worker = math.floor(num_trees_total / num_workers)
    trees_remainder = num_trees_total % num_workers

    root_dir = config.get('_root_dir', '.')

    strategies_path = os.path.join(root_dir, 'config', 'homogeneous_tasks.json')
    if(strategy == "homogeneous"):
        strategies_path = os.path.join(root_dir, 'config', 'homogeneous_tasks.json')
    else:
        strategies_path = os.path.join(root_dir, 'config', 'heterogeneous_tasks.json')
    
    try:
        with open(strategies_path, 'r') as f:
            all_strategies = json.load(f)
    except FileNotFoundError:
        print(f" [INFO ERROR] Missing strategies file: {strategies_path}")
        all_strategies = {}

    # Determine whether the dataset is for classification or regression
    ml_handler = ModelFactory.get_model(dataset)
    task_type = getattr(ml_handler, 'task_type', 'classification')

    target_strategies = []

    if strategy == "homogeneous":
        # Look for dataset, number of trees, in the specified configuration file
        dataset_params = all_strategies.get(dataset, {})
        conf = dataset_params.get(str(num_trees_total))

        if conf:
            # Generate same strategy for all workers
            target_strategies = [conf] * num_workers
        else:
            print(f" [INFO WARNING] No Gold Standard for {dataset} with {num_trees_total} trees.")

    else:
        # Look for
        dataset_params = all_strategies.get(dataset, {})

        # Get the strategy for the specified number of workers
        target_strategies = dataset_params.get(str(num_workers), [])

    if not target_strategies:
        print(f" [INFO WARNING] Using fallback params.")
        fallback_conf = {"max_depth": None, "max_features": "sqrt",
                         "criterion": "gini" if dataset == "airlines" else "squared_error"}
        target_strategies = [fallback_conf] * num_workers


    current_skip = 0
    
    """
    # ==========================================================
    # TEST 1.3 (CRASH BEFORE SENDING train_tasks)
    # ==========================================================
    print("\n" + "!"*50)
    print(" [TEST 1.3] VULNERABILITY WINDOW OPEN")
    print(" [TEST 1.3] You have 15 seconds to kill the Master!")
    print(" [TEST 1.3] Run: sudo docker restart master-node")
    print("!"*50 + "\n")

    time.sleep(15)
    # ==========================================================
    """

    
    print(f" [INFO] Distributing {num_trees_total} trees across {num_workers} training tasks...")
    for i in range(num_workers):
        trees = trees_per_worker + (1 if i < trees_remainder else 0)
        n_rows = rows_per_worker + (remainder_rows if i == num_workers - 1 else 0)
        conf = target_strategies[i % len(target_strategies)]

        raw_depth = conf['max_depth']
        if(raw_depth == "None"):
            max_depth = None
        else:
            try:
                max_depth = int(raw_depth)
            except (ValueError, TypeError):
                max_depth = None

        raw_features = conf['max_features']
        if raw_features in ["sqrt", "log2"]:
            max_features = raw_features  
        elif raw_features in ["None", "null", None]:
            max_features = None
        else:
            try:
                val_float = float(raw_features)
                if val_float.is_integer():
                    max_features = int(val_float)
                else:
                    max_features = val_float  
            except (ValueError, TypeError):
                max_features = "sqrt"


        max_samples = conf['max_samples']
        try:
            val_float = float(max_samples)
            max_samples = val_float
        except (ValueError, TypeError):
            max_samples = "1.0"

        task_payload = {
            "job_id": job_id,
            "task_id": f"task_{i + 1}",
            "seed": i * 1000,
            "dataset": dataset,
            "dataset_s3_path": train_s3_uri,
            "trees": trees,
            "skip_rows": current_skip,
            "num_rows": n_rows,

            # Specific parameters
            "max_depth": max_depth,
            "max_features": max_features,
            "criterion": conf.get('criterion'),
            "min_samples_split": conf.get('min_samples_split', 2),
            "min_samples_leaf": conf.get('min_samples_leaf', 1),
            "max_samples": max_samples,
            "class_weight": conf.get('class_weight', None),
            "n_jobs": conf.get('n_jobs', -1)
        }

        current_skip += n_rows
        sqs_client.send_message(QueueUrl=TRAIN_TASK_QUEUE, MessageBody=json.dumps(task_payload))
        print(f" Enqueued {task_payload['task_id']} ({trees} trees).")
        
        """
        # ==========================================================
        # TEST 1.5 (CRASH MID FAN-OUT)
        # ==========================================================
        if i == (num_workers // 2) - 1:  # Pauses exactly halfway through the workers
            print("\n" + "!"*50)
            print(f" [TEST 1.5] FAN-OUT INTERRUPTED HALFWAY! Sent {i+1} out of {num_workers} tasks.")
            print(" [TEST 1.5] You have 15 seconds to kill the Master before it finishes sending")
            print("!"*50 + "\n")
            time.sleep(15)
        # ==========================================================
        """
        

# Enqueues inference tasks mapped to completed training chunks
def generate_inference_tasks(job_id, train_resp, dataset, dataset_variant):
    # Get train task details
    task_id = train_resp['task_id']
    model_s3_uri = train_resp['s3_model_uri']

    config = load_config()
    target_bucket = config.get("s3_bucket", "distributed-random-forest-bkt")
    test_s3_key = config['datasets_metadata'][dataset][dataset_variant]['test_path']
    test_s3_uri = f"s3://{target_bucket}/{test_s3_key}"

    infer_task = {
        "job_id": job_id,
        "task_id": task_id,
        "dataset": dataset,
        "dataset_variant": dataset_variant,
        "test_dataset_uri": test_s3_uri,
        "model_s3_uri": model_s3_uri  
    }
    sqs_client.send_message(QueueUrl=INFER_TASK_QUEUE, MessageBody=json.dumps(infer_task))
    print(f" [INFER DISPATCH] Task {task_id} sent to inference queue.")

# Parses S3 URI into bucket and object key
def parse_s3_uri(s3_uri):
    parts = s3_uri.replace("s3://", "").split("/", 1)
    return parts[0], parts[1]

# Appends final job metrics to a persistent S3 CSV file
def save_metrics(dataset, dataset_variant, n_workers, n_trees, strategy_name, train_time, inf_time, metrics_dict, config):
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    target_bucket = config.get("s3_bucket", "distributed-random-forest-bkt")
    s3_key = f"results/{dataset}/{dataset}_{dataset_variant}_distributed_results.csv"
    
    # 1. Create the base row dictionary with standard information
    row_data = {
        'Dataset': dataset,
        'Workers': n_workers, 
        'Trees': n_trees,
        'System_type': "Distributed",
        'Strategy': strategy_name, 
        'Train_Time': round(train_time, 2), 
        'Infer_Time': round(inf_time, 2)
    }
    
    # 2. Unpack the metrics!
    # This command takes the keys (e.g., 'RMSE', 'MAE', 'R2 Score') and 
    # automatically transforms them into distinct columns within the row.
    row_data.update(metrics_dict)
    
    new_row_df = pd.DataFrame([row_data])

    try:
        # Attempt to download the existing CSV from S3
        obj = s3_client.get_object(Bucket=target_bucket, Key=s3_key)
        
        # Instruct Pandas to read the "Excel-style" formatted file
        # (using semicolon as separator and comma for decimals)
        df_existing = pd.read_csv(io.BytesIO(obj['Body'].read()), sep=';', decimal=',')
        
        # If it exists, append the new row at the end
        df_final = pd.concat([df_existing, new_row_df], ignore_index=True)
        
    except botocore.exceptions.ClientError as e:
        # If the file does not exist on S3 yet, the final file will consist only of the current row
        if e.response['Error']['Code'] == 'NoSuchKey':
            df_final = new_row_df
        else:
            print(f" [METRICS ERROR] Unexpected S3 error: {e}")
            return
            
    # Prepare the buffer to overwrite the updated file on S3
    csv_buffer = io.StringIO()
    
    # Save the CSV maintaining perfect formatting for Excel
    df_final.to_csv(csv_buffer, index=False, sep=';', decimal=',')
    
    s3_client.put_object(Bucket=target_bucket, Key=s3_key, Body=csv_buffer.getvalue())
    print(f" [METRICS] Results securely appended to: s3://{target_bucket}/{s3_key}")


# Downloads partial inferences, executes majority vote/averaging, and calculates global metrics.
def aggregate_and_evaluate(job_id, dataset_name, dataset_variant, s3_inference_results, num_workers, trees, weights, train_time, infer_time, strategy):
    print("\n" + "=" * 50)
    print(" FINAL AGGREGATION & EVALUATION PHASE")
    print("=" * 50)
    
    """ 
    # ==========================================================
    # TEST 2.4 (MASTER CRASH DURING FINAL AGGREGATION)
    # ==========================================================
    print("\n" + "!"*50)
    print(" [TEST 2.4] CRITICAL PHASE: ALL WORKERS HAVE COMPLETED")
    print(" [TEST 2.4] You have 15 seconds to kill the Master")
    print("!"*50 + "\n")
    time.sleep(15)
    # ==========================================================
    """


    ml_handler = ModelFactory.get_model(dataset_name)
    task_type = getattr(ml_handler, 'task_type', 'classification')

    s3 = boto3.client('s3')
    config = load_config()

    # Download all the .npy files from the workers
    predictions_list = []
    print(f" Downloading {len(s3_inference_results)} inference result files from S3...")

    for task_id, s3_uri in s3_inference_results.items():
        bucket, key = parse_s3_uri(s3_uri)
        local_path = f"/tmp/res_{task_id}.npy"
        s3.download_file(bucket, key, local_path)

        result_array = np.load(local_path)
        predictions_list.append(result_array)
        os.remove(local_path)

    # Downloads the real values from the test set
    target_col = ml_handler.target_column
    test_s3_key = config['datasets_metadata'][dataset_name][dataset_variant]['test_path']
    test_s3_uri = f"s3://{config.get('s3_bucket')}/{test_s3_key}"

    print(f" Reading Ground Truth from column '{target_col}'...")
    df_test = pd.read_csv(test_s3_uri, usecols=[target_col])
    y_true = df_test[target_col].values

    # Aggregation
    data_shape = predictions_list[0].shape
    
    if len(data_shape) == 2:
        # CLASSIFICATION (The shape is N_rows * 2_columns)
        print(" [EVALUATION] Classification task detected. Executing Majority Voting...")
        total_votes = np.sum(predictions_list, axis=0)
        votes_0 = total_votes[:, 0]
        votes_1 = total_votes[:, 1]
        
        y_prob = votes_1 / (votes_0 + votes_1)
        final_prediction = np.argmax(total_votes, axis=1)

        # Calculating all evaluation metrics
        auc = roc_auc_score(y_true, y_prob)
        acc = accuracy_score(y_true, final_prediction)
        precision = precision_score(y_true, final_prediction, zero_division=0)
        recall = recall_score(y_true, final_prediction, zero_division=0)
        f1 = f1_score(y_true, final_prediction, zero_division=0)

        print(f"\n GLOBAL DISTRIBUTED RESULTS:")
        print(f" ROC-AUC:   {auc:.4f}")
        print(f" Accuracy:  {acc:.4f}")
        print(f" Precision: {precision:.4f}")
        print(f" Recall:    {recall:.4f}")
        print(f" F1-Score:  {f1:.4f}")
        
        metrics_dict = {
            'ROC-AUC': float(round(auc, 4)), 
            'Accuracy': float(round(acc, 4)),
            'Precision': float(round(precision, 4)),
            'Recall': float(round(recall, 4)),
            'F1-Score': float(round(f1, 4))
        }

    else:
        # REGRESSION (The shape is N_rows)
        print(" [EVALUATION] Regression task detected. Executing Weighted Averaging...")
        y_pred = np.average(predictions_list, axis=0, weights=weights)

        # Calculating all evaluation metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred) 

        print(f"\n GLOBAL DISTRIBUTED RESULTS:")
        print(f" RMSE: {rmse:.4f}")
        print(f" MAE: {mae:.4f}") 
        print(f" R2 Score: {r2:.4f}")

        metrics_dict = {
            'RMSE': float(round(rmse, 4)), 
            'MAE': float(round(mae, 4)), 
            'R2 Score': float(round(r2, 4))
        }

    print("=" * 50 + "\n")

    strategy_name = "Homogeneous" if strategy == "homogeneous" else "Heterogeneous"

    # Save on S3
    save_metrics(
        dataset=dataset_name,
        dataset_variant=dataset_variant,
        n_workers=num_workers, 
        n_trees=trees, 
        strategy_name=strategy_name,
        train_time=train_time, 
        inf_time=infer_time, 
        metrics_dict=metrics_dict, 
        config=config
    )
    
    # 5. S3 Garbage Collection
    cleanup_s3_inference_files(s3_inference_results)


# Retrieves current job status from DynamoDB for fault tolerance recovery
def get_job_state(job_id):
    config = load_config() 
    table_name = config.get("dynamodb_table", "JobStatus")
    dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
    table = dynamodb.Table(table_name)
    try:
        response = table.get_item(Key={'job_id': job_id})
        if 'Item' in response:
            start_time = float(response['Item'].get('start_time'))
            tasks_dispatched = response['Item'].get('tasks_dispatched', False)
            
            # Note: DynamoDB Schema keys remain mapped to Italian labels to preserve backward compatibility
            training_time = float(response['Item'].get('tempo_training', 0.0))
            inference_time = float(response['Item'].get('tempo_inferenza', 0.0))
            
            return (set(response['Item'].get('completed_train', [])), 
                    response['Item'].get('completed_infer', {}), 
                    start_time, tasks_dispatched, training_time, inference_time)
    except Exception:
        pass
    return set(), {}, None, False, 0.0, 0.0


# Upserts current job progress into DynamoDB
def update_job_state(job_id, completed_train_set, completed_infer_dict, start_time, tasks_dispatched, training_time=0.0, inference_time=0.0):
    config = load_config() 
    table_name = config.get("dynamodb_table", "JobStatus")
    dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
    table = dynamodb.Table(table_name)

    # Note: DynamoDB Schema keys remain mapped to Italian labels to preserve backward compatibility
    table.put_item(Item={
        'job_id': job_id,
        'completed_train': list(completed_train_set),
        'completed_infer': completed_infer_dict,
        'start_time': str(start_time),
        'tasks_dispatched': tasks_dispatched,
        'tempo_training': str(training_time),
        'tempo_inferenza': str(inference_time)
    })

# Delete temporary .npy files from S3 after aggregation
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
    
# master event loop
def main():
    print(" Master Node initialized. Waiting for Client jobs...")

    while True:
        # 1. Wait the message from the client
        response = sqs_client.receive_message(QueueUrl=CLIENT_QUEUE_URL, MaxNumberOfMessages=1, WaitTimeSeconds=20)

        if 'Messages' in response:
            client_msg = response['Messages'][0]
            receipt_handle = client_msg['ReceiptHandle']
            job_data = json.loads(client_msg['Body'])

            # 2. We extract the descriptive Job ID created by the client (e.g., job_taxi_200trees_...)
            # We use the SQS ID only as a ‘fallback’ if, for some reason, it is missing from the JSON
            job_id = job_data.get('job_id', client_msg['MessageId'])
            job_data['job_id'] = job_id
            dataset = job_data['dataset']
            mode = job_data.get('mode', 'train')

            print(f"\n" + "=" * 50)
            print(f" INITIALIZING ASYNC PIPELINE FOR JOB: {job_id}")
            print("=" * 50)

            # 3. Start heartbeat thread
            stop_event_master = threading.Event()
            heartbeat_thread_master = threading.Thread(
                target=extend_client_sqs_visibility, 
                args=(CLIENT_QUEUE_URL, receipt_handle, stop_event_master)
            )
            heartbeat_thread_master.start()

            # 4. Execution of the task
            try:
                # BRANCH A: DISTRIBUTED TRAINING & BULK INFERENCE
                if mode == 'train':
                    num_workers = job_data['num_workers']
        
                    # 1. State recovery
                    completed_train_tasks, s3_inference_results, db_train_start_time, tasks_dispatched, training_time, inference_time = get_job_state(job_id)
                    
                    if db_train_start_time is None:
                        # NEW JOB: Set the timer in the database to protect against pre–fan-out crashes
                        total_start_time = time.time() 
                        start_train = time.time()
                        update_job_state(job_id, completed_train_tasks, s3_inference_results, start_train, False, training_time, inference_time)    
                    else:
                        # RECOVERY: Restore timers from the database
                        total_start_time = db_train_start_time
                        start_train = db_train_start_time
                        print(f" [RECOVERY] Global timers restored from Database.")
                        print(f" [RECOVERY] Current state: {len(completed_train_tasks)} Train and {len(s3_inference_results)} Infer tasks complete.")
        
                    # 2. PROVISIONING (Idempotent)
                    #scommenta delay per test veloci, è usato per spegnere bene le macchine e poi partire con le code, per evitare problemi
                    scale_worker_infrastructure(num_workers)
                    time.sleep(10)


                    # 3. Dynamic Data Split
                    calculated_train_rows = None
                    dataset_variant = job_data.get('dataset_variant', '1M')

                    if not tasks_dispatched:
                        bucket = config.get("s3_bucket")
                        metadata = config['datasets_metadata'].get(dataset, {}).get(dataset_variant)

                        if not metadata:
                            print(f" [CRITICAL] Dataset variant '{dataset_variant}' not found. Aborting.")
                            scale_worker_infrastructure(0)
                            continue

                        train_key = metadata['train_path']

                        split_already_exists = check_s3_file_exists(bucket, train_key)

                        if not split_already_exists:
                            try:
                                print(f" [PIPELINE] Dataset split required (Exists: {split_already_exists}. Streaming...")
                                calculated_train_rows = execute_streaming_split(dataset, dataset_variant)
                            except Exception as e:
                                print(f" [CRITICAL] Split execution failed: {e}")
                                scale_worker_infrastructure(0)
                                continue
                        else:
                            print(
                                " [PIPELINE] Dataset split already exists. Bypassing split to ensure test set consistency.")
                    else:
                        print(" [RECOVERY] Split block skipped. Tasks already dispatched.")

                    # 4. SQS FAN-OUT
                    if not tasks_dispatched:
                        generate_initial_training_tasks(job_data, total_rows=calculated_train_rows) 
                        tasks_dispatched = True
                        update_job_state(job_id, completed_train_tasks, s3_inference_results, start_train, tasks_dispatched, training_time, inference_time)
                    else:
                        print(" [RECOVERY] SQS Fan-Out skipped to prevent duplicates.")
        
                    start_infer = start_train # Fallback timestamp

                    # 5. POLLING EVENT LOOP
                    print("\n [EVENT LOOP] Master listening actively for Worker responses...\n")
                    while len(s3_inference_results) < num_workers:
                        
                        # Train response queue
                        res_train = sqs_client.receive_message(QueueUrl=TRAIN_RESPONSE_QUEUE, MaxNumberOfMessages=10, WaitTimeSeconds=2)
                        if 'Messages' in res_train:
                            for msg in res_train['Messages']:
                                train_resp = json.loads(msg['Body'])
                                task_id = train_resp['task_id']
        
                                if task_id not in completed_train_tasks:
                                    generate_inference_tasks(job_id, train_resp, dataset, dataset_variant)
                                    completed_train_tasks.add(task_id)
                                    print(f" [ACK] Worker completed training for {task_id}.")
                                    update_job_state(job_id, completed_train_tasks, s3_inference_results, start_train, tasks_dispatched, training_time, inference_time)
        
                                sqs_client.delete_message(QueueUrl=TRAIN_RESPONSE_QUEUE, ReceiptHandle=msg['ReceiptHandle'])
                        
                        # Timer switch: Train to Inference
                        if len(completed_train_tasks) == num_workers and training_time == 0.0:
                            training_time = time.time() - start_train
                            start_infer = time.time() 
                            update_job_state(job_id, completed_train_tasks, s3_inference_results, start_train, tasks_dispatched, training_time, inference_time)
        
                        # Inference response queue
                        res_infer = sqs_client.receive_message(QueueUrl=INFER_RESPONSE_QUEUE, MaxNumberOfMessages=10, WaitTimeSeconds=2)
                        if 'Messages' in res_infer:
                            for msg in res_infer['Messages']:
                            
                                """
                                # ==========================================================
                                # TEST 2.3 (MASTER CRASH ON INFERENCE ACK)
                                # ==========================================================
                                print("\n" + "!"*50)
                                print(" [TEST 2.3] INFERENCE ACK RETRIEVED FROM THE QUEUE")
                                print(" [TEST 2.3] You have 15 seconds to kill the Master before it deletes the message")
                                print("!"*50 + "\n")
                                time.sleep(15)
                                # ========================================================== 
                                """

                                body = json.loads(msg['Body'])
                                task_id = body['task_id']
                                
                                s3_votes_data = body['s3_voti_uri']
                                s3_votes_uri = s3_votes_data['valore'] if isinstance(s3_votes_data, dict) else s3_votes_data
                                
                                if task_id not in s3_inference_results:
                                    s3_inference_results[task_id] = s3_votes_uri
                                    print(f" [ACK] Worker completed inference for {task_id}! ({len(s3_inference_results)}/{num_workers})")

                                    # Halt inference timer if phase complete
                                    if len(s3_inference_results) == num_workers and inference_time == 0.0:
                                        inference_time = time.time() - start_infer

                                    update_job_state(job_id, completed_train_tasks, s3_inference_results, start_train, tasks_dispatched, training_time, inference_time)
        
                                sqs_client.delete_message(QueueUrl=INFER_RESPONSE_QUEUE, ReceiptHandle=msg['ReceiptHandle'])
        
                    # 6. FINAL AGGREGATION
                    if training_time == 0.0:
                        training_time = time.time() - start_train
                    if inference_time == 0.0:
                        inference_time = time.time() - total_start_time
                        
                    print("\n [PIPELINE] All Workers completed their end-to-end tasks!")
                    #commenta riga sotto per test veloci
                    #scale_worker_infrastructure(0)
                    
                    total_run_time = time.time() - total_start_time
        
                    try:
                        weights = []
                        trees_per_worker = math.floor(job_data['num_trees'] / num_workers)
                        trees_remainder = job_data['num_trees'] % num_workers
                        for i in range(num_workers):
                            weights.append(trees_per_worker + (1 if i < trees_remainder else 0))

                        aggregate_and_evaluate(job_id, dataset, dataset_variant, s3_inference_results, num_workers, job_data['num_trees'], weights, training_time, inference_time, job_data.get('strategy', 'homogeneous'))
                    except Exception as e:
                        print(f" [EVALUATION ERROR] Final aggregation failed: {e}")
                        
                    print(f" [TIMERS] Train: {training_time:.2f}s | Infer: {inference_time:.2f}s | Global: {total_run_time:.2f}s")


                # BRANCH B: REAL-TIME SINGLE INFERENCE 
                elif mode == 'infer':

                    # Instead of starting the timer now, use the Client's clock. Use time.time() as a safe fallback.
                    total_start_time = job_data.get('client_start_time', time.time())

                    target_model = job_data['target_model']
                    tuple_data = job_data['tuple_data']
                    
                    bucket = load_config().get("s3_bucket")
                    model_s3_uris = count_model_parts(bucket, dataset, target_model)
                    num_workers = len(model_s3_uris)
                    
                    print(f" [REAL-TIME] Target model '{target_model}' chunked into {num_workers} parts. Scaling workers...")
                    
                    # TIMER 1: AWS cold start
                    start_provisioning = time.time()
                    scale_worker_infrastructure(num_workers)
                    provisioning_time = time.time() - start_provisioning
                    
                    # TIMER 2: Inference
                    inference_pure_start = time.time()
                    for i, uri in enumerate(model_s3_uris):
                        task_id = f"task_infer_rt_{i+1}"
                        infer_task = {
                            "job_id": job_id, "task_id": task_id, "dataset": dataset,
                            "model_s3_uri": uri, "tuple_data": tuple_data
                        }
                        sqs_client.send_message(QueueUrl=INFER_TASK_QUEUE, MessageBody=json.dumps(infer_task))

                    # ==========================================================
                    # START INJECTION FOR TEST 3.3 (MASTER CRASH IN INFERENCE REAL-TIME)
                    # ==========================================================
                    print("\n" + "!"*50)
                    print(" [TEST 3.2] TASKS SENT TO WORKERS. WAITING FOR RESPONSES...")
                    print(" [TEST 3.2] You have 15 seconds to restart the Master!")
                    print("!"*50 + "\n")
                    time.sleep(15)
                    # ==========================================================

                    total_received_votes = []
                    read_messages = 0
                    
                    while read_messages < num_workers:
                        res = sqs_client.receive_message(QueueUrl=INFER_RESPONSE_QUEUE, WaitTimeSeconds=2)
                        if 'Messages' in res:
                            for msg in res['Messages']:
                                body = json.loads(msg['Body'])
                                res_data = body['s3_voti_uri']
                                
                                # Validate correct format for Real-Time Single prediction vs Bulk
                                if isinstance(res_data, dict) and res_data.get("tipo") == "singolo":
                                    worker_predictions = res_data['valore']
                                    total_received_votes.extend(worker_predictions)
                                    read_messages += 1

                                    print(f"   -> Gathered {len(worker_predictions)} votes from worker.")
                                    
                                sqs_client.delete_message(QueueUrl=INFER_RESPONSE_QUEUE, ReceiptHandle=msg['ReceiptHandle'])
                                
                    pure_inference_time = time.time() - inference_pure_start

                    # Comment this line for rapid test 
                    # scale_worker_infrastructure(0)
                    
                    # Real-Time Aggregation
                    ml_handler = ModelFactory.get_model(dataset)
                    if ml_handler.task_type == 'classification':
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
                
                    # REQUEST-REPLY: Send back the reply to the client
                    client_response_queue = config["sqs_queues"].get("client_response")
                    if client_response_queue:
                        try:
                            response_payload = {
                                "job_id": job_id,
                                "prediction": float(final_prediction),
                                "task_type": task_str,
                                "total_time_sec": round(total_run_time, 2)
                            }
                            sqs_client.send_message(
                                QueueUrl=client_response_queue,
                                MessageBody=json.dumps(response_payload)
                            )
                            print(f" [SUCCESS] Real-Time Prediction sent back to Client via SQS.")
                        except Exception as e:
                            print(f" [ERROR] Failed to send response to client: {e}")
                    
            finally:
                # STOP MASTER HEARTBEAT 
                stop_event_master.set()
                heartbeat_thread_master.join()
                
            sqs_client.delete_message(QueueUrl=CLIENT_QUEUE_URL, ReceiptHandle=receipt_handle)
            print(f" JOB {job_id} COMPLETED SUCCESSFULLY.\n")

if __name__ == "__main__":
    main()
