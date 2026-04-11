import sys
import json
from datetime import datetime
import time

import boto3

from src.utils.config import load_config

# DYNAMIC CONFIGURATION & AUTO-DISCOVERY 
try:
    config = load_config()
except Exception as e:
    print(f" [CRITICAL] Error during Auto-Discovery from S3: {e}")
    sys.exit(1)

CLIENT_QUEUE_URL = config["sqs_queues"]["client"]
AWS_REGION = config.get("aws_region")
S3_BUCKET = config.get("s3_bucket")
DATASETS_METADATA = config.get("datasets_metadata", {})


# Scans the S3 bucket to retrieve all trained model directories available for the specified dataset.
def list_available_models(s3_client, bucket, dataset):
    prefix = f"models/{dataset}/"
    resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')
    models = []
    if 'CommonPrefixes' in resp:
        for obj in resp['CommonPrefixes']:
            folder_name = obj['Prefix'].replace(prefix, '').strip('/')
            models.append(folder_name)
    return models
    
# Scans the first row of the dataset to obtain the names of the features
def get_feature_names_from_s3(s3_client, bucket, s3_key, target_column="Label"):
    try:
        # 1. Open a connection to the file. By default, Boto3 does not download everything at once,
        # but returns a "StreamingBody" that waits to be read.
        response = s3_client.get_object(Bucket=bucket, Key=s3_key)
        
        # 2. iter_lines() reads the network stream line by line.
        # We use next() to get ONLY the very first line as soon as TCP packets arrive.
        first_line = next(response['Body'].iter_lines()).decode('utf-8')
        
        # 3. CRITICAL: Close the HTTP connection.
        # This immediately stops the download, saving bandwidth and memory.
        response['Body'].close()
        
        # 4. Format the columns
        all_columns = [col.strip() for col in first_line.split(',')]
        
        # 5. Remove the target column
        if target_column in all_columns:
            all_columns.remove(target_column)
            
        return all_columns
        
    except Exception as e:
        print(f" [WARNING] Unable to read header from S3: {e}")
        return []

# Clears the terminal screen for better UI readability
def clear_screen():
    print("\n" * 2)

def main():
    sqs_client = boto3.client('sqs', region_name=AWS_REGION)
    s3_client = boto3.client('s3', region_name=AWS_REGION)

    clear_screen()
    print("=" * 60)
    print("  DISTRIBUTED RANDOM FOREST - CLI CLIENT ")
    print("=" * 60)

    print("\nSelect Operation Mode:")
    print("  1)  Distributed Training (+ Bulk Inference Test)")
    print("  2)  Real-Time Inference (Single Prediction)")
    
    while True:
        mode_choice = input("\n Enter 1 or 2: ").strip()
        if mode_choice in ['1', '2']:
            mode = 'train' if mode_choice == '1' else 'infer'
            break
        print(" Invalid choice. Please try again.")

    # DYNAMIC DATASET MENU
    print("\n" + "-" * 40)
    print(" Select Target Dataset:")
    
    available_datasets = list(DATASETS_METADATA.keys())
    if not available_datasets:
        print(" [ERROR] No datasets found in config metadata!")
        sys.exit(1)

    dataset_map = {}

    # Dynamically list all datasets registered in config.json
    for i, ds_name in enumerate(available_datasets, start=1):
        # We get the "type" (regression/classification) from the first available variant
        first_variant = list(DATASETS_METADATA[ds_name].keys())[0]
        ds_type = DATASETS_METADATA[ds_name][first_variant]["type"]
        print(f" {i}) {ds_name.capitalize()} ({ds_type.capitalize()})")
        dataset_map[str(i)] = ds_name
        
    while True:
        ds_choice = input(f"\n Enter a number [1-{len(available_datasets)}]: ").strip()
        if ds_choice in dataset_map:
            dataset = dataset_map[ds_choice]
            break
        print(" Invalid dataset selection.")

    # ==========================================
    # DYNAMIC VARIANT SELECTION
    # ==========================================
    print("\n" + "-" * 40)
    print(f" Select Dataset Variant for '{dataset.upper()}':")

    # Retrieves all dynamically discovered variants on S3 for this dataset
    available_variants = list(DATASETS_METADATA[dataset].keys())
    variant_map = {}

    for i, variant_name in enumerate(available_variants, start=1):
        print(f" {i}) {variant_name}")
        variant_map[str(i)] = variant_name

    while True:
        var_choice = input(f"\n Enter a number [1-{len(available_variants)}]: ").strip()
        if var_choice in variant_map:
            dataset_variant = variant_map[var_choice]
            break
        print(" Invalid variant selection.")



    if mode == 'train':
        print("\n" + "-" * 40)
        print(f"  Hyperparameter Configuration for: {dataset.upper()}({dataset_variant})")
        
        while True:
            try:
                workers = int(input(" Enter number of Workers (e.g., 4): "))
                trees = int(input(" Enter TOTAL number of Trees (e.g., 100): "))
                if workers > 0 and trees > 0:
                    break
                print(" Values must be greater than zero.")
            except ValueError:
                print(" Invalid input. Please enter integers only.")

        # AGGIUNTA: Scelta della Strategia (Homogeneous vs Heterogeneous)
        print("\n Select Training Strategy:")
        print("  1) Homogeneous  [Same parameters for all workers]")
        print("  2) Heterogeneous [Different parameters per worker, variance boosting]")

        while True:
            strat_choice = input(" Enter 1 or 2: ").strip()
            if strat_choice in ['1', '2']:
                strategy_type = "homogeneous" if strat_choice == '1' else "heterogeneous"
                break
            print(" Invalid choice. Please enter 1 or 2.")

        # Generate a unique and descriptive Job ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Ex: job_taxi_1M_100trees_4workers_homogeneous_20260328_180000
        job_id = f"job_{dataset}_{dataset_variant}_{trees}trees_{workers}workers_{strategy_type}_{timestamp}"
        
        payload = {
            "mode": "train",
            "job_id": job_id,
            "dataset": dataset,
            "dataset_variant": dataset_variant,
            "num_workers": workers,
            "num_trees": trees,
            "strategy": strategy_type,
            "client_start_time": time.time()
        }

    elif mode == 'infer':
        print("\n" + "-" * 40)
        print(f" [SEARCH] Scanning S3 for saved '{dataset}' models...")
        
        all_models = list_available_models(s3_client, S3_BUCKET, dataset)
        
        models = [m for m in all_models if f"_{dataset_variant}_" in m]
        
        if not models:
            print(f"\n [ERROR] No trained models found for '{dataset}' (Variant: {dataset_variant}). Run a training job first!")
            sys.exit(0)
            
        print("\n=== AVAILABLE MODELS ===")
        for i, m in enumerate(models):
            # Dividiamo la stringa usando l'underscore per estrarre i metadati
            parts = m.split('_')
            try:
                # 1. NEW FORMAT (Includes variant and strategy)
                # Ex: job_taxi_1M_100trees_4workers_homogeneous_20260328_180000
                if ("homogeneous" in m or "heterogeneous" in m) and len(parts) >= 8:
                    var_label = parts[2]
                    trees_count = parts[3].replace('trees', '')
                    workers_count = parts[4].replace('workers', '')
                    strat_label = parts[5][:4].upper()
                    raw_date = parts[6]
                    raw_time = parts[7]

                # 2. INTERMEDIATE FORMAT (Legacy: No variant, but has strategy)
                elif "homogeneous" in m or "heterogeneous" in m:
                    var_label = "? "
                    trees_count = parts[2].replace('trees', '')
                    workers_count = parts[3].replace('workers', '')
                    strat_label = parts[4][:4].upper()
                    raw_date = parts[5]
                    raw_time = parts[6]

                # 3. OLD FORMAT (Legacy: No strategy)
                elif "workers" in m:
                    var_label = "? "
                    trees_count = parts[2].replace('trees', '')
                    workers_count = parts[3].replace('workers', '')
                    strat_label = "N/A "
                    raw_date = parts[4]
                    raw_time = parts[5]

                # 4. VERY OLD FORMAT
                else:
                    var_label = "? "
                    trees_count = parts[2].replace('trees', '')
                    workers_count = "? "
                    strat_label = "N/A "
                    raw_date = parts[3]
                    raw_time = parts[4]
                    
                # Formattazione per la stampa a schermo
                date_formatted = f"{raw_date[6:8]}/{raw_date[4:6]}/{raw_date[0:4]}"
                time_formatted = f"{raw_time[0:2]}:{raw_time[2:4]}:{raw_time[4:6]}"
                
                print(f"  [{i}]  Trees: {trees_count:<4} | Workers: {workers_count:<2} | Strat: {strat_label} | Date: {date_formatted} {time_formatted}  (ID: {m})")
                
            except Exception:
                # Se la cartella ha un nome manuale o un formato illeggibile, la stampiamo "grezza" senza far crashare il Client
                print(f"  [{i}] {m}")
        
        while True:
            try:
                model_choice = int(input(f"\n Select Model ID [0-{len(models)-1}]: "))
                if 0 <= model_choice < len(models):
                    target_model = models[model_choice]
                    break
                print(" Invalid ID selected.")
            except ValueError:
                print(" Please enter a valid number.")

        # Retrieve the file path from the config
        dataset_s3_key = DATASETS_METADATA[dataset][dataset_variant]["test_path"]

        # Dynamically retrieve feature names
        feature_names = get_feature_names_from_s3(s3_client, S3_BUCKET, dataset_s3_key, target_column="Label")
        required_features = len(feature_names) if feature_names else DATASETS_METADATA[dataset][dataset_variant]["features"]

        print("\n" + "-" * 40)
        print(" Real-Time Prediction Input")
        print(f" WARNING: The '{dataset.upper()}' ({dataset_variant}) dataset requires EXACTLY {required_features} features!")

        if feature_names:
            print(f"\n Expected layout: \n {', '.join(feature_names)}")

        
        while True:
            raw_tuple = input(f" Enter {required_features} comma-separated values: ").strip()
            try:
                tuple_data = [float(x.strip()) for x in raw_tuple.split(',')]
                
                if len(tuple_data) == required_features:
                    break
                else:
                    print(f" [ERROR] You provided {len(tuple_data)} values, but the model expects {required_features}.")
            except ValueError:
                print(" [ERROR] Formatting error. Use numbers only (e.g., 10.5, 3).")

        req_id = f"req_{dataset}_{dataset_variant}_{int(datetime.now().timestamp())}"
        
        payload = {
            "mode": "infer",
            "job_id": req_id,
            "dataset": dataset,
            "dataset_variant": dataset_variant,
            "target_model": target_model,
            "tuple_data": tuple_data
        }

    # SQS DISPATCH 
    print("\n" + "=" * 60)
    print(" Dispatching request to Master Node...")
    
    try:
        # Enqueue the JSON payload into the FIFO queue
        sqs_client.send_message(
            QueueUrl=CLIENT_QUEUE_URL,
            MessageBody=json.dumps(payload),
            MessageGroupId="ML_Jobs",
            MessageDeduplicationId=payload['job_id']
        )
        print(f" [SUCCESS] Message enqueued successfully.")
        print(f" [INFO] Generated Job ID: {payload['job_id']}")
        
        # SYNC OVER ASYNC RESPONSE HANDLING (REQUEST-REPLY)
        if payload['mode'] == 'infer':
            print("\n [WAIT] Waiting for Real-Time prediction from the cluster...")
            print(" (If machines are cold-starting, this may take 60-90 seconds)")
            
            client_resp_queue = config["sqs_queues"].get("client_response")
            if not client_resp_queue:
                print("\n [ERROR] 'client_response' queue URL not found in config.json")
                sys.exit(1)

            start_wait = time.time()
            result_found = False
            
            # Polling with a 4 minute emergency timeout 
            while time.time() - start_wait < 240:
                # Long Polling (WaitTimeSeconds=20) 
                res = sqs_client.receive_message(
                    QueueUrl=client_resp_queue, 
                    MaxNumberOfMessages=1, 
                    WaitTimeSeconds=20
                )
                
                if 'Messages' in res:
                    for msg in res['Messages']:
                        body = json.loads(msg['Body'])
                        receipt = msg['ReceiptHandle']
                        
                        # Matching to find client's message
                        if body.get("job_id") == payload['job_id']:
                            print("\n" + "=" * 60)
                            print(" DISTRIBUTED PREDICTION RECEIVED!")
                            print("=" * 60)
                            print(f" Task Type       : {body.get('task_type')}")
                            print(f" PREDICTION      : >>> {body.get('prediction')} <<<")
                            print(f" Cluster Latency : {body.get('total_time_sec')} seconds")
                            print("=" * 60 + "\n")
                            
                            # Confirm and destroy the message
                            sqs_client.delete_message(QueueUrl=client_resp_queue, ReceiptHandle=receipt)
                            result_found = True
                            break
                        else:
                            # IMMEDIATE NACK: Release the message for the other clients
                            try:
                                sqs_client.change_message_visibility(
                                    QueueUrl=client_resp_queue,
                                    ReceiptHandle=receipt,
                                    VisibilityTimeout=0
                                )
                            except Exception:
                                pass 
                
                if result_found:
                    break
            
            if not result_found:
                print("\n [TIMEOUT] The cluster took too long to respond. Check Master logs.")
                
        else:
            # Batch Training Mode: Fire and Forget
            print("\n -> Distributed Training dispatched! You can monitor the progress on S3 or Master logs.")
            print("=" * 60 + "\n")
            
    except Exception as e:
        print(f"\n [CRITICAL ERROR] Failed to dispatch SQS message: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n [SYSTEM] Client terminated by user.")
        sys.exit(0)
