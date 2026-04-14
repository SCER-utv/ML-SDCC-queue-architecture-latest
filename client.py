import sys
import json
import os
import joblib
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


def list_available_models(s3_client, bucket, dataset):
    prefix = f"models/{dataset}/"
    resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')
    models = []
    if 'CommonPrefixes' in resp:
        for obj in resp['CommonPrefixes']:
            folder_name = obj['Prefix'].replace(prefix, '').strip('/')
            models.append(folder_name)
    return models


def get_feature_names_from_s3(s3_client, bucket, s3_key, target_column="Label"):
    try:
        response = s3_client.get_object(Bucket=bucket, Key=s3_key)
        first_line = next(response['Body'].iter_lines()).decode('utf-8')
        response['Body'].close()

        all_columns = [col.strip() for col in first_line.split(',')]
        if target_column in all_columns:
            all_columns.remove(target_column)

        return all_columns
    except Exception as e:
        print(f" [WARNING] Unable to read header from S3: {e}")
        return []


def download_and_merge_model(s3_client, bucket, dataset, job_id):
    print(f"\n" + "-" * 40)
    print(f" [DOWNLOAD] Fetching model chunks for {job_id}...")
    prefix = f"models/{dataset}/{job_id}/"
    resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

    if 'Contents' not in resp:
        print(" [ERROR] No chunks found for this model on S3.")
        return

    os.makedirs("tmp_downloads", exist_ok=True)
    downloaded_files = []

    for obj in resp['Contents']:
        if obj['Key'].endswith('.joblib'):
            file_name = obj['Key'].split('/')[-1]
            local_path = os.path.join("tmp_downloads", file_name)
            s3_client.download_file(bucket, obj['Key'], local_path)
            downloaded_files.append(local_path)
            print(f"   -> Downloaded {file_name}")

    if not downloaded_files:
        print(" [ERROR] No .joblib files found.")
        return

    print("\n [MERGE] Aggregating distributed trees into a single Scikit-Learn Model...")
    base_model = None

    for file in downloaded_files:
        rf_chunk = joblib.load(file)
        if base_model is None:
            base_model = rf_chunk
        else:
            base_model.estimators_.extend(rf_chunk.estimators_)
            base_model.n_estimators += len(rf_chunk.estimators_)
        os.remove(file)

    try:
        os.rmdir("tmp_downloads")
    except:
        pass

    output_filename = f"{job_id}_aggregated.pkl"
    joblib.dump(base_model, output_filename)

    print("\n" + "=" * 60)
    print(f" SUCCESS! Model aggregated and saved locally.")
    print(f" File Name  : {output_filename}")
    print(f" Total Trees: {base_model.n_estimators}")
    print("=" * 60 + "\n")


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
    print("  1)  Distributed Training (Training Only)")
    print("  2)  End-to-End Pipeline (Train + Auto-Evaluate)")
    print("  3)  Bulk Inference (Test Set Evaluation)")
    print("  4)  Real-Time Inference (Single Prediction)")
    print("  5)  Download Aggregated Model")

    while True:
        mode_choice = input("\n Enter 1, 2, 3, 4 or 5: ").strip()
        if mode_choice in ['1', '2', '3', '4', '5']:
            if mode_choice == '1':
                mode = 'train'
            elif mode_choice == '2':
                mode = 'train_and_infer'
            elif mode_choice == '3':
                mode = 'bulk_infer'
            elif mode_choice == '4':
                mode = 'infer'
            else:
                mode = 'download'
            break
        print(" Invalid choice. Please try again.")

    # ==========================================
    # DATASET SELECTION (Menu vs Custom URL)
    # ==========================================
    print("\n" + "-" * 40)
    print(" Select Dataset Source:")
    print("  1) Choose from predefined list (Golden Standard)")
    print("  2) Provide a Custom S3 URL")

    while True:
        data_source_choice = input("\n Enter 1 or 2: ").strip()
        if data_source_choice in ['1', '2']:
            break
        print(" Invalid choice. Please try again.")

    custom_s3_url = None
    custom_task_type = None
    custom_target_col = None
    needs_split = True

    if data_source_choice == '1':
        # --- PREDEFINED LIST MENU ---
        available_datasets = list(DATASETS_METADATA.keys())
        if not available_datasets:
            print(" [ERROR] No datasets found in config metadata!")
            sys.exit(1)

        dataset_map = {}
        for i, ds_name in enumerate(available_datasets, start=1):
            first_variant = list(DATASETS_METADATA[ds_name].keys())[0]
            ds_type = DATASETS_METADATA[ds_name][first_variant]["type"]
            print(f"  {i}) {ds_name.capitalize()} ({ds_type.capitalize()})")
            dataset_map[str(i)] = ds_name

        while True:
            ds_choice = input(f"\n Select dataset [1-{len(available_datasets)}]: ").strip()
            if ds_choice in dataset_map:
                dataset = dataset_map[ds_choice]
                break
            print(" Invalid dataset selection.")

        print(f"\n Select Dataset Variant for '{dataset.upper()}':")
        available_variants = list(DATASETS_METADATA[dataset].keys())
        variant_map = {}
        for i, variant_name in enumerate(available_variants, start=1):
            print(f"  {i}) {variant_name}")
            variant_map[str(i)] = variant_name

        while True:
            var_choice = input(f"\n Select variant [1-{len(available_variants)}]: ").strip()
            if var_choice in variant_map:
                dataset_variant = variant_map[var_choice]
                break
            print(" Invalid variant selection.")

    else:
        # --- CUSTOM S3 URL MENU ---
        print("\n [CUSTOM DATASET]")
        dataset = "custom"
        dataset_variant = "user_provided"

        while True:
            if mode == 'bulk_infer':
                custom_s3_url = input(
                    " Enter the full S3 URL of the TEST dataset (e.g., s3://my-bucket/test_data.csv): ").strip()
            else:
                custom_s3_url = input(" Enter the full S3 URL of the dataset (e.g., s3://my-bucket/data.csv): ").strip()

            if custom_s3_url.startswith("s3://") and custom_s3_url.endswith(".csv"):
                break
            print(" Invalid format. Must start with 's3://' and end with '.csv'.")

        if mode in ['train', 'train_and_infer']:
            print("\n How should the system handle this dataset?")
            print("  1) It's a FULL dataset -> Auto-Split into Train and Test")
            print("  2) It's already a TRAINING set -> Do not split (Use as is)")
            while True:
                split_choice = input(" Enter 1 or 2: ").strip()
                if split_choice in ['1', '2']:
                    needs_split = (split_choice == '1')
                    break
                print(" Invalid choice.")

        custom_target_col = input(" Enter the EXACT name of the Target Column to predict (e.g., Label): ").strip()

        print("\n Specify the ML Task Type for this dataset:")
        print("  1) Classification")
        print("  2) Regression")
        while True:
            task_choice = input(" Enter 1 or 2: ").strip()
            if task_choice in ['1', '2']:
                custom_task_type = "classification" if task_choice == '1' else "regression"
                break
            print(" Invalid choice.")

    # Inizializziamo le variabili per il payload
    workers = 0
    trees = 0
    strategy_type = 'homogeneous'
    custom_hyperparams = None

    if mode in ['train', 'train_and_infer']:
        print("\n" + "-" * 40)
        print(f"  Cluster Configuration for: {dataset.upper()}({dataset_variant})")

        while True:
            try:
                workers = int(input(" Enter number of Workers (e.g., 4): "))
                trees = int(input(" Enter TOTAL number of Trees (e.g., 100): "))
                if workers > 0 and trees > 0:
                    break
                print(" Values must be greater than zero.")
            except ValueError:
                print(" Invalid input. Please enter integers only.")

        print("\n Select Training Strategy:")
        print("  1) Homogeneous  [Same parameters for all workers]")
        print("  2) Heterogeneous [Different parameters per worker, variance boosting]")
        while True:
            strat_choice = input(" Enter 1 or 2: ").strip()
            if strat_choice in ['1', '2']:
                strategy_type = "homogeneous" if strat_choice == '1' else "heterogeneous"
                break
            print(" Invalid choice.")

        print("\n Select Hyperparameter Source:")
        if dataset == "custom":
            print("  1) Default Generic Parameters (Standard Scikit-Learn)")
            print("  2) Manual Configuration")
        else:
            print("  1) Golden Standard (Auto-optimized per dataset)")
            print("  2) Manual Configuration")

        while True:
            hyper_source = input(" Enter 1 or 2: ").strip()
            if hyper_source in ['1', '2']:
                break
            print(" Invalid choice.")

        custom_hyperparams = None

        if hyper_source == '2':
            print("\n [MANUAL HYPERPARAMETERS CONFIGURATION]")
            custom_hyperparams = []
            iterations = 1 if strategy_type == "homogeneous" else workers

            for w in range(iterations):
                if strategy_type == "heterogeneous":
                    print(f"\n --- Configuring Worker {w + 1}/{workers} ---")
                else:
                    print("\n --- Configuring Global Parameters ---")

                raw_depth = input(" Max Depth (int, or blank for None): ").strip()
                max_depth = int(raw_depth) if raw_depth.isdigit() else "None"

                raw_split = input(" Min Samples Split (int, default: 2): ").strip()
                min_samples_split = int(raw_split) if raw_split.isdigit() else 2

                raw_leaf = input(" Min Samples Leaf (int, default: 1): ").strip()
                min_samples_leaf = int(raw_leaf) if raw_leaf.isdigit() else 1

                raw_features = input(" Max Features ['sqrt', 'log2', or float < 1.0] (Default: sqrt): ").strip()
                if not raw_features:
                    max_features = "sqrt"
                elif raw_features in ["sqrt", "log2", "None"]:
                    max_features = raw_features
                else:
                    try:
                        max_features = str(float(raw_features))
                    except ValueError:
                        max_features = "sqrt"

                raw_samples = input(" Max Samples per Tree [0.1 - 1.0] (Default: 1.0): ").strip()
                try:
                    max_samples = str(float(raw_samples)) if raw_samples else "1.0"
                except ValueError:
                    max_samples = "1.0"

                criterion = input(" Criterion [gini, entropy, squared_error] (Leave blank for default): ").strip()
                if not criterion:
                    t_type = custom_task_type if dataset == "custom" else \
                    DATASETS_METADATA[dataset][list(DATASETS_METADATA[dataset].keys())[0]]["type"]
                    criterion = "gini" if "classification" in str(t_type).lower() else "squared_error"

                class_weight = None
                t_type = custom_task_type if dataset == "custom" else \
                DATASETS_METADATA[dataset][list(DATASETS_METADATA[dataset].keys())[0]]["type"]
                if "classification" in str(t_type).lower():
                    raw_cw = input(" Class Weight [balanced, balanced_subsample] (Leave blank for None): ").strip()
                    if raw_cw in ["balanced", "balanced_subsample"]:
                        class_weight = raw_cw

                worker_params = {
                    "max_depth": max_depth, "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf, "max_features": max_features,
                    "max_samples": max_samples, "criterion": criterion,
                    "class_weight": class_weight, "n_jobs": -1
                }
                custom_hyperparams.append(worker_params)

            if strategy_type == "homogeneous":
                custom_hyperparams = custom_hyperparams * workers

    # ==========================================
    # MODEL SELECTION (Per Inferenza o Download)
    # ==========================================
    target_model = None
    if mode in ['infer', 'bulk_infer', 'download']:
        print("\n" + "-" * 40)
        print(" Select Target Model ID:")
        print("  1) Paste a specific Model ID")
        print(f"  2) Scan S3 to select a model for '{dataset}' ({dataset_variant})")

        while True:
            sel_method = input("\n Enter 1 or 2: ").strip()
            if sel_method in ['1', '2']:
                break
            print(" Invalid choice.")

        if sel_method == '1':
            while True:
                target_model = input("\n Paste the exact Model ID (e.g., job_taxi_...): ").strip()
                if target_model.startswith("job_") or target_model.startswith("rf_"):
                    break
                print(" Invalid ID format. It should start with 'job_'")
        else:
            print(f"\n [SEARCH] Scanning S3 for saved '{dataset}' models...")
            all_models = list_available_models(s3_client, S3_BUCKET, dataset)
            models = [m for m in all_models if f"_{dataset_variant}_" in m]

            if not models:
                print(
                    f"\n [ERROR] No trained models found for '{dataset}' (Variant: {dataset_variant}). Run a training job first!")
                sys.exit(0)

            print("\n=== AVAILABLE MODELS ===")
            for i, m in enumerate(models):
                try:
                    parts = m.split('_')
                    trees_count = next((p.replace('trees', '') for p in parts if 'trees' in p), "?")
                    workers_count = next((p.replace('workers', '') for p in parts if 'workers' in p), "?")
                    strat_label = "HOMO" if "homogeneous" in m else ("HETE" if "heterogeneous" in m else "N/A ")

                    if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
                        raw_date, raw_time = parts[-2], parts[-1]
                        if len(raw_date) == 8 and len(raw_time) == 6:
                            date_formatted = f"{raw_date[6:8]}/{raw_date[4:6]}/{raw_date[0:4]}"
                            time_formatted = f"{raw_time[0:2]}:{raw_time[2:4]}:{raw_time[4:6]}"
                        else:
                            date_formatted, time_formatted = raw_date, raw_time
                    else:
                        date_formatted, time_formatted = "????/??/??", "??:??:??"
                    print(
                        f"  [{i}]  Trees: {trees_count:<4} | Workers: {workers_count:<2} | Strat: {strat_label} | Date: {date_formatted} {time_formatted}  (ID: {m})")
                except Exception:
                    print(f"  [{i}] {m}")

            while True:
                try:
                    model_choice = int(input(f"\n Select Model ID [0-{len(models) - 1}]: "))
                    if 0 <= model_choice < len(models):
                        target_model = models[model_choice]
                        break
                    print(" Invalid ID selected.")
                except ValueError:
                    print(" Please enter a valid number.")

        if mode == 'download':
            download_and_merge_model(s3_client, S3_BUCKET, dataset, target_model)
            sys.exit(0)

    # ==========================================
    # DATA-AGNOSTIC PAYLOAD TRANSLATOR
    # ==========================================
    tuple_data = None
    final_train_url = None
    final_test_url = None
    final_target_col = None
    final_task_type = None

    if dataset == "custom":
        final_train_url = custom_s3_url if mode != 'bulk_infer' else None

        # Deduciamo l'URL di Test
        if mode == 'bulk_infer':
            final_test_url = custom_s3_url
        else:
            original_filename = custom_s3_url.split('/')[-1].replace('.csv', '')
            final_test_url = f"s3://{S3_BUCKET}/splits/{original_filename}_test.csv"
            # Fallback se non c'era split
            try:
                s3_client.head_object(Bucket=S3_BUCKET, Key=f"splits/{original_filename}_test.csv")
            except Exception:
                final_test_url = custom_s3_url

        dataset_s3_key = final_test_url.replace(f"s3://{S3_BUCKET}/", "")
        final_target_col = custom_target_col
        final_task_type = custom_task_type

    else:
        metadata = DATASETS_METADATA[dataset][dataset_variant]
        train_key = metadata['interim_path'] if needs_split else metadata['train_path']

        final_train_url = f"s3://{S3_BUCKET}/{train_key}" if mode != 'bulk_infer' else None
        final_test_url = f"s3://{S3_BUCKET}/{metadata['test_path']}"
        dataset_s3_key = metadata['test_path']

        final_target_col = metadata.get("target_column", "Label")
        final_task_type = metadata.get("type", "classification")

    # ==========================================
    # INPUT INFERENZA SINGOLA
    # ==========================================
    if mode == 'infer':
        feature_names = get_feature_names_from_s3(s3_client, S3_BUCKET, dataset_s3_key, target_column=final_target_col)
        required_features = len(feature_names) if feature_names else (
            DATASETS_METADATA.get(dataset, {}).get(dataset_variant, {}).get("features", 0))

        print("\n" + "-" * 40)
        print(" Real-Time Prediction Input")
        if required_features > 0:
            print(f" WARNING: Dataset expects EXACTLY {required_features} features!")
        if feature_names:
            print(f"\n Expected layout: \n {', '.join(feature_names)}")

        while True:
            prompt_text = f" Enter {required_features} comma-separated values: " if required_features > 0 else " Enter the comma-separated values: "
            raw_tuple = input(prompt_text).strip()
            try:
                tuple_data = [float(x.strip()) for x in raw_tuple.split(',')]
                if required_features == 0 or len(tuple_data) == required_features:
                    break
                print(f" [ERROR] Expected {required_features} values, got {len(tuple_data)}.")
            except ValueError:
                print(" [ERROR] Formatting error. Use numbers only (e.g., 10.5, 3).")

    elif mode == 'bulk_infer':
        print("\n" + "-" * 40)
        print(f" Preparing Bulk Inference for model: {target_model}")
        print(f" Target Test Dataset: {final_test_url}")

    # ==========================================
    # SQS DISPATCH UNIFICATO
    # ==========================================
    req_id = f"req_{dataset}_{dataset_variant}_{int(datetime.now().timestamp())}"
    job_id = f"job_{dataset}_{dataset_variant}_{trees}trees_{workers}workers_{strategy_type}_{int(time.time())}" if mode in [
        'train', 'train_and_infer'] else req_id

    payload = {
        "mode": mode,
        "job_id": job_id,
        "dataset": dataset,
        "dataset_variant": dataset_variant,

        "train_s3_url": final_train_url,
        "test_s3_url": final_test_url,
        "target_column": final_target_col,
        "task_type": final_task_type,
        "needs_split": needs_split,

        "num_workers": workers,
        "num_trees": trees,
        "strategy": strategy_type,
        "custom_hyperparams": custom_hyperparams,

        "target_model": target_model,
        "tuple_data": tuple_data,
        "client_start_time": time.time()
    }

    print("\n" + "=" * 60)
    print(" Dispatching request to Master Node...")

    try:
        sqs_client.send_message(
            QueueUrl=CLIENT_QUEUE_URL, MessageBody=json.dumps(payload),
            MessageGroupId="ML_Jobs", MessageDeduplicationId=payload['job_id']
        )
        print(f" [SUCCESS] Message enqueued successfully.")
        print(f" [INFO] Generated Job ID: {payload['job_id']}")

        print(f"\n [WAIT] Waiting for response from the cluster...")
        print(" (If machines are cold-starting or training, this may take some time)")

        client_resp_queue = config["sqs_queues"].get("client_response")
        start_wait = time.time()
        result_found = False

        while time.time() - start_wait < 900:
            res = sqs_client.receive_message(QueueUrl=client_resp_queue, MaxNumberOfMessages=1, WaitTimeSeconds=20)
            if 'Messages' in res:
                for msg in res['Messages']:
                    body = json.loads(msg['Body'])
                    receipt = msg['ReceiptHandle']

                    if body.get("job_id") == payload['job_id']:
                        print("\n" + "=" * 60)
                        if payload['mode'] == 'train':
                            print(" DISTRIBUTED TRAINING COMPLETED!")
                            print("=" * 60)
                            print(f" YOUR MODEL ID IS: >>> {body.get('target_model')} <<<")
                        elif payload['mode'] == 'train_and_infer':
                            print(" END-TO-END PIPELINE COMPLETED SUCCESSFULLY!")
                            print("=" * 60)
                            print(f" YOUR MODEL ID IS: >>> {body.get('target_model')} <<<")
                            print(" Metrics(RMSE/ROC-AUC) saved on S3.")
                        elif payload['mode'] == 'infer':
                            print(" REAL-TIME PREDICTION RECEIVED!")
                            print("=" * 60)
                            print(f" Task Type       : {body.get('task_type')}")
                            print(f" PREDICTION      : >>> {body.get('prediction')} <<<")
                        elif payload['mode'] == 'bulk_infer':
                            print(" BULK INFERENCE COMPLETED!")
                            print("=" * 60)
                            print(f" Metrics saved to S3. Check the results file.")

                        print(f" Cluster Latency : {body.get('total_time_sec')} seconds")
                        print("=" * 60 + "\n")

                        sqs_client.delete_message(QueueUrl=client_resp_queue, ReceiptHandle=receipt)
                        result_found = True
                        break
                    else:
                        try:
                            sqs_client.change_message_visibility(QueueUrl=client_resp_queue, ReceiptHandle=receipt,
                                                                 VisibilityTimeout=0)
                        except Exception:
                            pass
            if result_found: break

        if not result_found: print("\n [TIMEOUT] The cluster took too long to respond. Check Master logs.")

    except Exception as e:
        print(f"\n [CRITICAL ERROR] Failed to dispatch SQS message: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n [SYSTEM] Client terminated by user.")
        sys.exit(0)