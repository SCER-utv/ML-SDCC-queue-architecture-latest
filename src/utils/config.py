import json
import os
import boto3

_cached_config = None


# Single Source of Truth Discovery: Scans S3 interim bucket for all dataset variants (e.g. 1M, 50M, optimized)
def discover_all_datasets(s3_bucket, region, dataset_registry):
    s3 = boto3.client('s3', region_name=region)
    datasets = {}
    prefix = "data/interim/"

    try:
        resp = s3.list_objects_v2(Bucket=s3_bucket, Prefix=prefix)

        if 'Contents' not in resp:
            print(f" [DISCOVERY WARN] No objects found in bucket under prefix '{prefix}'.")
            return datasets

        for obj in resp['Contents']:
            key = obj['Key']

            # Identify CSV datasets
            if key.endswith('.csv'):
                # Extract filename (e.g., "data/interim/airlines/airlines_1M.csv" -> "airlines_1M.csv")
                filename = key.split('/')[-1]

                # We expect the format "datasetName_variantName.csv" (e.g., "airlines_1M.csv")
                # Split by the LAST underscore to separate the name from the variant
                if '_' not in filename:
                    continue

                parts = filename.rsplit('_', 1)
                dataset_name = parts[0]  # e.g., "airlines"
                variant = parts[1].replace('.csv', '')  # e.g., "1M" or "optimized"

                if dataset_name not in dataset_registry:
                    continue

                target_col = dataset_registry[dataset_name]["target"]
                task_type = dataset_registry[dataset_name]["type"]

                # Define future paths for processed data splits for THIS SPECIFIC variant
                train_key = f"data/processed/{dataset_name}/{dataset_name}_{variant}_train.csv"
                test_key = f"data/processed/{dataset_name}/{dataset_name}_{variant}_test.csv"

                # S3 Select: Read only the header row
                select_resp = s3.select_object_content(
                    Bucket=s3_bucket,
                    Key=key,
                    ExpressionType='SQL',
                    Expression='SELECT * FROM S3Object LIMIT 1',
                    InputSerialization={'CSV': {'FileHeaderInfo': 'NONE'}},
                    OutputSerialization={'CSV': {}}
                )

                header = ""
                for event in select_resp['Payload']:
                    if 'Records' in event:
                        header += event['Records']['Payload'].decode('utf-8')

                columns = [col.strip() for col in header.split(',') if col.strip()]
                if not columns or target_col not in columns:
                    print(f" [DISCOVERY WARN] Target column '{target_col}' not found in '{filename}'. Skipping.")
                    continue

                features_count = len(columns) - 1

                # Initialize the dictionary for this dataset if it doesn't exist
                if dataset_name not in datasets:
                    datasets[dataset_name] = {}

                # Populate registry for THIS VARIANT
                datasets[dataset_name][variant] = {
                    "type": task_type,
                    "target": target_col,
                    "features": features_count,
                    "interim_path": key,  # We save the original S3 path for the split!
                    "train_path": train_key,
                    "test_path": test_key
                }

                print(f" [DISCOVERY] Registered: {dataset_name} (Variant: {variant})")

    except Exception as e:
        print(f" [DISCOVERY ERROR] S3 error during scan of prefix {prefix}: {e}")

    return datasets


# Loads configuration and executes S3 auto-discovery
def load_config():
    global _cached_config
    if _cached_config is not None:
        return _cached_config

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
    config_path = os.path.join(root_dir, 'config', 'config.json')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f" [SYSTEM ERROR] Configuration file not found at: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    dataset_registry = config.get("dataset_registry", {})
    print("\n [AUTO-DISCOVERY] Scanning S3 'data/interim/' prefix for valid datasets and variants...")

    # Inject dynamically discovered nested dictionary
    config['datasets_metadata'] = discover_all_datasets(config['s3_bucket'], config['aws_region'], dataset_registry)

    config['_root_dir'] = root_dir
    _cached_config = config
    return config