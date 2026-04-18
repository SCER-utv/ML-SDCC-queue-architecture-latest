import sys
import time
from datetime import datetime

from src.utils.config import load_config
from src.aws.client_aws_manager import ClientAWSManager
from src.client.cli_prompts import CLI

# main entry point for the interactive command-line interface
def main():
    try:
        config = load_config()
    except Exception as e:
        print(f" [CRITICAL] Error loading config.json: {e}")
        sys.exit(1)

    cli = CLI(config)
    aws = ClientAWSManager(config)

    cli.show_welcome()

    # gather user intentions and operation flow through interactive prompts
    mode = cli.prompt_operation_mode()
    dataset_info = cli.prompt_dataset_selection(mode)

    experiment_name = None
    if dataset_info['is_custom'] and mode in ['train', 'train_and_infer', 'bulk_infer']:
        experiment_name = cli.prompt_experiment_name()

    cluster_config = {}
    if mode in ['train', 'train_and_infer']:
        cluster_config = cli.prompt_cluster_config(dataset_info)

    target_model = None
    if mode in ['infer', 'bulk_infer', 'download']:
        target_model = cli.prompt_model_selection(aws, dataset_info)

    if mode == 'download':
        aws.download_and_merge_model(dataset_info['name'], target_model)
        sys.exit(0)

    # handle real-time inference inputs and feature extraction
    tuple_data = None
    if mode == 'infer':
        # pass the base url to download feature headers for prompt guidance
        s3_key = ""
        if dataset_info['is_custom']:
            s3_key = dataset_info['train_url'].replace(f"s3://{aws.bucket}/", "") if dataset_info['train_url'] else ""
        else:
            s3_key = cli.datasets_metadata[dataset_info['name']][dataset_info['variant']]['train_path']

        tuple_data = cli.prompt_realtime_input(aws, s3_key, dataset_info)

    # construct the thin payload representing the client contract for the master node
    dataset = dataset_info['name']
    dataset_variant = dataset_info['variant']

    req_id = f"req_{dataset}_{dataset_variant}_{int(datetime.now().timestamp())}"
    if mode in ['train', 'train_and_infer']:
        w = cluster_config.get('workers', 0)
        t = cluster_config.get('trees', 0)
        s = cluster_config.get('strategy', 'homogeneous')
        job_id = f"job_{dataset}_{dataset_variant}_{t}trees_{w}workers_{s}_{int(time.time())}"
    else:
        job_id = req_id

    # this dictionary encapsulates the entire job configuration sent to the master
    payload = {
        "mode": mode,
        "job_id": job_id,
        "experiment_name": experiment_name,
        "is_custom": dataset_info['is_custom'],
        "dataset": dataset,
        "dataset_variant": dataset_variant,

        # include manually entered urls if present
        "custom_train_url": dataset_info.get('train_url'),
        "custom_test_url": dataset_info.get('test_url'),

        "target_column": dataset_info['target_col'],
        "task_type": dataset_info['task_type'],
        "needs_split": dataset_info['needs_split'],

        "num_workers": cluster_config.get('workers', 0),
        "num_trees": cluster_config.get('trees', 0),
        "strategy": cluster_config.get('strategy', 'homogeneous'),
        "custom_hyperparams": cluster_config.get('custom_hyperparams'),

        "target_model": target_model,
        "tuple_data": tuple_data,
        "client_start_time": time.time()
    }

    # dispatch the payload to the queue and wait for cluster response
    aws.dispatch_and_wait(payload)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n [SYSTEM] Client terminated by user.")
        sys.exit(0)