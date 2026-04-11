import sys
import json
import itertools
from datetime import datetime

import boto3

from src.utils.config import load_config

# DYNAMIC CONFIGURATION & AUTO-DISCOVERY
try:
    config = load_config()
except Exception as e:
    print(f" Errore critico durante l'Auto-Discovery: {e}")
    sys.exit(1)

CLIENT_QUEUE_URL = config["sqs_queues"]["client"]
AWS_REGION = config.get("aws_region")

# =====================================================================
# HYPERPARAMETER GRID (Modify these arrays before running)
# ====================================================================
TARGET_DATASET = "airlines"          
WORKER_COUNTS = [1, 2, 3, 4, 5, 6, 7, 8, 9]     # Node scaling for Speedup measurement
TREE_COUNTS = [5, 10, 25, 50, 100, 200, 400]  # Tree scaling for workload variation
# =====================================================================

# Main execution loop. Generates job payloads and dispatches them to the SQS Client Queue as an automated batch.
def main():
    sqs_client = boto3.client('sqs', region_name=AWS_REGION)
    
    # Automatically generate the cartesian product (all possible combinations). Example: (2, 20), (2, 50), ..., (9, 400)
    grid_combinations = list(itertools.product(WORKER_COUNTS, TREE_COUNTS))

    print("\n" + "=" * 60)
    print(" DISTRIBUTED RANDOM FOREST - FULLY AUTOMATED GRID SEARCH")
    print("=" * 60)
    print(f" Target Dataset  : {TARGET_DATASET.upper()}")
    print(f" Workers Grid    : {WORKER_COUNTS}")
    print(f" Trees Grid      : {TREE_COUNTS}")
    print(f" Total Jobs      : {len(grid_combinations)}")
    print("=" * 60 + "\n")

    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Sequential dispatch to SQS
    for worker_count, tree_count in grid_combinations:
        job_id = f"job_{TARGET_DATASET}_W{worker_count}_T{tree_count}_TUNING_{batch_timestamp}"
        
        payload = {
            "mode": "train",
            "job_id": job_id,
            "dataset": TARGET_DATASET,
            "num_workers": worker_count,
            "num_trees": tree_count,
            # CRITICAL: We lock the data split to ensure scientific comparison!
            # The Master will reuse the existing S3 files (train/test) without regenerating them.
            "dynamic_split": False 
        }
        
        try:
            sqs_client.send_message(
                QueueUrl=CLIENT_QUEUE_URL,
                MessageBody=json.dumps(payload),
                MessageGroupId="ML_Jobs",
                MessageDeduplicationId=job_id
            )
            print(f" [ENQUEUED] Workers: {worker_count:<2} | Trees: {tree_count:<4} | Job ID: {job_id}")
        except Exception as e:
            print(f" [ERROR] Dispatch failed for W{worker_count}-T{tree_count}: {e}")

    print("\n [SUCCESS] All grid search configurations successfully dispatched to SQS!")
    print(" [INFO] The Master Node will process them sequentially and autonomously.")
    
    target_bucket = config.get('s3_bucket')
    print(f" [INFO] Monitor the final results at: s3://{target_bucket}/results/{TARGET_DATASET}/{TARGET_DATASET}_results.csv")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n [SYSTEM] Auto-Tuner terminated by user. Goodbye!")
        sys.exit(0)
