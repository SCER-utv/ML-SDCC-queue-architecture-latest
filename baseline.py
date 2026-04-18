import io
import sys
import time
import os
import gc

import boto3
import botocore
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, accuracy_score, mean_squared_error, r2_score, mean_absolute_error,
    f1_score, precision_score, recall_score, average_precision_score, mean_absolute_percentage_error
)

from src.model.model_factory import ModelFactory
from src.utils.config import load_config

# load system configuration and aws credentials
try:
    config = load_config()
except Exception as e:
    print(f" [CRITICAL] Error during Auto-Discovery: {e}")
    sys.exit(1)

AWS_REGION = config.get("aws_region")
TARGET_BUCKET = config.get("s3_bucket")


# define target datasets, tree variations, and golden standard hyperparameters
TARGET_DATASETS = ["airlines", "taxi"]
TREES_GRID = [25, 50, 75, 100, 150, 200, 300]

GOLD_STANDARD_PARAMS = {
    "airlines": {
        50: {"max_depth": 20, "min_samples_split": 50, "min_samples_leaf": 5, "max_features": 0.231,
             "max_samples": 0.598, "criterion": "gini", "class_weight": "balanced", "n_jobs": -1},
        75: {"max_depth": 28, "min_samples_split": 60, "min_samples_leaf": 5, "max_features": 0.2, "max_samples": 0.55,
             "criterion": "gini", "class_weight": "balanced", "n_jobs": -1},
        100: {"max_depth": 27, "min_samples_split": 50, "min_samples_leaf": 6, "max_features": 0.234,
              "max_samples": 0.564, "criterion": "gini", "class_weight": "balanced", "n_jobs": -1},
        200: {"max_depth": 19, "min_samples_split": 20, "min_samples_leaf": 4, "max_features": 0.300,
              "max_samples": 0.600, "criterion": "gini", "class_weight": "balanced", "n_jobs": -1},
        300: {"max_depth": 20, "min_samples_split": 45, "min_samples_leaf": 6, "max_features": 0.357,
              "max_samples": 0.539, "criterion": "gini", "class_weight": "balanced", "n_jobs": -1}
    },
    "taxi": {
        25: {"max_depth": 54, "min_samples_split": 2, "min_samples_leaf": 4, "max_features": 0.52, "max_samples": 0.7,
             "criterion": "friedman_mse", "n_jobs": -1},
        50: {"max_depth": 38, "min_samples_split": 2, "min_samples_leaf": 2, "max_features": "sqrt", "max_samples": 0.9,
             "criterion": "friedman_mse", "n_jobs": -1},
        75: {"max_depth": 42, "min_samples_split": 2, "min_samples_leaf": 2, "max_features": "sqrt", "max_samples": 0.9,
             "criterion": "friedman_mse", "n_jobs": -1},
        100: {"max_depth": 48, "min_samples_split": 2, "min_samples_leaf": 3, "max_features": 0.5, "max_samples": 1.0,
              "criterion": "friedman_mse", "n_jobs": -1}
    }
}


s3_client = boto3.client('s3', region_name=AWS_REGION)


# appends calculated baseline metrics to a historical csv file on s3 preserving excel formatting
def save_baseline_metrics(dataset, n_trees, train_time, inf_time, metrics_dict, config):
    s3_key = f"results/{dataset}/{dataset}_1M_baseline_results.csv"

    row_data = {
        'Dataset': dataset,
        'Trees': n_trees,
        'Train_Time': round(train_time, 2),
        'Infer_Time': round(inf_time, 2)
    }

    row_data.update(metrics_dict)
    new_row_df = pd.DataFrame([row_data])

    try:
        obj = s3_client.get_object(Bucket=TARGET_BUCKET, Key=s3_key)
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

    s3_client.put_object(Bucket=TARGET_BUCKET, Key=s3_key, Body=csv_buffer.getvalue())
    print(f" [METRICS] Baseline results securely appended to: s3://{TARGET_BUCKET}/{s3_key}")


# downloads a csv dataset from s3 directly into a pandas dataframe in memory
def load_dataset_from_s3(bucket, key):
    print(f" [DOWNLOAD] Fetching s3://{bucket}/{key} into RAM...")
    start_dl = time.time()
    response = s3_client.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(io.BytesIO(response['Body'].read()))
    print(f" [DOWNLOAD] Completed in {time.time() - start_dl:.2f}s. Loaded {len(df)} rows.")
    return df


# orchestrates the automated baseline benchmark across all configured datasets and tree sizes
def main():
    print("\n" + "=" * 60)
    print(" AUTOMATED MONOLITHIC BASELINE BENCHMARK - RANDOM FOREST")
    print("=" * 60)
    print(f" Target Datasets: {TARGET_DATASETS}")
    print(f" Trees Grid     : {TREES_GRID}")
    print("=" * 60 + "\n")

    # iterates over each target dataset to perform baseline benchmarking
    for dataset in TARGET_DATASETS:
        print("\n" + "*" * 50)
        print(f" STARTING DATASET ELABORATION: {dataset.upper()}")
        print("*" * 50)

        try:
            ml_handler = ModelFactory.get_model(dataset)
            task_type = getattr(ml_handler, 'task_type', 'classification')
        except ValueError as e:
            print(f" [CRITICAL] {e}. Skipping {dataset}.")
            continue

        train_s3_key = config['datasets_metadata'][dataset]['train_path']
        test_s3_key = config['datasets_metadata'][dataset]['test_path']

        try:
            print(" [RAM ALLOCATION] Loading entire Train and Test sets into memory...")
            df_train = load_dataset_from_s3(TARGET_BUCKET, train_s3_key)
            df_test = load_dataset_from_s3(TARGET_BUCKET, test_s3_key)
        except Exception as e:
            print(f" [CRITICAL] Memory or S3 Error during dataset loading: {e}")
            continue

        target_col = ml_handler.target_column
        y_true = df_test[target_col].values

        # trains and evaluates the model for each tree count in the grid
        for trees in TREES_GRID:
            print(f"\n --- STARTING BENCHMARK FOR {trees} TREES ---")

            try:
                params = GOLD_STANDARD_PARAMS[dataset][trees].copy()
            except KeyError:
                print(f" [WARNING] No parameter for {trees} trees on {dataset}. Skipping.")
                continue

            # inject runtime parameters
            params["n_estimators"] = trees
            params["trees"] = trees
            params["random_state"] = 42
            params["seed"] = 42
            params["n_jobs"] = -1

            print(f" [BASELINE TRAIN] Training model with params: {params}")
            train_start = time.time()
            rf_model = ml_handler.process_and_train(df_train, params)
            train_time = time.time() - train_start
            print(f" [BASELINE TRAIN] Completed in {train_time:.2f}s")

            print(f" [BASELINE INFER] Executing prediction on {len(df_test)} rows in chunks...")
            infer_start = time.time()

            # perform memory-efficient chunked inference on the test set
            chunk_size = 50000
            all_predictions = []

            for start_idx in range(0, len(df_test), chunk_size):
                end_idx = min(start_idx + chunk_size, len(df_test))
                chunk = df_test.iloc[start_idx:end_idx]

                chunk_preds = ml_handler.process_and_predict(rf_model, chunk)
                all_predictions.append(chunk_preds)

            predictions = np.vstack(all_predictions) if task_type == 'classification' else np.concatenate(
                all_predictions)

            infer_time = time.time() - infer_start
            print(f" [BASELINE INFER] Completed in {infer_time:.2f}s")

            # calculate and log evaluation metrics based on the task type
            if task_type == 'classification':
                votes_0 = predictions[:, 0]
                votes_1 = predictions[:, 1]
                y_prob = votes_1 / (votes_0 + votes_1)
                final_prediction = np.argmax(predictions, axis=1)

                auc = roc_auc_score(y_true, y_prob)
                acc = accuracy_score(y_true, final_prediction)
                f1 = f1_score(y_true, final_prediction, zero_division=0)
                prec = precision_score(y_true, final_prediction, zero_division=0)
                rec = recall_score(y_true, final_prediction, zero_division=0)
                pr_auc = average_precision_score(y_true, y_prob)

                print(
                    f" [EVALUATION] ROC-AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f} | F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | Acc: {acc:.4f}")

                metrics_dict = {
                    'ROC-AUC': round(auc, 4),
                    'PR-AUC': round(pr_auc, 4),
                    'F1-Score': round(f1, 4),
                    'Precision': round(prec, 4),
                    'Recall': round(rec, 4),
                    'Accuracy': round(acc, 4)
                }
            else:
                mse = mean_squared_error(y_true, predictions)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_true, predictions)
                mae = mean_absolute_error(y_true, predictions)
                mape = mean_absolute_percentage_error(y_true, predictions)

                print(f" [EVALUATION] RMSE: {rmse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.4f} | R2 Score: {r2:.4f}")

                metrics_dict = {
                    'RMSE': round(rmse, 4),
                    'MAE': round(mae, 4),
                    'MAPE': round(mape, 4),
                    'R2 Score': round(r2, 4)
                }

            save_baseline_metrics(dataset, trees, train_time, infer_time, metrics_dict, config)

        # force garbage collection to free ram before loading the next dataset
        print(f"\n [CLEANUP] Realeasing only memory used by dataset {dataset.upper()}...")
        del df_train
        del df_test
        del y_true
        gc.collect()

    print("\n [SUCCESS] All baselines completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n [SYSTEM] Baseline terminated by user.")
        sys.exit(0)