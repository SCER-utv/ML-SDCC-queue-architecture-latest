import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score, mean_absolute_error, \
    precision_score, recall_score, f1_score


# handles the aggregation of worker results and metrics calculation on the master
class EvaluationManager:

    # initializes the manager with an aws manager instance for s3 operations
    def __init__(self, aws_manager):
        self.aws = aws_manager

    # orchestrates the final aggregation and evaluation phase using metadata from the payload
    def aggregate_and_evaluate(self, job_data, job_id, dataset_name, dataset_variant, s3_inference_results, num_workers,
                               trees, weights, train_time, infer_time, strategy):
        print("\n" + "=" * 50)
        print(" FINAL AGGREGATION & EVALUATION PHASE")
        print("=" * 50)

        dataset_paths = job_data['dataset_paths']
        task_type = job_data['task_type']
        target_col = job_data['target_column']
        test_s3_uri = dataset_paths.test_url

        predictions_list = self._download_worker_results(s3_inference_results)
        if not predictions_list:
            return

        print(f" Reading Ground Truth from column '{target_col}'...")

        # reads the ground truth column directly from the test set on s3
        try:
            df_test = pd.read_csv(test_s3_uri, usecols=[target_col])
            y_true = df_test[target_col].values
        except Exception as e:
            print(f" [CRITICAL ERROR] Test set loading failure: {e}")
            return

        # routes to the appropriate evaluation logic based on task type
        if task_type == 'classification':
            metrics_dict = self._evaluate_classification(predictions_list, y_true, num_workers)
        else:
            metrics_dict = self._evaluate_regression(predictions_list, y_true, weights)

        print("=" * 50 + "\n")

        strategy_name = "Homogeneous" if strategy == "homogeneous" else "Heterogeneous"
        experiment_name = job_data['experiment_name']

        metrics_key = dataset_paths.metrics_key

        # saves the final metrics and cleans up temporary inference files
        self.aws.save_metrics(test_s3_uri, experiment_name, dataset_name, dataset_variant, num_workers, trees,
                              strategy_name, train_time,
                              infer_time, metrics_dict, metrics_key)
        self.aws.cleanup_s3_inference_files(s3_inference_results)

    # downloads and loads temporary .npy files containing worker predictions
    def _download_worker_results(self, s3_inference_results):
        predictions_list = []
        print(f" Downloading {len(s3_inference_results)} inference result files from S3...")
        for task_id, s3_uri in s3_inference_results.items():
            bucket, key = self.aws.parse_s3_uri(s3_uri)
            local_path = f"/tmp/res_{task_id}.npy"
            self.aws.s3_client.download_file(bucket, key, local_path)
            predictions_list.append(np.load(local_path))
            os.remove(local_path)

        if not predictions_list:
            print(" [CRITICAL ERROR] No result downloaded. Impossible to aggregate")
        return predictions_list

    # evaluates classification tasks using majority voting
    def _evaluate_classification(self, predictions_list, y_true, num_workers):
        print(" [EVALUATION] Classification task detected. Executing Majority Voting...")
        total_votes = np.sum(predictions_list, axis=0)

        if len(predictions_list[0].shape) == 2:
            votes_0, votes_1 = total_votes[:, 0], total_votes[:, 1]
            with np.errstate(divide='ignore', invalid='ignore'):
                y_prob = np.where((votes_0 + votes_1) == 0, 0, votes_1 / (votes_0 + votes_1))
            final_prediction = np.argmax(total_votes, axis=1)
        else:
            y_prob = total_votes / num_workers
            final_prediction = np.round(y_prob)

        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.0

        acc = accuracy_score(y_true, final_prediction)
        precision = precision_score(y_true, final_prediction, zero_division=0)
        recall = recall_score(y_true, final_prediction, zero_division=0)
        f1 = f1_score(y_true, final_prediction, zero_division=0)

        print(
            f"\n GLOBAL DISTRIBUTED RESULTS:\n ROC-AUC: {auc:.4f}\n Accuracy: {acc:.4f}\n Precision: {precision:.4f}\n Recall: {recall:.4f}\n F1-Score: {f1:.4f}")

        return {
            'ROC-AUC': float(round(auc, 4)), 'Accuracy': float(round(acc, 4)),
            'Precision': float(round(precision, 4)), 'Recall': float(round(recall, 4)), 'F1-Score': float(round(f1, 4))
        }

    # evaluates regression tasks using weighted averaging based on tree count
    def _evaluate_regression(self, predictions_list, y_true, weights):
        print(" [EVALUATION] Regression task detected. Executing Weighted Averaging...")
        y_pred = np.average(predictions_list, axis=0, weights=weights)

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        print(f"\n GLOBAL DISTRIBUTED RESULTS:\n RMSE: {rmse:.4f}\n MAE: {mae:.4f}\n R2 Score: {r2:.4f}")

        return {'RMSE': float(round(rmse, 4)), 'MAE': float(round(mae, 4)), 'R2 Score': float(round(r2, 4))}