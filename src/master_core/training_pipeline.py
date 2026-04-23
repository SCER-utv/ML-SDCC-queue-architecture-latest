import time
import json
import math
import os
from src.utils.config import load_config
from src.utils.job_paths import JobPaths


# orchestrates the distributed training workflow
class TrainingPipeline:

    def __init__(self, aws_manager):
        self.aws = aws_manager
        self.config = load_config()

    # executes the training process across distributed workers
    def run(self, job_data, job_id):
        num_workers = job_data['num_workers']

        # 1. fault tolerance state recovery
        completed_train_tasks, s3_inference_results, start_train, tasks_dispatched, training_time = self._recover_or_initialize_state(
            job_id)

        # 2. infrastructure provisioning
        self.aws.scale_worker_infrastructure(num_workers)
        time.sleep(10)

        # 3. data split management
        calculated_train_rows = None
        if not tasks_dispatched:
            calculated_train_rows = self._ensure_dataset_ready(job_data)

        # 4. fan-out task generation
        if not tasks_dispatched:
            self._generate_tasks(job_data, job_id, calculated_train_rows)
            tasks_dispatched = True
            self.aws.update_job_state(job_id, completed_train_tasks, s3_inference_results, start_train,
                                      tasks_dispatched, training_time, 0.0)
        else:
            print(" [RECOVERY] SQS Fan-Out skipped to prevent duplicates.")

        # 5. wait for worker results via sqs polling
        self._wait_for_workers(job_id, num_workers, completed_train_tasks, start_train, tasks_dispatched)

        # 6. closure and client notification
        total_run_time = time.time() - start_train
        print(f" [TIMERS] Distributed Training completed in {total_run_time:.2f}s")

        if job_data.get('mode') == 'train':
            self._send_client_response(job_id, "train", total_run_time)


    #support methods

    # recovers previous training state from dynamodb or initializes a new one
    def _recover_or_initialize_state(self, job_id):
        completed_train_tasks, s3_results, db_start, tasks_dispatched, train_time, _ = self.aws.get_job_state(job_id)

        if db_start is None:
            start_train = time.time()
            self.aws.update_job_state(job_id, completed_train_tasks, s3_results, start_train, False, train_time, 0.0)
        else:
            start_train = db_start
            print(f" [RECOVERY] Restored. Current state: {len(completed_train_tasks)} Train tasks complete.")

        return completed_train_tasks, s3_results, start_train, tasks_dispatched, train_time

    # handles dataset streaming split and row counting
    def _ensure_dataset_ready(self, job_data):
        dataset: JobPaths = job_data['dataset_paths']


        bucket, target_train_key = self.aws.parse_s3_uri(dataset.train_url)
        _, target_test_key = self.aws.parse_s3_uri(dataset.test_url)


        # native fault tolerance: does s3 file already exist?
        if self.aws.check_s3_file_exists(bucket, target_train_key):
            print(f" [PIPELINE] Dataset ready on {dataset.train_url}. (Bypass split / Recovery active)")
            return self.aws.get_total_rows_s3_select(bucket, target_train_key)

        # if it does not exist, it creates it from the raw file
        if dataset.raw_source_to_split:
            print(f" [PIPELINE] Train data not found. Starting split from {dataset.raw_source_to_split}...")

            try:
                raw_bucket, raw_key = self.aws.parse_s3_uri(dataset.raw_source_to_split)
            except Exception:
                raise ValueError(f"Invalid S3 URL format for raw source: {dataset.raw_source_to_split}")

            if not self.aws.check_s3_file_exists(raw_bucket, raw_key):
                raise ValueError(
                    f"Cannot perform split: the source file {dataset.raw_source_to_split} DOES NOT exist on S3!")

            try:
                calculated_train_rows, _ = self.aws.execute_streaming_split(
                    dataset.raw_source_to_split,
                    target_train_key=target_train_key,
                    target_test_key=target_test_key
                )
                return calculated_train_rows
            except Exception as e:
                print(f" [CRITICAL] Split failed: {e}")
                raise e
        else:
            # extreme error, file to split not found
            raise ValueError(
                f"Impossible to start job: {dataset.train_url} does not exist and the split has not been requested.")


    def _fetch_target_strategies(self, strategy, dataset, num_trees_total, num_workers):
        strategies_s3_key = f"config/{strategy}_tasks.json"
        target_strategies = []

        try:
            print(f" [PIPELINE] Downloading strategies from s3://{self.aws.bucket}/{strategies_s3_key}...")
            response = self.aws.s3_client.get_object(Bucket=self.aws.bucket, Key=strategies_s3_key)
            all_strategies = json.loads(response['Body'].read().decode('utf-8'))

            if strategy == "homogeneous":
                dataset_conf = all_strategies.get(dataset, {})
                conf = dataset_conf.get(str(num_trees_total))

                #this section finds the closest configuration to the input num_trees and uses it
                if not conf and dataset_conf:
                    try:
                        available_trees = [int(k) for k in dataset_conf.keys() if k.isdigit()]
                        if available_trees:
                            closest_trees = min(available_trees, key=lambda x: abs(x - num_trees_total))
                            conf = dataset_conf[str(closest_trees)]
                            print(
                                f" [INFO] Exact configuration for {num_trees_total} trees not found. Using closest available, configuration for: {closest_trees} trees.")
                    except Exception as e:
                        print(f" [WARNING] Error calculating closest tree configuration: {e}")

                if conf:
                    target_strategies = [conf] * num_workers
            else:
                # heterogeneous configuration
                target_strategies = all_strategies.get(dataset, {}).get(str(num_workers), [])

        except self.aws.s3_client.exceptions.NoSuchKey:
            print(f" [WARNING] File {strategies_s3_key} not found on S3. Using fallback parameters.")
        except Exception as e:
            print(f" [WARNING] Error while downloading strategies from S3: {e}. Using fallback parameters.")

        return target_strategies



    # generates and queues individual training tasks for each worker
    def _generate_tasks(self, job_data, job_id, total_rows):
        num_workers = job_data['num_workers']
        num_trees_total = job_data['num_trees']
        dataset = job_data['dataset']
        dataset_paths = job_data['dataset_paths']
        strategy = job_data.get('strategy', 'homogeneous')
        train_s3_uri = dataset_paths.train_url
        task_type = job_data['task_type']
        target_col = job_data['target_column']

        target_strategies = job_data.get('custom_hyperparams')

        if not target_strategies:
            target_strategies = self._fetch_target_strategies(strategy, dataset, num_trees_total, num_workers)

        # fallback configuration
        if not target_strategies:
            target_strategies = [{"max_depth": None, "max_features": "sqrt",
                                  "criterion": "gini" if task_type == 'classification' else 'squared_error'}] * num_workers

        rows_per_worker = total_rows // num_workers
        remainder_rows = total_rows % num_workers
        trees_per_worker = math.floor(num_trees_total / num_workers)
        trees_remainder = num_trees_total % num_workers
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

            raw_depth = conf.get('max_depth')
            max_depth = None if raw_depth in ["None", None] else (int(raw_depth) if str(raw_depth).lstrip('-').isdigit() else None)

            raw_features = conf.get('max_features', 'sqrt')
            if raw_features not in ["sqrt", "log2", "None", None]:
                try:
                    val_float = float(raw_features)
                    raw_features = int(val_float) if val_float.is_integer() else val_float
                except:
                    raw_features = "sqrt"

            raw_samples = float(conf.get('max_samples', 1.0))

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
                "max_features": raw_features,
                "criterion": conf.get('criterion'),
                "min_samples_split": conf.get('min_samples_split', 2),
                "min_samples_leaf": conf.get('min_samples_leaf', 1),
                "max_samples": raw_samples,
                "class_weight": conf.get('class_weight', None),
                "n_jobs": conf.get('n_jobs', -1),
                "is_custom": job_data.get('is_custom', False),
                "custom_target_col": target_col,
                "task_type": task_type
            }

            current_skip += n_rows
            self.aws.sqs_client.send_message(QueueUrl=self.aws.sqs_queues["train_task"], MessageBody=json.dumps(task_payload))
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

    # polls the sqs response queue until all workers complete their training tasks
    def _wait_for_workers(self, job_id, num_workers, completed_train_tasks, start_train, tasks_dispatched):
        print("\n [EVENT LOOP] Master listening actively for Worker responses...\n")
        train_resp_queue = self.aws.sqs_queues["train_response"]

        while len(completed_train_tasks) < num_workers:
            res_train = self.aws.sqs_client.receive_message(QueueUrl=train_resp_queue, MaxNumberOfMessages=10, WaitTimeSeconds=2)
            if 'Messages' in res_train:
                for msg in res_train['Messages']:
                    train_resp = json.loads(msg['Body'])
                    task_id = train_resp['task_id']
                    if task_id not in completed_train_tasks:
                        completed_train_tasks.add(task_id)
                        print(
                            f" [ACK] Worker completed training for {task_id}! ({len(completed_train_tasks)}/{num_workers})")
                        self.aws.update_job_state(job_id, completed_train_tasks, {}, start_train, tasks_dispatched, 0.0,
                                                  0.0)
                    self.aws.sqs_client.delete_message(QueueUrl=train_resp_queue, ReceiptHandle=msg['ReceiptHandle'])

        training_time = time.time() - start_train
        self.aws.update_job_state(job_id, completed_train_tasks, {}, start_train, tasks_dispatched, training_time, 0.0)
        print("\n [PIPELINE] All Workers completed their Training tasks!")

    # notifies the client that the training pipeline has finished
    def _send_client_response(self, job_id, mode, total_time):
        client_response_queue = self.aws.sqs_queues["client_response"]
        if client_response_queue:
            self.aws.sqs_client.send_message(QueueUrl=client_response_queue, MessageBody=json.dumps(
                {"job_id": job_id, "target_model": job_id, "mode": mode, "total_time_sec": round(total_time, 2)}))