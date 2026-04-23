import json
import os
import io
import time
import math
import random
import boto3
import botocore
import pandas as pd


# this class handles all the aws interactions from master
class AWSManager:

    def __init__(self, config):
        # 1. Retrieving base params from environment variables
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.bucket = os.getenv("S3_BUCKET_NAME")

        if not self.bucket:
            raise ValueError(" [CRITICAL] S3_BUCKET_NAME environment variable is not set!")

        #2. initializing aws clients
        self.s3_client = boto3.client('s3', region_name=self.region)
        self.asg_client = boto3.client('autoscaling', region_name=self.region)
        self.ec2_client = boto3.client('ec2', region_name=self.region)
        self.sqs_client = boto3.client('sqs', region_name=self.region)
        self.ssm_client = boto3.client('ssm', region_name=self.region)

        #get s3 key to retrieve configuration from ssm
        config_key = os.getenv("S3_CONFIG_KEY", "config/ssm_paths.json")
        ssm_paths = self._load_remote_config(config_key)

        #3. retrieve of ssm params
        print(" [INIT] Retrieving infrastructure from AWS SSM...")
        self.asg_name = self._get_ssm_parameter(ssm_paths.get("asg_worker"))
        self.dynamodb_table = self._get_ssm_parameter(ssm_paths.get("dynamodb_table"))

        self.dynamodb = boto3.resource('dynamodb', region_name=self.region)

        #4. initializing queues by retrieving name from ssm and url
        print(" [INIT] SQS URL dynamic runtime resolution ...")
        self.sqs_queues = {
            "client": self._resolve_sqs_url(self._get_ssm_parameter(ssm_paths.get("sqs_client"))),
            "client_response": self._resolve_sqs_url(self._get_ssm_parameter(ssm_paths.get("sqs_client_resp"))),
            "train_task": self._resolve_sqs_url(self._get_ssm_parameter(ssm_paths.get("sqs_train"))),
            "train_response": self._resolve_sqs_url(self._get_ssm_parameter(ssm_paths.get("sqs_train_resp"))),
            "infer_task": self._resolve_sqs_url(self._get_ssm_parameter(ssm_paths.get("sqs_infer"))),
            "infer_response": self._resolve_sqs_url(self._get_ssm_parameter(ssm_paths.get("sqs_infer_resp")))
        }

        self.config = config

    #used to load ssm path file from s3
    def _load_remote_config(self, key):
        try:
            print(f" [INIT] Download infrastructural configuration from s3://{self.bucket}/{key}...")
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            config_data = json.loads(response['Body'].read().decode('utf-8'))
            return config_data
        except Exception as e:
            print(f" [CRITICAL ERROR] Impossible to load configuration {key} da S3: {e}")
            raise e

    #extracts the value (resource name) from AWS SSM Parameter Store
    def _get_ssm_parameter(self, param_name):
        try:
            response = self.ssm_client.get_parameter(Name=param_name, WithDecryption=False)
            return response['Parameter']['Value']
        except Exception as e:
            print(f" [SSM ERROR] Impossible to extract parameter {param_name}: {e}")
            raise e

    #gets a complete URL of a SQS queue
    def _resolve_sqs_url(self, queue_name):
        try:
            response = self.sqs_client.get_queue_url(QueueName=queue_name)
            return response['QueueUrl']
        except Exception as e:
            print(f" [SQS ERROR] Impossible to resolute URL for the queue '{queue_name}': {e}")
            raise e

    # extract bucket and key from a s3 url
    @staticmethod
    def parse_s3_uri(s3_uri):
        if s3_uri is None or s3_uri == "":
            return "", ""
        parts = s3_uri.replace("s3://", "").split("/", 1)
        return parts[0], parts[1]

    # rapid check of a file existence
    def check_s3_file_exists(self, bucket, key):
        try:
            self.s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                return False
            raise e

    # executes a sql query on s3 to count rows of a csv
    def get_total_rows_s3_select(self, bucket, key):
        print(f" [S3-SELECT] Executing 'SELECT count(*)' on s3://{bucket}/{key}...")
        try:
            resp = self.s3_client.select_object_content(
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

    # retrieves all distributed model chunks (.joblib) saved on s3
    def count_model_parts(self, bucket, dataset, target_model):
        try:
            model_dataset_folder = target_model.split('_')[1]
        except Exception:
            model_dataset_folder = dataset

        prefix = f"models/{model_dataset_folder}/{target_model}/"
        resp = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

        chunks = [f"s3://{bucket}/{obj['Key']}" for obj in resp.get('Contents', []) if obj['Key'].endswith('.joblib')]

        if not chunks:
            print(f" [WARNING S3] No file .joblib found on s3://{bucket}/{prefix}")

        return chunks


    # streams a csv from s3 to dynamically split it into train and test sets
    def execute_streaming_split(self, source_url, target_train_key=None, target_test_key=None):
        config = self.config
        ratios = config.get("split_ratios", {"train": 0.70})
        train_threshold = ratios.get("train", 0.70)

        bucket, source_key = self.parse_s3_uri(source_url)
        file_name = source_key.split('/')[-1].replace('.csv', '')

        print(f" [SPLIT] Starting dynamic streaming split for '{file_name}'...")

        # fallback to default keys if dynamic paths are not provided by master
        train_key = target_train_key if target_train_key else f"splits/{file_name}_train.csv"
        test_key = target_test_key if target_test_key else f"splits/{file_name}_test.csv"

        local_train = f"/tmp/{file_name}_train.csv"
        local_test = f"/tmp/{file_name}_test.csv"

        try:
            print(f" [SPLIT] Line-by-line streaming in progress...")
            response = self.s3_client.get_object(Bucket=bucket, Key=source_key)
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
            print(" [SPLIT] Uploading to secure paths on S3...")

            self.s3_client.upload_file(local_train, bucket, train_key)
            self.s3_client.upload_file(local_test, bucket, test_key)

        except Exception as e:
            print(f" [SPLIT ERROR] Failed during streaming split: {e}")
            raise e
        finally:
            if os.path.exists(local_train): os.remove(local_train)
            if os.path.exists(local_test): os.remove(local_test)

        print(f" [SPLIT] Operation completed successfully.")
        return train_rows, f"s3://{bucket}/{train_key}"

    # appends inference results and metrics to a historical csv file on s3
    def save_metrics(self, test_set_url, experiment_name, dataset_name, dataset_variant, n_workers, n_trees, strategy_name, train_time, inf_time, metrics_dict, metrics_key):

        if experiment_name is None:
            # case of preconfigured datasets, using discovery
            s3_key = f"results/{dataset_name}/{dataset_name}_{dataset_variant}_distributed_results.csv"
        else:
            # case of custom datasets, experiment name used to keep track of different configurations on same dataset
            s3_key = f"experiments/{experiment_name}/{experiment_name}_distributed_results.csv"

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
            obj = self.s3_client.get_object(Bucket=self.bucket, Key=metrics_key)
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
        self.s3_client.put_object(Bucket=self.bucket, Key=metrics_key, Body=csv_buffer.getvalue())
        print(f" [METRICS] Results securely appended to: s3://{self.bucket}/{metrics_key}")

    # deletes temporary .npy prediction files generated by workers after aggregation
    def cleanup_s3_inference_files(self, s3_inference_results):
        print(" [CLEANUP] Deleting temporary .npy from S3...")
        deleted_count = 0
        for task_id, s3_uri in s3_inference_results.items():
            try:
                bucket, key = self.parse_s3_uri(s3_uri)
                self.s3_client.delete_object(Bucket=bucket, Key=key)
                deleted_count += 1
            except Exception as e:
                print(f" [CLEANUP ERROR] Delete error of {s3_uri}: {e}")
        print(f" [CLEANUP] Removed {deleted_count} temporary files successfully.")



    # dynamically scales ec2 instances using the auto-scaling group
    def scale_worker_infrastructure(self, num_workers):
        print(f" [ASG] Setting desired capacity to {num_workers} workers...")
        self.asg_client.update_auto_scaling_group(
            AutoScalingGroupName=self.asg_name, MinSize=0, DesiredCapacity=num_workers, MaxSize=10
        )

        if num_workers == 0:
            return

        print(f" [ASG] Waiting for instances to start for tagging...")
        found_instances = []

        # 120 seconds timeout loop
        for _ in range(24):
            time.sleep(5)
            response = self.ec2_client.describe_instances(
                Filters=[
                    {'Name': 'tag:aws:autoscaling:groupName', 'Values': [self.asg_name]},
                    {'Name': 'instance-state-name', 'Values': ['pending', 'running']}
                ]
            )

            for reservation in response.get('Reservations', []):
                for inst in reservation.get('Instances', []):
                    found_instances.append(inst['InstanceId'])

            if len(found_instances) >= num_workers:
                break

        if len(found_instances) > 0:
            # case of instances lack handling
            if len(found_instances) < num_workers:
                print(
                    f" [ASG WARN] Requested {num_workers} workers, but AWS provided {len(found_instances)}. Proceeding degraded.")
            else:
                print(f" [ASG] Found {len(found_instances)} instances. Applying name tags...")

            for i, instance_id in enumerate(found_instances):
                try:
                    self.ec2_client.create_tags(
                        Resources=[instance_id],
                        Tags=[{'Key': 'Name', 'Value': f"DRF-worker-{i + 1}"}]
                    )
                except Exception:
                    pass
            print(" [ASG] Name tags applied successfully.")
        else:
            print(" [ASG CRITICAL] No instances provided by ASG within timeout!")

    # retrieves the progress state of a specific job from dynamodb
    def get_job_state(self, job_id):
        table = self.dynamodb.Table(self.dynamodb_table)
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

    # updates the progress state of a job in dynamodb
    def update_job_state(self, job_id, completed_train_set, completed_infer_dict, start_time, tasks_dispatched,
                         training_time=0.0, inference_time=0.0):
        table = self.dynamodb.Table(self.dynamodb_table)
        table.put_item(Item={
            'job_id': job_id,
            'completed_train': list(completed_train_set),
            'completed_infer': completed_infer_dict,
            'start_time': str(start_time),
            'tasks_dispatched': tasks_dispatched,
            'tempo_training': str(training_time),
            'tempo_inferenza': str(inference_time)
        })

    # deletes a processed message from an sqs queue
    def delete_message(self, queue_url, receipt_handle):
        self.sqs_client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)