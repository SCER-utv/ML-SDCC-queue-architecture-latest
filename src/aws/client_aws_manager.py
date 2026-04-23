import os
import json
import time
import joblib
import boto3

# handles all communications between the local client and the aws infrastructure
class ClientAWSManager:

    def __init__(self, config):
        # 1. Retrieving base params from environment variables
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.bucket = os.getenv("S3_BUCKET_NAME")

        if not self.bucket:
            raise ValueError(" [CRITICAL] S3_BUCKET_NAME environment variable is not set!")

        #2. initializing aws clients
        self.s3_client = boto3.client('s3', region_name=self.region)
        self.sqs_client = boto3.client('sqs', region_name=self.region)
        self.ssm_client = boto3.client('ssm', region_name=self.region)

        config_key = os.getenv("S3_CONFIG_KEY", "config/ssm_paths.json")
        ssm_paths = self._load_remote_config(config_key)

        # 3. initializing queues by retrieving name from ssm and url
        print(" [INIT] SQS URL dynamic runtime resolution ......")
        self.client_queue_url = self._resolve_sqs_url(self._get_ssm_parameter(ssm_paths.get("sqs_client")))
        self.client_resp_queue = self._resolve_sqs_url(self._get_ssm_parameter(ssm_paths.get("sqs_client_resp")))

    # used to load ssm path file from s3
    def _load_remote_config(self, key):
        try:
            print(f" [INIT] Download infrastructural configuration from s3://{self.bucket}/{key}...")
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            config_data = json.loads(response['Body'].read().decode('utf-8'))
            return config_data
        except Exception as e:
            print(f" [CRITICAL ERROR] Impossible to load configuration {key} da S3: {e}")
            raise e

    def _get_ssm_parameter(self, param_name):
        try:
            response = self.ssm_client.get_parameter(Name=param_name, WithDecryption=False)
            return response['Parameter']['Value']
        except Exception as e:
            print(f" [SSM ERROR] Impossible to extract parameter {param_name}: {e}")
            raise e

    def _resolve_sqs_url(self, queue_name):
        try:
            response = self.sqs_client.get_queue_url(QueueName=queue_name)
            return response['QueueUrl']
        except Exception as e:
            print(f" [SQS ERROR] Impossible to resolute URL for the queue '{queue_name}': {e}")
            raise e


    # scans s3 to find all trained models for a given dataset
    def list_available_models(self, dataset):
        prefix = f"models/{dataset}/"
        resp = self.s3_client.list_objects_v2(Bucket=self.bucket, Prefix=prefix, Delimiter='/')
        models = []
        if 'CommonPrefixes' in resp:
            for obj in resp['CommonPrefixes']:
                folder_name = obj['Prefix'].replace(prefix, '').strip('/')
                models.append(folder_name)
        return models

    # reads the header of a csv on s3 on-the-fly to guide the user in real-time input
    def get_feature_names_from_s3(self, s3_key, target_column="Label"):
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
            first_line = next(response['Body'].iter_lines()).decode('utf-8')
            response['Body'].close()

            all_columns = [col.strip() for col in first_line.split(',')]
            if target_column in all_columns:
                all_columns.remove(target_column)

            return all_columns
        except Exception as e:
            print(f" [WARNING] Unable to read header from S3: {e}")
            return []

    # downloads worker .joblib files and merges them into a single local scikit-learn model
    def download_and_merge_model(self, dataset, job_id):
        print(f"\n" + "-" * 40)
        print(f" [DOWNLOAD] Fetching model chunks for {job_id}...")
        prefix = f"models/{dataset}/{job_id}/"
        resp = self.s3_client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)

        if 'Contents' not in resp:
            print(" [ERROR] No chunks found for this model on S3.")
            return

        os.makedirs("tmp_downloads", exist_ok=True)
        downloaded_files = []

        for obj in resp['Contents']:
            if obj['Key'].endswith('.joblib'):
                file_name = obj['Key'].split('/')[-1]
                local_path = os.path.join("tmp_downloads", file_name)
                self.s3_client.download_file(self.bucket, obj['Key'], local_path)
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

    # dispatches the job to the master node and patiently waits for the response
    def dispatch_and_wait(self, payload):
        print("\n" + "=" * 60)
        print(" Dispatching request to Master Node...")

        try:
            self.sqs_client.send_message(
                QueueUrl=self.client_queue_url, MessageBody=json.dumps(payload),
                MessageGroupId="ML_Jobs", MessageDeduplicationId=payload['job_id']
            )
            print(f" [SUCCESS] Message enqueued successfully.")
            print(f" [INFO] Generated Job ID: {payload['job_id']}")

            print(f"\n [WAIT] Waiting for response from the cluster...")
            print(" (If machines are cold-starting or training, this may take some time)")

            start_wait = time.time()
            result_found = False

            # 15 minutes timeout
            while time.time() - start_wait < 900:
                res = self.sqs_client.receive_message(QueueUrl=self.client_resp_queue, MaxNumberOfMessages=1,
                                                      WaitTimeSeconds=20)
                if 'Messages' in res:
                    for msg in res['Messages']:
                        body = json.loads(msg['Body'])
                        receipt = msg['ReceiptHandle']

                        # check if it's the response for our exact job
                        if body.get("job_id") == payload['job_id']:
                            print("\n" + "=" * 60)

                            if body.get("status") == "FAILED":
                                print("CRITICAL ERROR FROM MASTER NODE")
                                print("=" * 60)
                                print(f" REASON: {body.get('message', 'Unknown Error')}")
                                print("=" * 60 + "\n")

                                self.sqs_client.delete_message(QueueUrl=self.client_resp_queue, ReceiptHandle=receipt)
                                result_found = True
                                break

                            if payload['mode'] == 'train':
                                print(" DISTRIBUTED TRAINING COMPLETED!")
                                print("=" * 60)
                                print(f" YOUR MODEL ID IS: >>> {body.get('job_id')} <<<")
                            elif payload['mode'] == 'train_and_infer':
                                print(" END-TO-END PIPELINE COMPLETED SUCCESSFULLY!")
                                print("=" * 60)
                                print(f" YOUR MODEL ID IS: >>> {body.get('job_id')} <<<")
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

                            self.sqs_client.delete_message(QueueUrl=self.client_resp_queue, ReceiptHandle=receipt)
                            result_found = True
                            break
                        else:
                            # not for us, put the message back in the queue immediately
                            try:
                                self.sqs_client.change_message_visibility(QueueUrl=self.client_resp_queue,
                                                                          ReceiptHandle=receipt, VisibilityTimeout=0)
                            except Exception:
                                pass
                if result_found:
                    break

            if not result_found:
                print("\n [TIMEOUT] The cluster took too long to respond. Check Master logs.")

        except Exception as e:
            print(f"\n [CRITICAL ERROR] Failed to dispatch SQS message: {e}")