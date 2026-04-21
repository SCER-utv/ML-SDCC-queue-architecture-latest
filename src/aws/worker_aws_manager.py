import json
import os

import boto3
import threading
import time


# handles communications with aws sqs and s3 in an isolated manner
class WorkerAWSManager:

    def __init__(self, config):
        # 1. Retrieve region from env
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.bucket = os.getenv("S3_BUCKET_NAME")

        if not self.bucket:
            raise ValueError(" [CRITICAL] S3_BUCKET_NAME environment variable is not set!")

        # 2. initializing aws clients
        self.sqs_client = boto3.client('sqs', region_name=self.region)
        self.s3_client = boto3.client('s3', region_name=self.region)
        self.ssm_client = boto3.client('ssm', region_name=self.region)

        config_key = os.getenv("S3_CONFIG_KEY", "config/ssm_paths.json")
        ssm_paths = self._load_remote_config(config_key)

        # 3. initializing queues by retrieving name from ssm and url
        print(" [INIT] SQS URL dynamic runtime resolution ...")
        self.sqs_queues = {
            "train_task": self._resolve_sqs_url(self._get_ssm_parameter(ssm_paths.get("sqs_train"))),
            "train_response": self._resolve_sqs_url(self._get_ssm_parameter(ssm_paths.get("sqs_train_resp"))),
            "infer_task": self._resolve_sqs_url(self._get_ssm_parameter(ssm_paths.get("sqs_infer"))),
            "infer_response": self._resolve_sqs_url(self._get_ssm_parameter(ssm_paths.get("sqs_infer_resp")))
        }

    def _load_remote_config(self, key):
        try:
            print(f" [INIT] Download configurazioni infrastrutturali da s3://{self.bucket}/{key}...")
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            return json.loads(response['Body'].read().decode('utf-8'))
        except Exception as e:
            print(f" [CRITICAL ERROR] Impossibile caricare la configurazione {key} da S3: {e}")
            raise e

    # extracts the value (resource name) from AWS SSM Parameter Store
    def _get_ssm_parameter(self, param_name):
        try:
            response = self.ssm_client.get_parameter(Name=param_name, WithDecryption=False)
            return response['Parameter']['Value']
        except Exception as e:
            print(f" [SSM ERROR] Impossible to extract parameter {param_name}: {e}")
            raise e

    # gets a complete URL of a SQS queue
    def _resolve_sqs_url(self, queue_name):
        try:
            response = self.sqs_client.get_queue_url(QueueName=queue_name)
            return response['QueueUrl']
        except Exception as e:
            print(f" [SQS ERROR] Impossible to resolute URL for the queue '{queue_name}': {e}")
            raise e

    # extract bucket and key from a s3 url
    def parse_s3_uri(self, uri):
        parts = uri.replace("s3://", "").split("/", 1)
        return parts[0], parts[1]

    # polls an sqs queue for new messages
    def poll_queue(self, queue_url, wait_time=5):
        response = self.sqs_client.receive_message(
            QueueUrl=queue_url, MaxNumberOfMessages=1, WaitTimeSeconds=wait_time
        )
        return response.get('Messages', [None])[0]

    # deletes a successfully processed message from the queue
    def delete_message(self, queue_url, receipt_handle):
        self.sqs_client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)

    # performs an immediate nack to reassign the task on failure (visibilitytimeout=0)
    def release_message(self, queue_url, receipt_handle):
        try:
            self.sqs_client.change_message_visibility(
                QueueUrl=queue_url, ReceiptHandle=receipt_handle, VisibilityTimeout=0
            )
        except Exception as e:
            print(f" [FAULT TOLERANCE] Impossible message release: {e}")

    # launches a background thread to continually renew the sqs visibility timeout
    def start_heartbeat(self, queue_url, receipt_handle, stop_event):
        def heartbeat():
            while not stop_event.is_set():
                stop_event.wait(20)
                if not stop_event.is_set():
                    try:
                        self.sqs_client.change_message_visibility(
                            QueueUrl=queue_url, ReceiptHandle=receipt_handle, VisibilityTimeout=60
                        )
                        print(" [HEARTBEAT] SQS visibility extended to 60s.")
                    except:
                        pass

        hb_thread = threading.Thread(target=heartbeat, daemon=True)
        hb_thread.start()
        return hb_thread