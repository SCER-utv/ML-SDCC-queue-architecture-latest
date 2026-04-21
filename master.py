import json
import threading

from src.master_core.evaluation_manager import EvaluationManager
from src.master_core.inference_pipeline import InferencePipeline
from src.master_core.training_pipeline import TrainingPipeline
from src.utils.config import load_config

from src.aws.aws_manager import AWSManager


# extends sqs message visibility in the background to prevent timeout during long jobs
def extend_client_sqs_visibility(aws, queue_url, receipt_handle, stop_event):
    while not stop_event.is_set():
        stop_event.wait(20)
        if not stop_event.is_set():
            try:
                aws.sqs_client.change_message_visibility(QueueUrl=queue_url, ReceiptHandle=receipt_handle,
                                                         VisibilityTimeout=60)
            except Exception:
                pass


# translates the client's thin payload into concrete and secure s3 paths for the master
def resolve_paths(job_data, config, aws):
    is_custom = job_data.get('is_custom', False)
    mode = job_data.get('mode')
    bucket = aws.bucket
    job_id = job_data.get('job_id')
    needs_split = job_data.get('needs_split', False)

    experiment_name = job_data.get('experiment_name')

    if not is_custom:
        # handle official benchmark datasets (gold standard)
        meta = config['datasets_metadata'][job_data['dataset']][job_data['dataset_variant']]

        if mode in ['train', 'train_and_infer']:
            if needs_split:
                job_data['train_s3_url'] = f"s3://{bucket}/{meta['interim_path']}"
                job_data['target_train_key'] = meta['train_path']
                job_data['target_test_key'] = meta['test_path']
            else:
                job_data['train_s3_url'] = f"s3://{bucket}/{meta['train_path']}"
                job_data['test_s3_url'] = f"s3://{bucket}/{meta['test_path']}"
        elif mode == 'bulk_infer':
            job_data['test_s3_url'] = f"s3://{bucket}/{meta['test_path']}"

        job_data['metrics_s3_key'] = f"metrics/gold_standard/{job_data['dataset']}/{job_data['dataset_variant']}/results_{job_id}.json"

    else:
        # handle user-provided datasets and custom experiments
        folder_base = f"experiments/{experiment_name}" if experiment_name else f"splits/{job_id}"

        if mode in ['train', 'train_and_infer']:
            job_data['train_s3_url'] = job_data.get('custom_train_url')

            if needs_split:
                # force split outputs to target these exact keys
                job_data['target_train_key'] = f"{folder_base}/train.csv"
                job_data['target_test_key'] = f"{folder_base}/test.csv"
            else:
                job_data['test_s3_url'] = job_data.get('custom_test_url')

        elif mode == 'bulk_infer':
            job_data['test_s3_url'] = job_data.get('custom_test_url')

        # isolate evaluation metrics next to the experiment csv files
        job_data['metrics_s3_key'] = f"{folder_base}/results_{job_id}.json"

    return job_data



# initializes components and starts the main event loop to orchestrate client jobs
def main():
    print(" [MASTER] Components initialization...")
    config = load_config()

    aws = AWSManager(config)
    evaluator = EvaluationManager(aws)
    trainer = TrainingPipeline(aws)
    inferencer = InferencePipeline(aws, evaluator)

    CLIENT_QUEUE_URL = aws.sqs_queues["client"]
    print(" [MASTER] System ready. Listening for new Jobs from client...")

    while True:
        response = aws.sqs_client.receive_message(QueueUrl=CLIENT_QUEUE_URL, MaxNumberOfMessages=1, WaitTimeSeconds=20)

        if 'Messages' in response:
            client_msg = response['Messages'][0]
            receipt_handle = client_msg['ReceiptHandle']
            raw_job_data = json.loads(client_msg['Body'])

            raw_job_data.setdefault('job_id', client_msg['MessageId'])
            mode = raw_job_data.get('mode', 'train')
            job_id = raw_job_data['job_id']

            print(f"\n{'=' * 50}\n STARTING PIPELINE: {job_id} (Mode: {mode})\n{'=' * 50}")

            # securely resolve and assign authoritative s3 paths
            job_data = resolve_paths(raw_job_data, config, aws)

            stop_event = threading.Event()
            heartbeat_thread = threading.Thread(target=extend_client_sqs_visibility,
                                                args=(aws, CLIENT_QUEUE_URL, receipt_handle, stop_event))
            heartbeat_thread.start()

            try:
                # dynamically route the job to the appropriate pipeline
                if mode == 'train':
                    trainer.run(job_data, job_id)
                elif mode == 'bulk_infer':
                    inferencer.run_bulk(job_data, job_id)
                elif mode == 'infer':
                    inferencer.run_realtime(job_data, job_id)
                elif mode == 'train_and_infer':
                    trainer.run(job_data, job_id)
                    job_data['target_model'] = job_id
                    inferencer.run_bulk(job_data, job_id)
                else:
                    print(f" [WARNING] Unknown mode requested: {mode}")

            except Exception as e:
                print(f" [CRITICAL ERROR] Pipeline execution failed: {e}")
            finally:
                # ensure heartbeat stops and the processed message is deleted from the queue
                stop_event.set()
                heartbeat_thread.join()
                aws.delete_message(CLIENT_QUEUE_URL, receipt_handle)
                print(f" JOB {job_id} PROCESSED AND REMOVED FROM QUEUE.\n")


if __name__ == "__main__":
    main()