import boto3
import threading
import time


# handles communications with aws sqs and s3 in an isolated manner
class WorkerAWSManager:

    def __init__(self, config):
        self.region = config.get("aws_region")

        # initialize aws clients
        self.sqs = boto3.client('sqs', region_name=self.region)
        self.s3 = boto3.client('s3', region_name=self.region)

    # extract bucket and key from a s3 url
    def parse_s3_uri(self, uri):
        parts = uri.replace("s3://", "").split("/", 1)
        return parts[0], parts[1]

    # polls an sqs queue for new messages
    def poll_queue(self, queue_url, wait_time=5):
        response = self.sqs.receive_message(
            QueueUrl=queue_url, MaxNumberOfMessages=1, WaitTimeSeconds=wait_time
        )
        return response.get('Messages', [None])[0]

    # deletes a successfully processed message from the queue
    def delete_message(self, queue_url, receipt_handle):
        self.sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)

    # performs an immediate nack to reassign the task on failure (visibilitytimeout=0)
    def release_message(self, queue_url, receipt_handle):
        try:
            self.sqs.change_message_visibility(
                QueueUrl=queue_url, ReceiptHandle=receipt_handle, VisibilityTimeout=0
            )
        except Exception as e:
            print(f" [FAULT TOLERANCE] Impossible rilasciare il messaggio: {e}")

    # launches a background thread to continually renew the sqs visibility timeout
    def start_heartbeat(self, queue_url, receipt_handle, stop_event):
        def heartbeat():
            while not stop_event.is_set():
                stop_event.wait(20)
                if not stop_event.is_set():
                    try:
                        self.sqs.change_message_visibility(
                            QueueUrl=queue_url, ReceiptHandle=receipt_handle, VisibilityTimeout=60
                        )
                        print(" [HEARTBEAT] Visibilità SQS estesa di 60s.")
                    except:
                        pass

        hb_thread = threading.Thread(target=heartbeat, daemon=True)
        hb_thread.start()
        return hb_thread