import os
import json
import time
import joblib
import boto3


class ClientAWSManager:
    """Gestisce tutte le comunicazioni tra il Client locale e l'infrastruttura AWS."""

    def __init__(self, config):
        self.region = config.get("aws_region")
        self.bucket = config.get("s3_bucket")
        self.client_queue_url = config["sqs_queues"]["client"]
        self.client_resp_queue = config["sqs_queues"].get("client_response")

        # Inizializziamo i client AWS una volta sola (Connection Pooling)
        self.s3_client = boto3.client('s3', region_name=self.region)
        self.sqs_client = boto3.client('sqs', region_name=self.region)

    def list_available_models(self, dataset):
        """Scansiona S3 per trovare tutti i modelli addestrati per un certo dataset."""
        prefix = f"models/{dataset}/"
        resp = self.s3_client.list_objects_v2(Bucket=self.bucket, Prefix=prefix, Delimiter='/')
        models = []
        if 'CommonPrefixes' in resp:
            for obj in resp['CommonPrefixes']:
                folder_name = obj['Prefix'].replace(prefix, '').strip('/')
                models.append(folder_name)
        return models

    def get_feature_names_from_s3(self, s3_key, target_column="Label"):
        """Legge on-the-fly l'header di un CSV su S3 per guidare l'utente nell'input real-time."""
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

    def download_and_merge_model(self, dataset, job_id):
        """Scarica i file .joblib dei worker e li unisce in un unico modello Scikit-Learn locale."""
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

    def dispatch_and_wait(self, payload):
        """Spedisce il job al Master Node e attende pazientemente la risposta."""
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

            while time.time() - start_wait < 900:  # Timeout 15 minuti
                res = self.sqs_client.receive_message(QueueUrl=self.client_resp_queue, MaxNumberOfMessages=1,
                                                      WaitTimeSeconds=20)
                if 'Messages' in res:
                    for msg in res['Messages']:
                        body = json.loads(msg['Body'])
                        receipt = msg['ReceiptHandle']

                        # Controlliamo se è la risposta al nostro job esatto
                        if body.get("job_id") == payload['job_id']:
                            print("\n" + "=" * 60)
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
                            # Non è per noi, rimettiamo il messaggio in coda immediatamente
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