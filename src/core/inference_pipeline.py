import time
import json
import math
from src.utils.config import load_config


class InferencePipeline:
    """Orchestra i flussi di lavoro di Inferenza (Bulk e Real-Time)."""

    def __init__(self, aws_manager, evaluation_manager):
        self.aws = aws_manager
        self.evaluator = evaluation_manager
        self.config = load_config()

    # ==============================================================
    # 1. FLUSSO BULK INFERENCE (Valutazione su Test Set massivo)
    # ==============================================================
    def run_bulk(self, job_data, job_id):
        dataset = job_data['dataset']
        dataset_variant = job_data.get('dataset_variant', '1M')
        target_model = job_data['target_model']
        total_start_time = job_data.get('client_start_time', time.time())

        # 1. Analisi dei frammenti e Provisioning
        model_s3_uris = self.aws.count_model_parts(self.aws.bucket, dataset, target_model)
        num_workers = len(model_s3_uris)

        print(f" [BULK-INFER] Target model '{target_model}' chunked into {num_workers} parts. Scaling workers...")
        self.aws.scale_worker_infrastructure(num_workers)

        # 2. Recupero Stato (Fault Tolerance)
        historical_train_time, s3_inference_results, start_infer = self._recover_bulk_state(job_id, target_model)

        # 3. Task Generation (Fan-Out)
        self._dispatch_bulk_tasks(job_data, job_id, model_s3_uris, s3_inference_results)
        self.aws.update_job_state(job_id, set(), s3_inference_results, start_infer, True, historical_train_time, 0.0)

        # 4. Attesa dei Worker (Polling SQS)
        inference_time = self._wait_for_bulk_workers(job_id, num_workers, s3_inference_results, start_infer,
                                                     historical_train_time)

        # 5. Calcolo dei Pesi e Valutazione Finale (Delega all'Evaluator)
        num_trees, weights, strat = self._calculate_inference_weights(target_model, num_workers)
        self.evaluator.aggregate_and_evaluate(job_data, job_id, dataset, dataset_variant, s3_inference_results,
                                              num_workers, num_trees, weights, historical_train_time, inference_time,
                                              strat)

        # 6. Risposta al Client
        self._send_client_response(job_id, job_data.get('mode', 'bulk_infer'), time.time() - total_start_time)

    # ==============================================================
    # 2. FLUSSO REAL-TIME INFERENCE (Predizione Immediata su singola riga)
    # ==============================================================
    def run_realtime(self, job_data, job_id):
        dataset = job_data['dataset']
        target_model = job_data['target_model']
        tuple_data = job_data['tuple_data']
        task_type = job_data['task_type']
        total_start_time = job_data.get('client_start_time', time.time())

        # 1. Analisi dei frammenti e Provisioning
        model_s3_uris = self.aws.count_model_parts(self.aws.bucket, dataset, target_model)
        num_workers = len(model_s3_uris)

        print(f" [REAL-TIME] Target model '{target_model}' chunked into {num_workers} parts. Scaling workers...")
        start_provisioning = time.time()
        self.aws.scale_worker_infrastructure(num_workers)
        provisioning_time = time.time() - start_provisioning

        # 2. Invio dei dati da predire (Fan-Out)
        inference_pure_start = time.time()
        self._dispatch_realtime_tasks(job_data, job_id, dataset, model_s3_uris, tuple_data)

        # 3. Attesa votazioni dei Worker (Polling Veloce)
        total_received_votes, pure_inference_time = self._wait_for_realtime_workers(num_workers, inference_pure_start)

        # 4. Aggregazione dei voti (Majority o Mean)
        final_prediction, task_str = self._aggregate_realtime_results(task_type, total_received_votes)

        # 5. Stampa Latenze e Notifica
        total_run_time = time.time() - total_start_time
        print(f"\n{'=' * 60}\n REAL-TIME PREDICTION ({task_str}): {final_prediction:.2f}\n{'-' * 60}")
        print(f" AWS Provisioning Time (Cold Start):   {provisioning_time:.2f}s")
        print(f" Pure Inference Time (SQS + CPU):      {pure_inference_time:.2f}s")
        print(f" TOTAL Global System Latency:          {total_run_time:.2f}s\n{'=' * 60}\n")

        self._send_client_response(job_id, "infer", total_run_time, prediction=float(final_prediction),
                                   task_str=task_str)

    # ==============================================================
    # METODI PRIVATI DI SUPPORTO (Dettagli Implementativi)
    # ==============================================================

    def _recover_bulk_state(self, job_id, target_model):
        """Recupera il tempo storico di training del modello e lo stato attuale dell'inferenza."""
        _, _, _, _, historical_train_time, _ = self.aws.get_job_state(target_model)
        _, s3_inference_results, db_infer_start, _, _, _ = self.aws.get_job_state(job_id)
        start_infer = db_infer_start if db_infer_start else time.time()
        return historical_train_time, s3_inference_results, start_infer

    def _dispatch_bulk_tasks(self, job_data, job_id, model_s3_uris, s3_inference_results):
        """Invia i payload ai worker per valutare un intero dataset."""
        infer_queue = self.config["sqs_queues"]["infer_task"]
        for i, uri in enumerate(model_s3_uris):
            task_id = f"task_{i + 1}"
            if task_id not in s3_inference_results:
                infer_task = {
                    "job_id": job_id,
                    "task_id": task_id,
                    "dataset": job_data['dataset'],
                    "dataset_variant": job_data.get('dataset_variant', '1M'),
                    "test_dataset_uri": job_data['test_s3_url'],
                    "model_s3_uri": uri
                }
                self.aws.sqs_client.send_message(QueueUrl=infer_queue, MessageBody=json.dumps(infer_task))
                print(f" [INFER DISPATCH] Task {task_id} sent to inference queue.")

    def _wait_for_bulk_workers(self, job_id, num_workers, s3_inference_results, start_infer, historical_train_time):
        """Ascolta la coda delle risposte SQS finché tutti i worker non hanno finito il test set."""
        infer_resp_queue = self.config["sqs_queues"]["infer_response"]
        print("\n [EVENT LOOP] Master listening actively for Worker inference responses...\n")

        while len(s3_inference_results) < num_workers:
            res_infer = self.aws.sqs_client.receive_message(QueueUrl=infer_resp_queue, MaxNumberOfMessages=10,
                                                            WaitTimeSeconds=2)
            if 'Messages' in res_infer:
                for msg in res_infer['Messages']:
                    body = json.loads(msg['Body'])
                    task_id = body['task_id']

                    # Estrazione sicura dell'URI (gestisce vecchi dict o stringhe piane)
                    s3_votes_uri = body['s3_voti_uri']['valore'] if isinstance(body['s3_voti_uri'], dict) else body[
                        's3_voti_uri']

                    if task_id not in s3_inference_results:
                        s3_inference_results[task_id] = s3_votes_uri
                        print(
                            f" [ACK] Worker completed Bulk Inference for {task_id}! ({len(s3_inference_results)}/{num_workers})")
                        self.aws.update_job_state(job_id, set(), s3_inference_results, start_infer, True,
                                                  historical_train_time, time.time() - start_infer)

                    self.aws.sqs_client.delete_message(QueueUrl=infer_resp_queue, ReceiptHandle=msg['ReceiptHandle'])

        return time.time() - start_infer

    def _calculate_inference_weights(self, target_model, num_workers):
        """
        Versione intelligente: scansiona l'ID del modello, estrae il numero di alberi
        ovunque si trovi e identifica la strategia di addestramento.
        """
        # 1. Spezzettiamo l'ID in base agli underscore
        parts = target_model.split('_')

        # 2. RICERCA INTELLIGENTE DEL NUMERO DI ALBERI
        # Cerchiamo il primo pezzo che contenga la parola 'trees'
        trees_part = next((p for p in parts if 'trees' in p), None)

        if trees_part:
            try:
                # Puliamo la stringa (es: "100trees" -> "100") e convertiamo
                num_trees = int(trees_part.replace('trees', ''))
            except ValueError:
                print(f" [WARNING] Formato alberi non valido in '{trees_part}'. Uso fallback.")
                num_trees = num_workers * 25  # Fallback realistico (25 alberi per worker)
        else:
            print(f" [WARNING] Parola 'trees' non trovata nell'ID modello. Uso fallback.")
            num_trees = num_workers * 25

        # 3. IDENTIFICAZIONE STRATEGIA
        # Invece di un IF secco, cerchiamo la parola chiave nella stringa intera
        if "heterogeneous" in target_model:
            strat = "heterogeneous"
        else:
            strat = "homogeneous"

        # 4. CALCOLO DEI PESI (Bilanciamento del resto)
        # Questa logica è ottima: distribuisce il resto (remainder) sui primi worker
        weights = [
            math.floor(num_trees / num_workers) + (1 if i < (num_trees % num_workers) else 0)
            for i in range(num_workers)
        ]

        print(f" [RESOLVER] Modello: {strat.upper()} | Alberi Totali: {num_trees} | Pesi: {weights}")

        return num_trees, weights, strat

    def _dispatch_realtime_tasks(self, job_data, job_id, dataset, model_s3_uris, tuple_data):
        """Invia una singola riga di dati (tuple) ai worker per una predizione immediata."""
        infer_task_queue = self.config["sqs_queues"]["infer_task"]
        for i, uri in enumerate(model_s3_uris):
            task = {
                "job_id": job_id, "task_id": f"task_infer_rt_{i + 1}",
                "dataset": dataset, "model_s3_uri": uri, "tuple_data": tuple_data
            }
            self.aws.sqs_client.send_message(QueueUrl=infer_task_queue, MessageBody=json.dumps(task))

    def _wait_for_realtime_workers(self, num_workers, start_time):
        """Raccoglie le votazioni fulminee dalla memoria dei worker."""
        total_received_votes = []
        read_messages = 0
        infer_resp_queue = self.config["sqs_queues"]["infer_response"]

        while read_messages < num_workers:
            res = self.aws.sqs_client.receive_message(QueueUrl=infer_resp_queue, WaitTimeSeconds=2)
            if 'Messages' in res:
                for msg in res['Messages']:
                    body = json.loads(msg['Body'])
                    res_data = body['s3_voti_uri']

                    if isinstance(res_data, dict) and res_data.get("tipo") == "singolo":
                        worker_predictions = res_data['valore']
                        total_received_votes.extend(worker_predictions)
                        read_messages += 1
                        print(f"   -> Gathered {len(worker_predictions)} votes from worker.")

                    self.aws.sqs_client.delete_message(QueueUrl=infer_resp_queue, ReceiptHandle=msg['ReceiptHandle'])

        return total_received_votes, time.time() - start_time

    def _aggregate_realtime_results(self, task_type, total_received_votes):
        """Applica Majority Voting (Classificazione) o Simple Averaging (Regressione) ai voti in memoria."""
        if task_type == 'classification':
            final_prediction = max(set(total_received_votes), key=total_received_votes.count)
            votes_0 = total_received_votes.count(0)
            votes_1 = total_received_votes.count(1)
            task_str = "Classification (Majority Vote)"
            print(f" [POLL] Class 0: {votes_0} votes | Class 1: {votes_1} votes")
        else:
            final_prediction = sum(total_received_votes) / len(total_received_votes)
            task_str = "Regression (Mean)"

        return final_prediction, task_str

    def _send_client_response(self, job_id, mode, total_time, prediction=None, task_str=None):
        """Notifica il Client che il lavoro è terminato, allegando eventuali predizioni Real-Time."""
        client_response_queue = self.config.get("sqs_queues", {}).get("client_response")
        if client_response_queue:
            payload = {"job_id": job_id, "mode": mode, "total_time_sec": round(total_time, 2)}
            if prediction is not None:
                payload["prediction"] = prediction
                payload["task_type"] = task_str

            try:
                self.aws.sqs_client.send_message(QueueUrl=client_response_queue, MessageBody=json.dumps(payload))
                if prediction is not None:
                    print(f" [SUCCESS] Real-Time Prediction sent back to Client via SQS.")
            except Exception as e:
                print(f" [ERROR] Failed to send response to client: {e}")