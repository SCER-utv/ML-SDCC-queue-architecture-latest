# Distributed Random Forest on AWS

This project implements a **Cloud-Native** architecture for the distributed training and inference of Machine Learning models (Random Forest). Based on the **Master-Worker** pattern, the system leverages AWS managed services (SQS, S3, EC2 Auto Scaling) to ensure horizontal scalability, fault tolerance, and optimized memory management (Zero-Waste RAM) on large-scale datasets.

---

## Architecture and System Components

The system is built on a highly decoupled microservices architecture. Communication between components occurs exclusively via asynchronous message exchange over **Amazon SQS**, while heavy data (datasets and model weights) are stored in **Amazon S3**.

![Architecture Diagram](link_to_your_architecture_image_here)

### 1. The Client (User Interface)
The `client.py` file exposes an interactive CLI that guides the user through configuring the cluster and the Machine Learning job. Its primary purpose is to abstract the underlying cloud infrastructure complexity from the end user.

* **Thin Payload Contract:** Upon completing the configuration, the Client does not process any raw data. Instead, it builds a compact JSON dictionary (the *Thin Payload*) containing the user's intentions (number of trees, compute nodes, custom or gold standard hyperparameters, target column). This contract is dispatched to the Master node via SQS.
* **Dynamic Inference Management:** During Real-Time inference requests, the Client fetches the dataset header "on-the-fly" from S3 to validate user input, ensuring the provided tuple perfectly matches the format expected by the distributed model.

### 2. The Master Node (The Orchestrator)
The `master.py` file acts as the "brain" of the system. It is a persistent service constantly listening (via 20-second *Long Polling*) to the main Client queue. Upon receiving a *Thin Payload*, it triggers the following mechanisms:

* **Security & Path Resolution:** Through the `resolve_paths` function, the Master translates the Client's intentions into concrete and secure S3 URLs. If the user selects a "Gold Standard" dataset, the Master automatically injects the official paths retrieved from `config.json`. If a "Custom" dataset is used, the Master isolates the data and evaluation metrics within a unique `experiments/` folder, preventing collisions or unauthorized data manipulation by the Workers.
* **Master Heartbeat:** Since Machine Learning jobs can take tens of minutes, the Master spawns a dedicated background thread (`extend_client_sqs_visibility`) that cyclically renews the SQS message visibility for the Client's request (adding 60 seconds per tick). This prevents duplicate Master nodes from accidentally picking up the same job.
* **Dynamic Routing:** Depending on the requested mode (`train`, `bulk_infer`, `infer`), the Master routes the job to the specific pipeline orchestrator (e.g., `TrainingPipeline` or `InferencePipeline`), which handles the *Fan-Out* of micro-tasks to the Worker queues.

### 3. The Worker Node (The Compute Engine)
The `worker.py` file represents the pure computational node. It is designed to be completely *stateless* and horizontally scalable via EC2 Auto Scaling Groups.

* **Priority Polling:** The main event loop implements a strict priority queue system. The Worker always polls the Training queue (`q_train_in`) first. Only if no training tasks are available does it proceed to check the Inference queue (`q_infer_in`). This guarantees that bulk inference tasks never block the training pipeline of a new model.
* **Resilience & Immediate NACK:** Every task processed by the Worker (both `TrainingHandler` and `InferenceHandler`) is wrapped in `try/except` blocks and monitored by an isolated Heartbeat thread. If a Worker runs out of memory (OOM) or encounters a logical exception, it catches the error and executes an `aws.release_message` (setting the *VisibilityTimeout* to 0). This causes the task to instantly reappear on the SQS queue, allowing a healthy node to pick it up without delay.

---

## Data Management (Gold Standard vs. Custom)

The system supports two distinct logical flows for dataset ingestion, handled seamlessly by the architecture:

* **Gold Standard (Benchmark):** Leveraging metadata in the configuration file, the system skips data-split phases by utilizing pre-partitioned files on S3. It allows the user to bypass manual tuning by injecting hyperparameter sets that have been thoroughly tested and optimized to maximize performance (Accuracy/RMSE) for specific forest sizes.
* **Custom Datasets (User Upload):** By simply providing a raw `s3://` URL, the Master performs a Train/Test split in *Streaming Line-by-Line* mode. This prevents the entire dataset from being loaded into RAM, allowing the Master (often a lightweight T2 instance) to process multi-gigabyte datasets safely. Splits and metrics are grouped under the user-defined *Experiment Name* to ensure strict academic reproducibility.

---
