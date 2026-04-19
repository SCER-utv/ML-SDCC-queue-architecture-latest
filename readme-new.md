# Distributed Random Forest on AWS 

This project implements a **Cloud-Native** architecture for the distributed training and inference of Random Forest machine learning models. It aims to get better performance in terms of execution time, compared to a system running on a single node, while keeping almost the same performance in terms of metrics evaluated.

*Project developed for ML+SDCC 2025/26*

---

## Architecture and System Components

The system is built on a highly decoupled architecture, based on master-worker pattern. Communication between components occurs exclusively via asynchronous message exchange over **Amazon SQS**, while heavy data (datasets and model weights) are stored in **Amazon S3**.

![Training architecture Diagram](images/training_architecture.drawio.pdf) 
![Inference architecture Diagram](images/inference_architecture.drawio.pdf) 

### 1. The Client (User Interface)
The `client.py` file exposes an interactive CLI that guides the user through configuration of the cluster and the Machine Learning job. Its primary purpose is to abstract the underlying cloud infrastructure complexity from the end user.
* **Job Contract:** Upon completing the configuration, the client builds a JSON dictionary containing the user's intentions. This job contract is dispatched to the Master node via SQS.
* **Dynamic Inference Management:** During Real-Time inference requests, the Client fetches the dataset header "on-the-fly" from S3 to validate user input, ensuring the provided tuple perfectly matches the format expected by the distributed model.

### 2. The Master Node (The Orchestrator)
The `master.py` file acts as the orchestrator of the system. It is a persistent service constantly listening to the main Client queue. Upon receiving a *Job*, it triggers:
* **Security & Path Resolution:** Translates the Client's intentions into concrete and secure S3 URLs, isolating Custom data experiments from pre-configured data experiments.
* **Master Heartbeat:** Spawns a background thread that renews the SQS message visibility for the Client's request, preventing duplicate Master nodes from picking up the same long-running ML job.
* **Pipeline handlers:** Depending on the requested mode, the Master routes the job to the specific pipeline handler (`TrainingPipeline` or `InferencePipeline`).
  * **The Training Pipeline Workflow:** When a distributed training job is triggered, the Master executes a strict orchestration sequence:
    1. **Fault Tolerance & State Recovery:** Queries DynamoDB to check if the job is new or recovering from a previous Master crash, allowing the system to resume gracefully without restarting from scratch.
    2. **Infrastructure Provisioning:** Dynamically updates the AWS Auto Scaling Group (ASG) capacity to match the exact number of workers requested.
    3. **Zero-Waste Data Split:** If a custom dataset requires splitting, the Master streams it line-by-line from S3, probabilistically routing rows into Train/Test files directly back to S3 to bypass physical RAM limits.
    4. **Mathematical Fan-Out:** Calculates the optimal workload distribution (trees and rows per worker), applies the specified hyperparameter strategy (Homogeneous/Heterogeneous), and dispatches individual JSON micro-tasks to the `train_task` SQS queue.
    5. **Active Event Loop:** Continuously polls the `train_response` SQS queue for worker Acknowledgements (ACKs), updating the persistent DynamoDB state as `.joblib` artifacts are uploaded to S3.
    6. **Closure:** Once all workers report success, it calculates execution metrics and notifies the Client, or seamlessly passes the baton to the `InferencePipeline` for End-to-End evaluation jobs.

### 3. The Worker Node (The Compute Engine)
The `worker.py` file represents the pure computational node, designed to be completely *stateless* and horizontally scalable via EC2 Auto Scaling Groups.
* **Priority Polling:** The main event loop implements a strict priority queue system. The Worker always polls the Training queue first. Only if no training tasks are available does it proceed to check the Inference queue, guaranteeing that bulk inference tasks never block the training of a new model.
* **Resilience & Immediate NACK:** Every task is wrapped in `try/except` blocks and monitored by an isolated Heartbeat thread. On an Out-of-Memory (OOM) error or logical exception, the worker catches it and executes an immediate release (`VisibilityTimeout=0`), allowing a healthy node to pick up the task instantly.

---

## System Capabilities & Operation Modes

The system abstracts the complexity of distributed computing through a highly interactive CLI, allowing users to orchestrate the entire lifecycle of a Machine Learning model. It supports both **Classification** and **Regression** tasks through four distinct operation modes:

### 1. Distributed Training
The system parallelizes the training of a Random Forest across multiple EC2 Worker nodes. 
* **Mathematical Fan-Out:** The Master dynamically calculates the optimal distribution of trees (`n_estimators` / `num_workers`), assigning the remainder to the first available nodes. 
* **Zero-Waste RAM:** Workers do not load the entire dataset. The Master instructs each worker on the exact row range (`skip_rows`, `num_rows`) to fetch from S3.
* **Artifact Generation:** Each worker trains its assigned sub-forest and serializes it as an independent `.joblib` artifact securely stored on S3.

### 2. End-to-End Pipeline (Train + Auto-Evaluate)
A continuous, fully automated workflow ideal for rapid prototyping and academic benchmarking.
* The system sequentially chains the Distributed Training phase and the Bulk Inference phase. 
* Once training is complete, the Master immediately tasks the workers to evaluate the newly created distributed model against a hold-out Test Set.
* The pipeline culminates with the Master calculating and persisting global evaluation metrics (e.g., ROC-AUC, F1-Score, RMSE, MAPE) in both CSV (for historical tracking) and JSON formats.

### 3. Bulk Inference (Massive Test Set Evaluation)
Designed to evaluate previously trained distributed models against massive datasets that exceed the memory capacity of a single machine.
* **Memory-Safe Chunking:** Workers stream the test dataset from S3 in strictly defined block sizes (e.g., 500,000 rows at a time). After predicting a chunk, aggressive garbage collection (`gc.collect()`) is enforced to prevent Out-Of-Memory (OOM) crashes.
* **Smart Aggregation:** Workers upload temporary `.npy` prediction arrays to S3. The Master downloads them and applies task-specific aggregation: **Majority Voting** for classification and **Weighted Averaging** (based on the exact number of trees each worker trained) for regression.

### 4. Real-Time Inference (Single Tuple Prediction)
Engineered for ultra-low latency scenarios where an immediate prediction is required for a single data point.
* The Client automatically fetches the feature headers from the S3 dataset to guide the user's input safely.
* The tuple is broadcasted to all Workers simultaneously. Workers keep their respective model chunks in fast RAM, predict the outcome, and instantly return their votes to the Master via SQS for sub-second aggregation.

---

## Dataset Management: Preconfigured vs. Custom Workflows

A core architectural strength of this system is its completely *Data-Agnostic* design. It seamlessly handles two distinct ingestion logic flows without requiring manual code changes:

### A. Preconfigured Datasets (Gold Standard Benchmarking)
Built-in support for known academic datasets (e.g., *Airlines* for classification, *Taxi* for regression) defined directly within the `config.json`.
* **Zero-Setup Routing:** S3 paths for Train and Test sets, task types, and target columns are automatically resolved by the Master. The data-split phase is entirely bypassed since the system relies on pre-partitioned, standardized files to guarantee exact benchmark reproducibility.
* **Auto-Optimized Hyperparameters:** The user can opt to use "Golden Standard" parameters. The system automatically injects pre-calculated, grid-searched hyperparameter configurations (e.g., `max_depth`, `min_samples_split`) specifically tuned for the chosen dataset and the specified forest size.

### B. Custom Datasets (Bring Your Own Data)
Users can provide any raw `.csv` dataset by simply pasting its `s3://` URL into the CLI. The system dynamically adapts to the new schema.
* **Streaming Auto-Split:** If the user provides a single monolithic file, the Master executes a Train/Test split. To prevent RAM saturation on the Master node, this split is performed in **Streaming Mode** (reading and writing line-by-line via `boto3`). 
* **Target Masking:** The user specifies the exact target column name. Workers dynamically detect and drop this column during the training and inference phases, preventing data leakage.
* **Experiment Isolation:** When working with Custom data, the user is prompted to assign an *Experiment Name*. The system creates a dedicated S3 directory (e.g., `experiments/my_custom_test/`). The auto-split datasets and the final metric logs are permanently saved here. This allows users to rerun different model configurations (e.g., changing the number of workers or trees) on the exact same data splits, ensuring scientifically valid comparisons.

---
