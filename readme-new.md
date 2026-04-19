# Distributed Random Forest on AWS 

This project implements a **Cloud-Native** architecture for the distributed training and inference of Random Forest machine learning models. It aims to get better performance in terms of execution time, compared to a system running on a single node, while keeping almost the same performance in terms of metrics evaluated.

*Project developed for ML+SDCC 2025/26*

---

## Architecture and System Components

The system is built on a highly decoupled architecture, based on master-worker pattern. Communication between components occurs exclusively via asynchronous message exchange over **Amazon SQS**, while heavy data (datasets and model weights) are stored in **Amazon S3**.

![Training architecture Diagram](images/training_architecture.drawio.pdf) 
![Inference architecture Diagram](images/inference_architecture.drawio.pdf) 

---

### 1. Client
The `client.py` file exposes an interactive CLI that guides the user through configuration of the cluster and the Machine Learning job. Its primary purpose is to abstract the underlying cloud infrastructure complexity from the end user.
* **Job Contract:** Upon completing the configuration, the client builds a JSON dictionary containing the user's intentions. This job contract is dispatched to the Master node via SQS.
* **Dynamic Inference Management:** During Real-Time inference requests, the Client fetches the dataset header "on-the-fly" from S3 to validate user input, ensuring the provided tuple perfectly matches the format expected by the distributed model.

---

### 2. Master Node
The `master.py` file acts as the orchestrator of the system. It is a persistent service constantly listening to the main Client queue. The architecture is logically divided into three main components: Core Orchestration, the Training Pipeline, and the Inference Pipeline.

#### A. Core Orchestration Mechanisms
Upon receiving a *Job*, the Master triggers foundational tasks before routing the payload:
* **Security & Path Resolution:** Translates the Client's intentions into concrete and secure S3 URLs, isolating Custom data experiments from pre-configured data experiments.
* **Master Heartbeat:** Spawns a background thread that renews the SQS message visibility for the Client's request, preventing duplicate Master nodes from picking up the same long-running ML job.

#### B. The Training Pipeline Workflow
When a distributed training job is triggered, the Master routes it to the `TrainingPipeline` and executes a strict orchestration sequence:
1. **Fault Tolerance & State Recovery:** Queries DynamoDB to check if the job is new or recovering from a previous Master crash, allowing the system to resume gracefully without restarting from scratch.
2. **Infrastructure Provisioning:** Dynamically updates the AWS Auto Scaling Group (ASG) capacity to match the exact number of workers requested.
3. **Zero-Waste Data Split:** If a custom dataset requires splitting, the Master streams it line-by-line from S3, probabilistically routing rows into Train/Test files directly back to S3 to bypass physical RAM limits.
4. **Mathematical Fan-Out:** Calculates the optimal workload distribution (trees and rows per worker), applies the specified hyperparameter strategy (Homogeneous/Heterogeneous), and dispatches individual JSON micro-tasks to the `train_task` SQS queue.
5. **Active Event Loop:** Continuously polls the `train_response` SQS queue for worker Acknowledgements (ACKs), updating the persistent DynamoDB state as `.joblib` artifacts are uploaded to S3.
6. **Closure:** Once all workers report success, it calculates execution metrics and notifies the Client, or seamlessly passes the baton to the `InferencePipeline` for End-to-End evaluation jobs.

#### C. The Inference Pipeline Workflow
If the job entails testing a model, the `InferencePipeline` takes over. It handles two completely different execution paths based on the requested operation mode:

**Path 1: Bulk Inference (Massive Test Set Evaluation)**
1. **Model Discovery & Provisioning:** Scans S3 to count how many `.joblib` chunks the target distributed model consists of, and scales the AWS Auto Scaling Group to match this exact number.
2. **State Recovery:** Retrieves historical training times and the current evaluation state from DynamoDB to ensure fault tolerance.
3. **Fan-Out Dispatch:** Dispatches targeted JSON payloads via SQS, instructing each worker to download a specific model chunk and evaluate the entire test dataset.
4. **Active Event Loop:** Actively polls the SQS response queue, waiting for workers to upload their temporary `.npy` prediction arrays to S3.
5. **Smart Weighted Aggregation:** Intelligently parses the target Model ID to extract the exact number of trees and the hyperparameter strategy used. It calculates precise weights for each worker and delegates the final aggregation (Majority Voting or Weighted Averaging) to the `EvaluationManager`.

**Path 2: Real-Time Inference (Ultra-Low Latency)**
1. **Broadcast Dispatch:** Bypasses heavy S3 dataset operations entirely. It embeds the user's single data tuple directly into the SQS payload and broadcasts it to all workers.
2. **Rapid In-Memory Polling:** Swiftly gathers the individual tree votes directly from the SQS response messages (avoiding S3 file I/O overhead to minimize latency).
3. **Consensus Aggregation:** Computes the final prediction instantly (Majority Vote for classification, Mean for regression) and routes the precise prediction and system latency metrics directly back to the Client's CLI.

---

### 3. Worker Node
The Worker Node represents the pure computational engine of the distributed system. It is designed to be completely **stateless**, meaning it holds no permanent data and can be spun up or terminated at any time by the AWS Auto Scaling Group without affecting the system's integrity. 

The Worker code is logically divided into three main components: the Main Event Loop, the Training Handler, and the Inference Handler.

#### A. The Main Event Loop & Resilience (`worker.py`)
This component acts as the entry point and task router for the worker instance. It continuously listens to the SQS queues and guarantees that the system remains responsive and fault-tolerant.
* **Priority Polling:** The worker does not treat all tasks equally. It implements a strict priority system where it polls the `train_task` queue first. Only if the training queue is completely empty will it check the `infer_task` queue. This guarantees that massive bulk inference evaluations never block the creation of a new model.
* **Dynamic Heartbeat:** As soon as a task is picked up, the worker spawns an isolated background thread (`start_heartbeat`). This thread automatically pings AWS SQS every 20 seconds to extend the `VisibilityTimeout` of the message. This tells the queue: *"I am still alive and actively working on this, do not assign it to anyone else."*
* **Fault Tolerance (Immediate NACK):** Every task execution is wrapped in a `try/except` block. If the worker encounters a critical error (such as an Out-Of-Memory crash or corrupted data), it catches the exception and immediately triggers `release_message`. This sets the SQS visibility timeout to 0, instantly returning the task to the queue so a healthy worker can pick it up without waiting for a timeout.

#### B. The Training Handler (`TrainingHandler`)
When the main loop routes a training task, this handler is invoked to build a specific portion of the overall Random Forest.
1. **Zero-Waste RAM Data Loading:** The handler receives specific instructions (`skip_rows`, `num_rows`) from the Master. Instead of downloading the entire massive dataset from S3, it leverages pandas' partial reading capabilities to fetch *only* the exact subset of rows it has been assigned, drastically reducing RAM consumption and network overhead.
2. **Dynamic Model Building (Custom Dataset):** It dynamically constructs a raw Scikit-Learn `RandomForestClassifier` or `RandomForestRegressor` using the hyperparameters passed in the SQS payload. It seamlessly identifies and isolates the user-defined target column.
3. **Dynamic Model Building (Preconfigured Dataset):** It delegates the setup to the `ModelFactory`, ensuring strict adherence to the benchmark rules.
4. **Artifact Serialization:** Once the sub-forest is trained (`rf.fit`), the handler serializes the model into a `.joblib` file, temporarily saves it to the local `/tmp/` directory, uploads it to S3, and notifies the Master via the `train_response` SQS queue.

#### C. The Inference Handler (`InferenceHandler`)
When an inference task is routed, this handler is responsible for utilizing a previously saved `.joblib` artifact to generate predictions. It handles execution paths similarly to the Master's logic:
1. **Model Fetching:** Regardless of the mode, the very first step is securely downloading the worker's assigned `.joblib` model chunk from S3 into its local environment and loading it into memory.
* **Path 1: Real-Time Inference:** If the SQS payload contains a `tuple_data` array (meaning a user requested an immediate prediction for a single row), the handler reshapes the data, feeds it to the local sub-forest, and instantly returns an array of the individual tree votes back to the Master.
* **Path 2: Bulk Inference (Memory-Safe Chunking):** If the payload requires evaluating an entire test dataset, the handler employs extreme memory conservation tactics. It streams the dataset from S3 in strictly sized blocks (e.g., `chunksize=500000` rows). After predicting a chunk, it explicitly invokes Python's Garbage Collector (`gc.collect()`) to flush the RAM before loading the next chunk. Finally, it concatenates all chunk predictions into a highly compressed `.npy` (Numpy) file, uploads it to S3, and sends the S3 URI back to the Master for final aggregation.

---

## System Capabilities & Operation Modes

The system abstracts the complexity of distributed computing through a highly interactive CLI, empowering users to orchestrate the entire lifecycle of both **Classification** and **Regression** models. Rather than focusing on infrastructure management, users can directly leverage four distinct, high-level operation modes:

### 1. Distributed Training (Training Only)
Allows users to train large-scale Random Forest models across a distributed cluster of EC2 instances. Instead of being bottlenecked by a single machine's computational limits, the system horizontally parallelizes the workload. The result is a powerful, globally distributed model consisting of multiple independent `.joblib` sub-forests safely stored on S3, ready for future use.

### 2. End-to-End Pipeline (Train + Auto-Evaluate)
A fully automated, continuous workflow designed specifically for rapid prototyping and academic benchmarking. It seamlessly chains the Distributed Training phase directly into the Bulk Inference phase. Users can submit a job and walk away; the system will build the distributed model, test it against a hold-out dataset, and immediately output comprehensive evaluation metrics (such as ROC-AUC, F1-Score, RMSE, and MAPE) in both CLI and persistent CSV/JSON logs.

### 3. Bulk Inference (Massive Test Set Evaluation)
Enables the evaluation of previously trained distributed models against massive, out-of-core test datasets. This mode is designed to bypass traditional memory constraints entirely. It allows users to point the system to a test file containing millions of rows and receive a highly accurate, globally aggregated evaluation without risking local or cluster-wide Out-Of-Memory crashes.

### 4. Real-Time Inference (Single Prediction)
Engineered for ad-hoc, on-the-fly predictions requiring ultra-low latency. Users can interactively input a single data point (tuple) directly through the CLI. The system safely guides the input using fetched dataset headers, broadcasts the query to the active compute cluster, and aggregates the consensus (Majority Vote or Mean) in less than a second. It is the perfect tool for instant model querying and interactive testing.

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
