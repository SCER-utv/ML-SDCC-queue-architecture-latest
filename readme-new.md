# Distributed Random Forest on AWS 

This project implements a **Cloud-Native** architecture for the distributed training and inference of Random Forest machine learning models. It aims to get better performance in terms of execution time, compared to a system running on a single node, while keeping almost the same performance in terms of metrics evaluated.

*Project developed for ML+SDCC 2025/26*

---

## System Capabilities & Operation Modes

The system abstracts the complexity of distributed computing through a highly interactive CLI, empowering users to orchestrate the entire lifecycle of both **Classification** and **Regression** models. Rather than focusing on infrastructure management, users can directly leverage five distinct, high-level operation modes:

### 1. Distributed Training (Training Only)
Allows users to train large-scale Random Forest models across a distributed cluster of EC2 instances. Instead of being bottlenecked by a single machine's computational limits, the system horizontally parallelizes the workload. The result is a powerful, globally distributed model consisting of multiple independent `.joblib` sub-forests safely stored on S3, ready for future use.

### 2. End-to-End Pipeline (Train + Auto-Evaluate)
A fully automated, continuous workflow designed specifically for rapid prototyping and academic benchmarking. It seamlessly chains the Distributed Training phase directly into the Bulk Inference phase. Users can submit a job and walk away; the system will build the distributed model, test it against a hold-out dataset, and immediately output comprehensive evaluation metrics (such as ROC-AUC, F1-Score, RMSE, and MAPE) in both CLI and persistent CSV/JSON logs.

### 3. Bulk Inference (Massive Test Set Evaluation)
Enables the evaluation of previously trained distributed models against massive, out-of-core test datasets. This mode is designed to bypass traditional memory constraints entirely. It allows users to point the system to a test file containing millions of rows and receive a highly accurate, globally aggregated evaluation without risking local or cluster-wide Out-Of-Memory crashes.

### 4. Real-Time Inference (Single Prediction)
Engineered for ad-hoc, on-the-fly predictions requiring ultra-low latency. Users can interactively input a single data point (tuple) directly through the CLI. The system safely guides the input using fetched dataset headers, broadcasts the query to the active compute cluster, and aggregates the consensus (Majority Vote or Mean) in less than a second. It is the perfect tool for instant model querying and interactive testing.

### 5. Download & Merge Model (Local Export)
Allows users to export a globally trained distributed model to their local machine for offline use or standalone deployment. The Client automatically fetches all the individual `.joblib` sub-forests from S3, intelligently aggregates their estimators into a single, unified Scikit-Learn Random Forest model, and saves it locally as a `.pkl` file. This effectively bridges the gap between cloud-scale distributed training and edge/local inference.

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

## The Complete Configuration Flow (User Journey)
The `client.py` orchestrates job configuration through a dynamic, context-aware prompt system. Depending on the chosen operation mode and data source, the CLI alters its questions to gather only the necessary paths and prevents logical conflicts. 

Here is the step-by-step logical flow the user experiences:

**Step 1: The Primary Fork (Source Selection)**
The user is asked to choose the data source type:
* **Option 1 (Predefined):** The CLI dynamically lists available academic datasets (e.g., *Airlines*, *Taxi*) and their size variants from `config.json`. The system automatically infers the ML task and target column. If running an *End-to-End* pipeline, the user chooses whether to use pre-existing S3 splits or force a new Auto-Split.
* **Option 2 (Custom S3 URL):** The user enters the "Bring Your Own Data" mode.

**Step 2: Context-Aware Path Ingestion (Custom Data Only)**
Depending on the operation mode, the CLI prompts only for the required paths: a single `train_url` for training, a `test_url` for bulk inference, or both for an End-to-End pipeline. For Real-Time inference, it requests a dataset URL purely to extract the CSV headers for input validation.

**Step 3: Metadata & Isolation (Custom Data Only)**
The user explicitly defines the **Target Column** and the **Machine Learning Task** (Classification/Regression). For training or bulk evaluation, the CLI prompts for an **Experiment Name**, which the Master uses to safely isolate the generated S3 artifacts (`experiments/<experiment_name>/`) for reproducible A/B testing.

**Step 4: Cluster Sizing & Strategy (Training Only)**
The user configures the distributed infrastructure by inputting the number of **Workers** and the total number of **Trees**. Then, they select the training strategy:
* **Homogeneous:** Every worker trains its sub-forest using the exact same hyperparameters.
* **Heterogeneous:** Workers use different hyperparameters to maximize forest variance and prevent overfitting.

**Step 5: Smart Hyperparameter Routing**
The CLI dynamically adapts the hyperparameter options based on the dataset type and the chosen strategy:
* **Gold Standard Selected:** The CLI offers "Golden Standard (Auto-optimized)" parameters or Manual Configuration.
* **Custom Dataset + Homogeneous:** The CLI offers standard Scikit-Learn defaults or Manual Configuration.
* **Custom Dataset + Heterogeneous (Conflict Resolution):** The system detects a logical conflict (default parameters are identical, which defeats the purpose of a heterogeneous forest). It gracefully bypasses the menu and **forces the user into Manual Configuration**.

**Step 6: The Manual Configuration Loop**
If manual configuration is triggered, the user inputs values for `max_depth`, `min_samples_split`, `min_samples_leaf`, etc. 
* If the strategy is *Homogeneous*, the CLI prompts for these values only once and applies them globally. 
* If the strategy is *Heterogeneous*, the CLI loops through the prompts, asking for a unique hyperparameter configuration for *each* specific worker node.

**Step 7: The "Thin Payload" Dispatch**
Once the flow is complete, the CLI packages all the gathered configurations into a compact JSON dictionary. It attaches a unique `job_id`, dispatches it to the SQS queue, and waits for the Master's orchestration response.

---

## Advanced: Extending the Gold Standard Benchmarks

While the system supports completely custom datasets via the CLI, you may want to permanently integrate a new academic dataset into the "Preconfigured (Gold Standard)" list. This ensures that anyone using the system can run reproducible benchmarks with auto-optimized hyperparameters without manually configuring them each time.

Adding a new dataset requires updating the configuration files and adhering to a strict S3 directory structure.

### Step 1: Registering the Dataset Metadata
First, you must declare the dataset's basic properties in the main `config.json` file under the `dataset_registry` object. The Master node uses this to determine how to evaluate the model and which column to drop during training.

```json
  "dataset_registry": {
    "taxi": {
      "target": "Label",
      "type": "regression"
    },
    "airlines": {
      "target": "Label",
      "type": "classification"
    },
    "my_new_dataset": {
      "target": "MyTargetColumn",
      "type": "classification" 
    }
  }
```

### Step 2: Defining Homogeneous Hyperparameters
If a user selects the **Homogeneous** strategy, all workers will use the exact same parameters. Because tree depth and splitting rules often depend on the *total size of the forest*, the homogeneous configuration maps **Total Trees** to a single Scikit-Learn parameter dictionary.

Add your new dataset to the homogeneous JSON configuration file (e.g., `hyperparameters_homo.json`):

```json
{
  "my_new_dataset": {
    "50":  {"max_depth": 20, "min_samples_split": 50, "min_samples_leaf": 5, "max_features": "0.2", "criterion": "gini", "n_jobs": -1},
    "100": {"max_depth": 27, "min_samples_split": 50, "min_samples_leaf": 6, "max_features": "0.2", "criterion": "gini", "n_jobs": -1}
  }
}
```
*If a user requests 50 trees, every worker in the cluster will receive the exact dictionary assigned to the `"50"` key.*

### Step 3: Defining Heterogeneous Hyperparameters (Variance Boosting)
If a user selects the **Heterogeneous** strategy, each worker must receive a *different* set of hyperparameters to force the sub-forests to learn different patterns (Variance Boosting). Therefore, this configuration maps the **Number of Workers** to an **Array of dictionaries**.

Add your new dataset to the heterogeneous JSON configuration file (e.g., `hyperparameters_hetero.json`):

```json
{
  "my_new_dataset": {
    "1": [
      {"max_depth": 27, "min_samples_split": 50, "max_features": "0.234", "criterion": "gini", "n_jobs": -1}
    ],
    "2": [
      {"max_depth": 27, "min_samples_split": 50, "max_features": "0.234", "criterion": "gini", "n_jobs": -1},
      {"max_depth": 19, "min_samples_split": 20, "max_features": "0.300", "criterion": "gini", "n_jobs": -1}
    ],
    "3": [
      {"max_depth": 27, "min_samples_split": 50, "max_features": "0.234", "criterion": "gini", "n_jobs": -1},
      {"max_depth": 19, "min_samples_split": 20, "max_features": "0.300", "criterion": "gini", "n_jobs": -1},
      {"max_depth": 28, "min_samples_split": 60, "max_features": "0.200", "criterion": "entropy", "n_jobs": -1}
    ]
  }
}
```
**Important Design Rule:** The length of the array must perfectly match the dictionary key. If the key is `"3"` (meaning a 3-worker cluster), the array must contain exactly 3 distinct configuration dictionaries. The Master node will iterate through this array and assign index 0 to Worker 1, index 1 to Worker 2, and so on.

### Step 4: S3 Path Architecture & Auto-Discovery
The system does not require you to manually hardcode S3 URLs for every dataset variant. Instead, it features an **Auto-Discovery engine** built into the configuration loader (`config.py`). On startup, the Master node scans the S3 bucket and automatically maps the datasets based on strict naming conventions.

To successfully add your new dataset, adhere to the following S3 architecture:

#### 1. The Interim Directory (Raw Upload)
You must upload your full, monolithic `.csv` dataset into the `data/interim/` S3 prefix. 
The file name **must** follow the `<dataset_name>_<variant>.csv` convention. The system parses the string using the *last* underscore to separate the dataset's core name from its size/variant tag.

* Example: `s3://<bucket>/data/interim/airlines_1M.csv`

#### 2. S3 Select Header Validation
During Auto-Discovery, the Master node does not download these massive CSV files to check if they are valid. Instead, it uses **AWS S3 Select** to execute an SQL query directly on the bucket (`SELECT * FROM S3Object LIMIT 1`). This efficiently extracts *only* the CSV header row to verify that the `target` column (declared in `config.json`) actually exists in the file. If the target column is missing, the variant is safely ignored.

#### 3. Automated Processed Path Generation
If the dataset passes the header validation, the Auto-Discovery engine automatically computes and maps the future paths for the Train and Test splits. These processed files are strictly routed to the `data/processed/` prefix to prevent contamination of the raw data.

If a user requests the End-to-End Pipeline with the "Auto-Split" option on your new `my_new_dataset_optimized.csv`, the Master will stream the split directly into these resolved S3 keys:

```text
s3://<bucket>/
├── data/
│   ├── interim/
│   │   └── my_new_dataset_optimized.csv          <-- (You upload this)
│   │
│   └── processed/
│       └── my_new_dataset/
│           ├── my_new_dataset_optimized_train.csv <-- (Master generates this)
│           └── my_new_dataset_optimized_test.csv  <-- (Master generates this)
```

Because of this automated mapping, the interactive CLI will immediately display `My_new_dataset` and the `optimized` variant as selectable options without any further code modifications.
