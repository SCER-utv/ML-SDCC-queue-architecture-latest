import sys
import datetime

# handles all command-line interactions with the user
class CLI:

    def __init__(self, config):
        self.config = config
        self.datasets_metadata = config.get("datasets_metadata", {})

    # clears the terminal screen
    def clear_screen(self):
        print("\n" * 2)

    # displays the welcome banner
    def show_welcome(self):
        self.clear_screen()
        print("=" * 60)
        print("  DISTRIBUTED RANDOM FOREST - CLI CLIENT ")
        print("=" * 60)

    # prompts the user to select the primary operation mode
    def prompt_operation_mode(self):
        print("\nSelect Operation Mode:")
        print("  1)  Distributed Training (Training Only)")
        print("  2)  End-to-End Pipeline (Train + Auto-Evaluate)")
        print("  3)  Bulk Inference (Test Set Evaluation)")
        print("  4)  Real-Time Inference (Single Prediction)")
        print("  5)  Download Aggregated Model")

        while True:
            mode_choice = input("\n Enter 1, 2, 3, 4 or 5: ").strip()
            if mode_choice == '1': return 'train'
            if mode_choice == '2': return 'train_and_infer'
            if mode_choice == '3': return 'bulk_infer'
            if mode_choice == '4': return 'infer'
            if mode_choice == '5': return 'download'
            print(" Invalid choice. Please try again.")

    # guides the user through dataset selection (golden standard or custom s3 url)
    def prompt_dataset_selection(self, mode):
        print("\n" + "-" * 40)
        print(" Select Dataset Source:")
        print("  1) Choose from predefined list (Golden Standard)")
        print("  2) Provide a Custom S3 URL")

        while True:
            choice = input("\n Enter 1 or 2: ").strip()
            if choice in ['1', '2']: break
            print(" Invalid choice. Please try again.")

        dataset_info = {
            "name": None, "variant": None, "train_url": None, "test_url": None,
            "needs_split": False, "target_col": None, "task_type": None, "is_custom": False
        }

        # helper function to validate s3 uri input
        def get_s3_input(prompt_text):
            while True:
                url = input(prompt_text).strip()
                if url.startswith("s3://") and url.endswith(".csv"):
                    return url
                print(" Invalid format. Must start with 's3://' and end with '.csv'.")

        if choice == '1':
            # discovery datasets
            dataset_info["is_custom"] = False
            available_datasets = list(self.datasets_metadata.keys())
            if not available_datasets:
                print(" [ERROR] No datasets found in config metadata!")
                sys.exit(1)

            dataset_map = {str(i): name for i, name in enumerate(available_datasets, start=1)}
            for key, name in dataset_map.items():
                ds_type = self.datasets_metadata[name][list(self.datasets_metadata[name].keys())[0]]["type"]
                print(f"  {key}) {name.capitalize()} ({ds_type.capitalize()})")

            while True:
                ds_choice = input(f"\n Select dataset [1-{len(available_datasets)}]: ").strip()
                if ds_choice in dataset_map:
                    dataset_info["name"] = dataset_map[ds_choice]
                    break
                print(" Invalid dataset selection.")

            print(f"\n Select Dataset Variant for '{dataset_info['name'].upper()}':")
            available_variants = list(self.datasets_metadata[dataset_info["name"]].keys())
            variant_map = {str(i): var for i, var in enumerate(available_variants, start=1)}
            for key, var in variant_map.items():
                print(f"  {key}) {var}")

            while True:
                var_choice = input(f"\n Select variant [1-{len(available_variants)}]: ").strip()
                if var_choice in variant_map:
                    dataset_info["variant"] = variant_map[var_choice]
                    break
                print(" Invalid variant selection.")

            meta = self.datasets_metadata[dataset_info["name"]][dataset_info["variant"]]
            dataset_info["task_type"] = meta.get("type", "classification")
            dataset_info["target_col"] = meta.get("target", "Label")

            if mode == 'train_and_infer':
                print("\n How should we handle this Predefined Dataset?")
                print("  1) Auto-Split the full dataset in Train/test (if test & train do not exist already)")
                print("  2) Use pre-existing Train/Test splits")
                while True:
                    ans = input(" Enter 1 or 2: ").strip()
                    if ans in ['1', '2']:
                        dataset_info["needs_split"] = (ans == '1')
                        break
                    print(" Invalid choice.")

        else:
            # custom datasets
            print("\n [CUSTOM DATASET]")
            dataset_info["is_custom"] = True
            dataset_info["name"] = "custom"
            dataset_info["variant"] = "user_provided"

            if mode == 'train':
                dataset_info["train_url"] = get_s3_input(" Enter the S3 URL of the TRAINING Dataset: ")

            elif mode == 'bulk_infer':
                dataset_info["test_url"] = get_s3_input(" Enter the S3 URL of the TEST Dataset: ")

            elif mode == 'train_and_infer':
                print("\n How are your datasets organized?")
                print("  1) Single Full Dataset (Auto-Split will be executed if Train and Test do not exist in the experiment folder)")
                print("  2) Two Separate Datasets (Train file & Test file already existing)")
                while True:
                    ans = input(" Enter 1 or 2: ").strip()
                    if ans in ['1', '2']: break
                    print(" Invalid choice.")

                if ans == '1':
                    dataset_info["train_url"] = get_s3_input(" Enter the FULL Dataset S3 URL to split: ")
                    dataset_info["needs_split"] = True
                else:
                    dataset_info["train_url"] = get_s3_input(" Enter the TRAINING Dataset S3 URL: ")
                    dataset_info["test_url"] = get_s3_input(" Enter the TEST Dataset S3 URL: ")
                    dataset_info["needs_split"] = False

            elif mode == 'infer':
                dataset_info["train_url"] = get_s3_input(" Enter the S3 URL of the dataset (used to extract feature names): ")

            dataset_info["target_col"] = input(" Enter the EXACT name of the Target Column to predict (e.g., Label): ").strip()

            print("\n Specify the ML Task Type for this dataset:")
            print("  1) Classification\n  2) Regression")
            while True:
                task_choice = input(" Enter 1 or 2: ").strip()
                if task_choice in ['1', '2']:
                    dataset_info["task_type"] = "classification" if task_choice == '1' else "regression"
                    break
                print(" Invalid choice.")

        return dataset_info

    # prompts for a custom experiment name to organize s3 outputs
    def prompt_experiment_name(self):
        print("\n" + "-" * 40)
        print(" Experiment Configuration (Custom Dataset)")
        print(" By inserting a experiment name, Train/Test files will be saved for future executions, ")
        print("to guarantee constant benchmarks.")
        while True:
            exp_name = input("\n experiment name (press Enter to remain isolated): ").strip()
            if not exp_name:
                return None

            sanitized_name = exp_name.replace(" ", "-").replace("_", "-").lower()
            # validate: alphanumeric and dash/underscore only for safe s3 keys
            if all(c.isalnum() or c == "-" for c in sanitized_name):
                return sanitized_name
            print(" [ERROR] Use only letters, numbers, '-' or ' ' (blank space will be converted to -.")

    # gathers cluster settings and machine learning hyperparameters
    def prompt_cluster_config(self, dataset_info):
        print("\n" + "-" * 40)
        print(f"  Cluster Configuration for: {dataset_info['name'].upper()}({dataset_info['variant']})")

        config_data = {}
        while True:
            try:
                config_data['workers'] = int(input(" Enter number of Workers (1-8): "))
                config_data['trees'] = int(input(" Enter TOTAL number of Trees (e.g., 100) (if you want to use pre-tuned hyperparams, insert a valid num of trees, otherwise closest configuration will we used: "))
                if config_data['workers'] <= 0 or config_data['trees'] <= 0:
                    print(" [ERROR] Values must be greater than zero.")
                elif config_data['workers'] > 8:
                    print(" [ERROR] The maximum number of workers allowed is 8. Please try again.")
                else:
                    break
            except ValueError:
                print(" Invalid input. Please enter integers only.")

        print("\n Select Training Strategy:")
        print("  1) Homogeneous  [Same parameters for all workers]")
        print("  2) Heterogeneous [Different parameters per worker, variance boosting]")
        while True:
            strat_choice = input(" Enter 1 or 2: ").strip()
            if strat_choice in ['1', '2']:
                config_data['strategy'] = "homogeneous" if strat_choice == '1' else "heterogeneous"
                break
            print(" Invalid choice.")

        if dataset_info['is_custom'] and config_data['strategy'] == "heterogeneous":
            print("\n [INFO] Heterogeneous strategy requires different parameters for each worker.")
            print(" Forcing Manual Configuration...")
            hyper_source = '2'
        else:
            print("\n Select Hyperparameter Source:")
            if dataset_info['is_custom']:
                print("  1) Default Generic Parameters for fast execution (Standard Scikit-Learn)")
                print("  2) Manual Configuration")
            else:
                print("  1) Golden Standard (Auto-optimized per dataset)")
                print("  2) Manual Configuration")

            while True:
                hyper_source = input(" Enter 1 or 2: ").strip()
                if hyper_source in ['1', '2']:
                    break
                print(" Invalid choice.")


        config_data['custom_hyperparams'] = None

        if hyper_source == '2':
            print("\n [MANUAL HYPERPARAMETERS CONFIGURATION]")
            config_data['custom_hyperparams'] = []
            iterations = 1 if config_data['strategy'] == "homogeneous" else config_data['workers']

            for w in range(iterations):
                if config_data['strategy'] == "heterogeneous":
                    print(f"\n --- Configuring Worker {w + 1}/{config_data['workers']} ---")
                else:
                    print("\n --- Configuring Global Parameters ---")

                # MAX DEPTH (int or None)
                raw_depth = input(" Max Depth (int, or blank for None): ").strip()
                max_depth = int(raw_depth) if raw_depth.isdigit() else None

                # MIN SAMPLES SPLIT (int or float)
                raw_split = input(" Min Samples Split (int or float, default: 2): ").strip()
                try:
                    min_samples_split = float(raw_split) if '.' in raw_split else int(raw_split)
                except ValueError:
                    min_samples_split = 2

                # MIN SAMPLES LEAF (int or float)
                raw_leaf = input(" Min Samples Leaf (int or float, default: 1): ").strip()
                try:
                    min_samples_leaf = float(raw_leaf) if '.' in raw_leaf else int(raw_leaf)
                except ValueError:
                    min_samples_leaf = 1

                # MAX FEATURES ('sqrt', 'log2', float, or None)
                raw_features = input(
                    " Max Features ['sqrt', 'log2', float < 1.0, or blank for None] (Default: sqrt): ").strip()
                if not raw_features or raw_features == "sqrt":
                    max_features = "sqrt"
                elif raw_features == "log2":
                    max_features = "log2"
                elif raw_features.lower() == "none":
                    max_features = None
                else:
                    try:
                        max_features = float(raw_features)
                    except ValueError:
                        max_features = "sqrt"

                # MAX SAMPLES (float or None)
                raw_samples = input(" Max Samples per Tree [0.1 - 1.0, or blank for None] (Default: 1.0): ").strip()
                if raw_samples.lower() == "none":
                    max_samples = None
                else:
                    try:
                        max_samples = float(raw_samples)
                    except ValueError:
                        max_samples = 1.0

                criterion = input(" Criterion [gini, entropy, squared_error] (Leave blank for default): ").strip()
                if not criterion:
                    criterion = "gini" if "classification" in dataset_info['task_type'].lower() else "squared_error"

                class_weight = None
                if "classification" in dataset_info['task_type'].lower():
                    raw_cw = input(" Class Weight [balanced, balanced_subsample] (Leave blank for None): ").strip()
                    if raw_cw in ["balanced", "balanced_subsample"]:
                        class_weight = raw_cw

                worker_params = {
                    "max_depth": max_depth, "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf, "max_features": max_features,
                    "max_samples": max_samples, "criterion": criterion,
                    "class_weight": class_weight, "n_jobs": -1
                }
                config_data['custom_hyperparams'].append(worker_params)

            if config_data['strategy'] == "homogeneous":
                config_data['custom_hyperparams'] = config_data['custom_hyperparams'] * config_data['workers']

        return config_data

    # allows the user to select an existing distributed model from s3 or paste an id
    def prompt_model_selection(self, aws_manager, dataset_info):
        print("\n" + "-" * 40)
        print(" Select Target Model ID:")
        print("  1) Paste a specific Model ID")
        print(f"  2) Scan S3 to select a model for '{dataset_info['name']}' ({dataset_info['variant']})")

        while True:
            sel_method = input("\n Enter 1 or 2: ").strip()
            if sel_method in ['1', '2']: break
            print(" Invalid choice.")

        if sel_method == '1':
            while True:
                target_model = input("\n Paste the exact Model ID (e.g., job_taxi_...): ").strip()
                if target_model.startswith("job_") or target_model.startswith("rf_"):
                    return target_model
                print(" Invalid ID format. It should start with 'job_'")
        else:
            print(f"\n [SEARCH] Scanning S3 for saved '{dataset_info['name']}' models...")
            all_models = aws_manager.list_available_models(dataset_info['name'])
            models = [m for m in all_models if f"_{dataset_info['variant']}_" in m]

            if not models:
                print(
                    f"\n [ERROR] No trained models found for '{dataset_info['name']}' (Variant: {dataset_info['variant']}). Run a training job first!")
                sys.exit(0)

            print("\n=== AVAILABLE MODELS ===")
            for i, m in enumerate(models):
                try:
                    parts = m.split('_')
                    trees_count = next((p.replace('trees', '') for p in parts if 'trees' in p), "?")
                    workers_count = next((p.replace('workers', '') for p in parts if 'workers' in p), "?")
                    strat_label = "HOMO" if "homogeneous" in m else ("HETE" if "heterogeneous" in m else "N/A ")

                    date_formatted, time_formatted = "????/??/??", "??:??:??"

                    # CASO 1: Vecchio formato con Data e Ora separati (se ne hai ancora su S3)
                    if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
                        raw_date, raw_time = parts[-2], parts[-1]
                        if len(raw_date) == 8 and len(raw_time) == 6:
                            date_formatted = f"{raw_date[6:8]}/{raw_date[4:6]}/{raw_date[0:4]}"
                            time_formatted = f"{raw_time[0:2]}:{raw_time[2:4]}:{raw_time[4:6]}"
                        else:
                            date_formatted, time_formatted = raw_date, raw_time

                    # CASO 2: Nuovo formato con Unix Timestamp singolo (es. 1776195424)
                    elif len(parts) >= 1 and parts[-1].isdigit():
                        try:
                            timestamp = int(parts[-1])
                            # Converte i secondi in un oggetto datetime
                            dt = datetime.datetime.fromtimestamp(timestamp)
                            date_formatted = dt.strftime("%d/%m/%Y")
                            time_formatted = dt.strftime("%H:%M:%S")
                        except Exception:
                            pass
                    print(
                        f"  [{i}]  Trees: {trees_count:<4} | Workers: {workers_count:<2} | Strat: {strat_label} | Date: {date_formatted} {time_formatted}  (ID: {m})")
                except Exception:
                    print(f"  [{i}] {m}")

            while True:
                try:
                    model_choice = int(input(f"\n Select Model ID [0-{len(models) - 1}]: "))
                    if 0 <= model_choice < len(models):
                        return models[model_choice]
                    print(" Invalid ID selected.")
                except ValueError:
                    print(" Please enter a valid number.")

    # prompts for manual data entry of features to perform real-time inference
    def prompt_realtime_input(self, aws_manager, dataset_s3_key, dataset_info):
        feature_names = aws_manager.get_feature_names_from_s3(dataset_s3_key, target_column=dataset_info['target_col'])

        required_features = len(feature_names) if feature_names else (
            self.datasets_metadata.get(dataset_info['name'], {}).get(dataset_info['variant'], {}).get("features", 0)
        )

        print("\n" + "-" * 40)
        print(" Real-Time Prediction Input")
        if required_features > 0:
            print(f" WARNING: Dataset expects EXACTLY {required_features} features!")
        if feature_names:
            print(f"\n Expected layout: \n {', '.join(feature_names)}")

        while True:
            prompt_text = f" Enter {required_features} comma-separated values: " if required_features > 0 else " Enter the comma-separated values: "
            raw_tuple = input(prompt_text).strip()
            try:
                tuple_data = [float(x.strip()) for x in raw_tuple.split(',')]
                if required_features == 0 or len(tuple_data) == required_features:
                    return tuple_data
                print(f" [ERROR] Expected {required_features} values, got {len(tuple_data)}.")
            except ValueError:
                print(" [ERROR] Formatting error. Use numbers only (e.g., 10.5, 3).")