from src.model.classification_model import ClassificationModel
from src.model.regression_model import RegressionModel
from src.utils.config import load_config


# dynamically instantiates the correct machine learning model based on dataset metadata
class ModelFactory:

    @staticmethod
    # retrieves dataset configuration and returns the appropriate classification or regression model instance
    def get_model(dataset_name: str):

        # load dataset metadata directly from the configuration file
        config = load_config()
        metadata = config.get("dataset_registry", {})

        if dataset_name not in metadata:
            raise ValueError(f"Dataset '{dataset_name}' not found in config.json! Add before proceed.")

        # extract the machine learning task type and target column name
        ds_info = metadata[dataset_name]
        task_type = ds_info["type"]
        target_col = ds_info["target"]

        # return the corresponding model instance based on the task type
        if task_type == 'classification':
            return ClassificationModel(target_column=target_col)
        elif task_type == 'regression':
            return RegressionModel(target_column=target_col)
        else:
            raise ValueError(f"Task type '{task_type}' not supported. Use 'classification' or 'regression'.")