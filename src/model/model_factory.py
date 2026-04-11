from src.model.classification_model import ClassificationModel
from src.model.regression_model import RegressionModel
from src.utils.config import load_config

# IL FACTORY TOTALMENTE DINAMICO ---
class ModelFactory:
    @staticmethod
    def get_model(dataset_name: str):
        
        # Carica le info sui dataset direttamente dal config.json
        config = load_config()
        metadata = config.get("dataset_registry", {})

        if dataset_name not in metadata:
            raise ValueError(f"Dataset '{dataset_name}' non trovato nel config.json! Aggiungilo prima di procedere.")

        # Estrae tipo di ML e colonna target
        ds_info = metadata[dataset_name]
        task_type = ds_info["type"]
        target_col = ds_info["target"]

        # Ritorna l'istanza corretta
        if task_type == 'classification':
            return ClassificationModel(target_column=target_col)
        elif task_type == 'regression':
            return RegressionModel(target_column=target_col)
        else:
            raise ValueError(f"Tipo di task '{task_type}' non supportato. Usa 'classification' o 'regression'.")
