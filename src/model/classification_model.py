import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.model.base_model import BaseModel


class ClassificationModel(BaseModel):
    def __init__(self, target_column):
        self.target_column = target_column
        self.task_type = 'classification'

    def process_and_train(self, df, params):
        # 1. Estrazione sicura dei parametri (usa i default di scikit-learn se mancano)
        n_estimators = params.get('trees', 100)
        max_depth = params.get('max_depth', None)
        max_features = params.get('max_features', 'sqrt')
        criterion = params.get('criterion', 'gini')
        random_state = params.get('seed', 42)

        # I nuovi parametri del Gold Standard!
        min_samples_split = params.get('min_samples_split', 2)
        min_samples_leaf = params.get('min_samples_leaf', 1)
        max_samples = params.get('max_samples', 1.0)
        class_weight = params.get('class_weight', None)
        n_jobs = params.get('n_jobs', -1)

        print(f" Training {n_estimators} trees (Depth: {max_depth} | Split: {min_samples_split} | Leaf: {min_samples_leaf} | Feat: {max_features})")
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            criterion=criterion,
            random_state=random_state,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_samples=max_samples,
            class_weight=class_weight,
            n_jobs=n_jobs
        )
        rf.fit(X, y)
        return rf

    # Gathers predictions from individual trees for majority voting aggregation and returns a 2D array of shape containing vote counts
    def process_and_predict(self, rf_model, df):

        X = df.drop(columns=[self.target_column])

        # Convert to pure Numpy array to avoid sklearn "X has feature names" warning
        X_array = X.to_numpy(dtype=np.float32)

        """VA FATTO NEL NOTEBOOK"""
        # Sanitize inputs: replace Inf, -Inf, and NaN with 0.0 to prevent crash during predict
        X_clean = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)

        # 1. Collect predictions from each individual tree
        all_predictions = np.array([tree.predict(X_clean) for tree in rf_model.estimators_])

        # 2. Count votes for class 0 and class 1
        votes_0 = np.sum(all_predictions == 0, axis=0)
        votes_1 = np.sum(all_predictions == 1, axis=0)

        # 3. Stack arrays into a 2D matrix
        votes_matrix = np.column_stack((votes_0, votes_1))

        return votes_matrix
        
