import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.model.base_model import BaseModel


# handles training and prediction logic for classification tasks using random forest
class ClassificationModel(BaseModel):

    # initializes the model with the specified target column
    def __init__(self, target_column):
        self.target_column = target_column
        self.task_type = 'classification'

    # extracts hyperparameters, prepares data, and trains the random forest classifier
    def process_and_train(self, df, params):
        # safely extract hyperparameters falling back to scikit-learn defaults
        n_estimators = params.get('trees', 100)
        max_depth = params.get('max_depth', None)
        max_features = params.get('max_features', 'sqrt')
        criterion = params.get('criterion', 'gini')
        random_state = params.get('seed', 42)

        # advanced hyperparameters for golden standard and custom tuning
        min_samples_split = params.get('min_samples_split', 2)
        min_samples_leaf = params.get('min_samples_leaf', 1)
        max_samples = params.get('max_samples', 1.0)
        class_weight = params.get('class_weight', None)
        n_jobs = params.get('n_jobs', -1)

        print(
            f" Training {n_estimators} trees (Depth: {max_depth} | Split: {min_samples_split} | Leaf: {min_samples_leaf} | Feat: {max_features})")

        # separate features and target variable
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # initialize and train the random forest classifier
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

    # gathers predictions from individual trees for majority voting aggregation returning a 2d vote count matrix
    def process_and_predict(self, rf_model, df):
        X = df.drop(columns=[self.target_column])

        # convert to pure numpy array to avoid sklearn feature names warning
        X_array = X.to_numpy(dtype=np.float32)

        # sanitize inputs by replacing inf, -inf, and nan with 0.0 to prevent predict crashes
        X_clean = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)

        # collect predictions from each individual tree
        all_predictions = np.array([tree.predict(X_clean) for tree in rf_model.estimators_])

        # count votes for class 0 and class 1
        votes_0 = np.sum(all_predictions == 0, axis=0)
        votes_1 = np.sum(all_predictions == 1, axis=0)

        # stack arrays into a 2d matrix
        votes_matrix = np.column_stack((votes_0, votes_1))

        return votes_matrix