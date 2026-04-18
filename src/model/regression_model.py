import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.model.base_model import BaseModel


# handles training and prediction logic for regression tasks using random forest
class RegressionModel(BaseModel):

    # initializes the model with the specified target column
    def __init__(self, target_column):
        self.target_column = target_column
        self.task_type = 'regression'

    # extracts hyperparameters, prepares data, and trains the random forest regressor
    def process_and_train(self, df, params):
        # safely extract hyperparameters falling back to scikit-learn defaults
        n_estimators = params.get('trees', 100)
        max_depth = params.get('max_depth', None)
        max_features = params.get('max_features', 1.0)
        criterion = params.get('criterion', 'squared_error')
        random_state = params.get('seed', 42)

        # advanced hyperparameters for tuning (class_weight is not applicable for regression)
        min_samples_split = params.get('min_samples_split', 2)
        min_samples_leaf = params.get('min_samples_leaf', 1)
        max_samples = params.get('max_samples', 1.0)
        n_jobs = params.get('n_jobs', -1)

        print(
            f"-> Training {params['trees']} trees. (Depth: {params['max_depth']} | Max features: {params['max_features']} | Criterion: {params['criterion']})")

        # separate features and target variable
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # initialize and train the random forest regressor
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            criterion=criterion,
            random_state=random_state,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_samples=max_samples,
            n_jobs=n_jobs
        )

        rf.fit(X, y)
        return rf

    # executes prediction using the local forest and returns a 1d array containing the mean predictions of the trees
    def process_and_predict(self, rf_model, df):
        X = df.drop(columns=[self.target_column])

        # convert to pure numpy array to avoid sklearn feature names warning
        X_array = X.to_numpy(dtype=np.float32)

        # sanitize inputs by replacing inf, -inf, and nan with 0.0 to prevent predict crashes
        X_clean = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)

        # standard predict natively returns the mean of all estimators for regression
        predictions = rf_model.predict(X_clean)

        return predictions