from abc import ABC, abstractmethod


# Abstract base class for all Machine Learning models in the distributed system.
class BaseModel(ABC):

    # Trains the local model using the provided dataframe and hyperparameters.
    @abstractmethod
    def process_and_train(self, df, params):
        pass

    # Executes inference on the provided dataframe using the trained model and returns standardized output depending on the task type.
    @abstractmethod
    def process_and_predict(self, rf_model, df):    
        pass
