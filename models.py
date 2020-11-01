from typing import List
import numpy as np
import pickle
from customtypes import PredItemType


class Model:
    def __init__(self) -> None:
        self.names = ['AdaBoost', 'GaussianNB', 'GradientBoosting',
                      'LogisticRegression', 'MLP', 'RandomForest', 'SVC', 'XGBoost']
        self.classifiers: dict = {}
        self.load_models()
        self.model = None

    def load_models(self):
        for name in self.names:
            self.classifiers[name] = pickle.load(open(f"models/{name}", "rb+"))

    def select(self, model_name: str):
        if model_name not in self.names:
            raise IndexError
        self.model = self.classifiers[model_name]

    def prediction(self, data: List[float]) -> PredItemType:
        inp = np.array(data)
        return self.model.predict(inp)
