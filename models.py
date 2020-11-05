import pickle
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator

from customtypes import DataItemType


class Model:
    def __init__(self) -> None:
        self.names = ['AdaBoost', 'GaussianNB', 'GradientBoosting',
                      'LogisticRegression', 'MLP', 'RandomForest', 'SVC', 'XGBoost']
        self.classifiers: Dict[str, BaseEstimator] = {}
        self.load_models()
        self.model: BaseEstimator = None
        self.classes: Dict[int, str] = {0: "Star", 1: "Galaxy", 2: "Neither"}

    def load_models(self):
        for name in self.names:
            self.classifiers[name] = pickle.load(open(f"models/{name}", "rb+"))

    def select(self, model_name: str):
        if model_name not in self.names:
            raise IndexError
        self.model = self.classifiers[model_name]

    def prediction(self, data: DataItemType) -> str:
        input = []
        for d in data:
            if d[0] != "name":
                input.append(d[1])
        inp: np.ndarray = np.array(input)
        inp = np.reshape(inp, (1, -1))
        df = pd.DataFrame(
            inp, columns=["dered_i", "dered_z", "dered_u", "dered_g", "dered_r", "extinction_r", "run", "camCol", "field", "obj", "photoz", "ra", "dec"])
        pred = self.model.predict(df)
        print(pred[0])
        return self.classes[pred[0]]

    def test(self, name: str, data: pd.DataFrame):
        self.select(name)
        X = data.drop(['aclass'], axis=1)
        y = data['aclass']
        preds = self.model.predict(X)
        return confusion_matrix(preds, y)
