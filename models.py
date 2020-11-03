import numpy as np
import pandas as pd
import pickle
from customtypes import PredItemType, DataItemType


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

    def prediction(self, data: DataItemType) -> PredItemType:
        input = []
        for d in data:
            if d[0] != "name":
                input.append(d[1])
        inp: np.ndarray = np.array(input)
        inp = np.reshape(inp, (1, -1))
        df = pd.DataFrame(
            inp, columns=["dered_i", "dered_z", "dered_u", "dered_g", "dered_r", "extinction_r", "run", "camCol", "field", "obj", "photoz", "ra", "dec"])
        return self.model.predict(df)
