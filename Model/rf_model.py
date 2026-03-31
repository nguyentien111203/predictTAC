import numpy as np
from sklearn.ensemble import RandomForestClassifier
from config import RF_PARAMS

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestClassifier(**RF_PARAMS)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_top_k(self, X_input, k=3):
        if not hasattr(X_input, "columns"):
            X_input = X_input.to_frame().T

        probs = self.model.predict_proba(X_input)[0]
        top_k_idx = np.argsort(probs)[-k:][::-1]
        return list(top_k_idx)