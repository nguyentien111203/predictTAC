import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from config import RF_PARAMS

class HybridModel:
    def __init__(self, markov_model):
        self.model = RandomForestClassifier(**RF_PARAMS)
        self.markov = markov_model

    def add_markov_feature(self, X, y=None):
        X = X.copy()

        probs = []
        for i in range(len(X)):
            current = X.iloc[i]["current_TAC"]

            # nếu training → dùng y thật
            if y is not None:
                next_tac = y.iloc[i]
                p = self.markov.get_prob(current, next_tac)
            else:
                p = 0  # khi predict không biết next

            probs.append(p)

        X["markov_prob"] = probs
        return X

    def fit(self, X, y):
        X_new = self.add_markov_feature(X, y)
        self.model.fit(X_new, y)

    def predict_top_k(self, x, k=3):
        import pandas as pd

        if isinstance(x, pd.Series):
            x = x.to_frame().T

        x = x.copy()

        # dùng max prob của Markov
        current = x.iloc[0]["current_TAC"]
        probs_dict = self.markov.transition_prob.get(current, {})

        if probs_dict:
            max_prob = max(probs_dict.values())
        else:
            max_prob = 0

        x["markov_prob"] = max_prob

        probs = self.model.predict_proba(x)[0]
        top_k_idx = np.argsort(probs)[-k:][::-1]
        return list(top_k_idx)