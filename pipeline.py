import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from config import *
from Model.markov_model import MarkovModel
from Model.rf_model import RandomForestModel
from Model.hybrid_model import HybridModel
from Utils.evaluation import Evaluator
from Utils.logger import get_logger
from Utils.visualization import plot_cost, plot_accuracy_top1, plot_accuracy_top3

logger = get_logger()


class Pipeline:
    def __init__(self):
        self.le = LabelEncoder()
        self.markov = MarkovModel()
        self.rf = RandomForestModel()
        self.hybrid = HybridModel(self.markov)

    def load_data(self):
        logger.info("Loading data...")
        return pd.read_csv(DATA_PATH)

    def encode(self, df):
        logger.info("Encoding TAC...")

        all_tac = pd.concat([
            df["prev_2_TAC"],
            df["prev_TAC"],
            df["current_TAC"],
            df["next_TAC"]
        ])

        self.le.fit(all_tac)

        for col in ["prev_2_TAC", "prev_TAC", "current_TAC", "next_TAC"]:
            df[col] = self.le.transform(df[col])

        return df

    def split(self, df):
        logger.info("Splitting data...")

        X = df[FEATURES]  # RF dùng feature gốc
        y = df[TARGET]

        return train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

    def train(self, df, X_train, y_train):
        logger.info("Training models...")

        # Markov
        self.markov.fit(df)

        # RF (không dùng markov feature)
        self.rf.fit(X_train, y_train)

        # Hybrid (tự add markov feature bên trong)
        self.hybrid.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        logger.info("Evaluating models...")

        evaluator = Evaluator(self.markov, self.rf, self.hybrid)

        m_accu1, r_accu1, h_accu1, m_accu3, r_accu3, h_accu3 = evaluator.evaluate(X_test, y_test)
        plot_accuracy_top1(m_accu1, r_accu1, h_accu1)
        plot_accuracy_top3(m_accu3, r_accu3, h_accu3)

        m_cost, r_cost, h_cost = evaluator.paging_cost(X_test, y_test)
        plot_cost(m_cost, r_cost, h_cost)

        
    def save_models(self):
        logger.info("Saving models...")

        os.makedirs(MODEL_DIR, exist_ok=True)

        joblib.dump(self.rf.model, RF_MODEL_PATH)
        joblib.dump(self.hybrid.model, MODEL_DIR + "hybrid_model.pkl")
        joblib.dump(self.le, ENCODER_PATH)

        logger.info("Models saved successfully.")

    def run(self):
        df = self.load_data()
        df = self.encode(df)

        X_train, X_test, y_train, y_test = self.split(df)

        self.train(df, X_train, y_train)
        self.evaluate(X_test, y_test)

        self.save_models()