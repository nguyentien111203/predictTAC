
DATA_PATH = "Data/ue_tac_prediction_dataset.csv"

FEATURES = [
    "prev_2_TAC",
    "prev_TAC",
    "current_TAC",
    "hour",
    "day_of_week",
    "is_weekend"
]

TARGET = "next_TAC"

TEST_SIZE = 0.2
RANDOM_STATE = 42

TOP_K = 3

# Random Forest config
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": RANDOM_STATE
}

# Save paths
MODEL_DIR = "saved_models/"
RF_MODEL_PATH = MODEL_DIR + "rf_model.pkl"
ENCODER_PATH = MODEL_DIR + "label_encoder.pkl"
RF_MODEL_PATH = MODEL_DIR + "rf_model_v1.pkl"

# Logging
LOG_FILE = "logs/pipeline.log"