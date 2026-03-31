import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# -----------------------------
# CONFIG
# -----------------------------
NUM_UE = 1000
DAYS = 7
TAC_LIST = ["T1", "T2", "T3", "T4", "T5", "T6"]

# TAC graph
TAC_GRAPH = {
    "T1": ["T2"],
    "T2": ["T1", "T3", "T4"],
    "T3": ["T2", "T5"],
    "T4": ["T2", "T6"],
    "T5": ["T3"],
    "T6": ["T4"]
}

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def move_to_next(current):
    return random.choice(TAC_GRAPH[current])

def generate_path(start, end):
    path = [start]
    current = start
    visited = set()

    while current != end:
        visited.add(current)
        neighbors = TAC_GRAPH[current]

        # tránh loop
        next_tac = random.choice(neighbors)
        path.append(next_tac)
        current = next_tac

        if len(path) > 10:  # tránh loop vô hạn
            break

    return path

def random_move(current, prob=0.2):
    if random.random() < prob:
        return move_to_next(current)
    return current

# -----------------------------
# MAIN GENERATION
# -----------------------------
records = []

start_date = datetime(2024, 1, 1)

for ue in range(NUM_UE):
    ue_id = f"UE_{ue:04d}"
    
    home = random.choice(TAC_LIST)
    work = random.choice([t for t in TAC_LIST if t != home])
    
    current_time = start_date
    tac_sequence = []

    for day in range(DAYS):
        is_weekend = current_time.weekday() >= 5
        
        # -------- MORNING --------
        hour = random.randint(7, 9)
        current_time = current_time.replace(hour=hour, minute=0)
        
        if not is_weekend:
            path = generate_path(home, work)
        else:
            if random.random() < 0.6:
                path = [home]
            else:
                random_dest = random.choice(TAC_LIST)
                path = generate_path(home, random_dest)

        for tac in path:
            tac_sequence.append((current_time, tac))
            current_time += timedelta(minutes=10)

        # -------- DAYTIME --------
        for _ in range(random.randint(2, 5)):
            tac = work if not is_weekend else random.choice(TAC_LIST)
            tac = random_move(tac, prob=0.2)
            tac_sequence.append((current_time, tac))
            current_time += timedelta(minutes=60)

        # -------- EVENING --------
        hour = random.randint(17, 19)
        current_time = current_time.replace(hour=hour)

        if not is_weekend:
            path = generate_path(work, home)
        else:
            path = generate_path(tac, home)

        for tac in path:
            tac_sequence.append((current_time, tac))
            current_time += timedelta(minutes=10)

        # -------- NIGHT --------
        if random.random() < 0.3:
            night_dest = random.choice(TAC_LIST)
            path = generate_path(home, night_dest) + generate_path(night_dest, home)

            for tac in path:
                tac_sequence.append((current_time, tac))
                current_time += timedelta(minutes=15)

        current_time += timedelta(days=1)

    # -----------------------------
    # CREATE FEATURES FROM SEQUENCE
    # -----------------------------
    tac_sequence = sorted(tac_sequence, key=lambda x: x[0])

    for i in range(2, len(tac_sequence) - 1):
        prev_2 = tac_sequence[i-2][1]
        prev_1 = tac_sequence[i-1][1]
        current = tac_sequence[i][1]
        next_tac = tac_sequence[i+1][1]

        timestamp = tac_sequence[i][0]

        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0

        records.append([
            ue_id,
            prev_2,
            prev_1,
            current,
            hour,
            day_of_week,
            is_weekend,
            next_tac
        ])

# -----------------------------
# SAVE DATASET
# -----------------------------
columns = [
    "UE_ID",
    "prev_2_TAC",
    "prev_TAC",
    "current_TAC",
    "hour",
    "day_of_week",
    "is_weekend",
    "next_TAC"
]

df = pd.DataFrame(records, columns=columns)

df = df.sample(frac=1).reset_index(drop=True)  # shuffle

df.to_csv("ue_tac_prediction_dataset.csv", index=False)

print("Dataset shape:", df.shape)
print(df.head())