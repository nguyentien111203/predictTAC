from collections import defaultdict

class MarkovModel:
    def __init__(self):
        self.transition_prob = {}

    def fit(self, df):
        transition_counts = defaultdict(lambda: defaultdict(int))

        for _, row in df.iterrows():
            curr = row["current_TAC"]
            nxt = row["next_TAC"]
            transition_counts[curr][nxt] += 1

        # convert to probability
        for curr, next_dict in transition_counts.items():
            total = sum(next_dict.values())
            self.transition_prob[curr] = {
                nxt: count / total for nxt, count in next_dict.items()
            }

    def predict_top_k(self, current, k=3):
        probs = self.transition_prob.get(current, {})

        if not probs:
            return []

        sorted_tac = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return [t[0] for t in sorted_tac[:k]]
    
    def get_prob(self, current, next_tac):
        return self.transition_prob.get(current, {}).get(next_tac, 0)