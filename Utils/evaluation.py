class Evaluator:
    def __init__(self, markov_model, rf_model, hybrid_model):
        self.markov = markov_model
        self.rf = rf_model
        self.hybrid = hybrid_model

    def evaluate(self, X_test, y_test):
        m_top1 = m_top3 = 0
        r_top1 = r_top3 = 0
        h_top1 = h_top3 = 0

        total = len(X_test)

        for i in range(total):
            x = X_test.iloc[i]
            true = y_test.iloc[i]

            # -------- Markov --------
            m_pred = self.markov.predict_top_k(x["current_TAC"], k=3)
            if m_pred:
                if true == m_pred[0]:
                    m_top1 += 1
                if true in m_pred:
                    m_top3 += 1

            # -------- RF --------
            r_pred = self.rf.predict_top_k(x, k=3)
            if true == r_pred[0]:
                r_top1 += 1
            if true in r_pred:
                r_top3 += 1

            # -------- Hybrid --------
            h_pred = self.hybrid.predict_top_k(x, k=3)
            if true == h_pred[0]:
                h_top1 += 1
            if true in h_pred:
                h_top3 += 1

        # SAVE FILE
        with open("logs/result.txt", "w") as f:
            f.write("=== ACCURACY ===\n")
            f.write(f"Markov Top-1: {m_top1/total:.4f}\n")
            f.write(f"Markov Top-3: {m_top3/total:.4f}\n")

            f.write(f"RF Top-1: {r_top1/total:.4f}\n")
            f.write(f"RF Top-3: {r_top3/total:.4f}\n")

            f.write(f"Hybrid Top-1: {h_top1/total:.4f}\n")
            f.write(f"Hybrid Top-3: {h_top3/total:.4f}\n")

        return m_top1/total, r_top1/total, h_top1/total, m_top3/total, r_top3/total, h_top3/total

    def paging_cost(self, X_test, y_test):
        m_cost = r_cost = h_cost = 0
        total = len(X_test)

        for i in range(total):
            x = X_test.iloc[i]
            true = y_test.iloc[i]

            m_pred = self.markov.predict_top_k(x["current_TAC"], k=3)
            m_cost += (m_pred.index(true)+1) if true in m_pred else 3

            r_pred = self.rf.predict_top_k(x, k=3)
            r_cost += (r_pred.index(true)+1) if true in r_pred else 3

            h_pred = self.hybrid.predict_top_k(x, k=3)
            h_cost += (h_pred.index(true)+1) if true in h_pred else 3

        return m_cost/total, r_cost/total, h_cost/total