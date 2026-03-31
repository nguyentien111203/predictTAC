import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_cost(m_top3, r_top3, h_top3):
    models = ["Markov", "Random Forest", "Hybrid"]
    acc = [m_top3, r_top3, h_top3]

    plt.figure()
    plt.bar(models, acc)

    plt.ylabel("Paging cost")
    plt.title("Paging cost Comparison")
    plt.savefig("./Figure/cost.png")

    for i, v in enumerate(acc):
        plt.text(i, v, f"{v:.2f}", ha='center', va='bottom')

def plot_accuracy_top1(m_top1, r_top1, h_top1):
    models = ["Markov", "Random Forest", "Hybrid"]
    acc = [m_top1, r_top1, h_top1]

    plt.figure()
    plt.bar(models, acc)

    plt.ylabel("Top-1 Accuracy")
    plt.title("Top-1 Accuracy Comparison")

    for i, v in enumerate(acc):
        plt.text(i, v, f"{v:.2f}", ha='center', va='bottom')
    plt.savefig("./Figure/accuracy1.png")

def plot_accuracy_top3(m_top3, r_top3, h_top3):
    models = ["Markov", "Random Forest", "Hybrid"]
    acc = [m_top3, r_top3, h_top3]

    plt.figure()
    plt.bar(models, acc)

    plt.ylabel("Top-3 Accuracy")
    plt.title("Top-3 Accuracy Comparison")

    for i, v in enumerate(acc):
        plt.text(i, v, f"{v:.2f}", ha='center', va='bottom')
    plt.savefig("./Figure/accuracy3.png")
