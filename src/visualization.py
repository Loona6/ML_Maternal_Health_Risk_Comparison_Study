import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

def plot_accuracy_and_brier(cv_df):
    """
    Plots the mean accuracy and mean Brier Score error across stress tiers.
    """
    # Configure Seaborn style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 6))

    # Plot 1: Mean Accuracy with standard deviation error bars
    plt.subplot(1, 2, 1)
    sns.lineplot(data=cv_df, x="Tier", y="Accuracy", hue="Classifier",
                 marker="o", linewidth=2.5, err_style="bars", errorbar="sd")
    plt.title("Statistical Accuracy Under Degradation (MICE Imputed)", fontsize=13, fontweight='bold')
    plt.ylabel("Mean Accuracy (Higher is Better)")
    plt.xlabel("Stress Tier")
    plt.xticks(rotation=15)

    # Plot 2: Mean Brier Score Error with standard deviation error bars
    plt.subplot(1, 2, 2)
    sns.lineplot(data=cv_df, x="Tier", y="Brier Score", hue="Classifier",
                 marker="s", linewidth=2.5, err_style="bars", errorbar="sd")
    plt.title("Model Probability Reliability / Brier Score", fontsize=13, fontweight='bold')
    plt.ylabel("Mean Brier Score Error (Lower is Better)")
    plt.xlabel("Stress Tier")
    plt.xticks(rotation=15)

    plt.tight_layout()
    plt.show()

def display_summary_stats(cv_df):
    """
    Displays summary statistics table for the paper.
    """
    summary_stats = cv_df.groupby(["Classifier", "Tier"])[["Accuracy", "Brier Score"]].agg(["mean", "std"])
    summary_stats.columns = ['Acc_Mean', 'Acc_Std', 'Brier_Mean', 'Brier_Std']
    display(summary_stats.round(4))

def plot_heatmaps(cv_df):
    """
    Format cross-validation summary table into a clean pivot matrix and plots heatmaps.
    """
    pivot_acc = cv_df.groupby(["Classifier", "Tier"])["Accuracy"].mean().unstack()
    pivot_brier = cv_df.groupby(["Classifier", "Tier"])["Brier Score"].mean().unstack()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Heatmap 1: Accuracy Matrix
    sns.heatmap(pivot_acc, annot=True, fmt=".2%", cmap="YlGnBu", ax=axes[0], cbar=False, annot_kws={"size": 11})
    axes[0].set_title("Mean Model Accuracy Across Tiers (Higher is Better)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("")

    # Heatmap 2: Brier Score Matrix
    sns.heatmap(pivot_brier, annot=True, fmt=".3f", cmap="YlOrRd", ax=axes[1], cbar=False, annot_kws={"size": 11})
    axes[1].set_title("Mean Brier Score Probability Error (Lower is Better)", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("")

    plt.tight_layout()
    plt.show()
