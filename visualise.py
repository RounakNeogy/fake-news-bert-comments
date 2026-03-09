"""
Plotting utilities: label distribution bar charts and confusion matrix heatmap.
"""

import matplotlib.pyplot as plt
import seaborn as sns


def plot_label_distribution(y_train, y_test, save_path: str = "label_distribution.png"):
    """Bar charts of label counts for train and test sets."""
    train_counts = y_train.value_counts().sort_index()
    test_counts  = y_test.value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    sns.barplot(x=train_counts.index, y=train_counts.values, ax=axes[0])
    axes[0].set_title("Distribution of Labels in Training Set")
    axes[0].set_xlabel("Label (0: Non-Fake, 1: Fake)")
    axes[0].set_ylabel("Number of Posts")
    axes[0].set_xticks([0, 1])

    sns.barplot(x=test_counts.index, y=test_counts.values, ax=axes[1])
    axes[1].set_title("Distribution of Labels in Test Set")
    axes[1].set_xlabel("Label (0: Non-Fake, 1: Fake)")
    axes[1].set_ylabel("Number of Posts")
    axes[1].set_xticks([0, 1])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved label distribution plot → {save_path}")


def plot_confusion_matrix(cm, save_path: str = "confusion_matrix.png"):
    """Heatmap of the confusion matrix."""
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved confusion matrix plot → {save_path}")
