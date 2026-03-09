"""
Entry point — orchestrates all three tasks:

    Task-1  Run a forward pass through the custom BERT and print shapes.
    Task-2  Load Fakeddit, balance it, attach comments, run EDA.
    Task-3  Fine-tune BertForSequenceClassification, evaluate, save results.
"""

import json
import torch

from config import RESULTS_FILE
from custom_bert import run_task1_demo
from data import load_fakeddit, balance_dataset, attach_comments, cal_mean_std, get_splits, get_dataloaders
from train import fine_tune, evaluate
from visualise import plot_label_distribution, plot_confusion_matrix


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ── Task-1: Custom BERT demo ─────────────────────────────────────────────
    print("=" * 60)
    print("TASK-1: Custom BERT forward pass")
    print("=" * 60)
    run_task1_demo()

    # ── Task-2: Load & explore Fakeddit ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("TASK-2: Loading and exploring Fakeddit")
    print("=" * 60)

    full_df, comments_df = load_fakeddit()
    print(f"Full dataset size: {len(full_df)}")

    print("\nComment stats — entire dataset:")
    cal_mean_std(full_df.drop("2_way_label", axis=1), full_df["2_way_label"], comments_df)

    balanced_df = balance_dataset(full_df)
    balanced_df = attach_comments(balanced_df, comments_df)
    print(f"\nBalanced dataset size: {len(balanced_df)}")

    X_train, X_test, y_train, y_test = get_splits(balanced_df)

    print("\nComment stats — balanced dataset:")
    cal_mean_std(X_train, y_train, comments_df)

    plot_label_distribution(y_train, y_test)

    # ── Task-3: Fine-tune BERT ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TASK-3: Fine-tuning BERT for fake-news detection")
    print("=" * 60)

    train_loader, test_loader = get_dataloaders(X_train, X_test, y_train, y_test)

    model   = fine_tune(train_loader, device)
    metrics = evaluate(model, test_loader, device)

    plot_confusion_matrix(metrics["confusion_matrix"])

    # ── Save results ─────────────────────────────────────────────────────────
    results = {
        "task": "Fakeddit fake-news binary classification",
        "model": "bert-base-uncased (fine-tuned)",
        "metrics": metrics,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {RESULTS_FILE}")
    _print_summary(metrics)


def _print_summary(metrics: dict):
    print("\n" + "=" * 40)
    print("Final Test Metrics")
    print("=" * 40)
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:<12}: {v}")
    print("=" * 40)


if __name__ == "__main__":
    main()
