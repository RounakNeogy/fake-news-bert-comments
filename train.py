"""
Task-3: Fine-tuning BertForSequenceClassification and evaluating on the test set.
"""

import torch
from torch.optim import AdamW
from transformers import BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from config import BERT_MODEL_NAME, NUM_LABELS, LEARNING_RATE, EPOCHS


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def fine_tune(train_loader, device) -> BertForSequenceClassification:
    """
    Load bert-base-uncased, fine-tune on *train_loader* for EPOCHS epochs.

    Returns the trained model.
    """
    model = BertForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME, num_labels=NUM_LABELS
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0

        for batch in train_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS}  |  Avg loss: {avg_loss:.4f}")

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, test_loader, device) -> dict:
    """
    Run inference on *test_loader* and compute classification metrics.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, confusion_matrix
    """
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    acc           = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary"
    )
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\nAccuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    return {
        "accuracy":         round(float(acc),  4),
        "precision":        round(float(prec), 4),
        "recall":           round(float(rec),  4),
        "f1":               round(float(f1),   4),
        "confusion_matrix": cm.tolist(),
    }
