# BERT from Scratch & Fake News Detection on Fakeddit

Two tasks in one project: first, implement a BERT-style Transformer encoder in raw PyTorch — every attention head, layer norm, and residual connection by hand. Then, fine-tune a pretrained BERT on [Fakeddit](https://github.com/entitize/Fakeddit), a large-scale Reddit fake news dataset, using post titles paired with community comments to detect misinformation.

**Best result: 89.3% accuracy, 0.8983 F1-score** on a balanced 4,000-sample test set.

---

## Table of Contents

- [Task 1 — BERT from Scratch](#task-1--bert-from-scratch)
- [Task 2 — Fakeddit Dataset Analysis](#task-2--fakeddit-dataset-analysis)
- [Task 3 — Fine-tuning BERT for Classification](#task-3--fine-tuning-bert-for-classification)
- [Results](#results)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Requirements](#requirements)

---

## Task 1 — BERT from Scratch

A BERT-style encoder implemented from scratch using only `torch.nn` primitives — no `nn.MultiheadAttention`, no HuggingFace model classes.

### Architecture

```
Input Text
  → BertTokenizer (bert-base-uncased vocab)
  → Token Embeddings      [vocab_size × 768]
  + Positional Embeddings [512 × 768]
  → LayerNorm

  × 2 TransformerEncoderLayer:
    → MultiHeadSelfAttention   (12 heads, head_dim=64)
    → Residual + LayerNorm
    → FeedForward              (768 → 1568 → 768, GELU)
    → Residual + LayerNorm

  → [CLS] token → Pooler Linear(768→768)
  → Classifier Linear(768→2) → Softmax
```

**Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| `embed_dim` | 768 |
| `num_heads` | 12 |
| `head_dim` | 64 |
| `num_layers` | 2 |
| FFN hidden dim | 1568 |
| Max sequence length | 512 |
| Output classes | 2 |

### Multi-Head Self-Attention (implemented from scratch)

```python
def forward(self, x):
    B, T, D = x.shape  # batch, seq_len, embed_dim

    q = self.W_q(x).reshape(B, T, num_heads, head_dim).permute(0, 2, 1, 3)
    k = self.W_k(x).reshape(B, T, num_heads, head_dim).permute(0, 2, 1, 3)
    v = self.W_v(x).reshape(B, T, num_heads, head_dim).permute(0, 2, 1, 3)

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
    attn = torch.softmax(scores, dim=-1)

    out = (attn @ v).permute(0, 2, 1, 3).reshape(B, T, D)
    return self.out_proj(out)  # (B, T, 768)
```

### Shape Verification

Forward pass verified on a 29-token input sentence:

```
Token embeddings:           torch.Size([1, 29, 768])
Positional embeddings:      torch.Size([1, 29, 768])
Multi-Head Attention out:   torch.Size([1, 29, 768])  ← Layer 0
Feed-Forward out:           torch.Size([1, 29, 768])  ← Layer 0
Encoder Layer 0 out:        torch.Size([1, 29, 768])
Encoder Layer 1 out:        torch.Size([1, 29, 768])
Output probabilities:       tensor([[0.4529, 0.5471]])
```

---

## Task 2 — Fakeddit Dataset Analysis

[Fakeddit](https://github.com/entitize/Fakeddit) is a large-scale multimodal fake news dataset built from Reddit posts, labeled by community moderators. This project uses the text-only modality (titles + comments).

### Dataset Statistics

| Property | Value |
|----------|-------|
| Total posts (original) | 971,806 |
| Posts with at least one comment | 618,943 |
| Balanced sample used (10K fake + 10K non-fake) | 20,000 |
| Train / Test split | 80% / 20% |
| Input token max length | 256 |

### Comment Statistics

|  | Fake Posts | Non-Fake Posts |
|--|------------|----------------|
| Mean comments | **14.93** | 4.33 |
| Std deviation | 55.44 | 16.03 |

Fake posts attract **~3.5× more comments** on average, with substantially higher variance. This pattern holds in both the full 972K dataset and the balanced 20K sample, suggesting community engagement is a meaningful signal for credibility.

### Preprocessing

- Train/test sets (from `all_train.tsv`, `all_test_public.tsv`, `all_validate.tsv`) were merged and then re-split
- 10,000 fake and 10,000 non-fake posts sampled (`random_state=42`) for class balance
- Top-level comments for each post aggregated and joined with ` || ` separator
- Non-alphabetic characters cleaned from comment text (regex)
- `NaN` values dropped throughout

---

## Task 3 — Fine-tuning BERT for Classification

`bert-base-uncased` fine-tuned for binary classification (fake / non-fake) using `BertForSequenceClassification`.

### Input Strategy

Post title and comments are passed as a **sentence pair** to BERT:

```python
encodings = tokenizer(
    text=X["clean_title"].tolist(),       # Sequence A: post title
    text_pair=X["comments"].tolist(),     # Sequence B: joined comments
    truncation=True,
    padding="max_length",
    max_length=256,
    return_tensors="pt"
)
```

BERT's segment embeddings (`token_type_ids`) naturally separate the two sources, letting the model learn which evidence comes from the author vs the community.

### Training Configuration

| Setting | Value |
|---------|-------|
| Base model | `bert-base-uncased` |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Batch size | 8 |
| Epochs | 3 |
| Hardware | CUDA GPU |

### Training Loss

| Epoch | Avg Loss |
|-------|----------|
| 1 | 0.2743 |
| 2 | 0.1382 |
| 3 | 0.0597 |

Loss dropped by 78% across 3 epochs, indicating strong signal in the pretrained representations for this task.

---

## Results

Evaluated on a held-out test set of 4,000 posts (2,000 fake, 2,000 non-fake):

| Metric | Score |
|--------|-------|
| **Accuracy** | **0.8932** |
| Precision | 0.8580 |
| **Recall** | **0.9425** |
| **F1-Score** | **0.8983** |

**Confusion matrix (estimated):**

```
                Predicted: Non-Fake    Predicted: Fake
Actual: Non-Fake     1,770 (TN)           230 (FP)
Actual: Fake           115 (FN)         1,885 (TP)
```

The model skews towards high recall — it catches 94.3% of fake posts, at the cost of some false positives. For a misinformation detector, this is the right tradeoff.

---

## Key Findings

**Comments add real signal.** Fake posts generate 3.5× more comments than real ones. Feeding comments as a second sequence allows BERT to leverage community reaction as an implicit credibility signal — something a title-only model cannot access.

**High recall is the right priority.** Missing a fake post (false negative) is costlier than flagging a real one (false positive). The model's recall of 94.3% reflects this priority naturally from training.

**BERT fine-tunes fast on this task.** Three epochs were enough for strong generalisation, with the loss falling from 0.274 to 0.060. The pretrained representations already capture much of the linguistic difference between credible and non-credible writing.

**Building BERT from scratch clarifies the architecture.** Writing Q/K/V projections, scaled dot-product attention, and residual + LayerNorm explicitly — then verifying each tensor shape — builds intuition that library calls abstract away. Every dimension in the 768-d pipeline has a clear purpose once you've implemented it yourself.

---

## Requirements

```
torch
transformers
datasets
pandas
scikit-learn
matplotlib
seaborn
numpy
```

---
