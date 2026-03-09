"""
Global configuration constants for NLP Assignment 3.
Task-1: Custom BERT from scratch.
Task-2/3: Fine-tuned BERT for fake-news detection on Fakeddit.
"""

# ── Task-1: Custom BERT architecture ────────────────────────────────────────
EMBED_DIM       = 768
NUM_HEADS       = 12
NUM_LAYERS      = 2          # encoder layers in the custom BERT
FF_HIDDEN_DIM   = 1568       # feed-forward hidden size
MAX_SEQ_LEN     = 512        # maximum token positions

# ── Task-2/3: Dataset ────────────────────────────────────────────────────────
SAMPLE_SIZE     = 10000      # samples drawn per class for balanced dataset
TEST_SPLIT      = 0.2
RANDOM_STATE    = 42

# Paths for Fakeddit data (update these to match your local directory layout)
DATA_DIR        = "data"
TRAIN_TSV       = f"{DATA_DIR}/all_train.tsv"
TEST_TSV        = f"{DATA_DIR}/all_test_public.tsv"
VALIDATE_TSV    = f"{DATA_DIR}/all_validate.tsv"
COMMENTS_TSV    = f"{DATA_DIR}/all_comments.tsv"

# ── Task-3: Fine-tuning ──────────────────────────────────────────────────────
BERT_MODEL_NAME = "bert-base-uncased"
MAX_TOKEN_LEN   = 256        # tokeniser max_length for title + comments pair
BATCH_SIZE      = 8
EPOCHS          = 3
LEARNING_RATE   = 2e-5
NUM_LABELS      = 2

RESULTS_FILE    = "results.json"
