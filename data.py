"""
Task-2/3: Data loading and preprocessing for the Fakeddit dataset.

Pipeline
--------
1. load_fakeddit()      – read all TSV splits, concat, drop NaN
2. balance_dataset()    – sample equal fake / non-fake posts
3. attach_comments()    – join top-level comment bodies per post
4. get_splits()         – stratified train/test split
5. FakedditDataset      – PyTorch Dataset (tokenises title + comments)
6. get_dataloaders()    – returns (train_loader, test_loader)
"""

import re
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from config import (
    TRAIN_TSV, TEST_TSV, VALIDATE_TSV, COMMENTS_TSV,
    SAMPLE_SIZE, TEST_SPLIT, RANDOM_STATE,
    BERT_MODEL_NAME, MAX_TOKEN_LEN, BATCH_SIZE,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_fakeddit() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read train / test / validate TSV splits, concatenate them, and keep only
    the columns we need.

    Returns
    -------
    df           : full dataset (clean_title, 2_way_label)
    comments_df  : comments (body, submission_id, parent_id)  — cleaned
    """
    paths = [TRAIN_TSV, TEST_TSV, VALIDATE_TSV]
    frames = [pd.read_csv(p, sep="\t", index_col="id") for p in paths]
    df = pd.concat(frames)[["clean_title", "2_way_label"]].dropna()

    comments_df = pd.read_csv(COMMENTS_TSV, sep="\t", index_col="id")
    comments_df = comments_df[["body", "submission_id", "parent_id"]].dropna()

    # Keep only alphanumeric and basic punctuation
    comments_df["body"] = comments_df["body"].apply(
        lambda t: re.sub(r"[^a-zA-Z0-9.,!?\'\"\\s]", "", t)
    )

    return df, comments_df


# ---------------------------------------------------------------------------
# Balancing
# ---------------------------------------------------------------------------

def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Down-sample to SAMPLE_SIZE examples per class (fake=1, non-fake=0)
    and shuffle the result.
    """
    fake     = df[df["2_way_label"] == 1].sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    non_fake = df[df["2_way_label"] == 0].sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    return pd.concat([fake, non_fake]).sample(frac=1, random_state=RANDOM_STATE)


# ---------------------------------------------------------------------------
# Comment attachment
# ---------------------------------------------------------------------------

def attach_comments(df: pd.DataFrame, comments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join all top-level comments for each post (parent_id == "t3_<post_id>")
    into a single string separated by " || " and add as a new column.
    Missing comments are replaced with an empty string.
    """
    joined = comments_df.groupby("parent_id")["body"].apply(lambda s: " || ".join(s))
    df = df.copy()
    df["comments"] = ("t3_" + df.index).map(joined).fillna("")
    return df


# ---------------------------------------------------------------------------
# EDA helper
# ---------------------------------------------------------------------------

def cal_mean_std(X: pd.DataFrame, y: pd.Series, comments_df: pd.DataFrame):
    """Print mean / std of comment counts split by label."""
    fake_ids     = set(X[y == 1].index)
    non_fake_ids = set(X[y == 0].index)

    comment_counts = comments_df["submission_id"].value_counts()

    fake_counts     = [comment_counts.get(i, 0) for i in fake_ids]
    non_fake_counts = [comment_counts.get(i, 0) for i in non_fake_ids]

    print(f"  Fake     — mean: {np.mean(fake_counts):.2f}, std: {np.std(fake_counts):.2f}")
    print(f"  Non-fake — mean: {np.mean(non_fake_counts):.2f}, std: {np.std(non_fake_counts):.2f}")


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def get_splits(df: pd.DataFrame):
    """
    Stratified split on 2_way_label.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = df.drop("2_way_label", axis=1)
    y = df["2_way_label"]
    return train_test_split(X, y, test_size=TEST_SPLIT,
                            random_state=RANDOM_STATE, stratify=y)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class FakedditDataset(Dataset):
    """
    Tokenises (clean_title, comments) pairs with the BERT tokeniser.
    The title is passed as *text* and the comment body as *text_pair*,
    so BERT's [SEP] token separates the two segments.
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series):
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

        encodings = tokenizer(
            text=X["clean_title"].tolist(),
            text_pair=X["comments"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=MAX_TOKEN_LEN,
            return_tensors="pt",
        )

        self.input_ids      = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels         = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         self.labels[idx],
        }


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def get_dataloaders(X_train, X_test, y_train, y_test):
    """Build and return (train_loader, test_loader)."""
    train_ds = FakedditDataset(X_train, y_train)
    test_ds  = FakedditDataset(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader
