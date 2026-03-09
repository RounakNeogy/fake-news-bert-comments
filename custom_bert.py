"""
Task-1: Custom BERT model built from scratch with PyTorch.

Components
----------
MultiHeadSelfAttention  – scaled dot-product attention over H heads
FeedForward             – two-layer MLP with GELU activation
TransformerEncoderLayer – MHSA + FFN with residual connections & LayerNorm
customBert              – full encoder model with token + positional embeddings,
                          pooler, and a 2-class classifier head
"""

import torch
import torch.nn as nn
from transformers import BertTokenizer

from config import EMBED_DIM, NUM_HEADS, NUM_LAYERS, FF_HIDDEN_DIM, MAX_SEQ_LEN, BERT_MODEL_NAME


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert EMBED_DIM % NUM_HEADS == 0, "EMBED_DIM must be divisible by NUM_HEADS"
        self.head_dim = EMBED_DIM // NUM_HEADS

        self.W_q      = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.W_k      = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.W_v      = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.out_proj = nn.Linear(EMBED_DIM, EMBED_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        q = self.W_q(x).reshape(B, T, NUM_HEADS, self.head_dim).permute(0, 2, 1, 3)  # (B,H,T,Hd)
        k = self.W_k(x).reshape(B, T, NUM_HEADS, self.head_dim).permute(0, 2, 1, 3)
        v = self.W_v(x).reshape(B, T, NUM_HEADS, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        scores  = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)   # (B,H,T,T)
        weights = nn.functional.softmax(scores, dim=-1)
        context = weights @ v                                           # (B,H,T,Hd)

        # Merge heads → (B,T,D)
        context = context.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(context)


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(EMBED_DIM, FF_HIDDEN_DIM)
        self.fc2 = nn.Linear(FF_HIDDEN_DIM, EMBED_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(nn.functional.gelu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Transformer Encoder Layer
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mh_attn = MultiHeadSelfAttention()
        self.ln1     = nn.LayerNorm(EMBED_DIM)
        self.ff      = FeedForward()
        self.ln2     = nn.LayerNorm(EMBED_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MHSA sub-layer
        attn_out = self.mh_attn(x)
        print(f"  [Encoder] MHSA output shape      : {attn_out.shape}")
        x = self.ln1(x + attn_out)          # residual + LayerNorm

        # FFN sub-layer
        ff_out = self.ff(x)
        print(f"  [Encoder] FeedForward output shape: {ff_out.shape}")
        x = self.ln2(x + ff_out)            # residual + LayerNorm
        return x


# ---------------------------------------------------------------------------
# Custom BERT model
# ---------------------------------------------------------------------------

class customBert(nn.Module):
    """
    Minimal BERT-style encoder with:
      - Token + positional embeddings
      - N stacked TransformerEncoderLayer blocks
      - [CLS] pooler → 2-class classifier head
    """

    def __init__(self):
        super().__init__()
        self.tokenizer           = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        self.token_embeddings    = nn.Embedding(self.tokenizer.vocab_size, EMBED_DIM)
        self.position_embeddings = nn.Embedding(MAX_SEQ_LEN, EMBED_DIM)
        self.layernorm           = nn.LayerNorm(EMBED_DIM)
        self.encoders            = nn.ModuleList(
            [TransformerEncoderLayer() for _ in range(NUM_LAYERS)]
        )
        self.pooler     = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.classifier = nn.Linear(EMBED_DIM, 2)

    def forward(self, text: str) -> torch.Tensor:
        """
        Parameters
        ----------
        text : str  — a single input sentence

        Returns
        -------
        probs : (1, 2) softmax probabilities
        """
        encoding  = self.tokenizer(text, max_length=MAX_SEQ_LEN, return_tensors="pt")
        input_ids = encoding["input_ids"]                         # (1, T)
        B, T      = input_ids.shape

        pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1)     # (B, T)

        tok_emb = self.token_embeddings(input_ids)                # (B, T, D)
        pos_emb = self.position_embeddings(pos_ids)               # (B, T, D)

        print(f"Token embeddings shape     : {tok_emb.shape}")
        print(f"Positional embeddings shape: {pos_emb.shape}")

        x = self.layernorm(tok_emb + pos_emb)

        for i, enc in enumerate(self.encoders):
            x = enc(x)
            print(f"Encoder Layer {i} output shape: {x.shape}")

        # [CLS] token → pooler → classifier
        cls_token = x[:, 0, :]                                    # (B, D)
        pooled    = torch.tanh(self.pooler(cls_token))
        logits    = self.classifier(pooled)
        probs     = nn.functional.softmax(logits, dim=-1)

        print(f"Output probabilities shape : {probs.shape}")
        return probs


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

def run_task1_demo():
    """Run one forward pass and print all intermediate shapes."""
    sentence = (
        "Hi I am writing the sentence with more than 10 words, "
        "but still the sentence is short, maybe now it is fine."
    )
    print("=" * 60)
    print("Task-1: Custom BERT forward pass demo")
    print("=" * 60)
    model = customBert()
    probs = model(sentence)
    print(f"\nFinal output probabilities: {probs}")
    return probs


if __name__ == "__main__":
    run_task1_demo()
