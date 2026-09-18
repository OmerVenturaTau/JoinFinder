"""
Transformer-based Hebrew text encoder built on HuggingFace models.

This module provides a lightweight wrapper around models such as AlephBERT
to produce sentence/line embeddings that can later be fused with the vision
and character signals.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from collections import OrderedDict
from typing import Iterable, List, Optional, Sequence

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

# Avoid "forked after parallelism" warnings / deadlocks when DataLoader workers fork.
if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Fast tokenizers (Rust) are not thread-safe. DataParallel runs forward in worker threads;
# concurrent tokenizer() calls raise RuntimeError: Already borrowed.
FAST_TOKENIZER_CALL_LOCK = threading.Lock()


@dataclass
class TextEncoderConfig:
    model_name: str = "onlplab/alephbert-base"
    max_length: int = 256
    pooling: str = "cls"  # one of {"cls", "mean"}
    device: Optional[str] = None
    dtype: torch.dtype = torch.float32
    freeze_encoder: bool = False
    dropout: float = 0.1
    # Cache per-string tokenization to reduce tokenizer overhead for repeated words/strings.
    # Useful when you encode many short strings (e.g., words) repeatedly across epochs.
    enable_token_cache: bool = True
    token_cache_size: int = 50000


class HebrewTextEncoder(nn.Module):
    """
    Wrap a HuggingFace transformer to encode Hebrew manuscript text.
    """

    def __init__(self, config: Optional[TextEncoderConfig] = None):
        super().__init__()
        self.config = config or TextEncoderConfig()

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(self.config.model_name)
        self.transformer: PreTrainedModel = AutoModel.from_pretrained(self.config.model_name)

        if self.config.freeze_encoder:
            for param in self.transformer.parameters():
                param.requires_grad = False

        self.projection = nn.Identity()
        self.dropout = nn.Dropout(self.config.dropout)

        device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device=device, dtype=self.config.dtype)

        # Simple LRU cache for per-string tokenization (stores Python tuples on CPU).
        self._token_cache: "OrderedDict[str, tuple[tuple[int, ...], tuple[int, ...]]]" = OrderedDict()

    def _tokenize_one(self, text: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """
        Tokenize a single string with truncation/max_length and special tokens.
        Returns (input_ids, attention_mask) as Python tuples (CPU-side, cacheable).
        """
        with FAST_TOKENIZER_CALL_LOCK:
            encoded = self.tokenizer(
                text,
                padding=False,
                truncation=True,
                max_length=self.config.max_length,
                return_attention_mask=True,
                add_special_tokens=True,
            )
        input_ids = tuple(int(x) for x in encoded["input_ids"])
        attn = tuple(int(x) for x in encoded["attention_mask"])
        return input_ids, attn

    def _tokenize_many(self, texts: Sequence[str]) -> dict[str, torch.Tensor]:
        """
        Tokenize many strings. Uses an LRU cache when enabled.
        Returns a dict compatible with HuggingFace model forward: input_ids, attention_mask (padded).
        """
        if not texts:
            raise ValueError("No texts provided to encode.")

        if not self.config.enable_token_cache:
            with FAST_TOKENIZER_CALL_LOCK:
                encoded = self.tokenizer(
                    list(texts),
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                )
            return encoded

        # Gather tokenized sequences (cached) and pad manually.
        tokenized: List[tuple[tuple[int, ...], tuple[int, ...]]] = []
        max_len = 0
        for t in texts:
            if t in self._token_cache:
                ids, attn = self._token_cache.pop(t)  # refresh LRU order
                self._token_cache[t] = (ids, attn)
            else:
                ids, attn = self._tokenize_one(t)
                self._token_cache[t] = (ids, attn)
                # Evict LRU if over capacity
                if len(self._token_cache) > max(0, int(self.config.token_cache_size)):
                    self._token_cache.popitem(last=False)
            tokenized.append((ids, attn))
            if len(ids) > max_len:
                max_len = len(ids)

        # Pad to max_len for this batch
        pad_id = int(self.tokenizer.pad_token_id) if self.tokenizer.pad_token_id is not None else 0
        B = len(tokenized)
        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        for i, (ids, attn) in enumerate(tokenized):
            L = len(ids)
            input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :L] = torch.tensor(attn, dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        """
        Encode a list of strings into embeddings.
        """
        encoded = self._tokenize_many(texts)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        transformer_trainable = any(p.requires_grad for p in self.transformer.parameters())
        with torch.set_grad_enabled(transformer_trainable):
            outputs = self.transformer(**encoded)
            hidden = outputs.last_hidden_state  # [B, L, H]

            if self.config.pooling == "mean":
                attention_mask = encoded["attention_mask"].unsqueeze(-1)  # [B, L, 1]
                summed = torch.sum(hidden * attention_mask, dim=1)
                counts = attention_mask.sum(dim=1).clamp(min=1e-6)
                pooled = summed / counts
            elif self.config.pooling == "cls":
                pooled = hidden[:, 0]
            else:
                raise ValueError(f"Unknown pooling mode: {self.config.pooling}")

            pooled = self.dropout(pooled)
            return self.projection(pooled)

    @torch.inference_mode()
    def encode_dataset(self, texts: Sequence[str], batch_size: int = 16) -> torch.Tensor:
        """
        Encode many texts efficiently in batches.
        """
        embeddings: List[torch.Tensor] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            emb = self.forward(batch)
            embeddings.append(emb.detach().cpu())
        return torch.cat(embeddings, dim=0) if embeddings else torch.empty(0, device="cpu")


def chunk_iterable(iterable: Sequence[str], chunk_size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i : i + chunk_size]

