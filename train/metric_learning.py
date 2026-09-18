from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


MAX_QUANTILE_SCORES = 1_000_000


def _bounded_quantile_input(values: torch.Tensor, max_scores: int = MAX_QUANTILE_SCORES) -> torch.Tensor:
    """Keep quantile diagnostics bounded for large validation/test splits."""
    if values.numel() <= max_scores:
        return values
    # Deterministic row-order subsample. These are diagnostics/checkpoint-shaping
    # terms; exact p90/p95 over all negative pairs is not worth the memory/runtime.
    # Integer linspace can round its final CUDA index up to values.numel() for
    # some large sizes (for example 46,239,996 values and 1,000,000 steps).
    # Integer arithmetic guarantees every sampled index is in bounds.
    indices = (
        torch.arange(max_scores, device=values.device, dtype=torch.long)
        * values.numel()
        // max_scores
    )
    return values.index_select(0, indices)


class LabeledFeatureQueue:
    """Detached FIFO queue for normalized latent vectors and integer labels."""

    def __init__(self, capacity: int, feature_dim: int, device: torch.device):
        if int(capacity) <= 0:
            raise ValueError(f"Queue capacity must be positive, got {capacity}")
        self.capacity = int(capacity)
        self.feature_dim = int(feature_dim)
        self.device = device
        self.features = torch.zeros(self.capacity, self.feature_dim, device=device, dtype=torch.float32)
        self.labels = torch.full((self.capacity,), -1, device=device, dtype=torch.long)
        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return int(self.size)

    @torch.no_grad()
    def enqueue(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        if features.numel() == 0:
            return
        feats = F.normalize(features.detach().float(), dim=1, eps=1e-8).to(self.device)
        labs = labels.detach().long().to(self.device)
        if feats.ndim != 2 or feats.shape[1] != self.feature_dim:
            raise ValueError(f"Expected features [B,{self.feature_dim}], got {tuple(feats.shape)}")
        if labs.ndim != 1 or labs.shape[0] != feats.shape[0]:
            raise ValueError(f"Labels must be [B] aligned with features, got {tuple(labs.shape)}")

        n = feats.shape[0]
        if n >= self.capacity:
            feats = feats[-self.capacity :]
            labs = labs[-self.capacity :]
            n = self.capacity

        end = self.ptr + n
        if end <= self.capacity:
            self.features[self.ptr : end] = feats
            self.labels[self.ptr : end] = labs
        else:
            first = self.capacity - self.ptr
            self.features[self.ptr :] = feats[:first]
            self.labels[self.ptr :] = labs[:first]
            self.features[: end - self.capacity] = feats[first:]
            self.labels[: end - self.capacity] = labs[first:]

        self.ptr = end % self.capacity
        self.size = min(self.capacity, self.size + n)

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.size == 0:
            return (
                torch.empty(0, self.feature_dim, device=self.device, dtype=torch.float32),
                torch.empty(0, device=self.device, dtype=torch.long),
            )
        return self.features[: self.size], self.labels[: self.size]


class MemoryBankSupConLoss(nn.Module):
    """
    Supervised contrastive loss for small anchor batches using a detached memory bank.

    Gradients flow through anchors only. Batch positives are included, self matches
    are excluded, and queued features provide additional positives/negatives.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        if float(temperature) <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = float(temperature)

    def forward(
        self,
        anchor_features: torch.Tensor,
        anchor_labels: torch.Tensor,
        bank_features: Optional[torch.Tensor] = None,
        bank_labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        device = anchor_features.device
        anchors = F.normalize(anchor_features.float(), dim=1, eps=1e-8)
        labels = anchor_labels.long().to(device)
        batch_size = anchors.shape[0]

        contrast_features = [anchors]
        contrast_labels = [labels]
        if bank_features is not None and bank_features.numel() > 0:
            contrast_features.append(F.normalize(bank_features.detach().float().to(device), dim=1, eps=1e-8))
            contrast_labels.append(bank_labels.detach().long().to(device))

        contrast = torch.cat(contrast_features, dim=0)
        contrast_labs = torch.cat(contrast_labels, dim=0)
        logits = torch.matmul(anchors, contrast.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        self_mask = torch.zeros(batch_size, contrast.shape[0], dtype=torch.bool, device=device)
        self_mask[:, :batch_size] = torch.eye(batch_size, dtype=torch.bool, device=device)
        positive_mask = labels[:, None].eq(contrast_labs[None, :]) & ~self_mask
        valid_anchor_mask = positive_mask.any(dim=1)

        if not valid_anchor_mask.any():
            zero = anchors.sum() * 0.0
            stats = {
                "valid_anchor_frac": 0.0,
                "positive_pairs": 0.0,
                "contrast_count": float(contrast.shape[0]),
                "top1_acc": 0.0,
            }
            return zero, stats

        logits_mask = ~self_mask
        exp_logits = torch.exp(logits) * logits_mask.to(logits.dtype)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
        mean_log_prob_pos = (positive_mask.to(log_prob.dtype) * log_prob).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1)
        loss = -mean_log_prob_pos[valid_anchor_mask].mean()

        masked_logits = logits.masked_fill(self_mask, float("-inf"))
        top1_idx = masked_logits.argmax(dim=1)
        top1_is_positive = positive_mask.gather(1, top1_idx[:, None]).squeeze(1)
        top1_acc = top1_is_positive[valid_anchor_mask].float().mean()
        stats = {
            "valid_anchor_frac": float(valid_anchor_mask.float().mean().detach().cpu().item()),
            "positive_pairs": float(positive_mask.sum().detach().cpu().item()),
            "contrast_count": float(contrast.shape[0]),
            "top1_acc": float(top1_acc.detach().cpu().item()),
        }
        return loss, stats


@dataclass(frozen=True)
class RetrievalMetrics:
    knn_at_1: float
    knn_at_5: float
    knn_at_10: float
    mean_average_precision: float
    mean_positive_rank: float
    same_cosine_mean: float
    different_cosine_mean: float
    same_diff_cosine_gap: float
    hard_negative_cosine_p90: float
    hard_negative_cosine_p95: float
    hard_negative_cosine_max: float
    num_queries: int
    num_eval_images: int


def compute_retrieval_metrics(features: torch.Tensor, labels: torch.Tensor) -> RetrievalMetrics:
    feats = F.normalize(features.float(), dim=1, eps=1e-8)
    labs = labels.long()
    n = feats.shape[0]
    if n <= 1:
        return RetrievalMetrics(
            0.0, 0.0, 0.0, 0.0,
            float("nan"), float("nan"), float("nan"), float("nan"),
            float("nan"), float("nan"), float("nan"),
            0, n,
        )

    sim = torch.matmul(feats, feats.T)
    sim.fill_diagonal_(float("-inf"))
    same = labs[:, None].eq(labs[None, :])
    same.fill_diagonal_(False)
    has_positive = same.any(dim=1)
    query_idx = has_positive.nonzero(as_tuple=False).squeeze(1)
    if query_idx.numel() == 0:
        return RetrievalMetrics(
            0.0, 0.0, 0.0, 0.0,
            float("nan"), float("nan"), float("nan"), float("nan"),
            float("nan"), float("nan"), float("nan"),
            0, n,
        )

    sorted_idx = torch.argsort(sim[query_idx], dim=1, descending=True)
    same_sorted = same[query_idx].gather(1, sorted_idx)
    ks = {}
    for k in (1, 5, 10):
        kk = min(k, same_sorted.shape[1])
        ks[k] = same_sorted[:, :kk].any(dim=1).float().mean().item()

    ranks = torch.arange(1, same_sorted.shape[1] + 1, device=same_sorted.device, dtype=torch.float32)
    cumsum_pos = same_sorted.float().cumsum(dim=1)
    precision_at_rank = cumsum_pos / ranks[None, :]
    ap = (precision_at_rank * same_sorted.float()).sum(dim=1) / same_sorted.float().sum(dim=1).clamp_min(1)

    first_pos_rank = same_sorted.float().argmax(dim=1).float() + 1.0
    finite_sim = sim.masked_fill(torch.isinf(sim), float("nan"))
    same_scores = finite_sim[same]
    diff_scores = finite_sim[(~same) & (~torch.eye(n, dtype=torch.bool, device=sim.device))]
    same_mean = same_scores.nanmean() if same_scores.numel() else torch.tensor(float("nan"), device=sim.device)
    diff_mean = diff_scores.nanmean() if diff_scores.numel() else torch.tensor(float("nan"), device=sim.device)
    gap = same_mean - diff_mean if same_scores.numel() and diff_scores.numel() else torch.tensor(float("nan"), device=sim.device)
    if diff_scores.numel():
        finite_diff_scores = diff_scores[torch.isfinite(diff_scores)]
        if finite_diff_scores.numel():
            quantile_scores = _bounded_quantile_input(finite_diff_scores)
            hard_negative_p90 = torch.quantile(quantile_scores, 0.90)
            hard_negative_p95 = torch.quantile(quantile_scores, 0.95)
            hard_negative_max = finite_diff_scores.max()
        else:
            hard_negative_p90 = hard_negative_p95 = hard_negative_max = torch.tensor(float("nan"), device=sim.device)
    else:
        hard_negative_p90 = hard_negative_p95 = hard_negative_max = torch.tensor(float("nan"), device=sim.device)

    return RetrievalMetrics(
        knn_at_1=float(ks[1]),
        knn_at_5=float(ks[5]),
        knn_at_10=float(ks[10]),
        mean_average_precision=float(ap.mean().item()),
        mean_positive_rank=float(first_pos_rank.mean().item()),
        same_cosine_mean=float(same_mean.item()),
        different_cosine_mean=float(diff_mean.item()),
        same_diff_cosine_gap=float(gap.item()),
        hard_negative_cosine_p90=float(hard_negative_p90.item()),
        hard_negative_cosine_p95=float(hard_negative_p95.item()),
        hard_negative_cosine_max=float(hard_negative_max.item()),
        num_queries=int(query_idx.numel()),
        num_eval_images=int(n),
    )
