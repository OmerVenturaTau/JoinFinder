"""Symmetric, retrieval-first fusion for independently useful modalities."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricRetrievalFusion(nn.Module):
    """Concatenate mask-pooled modality summaries, with optional adapters."""

    MODALITIES = ("tile", "glyph", "word")

    def __init__(self, d_model: int, enabled_modalities: Tuple[str, ...],
                 reliability_hidden_dim: int = 64, *,
                 use_adapters: bool = False,
                 branch_dim: Optional[int] = None) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.use_adapters = bool(use_adapters)
        self.branch_dim = int(branch_dim) if branch_dim is not None else self.d_model
        if self.branch_dim <= 0:
            raise ValueError("branch_dim must be positive")
        self.enabled_modalities = tuple(enabled_modalities)
        if not self.enabled_modalities:
            raise ValueError("SymmetricRetrievalFusion requires at least one modality")
        if self.use_adapters:
            self.adapters = nn.ModuleDict({name: nn.Sequential(
                nn.LayerNorm(self.d_model),
                nn.Linear(self.d_model, self.branch_dim, bias=False),
            ) for name in self.enabled_modalities})
        # Shared scorer means identical capacity and no modality-specific prior.
        self.reliability = nn.Sequential(
            nn.Linear(3, reliability_hidden_dim), nn.GELU(),
            nn.Linear(reliability_hidden_dim, 1, bias=False),
        )
        nn.init.zeros_(self.reliability[-1].weight)  # equal weights initially
        output_branch_dim = self.branch_dim if self.use_adapters else self.d_model
        self.output_dim = output_branch_dim * len(self.enabled_modalities)

    @staticmethod
    def _pool(
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor],
        evidence_fraction: Optional[torch.Tensor] = None,
    ):
        if mask is None:
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        else:
            if mask.ndim != 2 or tuple(mask.shape) != tuple(tokens.shape[:2]):
                raise ValueError(
                    f"valid mask must have shape {tuple(tokens.shape[:2])}, "
                    f"got {tuple(mask.shape)}"
                )
            mask = mask.to(device=tokens.device, dtype=torch.bool)
        weights = mask.unsqueeze(-1).to(tokens.dtype)
        count = weights.sum(dim=1)
        pooled = (tokens * weights).sum(dim=1) / count.clamp_min(1.0)
        centered = (tokens - pooled.unsqueeze(1)) * weights
        variance = centered.square().sum(dim=(1, 2)) / (
            count.squeeze(-1).clamp_min(1.0) * tokens.shape[-1]
        )
        available = mask.any(dim=1)
        pooled = torch.where(available.unsqueeze(1), pooled, torch.zeros_like(pooled))
        if evidence_fraction is None:
            evidence_fraction = count.squeeze(-1) / max(1, tokens.shape[1])
        else:
            if evidence_fraction.ndim != 1 or evidence_fraction.shape[0] != tokens.shape[0]:
                raise ValueError(
                    "evidence_fraction must have shape "
                    f"({tokens.shape[0]},), got {tuple(evidence_fraction.shape)}"
                )
            evidence_fraction = evidence_fraction.to(
                device=tokens.device, dtype=tokens.dtype
            )
            if not torch.isfinite(evidence_fraction).all():
                raise ValueError("evidence_fraction must contain only finite values")
            if ((evidence_fraction < 0) | (evidence_fraction > 1)).any():
                raise ValueError("evidence_fraction values must be in [0, 1]")
        # An unavailable modality must never advertise evidence to the scorer.
        evidence_fraction = torch.where(
            available, evidence_fraction, torch.zeros_like(evidence_fraction)
        )
        stats = torch.stack((
            evidence_fraction,
            variance,
            pooled.float().norm(dim=1).to(tokens.dtype) / (tokens.shape[-1] ** 0.5),
        ), dim=1)
        return pooled, available, stats, evidence_fraction

    def forward(self, tile_tokens=None, tile_valid_mask=None, glyph_tokens=None,
                glyph_valid_mask=None, word_tokens=None, word_valid_mask=None,
                tile_evidence_fraction=None, glyph_evidence_fraction=None,
                word_evidence_fraction=None):
        supplied = {
            "tile": (tile_tokens, tile_valid_mask, tile_evidence_fraction),
            "glyph": (glyph_tokens, glyph_valid_mask, glyph_evidence_fraction),
            "word": (word_tokens, word_valid_mask, word_evidence_fraction),
        }
        reference = next(
            (tokens for tokens, _, _ in supplied.values() if tokens is not None),
            None,
        )
        if reference is None:
            raise ValueError("SymmetricRetrievalFusion received no modality tensors")
        batch_size = reference.shape[0]
        embeddings, availability, scores, evidence_fractions = [], [], [], []
        local_token_weights: Dict[str, torch.Tensor] = {}
        details: Dict[str, torch.Tensor] = {}
        for name in self.enabled_modalities:
            tokens, mask, evidence_fraction = supplied[name]
            if tokens is None:
                pooled = torch.zeros(batch_size, self.d_model, device=reference.device, dtype=reference.dtype)
                available = torch.zeros(batch_size, dtype=torch.bool, device=reference.device)
                stats = torch.zeros(batch_size, 3, device=reference.device, dtype=reference.dtype)
                effective_evidence = torch.zeros(
                    batch_size, device=reference.device, dtype=reference.dtype
                )
            else:
                if tokens.shape[0] != batch_size:
                    raise ValueError(f"{name} batch size differs from other modalities")
                pooled, available, stats, effective_evidence = self._pool(
                    tokens, mask, evidence_fraction
                )
                if mask is None:
                    token_mask = torch.ones(
                        tokens.shape[:2], dtype=torch.bool, device=tokens.device
                    )
                else:
                    if mask.ndim != 2 or tuple(mask.shape) != tuple(tokens.shape[:2]):
                        raise ValueError(
                            f"{name}_valid_mask must have shape {tuple(tokens.shape[:2])}, "
                            f"got {tuple(mask.shape)}"
                        )
                    token_mask = mask.to(device=tokens.device, dtype=torch.bool)
                token_weights = token_mask.to(tokens.dtype)
                token_weights = token_weights / token_weights.sum(
                    dim=1, keepdim=True
                ).clamp_min(1.0)
                local_token_weights[name] = token_weights
            if self.use_adapters:
                embedded = F.normalize(self.adapters[name](pooled), dim=1, eps=1e-8)
            else:
                # Adapter-free path keeps the native pooled representation.
                embedded = F.normalize(pooled, dim=1, eps=1e-8)
            embedded = torch.where(available.unsqueeze(1), embedded, torch.zeros_like(embedded))
            embeddings.append(embedded)
            availability.append(available)
            scores.append(self.reliability(stats).squeeze(1))
            evidence_fractions.append(effective_evidence)
            details[name] = embedded

        available_t = torch.stack(availability, dim=1)
        score_t = torch.stack(scores, dim=1).masked_fill(~available_t, -torch.inf)
        all_empty = ~available_t.any(dim=1)
        if all_empty.any():
            score_t = score_t.clone()
            score_t[all_empty] = 0.0
        reliability = torch.softmax(score_t, dim=1)
        reliability = torch.where(available_t, reliability, torch.zeros_like(reliability))
        reliability = reliability / reliability.sum(dim=1, keepdim=True).clamp_min(1e-8)
        branch_stack = torch.stack(embeddings, dim=1)
        # sqrt(reliability) makes each branch's squared norm—and therefore its
        # contribution to cosine similarity—equal to its reliability weight.
        # Clamp before sqrt to keep gradients finite for missing modalities,
        # then restore their exact zero contribution explicitly.
        reliability_scale = reliability.clamp_min(1e-8).sqrt()
        reliability_scale = torch.where(
            available_t, reliability_scale, torch.zeros_like(reliability_scale)
        )
        weighted = branch_stack * reliability_scale.unsqueeze(-1)
        fused_latent = F.normalize(weighted.flatten(1), dim=1, eps=1e-8)

        # Expose the exact reliability-weighted contribution used by attention
        # visualizations. A source token receives its modality reliability
        # multiplied by its normalized mask-mean weight.
        # The resulting [B, 1, total_tokens] tensor follows the same concatenated
        # token layout as the other fusion modules and sums to one for every
        # sample with at least one available modality.
        token_contributions = []
        for modality_idx, name in enumerate(self.enabled_modalities):
            if name in local_token_weights:
                token_contributions.append(
                    local_token_weights[name] * reliability[:, modality_idx].unsqueeze(1)
                )
        attention = (
            torch.cat(token_contributions, dim=1).unsqueeze(1)
            if token_contributions
            else reference.new_zeros(batch_size, 1, 0)
        )
        details.update({"fusion": fused_latent,
                        "reliability_weights": reliability,
                        "modality_available": available_t,
                        "evidence_fractions": torch.stack(evidence_fractions, dim=1),
                        "attention": attention})
        return fused_latent, details


class SymmetricRetrievalHead(nn.Module):
    """Classification head that does not transform the retrieval embedding."""

    def __init__(self, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        # Sequential preserves the legacy ``head.classifier[-1]`` checkpoint
        # metadata contract while remaining classification-only.
        self.classifier = nn.Sequential(nn.Linear(embedding_dim, num_classes))

    def forward(self, latent: torch.Tensor):
        return self.classifier(latent), latent
