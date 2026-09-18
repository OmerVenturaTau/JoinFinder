"""
VLAD-style fusion for branch token sets.

This module aggregates local tile/glyph/word descriptors with a differentiable
NetVLAD-like residual encoding. It is intentionally implemented as a fusion
module so the existing branches, masks, head, training loop, and projection
scripts can stay unchanged.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from system import (
    D_MODEL,
    DROPOUT,
    VLAD_ASSIGNMENT_TEMPERATURE,
    VLAD_NUM_CLUSTERS,
    VLAD_NORMALIZE_INPUT,
)


def _validate_modality_tokens(
    tokens: torch.Tensor,
    d_model: int,
    batch_size: Optional[int],
    name: str,
) -> int:
    if tokens.ndim != 3 or tokens.shape[-1] != d_model:
        raise ValueError(
            f"{name} must have shape [B, S, {d_model}], got {tuple(tokens.shape)}."
        )
    if batch_size is not None and tokens.shape[0] != batch_size:
        raise ValueError(
            f"{name} batch size must match other modalities: expected {batch_size}, got {tokens.shape[0]}."
        )
    return int(tokens.shape[0])


def _normalize_valid_mask(
    mask: Optional[torch.Tensor],
    tokens: torch.Tensor,
    name: str,
) -> torch.Tensor:
    expected_shape = tokens.shape[:2]
    if mask is None:
        return torch.ones(expected_shape, dtype=torch.bool, device=tokens.device)
    if mask.ndim != 2 or tuple(mask.shape) != tuple(expected_shape):
        raise ValueError(
            f"{name} must have shape {tuple(expected_shape)} to match tokens, got {tuple(mask.shape)}."
        )
    return mask.to(device=tokens.device, dtype=torch.bool)


def _validate_token_compatibility(
    tokens: torch.Tensor,
    expected_device: Optional[torch.device],
    expected_dtype: Optional[torch.dtype],
    name: str,
) -> Tuple[torch.device, torch.dtype]:
    if expected_device is None:
        return tokens.device, tokens.dtype
    if tokens.device != expected_device:
        raise ValueError(
            f"{name} device must match other modalities: expected {expected_device}, got {tokens.device}."
        )
    if tokens.dtype != expected_dtype:
        raise ValueError(
            f"{name} dtype must match other modalities: expected {expected_dtype}, got {tokens.dtype}."
        )
    return expected_device, expected_dtype


class VLADFusion(nn.Module):
    """
    Mask-aware NetVLAD-style fusion over multimodal local descriptors.

    Architecture:
    1. Add learned modality-type embeddings to tile/glyph/word tokens.
    2. Soft-assign every valid token to K learned visual-word centers.
    3. Accumulate assignment-weighted residuals ``x - c_k`` per center.
    4. Intra-normalize per center, flatten, normalize, and project to d_model.

    Returns ``[B, 1, d_model]`` so it is compatible with ``PerceiverHead``.
    """

    TYPE_TILE = 0
    TYPE_GLYPH = 1
    TYPE_WORD = 2
    NUM_MODALITY_TYPES = 3

    def __init__(
        self,
        d_model: int = D_MODEL,
        num_clusters: int = VLAD_NUM_CLUSTERS,
        assignment_temperature: float = VLAD_ASSIGNMENT_TEMPERATURE,
        normalize_input: bool = VLAD_NORMALIZE_INPUT,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_clusters = num_clusters
        self.assignment_temperature = float(assignment_temperature)
        self.normalize_input = bool(normalize_input)

        self.modality_type_embed = nn.Embedding(self.NUM_MODALITY_TYPES, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.assignment = nn.Linear(d_model, num_clusters)
        self.clusters = nn.Parameter(torch.empty(num_clusters, d_model))
        self.output_proj = nn.Sequential(
            nn.Linear(num_clusters * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.output_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.modality_type_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.clusters, std=0.02)
        nn.init.xavier_uniform_(self.assignment.weight)
        nn.init.zeros_(self.assignment.bias)

    def _add_modality_type(self, tokens: torch.Tensor, type_id: int) -> torch.Tensor:
        type_emb = self.modality_type_embed(
            torch.tensor(type_id, device=tokens.device)
        )
        return tokens + type_emb.view(1, 1, -1)

    def _collect_tokens(
        self,
        tile_tokens: Optional[torch.Tensor],
        tile_valid_mask: Optional[torch.Tensor],
        glyph_tokens: Optional[torch.Tensor],
        glyph_valid_mask: Optional[torch.Tensor],
        word_tokens: Optional[torch.Tensor],
        word_valid_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens_list: list[torch.Tensor] = []
        masks_list: list[torch.Tensor] = []
        B: Optional[int] = None
        device: Optional[torch.device] = None
        dtype: Optional[torch.dtype] = None

        if tile_tokens is not None:
            B = _validate_modality_tokens(tile_tokens, self.d_model, B, "tile_tokens")
            device, dtype = _validate_token_compatibility(tile_tokens, device, dtype, "tile_tokens")
            tokens_list.append(self._add_modality_type(tile_tokens, self.TYPE_TILE))
            masks_list.append(_normalize_valid_mask(tile_valid_mask, tile_tokens, "tile_valid_mask"))

        if glyph_tokens is not None:
            B = _validate_modality_tokens(glyph_tokens, self.d_model, B, "glyph_tokens")
            device, dtype = _validate_token_compatibility(glyph_tokens, device, dtype, "glyph_tokens")
            tokens_list.append(self._add_modality_type(glyph_tokens, self.TYPE_GLYPH))
            masks_list.append(_normalize_valid_mask(glyph_valid_mask, glyph_tokens, "glyph_valid_mask"))

        if word_tokens is not None:
            B = _validate_modality_tokens(word_tokens, self.d_model, B, "word_tokens")
            device, dtype = _validate_token_compatibility(word_tokens, device, dtype, "word_tokens")
            tokens_list.append(self._add_modality_type(word_tokens, self.TYPE_WORD))
            masks_list.append(_normalize_valid_mask(word_valid_mask, word_tokens, "word_valid_mask"))

        if not tokens_list:
            raise ValueError("VLADFusion requires at least one modality tensor")

        tokens = torch.cat(tokens_list, dim=1)
        valid_mask = torch.cat(masks_list, dim=1).to(dtype=torch.bool, device=tokens.device)
        return tokens, valid_mask

    def forward(
        self,
        tile_tokens: Optional[torch.Tensor] = None,
        tile_valid_mask: Optional[torch.Tensor] = None,
        glyph_tokens: Optional[torch.Tensor] = None,
        glyph_valid_mask: Optional[torch.Tensor] = None,
        word_tokens: Optional[torch.Tensor] = None,
        word_valid_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        tokens, valid_mask = self._collect_tokens(
            tile_tokens=tile_tokens,
            tile_valid_mask=tile_valid_mask,
            glyph_tokens=glyph_tokens,
            glyph_valid_mask=glyph_valid_mask,
            word_tokens=word_tokens,
            word_valid_mask=word_valid_mask,
        )
        B, S, _ = tokens.shape

        x = self.input_norm(tokens)
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=-1)

        logits = self.assignment(x) / max(self.assignment_temperature, 1e-6)
        assignments = F.softmax(logits, dim=-1)
        assignments = assignments * valid_mask.unsqueeze(-1).to(assignments.dtype)

        residuals = x.unsqueeze(2) - self.clusters.view(1, 1, self.num_clusters, self.d_model)
        vlad = (assignments.unsqueeze(-1) * residuals).sum(dim=1)

        vlad = F.normalize(vlad, p=2, dim=-1)
        vlad = vlad.reshape(B, self.num_clusters * self.d_model)
        vlad = F.normalize(vlad, p=2, dim=-1)
        fused = self.output_norm(self.output_proj(vlad)).unsqueeze(1)

        attention = None
        if return_attention:
            token_scores = torch.linalg.vector_norm(
                assignments.unsqueeze(-1) * residuals,
                ord=2,
                dim=(2, 3),
            )
            token_scores = token_scores * valid_mask.to(token_scores.dtype)
            denom = token_scores.sum(dim=1, keepdim=True).clamp(min=1e-12)
            attention = (token_scores / denom).masked_fill(~valid_mask, 0.0).unsqueeze(1)

        return fused, attention
