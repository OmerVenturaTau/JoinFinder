"""
Perceiver-Style Latent Fusion Module.

This module implements a Perceiver-style cross-attention fusion that:
1. Takes tokens from multiple modalities (tiles, glyph summaries, word lines)
2. Uses learnable latent queries to attend to all modality tokens
3. Outputs a fixed-size latent representation for classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from system import (
    D_MODEL,
    NUM_HEADS,
    DROPOUT,
    LATENT_DIM,
    PERCEIVER_NUM_LATENTS,
    PERCEIVER_NUM_CROSS_ATTN_LAYERS,
    PERCEIVER_POOLING,
    MAX_SELECTED_MANUSCRIPTS,
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


class PerceiverFusion(nn.Module):
    """
    Perceiver-style latent fusion using cross-attention.
    
    Architecture:
    - Learnable latent queries (fixed size)
    - Modality-type embeddings (tile / glyph / word) added to tokens
    - Cross-attention: queries attend to all modality tokens
    - Output: fixed-size latent representation [B, num_latents, d_model]
    """

    # Modality type IDs (shared convention with TransformerFusion)
    TYPE_TILE = 0
    TYPE_GLYPH = 1
    TYPE_WORD = 2
    NUM_MODALITY_TYPES = 3

    def __init__(
        self,
        d_model: int = D_MODEL,
        num_latents: int = PERCEIVER_NUM_LATENTS,
        num_heads: int = NUM_HEADS,
        num_cross_attn_layers: int = PERCEIVER_NUM_CROSS_ATTN_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_latents = num_latents
        
        # Learnable latent queries (initialized randomly)
        self.latent_queries = nn.Parameter(torch.randn(1, num_latents, d_model))
        nn.init.trunc_normal_(self.latent_queries, std=0.02)

        # Modality-type embeddings (tile / glyph / word) so cross-attention
        # can distinguish which modality each key/value token belongs to.
        self.modality_type_embed = nn.Embedding(self.NUM_MODALITY_TYPES, d_model)
        nn.init.trunc_normal_(self.modality_type_embed.weight, std=0.02)

        # A single "null" token (all zeros) used ONLY as a numerical safety fallback when
        # a batch contains samples where all modality tokens are masked out.
        #
        # - Not learnable (buffer), so it cannot become a shortcut feature.
        # - Not always appended; only used when needed to prevent NaNs.
        self.register_buffer("null_token", torch.zeros(1, 1, d_model), persistent=False)
        
        # Cross-attention layers (queries attend to modality tokens)
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_cross_attn_layers)
        ])
        
        # Layer norms for each cross-attention layer
        self.latent_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_cross_attn_layers)
        ])
        self.modality_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_cross_attn_layers)
        ])
        
        # Feedforward networks for latent queries (after cross-attention)
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(num_cross_attn_layers)
        ])
        self.ff_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_cross_attn_layers)
        ])
    
    def _add_modality_type(
        self,
        tokens: torch.Tensor,  # [B, S, d_model]
        type_id: int,
    ) -> torch.Tensor:
        """Add a modality-type embedding to every position in *tokens*."""
        type_emb = self.modality_type_embed(
            torch.tensor(type_id, device=tokens.device)
        )  # [d_model]
        return tokens + type_emb.unsqueeze(0).unsqueeze(0)  # broadcast [1, 1, d_model]

    def forward(
        self,
        tile_tokens: Optional[torch.Tensor] = None,  # [B, N, d_model]
        tile_valid_mask: Optional[torch.Tensor] = None,  # [B, N] bool
        glyph_tokens: Optional[torch.Tensor] = None,  # [B, M, d_model] (summary tokens)
        glyph_valid_mask: Optional[torch.Tensor] = None,  # [B, M] bool
        word_tokens: Optional[torch.Tensor] = None,  # [B, L, d_model] (line tokens)
        word_valid_mask: Optional[torch.Tensor] = None,  # [B, L] bool
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through Perceiver-style fusion.
        
        Args:
            tile_tokens: Optional tile tokens [B, N, d_model]
            tile_valid_mask: Optional valid mask for tiles [B, N] bool (True = valid)
            glyph_tokens: Optional glyph summary tokens [B, M, d_model]
            glyph_valid_mask: Optional valid mask for glyphs [B, M] bool (True = valid)
            word_tokens: Optional word line tokens [B, L, d_model]
            word_valid_mask: Optional valid mask for words [B, L] bool (True = valid)
            
        Returns:
            Tuple of:
            - latent_repr: [B, num_latents, d_model] - fused latent representation
            - attention_weights: Optional attention weights from last layer
        """
        B = None
        device: Optional[torch.device] = None
        dtype: Optional[torch.dtype] = None
        
        # Determine batch size and device from available inputs
        for name, tokens in (
            ("tile_tokens", tile_tokens),
            ("glyph_tokens", glyph_tokens),
            ("word_tokens", word_tokens),
        ):
            if tokens is not None:
                B = _validate_modality_tokens(tokens, self.d_model, B, name)
                device, dtype = _validate_token_compatibility(tokens, device, dtype, name)
        
        if B is None:
            raise ValueError("At least one modality must be provided")
        
        # Concatenate all modality tokens with modality-type embeddings.
        # Mask convention: valid_mask True = valid token; PyTorch key_padding_mask True = ignore.
        # So we pass ~valid_mask as key_padding_mask (invert once at concat).
        modality_tokens = []
        modality_mask = []
        
        if tile_tokens is not None:
            modality_tokens.append(self._add_modality_type(tile_tokens, self.TYPE_TILE))
            modality_mask.append(~_normalize_valid_mask(tile_valid_mask, tile_tokens, "tile_valid_mask"))
        
        if glyph_tokens is not None:
            modality_tokens.append(self._add_modality_type(glyph_tokens, self.TYPE_GLYPH))
            modality_mask.append(~_normalize_valid_mask(glyph_valid_mask, glyph_tokens, "glyph_valid_mask"))
        
        if word_tokens is not None:
            modality_tokens.append(self._add_modality_type(word_tokens, self.TYPE_WORD))
            modality_mask.append(~_normalize_valid_mask(word_valid_mask, word_tokens, "word_valid_mask"))
        
        if not modality_tokens:
            raise ValueError("At least one modality must be provided")

        # Concatenate all modality tokens
        all_modality_tokens = torch.cat(modality_tokens, dim=1)  # [B, total_tokens, d_model]
        all_modality_mask = torch.cat(modality_mask, dim=1)  # [B, total_tokens] bool (True = padding)
        total_modality_len = all_modality_tokens.shape[1]

        # Numerical safety: if any sample has ALL keys masked, attention softmax can produce NaNs.
        # Append a single unmasked zero token in that case (for the whole batch).
        all_empty = all_modality_mask.all(dim=1)
        if all_empty.any():
            all_modality_tokens = torch.cat([all_modality_tokens, self.null_token.expand(B, -1, -1)], dim=1)  # [B, total+1, d_model]
            null_padding_mask = ~all_empty[:, None]  # True = ignore for samples that already have real tokens.
            all_modality_mask = torch.cat([all_modality_mask, null_padding_mask], dim=1)  # [B, total+1]
        
        # Expand latent queries to batch size
        latents = self.latent_queries.expand(B, -1, -1)  # [B, num_latents, d_model]
        
        # Apply cross-attention layers
        attention_weights = None
        for i, (cross_attn, latent_norm, modality_norm, ff, ff_norm) in enumerate(
            zip(self.cross_attn_layers, self.latent_norms, self.modality_norms, self.ff_layers, self.ff_norms)
        ):
            # Normalize inputs
            latents_norm = latent_norm(latents)  # [B, num_latents, d_model]
            modality_normed = modality_norm(all_modality_tokens)  # [B, total_tokens, d_model]
            
            # Cross-attention: latents (queries) attend to modality tokens (keys/values)
            # Only request weights when needed for visualization; otherwise PyTorch
            # can use the faster attention path and avoid materializing weights.
            is_last = (i == len(self.cross_attn_layers) - 1)
            need_attention = bool(return_attention and is_last)
            latents_attn, attn_weights = cross_attn(
                query=latents_norm,  # [B, num_latents, d_model]
                key=modality_normed,  # [B, total_tokens, d_model]
                value=modality_normed,  # [B, total_tokens, d_model]
                key_padding_mask=all_modality_mask,  # [B, total_tokens] bool (True = ignore)
                need_weights=need_attention,
                average_attn_weights=False,
            )  # [B, num_latents, d_model]
            
            # Residual connection
            latents = latents + latents_attn
            
            # Feedforward on latents
            latents = latents + ff(ff_norm(latents))
            
            # Save attention weights from last layer
            if is_last:
                if attn_weights is not None and attn_weights.dim() == 4:
                    # Per-head weights [B, num_heads, num_latents, total_tokens] → average over heads
                    attention_weights = attn_weights.mean(dim=1)  # [B, num_latents, total_tokens]
                else:
                    attention_weights = attn_weights

        if attention_weights is not None:
            attention_weights = attention_weights[..., :total_modality_len]
        
        return latents, attention_weights  # [B, num_latents, d_model], [B, num_latents, total_tokens]


class PerceiverHead(nn.Module):
    """
    Head for manuscript classification after Perceiver fusion.
    
    Takes the latent representation from PerceiverFusion and:
    1. Pools latents (mean or CLS-style)
    2. Projects to latent_dim
    3. Classifies to num_classes
    """
    def __init__(
        self,
        d_model: int = D_MODEL,
        num_latents: int = PERCEIVER_NUM_LATENTS,
        latent_dim: int = LATENT_DIM,
        num_classes: int = MAX_SELECTED_MANUSCRIPTS,
        pooling: str = PERCEIVER_POOLING,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.pooling = pooling
        
        # Pooling: reduce [B, num_latents, d_model] to [B, d_model]
        if pooling == "cls":
            # Use first latent as CLS token
            self.pool = lambda x: x[:, 0]  # [B, d_model]
        elif pooling == "mean":
            self.pool = lambda x: x.mean(dim=1)  # [B, d_model]
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        
        # Project to latent_dim
        self.norm = nn.LayerNorm(d_model)
        self.latent_proj = nn.Sequential(
            nn.Linear(d_model, 1536),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1536, latent_dim),
        )
        self.latent_bn = nn.BatchNorm1d(latent_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.GELU(),
            nn.Linear(latent_dim, num_classes),
        )
    
    def forward(
        self,
        latent_repr: torch.Tensor,  # [B, num_latents, d_model]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through head.
        
        Args:
            latent_repr: [B, num_latents, d_model] from PerceiverFusion
            
        Returns:
            Tuple of:
            - logits: [B, num_classes]
            - latent: [B, latent_dim]
        """
        # Pool latents
        pooled = self.pool(latent_repr)  # [B, d_model]
        
        # Normalize and project to latent_dim
        pooled = self.norm(pooled)
        latent = self.latent_proj(pooled)  # [B, latent_dim]
        if self.training and latent.shape[0] == 1:
            latent = F.batch_norm(
                latent,
                self.latent_bn.running_mean,
                self.latent_bn.running_var,
                self.latent_bn.weight,
                self.latent_bn.bias,
                training=False,
                eps=self.latent_bn.eps,
            )
        else:
            latent = self.latent_bn(latent)
        
        # Classify
        logits = self.classifier(latent)  # [B, num_classes]
        
        return logits, latent
