"""
Self-Attention Transformer Fusion Module.

Concatenates all modality tokens (tiles, glyph summaries, word lines),
adds learnable modality-type embeddings so the model can distinguish
token origins, prepends a [CLS] token, and runs self-attention.

Returns [B, 1, d_model] (the contextualised CLS token) which is
compatible with PerceiverHead via its pool → project → classify pipeline.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from system import (
    D_MODEL,
    TRANSFORMER_NUM_LAYERS,
    TRANSFORMER_NUM_HEADS,
    TRANSFORMER_DIM_FEEDFORWARD,
    TRANSFORMER_DROPOUT,
    TRANSFORMER_USE_RESIDUAL_POOL,
    TRANSFORMER_RESIDUAL_CLS_GATE,
    TRANSFORMER_RESIDUAL_TILE_WEIGHT,
    TRANSFORMER_RESIDUAL_GLYPH_WEIGHT,
    TRANSFORMER_RESIDUAL_WORD_WEIGHT,
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


class TransformerFusion(nn.Module):
    """
    Self-Attention Transformer Fusion.

    Architecture:
    1. Concatenate modality tokens with learned modality-type embeddings
    2. Prepend a learnable [CLS] token
    3. Run pre-norm TransformerEncoder (self-attention over all tokens)
    4. Return the contextualised [CLS] token as [B, 1, d_model]

    The modality-type embeddings let the model distinguish tile vs glyph
    vs word tokens even though they share the same d_model dimension.
    A null-token safety fallback (matching PerceiverFusion) handles the
    edge case where all modality tokens are masked out.
    """

    # Modality type IDs (used for modality_type_embed)
    TYPE_TILE = 0
    TYPE_GLYPH = 1
    TYPE_WORD = 2
    NUM_MODALITY_TYPES = 3

    def __init__(
        self,
        d_model: int = D_MODEL,
        num_layers: int = TRANSFORMER_NUM_LAYERS,
        nhead: int = TRANSFORMER_NUM_HEADS,
        dim_feedforward: int = TRANSFORMER_DIM_FEEDFORWARD,
        dropout: float = TRANSFORMER_DROPOUT,
        use_residual_pool: bool = TRANSFORMER_USE_RESIDUAL_POOL,
        residual_cls_gate: float = TRANSFORMER_RESIDUAL_CLS_GATE,
        residual_tile_weight: float = TRANSFORMER_RESIDUAL_TILE_WEIGHT,
        residual_glyph_weight: float = TRANSFORMER_RESIDUAL_GLYPH_WEIGHT,
        residual_word_weight: float = TRANSFORMER_RESIDUAL_WORD_WEIGHT,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_residual_pool = bool(use_residual_pool)
        self.residual_cls_gate = float(min(1.0, max(0.0, residual_cls_gate)))
        self.residual_modality_weights = {
            self.TYPE_TILE: float(max(0.0, residual_tile_weight)),
            self.TYPE_GLYPH: float(max(0.0, residual_glyph_weight)),
            self.TYPE_WORD: float(max(0.0, residual_word_weight)),
        }

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Modality-type embeddings (tile / glyph / word)
        self.modality_type_embed = nn.Embedding(self.NUM_MODALITY_TYPES, d_model)

        # Numerical-safety fallback (non-learnable) for samples where every
        # modality token is masked.  Matches PerceiverFusion's null_token.
        self.register_buffer("null_token", torch.zeros(1, 1, d_model), persistent=False)

        # Pre-norm Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm is generally more stable
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        # Final layer-norm applied to CLS output
        self.final_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.modality_type_embed.weight, std=0.02)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _add_modality_type(
        self,
        tokens: torch.Tensor,   # [B, S, d_model]
        type_id: int,
    ) -> torch.Tensor:
        """Add a modality-type embedding to every position in *tokens*."""
        type_emb = self.modality_type_embed(
            torch.tensor(type_id, device=tokens.device)
        )  # [d_model]
        return tokens + type_emb.unsqueeze(0).unsqueeze(0)  # broadcast [1, 1, d_model]

    def _masked_mean(
        self,
        tokens: torch.Tensor,      # [B, S, d_model]
        valid_mask: torch.Tensor,  # [B, S] bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean-pool valid tokens for one modality, returning (pooled, available)."""
        mask = valid_mask.to(device=tokens.device, dtype=torch.bool)
        mask_f = mask.unsqueeze(-1).to(tokens.dtype)
        count = mask_f.sum(dim=1)  # [B, 1]
        pooled = (tokens * mask_f).sum(dim=1) / count.clamp_min(1.0)
        available = count.squeeze(1) > 0
        pooled = torch.where(available.unsqueeze(1), pooled, torch.zeros_like(pooled))
        return pooled, available

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        tile_tokens: Optional[torch.Tensor] = None,       # [B, N, d_model]
        tile_valid_mask: Optional[torch.Tensor] = None,    # [B, N] bool
        glyph_tokens: Optional[torch.Tensor] = None,      # [B, M, d_model]
        glyph_valid_mask: Optional[torch.Tensor] = None,   # [B, M] bool
        word_tokens: Optional[torch.Tensor] = None,        # [B, L, d_model]
        word_valid_mask: Optional[torch.Tensor] = None,     # [B, L] bool
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            tile_tokens:      Visual tile tokens
            tile_valid_mask:  True = valid, False = padding
            glyph_tokens:     Glyph summary tokens
            glyph_valid_mask: True = valid, False = padding
            word_tokens:      Word/line tokens
            word_valid_mask:  True = valid, False = padding
            return_attention: If True, extract and return self-attention
                weights from the last TransformerEncoder layer.

        Returns:
            cls_out:   [B, 1, d_model]  contextualised CLS token
            attention: [B, 1, total_modality_tokens] CLS-to-modality attention
                       from the last self-attention layer (averaged over heads),
                       in the same format as PerceiverFusion output.
                       None when return_attention=False.
        """
        # 1. Collect tokens + masks, adding modality-type embeddings ----------
        tokens_list: list[torch.Tensor] = []
        masks_list: list[torch.Tensor] = []
        pooled_list: list[torch.Tensor] = []
        available_list: list[torch.Tensor] = []
        residual_weight_list: list[float] = []
        B: Optional[int] = None
        device: Optional[torch.device] = None
        dtype: Optional[torch.dtype] = None

        if tile_tokens is not None:
            B = _validate_modality_tokens(tile_tokens, self.d_model, B, "tile_tokens")
            device, dtype = _validate_token_compatibility(tile_tokens, device, dtype, "tile_tokens")
            tile_valid = _normalize_valid_mask(tile_valid_mask, tile_tokens, "tile_valid_mask")
            tokens_list.append(self._add_modality_type(tile_tokens, self.TYPE_TILE))
            masks_list.append(tile_valid)
            if self.use_residual_pool:
                pooled, available = self._masked_mean(tile_tokens, tile_valid)
                pooled_list.append(pooled)
                available_list.append(available)
                residual_weight_list.append(self.residual_modality_weights[self.TYPE_TILE])

        if glyph_tokens is not None:
            B = _validate_modality_tokens(glyph_tokens, self.d_model, B, "glyph_tokens")
            device, dtype = _validate_token_compatibility(glyph_tokens, device, dtype, "glyph_tokens")
            glyph_valid = _normalize_valid_mask(glyph_valid_mask, glyph_tokens, "glyph_valid_mask")
            tokens_list.append(self._add_modality_type(glyph_tokens, self.TYPE_GLYPH))
            masks_list.append(glyph_valid)
            if self.use_residual_pool:
                pooled, available = self._masked_mean(glyph_tokens, glyph_valid)
                pooled_list.append(pooled)
                available_list.append(available)
                residual_weight_list.append(self.residual_modality_weights[self.TYPE_GLYPH])

        if word_tokens is not None:
            B = _validate_modality_tokens(word_tokens, self.d_model, B, "word_tokens")
            device, dtype = _validate_token_compatibility(word_tokens, device, dtype, "word_tokens")
            word_valid = _normalize_valid_mask(word_valid_mask, word_tokens, "word_valid_mask")
            tokens_list.append(self._add_modality_type(word_tokens, self.TYPE_WORD))
            masks_list.append(word_valid)
            if self.use_residual_pool:
                pooled, available = self._masked_mean(word_tokens, word_valid)
                pooled_list.append(pooled)
                available_list.append(available)
                residual_weight_list.append(self.residual_modality_weights[self.TYPE_WORD])

        if B is None:
            # No modality tokens at all (shouldn't happen in training).
            return torch.zeros(0, 1, self.d_model, device=device if device else self.cls_token.device), None
        if B == 0:
            return torch.zeros(0, 1, self.d_model, device=device if device else self.cls_token.device), None

        residual_pool = None
        if self.use_residual_pool and pooled_list:
            pooled_stack = torch.stack(pooled_list, dim=1)  # [B, n_modalities, d_model]
            available_stack = torch.stack(available_list, dim=1)  # [B, n_modalities]
            prior_weights = torch.tensor(
                residual_weight_list,
                device=pooled_stack.device,
                dtype=pooled_stack.dtype,
            ).view(1, -1, 1)
            available_f = available_stack.unsqueeze(-1).to(pooled_stack.dtype)
            effective_weights = available_f * prior_weights
            residual_pool = (pooled_stack * effective_weights).sum(dim=1)
            residual_weight_sum = effective_weights.sum(dim=1).clamp_min(1.0)
            residual_pool = residual_pool / residual_weight_sum
            any_available = available_stack.any(dim=1)
            residual_pool = torch.where(
                any_available.unsqueeze(1),
                residual_pool,
                torch.zeros_like(residual_pool),
            )

        # 2. Prepend [CLS] token (always valid) ------------------------------
        cls_tokens = self.cls_token.expand(B, -1, -1)           # [B, 1, d_model]
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=device)

        tokens_list.insert(0, cls_tokens)
        masks_list.insert(0, cls_mask)

        all_tokens = torch.cat(tokens_list, dim=1)              # [B, 1+S, d_model]
        valid_mask = torch.cat(masks_list, dim=1)               # [B, 1+S] bool (True = valid)

        # 3. Null-token safety (same logic as PerceiverFusion) ----------------
        # Track total modality tokens (excluding CLS and null) for attention slicing.
        total_modality_len = all_tokens.shape[1] - 1  # minus CLS at position 0

        modality_mask = valid_mask[:, 1:]  # exclude CLS
        all_empty = (~modality_mask).all(dim=1) if modality_mask.numel() > 0 else torch.ones(B, dtype=torch.bool, device=device)
        if all_empty.any():
            null_expanded = self.null_token.expand(B, -1, -1)   # [B, 1, d_model]
            null_mask = all_empty[:, None]  # True = valid only for samples with no real modality tokens.
            all_tokens = torch.cat([all_tokens, null_expanded], dim=1)
            valid_mask = torch.cat([valid_mask, null_mask], dim=1)

        # 4. Self-attention ---------------------------------------------------
        # PyTorch convention: src_key_padding_mask True = IGNORE
        padding_mask = ~valid_mask                               # [B, 1+S(+1)]

        attention_weights = None

        if not return_attention:
            # Fast path (training): single call, no attention extraction.
            x_out = self.transformer(
                all_tokens,
                src_key_padding_mask=padding_mask,
            )  # [B, 1+S(+1), d_model]
        else:
            # Visualization path: run layers individually so we can extract
            # self-attention weights from the last layer.
            x = all_tokens
            layers = self.transformer.layers
            for i, layer in enumerate(layers):
                if i < len(layers) - 1:
                    # Non-last layers: normal forward
                    x = layer(x, src_key_padding_mask=padding_mask)
                else:
                    # Last layer: replicate pre-norm TransformerEncoderLayer
                    # forward, but call self_attn directly to get weights.
                    # (norm_first=True layout)
                    x_normed = layer.norm1(x)
                    attn_out, attn_w = layer.self_attn(
                        x_normed, x_normed, x_normed,
                        key_padding_mask=padding_mask,
                        need_weights=True,
                        average_attn_weights=False,  # per-head → forces math path
                    )
                    x = x + layer.dropout1(attn_out)
                    x = x + layer._ff_block(layer.norm2(x))

                    # Average over heads: [B, nhead, S, S] → [B, S, S]
                    if attn_w is not None and attn_w.dim() == 4:
                        attn_w = attn_w.mean(dim=1)  # [B, 1+S(+1), 1+S(+1)]
                    attention_weights = attn_w
            x_out = x

        # 5. Extract & normalise export token ---------------------------------
        cls_token_out = x_out[:, 0, :]  # [B, d_model]
        if residual_pool is not None:
            gate = self.residual_cls_gate
            export_token = (1.0 - gate) * residual_pool + gate * cls_token_out
        else:
            export_token = cls_token_out
        cls_out = self.final_norm(export_token).unsqueeze(1)    # [B, 1, d_model]

        # 6. Reshape attention to Perceiver-compatible format ------------------
        # PerceiverFusion returns [B, num_latents, total_modality_tokens] where
        # columns are [tiles, glyphs, words].
        # TransformerFusion's raw self-attention is [B, 1+S(+null), 1+S(+null)].
        # Extract CLS row (row 0), keep only the modality-token columns
        # (indices 1..1+total_modality_len), skip CLS-self and null columns.
        # Result: [B, 1, total_modality_len] — same contract as PerceiverFusion.
        if attention_weights is not None:
            # CLS token is at position 0; modality tokens at 1..total_modality_len
            cls_attn = attention_weights[:, 0, 1:1 + total_modality_len]  # [B, S]
            attention_weights = cls_attn.unsqueeze(1)  # [B, 1, S]

        return cls_out, attention_weights
