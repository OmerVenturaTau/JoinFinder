"""
Tile Branch Architecture - New Architecture Implementation

This module implements the tile branch according to the new architecture:
1. Tile Visual Encoder: DINOv2 ViT-B/14, processes tiles as batch (B*N,3,H,W) → [B,N,768]
2. Tile Token Enrichment: proj(v) + pos_tile=f([x_tile,y_tile])
"""

import torch
import torch.nn as nn
import math
from typing import Optional
from system import (
    ENCODER_MODEL_NAME,
    TILE_ENCODE_CHUNK_SIZE,
    TILE_SIZE,
    D_MODEL,
    TILE_ENCODER_TYPE,
    TILE_CONVNEXT_MODEL_NAME,
    XML_PATCH_READING_DIRECTION_RTL,
    TILE_BRANCH_ENABLE_POS_ENCODING,
    TILE_BRANCH_USE_FOURIER_POS,
    TILE_BRANCH_NUM_FREQS,
    TILE_BRANCH_USE_TRANSFORMER,
    TILE_BRANCH_TRANSFORMER_LAYERS,
    TILE_BRANCH_TRANSFORMER_HEADS,
    TILE_BRANCH_TRANSFORMER_DIM_FEEDFORWARD,
    TILE_BRANCH_TRANSFORMER_DROPOUT,
    TILE_NUM_SUMMARY_TOKENS,
    TILE_MIN_SUMMARY_TOKENS,
    TILE_SUMMARIZER_CROSS_ATTN_LAYERS,
    TILE_SUMMARIZER_ATTENTION_HEADS,
    DROPOUT,
    USE_BRANCH_ADAPTERS_AND_SUMMARIZERS,
)

from .tile_backbones import create_tile_backbone


def _validate_tile_mask_shape(mask: torch.Tensor, expected_shape: torch.Size | tuple, context: str, name: str = "valid_mask") -> None:
    if mask.ndim != 2 or tuple(mask.shape) != tuple(expected_shape):
        raise ValueError(
            f"{context}: {name} must have shape {tuple(expected_shape)} to match the provided tiles/tokens; "
            f"got {tuple(mask.shape)}."
        )


def _validate_tile_coords_shape(coords: torch.Tensor, expected_shape: torch.Size | tuple, context: str) -> None:
    expected = tuple(expected_shape) + (2,)
    if coords.ndim != 3 or tuple(coords.shape) != expected:
        raise ValueError(
            f"{context}: tile_coords must have shape {expected} to match the provided tiles/tokens; "
            f"got {tuple(coords.shape)}."
        )


class TileVisualEncoder(nn.Module):
    """
    Tile Visual Encoder (selectable backbone).
    
    Processes tiles as batch: (B*N, 3, H, W) → [B, N, d_model]
    Backbone is selected via system.TILE_ENCODER_TYPE.
    """
    def __init__(
        self,
        tile_size: int = TILE_SIZE,
        encoder_model_name: str = ENCODER_MODEL_NAME,
        d_model: int = D_MODEL,
        pretrained: bool = True,
    ):
        super().__init__()

        # Pick backbone model name.
        # Keep backward compatibility: if caller passes the DINO default but the
        # config requests ConvNeXt, prefer the ConvNeXt default.
        backbone_type = str(TILE_ENCODER_TYPE).lower().strip()
        model_name = encoder_model_name
        if backbone_type == "convnext" and encoder_model_name == ENCODER_MODEL_NAME:
            model_name = TILE_CONVNEXT_MODEL_NAME

        spec = create_tile_backbone(
            encoder_type=backbone_type,
            model_name=model_name,
            tile_size=tile_size,
            pretrained=pretrained,
        )
        self.tile_encoder = spec.model
        self.tile_feat_dim = spec.feat_dim
        
        # Projection to d_model if needed
        if self.tile_feat_dim != d_model:
            self.proj = nn.Linear(self.tile_feat_dim, d_model)
        else:
            self.proj = nn.Identity()
        
        self.d_model = d_model
    
    def forward(self, tiles: torch.Tensor, valid_mask: Optional[torch.Tensor] = None, return_attention: bool = False) -> torch.Tensor:
        """
        Encode tiles using the selected backbone.
        
        Args:
            tiles: [B, N, 3, H, W] - batch of tiles
            valid_mask: [B, N] (bool) - True for valid tiles, False for padding.
                If None, all tiles are treated as valid.
            return_attention: If True, extract and return attention weights
            
        Returns:
            tile_features: [B, N, d_model] - encoded tile features (zeros for padded tiles)
            attention_weights: None - DINOv2 attention extraction is complex, use fusion attention instead
        """
        B, N, C, H, W = tiles.shape
        if valid_mask is None:
            valid_mask = torch.ones(B, N, dtype=torch.bool, device=tiles.device)
        else:
            _validate_tile_mask_shape(valid_mask, (B, N), "TileVisualEncoder")
            valid_mask = valid_mask.to(device=tiles.device, dtype=torch.bool)
        tiles_flat = tiles.reshape(B * N, C, H, W)
        valid_mask_flat = valid_mask.reshape(-1)
        
        # Process tiles in chunks to control memory
        enc_feats = torch.zeros(B * N, self.tile_feat_dim, device=tiles.device, dtype=tiles.dtype)
        
        if valid_mask_flat.any():
            valid_idx = torch.nonzero(valid_mask_flat, as_tuple=False).squeeze(1)
            
            # Process in chunks
            for start in range(0, valid_idx.numel(), TILE_ENCODE_CHUNK_SIZE):
                end = min(start + TILE_ENCODE_CHUNK_SIZE, valid_idx.numel())
                idx_chunk = valid_idx[start:end]
                chunk_tiles = tiles_flat[idx_chunk]
                
                # Forward pass
                encoded = self.tile_encoder(chunk_tiles)  # [chunk_size, tile_feat_dim]
                enc_feats[idx_chunk] = encoded
        
        # Project to d_model
        enc_feats = self.proj(enc_feats)  # [B*N, d_model]
        
        # Reshape back to [B, N, d_model]
        enc_feats = enc_feats.view(B, N, self.d_model)
        
        # Zero out padded positions
        enc_feats = enc_feats * valid_mask.unsqueeze(-1).to(enc_feats.dtype)
        
        if return_attention:
            # Backbone attention extraction is intentionally not exposed here.
            # Instead, we return None and let forward_with_attention use fusion attention,
            # which shows which tiles are important for classification (more interpretable).
            return enc_feats, None
        
        return enc_feats  # [B, N, d_model]


class PositionalEncoding2D(nn.Module):
    """
    2D Positional encoding using Fourier features or MLP.
    
    Maps (x, y) coordinates in [0, 1] to d_model-dimensional embeddings.
    Supports both Fourier features and MLP-based encoding.
    
    For RTL reading order: transforms x-coordinate for positional encoding purposes
    so that large x (right side) is treated as "earlier" in reading order.
    Actual coordinates remain unchanged - only the positional encoding interpretation changes.
    """
    def __init__(
        self,
        d_model: int = D_MODEL,
        use_fourier: bool = TILE_BRANCH_USE_FOURIER_POS,
        num_freqs: int = TILE_BRANCH_NUM_FREQS,
        rtl: bool = XML_PATCH_READING_DIRECTION_RTL,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_fourier = use_fourier
        self.rtl = rtl
        
        if use_fourier:
            # Fourier features: sin/cos encoding
            # Generate frequencies
            self.num_freqs = num_freqs
            # Create frequency bands
            freqs = torch.linspace(0.0, 1.0, num_freqs)
            self.register_buffer('freqs', freqs)
            # Project Fourier features to d_model
            fourier_dim = 2 * 2 * num_freqs  # (x,y) * (sin,cos) * num_freqs
            self.fourier_proj = nn.Linear(fourier_dim, d_model)
        else:
            # Simple MLP: (x, y) → d_model
            self.mlp = nn.Sequential(
                nn.Linear(2, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Encode 2D coordinates to d_model-dimensional embeddings.
        
        Args:
            coords: [B, N, 2] - normalized coordinates
              - x: [0, 1] (left=0, right=1) - actual spatial position
              - y: [0, 1] for single-page, [0, 2] for two-page (second page: y += 1.0)
            
        Returns:
            pos_emb: [B, N, d_model] - positional embeddings
            
        Note: For RTL, x-coordinate is transformed for positional encoding so that
        large x (right side) is treated as "earlier" in reading order. This only
        affects the positional encoding, not the actual coordinates.
        """
        # For RTL reading order, transform x-coordinate for positional encoding
        # Large x (right side) should be treated as "earlier" in reading order
        # We transform x to (1.0 - x) ONLY for encoding purposes
        coords_encoded = coords.clone()
        if self.rtl:
            coords_encoded[..., 0] = 1.0 - coords_encoded[..., 0]
        
        # For two-page manuscripts, y can be in [0, 2] (second page: y in [1, 2])
        # The encoding handles this naturally - y values > 1.0 represent the second page
        
        if self.use_fourier:
            # Fourier features (using transformed coordinates for RTL)
            x, y = coords_encoded[..., 0:1], coords_encoded[..., 1:2]  # [B, N, 1]
            
            # Generate sin/cos features for each frequency
            x_sin = torch.sin(2 * math.pi * self.freqs * x)  # [B, N, num_freqs]
            x_cos = torch.cos(2 * math.pi * self.freqs * x)  # [B, N, num_freqs]
            y_sin = torch.sin(2 * math.pi * self.freqs * y)  # [B, N, num_freqs]
            y_cos = torch.cos(2 * math.pi * self.freqs * y)  # [B, N, num_freqs]
            
            # Concatenate: [B, N, 4*num_freqs]
            fourier_feat = torch.cat([x_sin, x_cos, y_sin, y_cos], dim=-1)
            
            # Project to d_model
            pos_emb = self.fourier_proj(fourier_feat)  # [B, N, d_model]
        else:
            # MLP-based encoding (using transformed coordinates for RTL)
            pos_emb = self.mlp(coords_encoded)  # [B, N, d_model]
        
        return pos_emb


class TileTokenEnrichment(nn.Module):
    """
    Tile Token Enrichment Module.
    
    Enriches tile visual features with:
    1. Projection: v_tok = proj(v)
    2. Positional embedding: + pos_tile = f([x_tile, y_tile])
    
    Note: Coordinates represent actual spatial positions (left=0, right=1).
    RTL reading order is handled by patch ordering, not coordinate transformation.
    
    Input: [B, N, d_model] visual features + [B, N, 2] tile coordinates
    Output: [B, N, d_model] enriched tokens
    """
    def __init__(
        self,
        d_model: int = D_MODEL,
        enable_pos_encoding: bool = TILE_BRANCH_ENABLE_POS_ENCODING,
        use_fourier_pos: bool = TILE_BRANCH_USE_FOURIER_POS,
        num_freqs: int = TILE_BRANCH_NUM_FREQS,
        rtl: bool = XML_PATCH_READING_DIRECTION_RTL,
    ):
        super().__init__()
        self.d_model = d_model
        self.enable_pos_encoding = enable_pos_encoding
        
        # Projection layer (identity if already d_model, but kept for flexibility)
        self.proj = nn.Linear(d_model, d_model)
        
        # Kept for checkpoint compatibility; modality-type embeddings
        # are applied in the fusion module.
        self.type_embed = nn.Embedding(1, d_model)
        
        # Positional encoding for tile coordinates
        # RTL-aware: transforms x for encoding so large x (right) is treated as "earlier"
        self.pos_encoder = PositionalEncoding2D(
            d_model=d_model,
            use_fourier=use_fourier_pos,
            num_freqs=num_freqs,
            rtl=rtl,
        )
    
    def forward(
        self,
        visual_features: torch.Tensor,
        tile_coords: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Enrich tile tokens with positional embeddings.
        
        Args:
            visual_features: [B, N, d_model] - visual features from TileVisualEncoder
            tile_coords: [B, N, 2] - normalized tile coordinates (x_tile, y_tile) in [0, 1]
            valid_mask: [B, N] (bool) - True for valid tiles, False for padding.
                If None, all tokens are treated as valid.
            
        Returns:
            enriched_tokens: [B, N, d_model] - enriched tile tokens
        """
        if tile_coords is None and self.enable_pos_encoding:
            raise ValueError("TileTokenEnrichment requires tile_coords when tile positional encoding is enabled.")

        if valid_mask is None:
            valid_mask = torch.ones(
                visual_features.shape[:2],
                dtype=torch.bool,
                device=visual_features.device,
            )
        else:
            _validate_tile_mask_shape(valid_mask, visual_features.shape[:2], "TileTokenEnrichment")
            valid_mask = valid_mask.to(device=visual_features.device, dtype=torch.bool)

        if tile_coords is not None:
            _validate_tile_coords_shape(tile_coords, visual_features.shape[:2], "TileTokenEnrichment")
            tile_coords = tile_coords.to(device=visual_features.device, dtype=visual_features.dtype)

        # 1. Project visual features
        v_tok = self.proj(visual_features)  # [B, N, d_model]

        # 2. Add positional embedding (optional — disabled when position is irrelevant)
        if self.enable_pos_encoding:
            pos_tile = self.pos_encoder(tile_coords)  # [B, N, d_model]
            v_tok = v_tok + pos_tile  # [B, N, d_model]
        
        # Zero out padded positions
        v_tok = v_tok * valid_mask.unsqueeze(-1).to(v_tok.dtype)
        
        return v_tok  # [B, N, d_model]


def apply_page_segment_to_coords(
    coords: torch.Tensor,
    page_segments: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Apply page_segment shift to y-coordinates for two-page manuscripts (tiles only).
    
    For two-page manuscripts, we create a continuous coordinate space by stacking pages vertically:
    - First page (page_segment=0): y stays in [0, 1] (normalized within first page)
    - Second page (page_segment=1): y is shifted by +1.0, so y is in [1, 2] (normalized within second page, then shifted)
    
    This allows the positional encoding to naturally distinguish between the two pages:
    - y ∈ [0, 1) → first page
    - y ∈ [1, 2] → second page
    
    Example:
        Single page: tile at top-left has coords (0.0, 0.0), bottom-right has (1.0, 1.0)
        Two pages: 
          - First page top-left: (0.0, 0.0), bottom-right: (1.0, 1.0)
          - Second page top-left: (0.0, 1.0), bottom-right: (1.0, 2.0)
    
    Args:
        coords: [B, N, 2] - tile coordinates (x, y) where both are normalized to [0, 1]
        page_segments: [B, N] (long) - page segment IDs (0 or 1), None if single-page
        
    Returns:
        coords: [B, N, 2] - coordinates with y shifted for second page
    """
    # If no page_segments provided, this is a single-page manuscript - no shift needed
    if page_segments is None:
        return coords
    
    coords = coords.clone()
    # Shift y-coordinate by +1.0 for second page (page_segment=1)
    # page_segments is [B, N], coords is [B, N, 2]
    second_page_mask = (page_segments == 1).to(coords.dtype)  # [B, N]
    coords[..., 1] = coords[..., 1] + second_page_mask  # [B, N]
    
    return coords


class TileSetTransformer(nn.Module):
    """
    Transformer module to process the set of tile tokens after enrichment.
    Learns spatial relationships and interactions between tiles.
    """
    def __init__(
        self,
        d_model: int = D_MODEL,
        num_layers: int = TILE_BRANCH_TRANSFORMER_LAYERS,
        num_heads: int = TILE_BRANCH_TRANSFORMER_HEADS,
        dim_feedforward: int = TILE_BRANCH_TRANSFORMER_DIM_FEEDFORWARD,
        dropout: float = TILE_BRANCH_TRANSFORMER_DROPOUT,
    ):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Self-attention over tile tokens.
        
        Args:
            x: [B, N, d_model] - enriched tile tokens
            valid_mask: [B, N] (bool) - True for valid tiles.
                If None, all tokens are treated as valid.
            
        Returns:
            x: [B, N, d_model] - transformed tokens
        """
        if valid_mask is None:
            valid_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        else:
            _validate_tile_mask_shape(valid_mask, x.shape[:2], "TileSetTransformer")
            valid_mask = valid_mask.to(device=x.device, dtype=torch.bool)

        # Create attention mask for Transformer (False/0 is attend, True/1 is mask out)
        # Note: input mask is [B, N] bool where True=keep.
        # TransformerEncoder expects [B, N] bool where True=discard.
        src_key_padding_mask = ~valid_mask
        
        # Self-attention over tiles
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)
        
        # Zero out padded positions
        x = x * valid_mask.unsqueeze(-1).to(x.dtype)
        
        return x


class TileSetSummarizer(nn.Module):
    """Summarize contextualized tiles with learned cross-attention queries.

    This intentionally mirrors ``GlyphSetSummarizer``: fusion receives a
    fixed-size set of summary tokens for each nonempty modality, while diagnostic
    forwards can return the last-layer query-to-raw-tile attention matrix.
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        num_queries: int = TILE_NUM_SUMMARY_TOKENS,
        num_heads: int = TILE_SUMMARIZER_ATTENTION_HEADS,
        num_cross_attn_layers: int = TILE_SUMMARIZER_CROSS_ATTN_LAYERS,
        dropout: float = DROPOUT,
        min_active_queries: int = TILE_MIN_SUMMARY_TOKENS,
    ):
        super().__init__()
        if int(num_queries) <= 0:
            raise ValueError("TileSetSummarizer requires num_queries > 0")
        if int(num_cross_attn_layers) <= 0:
            raise ValueError("TileSetSummarizer requires num_cross_attn_layers > 0")
        self.d_model = int(d_model)
        self.num_queries = int(num_queries)
        self.num_cross_attn_layers = int(num_cross_attn_layers)
        # Retained for checkpoint/API compatibility. Query validity is no
        # longer tied to source-token count.
        self.min_active_queries = max(
            1, min(int(min_active_queries), self.num_queries)
        )

        self.query_tokens = nn.Parameter(
            torch.empty(1, self.num_queries, self.d_model)
        )
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.d_model,
                num_heads=int(num_heads),
                dropout=float(dropout),
                batch_first=True,
            )
            for _ in range(self.num_cross_attn_layers)
        ])
        self.query_norms = nn.ModuleList([
            nn.LayerNorm(self.d_model) for _ in range(self.num_cross_attn_layers)
        ])
        self.tile_norms = nn.ModuleList([
            nn.LayerNorm(self.d_model) for _ in range(self.num_cross_attn_layers)
        ])
        self.out_norm = nn.LayerNorm(self.d_model)

    def _build_summary_valid_mask(self, valid_mask: torch.Tensor) -> torch.Tensor:
        """Keep every learned query when the tile modality has evidence.

        Queries are global set aggregators, not one-to-one tile slots. Source
        padding is already removed by cross-attention's key-padding mask, so
        the query outputs must not be truncated according to source count.
        """
        has_evidence = valid_mask.any(dim=1, keepdim=True)
        return has_evidence.expand(-1, self.num_queries).clone()

    def forward(
        self,
        tile_tokens: torch.Tensor,
        valid_mask: torch.Tensor,
        return_attention: bool = False,
    ):
        B, N = tile_tokens.shape[:2]
        _validate_tile_mask_shape(valid_mask, (B, N), "TileSetSummarizer")
        valid_mask = valid_mask.to(device=tile_tokens.device, dtype=torch.bool)
        summary_valid_mask = self._build_summary_valid_mask(valid_mask)

        summary = tile_tokens.new_zeros(B, self.num_queries, self.d_model)
        attention_out = (
            tile_tokens.new_zeros(B, self.num_queries, N)
            if return_attention else None
        )
        if N == 0:
            if return_attention:
                return summary, summary_valid_mask, attention_out
            return summary, summary_valid_mask

        any_valid = valid_mask.any(dim=1)
        if any_valid.any():
            batch_indices = torch.nonzero(any_valid, as_tuple=False).squeeze(1)
            tiles_valid_batch = tile_tokens.index_select(0, batch_indices)
            mask_valid_batch = valid_mask.index_select(0, batch_indices)
            queries = self.query_tokens.expand(B, -1, -1).index_select(
                0, batch_indices
            )
            attention = None
            for layer_idx, (cross_attn, query_norm, tile_norm) in enumerate(
                zip(self.cross_attn_layers, self.query_norms, self.tile_norms)
            ):
                is_last = layer_idx == self.num_cross_attn_layers - 1
                attention_output, layer_attention = cross_attn(
                    query=query_norm(queries),
                    key=tile_norm(tiles_valid_batch),
                    value=tile_norm(tiles_valid_batch),
                    key_padding_mask=~mask_valid_batch,
                    need_weights=bool(return_attention and is_last),
                    average_attn_weights=False,
                )
                queries = queries + attention_output
                if is_last:
                    attention = layer_attention

            if attention is not None and attention.dim() == 4:
                if attention.shape[0] == queries.shape[0]:
                    attention = attention.mean(dim=1)
                else:
                    attention = attention.mean(dim=0)
            summary.index_copy_(0, batch_indices, self.out_norm(queries))
            if return_attention and attention is not None:
                attention_out.index_copy_(0, batch_indices, attention)

        if return_attention:
            return summary, summary_valid_mask, attention_out
        return summary, summary_valid_mask


class TileBranch(nn.Module):
    """
    Complete Tile Branch: Visual Encoder + Token Enrichment.
    
    This is the full tile processing pipeline according to the new architecture:
    1. High-res page → tile crops (currently N<=10, 560×560) + tile coords
    2. Tile Visual Encoder: configured backbone (currently ConvNeXt) → [B, N, 768]
    3. Tile Token Enrichment: proj + pos_emb → [B, N, 768]
    4. Tile Set Transformer → contextualized [B, N, 768]
    5. Return contextualized tile tokens directly when summarization is disabled
    
    Supports two-page manuscripts: y-coordinates can be in [0, 2] (second page: y += 1.0)
    """
    def __init__(
        self,
        tile_size: int = TILE_SIZE,
        encoder_model_name: str = ENCODER_MODEL_NAME,
        d_model: int = D_MODEL,
        pretrained: bool = True,
        use_fourier_pos: bool = TILE_BRANCH_USE_FOURIER_POS,
        num_freqs: int = TILE_BRANCH_NUM_FREQS,
        rtl: bool = XML_PATCH_READING_DIRECTION_RTL,
        num_summary_tokens: int = TILE_NUM_SUMMARY_TOKENS,
        min_summary_tokens: int = TILE_MIN_SUMMARY_TOKENS,
        enable_summarizer: bool = USE_BRANCH_ADAPTERS_AND_SUMMARIZERS,
    ):
        super().__init__()
        
        self.visual_encoder = TileVisualEncoder(
            tile_size=tile_size,
            encoder_model_name=encoder_model_name,
            d_model=d_model,
            pretrained=pretrained,
        )
        
        self.token_enrichment = TileTokenEnrichment(
            d_model=d_model,
            use_fourier_pos=use_fourier_pos,
            num_freqs=num_freqs,
            rtl=rtl,
        )
        
        # 3. Tile Set Transformer (optional)
        if TILE_BRANCH_USE_TRANSFORMER:
            self.set_transformer = TileSetTransformer(
                d_model=d_model,
                num_layers=TILE_BRANCH_TRANSFORMER_LAYERS,
                num_heads=TILE_BRANCH_TRANSFORMER_HEADS,
                dim_feedforward=TILE_BRANCH_TRANSFORMER_DIM_FEEDFORWARD,
                dropout=TILE_BRANCH_TRANSFORMER_DROPOUT,
            )
        else:
            self.set_transformer = None

        self.set_summarizer = (
            TileSetSummarizer(
                d_model=d_model,
                num_queries=num_summary_tokens,
                min_active_queries=min_summary_tokens,
            )
            if bool(enable_summarizer) and int(num_summary_tokens) > 0
            else None
        )
    
    def forward(
        self,
        tiles: torch.Tensor,
        tile_coords: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        page_segments: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        """
        Forward pass through the tile branch.
        
        Args:
            tiles: [B, N, 3, H, W] - tile crops
            tile_coords: [B, N, 2] - normalized tile coordinates (x_tile, y_tile)
              - x: [0, 1]
              - y: [0, 1] for single-page, or can be in [0, 2] if page_segments provided
            valid_mask: [B, N] (bool) - True for valid tiles, False for padding.
              If None, all tiles are treated as valid.
            page_segments: [B, N] (long, optional) - page segment IDs (0 or 1) for two-page manuscripts
              If provided, y-coordinates will be shifted: page_segment=1 → y += 1.0
            return_attention: If True, return last-layer query-to-tile attention.
            
        Returns:
            Tile tokens and their matching valid mask. With summarization
            disabled, their shapes are `[B, N, d_model]` and `[B, N]`.
            `return_attention=True` adds query-to-tile attention only when the
            optional summarizer is enabled; otherwise it adds `None`.
        """
        if tile_coords is None and self.token_enrichment.enable_pos_encoding:
            raise ValueError("TileBranch requires tile_coords when tile positional encoding is enabled.")

        if valid_mask is None:
            valid_mask = torch.ones(
                tiles.shape[:2],
                dtype=torch.bool,
                device=tiles.device,
            )
        else:
            _validate_tile_mask_shape(valid_mask, tiles.shape[:2], "TileBranch")
            valid_mask = valid_mask.to(device=tiles.device, dtype=torch.bool)

        if tile_coords is not None:
            _validate_tile_coords_shape(tile_coords, tiles.shape[:2], "TileBranch")
            tile_coords = tile_coords.to(device=tiles.device, dtype=tiles.dtype)

        # Apply page_segment shift if provided
        if page_segments is not None:
            if tile_coords is None:
                raise ValueError("TileBranch requires tile_coords when page_segments are provided.")
            _validate_tile_mask_shape(page_segments, tiles.shape[:2], "TileBranch", name="page_segments")
            page_segments = page_segments.to(device=tiles.device, dtype=torch.long)
            tile_coords = apply_page_segment_to_coords(tile_coords, page_segments)
        
        # 1. Visual encoding
        if return_attention:
            visual_features, _ = self.visual_encoder(tiles, valid_mask, return_attention=True)  # [B, N, d_model], None
        else:
            visual_features = self.visual_encoder(tiles, valid_mask)  # [B, N, d_model]
        
        # 2. Token enrichment
        enriched_tokens = self.token_enrichment(
            visual_features,
            tile_coords,
            valid_mask,
        )  # [B, N, d_model]
        
        # 3. Tile set transformation (contextualize tiles)
        if self.set_transformer is not None:
            enriched_tokens = self.set_transformer(enriched_tokens, valid_mask)
        
        # 4. Optional learned-query summarization. The active adapter-free
        # symmetric path disables it and mask-pools these tile tokens directly.
        if self.set_summarizer is not None:
            return self.set_summarizer(
                enriched_tokens,
                valid_mask,
                return_attention=return_attention,
            )

        if return_attention:
            return enriched_tokens, valid_mask, None
        return enriched_tokens, valid_mask
