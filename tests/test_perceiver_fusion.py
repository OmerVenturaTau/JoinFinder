"""
Tests for Perceiver Fusion module.
"""

import torch
import pytest
from models.perceiver_fusion import PerceiverFusion, PerceiverHead
from system import (
    D_MODEL,
    PERCEIVER_NUM_LATENTS,
    PERCEIVER_NUM_CROSS_ATTN_LAYERS,
    PERCEIVER_POOLING,
    LATENT_DIM,
    MAX_SELECTED_MANUSCRIPTS,
)


def test_perceiver_fusion_creation():
    """Test PerceiverFusion creation."""
    fusion = PerceiverFusion(d_model=D_MODEL)
    
    assert fusion.d_model == D_MODEL
    assert fusion.num_latents == PERCEIVER_NUM_LATENTS
    assert len(fusion.cross_attn_layers) == PERCEIVER_NUM_CROSS_ATTN_LAYERS
    assert fusion.latent_queries.shape == (1, PERCEIVER_NUM_LATENTS, D_MODEL)


def test_perceiver_fusion_forward_tiles_only():
    """Test PerceiverFusion forward with tile tokens only."""
    B, N = 2, 8
    fusion = PerceiverFusion(d_model=D_MODEL)
    
    tile_tokens = torch.rand(B, N, D_MODEL)
    tile_valid_mask = torch.ones(B, N, dtype=torch.bool)
    
    latents, attention_weights = fusion(
        tile_tokens=tile_tokens,
        tile_valid_mask=tile_valid_mask,
        return_attention=True,
    )
    
    assert latents.shape == (B, PERCEIVER_NUM_LATENTS, D_MODEL)
    assert attention_weights is not None
    assert attention_weights.shape == (B, PERCEIVER_NUM_LATENTS, N)


def test_perceiver_fusion_forward_glyphs_only():
    """Test PerceiverFusion forward with glyph tokens only."""
    B, M = 2, 32
    fusion = PerceiverFusion(d_model=D_MODEL)
    
    glyph_tokens = torch.rand(B, M, D_MODEL)
    glyph_valid_mask = torch.ones(B, M, dtype=torch.bool)
    
    latents, attention_weights = fusion(
        glyph_tokens=glyph_tokens,
        glyph_valid_mask=glyph_valid_mask,
        return_attention=True,
    )
    
    assert latents.shape == (B, PERCEIVER_NUM_LATENTS, D_MODEL)
    assert attention_weights is not None
    assert attention_weights.shape == (B, PERCEIVER_NUM_LATENTS, M)


def test_perceiver_fusion_forward_multimodal():
    """Test PerceiverFusion forward with multiple modalities."""
    B, N, M = 2, 8, 32
    fusion = PerceiverFusion(d_model=D_MODEL)
    
    tile_tokens = torch.rand(B, N, D_MODEL)
    tile_valid_mask = torch.ones(B, N, dtype=torch.bool)
    
    glyph_tokens = torch.rand(B, M, D_MODEL)
    glyph_valid_mask = torch.ones(B, M, dtype=torch.bool)
    
    latents, attention_weights = fusion(
        tile_tokens=tile_tokens,
        tile_valid_mask=tile_valid_mask,
        glyph_tokens=glyph_tokens,
        glyph_valid_mask=glyph_valid_mask,
        return_attention=True,
    )
    
    assert latents.shape == (B, PERCEIVER_NUM_LATENTS, D_MODEL)
    assert attention_weights is not None
    assert attention_weights.shape == (B, PERCEIVER_NUM_LATENTS, N + M)


def test_perceiver_fusion_with_padding():
    """Test PerceiverFusion handles padding correctly."""
    B, N = 2, 8
    fusion = PerceiverFusion(d_model=D_MODEL)
    
    tile_tokens = torch.rand(B, N, D_MODEL)
    tile_valid_mask = torch.ones(B, N, dtype=torch.bool)
    tile_valid_mask[0, -2:] = False  # Mark last 2 as padding
    
    latents, attention_weights = fusion(
        tile_tokens=tile_tokens,
        tile_valid_mask=tile_valid_mask,
        return_attention=True,
    )
    
    assert latents.shape == (B, PERCEIVER_NUM_LATENTS, D_MODEL)
    assert attention_weights.shape == (B, PERCEIVER_NUM_LATENTS, N)
    assert torch.all(attention_weights[0, :, -2:] == 0)


def test_perceiver_fusion_all_masked_attention_keeps_original_token_length():
    B, N = 2, 4
    fusion = PerceiverFusion(d_model=D_MODEL)

    tile_tokens = torch.rand(B, N, D_MODEL)
    tile_valid_mask = torch.zeros(B, N, dtype=torch.bool)

    latents, attention_weights = fusion(
        tile_tokens=tile_tokens,
        tile_valid_mask=tile_valid_mask,
        return_attention=True,
    )

    assert latents.shape == (B, PERCEIVER_NUM_LATENTS, D_MODEL)
    assert attention_weights.shape == (B, PERCEIVER_NUM_LATENTS, N)
    assert torch.isfinite(latents).all()
    assert torch.isfinite(attention_weights).all()


def test_perceiver_fusion_rejects_mask_shape_mismatch():
    fusion = PerceiverFusion(d_model=D_MODEL)
    tile_tokens = torch.rand(2, 4, D_MODEL)
    wrong_mask = torch.ones(2, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match="tile_valid_mask must have shape"):
        fusion(tile_tokens=tile_tokens, tile_valid_mask=wrong_mask)


def test_perceiver_fusion_attention_is_opt_in():
    B, N = 2, 4
    fusion = PerceiverFusion(d_model=D_MODEL)
    tile_tokens = torch.rand(B, N, D_MODEL)

    latents, attention_weights = fusion(tile_tokens=tile_tokens)

    assert latents.shape == (B, PERCEIVER_NUM_LATENTS, D_MODEL)
    assert attention_weights is None


def test_perceiver_fusion_rejects_dtype_mismatch():
    fusion = PerceiverFusion(d_model=D_MODEL)
    tile_tokens = torch.rand(2, 4, D_MODEL, dtype=torch.float32)
    glyph_tokens = torch.rand(2, 3, D_MODEL, dtype=torch.float64)

    with pytest.raises(ValueError, match="glyph_tokens dtype must match"):
        fusion(tile_tokens=tile_tokens, glyph_tokens=glyph_tokens)


def test_perceiver_head_creation():
    """Test PerceiverHead creation."""
    head = PerceiverHead(
        d_model=D_MODEL,
        num_classes=MAX_SELECTED_MANUSCRIPTS,
    )
    
    assert head.pooling == PERCEIVER_POOLING
    assert head.classifier[-1].out_features == MAX_SELECTED_MANUSCRIPTS


def test_perceiver_head_forward_mean_pooling():
    """Test PerceiverHead forward with mean pooling."""
    B = 2
    head = PerceiverHead(
        d_model=D_MODEL,
        num_classes=MAX_SELECTED_MANUSCRIPTS,
        pooling="mean",
    )
    
    latent_repr = torch.rand(B, PERCEIVER_NUM_LATENTS, D_MODEL)
    
    logits, latent = head(latent_repr)
    
    assert logits.shape == (B, MAX_SELECTED_MANUSCRIPTS)
    assert latent.shape == (B, LATENT_DIM)


def test_perceiver_head_forward_cls_pooling():
    """Test PerceiverHead forward with CLS pooling."""
    B = 2
    head = PerceiverHead(
        d_model=D_MODEL,
        num_classes=MAX_SELECTED_MANUSCRIPTS,
        pooling="cls",
    )
    
    latent_repr = torch.rand(B, PERCEIVER_NUM_LATENTS, D_MODEL)
    
    logits, latent = head(latent_repr)
    
    assert logits.shape == (B, MAX_SELECTED_MANUSCRIPTS)
    assert latent.shape == (B, LATENT_DIM)


def test_perceiver_fusion_and_head_integration():
    """Test PerceiverFusion and PerceiverHead integration."""
    B, N = 2, 8
    
    fusion = PerceiverFusion(d_model=D_MODEL)
    head = PerceiverHead(
        d_model=D_MODEL,
        num_classes=MAX_SELECTED_MANUSCRIPTS,
    )
    
    tile_tokens = torch.rand(B, N, D_MODEL)
    tile_valid_mask = torch.ones(B, N, dtype=torch.bool)
    
    latents, _ = fusion(
        tile_tokens=tile_tokens,
        tile_valid_mask=tile_valid_mask,
    )
    
    logits, latent = head(latents)
    
    assert logits.shape == (B, MAX_SELECTED_MANUSCRIPTS)
    assert latent.shape == (B, LATENT_DIM)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
