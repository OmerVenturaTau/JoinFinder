"""
Tests for VLAD-style fusion.
"""

import torch
import pytest

from models.vlad_fusion import VLADFusion
from system import D_MODEL, VLAD_NUM_CLUSTERS


def test_vlad_fusion_creation():
    fusion = VLADFusion(d_model=D_MODEL)

    assert fusion.d_model == D_MODEL
    assert fusion.num_clusters == VLAD_NUM_CLUSTERS
    assert fusion.clusters.shape == (VLAD_NUM_CLUSTERS, D_MODEL)


def test_vlad_fusion_forward_tiles_only():
    B, N = 2, 8
    fusion = VLADFusion(d_model=D_MODEL)

    tile_tokens = torch.rand(B, N, D_MODEL)
    tile_valid_mask = torch.ones(B, N, dtype=torch.bool)

    latents, attention = fusion(
        tile_tokens=tile_tokens,
        tile_valid_mask=tile_valid_mask,
        return_attention=True,
    )

    assert latents.shape == (B, 1, D_MODEL)
    assert attention.shape == (B, 1, N)
    assert torch.isfinite(latents).all()


def test_vlad_fusion_attention_uses_token_contribution_not_uniform_mask():
    fusion = VLADFusion(d_model=D_MODEL)
    with torch.no_grad():
        fusion.assignment.weight.zero_()
        fusion.assignment.bias.zero_()
        fusion.clusters.zero_()
        fusion.clusters[0, 0] = 5.0

    tile_tokens = torch.zeros(1, 2, D_MODEL)
    tile_tokens[0, 0, 0] = 1.0
    tile_tokens[0, 1, 1] = 1.0
    tile_valid_mask = torch.ones(1, 2, dtype=torch.bool)

    _, attention = fusion(
        tile_tokens=tile_tokens,
        tile_valid_mask=tile_valid_mask,
        return_attention=True,
    )

    assert attention.shape == (1, 1, 2)
    assert torch.isclose(attention.sum(), torch.tensor(1.0), atol=1e-6)
    assert not torch.allclose(attention[0, 0], torch.full((2,), 0.5))


def test_vlad_fusion_forward_multimodal_with_padding():
    B, N, M = 2, 8, 4
    fusion = VLADFusion(d_model=D_MODEL)

    tile_tokens = torch.rand(B, N, D_MODEL)
    tile_valid_mask = torch.ones(B, N, dtype=torch.bool)
    tile_valid_mask[0, -2:] = False

    glyph_tokens = torch.rand(B, M, D_MODEL)
    glyph_valid_mask = torch.ones(B, M, dtype=torch.bool)
    glyph_valid_mask[1, -1:] = False

    latents, attention = fusion(
        tile_tokens=tile_tokens,
        tile_valid_mask=tile_valid_mask,
        glyph_tokens=glyph_tokens,
        glyph_valid_mask=glyph_valid_mask,
        return_attention=True,
    )

    assert latents.shape == (B, 1, D_MODEL)
    assert attention.shape == (B, 1, N + M)
    assert torch.isfinite(latents).all()
    assert torch.all(attention[0, 0, N - 2:N] == 0)
    assert torch.all(attention[1, 0, -1:] == 0)


def test_vlad_fusion_all_masked_is_finite():
    B, N = 2, 3
    fusion = VLADFusion(d_model=D_MODEL)

    tile_tokens = torch.rand(B, N, D_MODEL)
    tile_valid_mask = torch.zeros(B, N, dtype=torch.bool)

    latents, attention = fusion(
        tile_tokens=tile_tokens,
        tile_valid_mask=tile_valid_mask,
        return_attention=True,
    )

    assert latents.shape == (B, 1, D_MODEL)
    assert attention.shape == (B, 1, N)
    assert torch.isfinite(latents).all()
    assert torch.all(attention == 0)


def test_vlad_fusion_rejects_mask_shape_mismatch():
    fusion = VLADFusion(d_model=D_MODEL)
    tile_tokens = torch.rand(2, 4, D_MODEL)
    wrong_mask = torch.ones(2, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match="tile_valid_mask must have shape"):
        fusion(tile_tokens=tile_tokens, tile_valid_mask=wrong_mask)


def test_vlad_fusion_rejects_dtype_mismatch():
    fusion = VLADFusion(d_model=D_MODEL)
    tile_tokens = torch.rand(2, 4, D_MODEL, dtype=torch.float32)
    glyph_tokens = torch.rand(2, 3, D_MODEL, dtype=torch.float64)

    with pytest.raises(ValueError, match="glyph_tokens dtype must match"):
        fusion(tile_tokens=tile_tokens, glyph_tokens=glyph_tokens)
