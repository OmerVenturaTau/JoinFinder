"""
Tests for Tile Branch module.
"""

import torch
import torch.nn as nn
import pytest
from models.tile_branch import (
    TileBranch,
    TileVisualEncoder,
    TileTokenEnrichment,
    TileSetSummarizer,
    PositionalEncoding2D,
    apply_page_segment_to_coords,
)
from system import (
    TILE_SIZE,
    D_MODEL,
    TILE_BRANCH_ENABLE_POS_ENCODING,
    TILE_NUM_SUMMARY_TOKENS,
    USE_BRANCH_ADAPTERS_AND_SUMMARIZERS,
)


class _TinyTileEncoder(nn.Module):
    def forward(self, x):
        return x.mean(dim=(2, 3))


def test_tile_positional_encoding_enabled_in_config():
    """RTL/page-segment tile positioning only works in normal runs when this is enabled."""
    assert TILE_BRANCH_ENABLE_POS_ENCODING is True


def test_positional_encoding_2d():
    """Test 2D positional encoding."""
    B, N = 2, 4
    d_model = D_MODEL
    
    pos_enc = PositionalEncoding2D(d_model=d_model, use_fourier=True, num_freqs=64)
    coords = torch.rand(B, N, 2)  # [B, N, 2] - (x, y)
    
    pos_emb = pos_enc(coords)
    assert pos_emb.shape == (B, N, d_model)
    
    # Test RTL flipping
    pos_enc_rtl = PositionalEncoding2D(d_model=d_model, rtl=True)
    coords_rtl = torch.tensor([[[0.0, 0.5], [1.0, 0.5]]])  # [1, 2, 2]
    pos_emb_rtl = pos_enc_rtl(coords_rtl)
    # x=0.0 should be encoded differently than x=1.0 in RTL mode
    assert not torch.allclose(pos_emb_rtl[0, 0], pos_emb_rtl[0, 1])


def test_apply_page_segment_to_coords():
    """Test page segment coordinate shifting."""
    B, N = 2, 4
    coords = torch.rand(B, N, 2)  # [B, N, 2] - (x, y)
    page_segments = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])  # [B, N]
    
    shifted = apply_page_segment_to_coords(coords, page_segments)
    
    # Check that y-coordinates are shifted for page_segment=1
    assert shifted.shape == coords.shape
    for b in range(B):
        for n in range(N):
            if page_segments[b, n] == 1:
                assert shifted[b, n, 1] == coords[b, n, 1] + 1.0


def test_tile_visual_encoder():
    """Test tile visual encoder."""
    B, N = 2, 8
    H, W = TILE_SIZE, TILE_SIZE
    
    encoder = TileVisualEncoder(tile_size=TILE_SIZE, d_model=D_MODEL)
    tiles = torch.rand(B, N, 3, H, W)
    valid_mask = torch.ones(B, N, dtype=torch.bool)
    valid_mask[0, -2:] = False  # Mark last 2 as invalid
    
    features = encoder(tiles, valid_mask)
    assert features.shape == (B, N, D_MODEL)
    
    # Check that invalid tiles are zeroed out
    assert torch.allclose(features[0, -2:], torch.zeros(2, D_MODEL))

    features_without_mask = encoder(tiles[:, :2], None)
    assert features_without_mask.shape == (B, 2, D_MODEL)


def test_tile_visual_encoder_accepts_non_contiguous_mask():
    """Sliced masks should not crash flattening inside the tile encoder."""
    encoder = TileVisualEncoder(tile_size=4, d_model=3)
    encoder.tile_encoder = _TinyTileEncoder()
    encoder.tile_feat_dim = 3
    encoder.proj = nn.Identity()

    tiles = torch.rand(2, 2, 3, 4, 4)
    valid_mask = torch.ones(2, 4, dtype=torch.bool)[:, :2]

    assert not valid_mask.is_contiguous()
    features = encoder(tiles, valid_mask)

    assert features.shape == (2, 2, 3)


def test_tile_visual_encoder_rejects_mask_shape_mismatch():
    encoder = TileVisualEncoder(tile_size=4, d_model=3)
    encoder.tile_encoder = _TinyTileEncoder()
    encoder.tile_feat_dim = 3
    encoder.proj = nn.Identity()

    tiles = torch.rand(2, 2, 3, 4, 4)
    wrong_mask = torch.ones(1, 4, dtype=torch.bool)

    with pytest.raises(ValueError, match="valid_mask must have shape"):
        encoder(tiles, wrong_mask)


def test_tile_token_enrichment():
    """Test tile token enrichment."""
    B, N = 2, 8
    d_model = D_MODEL
    
    enrichment = TileTokenEnrichment(d_model=d_model)
    visual_features = torch.rand(B, N, d_model)
    tile_coords = torch.rand(B, N, 2)  # [B, N, 2] - (x, y)
    valid_mask = torch.ones(B, N, dtype=torch.bool)
    
    enriched = enrichment(visual_features, tile_coords, valid_mask)
    assert enriched.shape == (B, N, d_model)
    
    # Enriched tokens should be different from input (due to type + pos embeddings)
    assert not torch.allclose(enriched, visual_features)

    enriched_without_mask = enrichment(visual_features, tile_coords, None)
    assert enriched_without_mask.shape == (B, N, d_model)


def test_tile_token_enrichment_requires_coords_when_positional_encoding_enabled():
    enrichment = TileTokenEnrichment(
        d_model=8,
        enable_pos_encoding=True,
        use_fourier_pos=False,
    )

    with pytest.raises(ValueError, match="requires tile_coords"):
        enrichment(torch.zeros(1, 2, 8), None, None)


def test_tile_token_enrichment_rejects_misaligned_mask_and_coords():
    enrichment = TileTokenEnrichment(
        d_model=8,
        enable_pos_encoding=True,
        use_fourier_pos=False,
    )
    features = torch.zeros(2, 3, 8)
    coords = torch.zeros(2, 3, 2)

    with pytest.raises(ValueError, match="valid_mask must have shape"):
        enrichment(features, coords, torch.ones(2, 2, dtype=torch.bool))

    with pytest.raises(ValueError, match="tile_coords must have shape"):
        enrichment(features, torch.zeros(2, 2, 2), torch.ones(2, 3, dtype=torch.bool))


def test_tile_set_summarizer_returns_masked_query_to_tile_attention():
    summarizer = TileSetSummarizer(
        d_model=8,
        num_queries=4,
        num_heads=2,
        num_cross_attn_layers=2,
        dropout=0.0,
        min_active_queries=1,
    ).eval()
    tokens = torch.randn(2, 3, 8)
    valid_mask = torch.tensor([[True, False, False], [False, False, False]])

    summary, summary_valid, attention = summarizer(
        tokens, valid_mask, return_attention=True
    )

    assert summary.shape == (2, 4, 8)
    assert summary_valid.tolist() == [[True] * 4, [False] * 4]
    assert attention.shape == (2, 4, 3)
    torch.testing.assert_close(attention[0, :, 1:], torch.zeros(4, 2))
    torch.testing.assert_close(attention[0].sum(dim=1), torch.ones(4))
    torch.testing.assert_close(attention[1], torch.zeros(4, 3))
    torch.testing.assert_close(summary[1], torch.zeros(4, 8))


def test_tile_branch():
    """Test complete tile branch."""
    B, N = 2, 8
    H, W = TILE_SIZE, TILE_SIZE
    
    branch = TileBranch(tile_size=TILE_SIZE, d_model=D_MODEL)
    tiles = torch.rand(B, N, 3, H, W)
    tile_coords = torch.rand(B, N, 2)  # [B, N, 2] - (x, y)
    valid_mask = torch.ones(B, N, dtype=torch.bool)
    page_segments = torch.tensor([[0, 0, 1, 1, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0]])
    
    tile_tokens, tile_valid = branch(
        tiles, tile_coords, valid_mask, page_segments
    )
    assert TILE_NUM_SUMMARY_TOKENS == 8
    if USE_BRANCH_ADAPTERS_AND_SUMMARIZERS:
        assert branch.set_summarizer is not None
        assert tile_tokens.shape == (B, TILE_NUM_SUMMARY_TOKENS, D_MODEL)
        assert tile_valid.all()
    else:
        assert branch.set_summarizer is None
        assert tile_tokens.shape == (B, N, D_MODEL)
        assert torch.equal(tile_valid, valid_mask)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
