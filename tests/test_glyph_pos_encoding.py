import torch
import pytest
import math
from models.glyph_branch import PositionalEncoding4D, GlyphTokenEnrichment, GlyphBranch
from models.tile_branch import PositionalEncoding2D
from models.word_branch import PositionalEncoding4D as WordPositionalEncoding4D
from system import D_MODEL, CHAR_PATCH_SIZE, GLYPH_NUM_SUMMARY_TOKENS

def test_rtl_encoding_transformation():
    """Verify that RTL flag correctly flips the x-coordinate for encoding."""
    d_model = 128
    # Create two encoders: one RTL, one LTR
    pos_enc_rtl = PositionalEncoding4D(d_model=d_model, rtl=True, use_fourier=False)
    pos_enc_ltr = PositionalEncoding4D(d_model=d_model, rtl=False, use_fourier=False)
    
    # Sync weights to ensure deterministic comparison (MLP is randomly initialized)
    # Using index-based access since it's an nn.Sequential
    pos_enc_ltr.mlp[0].weight.data.copy_(pos_enc_rtl.mlp[0].weight.data)
    pos_enc_ltr.mlp[0].bias.data.copy_(pos_enc_rtl.mlp[0].bias.data)
    pos_enc_ltr.mlp[2].weight.data.copy_(pos_enc_rtl.mlp[2].weight.data)
    pos_enc_ltr.mlp[2].bias.data.copy_(pos_enc_rtl.mlp[2].bias.data)
    
    # Coordinates: x=0.2, y=0.5, w=0.1, h=0.1
    coords = torch.tensor([[[0.2, 0.5, 0.1, 0.1]]])
    # Coordinates: x=0.8, y=0.5, w=0.1, h=0.1
    coords_flipped = torch.tensor([[[0.8, 0.5, 0.1, 0.1]]])
    
    with torch.no_grad():
        emb_rtl = pos_enc_rtl(coords)
        emb_ltr = pos_enc_ltr(coords_flipped)
    
    # If RTL flips x (1.0 - 0.2 = 0.8), then emb_rtl(0.2) should equal emb_ltr(0.8)
    assert torch.allclose(emb_rtl, emb_ltr, atol=1e-5), "RTL x-flip logic failed"


def test_tile_rtl_encoding_transformation():
    """Verify tile positional encoding flips x for RTL."""
    d_model = 128
    pos_enc_rtl = PositionalEncoding2D(d_model=d_model, rtl=True, use_fourier=False)
    pos_enc_ltr = PositionalEncoding2D(d_model=d_model, rtl=False, use_fourier=False)

    pos_enc_ltr.mlp[0].weight.data.copy_(pos_enc_rtl.mlp[0].weight.data)
    pos_enc_ltr.mlp[0].bias.data.copy_(pos_enc_rtl.mlp[0].bias.data)
    pos_enc_ltr.mlp[2].weight.data.copy_(pos_enc_rtl.mlp[2].weight.data)
    pos_enc_ltr.mlp[2].bias.data.copy_(pos_enc_rtl.mlp[2].bias.data)

    coords = torch.tensor([[[0.2, 0.5]]])
    coords_flipped = torch.tensor([[[0.8, 0.5]]])

    with torch.no_grad():
        emb_rtl = pos_enc_rtl(coords)
        emb_ltr = pos_enc_ltr(coords_flipped)

    assert torch.allclose(emb_rtl, emb_ltr, atol=1e-5), "Tile RTL x-flip logic failed"


def test_word_rtl_encoding_transformation():
    """Verify word/line positional encoding flips x for RTL."""
    d_model = 128
    pos_enc_rtl = WordPositionalEncoding4D(d_model=d_model, rtl=True, use_fourier=False)
    pos_enc_ltr = WordPositionalEncoding4D(d_model=d_model, rtl=False, use_fourier=False)

    pos_enc_ltr.mlp[0].weight.data.copy_(pos_enc_rtl.mlp[0].weight.data)
    pos_enc_ltr.mlp[0].bias.data.copy_(pos_enc_rtl.mlp[0].bias.data)
    pos_enc_ltr.mlp[2].weight.data.copy_(pos_enc_rtl.mlp[2].weight.data)
    pos_enc_ltr.mlp[2].bias.data.copy_(pos_enc_rtl.mlp[2].bias.data)

    coords = torch.tensor([[[0.2, 0.5, 0.1, 0.1]]])
    coords_flipped = torch.tensor([[[0.8, 0.5, 0.1, 0.1]]])

    with torch.no_grad():
        emb_rtl = pos_enc_rtl(coords)
        emb_ltr = pos_enc_ltr(coords_flipped)

    assert torch.allclose(emb_rtl, emb_ltr, atol=1e-5), "Word RTL x-flip logic failed"

def test_fourier_encoding_range():
    """Test that Fourier features handle coordinates outside [0, 1] (e.g., page 2)."""
    d_model = 128
    pos_enc = PositionalEncoding4D(d_model=d_model, use_fourier=True, num_freqs=16)
    
    # Page 1 coords vs Page 2 coords
    coords_p1 = torch.tensor([[[0.5, 0.5, 0.1, 0.1]]])
    coords_p2 = torch.tensor([[[0.5, 1.5, 0.1, 0.1]]]) # y = 1.5 (second page)
    
    with torch.no_grad():
        emb_p1 = pos_enc(coords_p1)
        emb_p2 = pos_enc(coords_p2)
    
    assert not torch.allclose(emb_p1, emb_p2), "Positional embeddings should differ for different pages"
    assert not torch.isnan(emb_p2).any(), "Fourier encoding produced NaNs for out-of-range y"

def test_token_enrichment_activation():
    """Verify GlyphTokenEnrichment adds pos encoding only when enabled."""
    d_model = D_MODEL
    
    enrich_enabled = GlyphTokenEnrichment(d_model=d_model, enable_pos_encoding=True)
    enrich_disabled = GlyphTokenEnrichment(d_model=d_model, enable_pos_encoding=False)
    
    # Sync weights for fair comparison of other parts
    enrich_disabled.proj.weight.data.copy_(enrich_enabled.proj.weight.data)
    enrich_disabled.proj.bias.data.copy_(enrich_enabled.proj.bias.data)
    enrich_disabled.type_embed.weight.data.copy_(enrich_enabled.type_embed.weight.data)
    
    feats = torch.randn(1, 4, d_model)
    coords = torch.rand(1, 4, 4)
    
    with torch.no_grad():
        out_enabled = enrich_enabled(feats, coords)
        out_disabled = enrich_disabled(feats, coords)
        
    assert not torch.allclose(out_enabled, out_disabled), "Enabling pos encoding had no effect"

def test_masking_consistency():
    """Ensure positional embeddings are zeroed out for masked tokens."""
    d_model = D_MODEL
    branch = GlyphBranch(d_model=d_model)
    
    B, M = 1, 10
    patches = torch.randn(B, M, 3, CHAR_PATCH_SIZE, CHAR_PATCH_SIZE)
    coords = torch.rand(B, M, 4)
    valid_mask = torch.ones(B, M, dtype=torch.bool)
    valid_mask[0, 5:] = False # Mask half
    
    # We need to manually check internal enriched_tokens if possible, 
    # but we can check if the output is stable or if we can hook it.
    # Alternatively, test GlyphTokenEnrichment directly for masking.
    
    enrichment = GlyphTokenEnrichment(d_model=d_model, enable_pos_encoding=True)
    with torch.no_grad():
        enriched = enrichment(torch.randn(B, M, d_model), coords, valid_mask=valid_mask)
    
    assert torch.all(enriched[0, 5:] == 0), "Masked tokens still have positional embeddings"

if __name__ == "__main__":
    pytest.main([__file__])
