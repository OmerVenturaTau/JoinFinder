"""
Tests for Glyph Branch module.
"""

import torch
import pytest
from models.glyph_branch import (
    GlyphBranch,
    GlyphHardQualityFilter,
    GlyphVisualEncoder,
    GlyphTokenEnrichment,
    GlyphSetSummarizer,
    PositionalEncoding4D,
    apply_page_segment_to_glyph_coords,
)
from system import (
    CHAR_PATCH_SIZE,
    D_MODEL,
    GLYPH_NUM_SUMMARY_TOKENS,
    USE_BRANCH_ADAPTERS_AND_SUMMARIZERS,
)


def test_glyph_hard_quality_filter():
    """Test glyph quality filter."""
    filter_module = GlyphHardQualityFilter(
        min_area=4,
        min_width=2,
        min_height=2,
        max_aspect_ratio=10.0,
        min_confidence=0.5,
    )
    
    # Create test glyphs
    glyph_patches = [
        torch.rand(3, CHAR_PATCH_SIZE, CHAR_PATCH_SIZE),
        torch.rand(3, CHAR_PATCH_SIZE, CHAR_PATCH_SIZE),
        torch.zeros(3, CHAR_PATCH_SIZE, CHAR_PATCH_SIZE),  # Blank patch
    ]
    
    glyph_metadata = [
        {'width': 10, 'height': 10, 'gc': 0.9, 'char': 'א'},
        {'width': 1, 'height': 1, 'gc': 0.9, 'char': 'ב'},  # Too small
        {'width': 10, 'height': 10, 'gc': 0.3, 'char': 'ג'},  # Low confidence
    ]
    
    filtered_patches, filtered_metadata = filter_module.filter_glyphs(glyph_patches, glyph_metadata)
    
    # Should keep only the first glyph (valid size and confidence)
    assert len(filtered_patches) == 1
    assert len(filtered_metadata) == 1


def test_glyph_hard_quality_filter_rejects_blank_high_confidence_patch():
    """Blank detection should not depend only on low OCR confidence."""
    filter_module = GlyphHardQualityFilter(
        min_area=4,
        min_width=2,
        min_height=2,
        max_aspect_ratio=10.0,
        min_confidence=0.5,
    )
    blank_patch = torch.zeros(3, CHAR_PATCH_SIZE, CHAR_PATCH_SIZE)
    metadata = [{'width': 10, 'height': 10, 'gc': 0.95, 'char': 'א'}]

    filtered_patches, filtered_metadata = filter_module.filter_glyphs([blank_patch], metadata)

    assert filtered_patches == []
    assert filtered_metadata == []


def test_sample_diverse_glyphs():
    """Test diverse glyph sampling."""
    glyph_patches = [torch.rand(3, CHAR_PATCH_SIZE, CHAR_PATCH_SIZE) for _ in range(20)]
    glyph_metadata = [
        {'width': 10, 'height': 10, 'gc': 0.9, 'char': 'א' if i % 3 == 0 else 'ב' if i % 3 == 1 else 'ג'}
        for i in range(20)
    ]
    
    sampled_patches, sampled_metadata = GlyphHardQualityFilter.sample_diverse_glyphs(
        glyph_patches, glyph_metadata, max_glyphs=10, min_confidence=0.8
    )
    
    assert len(sampled_patches) <= 10
    assert len(sampled_metadata) <= 10
    # Should prefer high confidence
    assert all(m['gc'] >= 0.8 for m in sampled_metadata)


def test_glyph_visual_encoder():
    """Test glyph visual encoder."""
    B, M = 2, 16
    H, W = CHAR_PATCH_SIZE, CHAR_PATCH_SIZE
    
    encoder = GlyphVisualEncoder(char_patch_size=CHAR_PATCH_SIZE, d_model=D_MODEL)
    glyph_patches = torch.rand(B, M, 3, H, W)
    valid_mask = torch.ones(B, M, dtype=torch.bool)
    valid_mask[0, -4:] = False  # Mark last 4 as invalid
    
    features = encoder(glyph_patches, valid_mask)
    assert features.shape == (B, M, D_MODEL)
    
    # Check that invalid glyphs are zeroed out
    assert torch.allclose(features[0, -4:], torch.zeros(4, D_MODEL))


def test_positional_encoding_4d():
    """Test 4D positional encoding."""
    B, M = 2, 8
    d_model = D_MODEL
    
    pos_enc = PositionalEncoding4D(d_model=d_model, use_fourier=True, num_freqs=64)
    coords = torch.rand(B, M, 4)  # [B, M, 4] - (x, y, w, h)
    
    pos_emb = pos_enc(coords)
    assert pos_emb.shape == (B, M, d_model)


def test_glyph_token_enrichment():
    """Test glyph token enrichment."""
    B, M = 2, 16
    d_model = D_MODEL
    
    enrichment = GlyphTokenEnrichment(d_model=d_model)
    glyph_features = torch.rand(B, M, d_model)
    glyph_coords = torch.rand(B, M, 4)  # [B, M, 4] - (x, y, w, h)
    valid_mask = torch.ones(B, M, dtype=torch.bool)
    
    enriched = enrichment(glyph_features, glyph_coords, valid_mask)
    assert enriched.shape == (B, M, d_model)


def test_glyph_token_enrichment_requires_coords_when_positional_encoding_enabled():
    enrichment = GlyphTokenEnrichment(
        d_model=8,
        enable_pos_encoding=True,
        use_fourier_pos=False,
    )

    with pytest.raises(ValueError, match="requires glyph_coords"):
        enrichment(torch.zeros(1, 2, 8), None, None)


def test_glyph_token_enrichment_rejects_misaligned_mask_coords_and_class_ids():
    enrichment = GlyphTokenEnrichment(
        d_model=8,
        enable_pos_encoding=True,
        use_fourier_pos=False,
    )
    features = torch.zeros(2, 3, 8)
    coords = torch.zeros(2, 3, 4)
    valid_mask = torch.ones(2, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match="valid_mask must have shape"):
        enrichment(features, coords, torch.ones(2, 2, dtype=torch.bool))

    with pytest.raises(ValueError, match="glyph_coords must have shape"):
        enrichment(features, torch.zeros(2, 2, 4), valid_mask)

    with pytest.raises(ValueError, match="char_class_ids must have shape"):
        enrichment(features, coords, valid_mask, torch.zeros(2, 1, dtype=torch.long))


def test_glyph_token_enrichment_rejects_invalid_char_class_ids():
    enrichment = GlyphTokenEnrichment(
        d_model=8,
        enable_pos_encoding=False,
        num_char_classes=4,
    )
    features = torch.zeros(1, 2, 8)
    coords = torch.zeros(1, 2, 4)
    valid_mask = torch.ones(1, 2, dtype=torch.bool)

    with pytest.raises(ValueError, match="char_class_ids must be in"):
        enrichment(features, coords, valid_mask, torch.tensor([[0, 4]], dtype=torch.long))

    with pytest.raises(ValueError, match="integer IDs"):
        enrichment(features, coords, valid_mask, torch.tensor([[0.0, 1.0]]))


def test_glyph_set_summarizer():
    """Test glyph set summarizer."""
    B, M = 2, 16
    d_model = D_MODEL
    num_queries = 8

    summarizer = GlyphSetSummarizer(d_model=d_model, num_queries=num_queries)
    glyph_tokens = torch.rand(B, M, d_model)
    valid_mask = torch.ones(B, M, dtype=torch.bool)
    valid_mask[0, -4:] = False  # Some padding

    summary, summary_valid = summarizer(glyph_tokens, valid_mask)
    assert summary.shape == (B, num_queries, d_model)
    assert summary_valid.shape == (B, num_queries)
    assert summary_valid.dtype == torch.bool
    # Query slots summarize sets rather than individual glyphs, so every query
    # remains active for every sample that has at least one valid glyph.
    counts = summary_valid.sum(dim=1).tolist()
    assert counts == [num_queries, num_queries]


def test_glyph_set_summarizer_empty_evidence():
    """Sample with no valid glyphs gets an all-False summary mask."""
    B, M = 3, 8
    d_model = D_MODEL
    num_queries = 8

    summarizer = GlyphSetSummarizer(d_model=d_model, num_queries=num_queries)
    glyph_tokens = torch.rand(B, M, d_model)
    valid_mask = torch.ones(B, M, dtype=torch.bool)
    valid_mask[1] = False  # second sample: no valid glyphs

    summary, summary_valid = summarizer(glyph_tokens, valid_mask)
    assert summary.shape == (B, num_queries, d_model)
    assert summary_valid[1].sum().item() == 0  # all-False for evidence-empty sample
    assert summary_valid[0].sum().item() == num_queries
    assert summary_valid[2].sum().item() == num_queries


def test_glyph_set_summarizer_empty_evidence_attention_is_zero():
    """Attention output should stay finite and zero for evidence-empty samples."""
    summarizer = GlyphSetSummarizer(
        d_model=8, num_queries=4, num_heads=2, dropout=0.0
    ).eval()
    glyph_tokens = torch.rand(2, 3, 8)
    valid_mask = torch.ones(2, 3, dtype=torch.bool)
    valid_mask[0, 1:] = False
    valid_mask[1] = False

    summary, summary_valid, attention = summarizer(glyph_tokens, valid_mask, return_attention=True)

    assert summary.shape == (2, 4, 8)
    assert attention.shape == (2, 4, 3)
    assert summary_valid[1].sum().item() == 0
    assert summary_valid[0].sum().item() == 4
    assert torch.all(attention[0, :, 1:] == 0)
    torch.testing.assert_close(attention[0].sum(dim=1), torch.ones(4))
    assert torch.all(summary[1] == 0)
    assert torch.all(attention[1] == 0)
    assert torch.isfinite(summary).all()
    assert torch.isfinite(attention).all()


def test_glyph_set_summarizer_rejects_mask_shape_mismatch():
    summarizer = GlyphSetSummarizer(d_model=8, num_queries=4, num_heads=2)
    glyph_tokens = torch.rand(2, 3, 8)

    with pytest.raises(ValueError, match="valid_mask must have shape"):
        summarizer(glyph_tokens, torch.ones(2, 2, dtype=torch.bool))


def test_apply_page_segment_to_glyph_coords():
    """Test page segment coordinate shifting for glyphs."""
    B, M = 2, 8
    coords = torch.rand(B, M, 4)  # [B, M, 4] - (x, y, w, h)
    page_segments = torch.tensor([[0, 0, 1, 1, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0]])
    
    shifted = apply_page_segment_to_glyph_coords(coords, page_segments)
    
    # Check that y-coordinates are shifted for page_segment=1
    assert shifted.shape == coords.shape
    for b in range(B):
        for m in range(M):
            if page_segments[b, m] == 1:
                assert shifted[b, m, 1] == coords[b, m, 1] + 1.0


def test_apply_page_segment_to_glyph_coords_rejects_broadcast_shape():
    coords = torch.zeros(2, 3, 4)
    page_segments = torch.tensor([[1], [0]])

    with pytest.raises(ValueError, match="page_segments must have shape"):
        apply_page_segment_to_glyph_coords(coords, page_segments)


def test_glyph_branch():
    """Test complete glyph branch."""
    B, M = 2, 16
    H, W = CHAR_PATCH_SIZE, CHAR_PATCH_SIZE

    branch = GlyphBranch(
        char_patch_size=CHAR_PATCH_SIZE,
        d_model=D_MODEL,
        num_summary_tokens=GLYPH_NUM_SUMMARY_TOKENS,
    )
    glyph_patches = torch.rand(B, M, 3, H, W)
    glyph_coords = torch.rand(B, M, 4)  # [B, M, 4] - (x, y, w, h)
    valid_mask = torch.ones(B, M, dtype=torch.bool)
    page_segments = torch.tensor([[0, 0, 1, 1] + [0] * 12, [0, 1, 0, 1] + [0] * 12])

    glyph_tokens, glyph_valid = branch(
        glyph_patches, glyph_coords, valid_mask=valid_mask, page_segments=page_segments
    )
    assert GLYPH_NUM_SUMMARY_TOKENS == 8
    if USE_BRANCH_ADAPTERS_AND_SUMMARIZERS:
        assert branch.set_summarizer is not None
        assert glyph_tokens.shape == (B, GLYPH_NUM_SUMMARY_TOKENS, D_MODEL)
        assert glyph_valid.all()
    else:
        assert branch.set_summarizer is None
        assert glyph_tokens.shape == (B, M, D_MODEL)
        assert torch.equal(glyph_valid, valid_mask)


def test_glyph_branch_rejects_invalid_char_class_ids_before_visual_encoding(monkeypatch):
    """Branch-level validation should fail before doing expensive visual encoding."""
    branch = GlyphBranch(
        char_patch_size=CHAR_PATCH_SIZE,
        d_model=D_MODEL,
        num_summary_tokens=GLYPH_NUM_SUMMARY_TOKENS,
    )

    def fail_visual_encoder(*_args, **_kwargs):
        raise AssertionError("visual encoder should not run for invalid char_class_ids")

    monkeypatch.setattr(branch.visual_encoder, "forward", fail_visual_encoder)

    glyph_patches = torch.zeros(1, 2, 3, CHAR_PATCH_SIZE, CHAR_PATCH_SIZE)
    glyph_coords = torch.zeros(1, 2, 4)
    valid_mask = torch.ones(1, 2, dtype=torch.bool)

    with pytest.raises(ValueError, match="char_class_ids must be in"):
        branch(
            glyph_patches,
            glyph_coords,
            valid_mask=valid_mask,
            char_class_ids=torch.tensor([[0, 999]], dtype=torch.long),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
