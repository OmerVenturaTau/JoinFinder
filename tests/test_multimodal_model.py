"""
Tests for MultiModal model.
"""

import torch
import pytest
from models import MultiModal
from models.multimodal_model import (
    _apply_batch_element_indices_to_lists,
    _word_page_segments_from_metadata,
)
from models.symmetric_fusion import SymmetricRetrievalFusion
from system import (
    TILE_SIZE,
    CHAR_PATCH_SIZE,
    D_MODEL,
    MAX_SELECTED_MANUSCRIPTS,
    MODALITY_DROPOUT_ENABLED,
    MODALITY_DROPOUT_PROB_VISUAL,
    MODALITY_DROPOUT_PROB_CHAR,
    MODALITY_DROPOUT_PROB_WORD,
    LATENT_DIM,
    SYMMETRIC_BRANCH_DIM,
    USE_BRANCH_ADAPTERS_AND_SUMMARIZERS,
)


class _CaptureTileBranch(torch.nn.Module):
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.d_model = d_model
        self.last_valid_mask = None

    def forward(self, tiles, tile_coords=None, valid_mask=None, page_segments=None, return_attention=False):
        self.last_valid_mask = valid_mask.detach().clone()
        tokens = torch.zeros(tiles.shape[0], tiles.shape[1], self.d_model, device=tiles.device)
        if return_attention:
            return tokens, None
        return tokens


class _AttentionTileBranch(torch.nn.Module):
    def forward(self, tiles, tile_coords=None, valid_mask=None,
                page_segments=None, return_attention=False):
        batch, tile_count = tiles.shape[:2]
        tokens = torch.zeros(batch, 2, D_MODEL, device=tiles.device)
        summary_valid = torch.ones(batch, 2, dtype=torch.bool, device=tiles.device)
        if return_attention:
            return tokens, summary_valid, None
        return tokens, summary_valid


class _DummyFusion(torch.nn.Module):
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.d_model = d_model
        self.last_tile_valid_mask = None
        self.last_word_valid_mask = None

    def forward(
        self,
        tile_tokens=None,
        tile_valid_mask=None,
        glyph_tokens=None,
        glyph_valid_mask=None,
        word_tokens=None,
        word_valid_mask=None,
    ):
        self.last_tile_valid_mask = (
            tile_valid_mask.detach().clone() if tile_valid_mask is not None else None
        )
        self.last_word_valid_mask = (
            word_valid_mask.detach().clone() if word_valid_mask is not None else None
        )
        for tokens in (tile_tokens, glyph_tokens, word_tokens):
            if tokens is not None:
                B = tokens.shape[0]
                device = tokens.device
                return torch.zeros(B, 1, self.d_model, device=device), None
        raise ValueError("At least one modality must be provided")


class _DummyHead(torch.nn.Module):
    def __init__(self, num_classes=MAX_SELECTED_MANUSCRIPTS):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, latent_repr):
        B = latent_repr.shape[0]
        device = latent_repr.device
        return (
            torch.zeros(B, self.num_classes, device=device),
            torch.zeros(B, LATENT_DIM, device=device),
        )


class _DummyWordQualityFilter:
    def filter_words_in_line(self, line_words):
        return [
            (idx, word, meta)
            for idx, word, meta in line_words
            if word != "bad" and float(meta.get("wc", 1.0)) >= 0.5
        ]


class _DummyWordBranch(torch.nn.Module):
    def __init__(self, d_model=D_MODEL, num_tokens=2):
        super().__init__()
        self.d_model = d_model
        self.num_tokens = num_tokens
        self.quality_filter = _DummyWordQualityFilter()

    def forward(self, words, word_metadata, device, page_segments=None):
        B = len(words)
        tokens = torch.zeros(B, self.num_tokens, self.d_model, device=device)
        valid = torch.zeros(B, self.num_tokens, dtype=torch.bool, device=device)
        for b, sample_words in enumerate(words):
            sample_metadata = word_metadata[b] if b < len(word_metadata) else []
            line_words = [
                (idx, word, meta)
                for idx, (word, meta) in enumerate(zip(sample_words, sample_metadata))
            ]
            if self.quality_filter.filter_words_in_line(line_words):
                valid[b, :] = True
        return tokens, valid


class _AttentionGlyphBranch(torch.nn.Module):
    def forward(self, glyph_patches, glyph_coords, valid_mask=None,
                page_segments=None, char_class_ids=None, return_attention=False,
                debug_paths=None):
        batch, glyph_count = glyph_patches.shape[:2]
        tokens = torch.zeros(batch, 2, D_MODEL, device=glyph_patches.device)
        summary_valid = torch.ones(batch, 2, dtype=torch.bool, device=glyph_patches.device)
        return tokens, summary_valid, None


class _AttentionWordBranch(torch.nn.Module):
    def forward(self, words, word_metadata, device, page_segments=None,
                return_attention=False, return_line_to_word_attn=False,
                return_summary_to_line_attn=False):
        batch = len(words)
        tokens = torch.zeros(batch, 2, D_MODEL, device=device)
        valid = torch.ones(batch, 2, dtype=torch.bool, device=device)
        local = torch.zeros(batch, 3, 3, device=device)
        line_to_word = torch.eye(2, device=device).unsqueeze(0).expand(batch, -1, -1).clone()
        summary_to_line = None
        return tokens, valid, local, line_to_word, summary_to_line


def test_apply_batch_element_indices_rejects_out_of_range_lists():
    with pytest.raises(ValueError, match="outside words length"):
        _apply_batch_element_indices_to_lists(
            torch.tensor([0, 2]),
            words=[["a"], ["b"]],
            word_metadata=[[{}], [{}]],
            word_page_segments=[[0], [0]],
            paths=["a", "b"],
        )


def test_source_evidence_fraction_uses_summary_capacity():
    mask = torch.tensor([
        [True, False, False, False],
        [True, True, True, True],
    ])

    torch.testing.assert_close(
        MultiModal._valid_fraction(mask, capacity=8),
        torch.tensor([1 / 8, 4 / 8]),
    )


def test_word_page_segments_from_metadata_accepts_float_like_strings():
    word_metadata = [[{"page_segment": "1.0"}, {"page_segment": 0}], [{"page_segment": None}]]

    assert _word_page_segments_from_metadata(word_metadata) == [[1, 0], [0]]


def test_forward_subsamples_default_tile_mask_when_mask_is_omitted():
    B, N = 2, 5
    model = MultiModal(
        num_classes=MAX_SELECTED_MANUSCRIPTS,
        use_visual_mod=True,
        use_char_mod=False,
        use_word_mod=False,
    )
    capture_tile_branch = _CaptureTileBranch()
    model.tile_branch = capture_tile_branch
    model.fusion = _DummyFusion()
    model.head = _DummyHead()
    model.train()
    model.modality_dropout_enabled = False
    model.token_subsample_enabled = True
    model.token_subsample_prob = 1.0
    model.token_subsample_min_tiles = 1
    model.token_subsample_frac_range = (0.0, 0.0)

    tiles = torch.rand(B, N, 3, TILE_SIZE, TILE_SIZE)
    tile_coords = torch.rand(B, N, 2)

    model(tiles=tiles, tile_coords=tile_coords, tile_valid_mask=None)

    assert capture_tile_branch.last_valid_mask.shape == (B, N)
    assert capture_tile_branch.last_valid_mask.sum(dim=1).tolist() == [1, 1]


def test_modality_dropout_rescue_uses_per_sample_availability():
    B, N = 1, 3
    model = MultiModal(
        num_classes=MAX_SELECTED_MANUSCRIPTS,
        use_visual_mod=False,
        use_char_mod=False,
        use_word_mod=False,
    )
    capture_tile_branch = _CaptureTileBranch()
    capture_fusion = _DummyFusion()
    model.use_visual_mod = True
    model.use_word_mod = True
    model.tile_branch = capture_tile_branch
    model.word_branch = _DummyWordBranch()
    model.fusion = capture_fusion
    model.head = _DummyHead()
    model.train()
    model.modality_dropout_enabled = True
    model.modality_dropout_prob_visual = 1.0
    model.modality_dropout_prob_word = 1.0
    model.token_subsample_enabled = False

    tiles = torch.rand(B, N, 3, TILE_SIZE, TILE_SIZE)
    tile_coords = torch.rand(B, N, 2)
    tile_valid_mask = torch.ones(B, N, dtype=torch.bool)

    model(
        tiles=tiles,
        tile_coords=tile_coords,
        tile_valid_mask=tile_valid_mask,
        words=[[]],
        word_metadata=[[]],
    )

    assert capture_tile_branch.last_valid_mask.tolist() == [[True, True, True]]
    assert capture_fusion.last_tile_valid_mask.tolist() == [[True, True, True]]
    assert capture_fusion.last_word_valid_mask.tolist() == [[False, False]]


def test_modality_dropout_rescue_ignores_filtered_word_only_stream():
    B, N = 1, 3
    model = MultiModal(
        num_classes=MAX_SELECTED_MANUSCRIPTS,
        use_visual_mod=False,
        use_char_mod=False,
        use_word_mod=False,
    )
    capture_tile_branch = _CaptureTileBranch()
    capture_fusion = _DummyFusion()
    model.use_visual_mod = True
    model.use_word_mod = True
    model.tile_branch = capture_tile_branch
    model.word_branch = _DummyWordBranch()
    model.fusion = capture_fusion
    model.head = _DummyHead()
    model.train()
    model.modality_dropout_enabled = True
    model.modality_dropout_prob_visual = 1.0
    model.modality_dropout_prob_word = 1.0
    model.token_subsample_enabled = False

    tiles = torch.rand(B, N, 3, TILE_SIZE, TILE_SIZE)
    tile_coords = torch.rand(B, N, 2)
    tile_valid_mask = torch.ones(B, N, dtype=torch.bool)

    model(
        tiles=tiles,
        tile_coords=tile_coords,
        tile_valid_mask=tile_valid_mask,
        words=[["bad"]],
        word_metadata=[[{"wc": 0.1, "width": 1.0, "height": 1.0}]],
    )

    assert capture_tile_branch.last_valid_mask.tolist() == [[True, True, True]]
    assert capture_fusion.last_tile_valid_mask.tolist() == [[True, True, True]]
    assert capture_fusion.last_word_valid_mask.tolist() == [[False, False]]


def test_forward_rejects_word_list_batch_size_mismatch_with_tensor_batch():
    B, N = 2, 2
    model = MultiModal(
        num_classes=MAX_SELECTED_MANUSCRIPTS,
        use_visual_mod=False,
        use_char_mod=False,
        use_word_mod=False,
    )
    model.use_visual_mod = True
    model.use_word_mod = True
    model.tile_branch = _CaptureTileBranch()
    model.fusion = _DummyFusion()
    model.head = _DummyHead()
    model.eval()

    tiles = torch.rand(B, N, 3, TILE_SIZE, TILE_SIZE)
    tile_coords = torch.rand(B, N, 2)

    with pytest.raises(ValueError, match="words batch size 2"):
        model(
            tiles=tiles,
            tile_coords=tile_coords,
            words=[["only-one-sample"]],
            word_metadata=[[{"wc": 1.0, "width": 1.0, "height": 1.0}]],
        )


def test_multimodal_creation():
    """Test MultiModal model creation with different modality configurations."""
    # Test with all modalities enabled
    model = MultiModal(
        num_classes=MAX_SELECTED_MANUSCRIPTS,
        use_visual_mod=True,
        use_char_mod=True,
        use_word_mod=True,
    )
    assert model.use_visual_mod is True
    assert model.use_char_mod is True
    assert model.use_word_mod is True
    assert hasattr(model, 'tile_branch')
    assert hasattr(model, 'glyph_branch')
    assert hasattr(model, 'word_branch')
    
    # Test with only visual modality
    model = MultiModal(
        num_classes=MAX_SELECTED_MANUSCRIPTS,
        use_visual_mod=True,
        use_char_mod=False,
        use_word_mod=False,
    )
    assert model.use_visual_mod is True
    assert model.use_char_mod is False
    assert model.use_word_mod is False
    assert hasattr(model, 'tile_branch')
    assert not hasattr(model, 'glyph_branch')
    assert not hasattr(model, 'word_branch')
    
    # Test with only char modality
    model = MultiModal(
        num_classes=MAX_SELECTED_MANUSCRIPTS,
        use_visual_mod=False,
        use_char_mod=True,
        use_word_mod=False,
    )
    assert model.use_visual_mod is False
    assert model.use_char_mod is True
    assert model.use_word_mod is False
    assert not hasattr(model, 'tile_branch')
    assert hasattr(model, 'glyph_branch')
    assert not hasattr(model, 'word_branch')


def test_multimodal_modality_dropout_settings():
    """Test that modality dropout settings are correctly initialized."""
    model = MultiModal(
        num_classes=MAX_SELECTED_MANUSCRIPTS,
        use_visual_mod=True,
        use_char_mod=True,
        use_word_mod=True,
    )
    
    assert model.modality_dropout_enabled == MODALITY_DROPOUT_ENABLED
    assert model.modality_dropout_prob_visual == MODALITY_DROPOUT_PROB_VISUAL
    assert model.modality_dropout_prob_char == MODALITY_DROPOUT_PROB_CHAR
    assert model.modality_dropout_prob_word == MODALITY_DROPOUT_PROB_WORD


def test_multimodal_forward_visual_only():
    """Test MultiModal forward pass with visual modality only."""
    B, N = 2, 8
    H, W = TILE_SIZE, TILE_SIZE
    
    model = MultiModal(
        num_classes=MAX_SELECTED_MANUSCRIPTS,
        use_visual_mod=True,
        use_char_mod=False,
        use_word_mod=False,
    )
    model.eval()  # Disable dropout for deterministic test
    
    tiles = torch.rand(B, N, 3, H, W)
    tile_coords = torch.rand(B, N, 2)
    tile_valid_mask = torch.ones(B, N, dtype=torch.bool)
    
    logits, latent, aux_latents = model(
        tiles=tiles,
        tile_coords=tile_coords,
        tile_valid_mask=tile_valid_mask,
    )
    
    assert logits.shape == (B, MAX_SELECTED_MANUSCRIPTS)
    assert latent.shape == (B, LATENT_DIM)
    assert isinstance(aux_latents, dict)


def test_multimodal_forward_char_only():
    """Test MultiModal forward pass with char modality only."""
    B, M = 2, 16
    H, W = CHAR_PATCH_SIZE, CHAR_PATCH_SIZE
    
    model = MultiModal(
        num_classes=MAX_SELECTED_MANUSCRIPTS,
        use_visual_mod=False,
        use_char_mod=True,
        use_word_mod=False,
    )
    model.eval()  # Disable dropout for deterministic test
    
    glyph_patches = torch.rand(B, M, 3, H, W)
    glyph_coords = torch.rand(B, M, 4)
    glyph_valid_mask = torch.ones(B, M, dtype=torch.bool)
    
    logits, latent, aux_latents = model(
        glyph_patches=glyph_patches,
        glyph_coords=glyph_coords,
        glyph_valid_mask=glyph_valid_mask,
    )
    
    assert logits.shape == (B, MAX_SELECTED_MANUSCRIPTS)
    assert latent.shape == (B, LATENT_DIM)
    assert isinstance(aux_latents, dict)


def test_multimodal_forward_visual_and_char():
    """Test MultiModal forward pass with visual and char modalities."""
    B, N, M = 2, 8, 16
    H_tile, W_tile = TILE_SIZE, TILE_SIZE
    H_char, W_char = CHAR_PATCH_SIZE, CHAR_PATCH_SIZE
    
    model = MultiModal(
        num_classes=MAX_SELECTED_MANUSCRIPTS,
        use_visual_mod=True,
        use_char_mod=True,
        use_word_mod=False,
    )
    model.eval()  # Disable dropout for deterministic test
    
    tiles = torch.rand(B, N, 3, H_tile, W_tile)
    tile_coords = torch.rand(B, N, 2)
    tile_valid_mask = torch.ones(B, N, dtype=torch.bool)
    
    glyph_patches = torch.rand(B, M, 3, H_char, W_char)
    glyph_coords = torch.rand(B, M, 4)
    glyph_valid_mask = torch.ones(B, M, dtype=torch.bool)
    
    logits, latent, aux_latents = model(
        tiles=tiles,
        tile_coords=tile_coords,
        tile_valid_mask=tile_valid_mask,
        glyph_patches=glyph_patches,
        glyph_coords=glyph_coords,
        glyph_valid_mask=glyph_valid_mask,
    )
    
    assert logits.shape == (B, MAX_SELECTED_MANUSCRIPTS)
    assert latent.shape == (B, LATENT_DIM)
    assert isinstance(aux_latents, dict)


def test_multimodal_modality_dropout_training():
    """Test that modality dropout only occurs during training."""
    B, N = 2, 8
    H, W = TILE_SIZE, TILE_SIZE
    
    model = MultiModal(
        num_classes=MAX_SELECTED_MANUSCRIPTS,
        use_visual_mod=True,
        use_char_mod=True,
        use_word_mod=True,
    )
    
    tiles = torch.rand(B, N, 3, H, W)
    tile_coords = torch.rand(B, N, 2)
    tile_valid_mask = torch.ones(B, N, dtype=torch.bool)
    
    glyph_patches = torch.rand(B, 16, 3, CHAR_PATCH_SIZE, CHAR_PATCH_SIZE)
    glyph_coords = torch.rand(B, 16, 4)
    glyph_valid_mask = torch.ones(B, 16, dtype=torch.bool)
    
    # In eval mode, all enabled modalities should be used
    model.eval()
    logits_eval, latent_eval, aux_latents_eval = model(
        tiles=tiles,
        tile_coords=tile_coords,
        tile_valid_mask=tile_valid_mask,
        glyph_patches=glyph_patches,
        glyph_coords=glyph_coords,
        glyph_valid_mask=glyph_valid_mask,
    )
    
    # In training mode, modalities may be dropped (if dropout is enabled)
    model.train()
    logits_train, latent_train, aux_latents_train = model(
        tiles=tiles,
        tile_coords=tile_coords,
        tile_valid_mask=tile_valid_mask,
        glyph_patches=glyph_patches,
        glyph_coords=glyph_coords,
        glyph_valid_mask=glyph_valid_mask,
    )
    
    # Both should produce valid outputs
    assert logits_eval.shape == (B, MAX_SELECTED_MANUSCRIPTS)
    assert logits_train.shape == (B, MAX_SELECTED_MANUSCRIPTS)
    assert latent_eval.shape == (B, LATENT_DIM)
    assert latent_train.shape == (B, LATENT_DIM)
    assert isinstance(aux_latents_train, dict)


def test_multimodal_forward_features():
    """Test forward_features method returns only latent representation."""
    B, N = 2, 8
    H, W = TILE_SIZE, TILE_SIZE
    
    model = MultiModal(
        num_classes=MAX_SELECTED_MANUSCRIPTS,
        use_visual_mod=True,
        use_char_mod=False,
        use_word_mod=False,
    )
    model.eval()
    
    tiles = torch.rand(B, N, 3, H, W)
    tile_coords = torch.rand(B, N, 2)
    tile_valid_mask = torch.ones(B, N, dtype=torch.bool)
    
    latent, aux_latents = model.forward_features(
        tiles=tiles,
        tile_coords=tile_coords,
        tile_valid_mask=tile_valid_mask,
    )
    
    assert latent.shape == (B, LATENT_DIM)
    assert len(latent.shape) == 2  # Should be 2D, not 3D
    assert isinstance(aux_latents, dict)


def test_multimodal_state_dict_saving():
    """Test that model can save and load state dict with modality flags."""
    model = MultiModal(
        num_classes=MAX_SELECTED_MANUSCRIPTS,
        use_visual_mod=True,
        use_char_mod=True,
        use_word_mod=False,
    )
    
    # Save state dict
    state_dict = model.state_dict()
    
    # Create new model with same configuration
    model2 = MultiModal(
        num_classes=MAX_SELECTED_MANUSCRIPTS,
        use_visual_mod=True,
        use_char_mod=True,
        use_word_mod=False,
    )
    
    # Load state dict (should work with strict=False if modalities differ)
    missing_keys, unexpected_keys = model2.load_state_dict(state_dict, strict=False)
    
    # Should load successfully (may have some missing/unexpected keys if architecture differs)
    assert isinstance(missing_keys, list)
    assert isinstance(unexpected_keys, list)


def test_multimodal_forward_without_glyph_coords():
    """Test that model skips glyph processing when glyph_coords is None."""
    B, N, M = 2, 8, 16
    H_tile, W_tile = TILE_SIZE, TILE_SIZE
    H_char, W_char = CHAR_PATCH_SIZE, CHAR_PATCH_SIZE
    
    model = MultiModal(
        num_classes=MAX_SELECTED_MANUSCRIPTS,
        use_visual_mod=True,
        use_char_mod=True,
        use_word_mod=False,
    )
    model.eval()
    
    tiles = torch.rand(B, N, 3, H_tile, W_tile)
    tile_coords = torch.rand(B, N, 2)
    tile_valid_mask = torch.ones(B, N, dtype=torch.bool)
    
    glyph_patches = torch.rand(B, M, 3, H_char, W_char)
    glyph_valid_mask = torch.ones(B, M, dtype=torch.bool)
    # glyph_coords is None - should skip glyph processing
    
    # Should not crash - glyphs will be skipped
    logits, latent, aux_latents = model(
        tiles=tiles,
        tile_coords=tile_coords,
        tile_valid_mask=tile_valid_mask,
        glyph_patches=glyph_patches,
        glyph_valid_mask=glyph_valid_mask,
        glyph_coords=None,  # Missing coordinates
    )
    
    # Should still produce valid output (using only tiles)
    assert logits.shape == (B, MAX_SELECTED_MANUSCRIPTS)
    assert latent.shape == (B, LATENT_DIM)
    assert isinstance(aux_latents, dict)


def test_multimodal_forward_without_glyph_patches():
    """Test that model skips glyph processing when glyph_patches is None."""
    B, N = 2, 8
    H, W = TILE_SIZE, TILE_SIZE
    
    model = MultiModal(
        num_classes=MAX_SELECTED_MANUSCRIPTS,
        use_visual_mod=True,
        use_char_mod=True,
        use_word_mod=False,
    )
    model.eval()
    
    tiles = torch.rand(B, N, 3, H, W)
    tile_coords = torch.rand(B, N, 2)
    tile_valid_mask = torch.ones(B, N, dtype=torch.bool)
    
    # glyph_patches is None - should skip glyph processing
    glyph_coords = torch.rand(B, 16, 4)
    glyph_valid_mask = torch.ones(B, 16, dtype=torch.bool)
    
    # Should not crash - glyphs will be skipped
    logits, latent, aux_latents = model(
        tiles=tiles,
        tile_coords=tile_coords,
        tile_valid_mask=tile_valid_mask,
        glyph_patches=None,  # Missing patches
        glyph_coords=glyph_coords,
        glyph_valid_mask=glyph_valid_mask,
    )
    
    # Should still produce valid output (using only tiles)
    assert logits.shape == (B, MAX_SELECTED_MANUSCRIPTS)
    assert latent.shape == (B, LATENT_DIM)
    assert isinstance(aux_latents, dict)


def test_multimodal_forward_with_attention_missing_glyph_coords():
    """Test forward_with_attention skips glyphs when glyph_coords is None."""
    B, N, M = 2, 8, 16
    H_tile, W_tile = TILE_SIZE, TILE_SIZE
    H_char, W_char = CHAR_PATCH_SIZE, CHAR_PATCH_SIZE
    
    model = MultiModal(
        num_classes=MAX_SELECTED_MANUSCRIPTS,
        use_visual_mod=True,
        use_char_mod=True,
        use_word_mod=False,
    )
    model.eval()
    
    tiles = torch.rand(B, N, 3, H_tile, W_tile)
    tile_coords = torch.rand(B, N, 2)
    tile_valid_mask = torch.ones(B, N, dtype=torch.bool)
    
    glyph_patches = torch.rand(B, M, 3, H_char, W_char)
    glyph_valid_mask = torch.ones(B, M, dtype=torch.bool)
    # glyph_coords is None
    
    # Should not crash - glyphs will be skipped
    logits, latent, attention_dict = model.forward_with_attention(
        tiles=tiles,
        tile_coords=tile_coords,
        tile_valid_mask=tile_valid_mask,
        glyph_patches=glyph_patches,
        glyph_valid_mask=glyph_valid_mask,
        glyph_coords=None,  # Missing coordinates
    )
    
    # Should still produce valid output
    assert logits.shape == (B, MAX_SELECTED_MANUSCRIPTS)
    assert latent.shape == (B, LATENT_DIM)
    # Character attention should not be in dict if glyphs were skipped
    assert 'character' not in attention_dict or attention_dict.get('character') is None


def test_symmetric_attention_maps_cover_tiles_glyphs_and_words():
    model = MultiModal(
        num_classes=MAX_SELECTED_MANUSCRIPTS,
        use_visual_mod=True,
        use_char_mod=True,
        use_word_mod=True,
    ).eval()
    model.tile_branch = _AttentionTileBranch()
    model.glyph_branch = _AttentionGlyphBranch()
    model.word_branch = _AttentionWordBranch()
    model.fusion = SymmetricRetrievalFusion(
        d_model=D_MODEL,
        enabled_modalities=("tile", "glyph", "word"),
        use_adapters=USE_BRANCH_ADAPTERS_AND_SUMMARIZERS,
        branch_dim=SYMMETRIC_BRANCH_DIM,
    )

    tile_mask = torch.tensor([[True, True]])
    glyph_mask = torch.tensor([[True, True]])
    _, _, attention = model.forward_with_attention(
        tiles=torch.zeros(1, 2, 3, 4, 4),
        tile_coords=torch.zeros(1, 2, 2),
        tile_valid_mask=tile_mask,
        glyph_patches=torch.zeros(1, 2, 3, 4, 4),
        glyph_coords=torch.zeros(1, 2, 4),
        glyph_valid_mask=glyph_mask,
        char_class_ids=torch.zeros(1, 2, dtype=torch.long),
        words=[["א", "מ"]],
        word_metadata=[[{}, {}]],
    )

    assert {"visual", "character", "word", "fusion"}.issubset(attention)
    torch.testing.assert_close(attention["fusion"].sum(), torch.tensor(1.0))

    tile_scores = attention["visual"][0, 0, 1:]
    glyph_scores = attention["character"][0, 0, 1:]
    word_scores = attention["word"][0, 0, 1:]
    torch.testing.assert_close(tile_scores, torch.tensor([1 / 6, 1 / 6]))
    torch.testing.assert_close(glyph_scores, torch.tensor([1 / 6, 1 / 6]))
    torch.testing.assert_close(word_scores, torch.tensor([1 / 6, 1 / 6]))
    torch.testing.assert_close(
        attention["reliability_weights"], torch.full((1, 3), 1 / 3)
    )
    assert "tile_query_to_tile" not in attention
    assert "glyph_query_to_glyph" not in attention
    assert "word_query_to_line" not in attention


def test_multimodal_forward_with_aux_latents():
    """Test that auxiliary latents are correctly returned when requested."""
    B, N, M = 2, 4, 4
    model = MultiModal(
        num_classes=MAX_SELECTED_MANUSCRIPTS,
        use_visual_mod=True,
        use_char_mod=True,
        use_word_mod=True,
    )
    model.eval()
    
    tiles = torch.rand(B, N, 3, TILE_SIZE, TILE_SIZE)
    tile_coords = torch.rand(B, N, 2)
    tile_valid_mask = torch.ones(B, N, dtype=torch.bool)
    
    glyph_patches = torch.rand(B, M, 3, CHAR_PATCH_SIZE, CHAR_PATCH_SIZE)
    glyph_coords = torch.rand(B, M, 4)
    glyph_valid_mask = torch.ones(B, M, dtype=torch.bool)
    
    # Words
    words = [["א", "ב"], ["ג", "ד"]]
    word_metadata = [[{}, {}], [{}, {}]]
    
    # Request auxiliary latents
    logits, latent, aux_latents = model(
        tiles=tiles,
        tile_coords=tile_coords,
        tile_valid_mask=tile_valid_mask,
        glyph_patches=glyph_patches,
        glyph_coords=glyph_coords,
        glyph_valid_mask=glyph_valid_mask,
        words=words,
        word_metadata=word_metadata,
        return_aux_latents=True,
    )
    
    assert isinstance(aux_latents, dict)
    assert 'tile' in aux_latents
    assert 'glyph' in aux_latents
    assert 'word' in aux_latents
    expected_branch_dim = (
        SYMMETRIC_BRANCH_DIM
        if USE_BRANCH_ADAPTERS_AND_SUMMARIZERS
        else D_MODEL
    )
    assert aux_latents['tile'].shape == (B, expected_branch_dim)
    assert aux_latents['glyph'].shape == (B, expected_branch_dim)
    assert aux_latents['word'].shape == (B, expected_branch_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
