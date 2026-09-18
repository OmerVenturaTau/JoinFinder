"""
Tests for character-class ID mapping, balanced glyph selection,
and alphabetical sorting used in visualization.
"""

import numpy as np
import torch
import pytest

from system import (
    GLYPH_INPUT_ALPHABET,
    HEBREW_ALPHABET,
    NUM_GLYPH_CLASSES,
    GLYPHS_PER_CLASS,
    MAX_CHARS_PER_IMAGE,
    D_MODEL,
    CHAR_PATCH_SIZE,
    GLYPH_NUM_SUMMARY_TOKENS,
    USE_BRANCH_ADAPTERS_AND_SUMMARIZERS,
    OCR_GLYPH_CONFIDENCE_THRESHOLD,
    OCR_STRING_CONFIDENCE_THRESHOLD,
)
from utilities.VisionModule.xml_character_extraction import (
    char_to_class_id,
    _select_balanced_glyphs,
    filter_glyphs_fallback,
    filter_strings_and_glyphs,
)
from utilities.VisionModule.alto_parser import GlyphInfo, StringInfo
from models.glyph_branch import GlyphTokenEnrichment, GlyphBranch


# ---------------------------------------------------------------------------
# char_to_class_id
# ---------------------------------------------------------------------------

class TestCharToClassId:
    def test_input_alphabet_uses_requested_letters(self):
        assert GLYPH_INPUT_ALPHABET == list("אמ")
        assert all(letter in HEBREW_ALPHABET for letter in GLYPH_INPUT_ALPHABET)

    def test_known_input_letters(self):
        for i, letter in enumerate(GLYPH_INPUT_ALPHABET):
            assert char_to_class_id(letter) == i

    def test_unknown_returns_last_class(self):
        unknown_id = NUM_GLYPH_CLASSES - 1
        assert char_to_class_id("ב") == unknown_id
        assert char_to_class_id("X") == unknown_id
        assert char_to_class_id("1") == unknown_id
        assert char_to_class_id("") == unknown_id

    def test_all_ids_in_range(self):
        for letter in GLYPH_INPUT_ALPHABET:
            cid = char_to_class_id(letter)
            assert 0 <= cid < NUM_GLYPH_CLASSES

    def test_unique_ids_per_letter(self):
        ids = [char_to_class_id(ch) for ch in GLYPH_INPUT_ALPHABET]
        assert len(set(ids)) == len(GLYPH_INPUT_ALPHABET)


# ---------------------------------------------------------------------------
# _select_balanced_glyphs
# ---------------------------------------------------------------------------

def _make_glyph(char: str, gc: float = 0.9, hpos: float = 0, vpos: float = 0) -> GlyphInfo:
    """Create a minimal GlyphInfo for testing."""
    return GlyphInfo(char=char, gc=gc, hpos=hpos, vpos=vpos, width=10.0, height=10.0)


class TestSelectBalancedGlyphs:
    def test_basic_balance(self):
        """Only configured input glyph classes should be selected."""
        word_glyphs = [_make_glyph("א", gc=0.9) for _ in range(GLYPHS_PER_CLASS)] + \
                      [_make_glyph("ב", gc=0.9) for _ in range(GLYPHS_PER_CLASS)]
        result = _select_balanced_glyphs(
            word_glyphs,
            [],
            max_chars=GLYPHS_PER_CLASS,
            per_class=GLYPHS_PER_CLASS,
        )
        aleph_count = sum(1 for g in result if g.char == "א")
        bet_count = sum(1 for g in result if g.char == "ב")
        assert aleph_count == GLYPHS_PER_CLASS
        assert bet_count == 0

    def test_respects_max_chars(self):
        word_glyphs = [_make_glyph(ch) for ch in GLYPH_INPUT_ALPHABET for _ in range(6)]
        result = _select_balanced_glyphs(word_glyphs, [], max_chars=16)
        assert len(result) <= 16

    def test_fills_fixed_slots_for_all_selected_inputs(self):
        word_glyphs = [_make_glyph(ch, gc=0.9) for ch in GLYPH_INPUT_ALPHABET for _ in range(12)] + \
                      [_make_glyph("ב", gc=0.9) for _ in range(12)]
        result = _select_balanced_glyphs(word_glyphs, [], max_chars=MAX_CHARS_PER_IMAGE)
        assert MAX_CHARS_PER_IMAGE == 16
        assert GLYPHS_PER_CLASS == 8
        assert len(result) == 16
        assert [g.char for g in result] == [ch for ch in GLYPH_INPUT_ALPHABET for _ in range(GLYPHS_PER_CLASS)]

    def test_fallback_fills_gaps(self):
        """Fallback glyphs should supply allowed input glyph classes."""
        word_glyphs = []
        fallback_glyphs = [_make_glyph("א") for _ in range(4)] + [_make_glyph("ב") for _ in range(4)]
        result = _select_balanced_glyphs(word_glyphs, fallback_glyphs, max_chars=8)
        chars = [g.char for g in result]
        assert "א" in chars
        assert "ב" not in chars

    def test_word_glyphs_preferred_over_fallback(self):
        """For the same class, word_glyphs should appear before fallback."""
        word_g = _make_glyph("א", gc=0.95)
        fb_g = _make_glyph("א", gc=0.85)
        result = _select_balanced_glyphs([word_g], [fb_g], max_chars=2, per_class=2)
        assert result[0] is word_g

    def test_empty_inputs(self):
        result = _select_balanced_glyphs([], [], max_chars=10)
        assert result == []

    def test_higher_confidence_preferred(self):
        """Within a class, higher confidence glyphs should be selected first."""
        glyphs = [
            _make_glyph("א", gc=0.5),
            _make_glyph("א", gc=0.99),
            _make_glyph("א", gc=0.7),
        ]
        result = _select_balanced_glyphs(glyphs, [], max_chars=2, per_class=2)
        assert len(result) == 2
        gcs = [g.gc for g in result]
        assert gcs[0] >= gcs[1], "Should be sorted by descending confidence"


# ---------------------------------------------------------------------------
# filter_glyphs_fallback
# ---------------------------------------------------------------------------

def _make_string(content: str, wc: float, glyphs: list) -> StringInfo:
    """Create a minimal StringInfo for testing."""
    s = StringInfo.__new__(StringInfo)
    s.content = content
    s.wc = wc
    s.glyphs = glyphs
    s.hpos = 0
    s.vpos = 0
    s.width = 100
    s.height = 20
    s.id = "test"
    return s


class TestFilterGlyphsFallback:
    def test_fallback_returns_low_word_confidence_glyphs(self):
        """Allowed glyphs from low-wc words that individually pass glyph threshold should be returned."""
        g1 = _make_glyph("א", gc=OCR_GLYPH_CONFIDENCE_THRESHOLD + 0.1)
        g2 = _make_glyph("ב", gc=OCR_GLYPH_CONFIDENCE_THRESHOLD + 0.1)
        s = _make_string("אב", wc=OCR_STRING_CONFIDENCE_THRESHOLD - 0.1, glyphs=[g1, g2])
        result = filter_glyphs_fallback([s], word_glyph_ids=set())
        assert result == [g1]

    def test_fallback_excludes_already_selected(self):
        g1 = _make_glyph("א", gc=OCR_GLYPH_CONFIDENCE_THRESHOLD + 0.1)
        s = _make_string("אב", wc=OCR_STRING_CONFIDENCE_THRESHOLD - 0.1, glyphs=[g1])
        result = filter_glyphs_fallback([s], word_glyph_ids={id(g1)})
        assert len(result) == 0

    def test_fallback_excludes_high_wc_words(self):
        """Glyphs from high-wc words are primary, not fallback."""
        g1 = _make_glyph("א", gc=0.9)
        s = _make_string("אב", wc=OCR_STRING_CONFIDENCE_THRESHOLD + 0.1, glyphs=[g1])
        result = filter_glyphs_fallback([s], word_glyph_ids=set())
        assert len(result) == 0

    def test_fallback_excludes_low_glyph_confidence(self):
        g1 = _make_glyph("א", gc=OCR_GLYPH_CONFIDENCE_THRESHOLD - 0.1)
        s = _make_string("אב", wc=OCR_STRING_CONFIDENCE_THRESHOLD - 0.1, glyphs=[g1])
        result = filter_glyphs_fallback([s], word_glyph_ids=set())
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Visualization alphabetical sort helper (unit-testable logic)
# ---------------------------------------------------------------------------

def sort_by_char_class(metadata_list, scores):
    """
    Replicates the sorting logic from visualize_character_patches_attention.
    Returns (sorted_indices, sorted_metadata, sorted_scores).
    """
    items = [(m.get('char_class_id', 999), i, m) for i, m in enumerate(metadata_list)]
    items.sort(key=lambda x: x[0])
    sorted_indices = [i for _, i, _ in items]
    sorted_metadata = [m for _, _, m in items]
    sorted_scores = [scores[i] for i in sorted_indices]
    return sorted_indices, sorted_metadata, sorted_scores


class TestVisualizationSorting:
    def test_sort_aleph_before_unknown(self):
        """Configured input letters should sort before unknown/non-input letters."""
        md = [
            {'char': 'ג', 'char_class_id': char_to_class_id('ג')},
            {'char': 'א', 'char_class_id': char_to_class_id('א')},
            {'char': 'ב', 'char_class_id': char_to_class_id('ב')},
        ]
        scores = [0.3, 0.8, 0.5]
        sorted_idx, sorted_md, sorted_sc = sort_by_char_class(md, scores)

        assert [m['char'] for m in sorted_md] == ['א', 'ג', 'ב']
        assert sorted_sc == [0.8, 0.3, 0.5]

    def test_sort_with_unknown_class(self):
        """Unknown characters (class_id = NUM_GLYPH_CLASSES-1) should sort last."""
        md = [
            {'char': 'X', 'char_class_id': char_to_class_id('X')},
            {'char': 'א', 'char_class_id': char_to_class_id('א')},
        ]
        scores = [0.1, 0.9]
        _, sorted_md, _ = sort_by_char_class(md, scores)
        assert sorted_md[0]['char'] == 'א'
        assert sorted_md[-1]['char'] == 'X'

    def test_sort_preserves_all_items(self):
        md = [{'char': ch, 'char_class_id': char_to_class_id(ch)} for ch in "שתרקצפעסנ"]
        scores = list(range(len(md)))
        sorted_idx, sorted_md, sorted_sc = sort_by_char_class(md, scores)
        assert len(sorted_md) == len(md)
        assert set(sorted_sc) == set(scores)

    def test_sort_empty_list(self):
        sorted_idx, sorted_md, sorted_sc = sort_by_char_class([], [])
        assert sorted_idx == []

    def test_sort_missing_class_id_goes_last(self):
        """Metadata without char_class_id should default to 999 and sort last."""
        md = [
            {'char': 'א', 'char_class_id': 0},
            {'char': '?'},
        ]
        scores = [0.5, 0.5]
        _, sorted_md, _ = sort_by_char_class(md, scores)
        assert sorted_md[0]['char'] == 'א'
        assert sorted_md[1]['char'] == '?'


# ---------------------------------------------------------------------------
# GlyphTokenEnrichment with char_class_ids
# ---------------------------------------------------------------------------

class TestGlyphTokenEnrichmentCharClass:
    def test_enrichment_with_char_class_ids(self):
        B, M = 2, 16
        enrichment = GlyphTokenEnrichment(d_model=D_MODEL)
        features = torch.rand(B, M, D_MODEL)
        coords = torch.rand(B, M, 4)
        valid_mask = torch.ones(B, M, dtype=torch.bool)
        char_class_ids = torch.randint(0, NUM_GLYPH_CLASSES, (B, M))

        result = enrichment(features, coords, valid_mask, char_class_ids=char_class_ids)
        assert result.shape == (B, M, D_MODEL)

    def test_enrichment_without_char_class_ids(self):
        """Backward compat: should work when char_class_ids is None."""
        B, M = 2, 8
        enrichment = GlyphTokenEnrichment(d_model=D_MODEL)
        features = torch.rand(B, M, D_MODEL)
        coords = torch.rand(B, M, 4)
        valid_mask = torch.ones(B, M, dtype=torch.bool)

        result = enrichment(features, coords, valid_mask, char_class_ids=None)
        assert result.shape == (B, M, D_MODEL)

    def test_char_class_embedding_affects_output(self):
        """Different char_class_ids should produce different enriched tokens."""
        B, M = 1, 4
        enrichment = GlyphTokenEnrichment(d_model=D_MODEL)
        features = torch.rand(B, M, D_MODEL)
        coords = torch.rand(B, M, 4)
        valid_mask = torch.ones(B, M, dtype=torch.bool)

        ids_a = torch.zeros(B, M, dtype=torch.long)
        ids_b = torch.ones(B, M, dtype=torch.long)
        result_a = enrichment(features, coords, valid_mask, char_class_ids=ids_a)
        result_b = enrichment(features, coords, valid_mask, char_class_ids=ids_b)
        assert not torch.allclose(result_a, result_b), \
            "Different class IDs should yield different token embeddings"


# ---------------------------------------------------------------------------
# GlyphBranch end-to-end with char_class_ids
# ---------------------------------------------------------------------------

class TestGlyphBranchCharClass:
    def test_branch_accepts_char_class_ids(self):
        B, M = 2, 16
        H, W = CHAR_PATCH_SIZE, CHAR_PATCH_SIZE
        branch = GlyphBranch(
            char_patch_size=CHAR_PATCH_SIZE,
            d_model=D_MODEL,
            num_summary_tokens=GLYPH_NUM_SUMMARY_TOKENS,
        )
        patches = torch.rand(B, M, 3, H, W)
        coords = torch.rand(B, M, 4)
        valid_mask = torch.ones(B, M, dtype=torch.bool)
        char_class_ids = torch.randint(0, NUM_GLYPH_CLASSES, (B, M))

        tokens, token_valid = branch(patches, coords, valid_mask=valid_mask, char_class_ids=char_class_ids)
        expected_count = (
            GLYPH_NUM_SUMMARY_TOKENS
            if USE_BRANCH_ADAPTERS_AND_SUMMARIZERS
            else M
        )
        assert tokens.shape == (B, expected_count, D_MODEL)
        assert token_valid.shape == (B, expected_count)
        assert token_valid.all()

    def test_branch_without_char_class_ids(self):
        B, M = 2, 8
        branch = GlyphBranch(
            char_patch_size=CHAR_PATCH_SIZE,
            d_model=D_MODEL,
            num_summary_tokens=GLYPH_NUM_SUMMARY_TOKENS,
        )
        patches = torch.rand(B, M, 3, CHAR_PATCH_SIZE, CHAR_PATCH_SIZE)
        coords = torch.rand(B, M, 4)
        valid_mask = torch.ones(B, M, dtype=torch.bool)

        tokens, token_valid = branch(patches, coords, valid_mask=valid_mask)
        expected_count = (
            GLYPH_NUM_SUMMARY_TOKENS
            if USE_BRANCH_ADAPTERS_AND_SUMMARIZERS
            else M
        )
        assert tokens.shape == (B, expected_count, D_MODEL)
        assert token_valid.shape == (B, expected_count)
        assert token_valid.all()


# ---------------------------------------------------------------------------
# Heatmap alphabetical reordering (numpy-level)
# ---------------------------------------------------------------------------

class TestHeatmapAlphabeticalReorder:
    def test_square_matrix_reorder(self):
        """Reordering rows + columns of a square matrix by char_class_id
        should group same-letter entries together."""
        n = 6
        attn = np.arange(n * n, dtype=float).reshape(n, n)
        md = [
            {'char': 'ג', 'char_class_id': 2},
            {'char': 'א', 'char_class_id': 0},
            {'char': 'ג', 'char_class_id': 2},
            {'char': 'ב', 'char_class_id': 1},
            {'char': 'א', 'char_class_id': 0},
            {'char': 'ב', 'char_class_id': 1},
        ]
        sort_keys = [(m['char_class_id'], i) for i, m in enumerate(md)]
        sort_order = [i for _, i in sorted(sort_keys)]
        reordered = attn[np.ix_(sort_order, sort_order)]
        labels = [md[i]['char'] for i in sort_order]

        assert labels == ['א', 'א', 'ב', 'ב', 'ג', 'ג']
        assert reordered.shape == (n, n)
        assert reordered[0, 0] == attn[1, 1]  # (א row 1) x (א col 1)

    def test_column_only_reorder(self):
        """For the glyph summarizer heatmap [Q, M], only columns are reordered."""
        Q, M = 4, 6
        attn = np.arange(Q * M, dtype=float).reshape(Q, M)
        md = [
            {'char': 'ב', 'char_class_id': 1},
            {'char': 'א', 'char_class_id': 0},
            {'char': 'ג', 'char_class_id': 2},
            {'char': 'א', 'char_class_id': 0},
            {'char': 'ב', 'char_class_id': 1},
            {'char': 'ג', 'char_class_id': 2},
        ]
        sort_keys = [(m['char_class_id'], i) for i, m in enumerate(md)]
        sort_order = [i for _, i in sorted(sort_keys)]
        reordered = attn[:, sort_order]
        labels = [md[i]['char'] for i in sort_order]

        assert labels == ['א', 'א', 'ב', 'ב', 'ג', 'ג']
        assert reordered.shape == (Q, M)
        assert reordered[0, 0] == attn[0, 1]  # First query → original col 1 (first א)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
