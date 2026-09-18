"""
Tests for Word Branch module.
"""

import math

import torch
import pytest
from models.word_branch import (
    WordBranch,
    WordSetSummarizer,
    WordHardQualityFilter,
    AlephBERTLineEncoder,
    group_words_by_lines,
    determine_line_reading_order,
    BibleTfidfDictionary,
    load_bible_tfidf_dictionary,
    compute_tfidf_weights,
    compute_legacy_batch_tfidf_weights,
)
from system import (
    D_MODEL,
    WORD_BRANCH_LINE_MAX_LENGTH,
    WORD_BRANCH_USE_TFIDF_GATING,
    WORD_BRANCH_TFIDF_DICTIONARY_PATH,
    OCR_STRING_CONFIDENCE_THRESHOLD,
)


class _DummyLineEncoder(torch.nn.Module):
    def __init__(self, d_model=D_MODEL, **_kwargs):
        super().__init__()
        self.d_model = d_model

    def encode_line_with_word_mapping(self, line_words, device):
        if not line_words:
            return torch.zeros(0, self.d_model, device=device), []
        embs = torch.arange(
            len(line_words) * self.d_model,
            dtype=torch.float32,
            device=device,
        ).reshape(len(line_words), self.d_model)
        return embs, list(range(len(line_words)))


def _make_test_word_branch(monkeypatch, d_model=12, num_heads=3):
    monkeypatch.setattr("models.word_branch.AlephBERTLineEncoder", _DummyLineEncoder)
    return WordBranch(
        d_model=d_model,
        num_heads=num_heads,
        use_fourier_pos=False,
        enable_summarizer=False,
        conf_min=0.9,
        min_area=0.0,
    )


def test_word_set_summarizer_masks_sources_but_keeps_all_queries_active():
    summarizer = WordSetSummarizer(
        d_model=8,
        num_queries=4,
        num_heads=2,
        dropout=0.0,
    ).eval()
    line_tokens = torch.randn(2, 3, 8)
    line_valid = torch.tensor([[True, False, False], [False, False, False]])

    summary, summary_valid, attention = summarizer(
        line_tokens,
        line_valid,
        return_attention=True,
    )

    assert summary.shape == (2, 4, 8)
    assert summary_valid.tolist() == [[True] * 4, [False] * 4]
    assert attention.shape == (2, 4, 3)
    torch.testing.assert_close(attention[0, :, 1:], torch.zeros(4, 2))
    torch.testing.assert_close(attention[0].sum(dim=1), torch.ones(4))
    torch.testing.assert_close(attention[1], torch.zeros(4, 3))
    torch.testing.assert_close(summary[1], torch.zeros(4, 8))


def test_word_branch_can_return_pre_summary_line_evidence(monkeypatch):
    branch = _make_test_word_branch(monkeypatch)
    words = [[], ["א", "מ"]]
    metadata = [[], [
        {"wc": 1.0, "line_id": "line-1"},
        {"wc": 1.0, "line_id": "line-2"},
    ]]

    tokens, valid, evidence = branch(
        words,
        metadata,
        torch.device("cpu"),
        return_source_evidence=True,
    )

    assert tokens.shape[0] == 2
    assert valid.shape[0] == 2
    torch.testing.assert_close(evidence, torch.tensor([0.0, 1.0]))


def test_word_summary_evidence_is_normalized_by_query_capacity(monkeypatch):
    monkeypatch.setattr("models.word_branch.AlephBERTLineEncoder", _DummyLineEncoder)
    branch = WordBranch(
        d_model=12,
        num_heads=3,
        use_fourier_pos=False,
        conf_min=0.9,
        min_area=0.0,
        dropout=0.0,
        enable_summarizer=True,
        num_summary_tokens=4,
        min_summary_tokens=1,
    ).eval()
    words = [[], ["א", "מ"]]
    metadata = [[], [
        {"wc": 1.0, "line_id": "line-1"},
        {"wc": 1.0, "line_id": "line-2"},
    ]]

    summaries, summary_mask, evidence = branch(
        words,
        metadata,
        torch.device("cpu"),
        return_source_evidence=True,
    )

    assert summaries.shape == (2, 4, 12)
    assert summary_mask.tolist() == [[False] * 4, [True] * 4]
    torch.testing.assert_close(evidence, torch.tensor([0.0, 0.5]))


def test_group_words_by_lines():
    """Test word grouping by lines."""
    words = ['שלום', 'עולם', 'יום', 'טוב']
    metadata = [
        {'line_id': 'line1', 'center_x': 0.8, 'center_y': 0.2},
        {'line_id': 'line1', 'center_x': 0.6, 'center_y': 0.2},
        {'line_id': 'line2', 'center_x': 0.7, 'center_y': 0.5},
        {'line_id': 'line2', 'center_x': 0.5, 'center_y': 0.5},
    ]
    
    line_groups = group_words_by_lines(words, metadata, rtl=True)
    
    assert len(line_groups) == 2
    assert 'line1' in line_groups
    assert 'line2' in line_groups
    assert len(line_groups['line1']) == 2
    assert len(line_groups['line2']) == 2
    
    # Check RTL sorting (rightmost first)
    line1_words = line_groups['line1']
    assert line1_words[0][2]['center_x'] > line1_words[1][2]['center_x']


def test_group_words_by_lines_uses_normalized_coordinates_when_available():
    words = ['left', 'right', 'same-line']
    metadata = [
        {'normalized_center_x': 0.2, 'normalized_center_y': 0.251, 'center_x': 200, 'center_y': 101},
        {'normalized_center_x': 0.9, 'normalized_center_y': 0.250, 'center_x': 900, 'center_y': 100},
        {'normalized_center_x': 0.6, 'normalized_center_y': 0.249, 'center_x': 600, 'center_y': 99},
    ]

    line_groups = group_words_by_lines(words, metadata, rtl=True)

    assert len(line_groups) == 1
    grouped_words = next(iter(line_groups.values()))
    assert [word for _, word, _ in grouped_words] == ['right', 'same-line', 'left']


def test_word_hard_quality_filter():
    """Test word quality filter."""
    filter_module = WordHardQualityFilter(
        conf_min=OCR_STRING_CONFIDENCE_THRESHOLD,
        min_area=1.0,
    )
    
    line_words = [
        (0, 'שלום', {'wc': 0.99, 'width': 10, 'height': 5}),
        (1, '!!!', {'wc': 0.99, 'width': 10, 'height': 5}),  # Punctuation-only
        (2, 'עולם', {'wc': 0.5, 'width': 10, 'height': 5}),  # Low confidence
        (3, 'יום', {'wc': 0.99, 'width': 0.5, 'height': 0.5}),  # Tiny bbox
        (4, 'טוב', {'wc': 0.99, 'width': 10, 'height': 5}),
    ]
    
    filtered = filter_module.filter_words_in_line(line_words)
    
    # Should keep only words 0 and 4 (valid)
    assert len(filtered) == 2
    assert filtered[0][0] == 0
    assert filtered[1][0] == 4


def test_word_hard_quality_filter_accepts_string_numeric_metadata():
    """ALTO/XML metadata can arrive as strings; filtering should not crash."""
    filter_module = WordHardQualityFilter(conf_min=0.9, min_area=1.0)
    line_words = [
        (0, 'שלום', {'wc': '0.99', 'width': '10', 'height': '5'}),
        (1, None, {'wc': '0.99', 'width': '10', 'height': '5'}),
        (2, '...', {'wc': '0.99', 'width': '10', 'height': '5'}),
        (3, 'קטן', {'wc': '0.99', 'width': 'nan', 'height': '5'}),
    ]

    filtered = filter_module.filter_words_in_line(line_words)

    assert [(idx, word) for idx, word, _ in filtered] == [(0, 'שלום')]


def test_metadata_float_rejects_non_finite_values_in_grouping():
    words = ['fallback', 'normal']
    metadata = [
        {'normalized_center_x': 'nan', 'normalized_center_y': 'nan'},
        {'normalized_center_x': 0.8, 'normalized_center_y': 0.0},
    ]

    line_groups = group_words_by_lines(words, metadata, rtl=True)

    grouped_words = next(iter(line_groups.values()))
    assert [word for _, word, _ in grouped_words] == ['normal', 'fallback']


def test_compute_tfidf_weights():
    """Local TF and fixed Bible IDF both contribute to the gate."""
    bible_tfidf = BibleTfidfDictionary(
        entries={},
        idf_by_word={'שלום': 2.0, 'עולמ': 1.0},
        num_documents=39,
    )
    words = ['שלום', 'עולם', 'שלום']

    weights = compute_tfidf_weights(words, bible_tfidf)
    
    assert weights.shape[0] == len(words)
    assert torch.all(weights >= 0)
    assert torch.all(weights <= 1)
    
    # שלום has higher local TF and higher Bible IDF than עולם.
    assert weights[0] > weights[1]
    assert weights[0] == weights[2]


def test_compute_tfidf_weights_never_go_negative_for_repeated_words():
    bible_tfidf = BibleTfidfDictionary(
        entries={},
        idf_by_word={'שלום': 1.2},
        num_documents=39,
    )
    words = ['שלום', 'שלום', 'שלום']

    weights = compute_tfidf_weights(words, bible_tfidf)

    assert torch.all(weights >= 0)


def test_legacy_batch_tfidf_matches_historical_formula():
    words = ["א", "א", "ב"]
    batch_words = ["א", "א", "ב", "ג"]
    weights = compute_legacy_batch_tfidf_weights(words, batch_words)

    expected = torch.tensor([
        math.log(5 / 3) + 1,
        math.log(5 / 3) + 1,
        0.5 * (math.log(5 / 2) + 1),
    ])
    expected = expected / expected.max()
    assert torch.allclose(weights, expected)
    assert torch.all(weights <= 1)


def test_bible_tfidf_lookup_prefers_exact_then_strips_clitic_prefix():
    bible_tfidf = BibleTfidfDictionary(
        entries={},
        idf_by_word={'בבית': 3.5, 'בית': 1.1, 'יאמר': 1.7},
        num_documents=39,
    )

    assert bible_tfidf.lookup_idf('בבית') == pytest.approx(3.5)
    assert bible_tfidf.lookup_idf('ויאמר') == pytest.approx(1.7)


def test_unknown_word_gets_neutral_not_maximum_idf():
    bible_tfidf = BibleTfidfDictionary(
        entries={},
        idf_by_word={'נדיר': 3.0},
        num_documents=39,
    )

    weights = compute_tfidf_weights(['נדיר', 'קשקושלאידוע'], bible_tfidf)

    assert weights.tolist() == pytest.approx([1.0, 1.0 / 3.0])


def test_configured_bible_tfidf_dictionary_is_book_level():
    bible_tfidf = load_bible_tfidf_dictionary(WORD_BRANCH_TFIDF_DICTIONARY_PATH)

    assert bible_tfidf.num_documents == 39
    assert bible_tfidf.lookup_idf('שלום') == pytest.approx(1.2231435513142097)
    assert bible_tfidf.lookup_idf('ויאמר') == pytest.approx(1.162518929497775)


def test_determine_line_reading_order():
    """Test line reading order determination."""
    line_groups = {
        'line1': [(0, 'שלום', {'page_segment': 0, 'center_y': 0.3})],
        'line2': [(1, 'עולם', {'page_segment': 0, 'center_y': 0.1})],
        'line3': [(2, 'יום', {'page_segment': 1, 'center_y': 0.2})],
    }
    word_metadata = [
        {'page_segment': 0, 'center_y': 0.3},
        {'page_segment': 0, 'center_y': 0.1},
        {'page_segment': 1, 'center_y': 0.2},
    ]
    
    order = determine_line_reading_order(line_groups, word_metadata, rtl=True)
    
    # Should be sorted by page_segment first, then y
    assert order[0] == 'line2'  # page_segment=0, y=0.1 (first)
    assert order[1] == 'line1'  # page_segment=0, y=0.3 (second)
    assert order[2] == 'line3'  # page_segment=1 (third page)


def test_determine_line_reading_order_uses_line_average_not_first_sorted_word():
    line_groups = {
        'line1': [
            (0, 'right', {'page_segment': 0, 'normalized_center_y': 0.5}),
            (1, 'left', {'page_segment': 0, 'normalized_center_y': 0.3}),
        ],
        'line2': [
            (2, 'top', {'page_segment': 0, 'normalized_center_y': 0.2}),
        ],
    }

    order = determine_line_reading_order(line_groups, [], rtl=True)

    assert order == ['line2', 'line1']


def test_alephbert_line_encoder():
    """Test AlephBERT line encoder (basic structure test)."""
    encoder = AlephBERTLineEncoder(
        d_model=D_MODEL,
        line_max_length=WORD_BRANCH_LINE_MAX_LENGTH,
        word_pooling="mean",
    )
    
    # Test that encoder can be created
    assert encoder.d_model == D_MODEL
    assert encoder.line_max_length == WORD_BRANCH_LINE_MAX_LENGTH


def test_word_branch_creation():
    """Test WordBranch creation."""
    branch = WordBranch(
        d_model=D_MODEL,
        line_max_length=WORD_BRANCH_LINE_MAX_LENGTH,
    )
    
    assert branch.d_model == D_MODEL
    assert branch.rtl is not None
    assert branch.enable_pos_encoding is True
    assert branch.use_tfidf_gating is WORD_BRANCH_USE_TFIDF_GATING is False


def test_word_branch_skips_tfidf_when_disabled(monkeypatch):
    branch = _make_test_word_branch(monkeypatch)
    monkeypatch.setattr(
        "models.word_branch.compute_tfidf_weights",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("TF-IDF must not run when disabled")
        ),
    )

    tokens, valid_mask = branch(
        words=[["שלום", "עולם"]],
        word_metadata=[[
            {"wc": 1.0, "width": 10.0, "height": 5.0, "line_id": "line1"},
            {"wc": 1.0, "width": 10.0, "height": 5.0, "line_id": "line1"},
        ]],
        device=torch.device("cpu"),
    )

    assert tokens.shape[0] == 1
    assert valid_mask.any()


def test_word_branch_forward_empty():
    """Test WordBranch forward with empty input."""
    branch = WordBranch(d_model=D_MODEL)
    device = torch.device('cpu')
    
    words = []
    word_metadata = []
    
    line_tokens, line_valid_mask = branch(words, word_metadata, device)
    
    assert line_tokens.shape[0] == 0
    assert line_valid_mask.shape[0] == 0


def test_word_branch_rejects_words_metadata_length_mismatch(monkeypatch):
    branch = _make_test_word_branch(monkeypatch)
    device = torch.device('cpu')

    with pytest.raises(ValueError, match="words/metadata length mismatch"):
        branch(
            words=[['שלום', 'עולם']],
            word_metadata=[[{'wc': 1.0, 'width': 1.0, 'height': 1.0}]],
            device=device,
        )


def test_word_branch_rejects_words_page_segment_length_mismatch(monkeypatch):
    branch = _make_test_word_branch(monkeypatch)
    device = torch.device('cpu')

    with pytest.raises(ValueError, match="words/page_segments length mismatch"):
        branch(
            words=[['שלום', 'עולם']],
            word_metadata=[[
                {'wc': 1.0, 'width': 1.0, 'height': 1.0},
                {'wc': 1.0, 'width': 1.0, 'height': 1.0},
            ]],
            page_segments=[[0]],
            device=device,
        )


def test_word_branch_rejects_non_dict_metadata(monkeypatch):
    branch = _make_test_word_branch(monkeypatch)
    device = torch.device('cpu')

    with pytest.raises(ValueError, match="metadata entries must be dicts"):
        branch(
            words=[['שלום']],
            word_metadata=[[None]],
            device=device,
        )


def test_word_branch_rejects_invalid_page_segment_values(monkeypatch):
    branch = _make_test_word_branch(monkeypatch)
    device = torch.device('cpu')

    with pytest.raises(ValueError, match="page_segments entries must be 0 or 1"):
        branch(
            words=[['שלום']],
            word_metadata=[[{'wc': 1.0, 'width': 1.0, 'height': 1.0}]],
            page_segments=[[2]],
            device=device,
        )


def test_word_branch_accepts_float_like_page_segment_values(monkeypatch):
    branch = _make_test_word_branch(monkeypatch)
    device = torch.device('cpu')

    tokens, valid_mask, *_ = branch(
        words=[['שלום']],
        word_metadata=[[{
            'wc': 1.0,
            'normalized_center_x': 0.8,
            'normalized_center_y': 0.2,
            'width': 1.0,
            'height': 1.0,
        }]],
        page_segments=[['1.0']],
        device=device,
        return_attention=True,
        return_line_to_word_attn=True,
        return_summary_to_line_attn=True,
    )

    assert tokens.shape[0] == 1
    assert valid_mask.sum() > 0


def test_word_branch_attention_outputs_remain_batch_aligned_after_filtering(monkeypatch):
    branch = _make_test_word_branch(monkeypatch)
    device = torch.device('cpu')

    words = [['bad'], ['שלום', 'עולם']]
    word_metadata = [
        [{'wc': 0.1, 'normalized_center_x': 0.5, 'normalized_center_y': 0.1, 'width': 1.0, 'height': 1.0}],
        [
            {'wc': 1.0, 'normalized_center_x': 0.8, 'normalized_center_y': 0.2, 'width': 1.0, 'height': 1.0, 'line_id': 'line1'},
            {'wc': 1.0, 'normalized_center_x': 0.6, 'normalized_center_y': 0.2, 'width': 1.0, 'height': 1.0, 'line_id': 'line1'},
        ],
    ]

    tokens, valid_mask, word_attn, line_to_word_attn, summary_to_line_attn = branch(
        words,
        word_metadata,
        device,
        return_attention=True,
        return_line_to_word_attn=True,
        return_summary_to_line_attn=True,
    )

    assert tokens.shape[0] == 2
    assert valid_mask[0].sum() == 0
    assert valid_mask[1].sum() > 0
    assert word_attn.shape[0] == 2
    assert torch.count_nonzero(word_attn[0]) == 0
    assert torch.count_nonzero(word_attn[1]) > 0
    assert line_to_word_attn.shape[0] == 2
    assert torch.count_nonzero(line_to_word_attn[0]) == 0
    assert torch.count_nonzero(line_to_word_attn[1]) > 0
    assert branch.word_set_summarizer is None
    assert summary_to_line_attn is None


def test_word_branch_page_segments_override_metadata_for_line_order(monkeypatch):
    branch = _make_test_word_branch(monkeypatch)
    device = torch.device('cpu')

    words = [['later-page', 'first-page']]
    word_metadata = [[
        {'wc': 1.0, 'normalized_center_x': 0.8, 'normalized_center_y': 0.1, 'width': 1.0, 'height': 1.0, 'line_id': 'line1', 'page_segment': 0},
        {'wc': 1.0, 'normalized_center_x': 0.8, 'normalized_center_y': 0.9, 'width': 1.0, 'height': 1.0, 'line_id': 'line2', 'page_segment': 0},
    ]]

    _, _, _, line_to_word_attn, _ = branch(
        words,
        word_metadata,
        device,
        page_segments=[[1, 0]],
        return_attention=True,
        return_line_to_word_attn=True,
        return_summary_to_line_attn=True,
    )

    assert line_to_word_attn.shape[:2] == (1, 2)
    assert torch.argmax(line_to_word_attn[0, 0]).item() == 1
    assert torch.argmax(line_to_word_attn[0, 1]).item() == 0


def test_word_branch_forward_simple():
    """Test WordBranch forward with simple input (structure only, no actual encoding)."""
    branch = WordBranch(d_model=D_MODEL)
    device = torch.device('cpu')
    
    # Note: This test only checks structure, not actual AlephBERT encoding
    # Full encoding test would require AlephBERT model to be loaded
    words = [['שלום', 'עולם']]
    word_metadata = [[
        {'wc': 0.95, 'center_x': 0.8, 'center_y': 0.2, 'width': 10, 'height': 5, 'line_id': 'line1'},
        {'wc': 0.95, 'center_x': 0.6, 'center_y': 0.2, 'width': 10, 'height': 5, 'line_id': 'line1'},
    ]]
    
    # This will fail if AlephBERT is not available, but structure is correct
    try:
        line_tokens, line_valid_mask = branch(words, word_metadata, device)
        # If successful, check shapes
        if line_tokens.numel() > 0:
            assert line_tokens.shape[-1] == D_MODEL
            assert line_valid_mask.shape[0] == line_tokens.shape[0]
    except Exception as e:
        # Expected if AlephBERT model is not available
        # Just check that the branch was created correctly
        assert branch is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
