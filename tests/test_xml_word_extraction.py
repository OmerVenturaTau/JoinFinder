from utilities.ContextModule.xml_word_extraction import (
    _select_complete_lines,
    _select_words_line_round_robin,
    extract_words_from_alto,
    normalize_ocr_word,
)


def test_normalize_ocr_word_removes_controls_and_edge_punctuation():
    assert normalize_ocr_word("\u202b(שלום!)\u202c") == "שלום"


def test_line_round_robin_enforces_four_words_per_line_even_below_global_cap():
    words = [f"l{line}-{index}" for line in range(3) for index in range(6)]
    metadata = [
        {"line_id": f"line-{line}"}
        for line in range(3)
        for _ in range(6)
    ]
    selected_words, selected_metadata = _select_words_line_round_robin(
        words, metadata, max_words=64, max_words_per_line=4
    )

    assert len(selected_words) == 12
    assert all(
        sum(meta["line_id"] == line for meta in selected_metadata) == 4
        for line in {"line-0", "line-1", "line-2"}
    )


def test_line_round_robin_balances_a_tight_global_budget():
    words = [f"l{line}-{index}" for line in range(3) for index in range(4)]
    metadata = [
        {"line_id": f"line-{line}"}
        for line in range(3)
        for _ in range(4)
    ]
    _, selected_metadata = _select_words_line_round_robin(
        words, metadata, max_words=5, max_words_per_line=4
    )
    counts = {
        line: sum(meta["line_id"] == line for meta in selected_metadata)
        for line in {"line-0", "line-1", "line-2"}
    }
    assert sorted(counts.values()) == [1, 2, 2]


def test_complete_line_selection_never_slices_a_line_that_fits():
    words = [f"l{line}-{index}" for line in range(4) for index in range(3)]
    metadata = [
        {"line_id": f"line-{line}", "wc": 0.99 - line * 0.01}
        for line in range(4)
        for _ in range(3)
    ]

    selected_words, selected_metadata = _select_complete_lines(
        words,
        metadata,
        max_words=7,
    )

    assert len(selected_words) == 6
    counts = {
        line: sum(meta["line_id"] == line for meta in selected_metadata)
        for line in {"line-0", "line-1", "line-2", "line-3"}
    }
    assert sorted(counts.values()) == [0, 0, 3, 3]


def test_complete_line_selection_uses_one_prefix_when_every_line_is_too_long():
    words = [f"l{line}-{index}" for line in range(2) for index in range(6)]
    metadata = [
        {"line_id": f"line-{line}", "wc": 0.99 - line * 0.01}
        for line in range(2)
        for _ in range(6)
    ]

    selected_words, selected_metadata = _select_complete_lines(
        words,
        metadata,
        max_words=4,
    )

    assert selected_words == ["l0-0", "l0-1", "l0-2", "l0-3"]
    assert {meta["line_id"] for meta in selected_metadata} == {"line-0"}


def test_alto_budget_keeps_complete_lines(tmp_path):
    lines = []
    for line_index, confidence in enumerate((0.99, 0.98, 0.97)):
        strings = "".join(
            f'<String CONTENT="שלום{word_index}" WC="{confidence}" '
            f'HPOS="{800 - word_index * 100}" VPOS="{100 + line_index * 100}" '
            'WIDTH="80" HEIGHT="30"><Glyph CONTENT="ש" GC="0.99" '
            'HPOS="0" VPOS="0" WIDTH="10" HEIGHT="20" /></String>'
            for word_index in range(3)
        )
        lines.append(f'<TextLine ID="line-{line_index}">{strings}</TextLine>')
    xml_path = tmp_path / "words.xml"
    xml_path.write_text(
        '<alto><Layout><Page WIDTH="1000" HEIGHT="1000"><PrintSpace>'
        + "".join(lines)
        + "</PrintSpace></Page></Layout></alto>",
        encoding="utf-8",
    )

    words, metadata = extract_words_from_alto(
        str(xml_path),
        string_conf_threshold=0.0,
        use_hebrew_dict=False,
        max_words=7,
    )

    assert len(words) == 6
    selected_lines = {meta["line_id"] for meta in metadata}
    assert selected_lines == {"line-0", "line-1"}
    assert all(
        sum(meta["line_id"] == line_id for meta in metadata) == 3
        for line_id in selected_lines
    )

    legacy_words, legacy_metadata = extract_words_from_alto(
        str(xml_path),
        string_conf_threshold=0.0,
        use_hebrew_dict=False,
        max_words=7,
        legacy_20260824=True,
    )

    assert len(legacy_words) == 7
    assert [
        sum(meta["line_id"] == line_id for meta in legacy_metadata)
        for line_id in ("line-0", "line-1", "line-2")
    ] == [3, 3, 1]
