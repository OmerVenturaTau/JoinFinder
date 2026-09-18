"""
Extract words from ALTO XML files with Hebrew dictionary validation.

Moved from `utilities/xml_word_extraction.py` into `utilities/ContextModule/`
to keep text-related utilities together.
"""

from __future__ import annotations

import os
import sys
import unicodedata
import xml.etree.ElementTree as ET
from collections import OrderedDict
from typing import List, Dict, Optional, Tuple
import logging

# Set up logger
logger = logging.getLogger(__name__)

# Add project root to path (this file is utilities/ContextModule/*)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from system import (
    OCR_STRING_CONFIDENCE_THRESHOLD,
    USE_HEBREW_DICT_CHECK,
    XML_PATCH_READING_DIRECTION_RTL,
)
from utilities.VisionModule.alto_parser import (
    parse_alto_strings,
    StringInfo,
    HebrewDictChecker,
    _is_hebrew,
)
from utilities.VisionModule.xml_patch_extraction import detect_page_layout


def normalize_ocr_word(value: object) -> str:
    """Remove invisible controls and punctuation surrounding an OCR word."""
    text = "".join(
        char for char in str(value or "").strip()
        if unicodedata.category(char) != "Cf"
    )
    while text and unicodedata.category(text[0]).startswith("P"):
        text = text[1:]
    while text and unicodedata.category(text[-1]).startswith("P"):
        text = text[:-1]
    return text.strip()


def _line_group_key(meta: Dict, fallback_index: int = 0) -> tuple:
    """Return a stable line key, including the page side when available."""
    segment = int(meta.get("page_segment", 0) or 0)
    line_index = meta.get("line_idx")
    if line_index is not None:
        try:
            return segment, "index", int(line_index)
        except (TypeError, ValueError):
            pass
    line_id = meta.get("line_id")
    if line_id is not None and str(line_id).strip():
        return segment, "id", str(line_id)
    try:
        y_position = float(meta.get("vpos", meta.get("center_y", fallback_index)) or 0.0)
    except (TypeError, ValueError):
        y_position = float(fallback_index)
    # ALTO words on one physical line normally share VPOS. Rounding is more
    # faithful here than putting every missing-line-id word into one giant line.
    return segment, "y", int(round(y_position))


def _select_words_line_round_robin(
    words: List[str],
    metadata: List[Dict],
    *,
    max_words: int,
    max_words_per_line: int,
) -> Tuple[List[str], List[Dict]]:
    """Balanced line sampler retained for diagnostics and small-data callers."""
    if len(words) != len(metadata):
        raise ValueError("words and metadata must have equal lengths")
    if max_words <= 0 or max_words_per_line <= 0:
        return [], []

    grouped: OrderedDict[tuple, List[int]] = OrderedDict()
    for index, meta in enumerate(metadata):
        grouped.setdefault(_line_group_key(meta, index), []).append(index)
    queues = [indices[:max_words_per_line] for indices in grouped.values()]

    selected: List[int] = []
    depth = 0
    while len(selected) < max_words:
        added = False
        for indices in queues:
            if depth < len(indices):
                selected.append(indices[depth])
                added = True
                if len(selected) == max_words:
                    break
        if not added:
            break
        depth += 1

    selected.sort()
    return [words[index] for index in selected], [metadata[index] for index in selected]


def _select_complete_lines(
    words: List[str],
    metadata: List[Dict],
    *,
    max_words: int,
) -> Tuple[List[str], List[Dict]]:
    """Select high-confidence complete lines, then restore reading order.

    The former top-confidence *word* truncation left most retained ALTO lines
    with only one or two words. AlephBERT is invoked once per line, so that both
    destroyed its sentence context and multiplied encoder calls. This selector
    treats a line as the atomic unit. Only a single over-budget line may be
    truncated, as an unavoidable fallback.
    """
    if len(words) != len(metadata):
        raise ValueError("words and metadata must have equal lengths")
    if max_words <= 0:
        return [], []
    if len(words) <= max_words:
        return list(words), list(metadata)

    grouped: OrderedDict[tuple, List[int]] = OrderedDict()
    for index, meta in enumerate(metadata):
        grouped.setdefault(_line_group_key(meta, index), []).append(index)

    candidates = []
    for reading_index, indices in enumerate(grouped.values()):
        confidences = []
        for index in indices:
            try:
                confidences.append(float(metadata[index].get("wc", 0.0) or 0.0))
            except (TypeError, ValueError):
                confidences.append(0.0)
        mean_confidence = sum(confidences) / max(1, len(confidences))
        candidates.append((-mean_confidence, reading_index, indices))

    selected: List[int] = []
    remaining = int(max_words)
    for _negative_confidence, _reading_index, indices in sorted(candidates):
        if len(indices) <= remaining:
            selected.extend(indices)
            remaining -= len(indices)
        if remaining == 0:
            break

    if not selected:
        # Every line is longer than the entire budget. Preserve one coherent
        # line prefix rather than reverting to unrelated high-confidence words.
        best_indices = sorted(candidates)[0][2]
        selected.extend(best_indices[:max_words])

    selected.sort()
    return [words[index] for index in selected], [metadata[index] for index in selected]


def _get_page_dimensions(alto_xml_path: str) -> Tuple[float, float]:
    """Extract page width/height from ALTO XML."""
    try:
        tree = ET.parse(alto_xml_path)
        root = tree.getroot()
    except Exception:
        logger.exception("Failed to parse page dimensions from ALTO XML")
        return 0.0, 0.0

    for page_elem in root.iter():
        tag = page_elem.tag.split('}', 1)[-1] if '}' in page_elem.tag else page_elem.tag
        if tag == 'Page':
            width = float(page_elem.attrib.get('WIDTH', 0) or 0)
            height = float(page_elem.attrib.get('HEIGHT', 0) or 0)
            return width, height
    return 0.0, 0.0


def extract_words_from_alto(
    alto_xml_path: str,
    string_conf_threshold: float = OCR_STRING_CONFIDENCE_THRESHOLD,
    use_hebrew_dict: bool = True,
    max_words: Optional[int] = None,
    legacy_20260824: bool = False,
) -> Tuple[List[str], List[Dict]]:
    """
    Extract words from ALTO XML file with Hebrew dictionary validation.

    Uses parse_alto_strings with the same filtering logic as alto_visualize:
    - Filters by string confidence threshold
    - Filters by Hebrew dictionary (if enabled)
    - Only processes Hebrew strings
    - Skips single-character strings

    Returns:
      (words, metadata_list)
    """
    if not os.path.exists(alto_xml_path):
        return [], []

    strings: List[StringInfo] = parse_alto_strings(alto_xml_path)
    page_width, page_height = _get_page_dimensions(alto_xml_path)

    # Initialize Hebrew dictionary checker if needed
    dict_checker = None
    if use_hebrew_dict and USE_HEBREW_DICT_CHECK:
        try:
            dict_checker = HebrewDictChecker()
            if dict_checker is not None and not dict_checker.is_available():
                logger.warning(
                    "Hebrew dictionary checker initialized but dictionary module not available. "
                    "Dictionary filtering disabled."
                )
        except Exception as e:
            logger.exception("Could not initialize Hebrew dictionary checker")
            print(f"Warning: Could not initialize Hebrew dictionary checker: {e}")
            dict_checker = None

    entries: List[Tuple[str, Dict]] = []

    for s in strings:
        word = s.content if legacy_20260824 else normalize_ocr_word(s.content)
        if len(word.strip()) <= 1:
            continue
        if not _is_hebrew(word):
            continue
        if s.wc is None or s.wc < string_conf_threshold:
            continue
        if use_hebrew_dict and dict_checker is not None:
            if not dict_checker.check_word(word):
                continue

        meta_entry: Dict[str, Optional[float] | str | int] = {
            'word': word,
            'wc': s.wc,
            'hpos': s.hpos,
            'vpos': s.vpos,
            'width': s.width,
            'height': s.height,
            'line_id': getattr(s.textline, "line_id", None),
        }
        entries.append((word, meta_entry))

    if legacy_20260824 and max_words is not None and len(entries) > max_words:
        entries.sort(key=lambda item: item[1].get("wc", 0.0) or 0.0, reverse=True)
        entries = entries[:max_words]

    words = [w for w, _ in entries]
    metadata_list = [m for _, m in entries]

    # Derive geometry and page layout from every eligible word. Doing this
    # before the word budget is applied prevents a confidence-skewed subset
    # from producing the wrong one-page/two-page decision.
    text_regions = []
    if metadata_list:
        for meta in metadata_list:
            hpos = meta.get('hpos')
            vpos = meta.get('vpos')
            width = meta.get('width')
            height = meta.get('height')

            center_x = None
            center_y = None
            if hpos is not None and width is not None:
                center_x = hpos + width / 2.0
                meta['center_x'] = center_x
            if vpos is not None and height is not None:
                center_y = vpos + height / 2.0
                meta['center_y'] = center_y

            if center_x is not None and center_y is not None:
                text_regions.append({'center_x': center_x, 'center_y': center_y})

            if center_x is not None and page_width > 0:
                meta['normalized_center_x'] = max(0.0, min(1.0, center_x / page_width))
            if center_y is not None and page_height > 0:
                meta['normalized_center_y'] = max(0.0, min(1.0, center_y / page_height))

    # Determine page segmentation (single vs two-page spread)
    if text_regions and page_width > 0 and page_height > 0:
        is_two_page, split_x = detect_page_layout(text_regions, page_width, page_height)
    else:
        is_two_page, split_x = (False, None)

    for meta in metadata_list:
        center_x = meta.get('center_x')
        page_segment = 0
        if is_two_page and split_x is not None and center_x is not None:
            is_right = center_x >= split_x
            if XML_PATCH_READING_DIRECTION_RTL:
                page_segment = 0 if is_right else 1
            else:
                page_segment = 0 if not is_right else 1
        meta['page_segment'] = page_segment
        meta['page_split_x'] = split_x
        meta['is_two_page'] = is_two_page

    if page_width > 0 and page_height > 0:
        for meta in metadata_list:
            w = meta.get("width")
            h = meta.get("height")
            if w is not None:
                meta["normalized_width"] = max(0.0, min(1.0, float(w) / float(page_width)))
            if h is not None:
                meta["normalized_height"] = max(0.0, min(1.0, float(h) / float(page_height)))

    # Sort by reading order with line grouping (same logic as original module)
    if len(words) > 1 and metadata_list:
        line_map: Dict[object, List[Tuple[str, Dict]]] = {}
        for w, m in zip(words, metadata_list):
            line_key_value = (
                str(m.get("line_id") or "__no_line__")
                if legacy_20260824
                else _line_group_key(m)
            )
            line_map.setdefault(line_key_value, []).append((w, m))

        def sort_key_word_rtl(pair):
            _, meta = pair
            cx = meta.get("center_x", 0.0) or 0.0
            return -cx

        def sort_key_word_ltr(pair):
            _, meta = pair
            cx = meta.get("center_x", 0.0) or 0.0
            return cx

        sort_word_key = sort_key_word_rtl if XML_PATCH_READING_DIRECTION_RTL else sort_key_word_ltr
        for _, lst in line_map.items():
            lst.sort(key=sort_word_key)

        def line_key(item):
            _, lst = item
            ys = [(m.get("center_y", 0.0) or 0.0) for _, m in lst]
            line_y = sum(ys) / max(1, len(ys))
            segs = [(m.get("page_segment", 0) or 0) for _, m in lst]
            seg = int(round(sum(segs) / max(1, len(segs))))
            return (seg, line_y)

        sorted_lines = sorted(line_map.items(), key=line_key)

        flat_pairs: List[Tuple[str, Dict]] = []
        for line_idx, (_, lst) in enumerate(sorted_lines):
            for word_in_line_idx, (w, m) in enumerate(lst):
                m["line_idx"] = int(line_idx)
                m["word_in_line_idx"] = int(word_in_line_idx)
                flat_pairs.append((w, m))

        words = [w for w, _ in flat_pairs]
        metadata_list = [m for _, m in flat_pairs]

    if not legacy_20260824 and max_words is not None and len(words) > max_words:
        words, metadata_list = _select_complete_lines(
            words,
            metadata_list,
            max_words=int(max_words),
        )

    return words, metadata_list
