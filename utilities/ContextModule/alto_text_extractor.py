"""
Utilities for extracting textual content from ALTO XML files.

We reuse the rich parsing utilities from ``utilities.VisionModule.alto_parser``
to obtain ordered words/strings per text line and convert them into plain text
suitable for language models.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from utilities.VisionModule.alto_parser import parse_alto_strings, StringInfo


@dataclass
class TextLine:
    """Simple container representing an extracted text line."""

    text: str
    words: List[str]
    confidence: Optional[float]
    polygon: Optional[List[Tuple[float, float]]]


def _string_confidence(s: StringInfo) -> Optional[float]:
    """Return per-string confidence if available, otherwise None."""
    return s.wc


def extract_text_lines(
    alto_xml_path: str,
    min_wc: float = 0.0,
    min_chars: int = 1,
    strip_empty: bool = True,
) -> List[TextLine]:
    """
    Extract ordered text lines from an ALTO XML file.

    Args:
        alto_xml_path: path to ALTO XML file.
        min_wc: minimum word confidence (0.0-1.0) to include; ignored if WC missing.
        min_chars: drop lines whose concatenated length is below this threshold.
        strip_empty: remove empty strings while forming lines.

    Returns:
        List of TextLine objects ordered by their vertical position.
    """
    if not Path(alto_xml_path).exists():
        raise FileNotFoundError(alto_xml_path)

    strings = parse_alto_strings(alto_xml_path)
    if not strings:
        return []

    # Group strings by their parent text line (fall back to vpos bucket)
    grouped: Dict[Tuple[int, float], List[StringInfo]] = defaultdict(list)
    for s in strings:
        if not s.content:
            continue
        wc = _string_confidence(s)
        if wc is not None and wc < min_wc:
            continue

        # Use textline identity when present, else bucket by approximate row (vpos)
        if s.textline is not None:
            key = (id(s.textline), float(s.vpos or 0.0))
        else:
            key = (-1, float(s.vpos or 0.0))
        grouped[key].append(s)

    # Sort groups by their vertical position (vpos)
    ordered_lines: List[Tuple[float, List[StringInfo]]] = []
    for key, slice_strings in grouped.items():
        _, vpos = key
        ordered_lines.append((vpos, slice_strings))
    ordered_lines.sort(key=lambda item: item[0])

    result: List[TextLine] = []
    for _, slice_strings in ordered_lines:
        # Sort tokens within line by horizontal position
        slice_strings.sort(key=lambda s: (s.hpos or 0.0))
        words = [s.content.strip() for s in slice_strings if s.content]
        if strip_empty:
            words = [w for w in words if w]
        line_text = " ".join(words).strip()
        if len(line_text) < min_chars:
            continue

        confidences = [c for c in (_string_confidence(s) for s in slice_strings) if c is not None]
        avg_conf = sum(confidences) / len(confidences) if confidences else None
        polygon = slice_strings[0].textline.polygon if slice_strings[0].textline else None

        result.append(TextLine(text=line_text, words=words, confidence=avg_conf, polygon=polygon))

    return result


def extract_text_blocks(
    alto_xml_path: str,
    min_wc: float = 0.0,
    min_chars: int = 1,
    join_with: str = "\n",
) -> List[str]:
    """
    Convenience helper that returns text blocks (strings) ready for the text encoder.

    Args:
        alto_xml_path: path to ALTO XML file.
        min_wc: minimum word confidence filter.
        min_chars: drop lines shorter than this many characters.
        join_with: separator when combining contiguous lines into a block.

    Returns:
        List of strings (one per text block / page).
    """
    lines = extract_text_lines(alto_xml_path, min_wc=min_wc, min_chars=min_chars)
    if not lines:
        return []

    # For now simply join all lines into one block per page; future work could split by paragraph.
    all_text = join_with.join(line.text for line in lines if line.text)
    return [all_text] if all_text else []

