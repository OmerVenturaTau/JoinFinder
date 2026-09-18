import argparse
import json
import math
import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import xml.etree.ElementTree as ET

# Ensure project root is on sys.path so we can import system.py when executed from subdirs
_HERE = os.path.abspath(os.path.dirname(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from system import (
    OCR_GLYPH_CONFIDENCE_THRESHOLD,
    OCR_STRING_CONFIDENCE_THRESHOLD,
    OCR_CHAR_ASPECT_RATIO_THRESHOLD,
    OCR_ALPHABET,
    OCR_CASE_SENSITIVE,
    OCR_INCLUDE_GLYPH_SIZE_STATS,
    OCR_INCLUDE_NORMALIZED_FREQUENCY,
)


class AltoGlyph:
    def __init__(self, char: str, gc: float, width: Optional[float], height: Optional[float]):
        self.char = char
        self.gc = gc
        self.width = width
        self.height = height


def _get_local_tag(tag: str) -> str:
    """Return the local tag name without namespace prefix."""
    if '}' in tag:
        return tag.split('}', 1)[1]
    return tag


def extract_glyphs_from_alto(
    alto_xml_path: str,
    glyph_conf_threshold: float = OCR_GLYPH_CONFIDENCE_THRESHOLD,
    string_conf_threshold: float = OCR_STRING_CONFIDENCE_THRESHOLD,
    aspect_ratio_threshold: Optional[float] = OCR_CHAR_ASPECT_RATIO_THRESHOLD,
) -> List[AltoGlyph]:
    """
    Parse an ALTO XML file and extract glyphs (characters) with their confidences.

    Filters (applied in order):
    1. Keeps only glyphs with GC >= glyph_conf_threshold
    2. If string_conf_threshold > 0, also requires parent String WC >= string_conf_threshold (when present)
    3. If aspect_ratio_threshold is set, filters out characters with aspect ratio > threshold
       (aspect ratio = max(width, height) / min(width, height))
    """
    tree = ET.parse(alto_xml_path)
    root = tree.getroot()

    glyphs: List[AltoGlyph] = []

    # Iterate over all String elements (namespace-agnostic)
    for string_elem in root.iter():
        if _get_local_tag(string_elem.tag) != 'String':
            continue

        # Optional filter on word confidence (WC)
        wc_attr = string_elem.attrib.get('WC')
        if wc_attr is not None and string_conf_threshold > 0.0:
            try:
                wc_val = float(wc_attr)
            except ValueError:
                wc_val = 0.0
            if wc_val < string_conf_threshold:
                continue

        # Iterate child Glyph elements
        for child in list(string_elem):
            if _get_local_tag(child.tag) != 'Glyph':
                continue
            char = child.attrib.get('CONTENT')
            gc_attr = child.attrib.get('GC')
            if char is None or gc_attr is None:
                continue
            try:
                gc_val = float(gc_attr)
            except ValueError:
                continue
            if gc_val < glyph_conf_threshold:
                continue

            width = None
            height = None
            if 'WIDTH' in child.attrib:
                try:
                    width = float(child.attrib['WIDTH'])
                except ValueError:
                    width = None
            if 'HEIGHT' in child.attrib:
                try:
                    height = float(child.attrib['HEIGHT'])
                except ValueError:
                    height = None

            # Apply aspect ratio filter (after confidence filtering)
            if aspect_ratio_threshold is not None and aspect_ratio_threshold > 0.0:
                if width is not None and height is not None and width > 0 and height > 0:
                    aspect_ratio = max(width, height) / min(width, height)
                    if aspect_ratio > aspect_ratio_threshold:
                        continue
                # If width or height is missing, keep the glyph (can't compute aspect ratio)

            glyphs.append(AltoGlyph(char=char, gc=gc_val, width=width, height=height))

    return glyphs


def _normalize_char(c: str, case_sensitive: bool) -> str:
    return c if case_sensitive else c.lower()


def aggregate_character_stats(
    glyphs: Sequence[AltoGlyph],
    alphabet: Optional[Sequence[str]] = OCR_ALPHABET,
    case_sensitive: bool = OCR_CASE_SENSITIVE,
    include_geometry: bool = OCR_INCLUDE_GLYPH_SIZE_STATS,
    include_norm_freq: bool = OCR_INCLUDE_NORMALIZED_FREQUENCY,
) -> Tuple[np.ndarray, List[str], Dict[str, Dict[str, float]]]:
    """
    Aggregate per-character statistics:
    - count: number of glyphs per character
    - mean_gc: mean glyph confidence
    - var_gc: variance of glyph confidence
    Optionally adds:
    - norm_freq: count / total_count
    - mean_width/var_width, mean_height/var_height

    Returns: (feature_vector, alphabet_used, per_char_stats)
    """
    # Normalize characters and collect arrays
    per_char_gc: Dict[str, List[float]] = {}
    per_char_w: Dict[str, List[float]] = {}
    per_char_h: Dict[str, List[float]] = {}

    for g in glyphs:
        char = _normalize_char(g.char, case_sensitive)
        per_char_gc.setdefault(char, []).append(g.gc)
        if include_geometry:
            if g.width is not None:
                per_char_w.setdefault(char, []).append(g.width)
            if g.height is not None:
                per_char_h.setdefault(char, []).append(g.height)

    if alphabet is None:
        # Use sorted unique characters to define order
        alphabet_used = sorted(per_char_gc.keys())
    else:
        # Keep provided order; normalize characters for consistency
        alphabet_used = [_normalize_char(c, case_sensitive) for c in alphabet]

    total_count = sum(len(per_char_gc.get(ch, [])) for ch in alphabet_used)
    feature_components: List[float] = []
    per_char_stats: Dict[str, Dict[str, float]] = {}

    for ch in alphabet_used:
        gc_vals = per_char_gc.get(ch, [])
        count = float(len(gc_vals))
        if count > 0:
            mean_gc = float(np.mean(gc_vals))
            var_gc = float(np.var(gc_vals))
        else:
            mean_gc = 0.0
            var_gc = 0.0

        # Base features: count, mean, variance
        features_for_char: List[float] = [count, mean_gc, var_gc]

        # Optional normalized frequency
        if include_norm_freq:
            norm_freq = float(count / total_count) if total_count > 0 else 0.0
            features_for_char.append(norm_freq)

        # Optional geometry stats
        if include_geometry:
            w_vals = per_char_w.get(ch, [])
            h_vals = per_char_h.get(ch, [])
            if len(w_vals) > 0:
                mean_w = float(np.mean(w_vals))
                var_w = float(np.var(w_vals))
            else:
                mean_w = 0.0
                var_w = 0.0
            if len(h_vals) > 0:
                mean_h = float(np.mean(h_vals))
                var_h = float(np.var(h_vals))
            else:
                mean_h = 0.0
                var_h = 0.0
            features_for_char.extend([mean_w, var_w, mean_h, var_h])

        feature_components.extend(features_for_char)

        stat_entry: Dict[str, float] = {
            'count': count,
            'mean_gc': mean_gc,
            'var_gc': var_gc,
        }
        if include_norm_freq:
            stat_entry['norm_freq'] = features_for_char[3] if include_norm_freq else 0.0
        if include_geometry:
            # Indices depend on include_norm_freq; compute directly for clarity
            stat_entry['mean_width'] = float(np.mean(per_char_w.get(ch, []))) if per_char_w.get(ch) else 0.0
            stat_entry['var_width'] = float(np.var(per_char_w.get(ch, []))) if per_char_w.get(ch) else 0.0
            stat_entry['mean_height'] = float(np.mean(per_char_h.get(ch, []))) if per_char_h.get(ch) else 0.0
            stat_entry['var_height'] = float(np.var(per_char_h.get(ch, []))) if per_char_h.get(ch) else 0.0

        per_char_stats[ch] = stat_entry

    feature_vector = np.array(feature_components, dtype=np.float32)
    return feature_vector, alphabet_used, per_char_stats


def compute_alto_feature_vector(
    alto_xml_path: str,
    glyph_conf_threshold: float = OCR_GLYPH_CONFIDENCE_THRESHOLD,
    string_conf_threshold: float = OCR_STRING_CONFIDENCE_THRESHOLD,
    aspect_ratio_threshold: Optional[float] = OCR_CHAR_ASPECT_RATIO_THRESHOLD,
    alphabet: Optional[Sequence[str]] = OCR_ALPHABET,
    case_sensitive: bool = OCR_CASE_SENSITIVE,
    include_geometry: bool = OCR_INCLUDE_GLYPH_SIZE_STATS,
    include_norm_freq: bool = OCR_INCLUDE_NORMALIZED_FREQUENCY,
) -> Tuple[np.ndarray, List[str], Dict[str, Dict[str, float]]]:
    glyphs = extract_glyphs_from_alto(
        alto_xml_path,
        glyph_conf_threshold=glyph_conf_threshold,
        string_conf_threshold=string_conf_threshold,
        aspect_ratio_threshold=aspect_ratio_threshold,
    )
    return aggregate_character_stats(
        glyphs,
        alphabet=alphabet,
        case_sensitive=case_sensitive,
        include_geometry=include_geometry,
        include_norm_freq=include_norm_freq,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-character feature vector from ALTO XML.")
    parser.add_argument("alto_xml", type=str, help="Path to ALTO XML file")
    parser.add_argument("--alphabet", type=str, default=None, help="Fixed alphabet string to use (order defines vector order)")
    parser.add_argument("--glyph-threshold", type=float, default=OCR_GLYPH_CONFIDENCE_THRESHOLD, help="Min GC to include glyph")
    parser.add_argument("--string-threshold", type=float, default=OCR_STRING_CONFIDENCE_THRESHOLD, help="Min WC to include String (0 to ignore)")
    parser.add_argument("--aspect-ratio-threshold", type=float, default=OCR_CHAR_ASPECT_RATIO_THRESHOLD, help="Max aspect ratio (max(w,h)/min(w,h)) to include character (0 or None to disable)")
    parser.add_argument("--case-sensitive", action="store_true", help="Treat characters case-sensitively")
    parser.add_argument("--include-geometry", action="store_true", help="Include glyph width/height stats")
    parser.add_argument("--no-norm-freq", action="store_true", help="Do not include normalized frequency feature")

    args = parser.parse_args()

    alphabet = list(args.alphabet) if args.alphabet is not None else None
    vec, alphabet_used, stats = compute_alto_feature_vector(
        args.alto_xml,
        glyph_conf_threshold=args.glyph_threshold,
        string_conf_threshold=args.string_threshold,
        aspect_ratio_threshold=args.aspect_ratio_threshold if args.aspect_ratio_threshold > 0 else None,
        alphabet=alphabet,
        case_sensitive=args.case_sensitive,
        include_geometry=args.include_geometry,
        include_norm_freq=(not args.no_norm_freq),
    )

    result = {
        'alto_xml': os.path.abspath(args.alto_xml),
        'alphabet': alphabet_used,
        'vector_length': int(vec.shape[0]),
        'nonzero_features': int(np.count_nonzero(vec)),
        'per_char_stats': stats,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


