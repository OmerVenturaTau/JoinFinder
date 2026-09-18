"""
Extract character patches from manuscript images using ALTO XML glyph coordinates.

Uses the existing parse_alto_strings function with threshold and Hebrew dictionary filtering.
"""

import os
import sys
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image
import torch
import logging

# Set up logger
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from system import (
    OCR_GLYPH_CONFIDENCE_THRESHOLD, 
    OCR_STRING_CONFIDENCE_THRESHOLD, 
    USE_HEBREW_DICT_CHECK,
    XML_PATCH_DETECT_ROTATION,
    GLYPH_INPUT_ALPHABET,
    NUM_GLYPH_CLASSES,
    GLYPHS_PER_CLASS,
    GLYPH_ONLY_MIDDLE_LETTERS,
    GLYPH_MIN_WORD_LENGTH_FOR_MIDDLE,
    GLYPH_TIGHTEN_BBOX_TO_INK,
    GLYPH_TIGHTEN_INK_PERCENTILE,
    GLYPH_TIGHTEN_INK_MAX_THRESHOLD,
    GLYPH_TIGHTEN_MAX_COLOR_SPREAD,
    GLYPH_TIGHTEN_MIN_PIXELS_PER_LINE,
    GLYPH_TIGHTEN_MARGIN_RATIO,
    GLYPH_TIGHTEN_MIN_SIZE_RATIO,
    GLYPH_MIN_INK_FRACTION,
    XML_PATCH_READING_DIRECTION_RTL,
)
from utilities.xml_loader import find_xml_path_pretrain
from utilities.VisionModule.alto_parser import (
    parse_alto_strings,
    StringInfo,
    GlyphInfo,
    HebrewDictChecker,
    _is_hebrew,
    _polygon_bbox,
)
from utilities.VisionModule.xml_patch_extraction import detect_page_layout, parse_alto_text_regions
from utilities.page_rotation import (
    detect_page_rotation,
    apply_rotation_correction,
)


_HEBREW_CHAR_TO_ID = {ch: i for i, ch in enumerate(GLYPH_INPUT_ALPHABET)}
_UNKNOWN_CLASS_ID = NUM_GLYPH_CLASSES - 1
_GLYPH_INPUT_CHAR_SET = set(GLYPH_INPUT_ALPHABET)


def char_to_class_id(char: str) -> int:
    """Map a Hebrew character to its class index (0..NUM_GLYPH_CLASSES-1)."""
    return _HEBREW_CHAR_TO_ID.get(char, _UNKNOWN_CLASS_ID)


def _select_balanced_glyphs(
    word_glyphs: List,
    fallback_glyphs: List,
    max_chars: int,
    per_class: int = GLYPHS_PER_CLASS,
    input_alphabet: str = GLYPH_INPUT_ALPHABET,
) -> List:
    """
    Select glyphs balanced across character classes.

    Priority:
    1. From *word_glyphs* (glyphs whose parent word passed the confidence threshold).
    2. If a class still has fewer than *per_class*, fill from *fallback_glyphs*
       (glyphs that individually pass the glyph threshold but whose word didn't).

    Within each class, pick by descending glyph confidence.
    Classes are capped at ``per_class`` so the selected glyph pool stays uniform
    across Hebrew letters; missing classes are left empty rather than overfilling
    other classes.
    """
    from collections import defaultdict

    char_to_id = {char: index for index, char in enumerate(input_alphabet)}
    input_char_set = set(input_alphabet)

    def _bucket(glyphs):
        buckets = defaultdict(list)
        for g in glyphs:
            if g.char not in input_char_set:
                continue
            cid = char_to_id[g.char]
            buckets[cid].append(g)
        for cid in buckets:
            buckets[cid].sort(key=lambda g: g.gc or 0.0, reverse=True)
        return buckets

    word_buckets = _bucket(word_glyphs)
    fb_buckets = _bucket(fallback_glyphs)

    selected: dict[int, list] = defaultdict(list)
    seen_ids: set = set()

    def _add(glyph, cid):
        gid = id(glyph)
        if gid not in seen_ids:
            seen_ids.add(gid)
            selected[cid].append(glyph)
            return True
        return False

    # Pass 1: fill each class up to per_class from word_glyphs first
    input_class_ids = list(range(len(input_alphabet)))

    for cid in input_class_ids:
        for g in word_buckets.get(cid, []):
            if len(selected[cid]) >= per_class:
                break
            _add(g, cid)

    # Pass 2: fill remaining slots per class from fallback_glyphs
    for cid in input_class_ids:
        if len(selected[cid]) >= per_class:
            continue
        for g in fb_buckets.get(cid, []):
            if len(selected[cid]) >= per_class:
                break
            _add(g, cid)

    # Flatten and return in a stable order (class-id then confidence)
    result = []
    for cid in input_class_ids:
        result.extend(selected[cid])
    return result[:max_chars]


def _sample_parchment_fill(image: Image.Image) -> Tuple[int, ...]:
    """
    Estimate a parchment-toned fill color by sampling the 4 corners and 4
    mid-edge pixels of ``image`` and taking the per-channel median.

    Median (rather than mean) is robust against an edge point that lands on
    ink: as long as a majority of the 8 samples sit on parchment, the result
    is a realistic background tone. Falls back to white for RGB / 255 for
    single-band modes when sampling fails (e.g. degenerate sizes).
    """
    mode = image.mode if image.mode else "RGB"
    w, h = image.size
    if w <= 0 or h <= 0:
        if mode == "RGB":
            return (255, 255, 255)
        if mode == "L":
            return (255,)
        return (0,)

    rx = max(0, w - 1)
    by = max(0, h - 1)
    mx = w // 2
    my = h // 2
    points = [(0, 0), (rx, 0), (0, by), (rx, by), (mx, 0), (mx, by), (0, my), (rx, my)]
    samples = [image.getpixel(p) for p in points]

    if mode == "RGB":
        arr = np.asarray(samples, dtype=np.int32)
        if arr.ndim == 1:  # PIL handed back ints (e.g. palette mode); broadcast
            arr = np.stack([arr, arr, arr], axis=-1)
        med = np.median(arr, axis=0)
        return (int(med[0]), int(med[1]), int(med[2]))
    if mode == "L":
        arr = np.asarray(samples, dtype=np.int32)
        return (int(np.median(arr)),)
    return (0,)


def _tighten_bbox_to_ink(
    patch: Image.Image,
    *,
    ink_percentile: float = GLYPH_TIGHTEN_INK_PERCENTILE,
    ink_max_threshold: float = GLYPH_TIGHTEN_INK_MAX_THRESHOLD,
    max_color_spread: int = GLYPH_TIGHTEN_MAX_COLOR_SPREAD,
    min_pixels_per_line: int = GLYPH_TIGHTEN_MIN_PIXELS_PER_LINE,
    margin_ratio: float = GLYPH_TIGHTEN_MARGIN_RATIO,
    min_size_ratio: float = GLYPH_TIGHTEN_MIN_SIZE_RATIO,
) -> Tuple[Image.Image, Optional[Tuple[int, int, int, int]]]:
    """
    Shrink a glyph crop to the bounding box of its actual ink pixels.

    A pixel counts as ink only when it is **both** dark (grayscale below
    ``ink_percentile`` of the patch, capped at ``ink_max_threshold``) **and**
    near-neutral (peak-to-peak across RGB <= ``max_color_spread``). The
    neutrality check is what makes this safe at the parchment edge: it
    excludes saturated library-board blue (high R-vs-B spread) that would
    otherwise be classified as ink and prevent useful tightening when an
    OCR bbox extends past the parchment.

    The implementation is fully vectorized in numpy so it runs in microseconds
    per glyph and is fine to call on hundreds of glyphs per page.

    Returns ``(possibly_cropped_patch, offsets_within_patch_or_None)`` where
    ``offsets_within_patch`` is ``(left, top, right, bottom)`` relative to
    the input patch. When the patch is returned unchanged, offsets are
    ``None`` so callers can branch cheaply.

    Defensive: leaves the patch untouched whenever (a) the patch has no
    ink-classified pixels, (b) trimming would reduce either axis below
    ``min_size_ratio`` of its current length (likely a misclassification),
    or (c) trimming wouldn't actually shrink the patch.
    """
    if patch.size[0] < 6 or patch.size[1] < 6:
        return patch, None

    arr = np.asarray(patch.convert("RGB"), dtype=np.int16)
    h, w = arr.shape[:2]
    gray = arr.mean(axis=2)
    threshold = float(np.percentile(gray, ink_percentile))
    threshold = min(threshold, float(ink_max_threshold))
    peak2peak = arr.max(axis=2) - arr.min(axis=2)
    ink = (gray < threshold) & (peak2peak <= int(max_color_spread))
    if not ink.any():
        return patch, None

    rows_with_ink = ink.sum(axis=1) >= int(min_pixels_per_line)
    cols_with_ink = ink.sum(axis=0) >= int(min_pixels_per_line)
    if not rows_with_ink.any() or not cols_with_ink.any():
        return patch, None

    rows = np.where(rows_with_ink)[0]
    cols = np.where(cols_with_ink)[0]
    top, bottom = int(rows[0]), int(rows[-1] + 1)
    left, right = int(cols[0]), int(cols[-1] + 1)

    margin = int(round(max(1.0, float(margin_ratio) * float(min(h, w)))))
    top = max(0, top - margin)
    bottom = min(h, bottom + margin)
    left = max(0, left - margin)
    right = min(w, right + margin)

    new_w = right - left
    new_h = bottom - top
    min_w = max(2, int(round(w * float(min_size_ratio))))
    min_h = max(2, int(round(h * float(min_size_ratio))))
    if new_w < min_w or new_h < min_h:
        return patch, None
    if new_w >= w - 1 and new_h >= h - 1:
        return patch, None

    return patch.crop((left, top, right, bottom)), (left, top, right, bottom)


def _glyph_patch_ink_fraction(
    patch: Image.Image,
    *,
    ink_percentile: float = GLYPH_TIGHTEN_INK_PERCENTILE,
    ink_max_threshold: float = GLYPH_TIGHTEN_INK_MAX_THRESHOLD,
    max_color_spread: int = GLYPH_TIGHTEN_MAX_COLOR_SPREAD,
) -> float:
    """
    Fraction of pixels in ``patch`` that count as manuscript ink (same rule as
    ``_tighten_bbox_to_ink``): dark relative to patch brightness percentile
    (capped) and low RGB spread so saturated backing is not counted.
    """
    arr = np.asarray(patch.convert("RGB"), dtype=np.int16)
    h, w = arr.shape[:2]
    if h == 0 or w == 0:
        return 0.0
    gray = arr.mean(axis=2)
    threshold = float(np.percentile(gray, ink_percentile))
    threshold = min(threshold, float(ink_max_threshold))
    peak2peak = arr.max(axis=2) - arr.min(axis=2)
    ink = (gray < threshold) & (peak2peak <= int(max_color_spread))
    return float(ink.sum()) / float(h * w)


def _resize_preserve_aspect_and_pad_to_square(patch: Image.Image, target_size: int) -> Image.Image:
    """
    Resize a glyph patch to fit within (target_size, target_size) preserving aspect ratio,
    then pad to a square of exactly (target_size, target_size).

    This avoids distorting tall/wide glyphs (important for stroke-level detail).
    The padding color is sampled from the resized glyph's own edges so the
    background blends with the surrounding parchment instead of leaving a
    hard solid (and previously red, due to a PIL packed-int bug) margin.
    """
    if target_size <= 0:
        raise ValueError(f"target_size must be > 0, got {target_size}")

    w, h = patch.size
    if w <= 0 or h <= 0:
        # Fallback: return blank square in the patch's mode using a proper tuple
        mode = patch.mode if patch.mode else "RGB"
        if mode == "RGB":
            fill: Tuple[int, ...] = (255, 255, 255)
        elif mode == "L":
            fill = (255,)
        else:
            fill = (0,)
        return Image.new(mode, (target_size, target_size), color=fill)

    # Scale so the long side matches target_size
    scale = float(target_size) / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = patch.resize((new_w, new_h), Image.Resampling.LANCZOS)

    fill_color = _sample_parchment_fill(resized)
    canvas = Image.new(resized.mode, (target_size, target_size), color=fill_color)

    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2
    canvas.paste(resized, (left, top))
    return canvas


def filter_strings_and_glyphs(
    strings: List[StringInfo],
    dict_checker: Optional[HebrewDictChecker] = None,
    glyph_context_by_id: Optional[Dict[int, Dict]] = None,
    only_middle_glyphs: bool = False,
    min_word_length_for_middle: int = GLYPH_MIN_WORD_LENGTH_FOR_MIDDLE,
) -> List[GlyphInfo]:
    """
    Apply standard filtering criteria to strings and glyphs (Hebrew-only, confidence thresholds, dictionary).
    Returns only glyphs from words that pass the word-level confidence threshold.
    """
    valid_glyphs = []
    for s in strings:
        if len(s.content.strip()) <= 1:
            continue
        if not _is_hebrew(s.content):
            continue
        if s.wc is None or s.wc < OCR_STRING_CONFIDENCE_THRESHOLD:
            continue
        if USE_HEBREW_DICT_CHECK and dict_checker is not None:
            if not dict_checker.check_word(s.content):
                continue
        
        for glyph in s.glyphs:
            if glyph.char not in _GLYPH_INPUT_CHAR_SET:
                continue
            if glyph.gc is None or glyph.gc < OCR_GLYPH_CONFIDENCE_THRESHOLD:
                continue
            if only_middle_glyphs:
                ctx = (glyph_context_by_id or {}).get(id(glyph), {})
                if int(ctx.get("word_length", 0)) < int(min_word_length_for_middle):
                    continue
                if not bool(ctx.get("is_word_middle_glyph", False)):
                    continue
            valid_glyphs.append(glyph)
    return valid_glyphs


def filter_glyphs_fallback(
    strings: List[StringInfo],
    word_glyph_ids: set,
    glyph_context_by_id: Optional[Dict[int, Dict]] = None,
    only_middle_glyphs: bool = False,
    min_word_length_for_middle: int = GLYPH_MIN_WORD_LENGTH_FOR_MIDDLE,
) -> List[GlyphInfo]:
    """
    Return glyphs that pass the *glyph*-level confidence threshold but whose
    parent word did NOT pass the word-level threshold (i.e. glyphs not already
    in the primary pool).
    """
    fallback = []
    for s in strings:
        if not _is_hebrew(s.content):
            continue
        # Only consider strings whose word confidence is below threshold
        # (strings above threshold are already handled by filter_strings_and_glyphs)
        word_passes = (
            len(s.content.strip()) > 1
            and s.wc is not None
            and s.wc >= OCR_STRING_CONFIDENCE_THRESHOLD
        )
        if word_passes:
            continue
        for glyph in s.glyphs:
            if glyph.char not in _GLYPH_INPUT_CHAR_SET:
                continue
            if id(glyph) in word_glyph_ids:
                continue
            if glyph.gc is None or glyph.gc < OCR_GLYPH_CONFIDENCE_THRESHOLD:
                continue
            if only_middle_glyphs:
                ctx = (glyph_context_by_id or {}).get(id(glyph), {})
                if int(ctx.get("word_length", 0)) < int(min_word_length_for_middle):
                    continue
                if not bool(ctx.get("is_word_middle_glyph", False)):
                    continue
            fallback.append(glyph)
    return fallback


def _build_glyph_context(strings: List[StringInfo]) -> Dict[int, Dict]:
    """Attach per-glyph word position metadata from the parsed ALTO structure."""
    glyph_context_by_id: Dict[int, Dict] = {}
    for s in strings:
        glyphs = list(s.glyphs or [])
        word_length = len(glyphs)
        line_id = getattr(s.textline, "line_id", None) if getattr(s, "textline", None) is not None else None
        for glyph_idx, glyph in enumerate(glyphs):
            glyph_context_by_id[id(glyph)] = {
                "word": s.content,
                "word_length": word_length,
                "glyph_index_in_word": glyph_idx,
                "glyph_index_from_word_end": max(0, word_length - glyph_idx - 1),
                "is_word_edge_glyph": bool(word_length > 0 and (glyph_idx == 0 or glyph_idx == word_length - 1)),
                "is_word_middle_glyph": bool(word_length >= 3 and 0 < glyph_idx < word_length - 1),
                "word_hpos": s.hpos,
                "word_vpos": s.vpos,
                "word_width": s.width,
                "word_height": s.height,
                "line_id": line_id,
            }
    return glyph_context_by_id


def extract_character_patches(
    image: Image.Image,
    image_path: str,
    char_patch_size: int = 32,
    max_chars: Optional[int] = None,
    xml_path: Optional[str] = None,
    only_middle_glyphs: bool = GLYPH_ONLY_MIDDLE_LETTERS,
    min_word_length_for_middle: int = GLYPH_MIN_WORD_LENGTH_FOR_MIDDLE,
    tighten_bbox_to_ink: bool = GLYPH_TIGHTEN_BBOX_TO_INK,
    min_ink_fraction: Optional[float] = None,
    input_alphabet: Optional[str] = None,
) -> Tuple[List[Image.Image], List[Dict]]:
    """
    Extract character patches from image using ALTO XML glyph coordinates.

    Uses parse_alto_strings with the same filtering logic as filter_strings_and_glyphs.

    When ``tighten_bbox_to_ink`` is True (default, controlled by the
    ``glyph.tighten_bbox_to_ink`` config flag), each cropped patch is
    trimmed to the actual ink pixels before being resized and padded to a
    square. This mitigates loose OCR bboxes and bboxes that extend past the
    parchment into the library backing.

    When ``min_ink_fraction`` is omitted, ``glyph.min_ink_fraction`` from
    config is used. Patches with an ink pixel fraction (same ink rule as
    tightening) below that threshold are skipped. Pass ``0.0`` to disable
    per call.
    """
    w, h = image.size
    original_size = (w, h)
    rotation_angle = 0

    eff_min_ink = GLYPH_MIN_INK_FRACTION if min_ink_fraction is None else float(min_ink_fraction)
    effective_alphabet = GLYPH_INPUT_ALPHABET if input_alphabet is None else input_alphabet
    if not effective_alphabet:
        raise ValueError("input_alphabet must contain at least one character")
    char_to_id = {char: index for index, char in enumerate(effective_alphabet)}
    # Find XML file
    if xml_path is None:
        xml_path = find_xml_path_pretrain(image_path)
    
    if xml_path is None or not os.path.exists(xml_path):
        return [], []
    
    # Detect and apply rotation if enabled (same as patch extraction)
    if XML_PATCH_DETECT_ROTATION:
        # Parse text regions to detect rotation
        text_regions, xml_image_size = parse_alto_text_regions(xml_path)
        
        # Scale text regions if XML size differs from actual image size
        if xml_image_size and xml_image_size[0] > 0 and xml_image_size[1] > 0:
            if xml_image_size != (w, h):
                scale_x = w / xml_image_size[0]
                scale_y = h / xml_image_size[1]
                for region in text_regions:
                    region['hpos'] *= scale_x
                    region['vpos'] *= scale_y
                    region['width'] *= scale_x
                    region['height'] *= scale_y
                    region['center_x'] = region['hpos'] + region['width'] / 2.0
                    region['center_y'] = region['vpos'] + region['height'] / 2.0
                    # Scale baseline coordinates if present
                    if region.get('baseline') is not None:
                        x1, y1, x2, y2 = region['baseline']
                        region['baseline'] = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
        
        # Detect rotation
        if text_regions:
            rotation_angle = detect_page_rotation(text_regions, w, h)
            
            # Apply rotation correction
            if rotation_angle != 0:
                image, text_regions = apply_rotation_correction(image, text_regions, rotation_angle)
                w, h = image.size
    
    # Parse strings using existing function
    strings = parse_alto_strings(xml_path)
    glyph_context_by_id = _build_glyph_context(strings)
    
    # Scale and transform glyph coordinates if needed
    if strings:
        # Get XML image size for scaling
        text_regions_for_size, xml_image_size = parse_alto_text_regions(xml_path)
        scale_x = 1.0
        scale_y = 1.0
        if xml_image_size and xml_image_size[0] > 0 and xml_image_size[1] > 0:
            if xml_image_size != original_size:
                scale_x = original_size[0] / xml_image_size[0]
                scale_y = original_size[1] / xml_image_size[1]
        
        # Transform glyph coordinates
        if rotation_angle != 0 or scale_x != 1.0 or scale_y != 1.0:
            from utilities.page_rotation import transform_point
            old_w, old_h = original_size
            
            for s in strings:
                for glyph in s.glyphs:
                    # Scale coordinates first (if needed)
                    if glyph.hpos is not None:
                        glyph.hpos = glyph.hpos * scale_x
                    if glyph.vpos is not None:
                        glyph.vpos = glyph.vpos * scale_y
                    if glyph.width is not None:
                        glyph.width = glyph.width * scale_x
                    if glyph.height is not None:
                        glyph.height = glyph.height * scale_y
                    
                    # Transform polygon if present
                    if glyph.polygon:
                        scaled_polygon = []
                        for px, py in glyph.polygon:
                            scaled_polygon.append((px * scale_x, py * scale_y))
                        glyph.polygon = scaled_polygon
                    
                    # Transform coordinates if rotation was applied
                    if rotation_angle != 0:
                        if glyph.hpos is not None and glyph.vpos is not None:
                            new_hpos, new_vpos = transform_point(glyph.hpos, glyph.vpos, rotation_angle, old_w, old_h)
                            glyph.hpos = new_hpos
                            glyph.vpos = new_vpos
                        
                        # Transform polygon if present
                        if glyph.polygon:
                            transformed_polygon = []
                            for px, py in glyph.polygon:
                                new_px, new_py = transform_point(px, py, rotation_angle, old_w, old_h)
                                transformed_polygon.append((new_px, new_py))
                            glyph.polygon = transformed_polygon
    
    # Initialize Hebrew dictionary checker if needed
    dict_checker = None
    if USE_HEBREW_DICT_CHECK and HebrewDictChecker is not None:
        try:
            dict_checker = HebrewDictChecker()
            if dict_checker is not None and not dict_checker.is_available():
                logger.warning("Hebrew dictionary checker initialized but dictionary module not available. Dictionary filtering disabled.")
        except Exception as e:
            logger.exception("Could not initialize Hebrew dictionary checker")
            print(f"Warning: Could not initialize Hebrew dictionary checker: {e}")
            dict_checker = None
    
    # Collect valid glyphs from word-passing strings (primary pool)
    word_glyphs = filter_strings_and_glyphs(
        strings,
        dict_checker,
        glyph_context_by_id=glyph_context_by_id,
        only_middle_glyphs=only_middle_glyphs,
        min_word_length_for_middle=min_word_length_for_middle,
    )
    
    # Collect fallback glyphs (glyph confidence OK, word confidence too low)
    word_glyph_ids = {id(g) for g in word_glyphs}
    fallback_glyphs = filter_glyphs_fallback(
        strings,
        word_glyph_ids,
        glyph_context_by_id=glyph_context_by_id,
        only_middle_glyphs=only_middle_glyphs,
        min_word_length_for_middle=min_word_length_for_middle,
    )
    
    if not word_glyphs and not fallback_glyphs:
        return [], []
    
    # Balanced class-aware selection (word-first priority)
    effective_max = max_chars if max_chars is not None else (len(word_glyphs) + len(fallback_glyphs))
    all_glyphs = _select_balanced_glyphs(
        word_glyphs,
        fallback_glyphs,
        effective_max,
        input_alphabet=effective_alphabet,
    )
    
    patches = []
    metadata_list = []
    glyph_layout_regions = []

    for glyph in all_glyphs:
        if glyph.polygon:
            min_x, min_y, max_x, max_y = _polygon_bbox(glyph.polygon)
            left, top, right, bottom = int(min_x), int(min_y), int(max_x), int(max_y)
        elif glyph.hpos is not None and glyph.vpos is not None:
            left, top = int(glyph.hpos), int(glyph.vpos)
            width = int(glyph.width) if glyph.width is not None else char_patch_size
            height = int(glyph.height) if glyph.height is not None else char_patch_size
            right, bottom = left + width, top + height
        else:
            continue
        glyph_layout_regions.append({
            'center_x': (left + right) / 2.0,
            'center_y': (top + bottom) / 2.0,
        })

    if glyph_layout_regions:
        is_two_page, split_x = detect_page_layout(glyph_layout_regions, w, h)
    else:
        is_two_page, split_x = (False, None)
    
    for glyph in all_glyphs:
        if glyph.polygon:
            min_x, min_y, max_x, max_y = _polygon_bbox(glyph.polygon)
            left, top, right, bottom = int(min_x), int(min_y), int(max_x), int(max_y)
        elif glyph.hpos is not None and glyph.vpos is not None:
            left, top = int(glyph.hpos), int(glyph.vpos)
            width = int(glyph.width) if glyph.width is not None else char_patch_size
            height = int(glyph.height) if glyph.height is not None else char_patch_size
            right, bottom = left + width, top + height
        else:
            continue
        
        left, top = max(0, min(left, w - 1)), max(0, min(top, h - 1))
        right, bottom = max(left + 1, min(right, w)), max(top + 1, min(bottom, h))
        
        try:
            ocr_left, ocr_top, ocr_right, ocr_bottom = left, top, right, bottom
            patch = image.crop((ocr_left, ocr_top, ocr_right, ocr_bottom))
            tightened = False
            if tighten_bbox_to_ink:
                patch, ink_offsets = _tighten_bbox_to_ink(patch)
                if ink_offsets is not None:
                    local_left, local_top, local_right, local_bottom = ink_offsets
                    left = ocr_left + local_left
                    top = ocr_top + local_top
                    right = ocr_left + local_right
                    bottom = ocr_top + local_bottom
                    tightened = True
            ink_frac = _glyph_patch_ink_fraction(patch)
            if eff_min_ink > 0.0 and ink_frac < eff_min_ink:
                continue
            patch = _resize_preserve_aspect_and_pad_to_square(patch, char_patch_size)
            patches.append(patch)

            norm_x, norm_y = float(left) / w, float(top) / h
            norm_width, norm_height = float(right - left) / w, float(bottom - top) / h
            center_x = (left + right) / 2.0
            page_segment = 0
            if is_two_page and split_x is not None:
                is_right = center_x >= split_x
                page_segment = 0 if is_right else 1
                if not XML_PATCH_READING_DIRECTION_RTL:
                    page_segment = 0 if not is_right else 1

            metadata_list.append({
                'char': glyph.char,
                'char_class_id': char_to_id[glyph.char],
                'gc': glyph.gc,
                'hpos': float(left),
                'vpos': float(top),
                'width': float(right - left),
                'height': float(bottom - top),
                'normalized_x': norm_x,
                'normalized_y': norm_y,
                'normalized_w': norm_width,
                'normalized_h': norm_height,
                'page_segment': page_segment,
                'page_split_x': split_x,
                'is_two_page': is_two_page,
                'rotation_angle': rotation_angle,
                'selection_only_middle_glyphs': bool(only_middle_glyphs),
                'selection_min_word_length_for_middle': int(min_word_length_for_middle),
                'bbox_tightened_to_ink': bool(tightened),
                'ocr_hpos': float(ocr_left),
                'ocr_vpos': float(ocr_top),
                'ocr_width': float(ocr_right - ocr_left),
                'ocr_height': float(ocr_bottom - ocr_top),
                'ink_fraction': float(ink_frac),
                'min_ink_fraction_applied': float(eff_min_ink),
                **glyph_context_by_id.get(id(glyph), {}),
            })
        except Exception as e:
            logger.exception(f"Could not extract patch for glyph {glyph.char}")
            continue
    
    return patches, metadata_list
