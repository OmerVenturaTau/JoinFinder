#!/usr/bin/env python3
"""
Run all visualizations (patches, words, glyphs) for a manuscript image.

You can run it in two input modes:
  1) By manuscript:
     --manuscript-id <id>
     (select a row for that manuscript)
  2) By exact image:
     --image <manuscript_id>/<image_id>
     (image_id is `picture_id` in the DB)

DB source for fetching `xml_path` and `image_path`:
  - --mode geniza: uses `geniza_image_information`
  - --mode pretrain: uses `pretrain_finetune_oriental_non_oriental_train_val_test_split`

Outputs are saved to:
  debug_tools/Visualization/outputs/<manuscript_id>/<image_stem>/
    viz_patches.jpg       (patch boundaries overlay on full page)
    viz_words.jpg         (word boxes overlay on full page)
    viz_chars.jpg         (glyph boxes overlay on full page)
    viz_glyph_grid.jpg   (montage of sampled glyphs, grouped by letter)
    viz_glyph_tighten_comparison.png  (OCR vs ink-tight glyph crops; header shows config ON/OFF)
    viz_patch_grid.jpg   (montage of extracted patches/tiles)
    summary.txt           (counts + pipeline summary)
    selected_row.json    (DB row used to select image/xml)

Examples:
  python debug_tools/Visualization/visualize_all_for_manuscript.py --manuscript 99005239975 --mode geniza --random
  python debug_tools/Visualization/visualize_all_for_manuscript.py --image 99005239975/IE168614219_P000002_FL168614222.jpg --mode geniza
  python debug_tools/Visualization/visualize_all_for_manuscript.py --image 99005239975/IE168614219_P000002_FL168614222.jpg --mode pretrain --split train
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from collections import Counter
from itertools import groupby
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib-cache")

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.utils import make_grid


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "system.py").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from system import (  # noqa: E402
    CHAR_PATCH_SIZE,
    DISABLE_PIL_LIMIT,
    DB_CONFIG_PATH,
    GENIZA_IMAGE_INFORMATION_TABLE,
    GENIZA_IMAGE_LATENTS_TABLE,
    GLYPH_TIGHTEN_BBOX_TO_INK,
    HEBREW_ALPHABET,
    MAX_CHARS_PER_IMAGE,
    MAX_TILES_EVAL,
    MAX_WORDS_PER_IMAGE,
    OCR_STRING_CONFIDENCE_THRESHOLD,
    USE_HEBREW_DICT_CHECK,
    PRETRAIN_TABLE_NAME,
    TILE_SIZE,
    TILE_STRIDE,
)
from models.word_branch import WordHardQualityFilter  # noqa: E402
from models.glyph_branch import GlyphHardQualityFilter  # noqa: E402
from utilities.ContextModule.xml_word_extraction import extract_words_from_alto  # noqa: E402
from utilities.VisionModule.xml_character_extraction import extract_character_patches  # noqa: E402
from utilities.VisionModule.xml_patch_extraction import extract_patches_with_xml  # noqa: E402
from utilities.xml_loader import find_xml_path_pretrain  # noqa: E402
from utilities.VisionModule.xml_patch_extraction import find_xml_path as find_xml_path_local  # noqa: E402


# --- OCR vs ink-tight bbox comparison (glyph debug) ---------------------------------
_GLYPH_TIGHTEN_GRID_TILE = 96
_GLYPH_TIGHTEN_COMPARISON_NCOLS = 8


def _glyph_tighten_resize_for_grid(patch: Image.Image, side: int) -> Image.Image:
    if patch.mode != "RGB":
        patch = patch.convert("RGB")
    if patch.size != (side, side):
        patch = patch.resize((side, side), Image.Resampling.LANCZOS)
    return patch


def _glyph_tighten_load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    if bold:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        ]
    for path in candidates:
        try:
            if os.path.exists(path):
                return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _glyph_tighten_build_comparison_grid(
    befores: List[Image.Image],
    afters: List[Image.Image],
    tightened_flags: List[bool],
    chars: List[str],
    *,
    tile: int = _GLYPH_TIGHTEN_GRID_TILE,
    ncols: int = _GLYPH_TIGHTEN_COMPARISON_NCOLS,
    bg: Tuple[int, int, int] = (245, 240, 225),
    color_tightened: Tuple[int, int, int] = (40, 160, 60),
    color_unchanged: Tuple[int, int, int] = (200, 70, 70),
) -> Image.Image:
    n = len(befores)
    if not (len(afters) == n == len(tightened_flags) == len(chars)):
        raise ValueError("Mismatched lengths for glyph tighten comparison grid.")
    if n == 0:
        return Image.new("RGB", (tile * 2, tile), color=bg)

    inner_pad = 4
    label_h = 28
    border = 3
    cell_w = 2 * tile + inner_pad + 2 * border
    cell_h = tile + label_h + 2 * border
    outer_pad = 8

    nrows = (n + ncols - 1) // ncols
    width = ncols * cell_w + (ncols + 1) * outer_pad
    height = nrows * cell_h + (nrows + 1) * outer_pad

    canvas = Image.new("RGB", (width, height), color=bg)
    draw = ImageDraw.Draw(canvas)
    font = _glyph_tighten_load_font(20, bold=True)

    for idx in range(n):
        r, c = divmod(idx, ncols)
        x = outer_pad + c * (cell_w + outer_pad)
        y = outer_pad + r * (cell_h + outer_pad)

        frame_color = color_tightened if tightened_flags[idx] else color_unchanged
        draw.rectangle([x, y, x + cell_w - 1, y + cell_h - 1], outline=frame_color, width=border)

        b = _glyph_tighten_resize_for_grid(befores[idx], tile)
        a = _glyph_tighten_resize_for_grid(afters[idx], tile)
        canvas.paste(b, (x + border, y + border + label_h))
        canvas.paste(a, (x + border + tile + inner_pad, y + border + label_h))

        tag = "T" if tightened_flags[idx] else "-"
        char = chars[idx] if chars[idx] else "?"
        label = f"{tag}  {char}"
        draw.text((x + border + 4, y + border + 2), label, fill=frame_color, font=font)

    return canvas


def _glyph_tighten_align_metadata(meta_before: List[dict], meta_after: List[dict]) -> None:
    if len(meta_before) != len(meta_after):
        raise RuntimeError(
            "Glyph counts disagree between tighten off/on extraction runs "
            f"({len(meta_before)} vs {len(meta_after)})."
        )
    keys = ("char", "ocr_hpos", "ocr_vpos", "ocr_width", "ocr_height")
    for i, (mb, ma) in enumerate(zip(meta_before, meta_after)):
        for k in keys:
            if mb.get(k) != ma.get(k):
                raise RuntimeError(
                    f"Glyph #{i} metadata mismatch on {k!r}: {mb.get(k)!r} vs {ma.get(k)!r}."
                )


def _glyph_tighten_add_page_header(body: Image.Image, *, page_title: str) -> Image.Image:
    """Stack a title block above ``body`` (same width). Shows whether training uses tightening."""
    w, h = body.size
    lines = [
        f"{page_title}   |   glyph.tighten_bbox_to_ink: {'ENABLED (active in training)' if GLYPH_TIGHTEN_BBOX_TO_INK else 'DISABLED (not used in training)'}",
        "LEFT column = OCR glyph crop (preview with tightening off)  |  RIGHT = ink-tight crop (preview with tightening on)",
        "Cell border: green = bbox shrank to ink  |  red = unchanged (OCR box kept)",
    ]
    font_top = _glyph_tighten_load_font(22, bold=True)
    font_sub = _glyph_tighten_load_font(17, bold=True)
    margin = 10
    line_gap = 5
    probe = Image.new("RGB", (max(w, 400), 200))
    dr = ImageDraw.Draw(probe)
    y_cursor = margin
    for i, line in enumerate(lines):
        font = font_top if i == 0 else font_sub
        dr.text((margin, y_cursor), line, fill=(0, 0, 0), font=font)
        bbox = dr.textbbox((margin, y_cursor), line, font=font)
        y_cursor = bbox[3] + line_gap
    banner_h = y_cursor + margin

    banner = Image.new("RGB", (w, banner_h), color=(245, 240, 225))
    dr2 = ImageDraw.Draw(banner)
    y_cursor = margin
    for i, line in enumerate(lines):
        font = font_top if i == 0 else font_sub
        dr2.text((margin, y_cursor), line, fill=(28, 28, 28), font=font)
        bbox = dr2.textbbox((margin, y_cursor), line, font=font)
        y_cursor = bbox[3] + line_gap

    # If the grid is narrower than some lines, wrap is not implemented; widen banner to body only.
    out = Image.new("RGB", (w, banner_h + h), color=(245, 240, 225))
    out.paste(banner, (0, 0))
    out.paste(body, (0, banner_h))
    return out


def _build_glyph_tightening_comparison_png(
    image_path: Path,
    xml_path: Path,
    *,
    page_title: str,
    char_patch_size: int = CHAR_PATCH_SIZE,
    max_chars: int = MAX_CHARS_PER_IMAGE,
) -> Image.Image:
    """Full-page PNG: header + [OCR | ink-tight] glyph montage (caps at ``max_chars``)."""
    if DISABLE_PIL_LIMIT:
        Image.MAX_IMAGE_PIXELS = None

    ip = str(image_path.resolve())
    xp = str(xml_path.resolve())
    if not image_path.exists():
        raise FileNotFoundError(ip)
    if not xml_path.exists():
        raise FileNotFoundError(xp)

    img = Image.open(ip).convert("RGB")
    before_patches, before_meta = extract_character_patches(
        img,
        ip,
        char_patch_size=char_patch_size,
        max_chars=max_chars,
        xml_path=xp,
        tighten_bbox_to_ink=False,
    )
    after_patches, after_meta = extract_character_patches(
        img,
        ip,
        char_patch_size=char_patch_size,
        max_chars=max_chars,
        xml_path=xp,
        tighten_bbox_to_ink=True,
    )

    if not before_patches:
        ph = Image.new("RGB", (720, 100), color=(245, 240, 225))
        dr = ImageDraw.Draw(ph)
        dr.text(
            (16, 40),
            "No glyphs extracted for bbox tightening comparison.",
            fill=(90, 85, 80),
            font=_glyph_tighten_load_font(20, bold=True),
        )
        return _glyph_tighten_add_page_header(ph, page_title=page_title)

    _glyph_tighten_align_metadata(before_meta, after_meta)
    tightened_flags = [bool(m.get("bbox_tightened_to_ink", False)) for m in after_meta]
    chars = [str(m.get("char") or "") for m in after_meta]
    body = _glyph_tighten_build_comparison_grid(
        before_patches,
        after_patches,
        tightened_flags,
        chars,
        tile=_GLYPH_TIGHTEN_GRID_TILE,
        ncols=_GLYPH_TIGHTEN_COMPARISON_NCOLS,
    )
    return _glyph_tighten_add_page_header(body, page_title=page_title)


def _load_db_config(db_config_path: str) -> Any:
    import configparser

    cfg_path = db_config_path
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(str(PROJECT_ROOT), cfg_path)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"DB config not found: {cfg_path}")

    config = configparser.ConfigParser()
    config.read(cfg_path)
    if "postgresql" not in config:
        raise ValueError(f"Missing [postgresql] section in DB config: {cfg_path}")
    return config["postgresql"]


def _get_db_connection(db_config_path: str):
    db = _load_db_config(db_config_path)
    from psycopg2 import connect as pg_connect

    return pg_connect(
        host=db["host"],
        database=db["database"],
        user=db["user"],
        password=db["password"],
        port=db.get("port", 5432),
    )


def _parse_image_arg(image_arg: str) -> Tuple[str, str]:
    """
    Parse `<manuscript_id>/<image_id>` and return `(manuscript_id, image_id)`.

    `image_id` corresponds to `picture_id` in the DB.
    """
    image_arg = image_arg.strip()
    if "/" not in image_arg:
        raise ValueError("`--image` must look like `<manuscript_id>/<image_id>` (e.g. 9900/IE...jpg).")
    manuscript_id, image_id = image_arg.split("/", 1)
    manuscript_id = manuscript_id.strip()
    image_id = image_id.strip()
    if not manuscript_id or not image_id:
        raise ValueError("`--image` must look like `<manuscript_id>/<image_id>` with non-empty parts.")
    return manuscript_id, image_id


def _select_row_from_geniza(
    conn,
    *,
    manuscript_id: str,
    picture_id: Optional[str],
    random_pick: bool,
    seed: Optional[int],
    require_counts: bool,
) -> Dict[str, Any]:
    rng = random.Random(seed)

    # Single-table WHERE clauses (no aliases).
    where_single = [
        "manuscript_id = %s",
        "image_path IS NOT NULL",
        "xml_path IS NOT NULL",
    ]
    params: list[Any] = [manuscript_id]
    if picture_id is not None:
        where_single.append("picture_id = %s")
        params.append(picture_id)

    if require_counts:
        # Join against latents table for cheap existence of counts.
        # Important: qualify columns (otherwise ambiguous column errors after JOIN).
        where_join = [
            "gii.manuscript_id = %s",
            "gii.image_path IS NOT NULL",
            "gii.xml_path IS NOT NULL",
        ]
        params_join: list[Any] = [manuscript_id]
        if picture_id is not None:
            where_join.append("gii.picture_id = %s")
            params_join.append(picture_id)

        where_sql = " AND ".join(where_join)
        count_sql = f"""
            SELECT COUNT(*)
            FROM {GENIZA_IMAGE_INFORMATION_TABLE} gii
            JOIN {GENIZA_IMAGE_LATENTS_TABLE} gil
              ON gil.manuscript_id = gii.manuscript_id
             AND gil.picture_id = gii.picture_id
            WHERE {where_sql}
              AND COALESCE(gil.num_visual_patches, 0) > 0
              AND COALESCE(gil.num_glyphs, 0) > 0
        """
        with conn.cursor() as cur:
            cur.execute(count_sql, tuple(params_join))
            n_rows = int(cur.fetchone()[0])
            if n_rows <= 0:
                raise RuntimeError(
                    f"No candidate rows found in geniza mode for manuscript_id={manuscript_id}, picture_id={picture_id} "
                    f"(require_counts={require_counts})."
                )
        offset = rng.randrange(n_rows) if random_pick and picture_id is None else 0

        pick_sql = f"""
            SELECT
                gii.manuscript_id,
                gii.picture_id,
                gii.page_number,
                gii.page_id,
                gii.image_path,
                gii.xml_path
            FROM {GENIZA_IMAGE_INFORMATION_TABLE} gii
            JOIN {GENIZA_IMAGE_LATENTS_TABLE} gil
              ON gil.manuscript_id = gii.manuscript_id
             AND gil.picture_id = gii.picture_id
            WHERE {where_sql}
              AND COALESCE(gil.num_visual_patches, 0) > 0
              AND COALESCE(gil.num_glyphs, 0) > 0
            ORDER BY gii.picture_id
            LIMIT 1 OFFSET %s
        """
        with conn.cursor() as cur:
            cur.execute(pick_sql, tuple(params_join) + (offset,))
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("DB row selection returned NULL unexpectedly (geniza).")
            colnames = [d.name for d in cur.description]  # type: ignore[union-attr]
            return dict(zip(colnames, row))

    where_sql = " AND ".join(where_single)
    with conn.cursor() as cur:
        if picture_id is not None:
            pick_sql = f"""
                SELECT
                    manuscript_id,
                    picture_id,
                    page_number,
                    page_id,
                    image_path,
                    xml_path
                FROM {GENIZA_IMAGE_INFORMATION_TABLE}
                WHERE {where_sql}
                LIMIT 1
            """
            cur.execute(pick_sql, tuple(params))
            row = cur.fetchone()
            if row is None:
                raise FileNotFoundError(f"No geniza_image_information row for manuscript_id={manuscript_id}, picture_id={picture_id}.")
            colnames = [d.name for d in cur.description]  # type: ignore[union-attr]
            return dict(zip(colnames, row))

        cur.execute(
            f"SELECT COUNT(*) FROM {GENIZA_IMAGE_INFORMATION_TABLE} WHERE {where_sql}",
            tuple(params),
        )
        n_rows = int(cur.fetchone()[0])
        if n_rows <= 0:
            raise RuntimeError(f"No geniza rows found for manuscript_id={manuscript_id} (require_counts={require_counts}).")
        offset = rng.randrange(n_rows) if random_pick else 0

        pick_sql = f"""
            SELECT
                manuscript_id,
                picture_id,
                page_number,
                page_id,
                image_path,
                xml_path
            FROM {GENIZA_IMAGE_INFORMATION_TABLE}
            WHERE {where_sql}
            ORDER BY picture_id
            LIMIT 1 OFFSET %s
        """
        cur.execute(pick_sql, tuple(params) + (offset,))
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("DB row selection returned NULL unexpectedly (geniza random pick).")
        colnames = [d.name for d in cur.description]  # type: ignore[union-attr]
        return dict(zip(colnames, row))


def _select_row_from_pretrain(
    conn,
    *,
    manuscript_id: str,
    picture_id: Optional[str],
    random_pick: bool,
    seed: Optional[int],
    split: str,
) -> Dict[str, Any]:
    rng = random.Random(seed)

    where = [
        "manuscript_id = %s",
        "image_path IS NOT NULL",
        "xml_path IS NOT NULL",
    ]
    params: list[Any] = [manuscript_id]

    if picture_id is not None:
        where.append("picture_id = %s")
        params.append(picture_id)

    split = (split or "any").strip().lower()
    if split != "any":
        where.append("LOWER(dataset_split::text) = %s")
        params.append(split)

    where_sql = " AND ".join(where)

    with conn.cursor() as cur:
        if picture_id is not None:
            pick_sql = f"""
                SELECT
                    manuscript_id,
                    picture_id,
                    dataset_split,
                    image_path,
                    xml_path
                FROM {PRETRAIN_TABLE_NAME}
                WHERE {where_sql}
                LIMIT 1
            """
            cur.execute(pick_sql, tuple(params))
            row = cur.fetchone()
            if row is None:
                raise FileNotFoundError(
                    f"No pretrain row for manuscript_id={manuscript_id}, picture_id={picture_id}, split={split}."
                )
            colnames = [d.name for d in cur.description]  # type: ignore[union-attr]
            return dict(zip(colnames, row))

        cur.execute(f"SELECT COUNT(*) FROM {PRETRAIN_TABLE_NAME} WHERE {where_sql}", tuple(params))
        n_rows = int(cur.fetchone()[0])
        if n_rows <= 0:
            raise RuntimeError(f"No pretrain rows found for manuscript_id={manuscript_id}, split={split}.")

        offset = rng.randrange(n_rows) if random_pick else 0

        pick_sql = f"""
            SELECT
                manuscript_id,
                picture_id,
                dataset_split,
                image_path,
                xml_path
            FROM {PRETRAIN_TABLE_NAME}
            WHERE {where_sql}
            ORDER BY picture_id
            LIMIT 1 OFFSET %s
        """
        cur.execute(pick_sql, tuple(params) + (offset,))
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("DB row selection returned NULL unexpectedly (pretrain random pick).")
        colnames = [d.name for d in cur.description]  # type: ignore[union-attr]
        return dict(zip(colnames, row))


def _compute_words_and_glyphs_summary(
    img_path: Path,
    xml_path: Path,
) -> Tuple[
    list[str],
    Dict[str, int],
    Dict[str, int],
    list[Any],  # glyph patch tensors [3, H, W]
    list[Dict[str, Any]],
    Dict[str, int],
]:
    """
    Run same word/glyph pipeline as training; return:
    - words_passed
    - per_letter_met (threshold)
    - per_letter_chosen (after diverse sampling)
    - chosen_patches (glyph PIL images)
    - chosen_meta
    """
    if DISABLE_PIL_LIMIT:
        Image.MAX_IMAGE_PIXELS = None

    # Words: extract + hard quality filter (no max_words so we get full list for summary)
    words, word_meta = extract_words_from_alto(
        alto_xml_path=str(xml_path),
        string_conf_threshold=OCR_STRING_CONFIDENCE_THRESHOLD,
        use_hebrew_dict=USE_HEBREW_DICT_CHECK,
        max_words=None,
    )
    quality_filter_w = WordHardQualityFilter()
    line_tuples = [(i, w, m) for i, (w, m) in enumerate(zip(words, word_meta))]
    filtered_tuples = quality_filter_w.filter_words_in_line(line_tuples)
    words_passed = [t[2].get("word") or t[1] for t in filtered_tuples]
    if MAX_WORDS_PER_IMAGE is not None and len(words_passed) > MAX_WORDS_PER_IMAGE:
        words_passed = words_passed[:MAX_WORDS_PER_IMAGE]

    # Glyphs: extract + hard quality filter, then diverse sampling (same as training/viz)
    img = Image.open(str(img_path)).convert("RGB")
    patches, glyph_meta = extract_character_patches(
        image=img,
        image_path=str(img_path),
        char_patch_size=CHAR_PATCH_SIZE,
        max_chars=None,
        xml_path=str(xml_path),
    )
    patch_tensors = [transforms.ToTensor()(p) for p in patches]
    extracted_counts = Counter(str(m.get("char") or "").strip() for m in glyph_meta)
    quality_filter_g = GlyphHardQualityFilter()
    filtered_patches, filtered_meta = quality_filter_g.filter_glyphs(patch_tensors, glyph_meta)

    glyph_counts = Counter(str(m.get("char") or "").strip() for m in filtered_meta)
    per_letter_met = {ch: glyph_counts.get(ch, 0) for ch in HEBREW_ALPHABET}
    for ch, count in glyph_counts.items():
        if ch and ch not in per_letter_met:
            per_letter_met[ch] = count

    chosen_patches, chosen_meta = quality_filter_g.sample_diverse_glyphs(
        filtered_patches, filtered_meta, MAX_CHARS_PER_IMAGE
    )
    chosen_counts = Counter(str(m.get("char") or "").strip() for m in chosen_meta)
    per_letter_chosen = {ch: chosen_counts.get(ch, 0) for ch in HEBREW_ALPHABET}
    for ch, count in chosen_counts.items():
        if ch and ch not in per_letter_chosen:
            per_letter_chosen[ch] = count

    return words_passed, per_letter_met, per_letter_chosen, chosen_patches, chosen_meta, dict(extracted_counts)


def _save_glyph_grid(patches: list[Any], meta: list[Dict[str, Any]], output_path: Path) -> None:
    """Save a grid montage of glyphs, sorted alphabetically by character."""
    if not patches:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No glyphs available for grid",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        plt.savefig(str(output_path), dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return

    # Sort by character (using HEBREW_ALPHABET as reference for order)
    char_to_order = {ch: i for i, ch in enumerate(HEBREW_ALPHABET)}

    indexed_meta = []
    for i, m in enumerate(meta):
        char = str(m.get("char") or "").strip()
        order = char_to_order.get(char, 999)
        indexed_meta.append((order, char, i))

    indexed_meta.sort(key=lambda x: (x[0], x[2]))
    sorted_patch_tensors = []
    sorted_chars = []
    for _, char, idx in indexed_meta:
        # `sample_diverse_glyphs` already returns tensors of shape [3, H, W].
        sorted_patch_tensors.append(patches[idx])
        sorted_chars.append(char)

    # Group consecutive same-character patches
    groups = []
    for char, g in groupby(zip(sorted_patch_tensors, sorted_chars), key=lambda x: x[1]):
        tensors = [t for t, _ in g]
        groups.append((char or "?", tensors))

    n_groups = len(groups)
    if n_groups == 0:
        return

    n_rows = n_groups
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4 * n_rows))
    if n_groups == 1:
        axes = [axes]

    for ax, (char, letter_patches) in zip(axes, groups):
        grid = make_grid(letter_patches, nrow=min(8, len(letter_patches)), padding=2)
        grid_np = grid.permute(1, 2, 0).numpy()
        ax.imshow(grid_np)
        ax.set_title(
            f"{char}\n(n={len(letter_patches)})",
            fontsize=18,
            fontweight="bold",
        )
        ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Glyph grid saved: {output_path}")


def _save_patch_grid(patches: list[Image.Image], output_path: Path, *, nrow: int = 8) -> None:
    """Save a simple montage/grid of extracted patches/tiles."""
    if not patches:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No patches extracted for grid",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        plt.savefig(str(output_path), dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return

    # Normalize to TILE_SIZE for consistent grids.
    tensors = []
    for p in patches:
        if p.size != (TILE_SIZE, TILE_SIZE):
            p = p.resize((TILE_SIZE, TILE_SIZE))
        tensors.append(transforms.ToTensor()(p))

    nrow = max(1, min(nrow, len(tensors)))
    grid = make_grid(tensors, nrow=nrow, padding=2)
    grid_np = grid.permute(1, 2, 0).numpy()

    fig = plt.figure(figsize=(max(6, nrow * 1.5), max(4, ((len(tensors) + nrow - 1) // nrow) * 1.5)))
    plt.imshow(grid_np)
    plt.axis("off")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Patch grid saved: {output_path}")


def _print_and_save_summary(
    *,
    image_name: str,
    words_passed: list[str],
    per_letter_extracted: Dict[str, int],
    per_letter_met: Dict[str, int],
    per_letter_chosen: Dict[str, int],
    patch_count: int,
    patch_config: Dict[str, Any],
    summary_path: Optional[Path],
) -> None:
    letters_per_word = [len(w) for w in words_passed]
    lines = [
        "=" * 60,
        f"VISUALIZATION SUMMARY — {image_name}",
        "=" * 60,
        "",
        "--- WORDS (met threshold) ---",
        f"  Count: {len(words_passed)}",
        f"  Words: {words_passed}",
        f"  Letters per word (same order as words): {letters_per_word}",
        "",
        "--- GLYPHS BY LETTER: extracted (after XML/middle/balanced selection) ---",
    ]
    for ch in HEBREW_ALPHABET:
        lines.append(f"  {ch}: {per_letter_extracted.get(ch, 0)}")
    others_extracted = {k: v for k, v in per_letter_extracted.items() if k not in HEBREW_ALPHABET}
    if others_extracted:
        lines.append("  (other): " + ", ".join(f"{k!r}:{v}" for k, v in sorted(others_extracted.items())))
    lines.extend(
        [
            "",
            f"  Total extracted: {sum(per_letter_extracted.values())}",
            "",
            "--- GLYPHS BY LETTER: met hard quality filter ---",
        ]
    )
    for ch in HEBREW_ALPHABET:
        lines.append(f"  {ch}: {per_letter_met.get(ch, 0)}")
    others_met = {k: v for k, v in per_letter_met.items() if k not in HEBREW_ALPHABET}
    if others_met:
        lines.append("  (other): " + ", ".join(f"{k!r}:{v}" for k, v in sorted(others_met.items())))

    lines.extend(
        [
            "",
            f"  Total met threshold: {sum(per_letter_met.values())}",
            "",
            "--- GLYPHS BY LETTER: chosen (after diverse sampling, max per image) ---",
        ]
    )
    for ch in HEBREW_ALPHABET:
        lines.append(f"  {ch}: {per_letter_chosen.get(ch, 0)}")
    others_chosen = {k: v for k, v in per_letter_chosen.items() if k not in HEBREW_ALPHABET}
    if others_chosen:
        lines.append("  (other): " + ", ".join(f"{k!r}:{v}" for k, v in sorted(others_chosen.items())))
    lines.extend(
        [
            "",
            f"  Total chosen: {sum(per_letter_chosen.values())}",
            "",
            "--- PATCHES (tiles) ---",
            f"  Count extracted: {patch_count}",
            f"  Patch config: {patch_config}",
        ]
    )

    text = "\n".join(lines)
    print(text)

    if summary_path:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(text, encoding="utf-8")
        print(f"Summary saved: {summary_path}")


def _extract_patches_for_grid(img_path: Path, xml_path: Path) -> list[Image.Image]:
    if DISABLE_PIL_LIMIT:
        Image.MAX_IMAGE_PIXELS = None

    img = Image.open(str(img_path)).convert("RGB")
    patches, _coords, _page_segments, _metadata = extract_patches_with_xml(
        image=img,
        image_path=str(img_path),
        patch_size=TILE_SIZE,
        stride=TILE_STRIDE,
        max_patches=MAX_TILES_EVAL,
        xml_path=str(xml_path),
    )
    return patches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    input_grp = parser.add_mutually_exclusive_group(required=True)
    input_grp.add_argument(
        "--manuscript-id",
        "--manuscript",
        dest="manuscript_id",
        help="Manuscript id to visualize (DB-backed).",
    )
    input_grp.add_argument(
        "--image",
        help="Exact image selection as `<manuscript_id>/<image_id>` where `image_id` == `picture_id` in the DB.",
    )
    parser.add_argument(
        "--mode",
        choices=("geniza", "pretrain"),
        default="geniza",
        help="Which DB table to use for fetching `image_path` and `xml_path` (default: geniza).",
    )
    parser.add_argument("--db-config", type=str, default=DB_CONFIG_PATH, help="Path to db_config.ini")
    parser.add_argument(
        "--random",
        action="store_true",
        help="Pick a random row among candidate rows (ignored when selecting an explicit --image).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (only used with --random).")
    parser.add_argument(
        "--no-require-counts",
        action="store_true",
        help="In geniza mode only: allow rows even if latent counts are missing/zero.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="any",
        help="In pretrain mode only: filter by dataset_split (train/val/test/any). Default: any.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_ROOT / "debug_tools" / "Visualization" / "outputs"),
        help="Output root directory (default: debug_tools/Visualization/outputs).",
    )
    parser.add_argument(
        "--skip-glyph-tighten-comparison",
        action="store_true",
        help=(
            "Skip viz_glyph_tighten_comparison.png (avoids two full extract_character_patches runs)."
        ),
    )

    args = parser.parse_args()

    if args.image:
        manuscript_id, picture_id = _parse_image_arg(args.image)
    else:
        manuscript_id = str(args.manuscript_id).strip()
        picture_id = None

    out_root = Path(args.out_dir).resolve()

    conn = _get_db_connection(args.db_config)
    try:
        if args.mode == "geniza":
            row = _select_row_from_geniza(
                conn,
                manuscript_id=manuscript_id,
                picture_id=picture_id,
                random_pick=bool(args.random),
                seed=args.seed,
                require_counts=not bool(args.no_require_counts),
            )
        else:
            row = _select_row_from_pretrain(
                conn,
                manuscript_id=manuscript_id,
                picture_id=picture_id,
                random_pick=bool(args.random),
                seed=args.seed,
                split=args.split,
            )
    finally:
        conn.close()

    image_path = Path(str(row["image_path"]))
    xml_path = Path(str(row["xml_path"]))
    if not image_path.exists():
        raise FileNotFoundError(f"image_path not found: {image_path}")
    if not xml_path.exists():
        # Fallback: sometimes DB has xml_path that may be missing locally; attempt discovery.
        xml_discovered = find_xml_path_pretrain(str(image_path)) or find_xml_path_local(str(image_path))
        if not xml_discovered:
            raise FileNotFoundError(f"xml_path not found and could not auto-discover for: {image_path}")
        xml_path = Path(xml_discovered)

    # Prefer filename stem; for geniza/pretrain picture_id often has an extension too.
    image_stem = image_path.stem
    manuscript_out_dir = out_root / manuscript_id
    image_out_dir = manuscript_out_dir / image_stem
    image_out_dir.mkdir(parents=True, exist_ok=True)

    (image_out_dir / "selected_row.json").write_text(
        json.dumps(row, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    patch_out = image_out_dir / "viz_patch_grid.jpg"
    glyph_grid_out = image_out_dir / "viz_glyph_grid.jpg"
    summary_out = image_out_dir / "summary.txt"

    # 1) Patch grid + patch count for summary.
    patch_images = _extract_patches_for_grid(image_path, xml_path)
    patch_config = {"tile_size": TILE_SIZE, "tile_stride": TILE_STRIDE, "max_tiles": MAX_TILES_EVAL}
    patch_count = len(patch_images)
    _save_patch_grid(patch_images, patch_out)

    # 2) Words + glyphs summary (and glyph grid input).
    (
        words_passed,
        per_letter_met,
        per_letter_chosen,
        chosen_patches,
        chosen_meta,
        per_letter_extracted,
    ) = _compute_words_and_glyphs_summary(image_path, xml_path)
    _print_and_save_summary(
        image_name=image_stem,
        words_passed=words_passed,
        per_letter_extracted=per_letter_extracted,
        per_letter_met=per_letter_met,
        per_letter_chosen=per_letter_chosen,
        patch_count=patch_count,
        patch_config=patch_config,
        summary_path=summary_out,
    )
    _save_glyph_grid(chosen_patches, chosen_meta, glyph_grid_out)

    glyph_tighten_comparison_out = image_out_dir / "viz_glyph_tighten_comparison.png"
    if not args.skip_glyph_tighten_comparison:
        tighten_grid = _build_glyph_tightening_comparison_png(
            image_path,
            xml_path,
            page_title=image_stem,
            char_patch_size=CHAR_PATCH_SIZE,
            max_chars=MAX_CHARS_PER_IMAGE,
        )
        tighten_grid.save(glyph_tighten_comparison_out, optimize=True)
        print(f"Glyph tighten comparison saved: {glyph_tighten_comparison_out}")

    # 3) Overlays on the full page (reuse existing Drafts visualization scripts).
    patches_overlay_out = image_out_dir / "viz_patches.jpg"
    words_overlay_out = image_out_dir / "viz_words.jpg"
    chars_overlay_out = image_out_dir / "viz_chars.jpg"

    patches_script = PROJECT_ROOT / "Drafts" / "Visualization" / "visualize_patches.py"
    words_script = PROJECT_ROOT / "Drafts" / "Visualization" / "visualize_words.py"
    chars_script = PROJECT_ROOT / "Drafts" / "Visualization" / "visualize_characters.py"

    if not patches_script.exists() or not words_script.exists() or not chars_script.exists():
        raise FileNotFoundError("Missing one or more Drafts/Visualization scripts.")

    print(f"Selected image: {image_path}")
    print(f"Selected xml:    {xml_path}")
    print(f"Output dir:      {image_out_dir}")

    # Patches overlay
    subprocess.run(
        [
            sys.executable,
            str(patches_script),
            str(image_path),
            str(xml_path),
            "-o",
            str(patches_overlay_out),
        ],
        check=True,
    )
    # Words overlay
    subprocess.run(
        [
            sys.executable,
            str(words_script),
            str(image_path),
            "--xml",
            str(xml_path),
            "-o",
            str(words_overlay_out),
        ],
        check=True,
    )
    # Glyph overlay
    subprocess.run(
        [
            sys.executable,
            str(chars_script),
            str(image_path),
            "--xml",
            str(xml_path),
            "-o",
            str(chars_overlay_out),
        ],
        check=True,
    )

    print("\nDone. Files saved:")
    print(f"  - {patches_overlay_out}")
    print(f"  - {words_overlay_out}")
    print(f"  - {chars_overlay_out}")
    print(f"  - {patch_out}")
    print(f"  - {glyph_grid_out}")
    if glyph_tighten_comparison_out.exists():
        print(f"  - {glyph_tighten_comparison_out}")
    print(f"  - {summary_out}")
    print(f"  - {image_out_dir / 'selected_row.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

