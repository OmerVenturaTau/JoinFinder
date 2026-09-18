#!/usr/bin/env python3
"""
Randomly select a Geniza or training image from the DB and save tile/glyph grids.

Examples:
  python debug_tools/Visualization/random_tiles_glyphs_grid.py --source geniza

  python debug_tools/Visualization/random_tiles_glyphs_grid.py \
      --source training \
      --stage stage1 \
      --split train
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib-cache")


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "system.py").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from system import (  # noqa: E402
    BASE_DIR,
    CHAR_PATCH_SIZE,
    DB_CONFIG_PATH,
    DISABLE_PIL_LIMIT,
    GENIZA_IMAGE_LATENTS_TABLE,
    GLYPH_MIN_WORD_LENGTH_FOR_MIDDLE,
    GLYPH_ONLY_MIDDLE_LETTERS,
    HEBREW_ALPHABET,
    MAX_CHARS_PER_IMAGE,
    MAX_TILES_EVAL,
    MAX_TILES_TRAIN,
    STAGE1_TABLE_NAME,
    STAGE2_TABLE_NAME,
    TILE_SIZE,
    TILE_STRIDE,
)


PREFERRED_COLUMNS = [
    "id",
    "manuscript_id",
    "picture_id",
    "page_number",
    "page_id",
    "parent_directory",
    "dataset_split",
    "image_path",
    "xml_path",
    "num_visual_patches",
    "num_glyphs",
    "num_words",
]


def _load_db_config(db_config_path: str) -> configparser.SectionProxy:
    cfg_path = Path(db_config_path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"DB config not found: {cfg_path}")

    config = configparser.ConfigParser()
    config.read(cfg_path)
    section = "postgresql" if "postgresql" in config else "database"
    if section not in config:
        raise ValueError(f"Missing [postgresql] or [database] section in {cfg_path}")
    return config[section]


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


def _quote_ident(part: str) -> str:
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", part):
        raise ValueError(f"Unsafe SQL identifier component: {part!r}")
    return '"' + part.replace('"', '""') + '"'


def _table_identifier(table_ref: str) -> str:
    parts = [part.strip() for part in table_ref.split(".") if part.strip()]
    if not parts or len(parts) > 2:
        raise ValueError(f"Invalid table name: {table_ref!r}")
    return ".".join(_quote_ident(part) for part in parts)


def _fetch_columns(conn, table_ref: str) -> List[str]:
    query = f"SELECT * FROM {_table_identifier(table_ref)} LIMIT 0"
    with conn.cursor() as cur:
        cur.execute(query)
        return [desc.name for desc in cur.description]


def _nonnull_text_clause(column: str) -> str:
    ident = _quote_ident(column)
    return f"{ident} IS NOT NULL AND {ident}::text <> ''"


def _build_where(
    *,
    source: str,
    columns: Sequence[str],
    split: str,
    require_counts: bool,
    require_xml: bool,
) -> Tuple[str, List[Any]]:
    clauses: List[str] = [_nonnull_text_clause("image_path")]
    params: List[Any] = []

    if require_xml and "xml_path" in columns:
        clauses.append(_nonnull_text_clause("xml_path"))

    if source == "geniza" and require_counts:
        if "num_visual_patches" in columns:
            clauses.append(f"COALESCE({_quote_ident('num_visual_patches')}, 0) > 0")
        if "num_glyphs" in columns:
            clauses.append(f"COALESCE({_quote_ident('num_glyphs')}, 0) > 0")

    if source == "training" and split != "any" and "dataset_split" in columns:
        clauses.append(f"LOWER({_quote_ident('dataset_split')}::text) = %s")
        params.append(split)

    return " AND ".join(clauses), params


def _order_by(columns: Sequence[str]) -> str:
    order_cols = [c for c in ("manuscript_id", "parent_directory", "page_number", "picture_id", "id", "image_path") if c in columns]
    if not order_cols:
        return _quote_ident("image_path")
    return ", ".join(_quote_ident(c) for c in order_cols)


def _count_candidates(conn, table_ref: str, where_sql: str, params: Sequence[Any]) -> int:
    query = f"SELECT COUNT(*) FROM {_table_identifier(table_ref)} WHERE {where_sql}"
    with conn.cursor() as cur:
        cur.execute(query, tuple(params))
        return int(cur.fetchone()[0])


def _select_random_row(
    conn,
    *,
    table_ref: str,
    source: str,
    split: str,
    require_counts: bool,
    require_xml: bool,
    rng: random.Random,
) -> Tuple[Dict[str, Any], int]:
    columns = _fetch_columns(conn, table_ref)
    if "image_path" not in columns:
        raise RuntimeError(f"Table {table_ref!r} has no image_path column.")

    selected_columns = [c for c in PREFERRED_COLUMNS if c in columns]
    where_sql, params = _build_where(
        source=source,
        columns=columns,
        split=split,
        require_counts=require_counts,
        require_xml=require_xml,
    )
    candidate_count = _count_candidates(conn, table_ref, where_sql, params)
    if candidate_count <= 0:
        raise RuntimeError(f"No candidate rows found in {table_ref!r} for source={source!r}, split={split!r}.")

    offset = rng.randrange(candidate_count)
    query = (
        f"SELECT {', '.join(_quote_ident(c) for c in selected_columns)} "
        f"FROM {_table_identifier(table_ref)} "
        f"WHERE {where_sql} "
        f"ORDER BY {_order_by(columns)} "
        "LIMIT 1 OFFSET %s"
    )
    with conn.cursor() as cur:
        cur.execute(query, tuple(params) + (offset,))
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("Random DB offset returned no row.")
        names = [desc.name for desc in cur.description]
        return dict(zip(names, row)), candidate_count


def _safe_filename(text: str, max_len: int = 120) -> str:
    out = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text)).strip("_")
    return (out[:max_len] or "unknown")


def _resolve_image_path(row: Dict[str, Any]) -> Optional[str]:
    image_path = row.get("image_path")
    if image_path:
        return str(image_path)
    required = ("manuscript_id", "parent_directory", "picture_id")
    if all(row.get(k) for k in required):
        return str(Path(BASE_DIR) / str(row["manuscript_id"]) / str(row["parent_directory"]) / str(row["picture_id"]))
    return None


def _resolve_xml_path(image_path: str, row: Dict[str, Any]) -> Optional[str]:
    xml_path = row.get("xml_path")
    if xml_path and Path(str(xml_path)).exists():
        return str(xml_path)
    from utilities.VisionModule.xml_patch_extraction import find_xml_path
    from utilities.xml_loader import find_xml_path_pretrain

    return find_xml_path_pretrain(image_path) or find_xml_path(image_path)


def _shorten(value: Any, limit: int = 32) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _load_font(size: int) -> ImageFont.ImageFont:
    from PIL import ImageFont

    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for font_path in font_paths:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def _draw_text_fit(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    *,
    max_width: int,
    font: ImageFont.ImageFont,
    fill: Tuple[int, int, int],
) -> None:
    shortened = text
    while shortened:
        bbox = draw.textbbox((0, 0), shortened, font=font)
        if bbox[2] - bbox[0] <= max_width:
            break
        shortened = shortened[:-4].rstrip() + "..."
    draw.text(xy, shortened or "...", font=font, fill=fill)


def _save_grid(
    images: Sequence[Image.Image],
    titles: Sequence[str],
    output_path: Path,
    *,
    cols: int,
    suptitle: str,
    cell_inches: float,
) -> None:
    from PIL import Image, ImageDraw, ImageOps

    output_path.parent.mkdir(parents=True, exist_ok=True)
    title_font = _load_font(16)
    cell_title_font = _load_font(12)
    background = (255, 255, 255)

    if not images:
        canvas = Image.new("RGB", (720, 240), background)
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 24), suptitle, font=title_font, fill=(0, 0, 0))
        draw.text((24, 104), "No images extracted", font=title_font, fill=(0, 0, 0))
        canvas.save(output_path)
        return

    cols = max(1, min(cols, len(images)))
    rows = (len(images) + cols - 1) // cols

    # Keep PIL output compact but readable. cell_inches is retained in the API
    # so tile and glyph grids can request different visual scales.
    cell_px = max(72, int(cell_inches * 100))
    title_h = 30
    pad = 10
    header_h = 42
    canvas_w = cols * (cell_px + pad) + pad
    canvas_h = header_h + rows * (cell_px + title_h + pad) + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), background)
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 10), suptitle, font=title_font, fill=(0, 0, 0))

    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = pad + col * (cell_px + pad)
        y = header_h + row * (cell_px + title_h + pad)

        tile = ImageOps.contain(image.convert("RGB"), (cell_px, cell_px), Image.Resampling.LANCZOS)
        framed = Image.new("RGB", (cell_px, cell_px), (248, 248, 248))
        framed.paste(tile, ((cell_px - tile.width) // 2, (cell_px - tile.height) // 2))
        canvas.paste(framed, (x, y))
        draw.rectangle([x, y, x + cell_px - 1, y + cell_px - 1], outline=(210, 210, 210))
        _draw_text_fit(
            draw,
            (x, y + cell_px + 6),
            _shorten(titles[idx] if idx < len(titles) else str(idx + 1), 40),
            max_width=cell_px,
            font=cell_title_font,
            fill=(25, 25, 25),
        )

    canvas.save(output_path)


def _extract_tiles(image: Image.Image, image_path: str, xml_path: str, max_tiles: int) -> Tuple[List[Image.Image], Dict[str, Any]]:
    from utilities.VisionModule.xml_patch_extraction import extract_patches_with_xml

    tiles, coords, page_segments, metadata = extract_patches_with_xml(
        image=image,
        image_path=image_path,
        patch_size=TILE_SIZE,
        stride=TILE_STRIDE,
        max_patches=max_tiles,
        xml_path=xml_path,
    )
    metadata = dict(metadata)
    metadata["coords"] = coords.tolist() if hasattr(coords, "tolist") else []
    metadata["page_segments"] = page_segments.tolist() if hasattr(page_segments, "tolist") else []
    return tiles, metadata


def _extract_glyphs(
    image: Image.Image,
    image_path: str,
    xml_path: str,
    max_glyphs: int,
) -> Tuple[List[Image.Image], List[Dict[str, Any]], Dict[str, int]]:
    from torchvision import transforms

    from models.glyph_branch import GlyphHardQualityFilter
    from utilities.VisionModule.xml_character_extraction import extract_character_patches

    raw_patches, raw_meta = extract_character_patches(
        image=image,
        image_path=image_path,
        char_patch_size=CHAR_PATCH_SIZE,
        max_chars=None,
        xml_path=xml_path,
        only_middle_glyphs=GLYPH_ONLY_MIDDLE_LETTERS,
        min_word_length_for_middle=GLYPH_MIN_WORD_LENGTH_FOR_MIDDLE,
    )
    if not raw_patches:
        return [], [], {"raw": 0, "filtered": 0, "sampled": 0}

    patch_tensors = [transforms.ToTensor()(patch) for patch in raw_patches]
    pil_by_tensor_id = {id(tensor): patch for tensor, patch in zip(patch_tensors, raw_patches)}

    quality_filter = GlyphHardQualityFilter()
    filtered_tensors, filtered_meta = quality_filter.filter_glyphs(patch_tensors, raw_meta)
    sampled_tensors, sampled_meta = quality_filter.sample_diverse_glyphs(
        filtered_tensors,
        filtered_meta,
        max_glyphs=max_glyphs,
    )

    sampled_pils: List[Image.Image] = []
    for tensor in sampled_tensors:
        pil = pil_by_tensor_id.get(id(tensor))
        if pil is None:
            pil = transforms.ToPILImage()(tensor.detach().cpu().clamp(0.0, 1.0))
        sampled_pils.append(pil)

    return sampled_pils, sampled_meta, {
        "raw": len(raw_patches),
        "filtered": len(filtered_tensors),
        "sampled": len(sampled_pils),
    }


def _sorted_glyph_items(
    glyphs: Sequence[Image.Image],
    metadata: Sequence[Dict[str, Any]],
) -> Tuple[List[Image.Image], List[Dict[str, Any]]]:
    order = {char: idx for idx, char in enumerate(HEBREW_ALPHABET)}
    indexed = list(enumerate(zip(glyphs, metadata)))
    indexed.sort(
        key=lambda item: (
            order.get(str(item[1][1].get("char") or "").strip(), 999),
            -float(item[1][1].get("gc") or 0.0),
            item[0],
        )
    )
    sorted_pairs = [pair for _, pair in indexed]
    return [pair[0] for pair in sorted_pairs], [pair[1] for pair in sorted_pairs]


def _glyph_titles(metadata: Iterable[Dict[str, Any]]) -> List[str]:
    titles = []
    for idx, meta in enumerate(metadata, start=1):
        char = str(meta.get("char") or "?").strip() or "?"
        gc = meta.get("gc")
        gc_text = f"{float(gc):.2f}" if gc is not None else "na"
        titles.append(f"{idx}: {char} gc={gc_text}")
    return titles


def _try_extract_from_row(
    row: Dict[str, Any],
    *,
    max_tiles: int,
    max_glyphs: int,
) -> Tuple[str, str, Image.Image, List[Image.Image], Dict[str, Any], List[Image.Image], List[Dict[str, Any]], Dict[str, int]]:
    from PIL import Image

    image_path = _resolve_image_path(row)
    if not image_path:
        raise RuntimeError("Selected row has no resolvable image path.")
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image path not found: {image_path}")

    xml_path = _resolve_xml_path(image_path, row)
    if not xml_path or not Path(xml_path).exists():
        raise FileNotFoundError(f"XML path not found for image: {image_path}")

    image = Image.open(image_path).convert("RGB")
    tiles, tile_metadata = _extract_tiles(image, image_path, xml_path, max_tiles)
    glyphs, glyph_meta, glyph_counts = _extract_glyphs(image, image_path, xml_path, max_glyphs)
    return image_path, xml_path, image, tiles, tile_metadata, glyphs, glyph_meta, glyph_counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=("geniza", "training"), default="geniza")
    parser.add_argument("--db-config", default=DB_CONFIG_PATH)
    parser.add_argument("--table", default=None, help="DB table override. Defaults to geniza latents or the selected training stage table.")
    parser.add_argument("--stage", choices=("stage1", "stage2"), default="stage1", help="Training table preset when --source training.")
    parser.add_argument("--split", choices=("train", "val", "test", "any"), default="train", help="Training split filter if dataset_split exists.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-db-attempts", type=int, default=25)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--max-glyphs", type=int, default=MAX_CHARS_PER_IMAGE)
    parser.add_argument("--tile-cols", type=int, default=4)
    parser.add_argument("--glyph-cols", type=int, default=12)
    parser.add_argument("--allow-empty", action="store_true", help="Save grids even if tile or glyph extraction returns zero items.")
    parser.add_argument("--no-require-geniza-counts", action="store_true", help="Do not require num_visual_patches/num_glyphs > 0 for Geniza rows.")
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "debug_tools" / "Visualization" / "outputs" / "random_tiles_glyphs_grid"),
    )
    args = parser.parse_args()

    if DISABLE_PIL_LIMIT:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None

    rng = random.Random(args.seed)
    table_ref = args.table
    if table_ref is None:
        table_ref = GENIZA_IMAGE_LATENTS_TABLE if args.source == "geniza" else (STAGE2_TABLE_NAME if args.stage == "stage2" else STAGE1_TABLE_NAME)

    max_tiles = args.max_tiles
    if max_tiles is None:
        max_tiles = MAX_TILES_TRAIN if args.source == "training" and args.split == "train" else MAX_TILES_EVAL

    conn = _get_db_connection(args.db_config)
    last_error: Optional[BaseException] = None
    selected: Optional[Tuple[Dict[str, Any], int, str, str, Image.Image, List[Image.Image], Dict[str, Any], List[Image.Image], List[Dict[str, Any]], Dict[str, int]]] = None

    try:
        for attempt in range(1, max(1, args.max_db_attempts) + 1):
            row, candidate_count = _select_random_row(
                conn,
                table_ref=table_ref,
                source=args.source,
                split=args.split,
                require_counts=not args.no_require_geniza_counts,
                require_xml=True,
                rng=rng,
            )
            try:
                image_path, xml_path, image, tiles, tile_metadata, glyphs, glyph_meta, glyph_counts = _try_extract_from_row(
                    row,
                    max_tiles=max_tiles,
                    max_glyphs=args.max_glyphs,
                )
                if args.allow_empty or (tiles and glyphs):
                    selected = (
                        row,
                        candidate_count,
                        image_path,
                        xml_path,
                        image,
                        tiles,
                        tile_metadata,
                        glyphs,
                        glyph_meta,
                        glyph_counts,
                    )
                    break
                raise RuntimeError(f"Extraction returned tiles={len(tiles)}, glyphs={len(glyphs)}")
            except Exception as exc:
                last_error = exc
                print(f"[attempt {attempt}] Skipping row: {exc}")
    finally:
        conn.close()

    if selected is None:
        raise RuntimeError(f"Could not find a usable random row after {args.max_db_attempts} attempts. Last error: {last_error}")

    (
        row,
        candidate_count,
        image_path,
        xml_path,
        image,
        tiles,
        tile_metadata,
        glyphs,
        glyph_meta,
        glyph_counts,
    ) = selected

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_stem = _safe_filename(Path(image_path).stem)
    output_dir = Path(args.out_dir) / f"{args.source}_{image_stem}_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_titles = [f"{idx + 1}" for idx in range(len(tiles))]
    sorted_glyphs, sorted_glyph_meta = _sorted_glyph_items(glyphs, glyph_meta)

    _save_grid(
        tiles,
        tile_titles,
        output_dir / "tiles_grid.png",
        cols=args.tile_cols,
        suptitle=f"Tiles ({len(tiles)})",
        cell_inches=3.2,
    )
    _save_grid(
        sorted_glyphs,
        _glyph_titles(sorted_glyph_meta),
        output_dir / "glyphs_grid.png",
        cols=args.glyph_cols,
        suptitle=f"Glyphs ({len(sorted_glyphs)})",
        cell_inches=1.35,
    )

    selected_row_path = output_dir / "selected_row.json"
    selected_row_path.write_text(json.dumps(row, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    summary = {
        "source": args.source,
        "table": table_ref,
        "candidate_count": candidate_count,
        "training_stage": args.stage if args.source == "training" else None,
        "training_split": args.split if args.source == "training" else None,
        "image_path": image_path,
        "xml_path": xml_path,
        "image_size": image.size,
        "tile_config": {
            "tile_size": TILE_SIZE,
            "tile_stride": TILE_STRIDE,
            "max_tiles": max_tiles,
        },
        "glyph_config": {
            "char_patch_size": CHAR_PATCH_SIZE,
            "max_glyphs": args.max_glyphs,
            "only_middle_glyphs": GLYPH_ONLY_MIDDLE_LETTERS,
            "min_word_length_for_middle": GLYPH_MIN_WORD_LENGTH_FOR_MIDDLE,
        },
        "counts": {
            "tiles_saved": len(tiles),
            "glyphs_saved": len(sorted_glyphs),
            "glyphs_raw": glyph_counts["raw"],
            "glyphs_after_quality_filter": glyph_counts["filtered"],
        },
        "tile_metadata": tile_metadata,
        "glyph_metadata": sorted_glyph_meta,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"Selected {args.source} row from {table_ref} ({candidate_count} candidates)")
    print(f"Image: {image_path}")
    print(f"XML:   {xml_path}")
    print(f"Saved: {output_dir / 'tiles_grid.png'}")
    print(f"Saved: {output_dir / 'glyphs_grid.png'}")
    print(f"Saved: {selected_row_path}")
    print(f"Saved: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
