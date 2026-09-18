"""
Inspect Geniza glyph crops against known library background templates.

This debug script samples rows from geniza_image_information, joins
geniza_manuscript_shelfmark for normalized_library, extracts glyph crops from
ALTO XML, optionally keeps only middle letters inside words, and saves:
- a CSV with glyph/background diagnostic metrics
- per-image visual grids grouped by library

Example:
  python Debugs/BackgroundLeakage/inspect_geniza_glyphs_by_library_background.py \
      --limit 40 \
      --max-glyphs 24 \
      --only-middle-glyphs
"""

from __future__ import annotations

import argparse
import configparser
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib-cache")

import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "system.py").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.glyph_branch import GlyphHardQualityFilter  # noqa: E402
from system import (  # noqa: E402
    CHAR_PATCH_SIZE,
    CLUSTERING_DB_CONFIG_PATH,
    GENIZA_IMAGE_INFORMATION_TABLE,
    GLYPH_MIN_WORD_LENGTH_FOR_MIDDLE,
    GLYPH_ONLY_MIDDLE_LETTERS,
    MAX_CHARS_PER_IMAGE,
)
from utilities.VisionModule.xml_character_extraction import extract_character_patches  # noqa: E402


SHELFMARK_TABLE = "geniza_manuscript_shelfmark"


def get_db_connection(db_config_path: str):
    cfg = db_config_path
    if not os.path.isabs(cfg):
        cfg = str(PROJECT_ROOT / cfg)
    config = configparser.ConfigParser()
    config.read(cfg)
    section = "postgresql" if "postgresql" in config else "database"
    db = config[section]
    import psycopg2

    return psycopg2.connect(
        host=db["host"],
        database=db["database"],
        user=db["user"],
        password=db["password"],
        port=db.get("port", 5432),
    )


def load_rows(
    *,
    db_config: str,
    limit: int,
    offset: int,
    manuscript_id: Optional[str],
    normalized_library: Optional[str],
) -> List[Dict[str, Any]]:
    where = ["i.image_path IS NOT NULL", "i.image_path != ''", "i.xml_path IS NOT NULL", "i.xml_path != ''"]
    params: List[Any] = []
    if manuscript_id:
        where.append("i.manuscript_id = %s")
        params.append(manuscript_id)
    if normalized_library:
        where.append("s.normalized_library = %s")
        params.append(normalized_library)
    params.extend([limit, offset])

    sql = f"""
        SELECT
            i.manuscript_id,
            i.picture_id,
            i.parent_directory,
            i.page_number,
            i.image_path,
            i.xml_path,
            s.normalized_library
        FROM {GENIZA_IMAGE_INFORMATION_TABLE} i
        LEFT JOIN {SHELFMARK_TABLE} s
          ON i.manuscript_id::text = s.manuscript_id::text
        WHERE {' AND '.join(where)}
        ORDER BY i.manuscript_id, i.parent_directory, i.picture_id
        LIMIT %s OFFSET %s
    """
    conn = get_db_connection(db_config)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def safe_filename(text: str, max_len: int = 120) -> str:
    keep = []
    for ch in str(text):
        keep.append(ch if ch.isalnum() or ch in "._-" else "_")
    out = "".join(keep).strip("_")
    return (out[:max_len] or "unknown")


def find_background_path(backgrounds_dir: str, normalized_library: Optional[str]) -> Optional[str]:
    if not normalized_library:
        return None
    direct = os.path.join(backgrounds_dir, f"{normalized_library}.png")
    if os.path.exists(direct):
        return direct
    norm_target = safe_filename(normalized_library).lower()
    for file in os.listdir(backgrounds_dir):
        if not file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        stem = safe_filename(os.path.splitext(file)[0]).lower()
        if stem == norm_target:
            return os.path.join(backgrounds_dir, file)
    return None


def image_array(image: Image.Image, size: Tuple[int, int]) -> np.ndarray:
    return np.asarray(image.convert("RGB").resize(size, Image.Resampling.BICUBIC), dtype=np.float32) / 255.0


def background_distance(patch: Image.Image, background: Optional[Image.Image]) -> float:
    if background is None:
        return float("nan")
    size = patch.size
    p = image_array(patch, size)
    b = image_array(background, size)
    return float(np.mean(np.abs(p - b)))


def patch_ink_fraction(patch: Image.Image) -> float:
    arr = np.asarray(patch.convert("L"), dtype=np.float32) / 255.0
    return float(np.mean(arr < 0.72))


def draw_page_overview(image: Image.Image, metadata_list: List[Dict[str, Any]]) -> Image.Image:
    max_w = 1200
    scale = min(1.0, max_w / max(1, image.size[0]))
    overview = image.convert("RGB").resize(
        (int(image.size[0] * scale), int(image.size[1] * scale)),
        Image.Resampling.BICUBIC,
    )
    draw = ImageDraw.Draw(overview)
    for idx, meta in enumerate(metadata_list):
        left = int(float(meta.get("hpos", 0.0)) * scale)
        top = int(float(meta.get("vpos", 0.0)) * scale)
        right = int((float(meta.get("hpos", 0.0)) + float(meta.get("width", 0.0))) * scale)
        bottom = int((float(meta.get("vpos", 0.0)) + float(meta.get("height", 0.0))) * scale)
        draw.rectangle([left, top, right, bottom], outline=(255, 30, 30), width=2)
        draw.text((left + 2, top + 2), str(idx), fill=(255, 30, 30))
    return overview


def save_grid(
    *,
    out_path: str,
    row: Dict[str, Any],
    image: Image.Image,
    patches: List[Image.Image],
    metadata_list: List[Dict[str, Any]],
    distances: List[float],
    ink_fracs: List[float],
    background: Optional[Image.Image],
) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = len(patches)
    overview = draw_page_overview(image, metadata_list)
    cols = max(2, min(4, n))
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(4 * cols, 4 * (rows + 1)))

    ax0 = fig.add_subplot(rows + 1, cols, 1)
    ax0.imshow(overview)
    ax0.axis("off")
    ax0.set_title("selected glyph boxes", fontsize=9)

    if background is not None:
        ax_bg = fig.add_subplot(rows + 1, cols, 2)
        ax_bg.imshow(background.convert("RGB"))
        ax_bg.axis("off")
        ax_bg.set_title("library background", fontsize=9)

    for idx, patch in enumerate(patches):
        ax = fig.add_subplot(rows + 1, cols, cols + idx + 1)
        ax.imshow(patch.convert("RGB"))
        ax.axis("off")
        meta = metadata_list[idx]
        dist = distances[idx]
        dist_s = "nan" if not np.isfinite(dist) else f"{dist:.3f}"
        pos = f"{int(meta.get('glyph_index_in_word', -1)) + 1}/{int(meta.get('word_length', 0))}"
        middle = "mid" if bool(meta.get("is_word_middle_glyph", False)) else "edge"
        ax.set_title(
            f"{meta.get('char', '?')} gc={float(meta.get('gc', 0.0)):.2f}\n"
            f"{middle} pos={pos} ink={ink_fracs[idx]:.3f}\n"
            f"bg_l1={dist_s}",
            fontsize=8,
        )

    fig.suptitle(
        f"{row.get('manuscript_id')} | {row.get('picture_id')}\n"
        f"library={row.get('normalized_library')} only_middle={bool(metadata_list and metadata_list[0].get('selection_only_middle_glyphs', False))}\n"
        f"{row.get('image_path')}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-config", default=CLUSTERING_DB_CONFIG_PATH)
    parser.add_argument("--backgrounds-dir", default=str(PROJECT_ROOT / "Backgrounds"))
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "outputs" / "geniza_glyph_background_inspection"))
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--manuscript-id", default="")
    parser.add_argument("--normalized-library", default="")
    parser.add_argument("--char-patch-size", type=int, default=CHAR_PATCH_SIZE)
    parser.add_argument("--max-glyphs", type=int, default=MAX_CHARS_PER_IMAGE)
    parser.add_argument("--only-middle-glyphs", action="store_true", default=GLYPH_ONLY_MIDDLE_LETTERS)
    parser.add_argument("--min-word-length-for-middle", type=int, default=GLYPH_MIN_WORD_LENGTH_FOR_MIDDLE)
    parser.add_argument("--save-grids", type=int, default=40)
    args = parser.parse_args()

    rows = load_rows(
        db_config=args.db_config,
        limit=args.limit,
        offset=args.offset,
        manuscript_id=args.manuscript_id.strip() or None,
        normalized_library=args.normalized_library.strip() or None,
    )
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "glyph_background_metrics.csv")
    metrics: List[Dict[str, Any]] = []
    grids_saved = 0
    quality_filter = GlyphHardQualityFilter()

    for row_idx, row in enumerate(rows):
        image_path = str(row.get("image_path") or "")
        xml_path = str(row.get("xml_path") or "").strip() or None
        normalized_library = row.get("normalized_library")
        if not image_path or not os.path.exists(image_path):
            print(f"[Skip] missing image: {image_path}")
            continue
        try:
            image = Image.open(image_path).convert("RGB")
            glyph_patches, glyph_metadata = extract_character_patches(
                image=image,
                image_path=image_path,
                char_patch_size=args.char_patch_size,
                max_chars=None,
                xml_path=xml_path,
                only_middle_glyphs=bool(args.only_middle_glyphs),
                min_word_length_for_middle=int(args.min_word_length_for_middle),
            )
        except Exception as exc:
            print(f"[Skip] extraction failed for {image_path}: {type(exc).__name__}: {exc}")
            continue

        if not glyph_patches:
            print(f"[Skip] no glyphs: {image_path}")
            continue

        patch_tensors = [transforms.ToTensor()(p) for p in glyph_patches]
        pil_by_meta_id = {id(meta): pil for meta, pil in zip(glyph_metadata, glyph_patches)}
        filtered_patches, filtered_metadata = quality_filter.filter_glyphs(patch_tensors, glyph_metadata)
        _sampled_tensors, sampled_metadata = quality_filter.sample_diverse_glyphs(
            filtered_patches,
            filtered_metadata,
            args.max_glyphs,
        )
        sampled_patches = [pil_by_meta_id[id(meta)] for meta in sampled_metadata if id(meta) in pil_by_meta_id]
        sampled_metadata = [meta for meta in sampled_metadata if id(meta) in pil_by_meta_id]

        bg_path = find_background_path(args.backgrounds_dir, normalized_library)
        background = Image.open(bg_path).convert("RGB") if bg_path else None
        distances = [background_distance(patch, background) for patch in sampled_patches]
        ink_fracs = [patch_ink_fraction(patch) for patch in sampled_patches]

        middle_count = sum(bool(meta.get("is_word_middle_glyph", False)) for meta in sampled_metadata)
        edge_count = len(sampled_metadata) - middle_count
        print(
            f"[Image {row_idx + 1}/{len(rows)}] ms={row.get('manuscript_id')} "
            f"lib={normalized_library!r} glyphs={len(sampled_patches)} "
            f"middle={middle_count} edge={edge_count} "
            f"bg_l1_mean={np.nanmean(distances) if distances else float('nan'):.3f}"
        )

        for glyph_idx, (meta, dist, ink) in enumerate(zip(sampled_metadata, distances, ink_fracs)):
            metrics.append({
                "manuscript_id": row.get("manuscript_id"),
                "picture_id": row.get("picture_id"),
                "image_path": image_path,
                "xml_path": xml_path,
                "normalized_library": normalized_library,
                "background_path": bg_path or "",
                "glyph_index": glyph_idx,
                "char": meta.get("char"),
                "gc": meta.get("gc"),
                "hpos": meta.get("hpos"),
                "vpos": meta.get("vpos"),
                "width": meta.get("width"),
                "height": meta.get("height"),
                "ink_fraction": ink,
                "background_l1_distance": dist,
                "word": meta.get("word"),
                "word_length": meta.get("word_length"),
                "glyph_index_in_word": meta.get("glyph_index_in_word"),
                "is_word_edge_glyph": meta.get("is_word_edge_glyph"),
                "is_word_middle_glyph": meta.get("is_word_middle_glyph"),
                "selection_only_middle_glyphs": meta.get("selection_only_middle_glyphs"),
                "selection_min_word_length_for_middle": meta.get("selection_min_word_length_for_middle"),
                "num_glyphs": len(sampled_patches),
            })

        if grids_saved < args.save_grids and sampled_patches:
            lib_dir = safe_filename(normalized_library or "unknown_library")
            suffix = "middle" if args.only_middle_glyphs else "all"
            out_path = os.path.join(
                args.out_dir,
                "grids",
                lib_dir,
                f"{safe_filename(row.get('manuscript_id'))}_{safe_filename(row.get('picture_id'))}_{suffix}.png",
            )
            save_grid(
                out_path=out_path,
                row=row,
                image=image,
                patches=sampled_patches,
                metadata_list=sampled_metadata,
                distances=distances,
                ink_fracs=ink_fracs,
                background=background,
            )
            grids_saved += 1

    if metrics:
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()))
            writer.writeheader()
            writer.writerows(metrics)
    else:
        Path(csv_path).write_text("", encoding="utf-8")

    print(f"[Done] rows={len(rows)} glyph_metrics={len(metrics)} grids_saved={grids_saved}")
    print(f"  csv: {csv_path}")
    print(f"  grids: {os.path.join(args.out_dir, 'grids')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
