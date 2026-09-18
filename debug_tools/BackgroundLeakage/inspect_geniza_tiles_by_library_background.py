"""
Inspect Geniza projection tiles against known library background templates.

This debug script samples rows from geniza_image_information, joins
geniza_manuscript_shelfmark for normalized_library, extracts XML-guided tiles
with the current strict coverage settings, and saves:
- a CSV with tile coverage and background-template distance metrics
- per-image visual grids grouped by library

The goal is not automatic rejection. It is to verify whether selected projection
tiles are text-heavy or still dominated by library/background artifacts.

Example:
  python debug_tools/BackgroundLeakage/inspect_geniza_tiles_by_library_background.py \
      --limit 40 \
      --min-text-coverage 0.85 \
      --max-tiles 8
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


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "system.py").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utilities.VisionModule.xml_patch_extraction as xpe  # noqa: E402
from system import (  # noqa: E402
    CLUSTERING_DB_CONFIG_PATH,
    CLUSTERING_TILE_SIZE,
    CLUSTERING_TILE_STRIDE,
    GENIZA_IMAGE_INFORMATION_TABLE,
    XML_PATCH_MIN_TEXT_COVERAGE,
    CLUSTERING_MAX_TILES_EVAL,
)


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
    where = ["i.image_path IS NOT NULL", "i.image_path != ''"]
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
    # Very simple diagnostic: dark-ish pixels. This is not used for extraction.
    return float(np.mean(arr < 0.72))


def draw_page_overview(image: Image.Image, positions: List[Tuple[int, int]], patch_size: int) -> Image.Image:
    max_w = 1200
    scale = min(1.0, max_w / max(1, image.size[0]))
    overview = image.convert("RGB").resize(
        (int(image.size[0] * scale), int(image.size[1] * scale)),
        Image.Resampling.BICUBIC,
    )
    draw = ImageDraw.Draw(overview)
    for idx, (top, left) in enumerate(positions):
        box = [
            int(left * scale),
            int(top * scale),
            int((left + patch_size) * scale),
            int((top + patch_size) * scale),
        ]
        draw.rectangle(box, outline=(255, 30, 30), width=3)
        draw.text((box[0] + 4, box[1] + 4), str(idx), fill=(255, 30, 30))
    return overview


def save_grid(
    *,
    out_path: str,
    row: Dict[str, Any],
    image: Image.Image,
    patches: List[Image.Image],
    positions: List[Tuple[int, int]],
    coverages: List[float],
    distances: List[float],
    ink_fracs: List[float],
    background: Optional[Image.Image],
    patch_size: int,
) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = len(patches)
    overview = draw_page_overview(image, positions, patch_size)
    cols = max(2, min(4, n))
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(4 * cols, 4 * (rows + 1)))

    ax0 = fig.add_subplot(rows + 1, cols, 1)
    ax0.imshow(overview)
    ax0.axis("off")
    ax0.set_title("selected tile boxes", fontsize=9)

    if background is not None:
        ax_bg = fig.add_subplot(rows + 1, cols, 2)
        ax_bg.imshow(background.convert("RGB"))
        ax_bg.axis("off")
        ax_bg.set_title("library background", fontsize=9)

    for idx, patch in enumerate(patches):
        ax = fig.add_subplot(rows + 1, cols, cols + idx + 1)
        ax.imshow(patch.convert("RGB"))
        ax.axis("off")
        dist = distances[idx]
        dist_s = "nan" if not np.isfinite(dist) else f"{dist:.3f}"
        ax.set_title(
            f"tile {idx} cov={coverages[idx]:.3f}\nink={ink_fracs[idx]:.3f} bg_l1={dist_s}",
            fontsize=8,
        )

    fig.suptitle(
        f"{row.get('manuscript_id')} | {row.get('picture_id')}\n"
        f"library={row.get('normalized_library')}\n{row.get('image_path')}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-config", default=CLUSTERING_DB_CONFIG_PATH)
    parser.add_argument("--backgrounds-dir", default=str(PROJECT_ROOT / "Backgrounds"))
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "outputs" / "geniza_tile_background_inspection"))
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--manuscript-id", default="")
    parser.add_argument("--normalized-library", default="")
    parser.add_argument("--tile-size", type=int, default=CLUSTERING_TILE_SIZE)
    parser.add_argument("--stride", type=int, default=CLUSTERING_TILE_STRIDE)
    parser.add_argument("--max-tiles", type=int, default=CLUSTERING_MAX_TILES_EVAL)
    parser.add_argument("--min-text-coverage", type=float, default=XML_PATCH_MIN_TEXT_COVERAGE)
    parser.add_argument("--save-grids", type=int, default=40)
    args = parser.parse_args()

    xpe.XML_PATCH_MIN_TEXT_COVERAGE = float(args.min_text_coverage)
    xpe.XML_PATCH_FILTER_BY_TEXT_COVERAGE = True
    xpe.XML_PATCH_CONSTRAIN_TO_BOUNDS = True

    rows = load_rows(
        db_config=args.db_config,
        limit=args.limit,
        offset=args.offset,
        manuscript_id=args.manuscript_id.strip() or None,
        normalized_library=args.normalized_library.strip() or None,
    )
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "tile_background_metrics.csv")
    metrics: List[Dict[str, Any]] = []
    grids_saved = 0

    for row_idx, row in enumerate(rows):
        image_path = str(row.get("image_path") or "")
        xml_path = str(row.get("xml_path") or "").strip() or None
        normalized_library = row.get("normalized_library")
        if not image_path or not os.path.exists(image_path):
            print(f"[Skip] missing image: {image_path}")
            continue
        try:
            image = Image.open(image_path).convert("RGB")
            patches, _coords, _page_segments, metadata = xpe.extract_patches_with_xml(
                image=image,
                image_path=image_path,
                patch_size=args.tile_size,
                stride=args.stride,
                max_patches=args.max_tiles,
                xml_path=xml_path,
            )
        except Exception as exc:
            print(f"[Skip] extraction failed for {image_path}: {type(exc).__name__}: {exc}")
            continue

        positions = metadata.get("patch_positions") or []
        text_regions, xml_image_size = xpe.parse_alto_text_regions(xml_path) if xml_path else ([], image.size)
        if text_regions and xml_image_size and tuple(xml_image_size) != tuple(image.size):
            sx = image.size[0] / max(1, xml_image_size[0])
            sy = image.size[1] / max(1, xml_image_size[1])
            for region in text_regions:
                region["hpos"] *= sx
                region["vpos"] *= sy
                region["width"] *= sx
                region["height"] *= sy

        bg_path = find_background_path(args.backgrounds_dir, normalized_library)
        background = Image.open(bg_path).convert("RGB") if bg_path else None
        coverages = [
            xpe.calculate_patch_text_coverage(top, left, args.tile_size, text_regions)
            for top, left in positions
        ]
        distances = [background_distance(patch, background) for patch in patches]
        ink_fracs = [patch_ink_fraction(patch) for patch in patches]

        print(
            f"[Image {row_idx + 1}/{len(rows)}] ms={row.get('manuscript_id')} "
            f"lib={normalized_library!r} patches={len(patches)} "
            f"coverage_mean={np.mean(coverages) if coverages else float('nan'):.3f} "
            f"bg_l1_mean={np.nanmean(distances) if distances else float('nan'):.3f}"
        )

        for tile_idx, (pos, cov, dist, ink) in enumerate(zip(positions, coverages, distances, ink_fracs)):
            metrics.append({
                "manuscript_id": row.get("manuscript_id"),
                "picture_id": row.get("picture_id"),
                "image_path": image_path,
                "xml_path": xml_path,
                "normalized_library": normalized_library,
                "background_path": bg_path or "",
                "tile_index": tile_idx,
                "top": pos[0],
                "left": pos[1],
                "text_coverage": cov,
                "ink_fraction": ink,
                "background_l1_distance": dist,
                "num_patches": len(patches),
            })

        if grids_saved < args.save_grids and patches:
            lib_dir = safe_filename(normalized_library or "unknown_library")
            out_path = os.path.join(
                args.out_dir,
                "grids",
                lib_dir,
                f"{safe_filename(row.get('manuscript_id'))}_{safe_filename(row.get('picture_id'))}.png",
            )
            save_grid(
                out_path=out_path,
                row=row,
                image=image,
                patches=patches,
                positions=positions,
                coverages=coverages,
                distances=distances,
                ink_fracs=ink_fracs,
                background=background,
                patch_size=args.tile_size,
            )
            grids_saved += 1

    if metrics:
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()))
            writer.writeheader()
            writer.writerows(metrics)
    else:
        Path(csv_path).write_text("", encoding="utf-8")

    print(f"[Done] rows={len(rows)} tile_metrics={len(metrics)} grids_saved={grids_saved}")
    print(f"  csv: {csv_path}")
    print(f"  grids: {os.path.join(args.out_dir, 'grids')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
