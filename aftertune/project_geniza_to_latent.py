#!/usr/bin/env python3
"""
Project geniza manuscript images to the model latent space and save per-image rows.

1. Loads a single checkpoint (pretrain or finetune model).
2. Reads image list from geniza_image_information (DB only; no file).
3. For each row, loads the image from image_path and runs the model to get a latent vector.
   XML (for patch/char/word extraction): uses xml_path from the table when present and non-empty.
   If xml_path is NULL or empty, ManuscriptDataset falls back to find_xml_path_pretrain(image_path)
   (filesystem convention). If no XML is found, projection uses visual patches only (no char/word).
4. Saves one row per image to geniza_image_latents with columns:
   manuscript_id, picture_id, is_color, parent_directory, page_number, page_id,
   image_path, xml_path, latent_vector, latent_vector_search.
   Latents are computed from both visual patches and character (glyph) features when XML is available.
5. PCA / latent_vector_search should be recomputed in a separate script once all latent_vector rows exist.

Run geniza_top_neighbors.py afterwards to build the table of top-K neighbours
per image (excluding same manuscript).

To debug projection (patches/letters for one image), run:
  python aftertune/debug_projection_one_image.py --index 0

Table names and paths are in system.py (GENIZA_IMAGE_INFORMATION_TABLE,
GENIZA_IMAGE_LATENTS_TABLE, CLUSTERING_IMAGES_ROOT).
"""

import os
import sys
import argparse
import configparser
import logging
import re
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import psycopg2

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models import MultiModal
from train.dataset import ManuscriptDataset, tile_collate_with_padding
from system import (
    DB_CONFIG_PATH,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    LATENT_DIM,
    USE_VISUAL_MOD,
    USE_CHAR_MOD,
    USE_WORD_MOD,
    MAX_SELECTED_MANUSCRIPTS,
    CLUSTERING_DB_CONFIG_PATH,
    CLUSTERING_IMAGES_ROOT,
    CLUSTERING_TILE_SIZE,
    CLUSTERING_TILE_STRIDE,
    CLUSTERING_MAX_TILES_EVAL,
    CLUSTERING_SEARCH_VECTOR_DIM,
    BEST_MODEL_PATH,
    GENIZA_IMAGE_INFORMATION_TABLE,
    GENIZA_IMAGE_LATENTS_TABLE,
)

logger = logging.getLogger(__name__)


def _pgvector_column_dim(cur, table_name: str, column_name: str) -> int | None:
    cur.execute(
        """
        SELECT format_type(a.atttypid, a.atttypmod) AS type_name
        FROM pg_attribute a
        WHERE a.attrelid = %s::regclass
          AND a.attname = %s
          AND NOT a.attisdropped
        """,
        (table_name, column_name),
    )
    row = cur.fetchone()
    if not row:
        return None
    type_name = row[0]
    match = re.fullmatch(r"vector\((\d+)\)", str(type_name))
    return int(match.group(1)) if match else None


def _ensure_pgvector_dim(cur, table_name: str, column_name: str, expected_dim: int) -> None:
    actual_dim = _pgvector_column_dim(cur, table_name, column_name)
    if actual_dim is None:
        raise RuntimeError(
            f"Could not determine pgvector dimension for {table_name}.{column_name}. "
            "Check that the column exists and is a pgvector column."
        )
    if int(actual_dim) != int(expected_dim):
        raise RuntimeError(
            f"{table_name}.{column_name} is vector({actual_dim}), but this run expects "
            f"vector({expected_dim}). This usually happens after changing LATENT_DIM or "
            "CLUSTERING_SEARCH_VECTOR_DIM. Use a fresh latents table, migrate/drop the old "
            "column/table, or project with a checkpoint/config that matches the existing DB schema."
        )


def setup_logger(log_dir=None):
    if log_dir is None:
        log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"project_geniza_to_latent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def get_db_connection(db_config_path):
    cfg = db_config_path
    if not os.path.isabs(cfg):
        cfg = os.path.join(project_root, cfg)
    config = configparser.ConfigParser()
    config.read(cfg)
    db = config["postgresql"]
    return psycopg2.connect(
        host=db["host"],
        database=db["database"],
        user=db["user"],
        password=db["password"],
        port=db.get("port", 5432),
    )


def load_image_rows_from_db(
    conn,
    offset: int,
    limit: int,
    images_root: str,
    log,
    colorful_only: bool = False,
    manuscript_min: str | None = None,
    manuscript_max: str | None = None,
):
    """
    Load image rows from geniza_image_information that don't already exist in geniza_image_latents.
    Returns list of dicts:
    manuscript_id, picture_id, is_color, parent_directory, page_number, page_id,
    image_path, xml_path (from table, or None), local_path (path to load image from disk).
    If colorful_only is True, only rows with is_color = true are returned.
    Automatically excludes images that already have latent vectors computed.
    Note: geniza_image_latents table must exist (call ensure_table first).
    """
    # Build WHERE clause - exclude images that already have latents
    where_clause = "WHERE i.image_path IS NOT NULL AND i.image_path != '' AND l.image_path IS NULL"
    if colorful_only:
        where_clause += " AND i.is_color = true"
    if manuscript_min is not None:
        where_clause += " AND i.manuscript_id >= %s"
    if manuscript_max is not None:
        where_clause += " AND i.manuscript_id <= %s"
    
    # limit <= 0 means no limit (fetch all rows that match criteria)
    limit_param = limit if limit and limit > 0 else None
    params = []
    if manuscript_min is not None:
        params.append(manuscript_min)
    if manuscript_max is not None:
        params.append(manuscript_max)
    params.extend([limit_param, offset])
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT i.manuscript_id, i.picture_id, i.is_color, i.parent_directory, i.page_number, i.page_id, i.image_path, i.xml_path
            FROM {GENIZA_IMAGE_INFORMATION_TABLE} i
            LEFT JOIN {GENIZA_IMAGE_LATENTS_TABLE} l ON i.image_path = l.image_path
            {where_clause}
            ORDER BY i.manuscript_id, i.parent_directory, i.picture_id
            LIMIT %s OFFSET %s
        """, tuple(params))
        rows = cur.fetchall()
    result = []
    for r in rows:
        manuscript_id, picture_id, is_color, parent_directory, page_number, page_id, image_path, xml_path = r
        # image_path in DB may be full path or relative; use as-is if absolute, else build from root
        if image_path and os.path.isabs(image_path):
            local_path = image_path
        else:
            local_path = os.path.join(images_root.rstrip("/"), str(manuscript_id), str(parent_directory or ""), str(picture_id or ""))
        result.append({
            "manuscript_id": str(manuscript_id),
            "picture_id": picture_id or "",
            "is_color": bool(is_color) if is_color is not None else None,
            "parent_directory": parent_directory or "",
            "page_number": page_number or "",
            "page_id": page_id or "",
            "image_path": image_path or "",
            "xml_path": (xml_path or "").strip() or None,
            "local_path": local_path,
        })
    log.info(
        "Loaded %d image rows from %s (offset=%d, limit=%s%s%s) - excluding images already in %s",
        len(result), GENIZA_IMAGE_INFORMATION_TABLE, offset,
        "all" if limit_param is None else limit,
        ", colorful_only=True" if colorful_only else "",
        f", manuscript_range=[{manuscript_min},{manuscript_max}]" if (manuscript_min is not None or manuscript_max is not None) else "",
        GENIZA_IMAGE_LATENTS_TABLE,
    )
    return result


def load_model(checkpoint_path, device, log):
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(project_root, checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        model_config = checkpoint.get("model_config", {})
        num_classes = model_config.get("num_classes", MAX_SELECTED_MANUSCRIPTS)
        tile_size = model_config.get("tile_size", CLUSTERING_TILE_SIZE)
        use_visual_mod = model_config.get("use_visual_mod", USE_VISUAL_MOD)
        use_char_mod = model_config.get("use_char_mod", USE_CHAR_MOD)
        use_word_mod = model_config.get("use_word_mod", USE_WORD_MOD)
        log.info("Checkpoint config: num_classes=%s, tile_size=%s", num_classes, tile_size)
    else:
        state_dict = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or (checkpoint if isinstance(checkpoint, dict) else None)
        )
        if state_dict is None:
            state_dict = checkpoint.state_dict()
        num_classes = MAX_SELECTED_MANUSCRIPTS
        tile_size = CLUSTERING_TILE_SIZE
        use_visual_mod = USE_VISUAL_MOD
        use_char_mod = USE_CHAR_MOD
        use_word_mod = USE_WORD_MOD

    model = MultiModal(
        num_classes=num_classes,
        tile_size=tile_size,
        use_visual_mod=use_visual_mod,
        use_char_mod=use_char_mod,
        use_word_mod=use_word_mod,
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Expose the modality flags used by the model (from checkpoint, not system.py)
    # so that the dataset can be configured to match.
    model._ckpt_use_visual_mod = use_visual_mod
    model._ckpt_use_char_mod = use_char_mod
    model._ckpt_use_word_mod = use_word_mod
    model._ckpt_tile_size = tile_size

    log.info("Model modalities: visual=%s, char=%s, word=%s (from checkpoint)",
             use_visual_mod, use_char_mod, use_word_mod)
    return model


def ensure_table(conn, latent_dim, search_dim):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {GENIZA_IMAGE_LATENTS_TABLE} (
                id SERIAL PRIMARY KEY,
                manuscript_id TEXT NOT NULL,
                picture_id TEXT NOT NULL,
                is_color BOOLEAN,
                parent_directory TEXT,
                page_number TEXT,
                page_id TEXT,
                image_path TEXT NOT NULL UNIQUE,
                xml_path TEXT,
                latent_vector vector({latent_dim}),
                latent_vector_search vector({search_dim})
            )
        """)
        # Add xml_path to tables created before this column existed
        cur.execute(f"""
            ALTER TABLE {GENIZA_IMAGE_LATENTS_TABLE} ADD COLUMN IF NOT EXISTS xml_path TEXT
        """)
        # Add feature count columns for diagnostics (if they don't exist)
        cur.execute(f"""
            ALTER TABLE {GENIZA_IMAGE_LATENTS_TABLE} ADD COLUMN IF NOT EXISTS num_visual_patches INTEGER
        """)
        cur.execute(f"""
            ALTER TABLE {GENIZA_IMAGE_LATENTS_TABLE} ADD COLUMN IF NOT EXISTS num_glyphs INTEGER
        """)
        cur.execute(f"""
            ALTER TABLE {GENIZA_IMAGE_LATENTS_TABLE} ADD COLUMN IF NOT EXISTS num_words INTEGER
        """)
        _ensure_pgvector_dim(cur, GENIZA_IMAGE_LATENTS_TABLE, "latent_vector", latent_dim)
        _ensure_pgvector_dim(cur, GENIZA_IMAGE_LATENTS_TABLE, "latent_vector_search", search_dim)
        # Ensure pgvector indexes exist for fast KNN search on both columns.
        # latent_vector_search (PCA, non-whitened, L2-normalised) is the default for KNN,
        # using cosine distance (<=> operator) with the vector_cosine_ops opclass.
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS geniza_image_latents_search_cosine_idx
            ON {GENIZA_IMAGE_LATENTS_TABLE}
            USING ivfflat (latent_vector_search vector_cosine_ops)
            WITH (lists = 100)
        """)
        # Also index the full latent_vector with cosine ops so --vector full remains fast.
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS geniza_image_latents_full_cosine_idx
            ON {GENIZA_IMAGE_LATENTS_TABLE}
            USING ivfflat (latent_vector vector_cosine_ops)
            WITH (lists = 100)
        """)
    conn.commit()


def run(
    checkpoint_path,
    offset,
    limit,
    images_root,
    db_config_path,
    batch_size,
    search_dim,
    log,
    colorful_only: bool = False,
    manuscript_min: str | None = None,
    manuscript_max: str | None = None,
):
    conn = get_db_connection(db_config_path or CLUSTERING_DB_CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    log.info("Loading checkpoint: %s", checkpoint_path)
    model = load_model(checkpoint_path, device, log)

    # Verify dataset modality flags (from system.py) match the model (from checkpoint).
    # A mismatch means the dataset would extract different inputs than the model expects.
    for flag_name, sys_val, ckpt_val in [
        ("USE_VISUAL_MOD", USE_VISUAL_MOD, model._ckpt_use_visual_mod),
        ("USE_CHAR_MOD", USE_CHAR_MOD, model._ckpt_use_char_mod),
        ("USE_WORD_MOD", USE_WORD_MOD, model._ckpt_use_word_mod),
    ]:
        if sys_val != ckpt_val:
            log.warning(
                "MODALITY MISMATCH: system.py %s=%s but checkpoint has %s. "
                "Dataset will use system.py value; model may receive unexpected input.",
                flag_name, sys_val, ckpt_val,
            )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])

    # Ensure table exists before querying (creates if needed)
    ensure_table(conn, LATENT_DIM, search_dim)

    image_rows = load_image_rows_from_db(
        conn,
        offset,
        limit,
        images_root,
        log,
        colorful_only=colorful_only,
        manuscript_min=manuscript_min,
        manuscript_max=manuscript_max,
    )
    if not image_rows:
        log.error("No images from %s (offset=%d, limit=%d)", GENIZA_IMAGE_INFORMATION_TABLE, offset, limit)
        conn.close()
        return

    paths = [r["local_path"] for r in image_rows]
    labels = [r["manuscript_id"] for r in image_rows]
    unique_labels = sorted(set(labels))
    label2idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    # Use xml_path from table when present; otherwise ManuscriptDataset falls back to find_xml_path_pretrain().
    xml_paths = [r.get("xml_path") or None for r in image_rows]

    dataset = ManuscriptDataset(
        paths,
        labels,
        transform,
        label2idx,
        xml_paths=xml_paths,
        patch_size=model._ckpt_tile_size,
        stride=CLUSTERING_TILE_STRIDE,
        max_tiles_per_image=CLUSTERING_MAX_TILES_EVAL,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        collate_fn=tile_collate_with_padding,
    )

    # Match latent to DB row by path (robust to dataloader order with num_workers)
    image_rows_by_local_path = {os.path.normpath(r["local_path"]): r for r in image_rows}
    # Store processed image paths so we can verify every selected row was written.
    latent_by_image_path = {}

    inserted = 0
    manuscripts_since_commit = set()
    global_idx = 0
    # Track per-image modality statistics
    n_images_no_glyphs = 0
    n_images_no_visual = 0
    n_images_total = 0
    glyph_counts = []   # number of valid glyphs per image

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting latents"):
            if batch is None:
                continue
            (
                tiles,
                valid_mask,
                coords,
                tile_page_segments,
                char_patches,
                char_valid_mask,
                glyph_coords,
                glyph_page_segments,
                char_class_ids,
                char_metadata,
                words,
                word_metadata,
                labels_tensor,
                batch_paths,
            ) = batch

            # Count valid glyphs, visual patches, and words per image in this batch
            for bi in range(valid_mask.shape[0]):
                n_vis = int(valid_mask[bi].sum())
                n_glyph = int(char_valid_mask[bi].sum())
                n_word = len(words[bi]) if words and bi < len(words) else 0
                glyph_counts.append(n_glyph)
                n_images_total += 1
                if n_glyph == 0:
                    n_images_no_glyphs += 1
                if n_vis == 0:
                    n_images_no_visual += 1

            tiles = tiles.to(device)
            valid_mask = valid_mask.to(device)
            coords = coords.to(device)
            tile_page_segments = tile_page_segments.to(device)
            char_patches = char_patches.to(device)
            char_valid_mask = char_valid_mask.to(device)
            glyph_coords = glyph_coords.to(device)
            glyph_page_segments = glyph_page_segments.to(device)
            char_class_ids = char_class_ids.to(device)

            latents, aux_latents = model.forward_features(
                tiles=tiles,
                tile_coords=coords,
                tile_valid_mask=valid_mask,
                tile_page_segments=tile_page_segments,
                glyph_patches=char_patches,
                glyph_coords=glyph_coords,
                glyph_valid_mask=char_valid_mask,
                glyph_page_segments=glyph_page_segments,
                char_class_ids=char_class_ids,
                words=words,
                word_metadata=word_metadata,
                word_page_segments=None,
                device=device,
                return_aux_latents=False,
            )
            latents = latents.cpu().float().numpy()
            norms = np.linalg.norm(latents, axis=1, keepdims=True)
            safe_norms = np.maximum(norms, 1e-12)
            latents = latents / safe_norms
            batch_size_actual = latents.shape[0]
            for i in range(batch_size_actual):
                path_key = os.path.normpath(batch_paths[i])
                row = image_rows_by_local_path.get(path_key)
                if row is None:
                    row = image_rows[global_idx + i] if (global_idx + i) < len(image_rows) else None
                    if row is None or os.path.normpath(row["local_path"]) != path_key:
                        log.warning("Latent row mismatch: batch path %s not in image_rows; using index fallback", path_key)
                        row = image_rows[global_idx + i]
                latent = latents[i]
                latent_by_image_path[row["image_path"]] = latent
                
                # Get feature counts for this image
                n_vis = int(valid_mask[i].sum())
                n_glyph = int(char_valid_mask[i].sum())
                n_word = len(words[i]) if words and i < len(words) else 0
                
                # Warn if image has very few features (likely to cause similarity issues)
                if n_vis == 0:
                    log.warning("Image %s has ZERO visual patches - will produce generic representation", row["image_path"])
                elif n_vis < 3:
                    log.warning("Image %s has only %d visual patches - may cause similarity issues", row["image_path"], n_vis)
                
                latent_str = "[" + ",".join(f"{x:.6f}" for x in latent.tolist()) + "]"
                try:
                    with conn.cursor() as cur:
                        cur.execute(f"""
                            INSERT INTO {GENIZA_IMAGE_LATENTS_TABLE}
                            (manuscript_id, picture_id, is_color, parent_directory, page_number, page_id, image_path, xml_path, latent_vector, latent_vector_search, num_visual_patches, num_glyphs, num_words)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector, NULL, %s, %s, %s)
                            ON CONFLICT (image_path) DO UPDATE SET
                            manuscript_id = EXCLUDED.manuscript_id,
                            picture_id = EXCLUDED.picture_id,
                            is_color = EXCLUDED.is_color,
                            parent_directory = EXCLUDED.parent_directory,
                            page_number = EXCLUDED.page_number,
                            page_id = EXCLUDED.page_id,
                            xml_path = EXCLUDED.xml_path,
                            latent_vector = EXCLUDED.latent_vector,
                            num_visual_patches = EXCLUDED.num_visual_patches,
                            num_glyphs = EXCLUDED.num_glyphs,
                            num_words = EXCLUDED.num_words
                            """, (
                                row["manuscript_id"],
                                row["picture_id"],
                                row["is_color"],
                                row["parent_directory"],
                                row["page_number"],
                                row["page_id"],
                                row["image_path"],
                                row.get("xml_path"),
                                latent_str,
                                n_vis,
                                n_glyph,
                                n_word,
                            ))
                        if cur.rowcount:
                            inserted += 1
                except Exception as e:
                    log.exception("Insert failed for %s: %s", row["image_path"], e)
                    conn.rollback()
                    continue
                manuscripts_since_commit.add(row["manuscript_id"])
                if len(manuscripts_since_commit) >= 10:
                    conn.commit()
                    manuscripts_since_commit.clear()
            global_idx += batch_size_actual

    conn.commit()

    # ── Modality coverage report ──────────────────────────────────────
    log.info("=" * 70)
    log.info("  MODALITY COVERAGE REPORT")
    log.info("=" * 70)
    log.info("Total images projected: %d", n_images_total)
    log.info("Images with 0 valid glyphs (visual-only):  %d / %d  (%.1f%%)",
             n_images_no_glyphs, n_images_total,
             100.0 * n_images_no_glyphs / n_images_total if n_images_total else 0)
    log.info("Images with 0 valid visual patches:        %d / %d  (%.1f%%)",
             n_images_no_visual, n_images_total,
             100.0 * n_images_no_visual / n_images_total if n_images_total else 0)
    if n_images_no_visual > 0:
        log.warning("⚠️  %d images have ZERO visual patches - these will produce identical/generic latent vectors and cause false high similarity!", n_images_no_visual)
    if glyph_counts:
        gc = np.array(glyph_counts)
        log.info("Glyph count per image: min=%d  max=%d  median=%d  mean=%.1f",
                 int(gc.min()), int(gc.max()), int(np.median(gc)), float(gc.mean()))
        # Distribution buckets
        n_0 = int(np.sum(gc == 0))
        n_1_10 = int(np.sum((gc >= 1) & (gc <= 10)))
        n_11_30 = int(np.sum((gc >= 11) & (gc <= 30)))
        n_31plus = int(np.sum(gc >= 31))
        log.info("Glyph distribution:  0=%d  1-10=%d  11-30=%d  31+=%d",
                 n_0, n_1_10, n_11_30, n_31plus)
    log.info("=" * 70)

    missing = [r["image_path"] for r in image_rows if r["image_path"] not in latent_by_image_path]
    if missing:
        log.error("Missing latents for %d rows (first 3): %s", len(missing), missing[:3])
        raise RuntimeError(f"Missing latents for {len(missing)} rows; latent_by_image_path has {len(latent_by_image_path)} entries")
    log.info("Stored L2-normalised full latent vectors for %d new images.", len(latent_by_image_path))

    # PCA / latent_vector_search is now handled by aftertune/recompute_geniza_pca.py
    conn.close()
    log.info(
        "Done. Inserted/updated %d rows into %s (latent_vector + diagnostics only).",
        inserted,
        GENIZA_IMAGE_LATENTS_TABLE,
    )
    log.info(
        "Note: Run python aftertune/recompute_geniza_pca.py after all projections "
        "are complete to refresh latent_vector_search for all images."
    )
    return inserted


def main():
    parser = argparse.ArgumentParser(
        description="Project geniza manuscript images to latent space (reads from geniza_image_information)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=BEST_MODEL_PATH,
        help="Path to pretrain/finetune checkpoint (.pth)",
    )
    parser.add_argument("--offset", type=int, default=0, help="Skip this many rows from geniza_image_information")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to process (0 = no limit, all rows matching criteria)")
    parser.add_argument(
        "--images_root",
        type=str,
        default=CLUSTERING_IMAGES_ROOT,
        help="Base path for image_path when not absolute (from system.py)",
    )
    parser.add_argument("--db-config", type=str, default=None, help="Path to db_config.ini")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for encoding")
    parser.add_argument(
        "--search_dim",
        type=int,
        default=CLUSTERING_SEARCH_VECTOR_DIM,
        help="PCA dimension for latent_vector_search",
    )
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument(
        "--colorful-only",
        default=True,
        action="store_true",
        help="Project only images marked as colorful (is_color = true) in geniza_image_information",
    )
    parser.add_argument(
        "--manuscript-min",
        type=str,
        default=None,
        help="Optional: minimum manuscript_id (inclusive) to process on this run (for sharding across GPUs)",
    )
    parser.add_argument(
        "--manuscript-max",
        type=str,
        default=None,
        help="Optional: maximum manuscript_id (inclusive) to process on this run (for sharding across GPUs)",
    )
    args = parser.parse_args()

    log = setup_logger(args.log_dir)

    run(
        checkpoint_path=args.checkpoint,
        offset=args.offset,
        limit=args.limit,
        images_root=args.images_root,
        db_config_path=args.db_config,
        batch_size=args.batch_size,
        search_dim=args.search_dim,
        log=log,
        colorful_only=args.colorful_only,
        manuscript_min=args.manuscript_min,
        manuscript_max=args.manuscript_max,
    )
    print("Next: run python aftertune/geniza_top_neighbors.py to build top-K neighbours (excluding same manuscript).")


if __name__ == "__main__":
    main()
