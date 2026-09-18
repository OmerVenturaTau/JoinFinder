#!/usr/bin/env python3
"""
Compare every pair of images in clusters_images_metadata.xlsx/csv and add
are_they_same_clusters and similarity_score (from geniza_image_latents).

Input: results_analysis/test_set/clusters_images_metadata.csv or xlsx
  Headers: manuscript_id, picture_id, parent_directory, page_number, width,
           height, xml_path, image_path, cluster_id

Output: results_analysis/test_set/clusters_images_metadata_pairs.xlsx (or --output)
  One row per pair (i, j) with i < j. Columns from both rows (suffix _1, _2)
  plus: are_they_same_clusters, similarity_score, image_1_in_db, image_2_in_db,
  diagnose_cli (command to run diagnose_similarity_issue.py for this pair).

  Similarity is missing (NaN) when either image has no row in geniza_image_latents
  (e.g. not yet projected, or image_path in the xlsx doesn't match the DB).
  image_1_in_db / image_2_in_db indicate which image(s) had a vector in the DB.

  Optional: --relaxed-alto-filters relaxes ALTO word/glyph filtering for this
  process only when using --checkpoint (no Hebrew dict; WC/GC off). Default follows system.py.

  Older checkpoints (e.g. d_model=1200, latent_dim=2048) are supported via --checkpoint:
  architecture widths are inferred from saved weights before the model is built. Override
  manually with --model-d-model / --model-latent-dim if needed.
"""

import os
import sys
import configparser
import argparse
import json
import math
from itertools import combinations

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ---------------------------------------------------------------------------
# ALTO / OCR filters for THIS PROCESS ONLY (does not edit system.py on disk).
# When using --checkpoint, ManuscriptDataset loads before submodules bind constants;
# patch `system` here *before* importing train.dataset / models.
# Default: same as system.py. Pass --relaxed-alto-filters to disable Hebrew dict
# and WC/GC filtering for this process only (only affects --checkpoint / ManuscriptDataset).
# ---------------------------------------------------------------------------
_pre_alto = argparse.ArgumentParser(add_help=False)
_pre_alto.add_argument(
    "--checkpoint",
    type=str,
    default="",
    help=argparse.SUPPRESS,
)
_pre_alto.add_argument(
    "--relaxed-alto-filters",
    action="store_true",
    help="Ignore Hebrew dict + ALTO WC/GC thresholds for this run (--checkpoint only).",
)
_pre_alto.add_argument(
    "--model-glyph-summary-tokens",
    type=int,
    default=None,
    help="Evaluation-only compatibility override for old checkpoints trained with a different glyph summary-token count.",
)
_pre_alto.add_argument(
    "--model-use-word",
    choices=("true", "false"),
    default=None,
    help="Evaluation-only compatibility override for old checkpoints trained with/without the word branch.",
)
_pre_alto.add_argument(
    "--model-d-model",
    type=int,
    default=None,
    help="Override fusion d_model for old checkpoints (default: infer from weights).",
)
_pre_alto.add_argument(
    "--model-latent-dim",
    type=int,
    default=None,
    help="Override exported latent_dim for old checkpoints (default: infer from weights).",
)
_pre_alto.add_argument(
    "--legacy-stage2-20260824",
    action="store_true",
    help="Reproduce the 2026-08-24 checkpoint's word filtering and legacy batch TF-IDF path.",
)
_pre_alto_args, _ = _pre_alto.parse_known_args()


def _resolve_project_path(path: str) -> str:
    if not path:
        return ""
    return path if os.path.isabs(path) else os.path.join(project_root, path)


# Inspect the checkpoint and patch system constants BEFORE importing model
# modules, which snapshot system.* values at import time.
from utilities.checkpoint_utils import (  # noqa: E402
    inspect_checkpoint,
    apply_inspection_to_system,
    load_state_dict_with_report,
    warn_if_tile_size_mismatch,
)

_checkpoint_path_early = _resolve_project_path(_pre_alto_args.checkpoint)
_checkpoint_inspection = inspect_checkpoint(_checkpoint_path_early)
_needs_system_patch = (
    bool(_checkpoint_path_early)
    or _pre_alto_args.relaxed_alto_filters
    or _pre_alto_args.model_glyph_summary_tokens is not None
    or _pre_alto_args.model_use_word is not None
    or _pre_alto_args.model_d_model is not None
    or _pre_alto_args.model_latent_dim is not None
    or _pre_alto_args.legacy_stage2_20260824
    or any(
        getattr(_checkpoint_inspection, name) is not None
        for name in (
            "use_visual_mod",
            "use_char_mod",
            "use_word_mod",
            "glyph_summary_tokens",
            "d_model",
            "latent_dim",
            "fusion_dim_feedforward",
            "tile_dim_feedforward",
        )
    )
)
if _needs_system_patch:
    import system as _system_for_compare_clusters  # noqa: E402

    if _pre_alto_args.relaxed_alto_filters:
        _system_for_compare_clusters.USE_HEBREW_DICT_CHECK = False
        _system_for_compare_clusters.OCR_STRING_CONFIDENCE_THRESHOLD = 0.0
        _system_for_compare_clusters.OCR_GLYPH_CONFIDENCE_THRESHOLD = 0.0

    if _pre_alto_args.legacy_stage2_20260824:
        # Recorded run revision used 0.98 at the WordBranch's effective hard
        # filter and the historical batch-local TF-IDF gate.
        _system_for_compare_clusters.OCR_STRING_CONFIDENCE_THRESHOLD = 0.98

    if _checkpoint_path_early:
        apply_inspection_to_system(_checkpoint_inspection, _system_for_compare_clusters)

    # Manual CLI overrides take precedence over inferred values.
    if _pre_alto_args.model_glyph_summary_tokens is not None:
        _system_for_compare_clusters.GLYPH_NUM_SUMMARY_TOKENS = int(_pre_alto_args.model_glyph_summary_tokens)
    if _pre_alto_args.model_use_word is not None:
        _system_for_compare_clusters.USE_WORD_MOD = _pre_alto_args.model_use_word == "true"
    if _pre_alto_args.model_d_model is not None:
        _system_for_compare_clusters.D_MODEL = int(_pre_alto_args.model_d_model)
    if _pre_alto_args.model_latent_dim is not None:
        _system_for_compare_clusters.LATENT_DIM = int(_pre_alto_args.model_latent_dim)

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from scipy import ndimage
from torch.utils.data import DataLoader
from torchvision import transforms

from system import (
    GENIZA_IMAGE_LATENTS_TABLE,
    CLUSTERING_DB_CONFIG_PATH,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    CLUSTERING_TILE_SIZE,
    CLUSTERING_TILE_STRIDE,
    CLUSTERING_MAX_TILES_EVAL,
    MAX_SELECTED_MANUSCRIPTS,
    D_MODEL,
    FUSION_METHOD,
    BASE_DIR,
    STAGE2_TABLE_NAME,
    GENIZA_IMAGE_BASE,
    GENIZA_XML_BASE,
    GENIZA_XML_FILENAME_SUFFIX,
)
from train.dataset import ManuscriptDataset, tile_collate_with_padding
from models import MultiModal
from utilities.augmentations.background_overlays import RandomLibraryBackgroundOverlay
from utilities.augmentations.manuscript_augmentations import RandomWhiteBackground

if _pre_alto_args.legacy_stage2_20260824:
    # The training-era dataset admitted ALTO strings at 0.90 before the
    # WordBranch applied its stricter 0.98 hard filter.  Keep those thresholds
    # separate: max_words is enforced during extraction, so using 0.98 at both
    # stages can change which later high-confidence words reach the model.
    import train.dataset as _dataset_for_compare_clusters

    _dataset_for_compare_clusters.OCR_STRING_CONFIDENCE_THRESHOLD = 0.90


class CanonicalizeExternalBackground:
    """Replace only border-connected mounting/padding with neutral parchment."""

    def __init__(self, max_coverage: float = 0.55) -> None:
        self.max_coverage = float(max_coverage)
        self._glyph_detector = RandomLibraryBackgroundOverlay(p=0.0)

    @staticmethod
    def _parchment_color(arr: np.ndarray) -> np.ndarray:
        rgb = arr.astype(np.float32)
        lum = rgb.mean(axis=2)
        spread = rgb.max(axis=2) - rgb.min(axis=2)
        h, w = lum.shape
        center = np.zeros_like(lum, dtype=bool)
        center[max(0, h // 10):max(1, h - h // 10), max(0, w // 10):max(1, w - w // 10)] = True
        plausible = center & (lum >= np.percentile(lum, 45)) & (spread < 85)
        if plausible.sum() < 32:
            plausible = lum >= np.percentile(lum, 60)
        return np.median(rgb[plausible], axis=0) if plausible.any() else np.median(rgb.reshape(-1, 3), axis=0)

    def _tile_foreground_mask(self, image: Image.Image) -> Image.Image | None:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        parchment = self._parchment_color(arr)
        lum = arr.mean(axis=2)
        parchment_lum = float(parchment.mean())
        spread = arr.max(axis=2) - arr.min(axis=2)
        distance = np.sqrt(np.mean((arr - parchment[None, None, :]) ** 2, axis=2))
        blue = (arr[:, :, 2] - arr[:, :, 0] > 24) & (arr[:, :, 2] - arr[:, :, 1] > 8)
        chromatic = spread > 45
        very_bright = lum > max(235.0, parchment_lum + 35.0)
        very_dark = lum < parchment_lum - 95.0
        candidate = (distance > 38.0) & (blue | chromatic | very_bright | very_dark)

        labels, _ = ndimage.label(candidate)
        border_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
        border_labels = border_labels[border_labels != 0]
        if border_labels.size == 0:
            return None
        background = np.isin(labels, border_labels)
        background = ndimage.binary_closing(background, iterations=1, border_value=1)
        coverage = float(background.mean())
        if coverage < 0.002 or coverage > self.max_coverage:
            return None
        foreground = np.where(background, 0, 255).astype(np.uint8)
        return Image.fromarray(foreground, mode="L").filter(ImageFilter.GaussianBlur(radius=2.0))

    def __call__(self, image: Image.Image) -> Image.Image:
        img = image.convert("RGB")
        if max(img.size) <= 192:
            foreground = self._glyph_detector._detect_connected_edge_background(img)
        else:
            foreground = self._tile_foreground_mask(img)
        if foreground is None:
            return img
        arr = np.asarray(img, dtype=np.uint8)
        parchment = tuple(int(round(v)) for v in self._parchment_color(arr))
        neutral = Image.new("RGB", img.size, parchment)
        return Image.composite(img, neutral, foreground)


def get_db_connection(db_config_path: str):
    cfg = db_config_path
    if not os.path.isabs(cfg):
        cfg = os.path.join(project_root, cfg)
    if not os.path.exists(cfg):
        raise FileNotFoundError(f"Config not found: {cfg}")
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


def load_latent_vectors_for_paths(conn, image_paths: list) -> dict:
    """
    Load latent_vector and feature counts for each image_path.
    Returns (vecs_by_path, counts_by_path).

    - vecs_by_path: image_path -> np.array (L2-normalized) (or None if parse fails)
    - counts_by_path: image_path -> (num_visual_patches, num_glyphs, num_words) (may be None if missing)
    """
    import psycopg2.extras
    paths = [p for p in image_paths if pd.notna(p) and str(p).strip()]
    if not paths:
        return {}, {}
    placeholders = ",".join(["%s"] * len(paths))
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            f"""
            SELECT
                image_path,
                latent_vector::text AS vec,
                num_visual_patches,
                num_glyphs,
                num_words
            FROM {GENIZA_IMAGE_LATENTS_TABLE}
            WHERE image_path IN ({placeholders}) AND latent_vector IS NOT NULL
            """,
            tuple(paths),
        )
        rows = cur.fetchall()
    by_path = {}
    counts_by_path = {}
    for r in rows:
        key = (r["image_path"] or "").strip()
        if key:
            vec_str = r["vec"]
            try:
                arr = np.array([float(x) for x in vec_str.strip("[]").split(",")])
                n = np.linalg.norm(arr)
                if n >= 1e-12:
                    arr = arr / n
                by_path[key] = arr
            except Exception:
                by_path[key] = None
            counts_by_path[key] = (
                int(r["num_visual_patches"]) if r.get("num_visual_patches") is not None else None,
                int(r["num_glyphs"]) if r.get("num_glyphs") is not None else None,
                int(r["num_words"]) if r.get("num_words") is not None else None,
            )
    # Also match normpath variants
    for r in rows:
        key = os.path.normpath((r["image_path"] or "").strip())
        if key and key not in by_path and r.get("vec"):
            # Reuse already-parsed vector if present for the raw key
            raw_key = (r["image_path"] or "").strip()
            if raw_key in by_path:
                by_path[key] = by_path[raw_key]
                if raw_key in counts_by_path:
                    counts_by_path[key] = counts_by_path[raw_key]
                continue
            try:
                arr = np.array([float(x) for x in r["vec"].strip("[]").split(",")])
                n = np.linalg.norm(arr)
                if n >= 1e-12:
                    arr = arr / n
                by_path[key] = arr
                counts_by_path[key] = counts_by_path.get(raw_key)
            except Exception:
                pass
    return by_path, counts_by_path


SHELFMARK_TABLE = "geniza_manuscript_shelfmark"


def load_manuscript_metadata(conn, manuscript_ids: list) -> dict[str, dict[str, str]]:
    """Fetch ``normalized_library`` and ``shelfmark_root`` for the given manuscripts.

    Returns ``{manuscript_id: {"library": ..., "shelfmark_root": ...}}``. Missing
    manuscripts are simply absent from the dict so callers can default safely.
    Failures (missing table, bad permissions) are logged and the function
    returns an empty dict — this metadata is descriptive only and should not
    block the main pair analysis.
    """
    ids = [str(m).strip() for m in manuscript_ids if pd.notna(m) and str(m).strip()]
    ids = sorted(set(ids))
    if not ids:
        return {}
    placeholders = ",".join(["%s"] * len(ids))
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT manuscript_id, normalized_library, shelfmark_root
                FROM {SHELFMARK_TABLE}
                WHERE manuscript_id IN ({placeholders})
                """,
                tuple(ids),
            )
            rows = cur.fetchall()
    except Exception as exc:
        print(
            f"[Metadata] Could not load library/shelfmark from {SHELFMARK_TABLE}: {exc}. "
            "Continuing without library/shelfmark columns."
        )
        return {}

    out: dict[str, dict[str, str]] = {}
    for ms_id, library, shelfmark in rows:
        key = str(ms_id or "").strip()
        if not key:
            continue
        out[key] = {
            "library": str(library or "").strip(),
            "shelfmark_root": str(shelfmark or "").strip(),
        }
    return out


def _load_member_table(input_path: str, sheet_name: str | None = None) -> pd.DataFrame:
    """Load either canonical fixed-test format as member-level rows.

    ``cluster_members.xlsx`` stores paths relative to the Geniza roots.  They
    are resolved here using the same rules as the training evaluator, while a
    conventional CSV/XLSX containing ``image_path`` is consumed directly.
    """
    suffix = os.path.splitext(input_path)[1].lower()
    if suffix == ".csv":
        df = pd.read_csv(input_path)
    elif suffix in {".xlsx", ".xls", ".xlsm"}:
        df = pd.read_excel(input_path, sheet_name=sheet_name or 0)
    else:
        raise ValueError(f"Input must be CSV or Excel, got {suffix!r}: {input_path}")

    cluster_member_columns = {"cluster_id", "image_id", "manuscript_id", "relative_path"}
    if cluster_member_columns.issubset(df.columns) and "image_path" not in df.columns:
        df = df.copy()
        image_paths: list[str] = []
        xml_paths: list[str] = []
        parent_directories: list[str] = []
        picture_ids: list[str] = []
        for row in df.itertuples(index=False):
            relative = os.path.normpath(str(row.relative_path).strip())
            parts = list(relative.split(os.sep))
            if parts and parts[0] == os.path.basename(os.path.normpath(GENIZA_IMAGE_BASE)):
                parts = parts[1:]
            manuscript_id = str(row.manuscript_id).strip()
            if len(parts) < 3 or parts[0] != manuscript_id:
                raise ValueError(
                    f"Invalid cluster-member relative path for manuscript {manuscript_id!r}: "
                    f"{row.relative_path!r}"
                )
            filename = parts[-1]
            parent = parts[-2]
            stem = os.path.splitext(filename)[0]
            image_paths.append(os.path.normpath(os.path.join(GENIZA_IMAGE_BASE, *parts)))
            xml_paths.append(os.path.normpath(os.path.join(
                GENIZA_XML_BASE,
                manuscript_id,
                parent,
                f"{stem}{GENIZA_XML_FILENAME_SUFFIX}",
            )))
            parent_directories.append(parent)
            picture_ids.append(filename)
        df["image_path"] = image_paths
        df["xml_path"] = xml_paths
        df["parent_directory"] = parent_directories
        df["picture_id"] = picture_ids

    required = {"manuscript_id", "picture_id", "image_path", "cluster_id"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Input {input_path} is missing required member columns: {missing}")
    if "xml_path" not in df.columns:
        df["xml_path"] = None
    df = df.copy()
    df["image_path"] = df["image_path"].astype(str).str.strip().map(os.path.normpath)
    df["cluster_id"] = df["cluster_id"].astype(str).str.strip()
    duplicate_paths = df.loc[df["image_path"].duplicated(keep=False), "image_path"].unique()
    if len(duplicate_paths):
        raise ValueError(f"Input contains duplicate image paths: {duplicate_paths[:3].tolist()}")
    missing_images = [path for path in df["image_path"] if not os.path.isfile(path)]
    if missing_images:
        raise FileNotFoundError(
            f"Input path preflight found {len(missing_images)} missing images; "
            f"sample={missing_images[:2]}"
        )
    return df.reset_index(drop=True)


def _load_stage2_validation_members() -> pd.DataFrame:
    """Reconstruct the Stage-2 classification validation split from the DB.

    The checkpoint run used the table's authoritative ``dataset_split`` field.
    The live validation side is stable at 6,803 rows even though training rows
    have since grown, so this reproduces the checkpoint's validation members.
    """
    from train.split_data import build_splits

    splits, _stats = build_splits(
        BASE_DIR,
        table_name=STAGE2_TABLE_NAME,
        dataset_stage="stage2",
    )
    rows: list[dict[str, object]] = []
    for manuscript_id, items in splits["val"].items():
        for image_path, xml_path in items:
            image_path = os.path.normpath(str(image_path))
            rows.append({
                "manuscript_id": str(manuscript_id),
                "picture_id": os.path.basename(image_path),
                "parent_directory": os.path.basename(os.path.dirname(image_path)),
                "xml_path": xml_path,
                "image_path": image_path,
                "cluster_id": str(manuscript_id),
            })
    df = pd.DataFrame(rows)
    if len(df) != 6803:
        raise RuntimeError(
            "Stage-2 validation membership changed: expected the checkpoint run's "
            f"6,803 rows, reconstructed {len(df):,}. Refusing a non-comparable evaluation."
        )
    return df


def _is_valid_vector(vec: np.ndarray | None) -> bool:
    if vec is None:
        return False
    arr = np.asarray(vec)
    return arr.ndim == 1 and arr.size > 0 and np.isfinite(arr).all()


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    if not _is_valid_vector(v1) or not _is_valid_vector(v2):
        return np.nan
    return float(np.dot(v1, v2))


def _load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> MultiModal:
    """Load MultiModal model from checkpoint for latent extraction."""
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(project_root, checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Reuse the early inspection when possible to avoid loading the .pth twice.
    if (
        _checkpoint_inspection is not None
        and _checkpoint_inspection.path == checkpoint_path
        and _checkpoint_inspection.state_dict
    ):
        inspection = _checkpoint_inspection
    else:
        inspection = inspect_checkpoint(checkpoint_path)

    state_dict = inspection.state_dict
    if not state_dict:
        raise RuntimeError(f"Could not extract a state_dict from checkpoint: {checkpoint_path}")

    num_classes = inspection.num_classes if inspection.num_classes is not None else MAX_SELECTED_MANUSCRIPTS

    # Sanity-check tile size against the active runtime tiling.
    warn_if_tile_size_mismatch(inspection, CLUSTERING_TILE_SIZE, label="Checkpoint")

    if inspection.d_model is not None and int(inspection.d_model) != int(D_MODEL):
        print(
            f"[Checkpoint] Building model with d_model={D_MODEL} (inferred from checkpoint; "
            f"current config may differ)."
        )
    model = MultiModal(
        num_classes=num_classes,
        d_model=int(D_MODEL),
        word_legacy_batch_tfidf=_pre_alto_args.legacy_stage2_20260824,
    ).to(device)
    load_state_dict_with_report(model, state_dict, label="Checkpoint")
    model.eval()

    if hasattr(model, "modality_dropout_enabled"):
        model.modality_dropout_enabled = False
    if hasattr(model, "token_subsample_enabled"):
        model.token_subsample_enabled = False

    # Note: DataParallel is intentionally NOT used here. The forward path below
    # always goes through ``model.module if hasattr(model, 'module') else model``
    # (to avoid AlephBERT device-mismatch in the word branch), so wrapping in DP
    # would only add memory replication with no parallel execution.

    return model


_MODALITY_ORDER = ("tiles", "glyphs", "words")


def _fusion_subsets(enabled_modalities: frozenset[str]) -> list[frozenset[str]]:
    subsets: list[frozenset[str]] = []
    keys = tuple(k for k in _MODALITY_ORDER if k in enabled_modalities)
    for r in range(1, len(keys) + 1):
        for comb in combinations(keys, r):
            subsets.append(frozenset(comb))
    return subsets


def _fusion_mode_name(subset: frozenset[str]) -> str:
    abbrev = "".join(k[0] for k in _MODALITY_ORDER if k in subset)
    return f"fuse_{abbrev}"


_FUSE_MODE_LABELS = {
    "fuse_t": "Fusion ablation: tiles only",
    "fuse_g": "Fusion ablation: glyphs only",
    "fuse_w": "Fusion ablation: words only",
    "fuse_tg": "Fusion ablation: tiles + glyphs",
    "fuse_tw": "Fusion ablation: tiles + words",
    "fuse_gw": "Fusion ablation: glyphs + words",
    "fuse_tgw": "Fusion ablation: tiles + glyphs + words",
}

_LATE_FUSION_SCORE_SPECS: tuple[tuple[str, str, dict[str, float]], ...] = (
    (
        "late_tg_equal_similarity_score",
        "Late score fusion: tile 0.50 + glyph 0.50",
        {
            "tile_branch_similarity_score": 0.50,
            "glyph_branch_similarity_score": 0.50,
        },
    ),
    (
        "late_tg_tile70_glyph30_similarity_score",
        "Late score fusion: tile 0.70 + glyph 0.30",
        {
            "tile_branch_similarity_score": 0.70,
            "glyph_branch_similarity_score": 0.30,
        },
    ),
    (
        "late_tgf_tile70_glyph20_full10_similarity_score",
        "Late score fusion: tile 0.70 + glyph 0.20 + full 0.10",
        {
            "tile_branch_similarity_score": 0.70,
            "glyph_branch_similarity_score": 0.20,
            "similarity_score": 0.10,
        },
    ),
    (
        "late_tgfb_tile70_glyph20_fusion10_similarity_score",
        "Late score fusion: tile 0.70 + glyph 0.20 + pre-head fusion 0.10",
        {
            "tile_branch_similarity_score": 0.70,
            "glyph_branch_similarity_score": 0.20,
            "fusion_branch_similarity_score": 0.10,
        },
    ),
)


def _score_column_label(score_col: str) -> str:
    """Human-readable label for a similarity column in outputs / console."""
    static = {
        "similarity_score": "Full multimodal fusion (DB latent_vector or checkpoint)",
        "fusion_branch_similarity_score": "Pre-head fusion token",
        "tile_branch_similarity_score": "Pre-fusion tile branch only",
        "glyph_branch_similarity_score": "Pre-fusion glyph branch only",
        "word_branch_similarity_score": "Pre-fusion word branch only",
    }
    if score_col in static:
        return static[score_col]
    for col, label, _weights in _LATE_FUSION_SCORE_SPECS:
        if score_col == col:
            return label
    if score_col.endswith("_similarity_score"):
        mode = score_col[: -len("_similarity_score")]
        return _FUSE_MODE_LABELS.get(mode, f"Fusion subset: {mode}")
    return score_col


_PAIR_SEPARATION_PRINT_METRICS = (
    "n_pairs",
    "roc_auc",
    "pr_auc_average_precision",
    "same_mean",
    "different_mean",
    "mean_gap_same_minus_different",
    "best_f1_threshold",
    "best_f1",
    "best_f1_precision",
    "best_f1_recall",
)

_PAIR_RETRIEVAL_PRINT_METRICS = (
    "n_images_with_scores",
    "n_queries_with_relevant",
    "mAP",
    "recall@1",
    "recall@5",
    "recall@10",
    "hit@1",
    "hit@5",
    "hit@10",
)


def _print_all_score_metrics(metrics_df: pd.DataFrame, score_columns: list[str]) -> None:
    """Print pair-separation and retrieval stats for every score column (branches + ablations)."""
    if metrics_df.empty or not score_columns:
        return

    def metric_value(metric_set: str, name: str):
        frame = metrics_df[metrics_df["metric_set"] == metric_set]
        values = frame.loc[frame["metric"] == name, "value"]
        return values.iloc[0] if len(values) else None

    print("\n" + "=" * 72)
    print("METRICS BY EMBEDDING (full fusion, single branches, fusion ablations)")
    print("=" * 72)

    for score_col in score_columns:
        label = _score_column_label(score_col)
        print(f"\n--- {label} ---")
        print(f"    column: {score_col}")

        n_pairs = metric_value(score_col, "n_pairs")
        if n_pairs is None:
            print("    (no finite scores — use --checkpoint for branch/ablation embeddings)")
            continue
        print(f"    n_pairs_with_score: {int(n_pairs)}")

        print("    Pair separation (same vs different cluster):")
        for name in _PAIR_SEPARATION_PRINT_METRICS:
            if name == "n_pairs":
                continue
            value = metric_value(score_col, name)
            if value is not None:
                print(f"      {name}: {float(value):.6f}")

        retrieval_set = f"{score_col}/cluster_retrieval"
        if metrics_df["metric_set"].eq(retrieval_set).any():
            print("    Cluster retrieval (from pairwise scores):")
            for name in _PAIR_RETRIEVAL_PRINT_METRICS:
                value = metric_value(retrieval_set, name)
                if value is not None:
                    if name.startswith("n_"):
                        print(f"      {name}: {int(value)}")
                    else:
                        print(f"      {name}: {float(value):.6f}")


def _compute_latents_with_checkpoint(
    checkpoint_path: str,
    image_paths: list,
    xml_paths: list | None = None,
    batch_size: int = 8,
    num_workers: int = 8,
    mask_external_background: bool = False,
    ink_only_background: bool = False,
    ink_only_tiles: bool = False,
) -> tuple[dict, dict, dict[str, dict[str, np.ndarray | None]], dict[str, dict[str, np.ndarray | None]]]:
    """
    Compute normalized latents for the given image paths using a checkpoint,
    returned as dict image_path -> np.ndarray.

    Also returns checkpoint-derived pre-fusion branch vectors:
    - tile: after tile branch transformer, before fusion
    - glyph: after glyph summarizer, before fusion
    - word: after word summarizer, before fusion
    and fused vectors for every non-empty subset of active modalities:
    - fuse_t, fuse_g, fuse_w, fuse_tg, fuse_tw, fuse_gw, fuse_tgw
    """
    assert len(image_paths) > 0
    if xml_paths is None:
        xml_paths = [None] * len(image_paths)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device for checkpoint projection: {device}")
    model = _load_model_from_checkpoint(checkpoint_path, device)

    if ink_only_background or ink_only_tiles:
        inference_pil_ops = [RandomWhiteBackground(p=1.0)]
    elif mask_external_background:
        inference_pil_ops = [CanonicalizeExternalBackground()]
    else:
        inference_pil_ops = []
    transform = transforms.Compose([
        *inference_pil_ops,
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])

    # Dummy labels / mapping – we only care about latents, not logits.
    labels = ["0"] * len(image_paths)
    label2idx = {"0": 0}

    dataset = ManuscriptDataset(
        image_paths,
        labels,
        transform,
        label2idx,
        xml_paths=xml_paths,
        patch_size=CLUSTERING_TILE_SIZE,
        stride=CLUSTERING_TILE_STRIDE,
        max_tiles_per_image=CLUSTERING_MAX_TILES_EVAL,
        split="test",
        legacy_word_extraction_20260824=_pre_alto_args.legacy_stage2_20260824,
    )
    if mask_external_background or ink_only_background:
        inference_char_transform = (
            RandomWhiteBackground(p=1.0)
            if ink_only_background
            else CanonicalizeExternalBackground()
        )
        char_ops = [inference_char_transform, transforms.ToTensor()]
        if bool(getattr(__import__("system"), "CHAR_PATCH_APPLY_IMAGENET_NORM", True)):
            char_ops.append(transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD))
        dataset.char_transform = transforms.Compose(char_ops)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=tile_collate_with_padding,
    )

    by_path: dict[str, np.ndarray | None] = {}
    counts_by_path: dict[str, tuple[int | None, int | None, int | None]] = {}
    branch_vecs_by_name: dict[str, dict[str, np.ndarray | None]] = {
        "fusion": {},
        "tile": {},
        "glyph": {},
        "word": {},
    }
    model_for_forward = model.module if hasattr(model, "module") else model
    enabled_modalities: list[str] = []
    if getattr(model_for_forward, "use_visual_mod", False):
        enabled_modalities.append("tiles")
    if getattr(model_for_forward, "use_char_mod", False):
        enabled_modalities.append("glyphs")
    if getattr(model_for_forward, "use_word_mod", False):
        enabled_modalities.append("words")
    fusion_sets = _fusion_subsets(frozenset(enabled_modalities))
    fusion_mode_to_subset = {_fusion_mode_name(s): s for s in fusion_sets}
    fusion_vecs_by_mode: dict[str, dict[str, np.ndarray | None]] = {
        mode: {} for mode in fusion_mode_to_subset.keys()
    }

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
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

            if batch_index == 1 or batch_index % 50 == 0 or batch_index == len(loader):
                print(f"Projected batch {batch_index}/{len(loader)}", flush=True)

            tiles = tiles.to(device)
            valid_mask = valid_mask.to(device)
            coords = coords.to(device)
            tile_page_segments = tile_page_segments.to(device)
            char_patches = char_patches.to(device) if char_patches is not None else None
            char_valid_mask = char_valid_mask.to(device) if char_valid_mask is not None else None
            glyph_coords = glyph_coords.to(device) if glyph_coords is not None else None
            glyph_page_segments = glyph_page_segments.to(device) if glyph_page_segments is not None else None
            char_class_ids = char_class_ids.to(device) if char_class_ids is not None else None
            batch_element_indices = torch.arange(
                tiles.shape[0], dtype=torch.long, device=device
            )

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(device.type == "cuda" and _pre_alto_args.legacy_stage2_20260824),
            ):
                _, latents, aux_latents = model_for_forward(
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
                    paths=list(batch_paths),
                    batch_element_indices=batch_element_indices,
                    return_aux_latents=True,
                )
            latents = latents.detach().cpu().float().numpy()
            aux_latents_np = {
                name: tensor.detach().cpu().float().numpy()
                for name, tensor in aux_latents.items()
                if tensor is not None
            }
            subset_latents_np: dict[str, np.ndarray] = {}
            if FUSION_METHOD == "symmetric":
                # Symmetric fusion's subset vectors can be reconstructed exactly
                # from one full forward pass. The full softmax weights only need
                # renormalization over the retained modalities, avoiding seven
                # redundant ConvNeXt/AlephBERT passes per batch.
                schema = tuple(model_for_forward.fusion.enabled_modalities)
                reliability = aux_latents["reliability_weights"]
                branch_tensor = {name: aux_latents[name] for name in schema}
                plural_to_singular = {"tiles": "tile", "glyphs": "glyph", "words": "word"}
                for mode, subset in fusion_mode_to_subset.items():
                    selected = {plural_to_singular[name] for name in subset}
                    selected_indices = [idx for idx, name in enumerate(schema) if name in selected]
                    denom = reliability[:, selected_indices].sum(dim=1, keepdim=True)
                    subset_weights = torch.zeros_like(reliability)
                    if selected_indices:
                        subset_weights[:, selected_indices] = (
                            reliability[:, selected_indices] / denom.clamp_min(1e-8)
                        )
                    blocks = [
                        branch_tensor[name] * subset_weights[:, idx:idx + 1].clamp_min(0).sqrt()
                        if name in selected else torch.zeros_like(branch_tensor[name])
                        for idx, name in enumerate(schema)
                    ]
                    subset_latent = torch.nn.functional.normalize(
                        torch.cat(blocks, dim=1), dim=1, eps=1e-8
                    )
                    subset_latents_np[mode] = subset_latent.detach().cpu().numpy()
            else:
                for mode, subset in fusion_mode_to_subset.items():
                    use_t = "tiles" in subset
                    use_g = "glyphs" in subset
                    use_w = "words" in subset
                    _, subset_latent, _ = model_for_forward(
                        tiles=tiles if use_t else None,
                        tile_coords=coords if use_t else None,
                        tile_valid_mask=valid_mask if use_t else None,
                        tile_page_segments=tile_page_segments if use_t else None,
                        glyph_patches=char_patches if use_g else None,
                        glyph_coords=glyph_coords if use_g else None,
                        glyph_valid_mask=char_valid_mask if use_g else None,
                        glyph_page_segments=glyph_page_segments if use_g else None,
                        char_class_ids=char_class_ids if use_g else None,
                        words=words if use_w else None,
                        word_metadata=word_metadata if use_w else None,
                        paths=list(batch_paths),
                        batch_element_indices=batch_element_indices,
                        return_aux_latents=False,
                    )
                    subset_latents_np[mode] = subset_latent.detach().cpu().float().numpy()

            for i, p in enumerate(batch_paths):
                key = (p or "").strip()
                if not key:
                    continue

                def sample_has_subset_evidence(subset: frozenset[str]) -> bool:
                    """True when this row has at least one real token in the requested subset."""
                    if "tiles" in subset and valid_mask is not None and bool(valid_mask[i].any().item()):
                        return True
                    if "glyphs" in subset and char_valid_mask is not None and bool(char_valid_mask[i].any().item()):
                        return True
                    if "words" in subset and words and i < len(words) and words[i]:
                        return True
                    return False

                vec = latents[i]
                n = np.linalg.norm(vec) if np.isfinite(vec).all() else 0.0
                if n >= 1e-12:
                    vec = vec / n
                    by_path[key] = vec if _is_valid_vector(vec) else None
                else:
                    by_path[key] = None
                # Feature counts (what the model actually received as valid tokens)
                n_vis = int(valid_mask[i].sum().item()) if valid_mask is not None else None
                n_glyph = int(char_valid_mask[i].sum().item()) if char_valid_mask is not None else None
                n_word = len(words[i]) if words and i < len(words) and words[i] is not None else 0
                counts_by_path[key] = (n_vis, n_glyph, n_word)
                for branch_name in ("fusion", "tile", "glyph", "word"):
                    branch_arr = aux_latents_np.get(branch_name)
                    if branch_arr is None:
                        continue
                    branch_vec = branch_arr[i]
                    branch_norm = np.linalg.norm(branch_vec) if np.isfinite(branch_vec).all() else 0.0
                    if branch_norm >= 1e-12:
                        branch_vec = branch_vec / branch_norm
                        branch_vecs_by_name[branch_name][key] = branch_vec if _is_valid_vector(branch_vec) else None
                    else:
                        branch_vecs_by_name[branch_name][key] = None
                for mode, mode_latents in subset_latents_np.items():
                    subset = fusion_mode_to_subset[mode]
                    if not sample_has_subset_evidence(subset):
                        fusion_vecs_by_mode[mode][key] = None
                        continue
                    mode_vec = mode_latents[i]
                    mode_norm = np.linalg.norm(mode_vec) if np.isfinite(mode_vec).all() else 0.0
                    if mode_norm >= 1e-12:
                        mode_vec = mode_vec / mode_norm
                        fusion_vecs_by_mode[mode][key] = mode_vec if _is_valid_vector(mode_vec) else None
                    else:
                        fusion_vecs_by_mode[mode][key] = None

    # Also index by normpath variants for robustness (match DB loader behaviour)
    for key, vec in list(by_path.items()):
        norm_key = os.path.normpath(key)
        if norm_key not in by_path:
            by_path[norm_key] = vec
        if norm_key not in counts_by_path and key in counts_by_path:
            counts_by_path[norm_key] = counts_by_path[key]
    for branch_name, branch_map in branch_vecs_by_name.items():
        for key, vec in list(branch_map.items()):
            norm_key = os.path.normpath(key)
            if norm_key not in branch_map:
                branch_map[norm_key] = vec
    for mode, mode_map in fusion_vecs_by_mode.items():
        for key, vec in list(mode_map.items()):
            norm_key = os.path.normpath(key)
            if norm_key not in mode_map:
                mode_map[norm_key] = vec

    return by_path, counts_by_path, branch_vecs_by_name, fusion_vecs_by_mode


def _add_late_fusion_score_columns(pairs_df: pd.DataFrame) -> list[str]:
    """
    Add production-style score-level fusion columns.

    These blend already-normalized cosine scores and renormalize over available
    components, so rows without glyph evidence fall back to the finite tile/full
    components instead of becoming a null-token artifact.
    """
    added: list[str] = []
    for col, _label, weights_by_col in _LATE_FUSION_SCORE_SPECS:
        src_cols = [src for src in weights_by_col if src in pairs_df.columns]
        if not src_cols:
            continue
        values = pairs_df[src_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        weights = np.array([float(weights_by_col[src]) for src in src_cols], dtype=float)
        finite = np.isfinite(values)
        if int(finite.any(axis=0).sum()) < 2:
            continue
        numerator = (np.where(finite, values, 0.0) * weights).sum(axis=1)
        denominator = (finite * weights).sum(axis=1)
        fused = np.full_like(numerator, np.nan, dtype=float)
        has_score = denominator > 0.0
        fused[has_score] = numerator[has_score] / denominator[has_score]
        pairs_df[col] = fused
        if pairs_df[col].notna().any():
            added.append(col)
    return added


def _finite_floats(values) -> list[float]:
    out: list[float] = []
    for v in values:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fv):
            out.append(fv)
    return out


def _try_plot_similarity_histograms(
    same_vals: list[float],
    diff_vals: list[float],
    out_path: str,
) -> None:
    """Single-panel same-vs-different histogram for the main fused score.

    Kept for backwards-compatibility; the multi-panel version below is the
    preferred entry point and matches the layout used by
    discriminability_calibration.py.
    """
    try:
        import matplotlib.pyplot as plt

        same_vals = _finite_floats(same_vals)
        diff_vals = _finite_floats(diff_vals)
        if not same_vals and not diff_vals:
            return

        all_vals = same_vals + diff_vals
        lo = float(min(all_vals)) - 0.005
        hi = float(max(all_vals)) + 0.005
        bins = 50

        plt.figure(figsize=(7, 4))
        if same_vals:
            plt.hist(same_vals, bins=bins, range=(lo, hi), density=True, alpha=0.55, label="same cluster", color="steelblue")
        if diff_vals:
            plt.hist(diff_vals, bins=bins, range=(lo, hi), density=True, alpha=0.55, label="different clusters", color="tomato")
        plt.title("Similarity score distribution")
        plt.xlabel("cosine similarity")
        plt.ylabel("density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    except Exception:
        return


def _try_plot_similarity_histograms_multi(
    pairs_df: pd.DataFrame,
    score_columns: list[str],
    out_path: str,
    *,
    title_suffix: str = "",
) -> None:
    """Multi-panel same-vs-different histograms, one panel per score column.

    Uses ``image_1_in_db`` / ``image_2_in_db`` to filter for pairs where the
    fused vector exists, then per-column drops non-finite values so per-branch
    panels (which can be missing when a modality has no tokens) are plotted on
    whatever pairs do have a finite score.
    """
    try:
        import matplotlib.pyplot as plt

        if not score_columns:
            return

        same_data: dict[str, list[float]] = {}
        diff_data: dict[str, list[float]] = {}
        same_mask = pairs_df["are_they_same_clusters"] == True  # noqa: E712
        diff_mask = pairs_df["are_they_same_clusters"] == False  # noqa: E712
        if "image_1_in_db" in pairs_df.columns and "image_2_in_db" in pairs_df.columns:
            in_db = pairs_df["image_1_in_db"].astype(bool) & pairs_df["image_2_in_db"].astype(bool)
            same_mask = same_mask & in_db
            diff_mask = diff_mask & in_db

        present_columns: list[str] = []
        for col in score_columns:
            if col not in pairs_df.columns:
                continue
            sv = _finite_floats(pairs_df.loc[same_mask, col].tolist())
            dv = _finite_floats(pairs_df.loc[diff_mask, col].tolist())
            if not sv and not dv:
                continue
            same_data[col] = sv
            diff_data[col] = dv
            present_columns.append(col)

        if not present_columns:
            return

        histogram_range = (-1.0, 1.0)
        histogram_edges = np.linspace(histogram_range[0], histogram_range[1], 41)
        densities: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for col in present_columns:
            same_density, _ = np.histogram(same_data[col], bins=histogram_edges, density=True)
            diff_density, _ = np.histogram(diff_data[col], bins=histogram_edges, density=True)
            same_density = np.nan_to_num(same_density)
            diff_density = np.nan_to_num(diff_density)
            densities[col] = (same_density, diff_density)

        n_panels = len(present_columns)
        ncols = min(4, n_panels)
        nrows = int(math.ceil(n_panels / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5 * ncols, 4 * nrows),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        flat_axes = axes.ravel()
        # Cosine similarity has a fixed mathematical range. Every panel uses
        # the same bin edges and a shared y maximum derived across the entire
        # figure, so both axes are directly comparable without clipping.
        # Fixed across every dataset figure (curated clusters, cluster members,
        # and validation) so separate PNGs are directly comparable too.
        histogram_density_range = (0.0, 18.0)
        for ax, col in zip(flat_axes, present_columns):
            same_density, diff_density = densities[col]
            ax.stairs(same_density, histogram_edges, fill=True, alpha=0.55, label="same cluster", color="steelblue")
            ax.stairs(diff_density, histogram_edges, fill=True, alpha=0.55, label="different clusters", color="tomato")
            ax.set_xlim(*histogram_range)
            ax.set_ylim(*histogram_density_range)
            ax.set_title(_score_column_label(col), fontsize=9)
            ax.set_xlabel("cosine similarity")
            ax.set_ylabel("density")
            ax.legend()
        for ax in flat_axes[n_panels:]:
            ax.set_visible(False)
        if title_suffix:
            fig.suptitle(title_suffix)
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97) if title_suffix else None)
        plt.savefig(out_path)
        plt.close()
    except Exception:
        return


def _try_plot_similarity_histograms_direct(
    df: pd.DataFrame,
    representations: list[tuple[str, object]],
    out_path: str,
    *,
    device: torch.device,
    title_suffix: str = "",
    chunk_size: int = 256,
) -> None:
    """Plot same/different-cluster histograms without building an O(N^2) table.

    Validation has millions of image pairs, so similarities are accumulated in
    fixed bins on the accelerator one query chunk at a time. Only each unique
    unordered pair is counted, matching the Excel-backed cluster plots.
    """
    try:
        import matplotlib.pyplot as plt

        histogram_edges = np.linspace(-1.0, 1.0, 41)
        histogram_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for score_col, get_vec in representations:
            vectors: list[np.ndarray] = []
            raw_labels: list[str] = []
            for _, row in df.iterrows():
                vec = get_vec(row["image_path"])
                if not _is_valid_vector(vec):
                    continue
                vectors.append(np.asarray(vec, dtype=np.float32))
                raw_labels.append(str(row["cluster_id"]))
            if len(vectors) < 2:
                continue

            label_to_idx = {label: idx for idx, label in enumerate(dict.fromkeys(raw_labels))}
            features = torch.from_numpy(np.stack(vectors)).to(device=device, dtype=torch.float32)
            features = torch.nn.functional.normalize(features, dim=1)
            labels = torch.tensor(
                [label_to_idx[label] for label in raw_labels],
                device=device,
                dtype=torch.long,
            )
            n_images = int(features.shape[0])
            same_counts = torch.zeros(40, device=device, dtype=torch.float64)
            diff_counts = torch.zeros(40, device=device, dtype=torch.float64)
            candidate_indices = torch.arange(n_images, device=device)

            for start in range(0, n_images, chunk_size):
                end = min(start + chunk_size, n_images)
                similarities = features[start:end] @ features.T
                unique_pair = candidate_indices.unsqueeze(0) > torch.arange(
                    start, end, device=device
                ).unsqueeze(1)
                same_cluster = labels[start:end].unsqueeze(1) == labels.unsqueeze(0)
                same_values = similarities[unique_pair & same_cluster]
                diff_values = similarities[unique_pair & ~same_cluster]
                if same_values.numel():
                    same_counts += torch.histc(same_values.float(), bins=40, min=-1.0, max=1.0).double()
                if diff_values.numel():
                    diff_counts += torch.histc(diff_values.float(), bins=40, min=-1.0, max=1.0).double()

            bin_width = float(histogram_edges[1] - histogram_edges[0])
            same_total = float(same_counts.sum().item())
            diff_total = float(diff_counts.sum().item())
            same_density = (
                same_counts / (same_total * bin_width)
                if same_total > 0.0
                else torch.zeros_like(same_counts)
            ).cpu().numpy()
            diff_density = (
                diff_counts / (diff_total * bin_width)
                if diff_total > 0.0
                else torch.zeros_like(diff_counts)
            ).cpu().numpy()
            histogram_data[score_col] = (same_density, diff_density)
            del features, labels, same_counts, diff_counts
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if not histogram_data:
            return

        n_panels = len(histogram_data)
        ncols = min(4, n_panels)
        nrows = int(math.ceil(n_panels / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5 * ncols, 4 * nrows),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        flat_axes = axes.ravel()
        y_max = 18.0
        for ax, (score_col, (same_density, diff_density)) in zip(flat_axes, histogram_data.items()):
            ax.stairs(same_density, histogram_edges, fill=True, alpha=0.55, label="same cluster", color="steelblue")
            ax.stairs(diff_density, histogram_edges, fill=True, alpha=0.55, label="different clusters", color="tomato")
            ax.set_xlim(-1.0, 1.0)
            ax.set_ylim(0.0, y_max)
            ax.set_title(_score_column_label(score_col), fontsize=9)
            ax.set_xlabel("cosine similarity")
            ax.set_ylabel("density")
            ax.legend()
        for ax in flat_axes[n_panels:]:
            ax.set_visible(False)
        if title_suffix:
            fig.suptitle(title_suffix)
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97) if title_suffix else None)
        plt.savefig(out_path)
        plt.close()
    except Exception as exc:
        print(f"Warning: could not create direct histogram grid: {exc}")


def _requested_histogram_columns() -> list[str]:
    """The three standalone branches plus all seven through-fusion subsets."""
    return [
        "glyph_branch_similarity_score",
        "tile_branch_similarity_score",
        "word_branch_similarity_score",
        "fuse_g_similarity_score",
        "fuse_t_similarity_score",
        "fuse_w_similarity_score",
        "fuse_gw_similarity_score",
        "fuse_tg_similarity_score",
        "fuse_tw_similarity_score",
        "fuse_tgw_similarity_score",
    ]


def _pair_separation_metrics(
    out_df: pd.DataFrame,
    score_col: str,
    *,
    label_col: str = "are_they_same_clusters",
) -> list[dict[str, object]]:
    """Compute pair-level same/different-cluster separation metrics."""
    if score_col not in out_df.columns:
        return []

    valid = out_df[[label_col, score_col]].dropna().copy()
    if valid.empty:
        return []

    valid[label_col] = valid[label_col].astype(bool)
    y_true = valid[label_col].astype(int).to_numpy()
    scores = valid[score_col].astype(float).to_numpy()
    same_scores = scores[y_true == 1]
    diff_scores = scores[y_true == 0]

    rows: list[dict[str, object]] = []

    def add(metric: str, value: object) -> None:
        rows.append({"metric_set": score_col, "metric": metric, "value": value})

    add("n_pairs", int(len(scores)))
    add("n_same_pairs", int(len(same_scores)))
    add("n_different_pairs", int(len(diff_scores)))
    for prefix, vals in (("same", same_scores), ("different", diff_scores)):
        if len(vals) == 0:
            continue
        add(f"{prefix}_mean", float(np.mean(vals)))
        add(f"{prefix}_std", float(np.std(vals)))
        add(f"{prefix}_median", float(np.median(vals)))
        add(f"{prefix}_p10", float(np.percentile(vals, 10)))
        add(f"{prefix}_p90", float(np.percentile(vals, 90)))
    if len(same_scores) > 0 and len(diff_scores) > 0:
        add("mean_gap_same_minus_different", float(np.mean(same_scores) - np.mean(diff_scores)))

    if len(np.unique(y_true)) < 2:
        return rows

    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

    add("roc_auc", float(roc_auc_score(y_true, scores)))
    add("pr_auc_average_precision", float(average_precision_score(y_true, scores)))

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision/recall include one extra point with no threshold; ignore it for
    # threshold selection so the reported threshold is actionable.
    if len(thresholds) > 0:
        p = precision[:-1]
        r = recall[:-1]
        f1 = (2 * p * r) / np.clip(p + r, 1e-12, None)
        best_idx = int(np.nanargmax(f1))
        best_threshold = float(thresholds[best_idx])
        pred = scores >= best_threshold
        add("best_f1_threshold", best_threshold)
        add("best_f1", float(f1[best_idx]))
        add("best_f1_precision", float(p[best_idx]))
        add("best_f1_recall", float(r[best_idx]))
        add("best_f1_accuracy", float((pred.astype(int) == y_true).mean()))

    return rows


def _cluster_retrieval_metrics(
    df: pd.DataFrame,
    get_vec,
    *,
    k_values: tuple[int, ...] = (1, 5, 10),
) -> list[dict[str, object]]:
    """
    Image-level retrieval metrics: each image queries all other images, and
    relevant images are those with the same cluster_id.
    """
    records = []
    for idx, row in df.iterrows():
        vec = get_vec(row["image_path"])
        if vec is None:
            continue
        records.append(
            {
                "idx": idx,
                "image_path": row["image_path"],
                "cluster_id": row["cluster_id"],
                "vec": vec,
            }
        )

    rows: list[dict[str, object]] = []

    def add(metric: str, value: object) -> None:
        rows.append({"metric_set": "cluster_retrieval", "metric": metric, "value": value})

    add("n_images_with_vectors", int(len(records)))
    if len(records) < 2:
        return rows

    aps: list[float] = []
    recalls_at_k = {k: [] for k in k_values}
    hits_at_k = {k: [] for k in k_values}
    evaluated_queries = 0

    for query in records:
        scored = []
        for cand in records:
            if cand["idx"] == query["idx"]:
                continue
            sim = cosine_similarity(query["vec"], cand["vec"])
            scored.append((sim, cand["cluster_id"] == query["cluster_id"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        relevant_total = sum(1 for _, is_rel in scored if is_rel)
        if relevant_total == 0:
            continue

        evaluated_queries += 1
        hit_count = 0
        precision_sum = 0.0
        for rank, (_, is_rel) in enumerate(scored, start=1):
            if is_rel:
                hit_count += 1
                precision_sum += hit_count / rank
        aps.append(precision_sum / relevant_total)

        for k in k_values:
            top_k = scored[:k]
            rel_in_top_k = sum(1 for _, is_rel in top_k if is_rel)
            recalls_at_k[k].append(rel_in_top_k / relevant_total)
            hits_at_k[k].append(float(rel_in_top_k > 0))

    add("n_queries_with_relevant", int(evaluated_queries))
    if evaluated_queries == 0:
        return rows

    add("mAP", float(np.mean(aps)))
    for k in k_values:
        add(f"recall@{k}", float(np.mean(recalls_at_k[k])))
        add(f"hit@{k}", float(np.mean(hits_at_k[k])))
    return rows


def _cluster_retrieval_metrics_from_pairs(
    pairs_df: pd.DataFrame,
    score_col: str,
    *,
    k_values: tuple[int, ...] = (1, 5, 10),
) -> list[dict[str, object]]:
    """
    Retrieval metrics from a pairwise score column. This naturally excludes
    pairs where that modality/embedding is unavailable or non-finite.
    """
    if score_col not in pairs_df.columns:
        return []

    valid = pairs_df.copy()
    valid[score_col] = pd.to_numeric(valid[score_col], errors="coerce")
    valid = valid[np.isfinite(valid[score_col].to_numpy(dtype=float, na_value=np.nan))]

    rows: list[dict[str, object]] = []
    metric_set = f"{score_col}/cluster_retrieval"

    def add(metric: str, value: object) -> None:
        rows.append({"metric_set": metric_set, "metric": metric, "value": value})

    if valid.empty:
        add("n_images_with_scores", 0)
        return rows

    cluster_by_path = {}
    adjacency = {}
    for _, r in valid.iterrows():
        a = str(r["image_path_1"])
        b = str(r["image_path_2"])
        ca = r["cluster_id_1"]
        cb = r["cluster_id_2"]
        score = float(r[score_col])
        cluster_by_path[a] = ca
        cluster_by_path[b] = cb
        adjacency.setdefault(a, []).append((score, b, ca == cb))
        adjacency.setdefault(b, []).append((score, a, ca == cb))

    add("n_images_with_scores", int(len(adjacency)))

    aps: list[float] = []
    recalls_at_k = {k: [] for k in k_values}
    hits_at_k = {k: [] for k in k_values}
    evaluated_queries = 0

    for query_path, scored in adjacency.items():
        relevant_total = sum(1 for _, _, is_rel in scored if is_rel)
        if relevant_total == 0:
            continue
        evaluated_queries += 1
        scored.sort(key=lambda x: x[0], reverse=True)

        hit_count = 0
        precision_sum = 0.0
        for rank, (_, _, is_rel) in enumerate(scored, start=1):
            if is_rel:
                hit_count += 1
                precision_sum += hit_count / rank
        aps.append(precision_sum / relevant_total)

        for k in k_values:
            top_k = scored[:k]
            rel_in_top_k = sum(1 for _, _, is_rel in top_k if is_rel)
            recalls_at_k[k].append(rel_in_top_k / relevant_total)
            hits_at_k[k].append(float(rel_in_top_k > 0))

    add("n_queries_with_relevant", int(evaluated_queries))
    if evaluated_queries == 0:
        return rows

    add("mAP", float(np.mean(aps)))
    for k in k_values:
        add(f"recall@{k}", float(np.mean(recalls_at_k[k])))
        add(f"hit@{k}", float(np.mean(hits_at_k[k])))
    return rows


def _direct_retrieval_metrics(
    df: pd.DataFrame,
    representations: list[tuple[str, object]],
    *,
    device: torch.device,
) -> pd.DataFrame:
    """Compute scalable retrieval metrics directly from per-image vectors."""
    from train.metric_learning import compute_retrieval_metrics

    rows: list[dict[str, object]] = []
    for score_col, get_vec in representations:
        vectors: list[np.ndarray] = []
        raw_labels: list[str] = []
        for _, row in df.iterrows():
            vec = get_vec(row["image_path"])
            if not _is_valid_vector(vec):
                continue
            vectors.append(np.asarray(vec, dtype=np.float32))
            raw_labels.append(str(row["cluster_id"]))
        metric_set = f"{score_col}/cluster_retrieval"
        rows.append({"metric_set": metric_set, "metric": "n_images_with_vectors", "value": len(vectors)})
        if len(vectors) < 2:
            continue
        label_to_idx = {label: idx for idx, label in enumerate(dict.fromkeys(raw_labels))}
        features = torch.from_numpy(np.stack(vectors)).to(device=device, dtype=torch.float32)
        labels = torch.tensor([label_to_idx[label] for label in raw_labels], device=device, dtype=torch.long)
        result = compute_retrieval_metrics(features, labels)
        values = {
            "n_queries_with_relevant": result.num_queries,
            "mAP": result.mean_average_precision,
            "knn_at_1": result.knn_at_1,
            "knn_at_5": result.knn_at_5,
            "knn_at_10": result.knn_at_10,
        }
        for metric_name, value in values.items():
            rows.append({"metric_set": metric_set, "metric": metric_name, "value": value})
        del features, labels
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def _score_modalities(score_col: str) -> list[str]:
    static = {
        "similarity_score": ["tile", "glyph", "word"],
        "fusion_branch_similarity_score": ["tile", "glyph", "word"],
        "tile_branch_similarity_score": ["tile"],
        "glyph_branch_similarity_score": ["glyph"],
        "word_branch_similarity_score": ["word"],
    }
    if score_col in static:
        return static[score_col]
    if score_col.startswith("fuse_") and score_col.endswith("_similarity_score"):
        code = score_col[len("fuse_") : -len("_similarity_score")]
        return [name for letter, name in (("t", "tile"), ("g", "glyph"), ("w", "word")) if letter in code]
    for col, _label, weights in _LATE_FUSION_SCORE_SPECS:
        if score_col == col:
            modalities = []
            for source in weights:
                if source.startswith("tile_"):
                    modalities.append("tile")
                elif source.startswith("glyph_"):
                    modalities.append("glyph")
                elif source.startswith("word_"):
                    modalities.append("word")
                elif source in {"similarity_score", "fusion_branch_similarity_score"}:
                    modalities.append("fusion")
            return modalities
    return []


def _representation_key(score_col: str) -> str:
    static = {
        "similarity_score": "full_model",
        "fusion_branch_similarity_score": "pre_head_fusion",
        "tile_branch_similarity_score": "branch_tile",
        "glyph_branch_similarity_score": "branch_glyph",
        "word_branch_similarity_score": "branch_word",
    }
    if score_col in static:
        return static[score_col]
    if score_col.startswith("fuse_") and score_col.endswith("_similarity_score"):
        code = score_col[len("fuse_") : -len("_similarity_score")]
        names = _score_modalities(score_col)
        return "fusion_" + "_".join(names) if names else f"fusion_{code}"
    return score_col.removesuffix("_similarity_score")


def _compact_metrics_payload(
    metrics_df: pd.DataFrame,
    score_columns: list[str],
    *,
    dataset_name: str,
    dataset_source: str,
    df: pd.DataFrame,
    checkpoint_path: str,
    inference_batch_size: int,
    inference_num_workers: int,
) -> dict[str, object]:
    def value(metric_set: str, *metric_names: str):
        frame = metrics_df[metrics_df["metric_set"] == metric_set]
        for metric_name in metric_names:
            values = frame.loc[frame["metric"] == metric_name, "value"]
            if len(values):
                raw = values.iloc[0]
                return int(raw) if metric_name.startswith("n_") else float(raw)
        return None

    checkpoint = _checkpoint_inspection.raw if isinstance(_checkpoint_inspection.raw, dict) else {}
    representations: dict[str, object] = {}
    for score_col in score_columns:
        metric_set = f"{score_col}/cluster_retrieval"
        map_value = value(metric_set, "mAP")
        if map_value is None:
            continue
        representations[_representation_key(score_col)] = {
            "label": _score_column_label(score_col),
            "score_column": score_col,
            "modalities": _score_modalities(score_col),
            "n_images": value(metric_set, "n_images_with_vectors", "n_images_with_scores"),
            "n_queries": value(metric_set, "n_queries_with_relevant"),
            "mAP": map_value,
            "knn@1": value(metric_set, "knn_at_1", "hit@1"),
            "knn@5": value(metric_set, "knn_at_5", "hit@5"),
            "knn@10": value(metric_set, "knn_at_10", "hit@10"),
        }
    return {
        "schema_version": 1,
        "checkpoint": {
            "path": os.path.abspath(checkpoint_path) if checkpoint_path else None,
            "filename": os.path.basename(checkpoint_path) if checkpoint_path else None,
            "epoch": checkpoint.get("epoch"),
            "training_mode": checkpoint.get("training_mode"),
        },
        "evaluation": {
            "legacy_stage2_20260824": bool(_pre_alto_args.legacy_stage2_20260824),
            "alto_string_extraction_threshold": (
                0.90
                if _pre_alto_args.legacy_stage2_20260824
                else float(getattr(__import__("system"), "OCR_STRING_CONFIDENCE_THRESHOLD"))
            ),
            "word_branch_confidence_threshold": float(
                getattr(__import__("system"), "OCR_STRING_CONFIDENCE_THRESHOLD")
            ),
            "legacy_batch_tfidf_gating": bool(_pre_alto_args.legacy_stage2_20260824),
            "autocast_dtype": (
                "bfloat16" if _pre_alto_args.legacy_stage2_20260824 else None
            ),
            "inference_batch_size": int(inference_batch_size),
            "inference_num_workers": int(inference_num_workers),
        },
        "dataset": {
            "name": dataset_name,
            "source": dataset_source,
            "num_images": int(len(df)),
            "num_clusters": int(df["cluster_id"].astype(str).nunique()),
        },
        "representations": representations,
    }


def _write_compact_metrics_json(path: str, payload: dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(f"Wrote compact retrieval metrics to {path}")


def _try_plot_retrieval_summary(
    payload: dict[str, object],
    out_path: str,
    series: list[tuple[str, str]],
    title_group: str,
) -> None:
    """Plot mAP and KNN@1/5/10 for a requested, ordered representation group."""
    try:
        import matplotlib.pyplot as plt

        all_representations = payload.get("representations", {})
        representations = [
            (label, all_representations[key])
            for key, label in series
            if key in all_representations
        ]
        if not representations:
            return
        labels = [label for label, _entry in representations]
        metric_names = ["mAP", "knn@1", "knn@5", "knn@10"]
        matrix = np.array([
            [entry[name] for name in metric_names]
            for _label, entry in representations
        ], dtype=float)
        fig_height = max(5.5, 0.52 * len(labels) + 1.8)
        fig, ax = plt.subplots(figsize=(10.5, fig_height))
        image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(metric_names)), metric_names)
        ax.set_yticks(range(len(labels)), labels)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                ax.text(col, row, f"{value:.3f}", ha="center", va="center",
                        color="white" if value >= 0.58 else "black", fontsize=9)
        dataset = payload.get("dataset", {})
        checkpoint = payload.get("checkpoint", {})
        ax.set_title(
            f"Retrieval performance — {title_group}\n"
            f"{dataset.get('name')} — Stage 2 epoch {checkpoint.get('epoch')}",
            fontweight="bold",
            pad=14,
        )
        fig.colorbar(image, ax=ax, label="score", fraction=0.035, pad=0.03)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Wrote retrieval summary image to {out_path}")
    except Exception as exc:
        print(f"Warning: could not write retrieval summary plot: {exc}")


def _write_retrieval_summary_images(payload: dict[str, object], output_stem: str) -> None:
    fusion_series = [
        ("fusion_glyph", "Glyphs"),
        ("fusion_tile", "Tiles"),
        ("fusion_word", "Words"),
        ("fusion_glyph_word", "Glyphs + words"),
        ("fusion_tile_glyph", "Glyphs + tiles"),
        ("fusion_tile_word", "Words + tiles"),
        ("fusion_tile_glyph_word", "Glyphs + words + tiles"),
    ]
    branch_series = [
        ("branch_glyph", "Glyph branch"),
        ("branch_tile", "Tile branch"),
        ("branch_word", "Word branch"),
    ]
    _try_plot_retrieval_summary(
        payload,
        output_stem + "_retrieval_metrics_fusion.png",
        fusion_series,
        "fusion combinations",
    )
    _try_plot_retrieval_summary(
        payload,
        output_stem + "_retrieval_metrics_branches.png",
        branch_series,
        "pre-fusion branches",
    )


def _write_outputs(output_path: str, pairs_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    """Write pair rows plus a compact metrics table."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ext = os.path.splitext(output_path)[1].lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        with pd.ExcelWriter(output_path) as writer:
            pairs_df.to_excel(writer, index=False, sheet_name="pairs")
            if not metrics_df.empty:
                metrics_df.to_excel(writer, index=False, sheet_name="metrics")
    else:
        pairs_df.to_csv(output_path, index=False)

    metrics_path = os.path.splitext(output_path)[0] + "_metrics.csv"
    if not metrics_df.empty:
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Wrote metrics to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=os.path.join(script_dir, "clusters_images_metadata.csv"),
        help="Input CSV/XLSX member file (default: clusters_images_metadata.csv in this script's folder)",
    )
    parser.add_argument("--sheet-name", default=None, help="Excel sheet name (default: first sheet)")
    parser.add_argument(
        "--stage2-validation",
        action="store_true",
        help="Evaluate the authoritative Stage-2 validation split reconstructed from the DB.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip O(N^2) pair rows/Excel and compute retrieval metrics directly from embeddings.",
    )
    parser.add_argument("--dataset-name", default=None, help="Dataset label stored in JSON/plot titles")
    parser.add_argument(
        "--skip-manuscript-metadata",
        action="store_true",
        help="Skip optional library/shelfmark DB enrichment.",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output Excel file (default: same dir as input, name clusters_images_metadata_pairs.xlsx)",
    )
    parser.add_argument("--db-config", type=str, default=CLUSTERING_DB_CONFIG_PATH, help="DB config path (used when no checkpoint is provided)")
    parser.add_argument("--limit", type=int, default=0, help="Max number of pairs to compute (0 = all)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Optional model checkpoint (.pth). If provided, latents are computed from this checkpoint instead of loaded from DB.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for latent extraction when using --checkpoint",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Data-loader workers for checkpoint projection (default: 8)",
    )
    parser.add_argument(
        "--inference-background",
        choices=("original", "masked", "ink-only", "ink-only-tiles", "ensemble"),
        default="original",
        help="Checkpoint inference view: original, external-background masked, or normalized average of both.",
    )
    parser.add_argument(
        "--relaxed-alto-filters",
        action="store_true",
        help="Same as early parse: relaxed ALTO for --checkpoint (this process only).",
    )
    parser.add_argument(
        "--model-glyph-summary-tokens",
        type=int,
        default=None,
        help="Same as early parse: override glyph summary-token count for old checkpoint compatibility.",
    )
    parser.add_argument(
        "--model-use-word",
        choices=("true", "false"),
        default=None,
        help="Same as early parse: override word branch on/off for old checkpoint compatibility.",
    )
    parser.add_argument(
        "--model-d-model",
        type=int,
        default=None,
        help="Same as early parse: override fusion d_model (default: infer from checkpoint weights).",
    )
    parser.add_argument(
        "--model-latent-dim",
        type=int,
        default=None,
        help="Same as early parse: override latent_dim (default: infer from checkpoint weights).",
    )
    parser.add_argument(
        "--legacy-stage2-20260824",
        action="store_true",
        help="Use the historical word confidence/TF-IDF behavior from this checkpoint run.",
    )
    args = parser.parse_args()

    if args.stage2_validation:
        input_path = "stage2_validation_from_database"
        args.retrieval_only = True
        df = _load_stage2_validation_members()
    else:
        input_path = args.input_csv
        if not os.path.isabs(input_path):
            # Prefer script dir; also accept paths relative to project root.
            candidate_script = os.path.join(script_dir, input_path)
            candidate_root = os.path.join(project_root, input_path)
            if os.path.exists(candidate_root):
                input_path = candidate_root
            else:
                input_path = candidate_script
        if not os.path.exists(input_path):
            print(f"Error: input file not found: {input_path}")
            return 1
        df = _load_member_table(input_path, sheet_name=args.sheet_name)

    if args.output:
        output_path = os.path.join(project_root, args.output) if not os.path.isabs(args.output) else args.output
    else:
        if args.stage2_validation:
            output_path = os.path.join(script_dir, "stage2_validation_metrics.xlsx")
        else:
            output_path = os.path.join(os.path.dirname(input_path), "clusters_images_metadata_pairs.xlsx")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    required = ["manuscript_id", "picture_id", "image_path", "cluster_id"]
    for c in required:
        if c not in df.columns:
            print(f"Error: missing column '{c}' in input. Columns: {list(df.columns)}")
            return 1

    n = len(df)
    pairs = [] if args.retrieval_only else [(i, j) for i in range(n) for j in range(i + 1, n)]
    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]
    if args.retrieval_only:
        print(f"Loaded {n} rows. Retrieval-only mode: pair workbook disabled.")
    else:
        print(f"Loaded {n} rows. Computing {len(pairs)} pairs...")

    image_paths = df["image_path"].astype(str).unique().tolist()
    print(f"Preparing latent vectors for {len(image_paths)} images...")

    # Best-effort metadata enrichment: fetch normalized_library and
    # shelfmark_root for every manuscript in the input. The `_1`/`_2`-suffix
    # loop below picks them up automatically once they're columns on `df`.
    if "manuscript_id" in df.columns and not args.skip_manuscript_metadata and not args.stage2_validation:
        unique_ms_ids = df["manuscript_id"].dropna().astype(str).unique().tolist()
        try:
            meta_conn = get_db_connection(args.db_config)
        except Exception as exc:
            print(
                f"[Metadata] Could not connect to {args.db_config} for shelfmark metadata: {exc}. "
                "Continuing without library/shelfmark columns."
            )
            ms_meta: dict[str, dict[str, str]] = {}
        else:
            try:
                ms_meta = load_manuscript_metadata(meta_conn, unique_ms_ids)
            finally:
                meta_conn.close()
        if ms_meta:
            print(f"[Metadata] Loaded library/shelfmark for {len(ms_meta)}/{len(unique_ms_ids)} manuscripts.")
            df["library"] = df["manuscript_id"].astype(str).map(
                lambda m: ms_meta.get(m, {}).get("library", "")
            )
            df["shelfmark_root"] = df["manuscript_id"].astype(str).map(
                lambda m: ms_meta.get(m, {}).get("shelfmark_root", "")
            )

    if args.checkpoint:
        # Compute latents directly from checkpoint (no DB required)
        if "xml_path" in df.columns:
            # Align XML paths with image_paths order
            # Build lookup from full row set, then re-order by unique image_paths list
            xml_by_path = {
                str(row["image_path"]).strip(): (str(row["xml_path"]).strip() if pd.notna(row["xml_path"]) and str(row["xml_path"]).strip() else None)
                for _, row in df.iterrows()
            }
            xml_paths = [xml_by_path.get(p, None) for p in image_paths]
        else:
            xml_paths = [None] * len(image_paths)

        def project(mode: str):
            return _compute_latents_with_checkpoint(
                checkpoint_path=args.checkpoint,
                image_paths=image_paths,
                xml_paths=xml_paths,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                mask_external_background=mode == "masked",
                ink_only_background=mode == "ink-only",
                ink_only_tiles=mode == "ink-only-tiles",
            )

        if args.inference_background == "ensemble":
            original = project("original")
            masked = project("masked")

            def average_vectors(left: dict, right: dict) -> dict:
                out = {}
                for key in left.keys() | right.keys():
                    a, b = left.get(key), right.get(key)
                    if _is_valid_vector(a) and _is_valid_vector(b) and np.asarray(a).shape == np.asarray(b).shape:
                        value = np.asarray(a) + np.asarray(b)
                        norm = np.linalg.norm(value)
                        out[key] = value / norm if norm >= 1e-12 else None
                    else:
                        out[key] = a if _is_valid_vector(a) else b
                return out

            vecs = average_vectors(original[0], masked[0])
            counts = original[1]
            branch_vecs = {
                name: average_vectors(original[2].get(name, {}), masked[2].get(name, {}))
                for name in original[2].keys() | masked[2].keys()
            }
            fusion_subset_vecs = {
                name: average_vectors(original[3].get(name, {}), masked[3].get(name, {}))
                for name in original[3].keys() | masked[3].keys()
            }
        else:
            vecs, counts, branch_vecs, fusion_subset_vecs = project(args.inference_background)
        print(f"Computed {len(vecs)} latent vectors from checkpoint.")
        if fusion_subset_vecs:
            print(f"Computed fused subset embeddings: {sorted(fusion_subset_vecs.keys())}")
    else:
        # Fallback: load from DB as before
        conn = get_db_connection(args.db_config)
        print("Loading latent vectors from database...")
        vecs, counts = load_latent_vectors_for_paths(conn, image_paths)
        conn.close()
        print(f"Found {len(vecs)} vectors in DB.")
        branch_vecs = {"fusion": {}, "tile": {}, "glyph": {}, "word": {}}
        fusion_subset_vecs = {}

    # Build path -> normalized path for lookup (try both). Count gates prevent
    # missing modalities from producing fake similarity through all-zero/dummy
    # streams. The main fused score is treated as visual-grounded for clustering.
    def get_counts(path):
        if pd.isna(path):
            return (None, None, None)
        s = str(path).strip()
        c = counts.get(s)
        if c is not None:
            return c
        return counts.get(os.path.normpath(s), (None, None, None))

    def _has_count(path, count_idx: int) -> bool:
        c = get_counts(path)
        try:
            return c[count_idx] is not None and int(c[count_idx]) > 0
        except Exception:
            return False

    def _lookup_vec(mapping, path):
        if pd.isna(path):
            return None
        s = str(path).strip()
        v = mapping.get(s)
        if _is_valid_vector(v):
            return v
        v = mapping.get(os.path.normpath(s))
        return v if _is_valid_vector(v) else None

    def get_vec(path):
        # Skip images with no visual stream for the main clustering metric.
        if not _has_count(path, 0):
            return None
        return _lookup_vec(vecs, path)

    def get_branch_vec(branch_name, path):
        count_idx_by_branch = {"tile": 0, "glyph": 1, "word": 2}
        if branch_name == "fusion":
            counts_for_path = get_counts(path)
            if not any(c is not None and int(c) > 0 for c in counts_for_path):
                return None
        count_idx = count_idx_by_branch.get(branch_name)
        if count_idx is not None and not _has_count(path, count_idx):
            return None
        branch_map = branch_vecs.get(branch_name, {})
        return _lookup_vec(branch_map, path)

    def get_fusion_subset_vec(mode_name, path):
        mode_map = fusion_subset_vecs.get(mode_name, {})
        return _lookup_vec(mode_map, path)

    dataset_name = args.dataset_name or (
        "stage2_validation" if args.stage2_validation else os.path.splitext(os.path.basename(input_path))[0]
    )
    dataset_source = input_path

    if args.retrieval_only:
        direct_representations: list[tuple[str, object]] = [
            ("similarity_score", get_vec),
            ("fusion_branch_similarity_score", lambda path: get_branch_vec("fusion", path)),
            ("tile_branch_similarity_score", lambda path: get_branch_vec("tile", path)),
            ("glyph_branch_similarity_score", lambda path: get_branch_vec("glyph", path)),
            ("word_branch_similarity_score", lambda path: get_branch_vec("word", path)),
        ]
        for mode_name in sorted(fusion_subset_vecs):
            direct_representations.append((
                f"{mode_name}_similarity_score",
                lambda path, mode=mode_name: get_fusion_subset_vec(mode, path),
            ))
        metrics_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        metrics_df = _direct_retrieval_metrics(
            df,
            direct_representations,
            device=metrics_device,
        )
        score_columns = [name for name, _getter in direct_representations]
        metrics_path = os.path.splitext(output_path)[0] + "_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Wrote metrics to {metrics_path}")
        payload = _compact_metrics_payload(
            metrics_df,
            score_columns,
            dataset_name=dataset_name,
            dataset_source=dataset_source,
            df=df,
            checkpoint_path=args.checkpoint,
            inference_batch_size=args.batch_size,
            inference_num_workers=args.num_workers,
        )
        json_path = os.path.splitext(output_path)[0] + "_metrics.json"
        _write_compact_metrics_json(json_path, payload)
        _write_retrieval_summary_images(payload, os.path.splitext(output_path)[0])
        requested = set(_requested_histogram_columns())
        histogram_representations = [
            item for item in direct_representations if item[0] in requested
        ]
        histogram_path = os.path.splitext(output_path)[0] + "_hist_all.png"
        _try_plot_similarity_histograms_direct(
            df,
            histogram_representations,
            histogram_path,
            device=metrics_device,
            title_suffix="Standalone branches and fusion subsets",
        )
        if os.path.exists(histogram_path):
            print(f"Wrote histogram grid to {histogram_path}")
        return 0

    rows = []
    for k, (i, j) in enumerate(pairs):
        if (k + 1) % 5000 == 0 or k == 0:
            print(f"  Pair {k + 1}/{len(pairs)}")
        row_i = df.iloc[i]
        row_j = df.iloc[j]
        cluster_i = row_i["cluster_id"]
        cluster_j = row_j["cluster_id"]
        are_same = cluster_i == cluster_j
        lib_i = str(row_i["library"]).strip() if "library" in df.columns and pd.notna(row_i["library"]) else ""
        lib_j = str(row_j["library"]).strip() if "library" in df.columns and pd.notna(row_j["library"]) else ""
        are_same_library = bool(lib_i and lib_j and lib_i == lib_j)
        path_i = row_i["image_path"]
        path_j = row_j["image_path"]
        v1 = get_vec(path_i)
        v2 = get_vec(path_j)
        tile_v1 = get_branch_vec("tile", path_i)
        tile_v2 = get_branch_vec("tile", path_j)
        glyph_v1 = get_branch_vec("glyph", path_i)
        glyph_v2 = get_branch_vec("glyph", path_j)
        fusion_v1 = get_branch_vec("fusion", path_i)
        fusion_v2 = get_branch_vec("fusion", path_j)
        word_v1 = get_branch_vec("word", path_i)
        word_v2 = get_branch_vec("word", path_j)
        c1 = get_counts(path_i)
        c2 = get_counts(path_j)
        sim = np.nan
        if v1 is not None and v2 is not None:
            sim = cosine_similarity(v1, v2)
        tile_sim = cosine_similarity(tile_v1, tile_v2) if tile_v1 is not None and tile_v2 is not None else np.nan
        glyph_sim = cosine_similarity(glyph_v1, glyph_v2) if glyph_v1 is not None and glyph_v2 is not None else np.nan
        fusion_sim = cosine_similarity(fusion_v1, fusion_v2) if fusion_v1 is not None and fusion_v2 is not None else np.nan
        word_sim = cosine_similarity(word_v1, word_v2) if word_v1 is not None and word_v2 is not None else np.nan
        out = {}
        for c in df.columns:
            out[f"{c}_1"] = row_i[c]
            out[f"{c}_2"] = row_j[c]
        out["are_they_same_clusters"] = are_same
        out["are_they_same_library"] = are_same_library
        out["similarity_score"] = sim
        out["fusion_branch_similarity_score"] = fusion_sim
        out["tile_branch_similarity_score"] = tile_sim
        out["glyph_branch_similarity_score"] = glyph_sim
        out["word_branch_similarity_score"] = word_sim
        for mode_name in sorted(fusion_subset_vecs.keys()):
            mode_v1 = get_fusion_subset_vec(mode_name, path_i)
            mode_v2 = get_fusion_subset_vec(mode_name, path_j)
            out[f"{mode_name}_similarity_score"] = (
                cosine_similarity(mode_v1, mode_v2)
                if mode_v1 is not None and mode_v2 is not None
                else np.nan
            )
        out["image_1_in_db"] = v1 is not None
        out["image_2_in_db"] = v2 is not None
        out["image_1_num_visual_patches"] = c1[0]
        out["image_1_num_glyphs"] = c1[1]
        out["image_1_num_words"] = c1[2]
        out["image_2_num_visual_patches"] = c2[0]
        out["image_2_num_glyphs"] = c2[1]
        out["image_2_num_words"] = c2[2]
        ms1 = row_i["manuscript_id"] if pd.notna(row_i["manuscript_id"]) else ""
        pic1 = row_i["picture_id"] if pd.notna(row_i["picture_id"]) else ""
        ms2 = row_j["manuscript_id"] if pd.notna(row_j["manuscript_id"]) else ""
        pic2 = row_j["picture_id"] if pd.notna(row_j["picture_id"]) else ""
        out["diagnose_cli"] = (
            f"python Debugs/impact_analysis/diagnose_similarity_issue.py "
            f"--ms1 {ms1} --pic1 {pic1} --ms2 {ms2} --pic2 {pic2}"
        )
        rows.append(out)

    out_df = pd.DataFrame(rows)
    late_fusion_score_columns = _add_late_fusion_score_columns(out_df)
    metrics_rows: list[dict[str, object]] = []
    score_columns = [
        col
        for col in (
            "similarity_score",
            "fusion_branch_similarity_score",
            "tile_branch_similarity_score",
            "glyph_branch_similarity_score",
            "word_branch_similarity_score",
        )
        if col in out_df.columns
    ]
    score_columns.extend(
        sorted(col for col in out_df.columns if col.startswith("fuse_") and col.endswith("_similarity_score"))
    )
    score_columns.extend(late_fusion_score_columns)
    for score_col in score_columns:
        metrics_rows.extend(_pair_separation_metrics(out_df, score_col))
        metrics_rows.extend(_cluster_retrieval_metrics_from_pairs(out_df, score_col))
    metrics_rows.extend(_cluster_retrieval_metrics(df, get_vec))
    metrics_df = pd.DataFrame(metrics_rows)

    _write_outputs(output_path, out_df, metrics_df)
    print(f"Wrote {len(out_df)} rows to {output_path}")

    payload = _compact_metrics_payload(
        metrics_df,
        score_columns,
        dataset_name=dataset_name,
        dataset_source=dataset_source,
        df=df,
        checkpoint_path=args.checkpoint,
        inference_batch_size=args.batch_size,
        inference_num_workers=args.num_workers,
    )
    json_path = os.path.splitext(output_path)[0] + "_metrics.json"
    _write_compact_metrics_json(json_path, payload)
    _write_retrieval_summary_images(payload, os.path.splitext(output_path)[0])

    if not metrics_df.empty:
        _print_all_score_metrics(metrics_df, score_columns)

        retrieval_metrics = metrics_df[metrics_df["metric_set"] == "cluster_retrieval"]
        if not retrieval_metrics.empty:
            print("\n--- Image-level retrieval (full fusion vectors, all images in CSV) ---")
            for name in _PAIR_RETRIEVAL_PRINT_METRICS:
                values = retrieval_metrics.loc[retrieval_metrics["metric"] == name, "value"]
                if len(values):
                    value = values.iloc[0]
                    if name.startswith("n_"):
                        print(f"    {name}: {int(value)}")
                    else:
                        print(f"    {name}: {float(value):.6f}")

    # Plot exactly one histogram figure: a grid with all relevant score modes.
    try:
        plot_path_all = os.path.splitext(output_path)[0] + "_hist_all.png"
        requested = set(_requested_histogram_columns())
        histogram_columns = [col for col in score_columns if col in requested]
        _try_plot_similarity_histograms_multi(
            out_df,
            histogram_columns,
            plot_path_all,
            title_suffix="Standalone branches and fusion subsets",
        )
        if os.path.exists(plot_path_all):
            print(f"Wrote histogram grid to {plot_path_all}")
    except Exception:
        pass

    # Summary of missing similarities
    has_both = out_df["image_1_in_db"] & out_df["image_2_in_db"]
    n_with_sim = has_both.sum()
    n_missing = len(out_df) - n_with_sim
    if n_missing > 0:
        only_1 = (~out_df["image_1_in_db"]) & out_df["image_2_in_db"]
        only_2 = out_df["image_1_in_db"] & (~out_df["image_2_in_db"])
        neither = (~out_df["image_1_in_db"]) & (~out_df["image_2_in_db"])
        print(f"Similarity: {n_with_sim} pairs have score, {n_missing} missing (image_1 not in DB: {only_1.sum()}, image_2 not in DB: {only_2.sum()}, both not in DB: {neither.sum()})")
        n_in_xlsx = len(df)
        n_in_db = len(vecs)
        print(f"Vectors: {n_in_db}/{n_in_xlsx} images from the xlsx exist in geniza_image_latents.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
