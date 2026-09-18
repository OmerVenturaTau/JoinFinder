#!/usr/bin/env python3
"""
Diagnostic tool to investigate why dissimilar images have high similarity scores.

Checks:
1. How many visual patches were extracted for each image
2. How many glyphs were extracted for each image
3. Whether images with high similarity have similar feature counts
4. Visualizes patches/glyphs for problematic image pairs

Similarity source:
  - By default, similarity is loaded from the DB (geniza_image_latents).
  - If --checkpoint is provided, the model is loaded and similarity is computed
    on-the-fly from the checkpoint's latent vectors.

Optional: --relaxed-alto-filters relaxes ALTO word/glyph filtering for this
process only (no Hebrew dict; WC/GC off). Default follows system.py.
"""

import os
import sys
import subprocess
import configparser
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ---------------------------------------------------------------------------
# Early arg parse: must run BEFORE importing model modules so we can patch
# architecture-related system constants while they are still being bound.
# ---------------------------------------------------------------------------
_pre_alto = argparse.ArgumentParser(add_help=False)
_pre_alto.add_argument("--checkpoint", type=str, default="", help=argparse.SUPPRESS)
_pre_alto.add_argument(
    "--relaxed-alto-filters",
    action="store_true",
    help="Ignore Hebrew dict + ALTO WC/GC thresholds for this run.",
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
_pre_alto_args, _ = _pre_alto.parse_known_args()


def _resolve_project_path(path: str) -> str:
    if not path:
        return ""
    return path if os.path.isabs(path) else os.path.join(project_root, path)


from utilities.checkpoint_utils import (  # noqa: E402
    inspect_checkpoint,
    apply_inspection_to_system,
    load_state_dict_with_report,
    warn_if_tile_size_mismatch,
)

_checkpoint_inspection = inspect_checkpoint(_resolve_project_path(_pre_alto_args.checkpoint))
_needs_system_patch = (
    _pre_alto_args.relaxed_alto_filters
    or _pre_alto_args.model_glyph_summary_tokens is not None
    or _pre_alto_args.model_use_word is not None
    or any(
        getattr(_checkpoint_inspection, name) is not None
        for name in ("use_visual_mod", "use_char_mod", "use_word_mod", "glyph_summary_tokens")
    )
)
if _needs_system_patch:
    import system as _system_for_diagnose  # noqa: E402

    if _pre_alto_args.relaxed_alto_filters:
        _system_for_diagnose.USE_HEBREW_DICT_CHECK = False
        _system_for_diagnose.OCR_STRING_CONFIDENCE_THRESHOLD = 0.0
        _system_for_diagnose.OCR_GLYPH_CONFIDENCE_THRESHOLD = 0.0

    apply_inspection_to_system(_checkpoint_inspection, _system_for_diagnose)

    if _pre_alto_args.model_glyph_summary_tokens is not None:
        _system_for_diagnose.GLYPH_NUM_SUMMARY_TOKENS = int(_pre_alto_args.model_glyph_summary_tokens)
    if _pre_alto_args.model_use_word is not None:
        _system_for_diagnose.USE_WORD_MOD = _pre_alto_args.model_use_word == "true"

# ---------------------------------------------------------------------------
# Now safe to import model / dataset modules that snapshot system.* at import.
# ---------------------------------------------------------------------------
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle  # noqa: F401
from torch.utils.data import DataLoader
from torchvision import transforms

from train.dataset import ManuscriptDataset, tile_collate_with_padding
from models import MultiModal
from system import (
    CLUSTERING_DB_CONFIG_PATH,
    CLUSTERING_IMAGES_ROOT,
    CLUSTERING_TILE_SIZE,
    CLUSTERING_TILE_STRIDE,
    CLUSTERING_MAX_TILES_EVAL,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    CHAR_PATCH_SIZE,
    USE_VISUAL_MOD,
    USE_CHAR_MOD,
    USE_WORD_MOD,
    MAX_SELECTED_MANUSCRIPTS,
    BEST_MODEL_PATH,
    GENIZA_IMAGE_LATENTS_TABLE,
    GENIZA_IMAGE_INFORMATION_TABLE,
    GENIZA_KNN_RESULTS_TABLE,
)


# ---------------------------------------------------------------------------
# Checkpoint-based similarity
# ---------------------------------------------------------------------------

def _load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> MultiModal:
    """Load MultiModal model from checkpoint for latent extraction."""
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(project_root, checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

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
    warn_if_tile_size_mismatch(inspection, CLUSTERING_TILE_SIZE, label="Checkpoint")

    model = MultiModal(num_classes=num_classes).to(device)
    load_state_dict_with_report(model, state_dict, label="Checkpoint")
    model.eval()

    if hasattr(model, "modality_dropout_enabled"):
        model.modality_dropout_enabled = False
    if hasattr(model, "token_subsample_enabled"):
        model.token_subsample_enabled = False

    return model


def compute_similarity_from_checkpoint(
    checkpoint_path: str,
    image_paths: list,
    xml_paths: list | None = None,
    batch_size: int = 4,
) -> tuple[dict, dict]:
    """
    Compute normalized latent vectors for the given image paths using a
    checkpoint and return cosine similarity and per-image feature counts.

    Returns (vecs_by_path, counts_by_path).
    """
    assert len(image_paths) > 0
    if xml_paths is None:
        xml_paths = [None] * len(image_paths)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"  Using device for checkpoint projection: {device}")
    model = _load_model_from_checkpoint(checkpoint_path, device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])

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
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=tile_collate_with_padding,
    )

    by_path: dict[str, np.ndarray | None] = {}
    counts_by_path: dict[str, tuple] = {}
    model_for_forward = model.module if hasattr(model, "module") else model

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            (
                tiles, valid_mask, coords, tile_page_segments,
                char_patches, char_valid_mask, glyph_coords, glyph_page_segments,
                char_class_ids, char_metadata, words, word_metadata,
                labels_tensor, batch_paths,
            ) = batch

            tiles = tiles.to(device)
            valid_mask = valid_mask.to(device)
            coords = coords.to(device)
            tile_page_segments = tile_page_segments.to(device)
            char_patches = char_patches.to(device) if char_patches is not None else None
            char_valid_mask = char_valid_mask.to(device) if char_valid_mask is not None else None
            glyph_coords = glyph_coords.to(device) if glyph_coords is not None else None
            glyph_page_segments = glyph_page_segments.to(device) if glyph_page_segments is not None else None
            char_class_ids = char_class_ids.to(device) if char_class_ids is not None else None
            batch_element_indices = torch.arange(tiles.shape[0], dtype=torch.long, device=device)

            _, latents, _ = model_for_forward(
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

            for i, p in enumerate(batch_paths):
                key = (p or "").strip()
                if not key:
                    continue
                vec = latents[i]
                n = np.linalg.norm(vec) if np.isfinite(vec).all() else 0.0
                if n >= 1e-12:
                    vec = vec / n
                    by_path[key] = vec
                else:
                    by_path[key] = None
                n_vis = int(valid_mask[i].sum().item()) if valid_mask is not None else None
                n_glyph = int(char_valid_mask[i].sum().item()) if char_valid_mask is not None else None
                n_word = len(words[i]) if words and i < len(words) and words[i] is not None else 0
                counts_by_path[key] = (n_vis, n_glyph, n_word)

    for key, vec in list(by_path.items()):
        norm_key = os.path.normpath(key)
        if norm_key not in by_path:
            by_path[norm_key] = vec
        if norm_key not in counts_by_path and key in counts_by_path:
            counts_by_path[norm_key] = counts_by_path[key]

    return by_path, counts_by_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _glyph_grid_image(char_patches: torch.Tensor, n_cols: int = 12) -> np.ndarray:
    """Build a single image that is a grid of denormalized glyph patches. char_patches: [M, 3, H, W]."""
    M = char_patches.shape[0] if char_patches.dim() >= 2 else 0
    if M == 0:
        return np.zeros((CHAR_PATCH_SIZE, n_cols * CHAR_PATCH_SIZE, 3), dtype=np.uint8) + 255
    mean = np.array(NORMALIZE_MEAN).reshape(1, 1, 3)
    std = np.array(NORMALIZE_STD).reshape(1, 1, 3)
    n_rows = (M + n_cols - 1) // n_cols
    H, W = CHAR_PATCH_SIZE, CHAR_PATCH_SIZE
    grid = np.zeros((n_rows * H, n_cols * W, 3), dtype=np.float32)
    grid += 1.0
    for i in range(M):
        r, c = i // n_cols, i % n_cols
        patch = char_patches[i]
        if patch.dim() == 3:
            p = patch.permute(1, 2, 0).numpy()
            p = p * std + mean
            p = np.clip(p, 0, 1)
            grid[r * H : (r + 1) * H, c * W : (c + 1) * W] = p
    return (np.clip(grid, 0, 1) * 255).astype(np.uint8)


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


def analyze_image_features(image_path: str, xml_path: str = None, images_root: str = None):
    """
    Extract and count features for a single image.
    Returns dict with counts and sample patches.
    """
    if images_root and not os.path.isabs(image_path):
        full_path = os.path.join(images_root, image_path)
    else:
        full_path = image_path
    
    if not os.path.exists(full_path):
        return {"error": f"Image not found: {full_path}"}
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])
    
    dataset = ManuscriptDataset(
        [full_path],
        ["dummy"],
        transform,
        {"dummy": 0},
        xml_paths=[xml_path] if xml_path else None,
        patch_size=CLUSTERING_TILE_SIZE,
        stride=CLUSTERING_TILE_STRIDE,
        max_tiles_per_image=CLUSTERING_MAX_TILES_EVAL,
    )
    
    try:
        tiles, coords, tile_page_segments, char_patches, char_metadata, words, word_metadata, label, path = dataset[0]
    except Exception as e:
        return {"error": f"Failed to extract features: {e}"}
    
    n_tiles = tiles.shape[0] if len(tiles.shape) > 1 else 0
    n_glyphs = char_patches.shape[0] if len(char_patches.shape) > 1 else 0
    
    try:
        pil_img = Image.open(full_path).convert('RGB')
        img_array = np.array(pil_img)
    except Exception:
        img_array = None

    xml_exists = bool(xml_path and os.path.exists(xml_path))
    
    return {
        "image_path": image_path,
        "n_visual_patches": n_tiles,
        "n_glyphs": n_glyphs,
        "n_words": len(words) if words else 0,
        "image_array": img_array,
        "tiles": tiles,
        "coords": coords,
        "char_patches": char_patches,
        "char_metadata": char_metadata,
        "xml_path": xml_path,
        "xml_exists": xml_exists,
    }


def compare_two_images(image1_path: str, image2_path: str, xml1_path: str = None, xml2_path: str = None,
                       images_root: str = None, output_dir: str = "aftertune/diagnosis", db_config_path: str = None,
                       checkpoint_path: str = None):
    """
    Compare two images and diagnose why they might have high similarity.
    When checkpoint_path is provided, similarity is computed from the model
    instead of the DB.
    """
    os.makedirs(output_dir, exist_ok=True)
    if db_config_path is None:
        db_config_path = CLUSTERING_DB_CONFIG_PATH

    report_lines = []
    def _log(s=""):
        print(s)
        report_lines.append(s)

    _log(f"\n{'='*70}")
    _log(f"  DIAGNOSING IMAGE SIMILARITY ISSUE")
    _log(f"{'='*70}")
    _log(f"  Output folder: {output_dir}")
    _log(f"\nImage 1: {image1_path}")
    _log(f"Image 2: {image2_path}")

    # --- Similarity ---
    sim = None
    ckpt_counts = {}

    if checkpoint_path:
        _log(f"\n  Checkpoint: {checkpoint_path}")
        try:
            full1 = os.path.join(images_root, image1_path) if images_root and not os.path.isabs(image1_path) else image1_path
            full2 = os.path.join(images_root, image2_path) if images_root and not os.path.isabs(image2_path) else image2_path
            vecs, counts = compute_similarity_from_checkpoint(
                checkpoint_path,
                [full1, full2],
                xml_paths=[xml1_path, xml2_path],
            )
            v1 = vecs.get(full1) or vecs.get(os.path.normpath(full1))
            v2 = vecs.get(full2) or vecs.get(os.path.normpath(full2))
            ckpt_counts = counts
            if v1 is not None and v2 is not None:
                sim = float(np.dot(v1, v2))
                _log(f"\n  Similarity (from checkpoint): {sim:.4f}")
            else:
                missing = []
                if v1 is None:
                    missing.append("image 1")
                if v2 is None:
                    missing.append("image 2")
                _log(f"\n  Similarity (from checkpoint): could not compute — missing vector for {', '.join(missing)}")
        except Exception as e:
            _log(f"\n  Similarity (from checkpoint): error — {e}")
    else:
        try:
            conn = get_db_connection(db_config_path)
            sim, err = get_similarity_from_db(conn, image1_path, image2_path)
            conn.close()
            if err:
                _log(f"\n  Similarity (from DB): could not compute — {err}")
            else:
                _log(f"\n  Similarity (from DB geniza_image_latents): {sim:.4f}")
        except Exception as e:
            _log(f"\n  Similarity (from DB): error — {e}")

    # Analyze both images
    feat1 = analyze_image_features(image1_path, xml1_path, images_root)
    feat2 = analyze_image_features(image2_path, xml2_path, images_root)
    
    if "error" in feat1:
        _log(f"\nERROR analyzing image 1: {feat1['error']}")
        return
    if "error" in feat2:
        _log(f"\nERROR analyzing image 2: {feat2['error']}")
        return

    _log(f"\n{'─'*70}")
    _log("  FEATURE COUNTS")
    _log(f"{'─'*70}")
    _log(f"Image 1:")
    _log(f"  Visual patches: {feat1['n_visual_patches']}")
    _log(f"  Glyphs:         {feat1['n_glyphs']}")
    _log(f"  Words:          {feat1['n_words']}")
    _log(f"  XML path:       {feat1.get('xml_path') or xml1_path or 'None'}")
    _log(f"  XML exists:     {feat1.get('xml_exists') if 'xml_exists' in feat1 else bool(xml1_path and os.path.exists(xml1_path))}")
    _log(f"\nImage 2:")
    _log(f"  Visual patches: {feat2['n_visual_patches']}")
    _log(f"  Glyphs:         {feat2['n_glyphs']}")
    _log(f"  Words:          {feat2['n_words']}")
    _log(f"  XML path:       {feat2.get('xml_path') or xml2_path or 'None'}")
    _log(f"  XML exists:     {feat2.get('xml_exists') if 'xml_exists' in feat2 else bool(xml2_path and os.path.exists(xml2_path))}")

    if ckpt_counts:
        full1 = os.path.join(images_root, image1_path) if images_root and not os.path.isabs(image1_path) else image1_path
        full2 = os.path.join(images_root, image2_path) if images_root and not os.path.isabs(image2_path) else image2_path
        c1 = ckpt_counts.get(full1) or ckpt_counts.get(os.path.normpath(full1))
        c2 = ckpt_counts.get(full2) or ckpt_counts.get(os.path.normpath(full2))
        if c1:
            _log(f"\n  [Checkpoint] Image 1 valid tokens: visual={c1[0]}, glyphs={c1[1]}, words={c1[2]}")
        if c2:
            _log(f"  [Checkpoint] Image 2 valid tokens: visual={c2[0]}, glyphs={c2[1]}, words={c2[2]}")

    low_features_threshold = 3
    both_low = (feat1['n_visual_patches'] < low_features_threshold and
                feat2['n_visual_patches'] < low_features_threshold)
    both_no_glyphs = (feat1['n_glyphs'] == 0 and feat2['n_glyphs'] == 0)

    _log(f"\n{'─'*70}")
    _log("  DIAGNOSIS")
    _log(f"{'─'*70}")
    if both_low:
        _log("⚠️  WARNING: Both images have very few visual patches (< 3)")
        _log("   This can cause them to produce similar generic representations.")
    if both_no_glyphs:
        _log("⚠️  WARNING: Both images have zero glyphs extracted")
        _log("   Possible reasons:")
        _log("   - Missing or empty XML files")
        _log("   - XML parsing failed")
        _log("   - No valid glyphs found in XML (confidence/quality filters)")
        _log("   - Image is mostly blank/background")

    if feat1['n_visual_patches'] == 0 or feat2['n_visual_patches'] == 0:
        _log("⚠️  CRITICAL: One or both images have ZERO visual patches!")
        _log("   This will cause the model to produce near-identical latent vectors.")
    
    stem1 = Path(image1_path).stem
    stem2 = Path(image2_path).stem
    comparison_dir = os.path.join(output_dir, f"{stem1}-{stem2}")
    subdir1 = os.path.join(comparison_dir, stem1)
    subdir2 = os.path.join(comparison_dir, stem2)
    os.makedirs(comparison_dir, exist_ok=True)
    os.makedirs(subdir1, exist_ok=True)
    os.makedirs(subdir2, exist_ok=True)

    if feat1.get('image_array') is not None and feat2.get('image_array') is not None:
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes[0, 0].imshow(feat1['image_array'])
            axes[0, 0].set_title(f"Image 1\n{feat1['n_visual_patches']} patches, {feat1['n_glyphs']} glyphs")
            axes[0, 0].axis('off')
            axes[0, 1].imshow(feat2['image_array'])
            axes[0, 1].set_title(f"Image 2\n{feat2['n_visual_patches']} patches, {feat2['n_glyphs']} glyphs")
            axes[0, 1].axis('off')
            grid1 = _glyph_grid_image(feat1['char_patches'])
            axes[1, 0].imshow(grid1)
            axes[1, 0].set_title(f"Glyphs extracted (Image 1): {feat1['n_glyphs']}")
            axes[1, 0].axis('off')
            grid2 = _glyph_grid_image(feat2['char_patches'])
            axes[1, 1].imshow(grid2)
            axes[1, 1].set_title(f"Glyphs extracted (Image 2): {feat2['n_glyphs']}")
            axes[1, 1].axis('off')
            plt.tight_layout()
            comparison_file = os.path.join(comparison_dir, "comparison.png")
            plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
            _log(f"\n✓ Comparison saved to: {comparison_file}")
            plt.close()
        except Exception as e:
            _log(f"\n⚠️  Could not create comparison: {e}")

    _log(f"\n{'─'*70}")
    _log("  PATCHES / WORDS / GLYPHS (visualize_all)")
    _log(f"{'─'*70}")
    full1 = os.path.join(images_root, image1_path) if images_root and not os.path.isabs(image1_path) else image1_path
    full2 = os.path.join(images_root, image2_path) if images_root and not os.path.isabs(image2_path) else image2_path
    _log(f"Image 1: {os.path.basename(full1)} → {subdir1}")
    if run_visualize_all_for_image(image1_path, xml1_path, images_root, subdir1):
        _log(f"  → {stem1}_viz_patches.jpg, _viz_words.jpg, _viz_chars.jpg")
    _log(f"Image 2: {os.path.basename(full2)} → {subdir2}")
    if run_visualize_all_for_image(image2_path, xml2_path, images_root, subdir2):
        _log(f"  → {stem2}_viz_patches.jpg, _viz_words.jpg, _viz_chars.jpg")

    _log(f"\n{'='*70}\n")
    try:
        report_file = os.path.join(comparison_dir, "report.txt")
        with open(report_file, "w") as f:
            f.write("\n".join(report_lines))
        print(f"Report written to: {report_file}")
    except Exception as e:
        print(f"Could not write report.txt: {e}")
    print(f"Comparison outputs in: {comparison_dir}")


def query_high_similarity_pairs(db_config_path: str, limit: int = 10, min_similarity: float = 0.95):
    """
    Query the database for image pairs with unexpectedly high similarity.
    """
    conn = get_db_connection(db_config_path)
    import psycopg2.extras
    
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(f"""
            SELECT 
                q1.query_manuscript_id AS ms1,
                q1.query_image_path AS img1,
                q2.query_manuscript_id AS ms2,
                q2.query_image_path AS img2,
                q1.similarity_score AS sim
            FROM {GENIZA_KNN_RESULTS_TABLE} q1
            JOIN {GENIZA_KNN_RESULTS_TABLE} q2
                ON q1.query_image_path = q2.query_image_path
            WHERE q1.neighbor_rank = 1
              AND q2.neighbor_rank = 1
              AND q1.query_manuscript_id != q2.query_manuscript_id
              AND q1.similarity_score >= %s
            ORDER BY q1.similarity_score DESC
            LIMIT %s
        """, (min_similarity, limit))
        
        pairs = cur.fetchall()
    
    conn.close()
    return pairs


def get_similarity_from_db(conn, image1_path: str, image2_path: str, vector_column: str = "latent_vector"):
    """
    Load latent vectors for two images from geniza_image_latents and return cosine similarity.
    Returns (similarity, None) or (None, error_message) if one/both vectors missing.
    """
    import psycopg2.extras
    path1 = os.path.normpath(image1_path)
    path2 = os.path.normpath(image2_path)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            f"""
            SELECT image_path, {vector_column}::text AS vec
            FROM {GENIZA_IMAGE_LATENTS_TABLE}
            WHERE image_path IN (%s, %s) AND {vector_column} IS NOT NULL
            """,
            (image1_path, image2_path),
        )
        rows = cur.fetchall()
    by_path = {}
    for r in rows:
        key = os.path.normpath(r["image_path"]) if r["image_path"] else None
        if key:
            by_path[key] = r["vec"]
    for r in rows:
        if r["image_path"] and r["image_path"] not in by_path:
            by_path[r["image_path"]] = r["vec"]
    v1_str = by_path.get(path1) or by_path.get(image1_path)
    v2_str = by_path.get(path2) or by_path.get(image2_path)
    if not v1_str:
        return None, f"No latent vector in DB for image 1: {image1_path}"
    if not v2_str:
        return None, f"No latent vector in DB for image 2: {image2_path}"
    try:
        v1 = np.array([float(x) for x in v1_str.strip("[]").split(",")])
        v2 = np.array([float(x) for x in v2_str.strip("[]").split(",")])
    except Exception as e:
        return None, f"Failed to parse vectors: {e}"
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return None, "One or both vectors have zero norm"
    sim = float(np.dot(v1, v2) / (n1 * n2))
    return sim, None


def run_visualize_all_for_image(image_path: str, xml_path: str, images_root: str, output_dir: str) -> bool:
    """Run Drafts/Visualization/visualize_all.py for one image. Returns True on success."""
    full_path = os.path.join(images_root, image_path) if images_root and not os.path.isabs(image_path) else image_path
    if not os.path.exists(full_path):
        print(f"  Skip visualization: image not found at {full_path}")
        return False
    viz_script = os.path.join(project_root, "Drafts", "Visualization", "visualize_all.py")
    if not os.path.exists(viz_script):
        print(f"  Skip visualization: script not found at {viz_script}")
        return False
    cmd = [sys.executable, viz_script, "--image", full_path, "--out-dir", output_dir]
    if xml_path and os.path.exists(xml_path):
        cmd.extend(["--xml", xml_path])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  visualize_all failed for {os.path.basename(full_path)}: {result.stderr[:200] if result.stderr else result.stdout[:200]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  visualize_all timed out for {os.path.basename(full_path)}")
        return False
    except Exception as e:
        print(f"  visualize_all error for {os.path.basename(full_path)}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Diagnose why dissimilar images have high similarity")
    parser.add_argument("--image1", type=str, help="Image path for first image (as in geniza_image_information.image_path)")
    parser.add_argument("--image2", type=str, help="Image path for second image (as in geniza_image_information.image_path)")
    parser.add_argument("--xml1", type=str, default=None, help="Override XML path for image1 (otherwise read from geniza_image_information)")
    parser.add_argument("--xml2", type=str, default=None, help="Override XML path for image2 (otherwise read from geniza_image_information)")
    parser.add_argument("--ms1", type=str, default=None, help="manuscript_id for first image (geniza_image_information)")
    parser.add_argument("--pic1", type=str, default=None, help="picture_id for first image (geniza_image_information)")
    parser.add_argument("--ms2", type=str, default=None, help="manuscript_id for second image (geniza_image_information)")
    parser.add_argument("--pic2", type=str, default=None, help="picture_id for second image (geniza_image_information)")
    parser.add_argument("--images-root", type=str, default=CLUSTERING_IMAGES_ROOT)
    parser.add_argument("--db-config", type=str, default=CLUSTERING_DB_CONFIG_PATH)
    parser.add_argument("--find-problematic", action="store_true", help="Find high-similarity pairs from DB")
    parser.add_argument("--min-similarity", type=float, default=0.95, help="Minimum similarity threshold")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output folder (default: Debugs/impact_analysis/diagnosis/<ms1>-<ms2>)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Optional model checkpoint (.pth). If provided, similarity is computed from this checkpoint instead of loaded from DB.",
    )
    parser.add_argument(
        "--relaxed-alto-filters",
        action="store_true",
        help="Ignore Hebrew dict + ALTO WC/GC thresholds for this run (affects feature extraction).",
    )
    parser.add_argument(
        "--model-glyph-summary-tokens",
        type=int,
        default=None,
        help="Evaluation-only compatibility override for old checkpoints trained with a different glyph summary-token count.",
    )
    parser.add_argument(
        "--model-use-word",
        choices=("true", "false"),
        default=None,
        help="Evaluation-only compatibility override for old checkpoints trained with/without the word branch.",
    )
    
    args = parser.parse_args()
    
    def _safe_folder(s: str) -> str:
        if not s:
            return "unknown"
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(s))
    
    if args.find_problematic:
        print("Finding problematic high-similarity pairs from database...")
        pairs = query_high_similarity_pairs(args.db_config, limit=20, min_similarity=args.min_similarity)
        print(f"\nFound {len(pairs)} high-similarity pairs:")
        for i, p in enumerate(pairs[:5], 1):
            print(f"\n{i}. Similarity: {p['sim']:.4f}")
            print(f"   Image 1: {p['img1']}")
            print(f"   Image 2: {p['img2']}")
    else:
        image1_path = args.image1
        image2_path = args.image2
        xml1_path = args.xml1
        xml2_path = args.xml2
        ms1 = args.ms1
        ms2 = args.ms2

        if (args.ms1 and args.pic1) or (args.ms2 and args.pic2) or (xml1_path is None or xml2_path is None) or (ms1 is None and image1_path) or (ms2 is None and image2_path):
            conn = get_db_connection(args.db_config)
            import psycopg2.extras
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if args.ms1 and args.pic1:
                    cur.execute(
                        f"""
                        SELECT image_path, xml_path, manuscript_id
                        FROM {GENIZA_IMAGE_INFORMATION_TABLE}
                        WHERE manuscript_id = %s AND picture_id = %s
                        LIMIT 1
                        """,
                        (args.ms1, args.pic1),
                    )
                    row1 = cur.fetchone()
                    if row1:
                        image1_path = row1["image_path"]
                        if xml1_path is None:
                            xml1_path = (row1.get("xml_path") or "").strip() or None
                        if ms1 is None:
                            ms1 = (row1.get("manuscript_id") or "").strip() or args.ms1
                if image1_path and (xml1_path is None or ms1 is None):
                    cur.execute(
                        f"""
                        SELECT xml_path, manuscript_id
                        FROM {GENIZA_IMAGE_INFORMATION_TABLE}
                        WHERE image_path = %s
                        LIMIT 1
                        """,
                        (image1_path,),
                    )
                    row1b = cur.fetchone()
                    if row1b:
                        if xml1_path is None:
                            xml1_path = (row1b.get("xml_path") or "").strip() or None
                        if ms1 is None:
                            ms1 = (row1b.get("manuscript_id") or "").strip()

                if args.ms2 and args.pic2:
                    cur.execute(
                        f"""
                        SELECT image_path, xml_path, manuscript_id
                        FROM {GENIZA_IMAGE_INFORMATION_TABLE}
                        WHERE manuscript_id = %s AND picture_id = %s
                        LIMIT 1
                        """,
                        (args.ms2, args.pic2),
                    )
                    row2 = cur.fetchone()
                    if row2:
                        image2_path = row2["image_path"]
                        if xml2_path is None:
                            xml2_path = (row2.get("xml_path") or "").strip() or None
                        if ms2 is None:
                            ms2 = (row2.get("manuscript_id") or "").strip() or args.ms2
                if image2_path and (xml2_path is None or ms2 is None):
                    cur.execute(
                        f"""
                        SELECT xml_path, manuscript_id
                        FROM {GENIZA_IMAGE_INFORMATION_TABLE}
                        WHERE image_path = %s
                        LIMIT 1
                        """,
                        (image2_path,),
                    )
                    row2b = cur.fetchone()
                    if row2b:
                        if xml2_path is None:
                            xml2_path = (row2b.get("xml_path") or "").strip() or None
                        if ms2 is None:
                            ms2 = (row2b.get("manuscript_id") or "").strip()
            conn.close()

        if image1_path and image2_path:
            if args.output_dir:
                output_dir = os.path.join(project_root, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
            else:
                output_dir = os.path.join(project_root, "Debugs", "impact_analysis", "diagnosis", f"{_safe_folder(ms1)}-{_safe_folder(ms2)}")
            compare_two_images(
                image1_path,
                image2_path,
                xml1_path,
                xml2_path,
                args.images_root,
                output_dir,
                args.db_config,
                checkpoint_path=_resolve_project_path(args.checkpoint) if args.checkpoint else None,
            )
        else:
            parser.print_help()
            print("\nExample usage:")
            print("  # By explicit paths (must match geniza_image_information.image_path)")
            print("  python Debugs/impact_analysis/diagnose_similarity_issue.py --image1 9900.../IE....jpg --image2 9900.../IE....jpg")
            print("  # Or by manuscript_id + picture_id:")
            print("  python Debugs/impact_analysis/diagnose_similarity_issue.py --ms1 9900... --pic1 IE....jpg --ms2 9900... --pic2 IE....jpg")
            print("  # With a checkpoint for on-the-fly similarity:")
            print("  python Debugs/impact_analysis/diagnose_similarity_issue.py --ms1 9900... --pic1 IE....jpg --ms2 9900... --pic2 IE....jpg --checkpoint path/to/model.pth")
            print("  # With relaxed ALTO filters:")
            print("  python Debugs/impact_analysis/diagnose_similarity_issue.py --ms1 9900... --pic1 IE....jpg --ms2 9900... --pic2 IE....jpg --checkpoint path/to/model.pth --relaxed-alto-filters")
            print("  # Or to search for problematic pairs:")
            print("  python Debugs/impact_analysis/diagnose_similarity_issue.py --find-problematic")


if __name__ == "__main__":
    main()
