"""
Glyph vs. Tile Swap Sensitivity Test.

This script loads a trained checkpoint, selects two images from the DB‑driven
dataset (same pipeline as training), and runs two kinds of ablation:

1. Glyph swap (keep tiles A, swap glyphs from B)
2. Tile swap  (keep glyphs A, swap tiles from B)
3. Full grid: for every non-empty subset of {tiles, glyphs, words}, hybrid forward
   (A-centric: named modalities from B, rest from A; and symmetrically for B)

For each image, it reports:
- Original logits and latent vector
- Swapped logits and latent vector
- Cosine similarity and Δ = 1 − cos(e_orig, e_swapped)
- Pre-fusion mean-pooled branch vectors (tiles, glyphs, words): cos / Δ between A and B

Interpretation:
- If logits / embeddings barely change (Δ ≈ 0): glyph (or tile) branch is being
  ignored for that swap.
- If they shift noticeably (Δ larger): that modality matters.

Pair-type filtering (--pair-type):
  all : any cross-manuscript pair (default)
  oo  : both manuscripts are oriental
  nn  : both manuscripts are non-oriental
  on  : one oriental, one non-oriental (cross-type)

Usage (examples):
    conda activate DeepEnv
    python debug_tools/impact_analysis/swap_sensitivity.py --checkpoint PATH_TO_PTH
    python debug_tools/impact_analysis/swap_sensitivity.py --checkpoint PATH_TO_PTH --source geniza
    python debug_tools/impact_analysis/swap_sensitivity.py --checkpoint PATH_TO_PTH --pair-seed 1234
    python debug_tools/impact_analysis/swap_sensitivity.py --checkpoint PATH_TO_PTH --split train --index-a 0 --index-b 1
    python debug_tools/impact_analysis/swap_sensitivity.py --checkpoint PATH_TO_PTH --pair-type on
    python debug_tools/impact_analysis/swap_sensitivity.py --checkpoint PATH_TO_PTH --pair-type oo --num-tests 5
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from itertools import combinations
from typing import Any, Dict, FrozenSet, List, Tuple

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path (so "import system", "from models import MultiModal", etc. work)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Parse --checkpoint early so we can patch system constants BEFORE importing model modules
# (which snapshot system.* values at import time). Older checkpoints (e.g. d_model=1200) need this.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--checkpoint", type=str, default=None, help=argparse.SUPPRESS)
_pre.add_argument("--model-d-model", type=int, default=None, help=argparse.SUPPRESS)
_pre.add_argument("--model-latent-dim", type=int, default=None, help=argparse.SUPPRESS)
_pre.add_argument(
    "--model-glyph-summary-tokens",
    type=int,
    default=None,
    help=argparse.SUPPRESS,
)
_pre_args, _ = _pre.parse_known_args()


def _resolve_project_path(path: str) -> str:
    if not path:
        return ""
    return path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)


import system  # noqa: E402

_checkpoint_path_early = _resolve_project_path(_pre_args.checkpoint or system.BEST_MODEL_PATH)
if _checkpoint_path_early and os.path.exists(_checkpoint_path_early):
    from utilities.checkpoint_utils import (  # noqa: E402
        apply_inspection_to_system,
        inspect_checkpoint,
    )

    _checkpoint_inspection = inspect_checkpoint(_checkpoint_path_early)
    apply_inspection_to_system(_checkpoint_inspection, system)
    if _pre_args.model_glyph_summary_tokens is not None:
        system.GLYPH_NUM_SUMMARY_TOKENS = int(_pre_args.model_glyph_summary_tokens)
    if _pre_args.model_d_model is not None:
        system.D_MODEL = int(_pre_args.model_d_model)
    if _pre_args.model_latent_dim is not None:
        system.LATENT_DIM = int(_pre_args.model_latent_dim)
else:
    _checkpoint_inspection = None

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid

from system import (  # noqa: E402
    BASE_DIR,
    D_MODEL,
    GENIZA_IMAGE_BASE,
    GENIZA_IMAGE_INFORMATION_TABLE,
    GENIZA_IMAGE_LATENTS_TABLE,
    LABEL_HEAD,
    MAX_CHARS_PER_IMAGE,
    MAX_TILES_EVAL,
    MAX_WORDS_PER_IMAGE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
)

from models import MultiModal  # noqa: E402
from train.split_data import build_splits  # noqa: E402
from train.dataset import ManuscriptDataset, tile_collate_with_padding  # noqa: E402
from train.db_loader import get_db_connection, get_table_as_df  # noqa: E402
from tasks.label_heads.factory import create_label_head  # noqa: E402
from tasks.label_heads.base import build_label_maps  # noqa: E402
from utilities.checkpoint_utils import load_state_dict_with_report  # noqa: E402


def _fetch_oriental_map() -> dict:
    """
    Query the DB to build a mapping {manuscript_id (str) -> is_oriental (bool | None)}.
    Returns an empty dict if the column does not exist or the DB is unreachable.
    """
    try:
        from system import PRETRAIN_TABLE_NAME
        conn = get_db_connection()
        try:
            df = get_table_as_df(conn, PRETRAIN_TABLE_NAME)
        finally:
            conn.close()

        if "is_oriental" not in df.columns or "manuscript_id" not in df.columns:
            print("[Oriental] 'is_oriental' column not found in DB table; oriental filtering disabled.")
            return {}

        result: dict = {}
        for _, row in df[["manuscript_id", "is_oriental"]].drop_duplicates("manuscript_id").iterrows():
            mid = str(row["manuscript_id"])
            val = row["is_oriental"]
            if val is None or (isinstance(val, float) and val != val):  # NaN check
                result[mid] = None
            elif isinstance(val, bool):
                result[mid] = val
            elif isinstance(val, str):
                result[mid] = val.strip().lower() in ("true", "t", "1", "yes")
            else:
                result[mid] = bool(val)
        print(f"[Oriental] Loaded is_oriental for {len(result)} manuscripts.")
        return result
    except Exception as exc:  # noqa: BLE001
        print(f"[Oriental] Could not load is_oriental from DB: {exc}")
        return {}


def _build_dataset(
    split: str,
    verbose: bool = True,
) -> Tuple[ManuscriptDataset, dict, dict]:
    """
    Build a ManuscriptDataset for a given split ('train' | 'val' | 'test'),
    mirroring the training pipeline (DB → build_splits → label_head).
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Invalid split '{split}'. Must be one of: train, val, test.")

    # Build DB-driven splits (uses system.TRAINING_MODE internally)
    splits, split_stats = build_splits(BASE_DIR)

    # Label head (usually manuscript_id)
    label_head = create_label_head(name=LABEL_HEAD)
    flat = label_head.flatten_splits(splits)

    if split == "train":
        paths, labels, xmls = flat.train_paths, flat.train_labels, flat.train_xmls
    elif split == "val":
        paths, labels, xmls = flat.val_paths, flat.val_labels, flat.val_xmls
    else:
        paths, labels, xmls = flat.test_paths, flat.test_labels, flat.test_xmls

    if not paths:
        raise RuntimeError(f"No samples found in split '{split}'.")

    # Build label2idx / idx2label over ALL splits to match training
    all_labels = flat.train_labels + flat.val_labels + flat.test_labels
    label2idx, idx2label = build_label_maps(all_labels=all_labels)

    # Simple eval-style transform for tiles (same as main.py eval_transform)
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )

    dataset = ManuscriptDataset(
        paths,
        labels,
        eval_transform,
        label2idx,
        xml_paths=xmls,
        max_tiles_per_image=MAX_TILES_EVAL,
        split=split,
    )

    if verbose:
        print(f"[Dataset] Split='{split}', num_images={len(dataset)}, num_classes={len(label2idx)}")

    return dataset, label2idx, idx2label


def _build_geniza_dataset(verbose: bool = True) -> Tuple[ManuscriptDataset, dict, dict]:
    """Build a Geniza dataset containing only full-capacity multimodal pages."""
    required_tiles = int(MAX_TILES_EVAL)
    required_glyphs = int(MAX_CHARS_PER_IMAGE)
    required_words = int(MAX_WORDS_PER_IMAGE)
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT i.manuscript_id, i.image_path, i.xml_path,
                       i.parent_directory, i.picture_id
                FROM {GENIZA_IMAGE_INFORMATION_TABLE} i
                JOIN {GENIZA_IMAGE_LATENTS_TABLE} l
                  ON l.image_path = i.image_path
                WHERE i.manuscript_id IS NOT NULL
                  AND i.image_path IS NOT NULL
                  AND i.image_path != ''
                  AND l.num_visual_patches >= %s
                  AND l.num_glyphs >= %s
                  AND l.num_words >= %s
                ORDER BY i.manuscript_id, i.image_path
                """,
                (required_tiles, required_glyphs, required_words),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    paths: List[str] = []
    labels: List[str] = []
    xmls: List[str | None] = []
    for manuscript_id, image_path, xml_path, parent_directory, picture_id in rows:
        manuscript_id = str(manuscript_id)
        if os.path.isabs(str(image_path)):
            local_path = str(image_path)
        else:
            local_path = os.path.join(
                GENIZA_IMAGE_BASE,
                manuscript_id,
                str(parent_directory or ""),
                str(picture_id or image_path),
            )
        paths.append(local_path)
        labels.append(manuscript_id)
        xmls.append((str(xml_path).strip() if xml_path else None))

    if not paths:
        raise RuntimeError(
            f"No usable image rows found in Geniza table {GENIZA_IMAGE_INFORMATION_TABLE!r}."
        )

    label2idx, idx2label = build_label_maps(all_labels=labels)
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )
    dataset = ManuscriptDataset(
        paths,
        labels,
        eval_transform,
        label2idx,
        xml_paths=xmls,
        max_tiles_per_image=MAX_TILES_EVAL,
        split="val",
    )
    dataset.required_feature_counts = {
        "tiles": required_tiles,
        "glyphs": required_glyphs,
        "words": required_words,
    }

    if verbose:
        print(
            f"[Dataset] Source='geniza' table={GENIZA_IMAGE_INFORMATION_TABLE!r}, "
            f"num_images={len(dataset)}, num_manuscripts={len(label2idx)}, "
            f"required_counts=(tiles>={required_tiles}, glyphs>={required_glyphs}, "
            f"words>={required_words}) from {GENIZA_IMAGE_LATENTS_TABLE!r}"
        )

    return dataset, label2idx, idx2label


def _load_model(checkpoint_path: str, num_classes: int, device: torch.device) -> MultiModal:
    """
    Create MultiModal model and load weights from checkpoint (pretrain or finetune).
    """
    checkpoint_path = _resolve_project_path(checkpoint_path)
    print(
        f"[Model] Initializing MultiModal (num_classes={num_classes}, d_model={D_MODEL})"
    )
    model = MultiModal(num_classes=num_classes, d_model=int(D_MODEL)).to(device)
    # Ensure deterministic eval-time behavior
    model.eval()

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    print(f"[Model] Loading checkpoint from: {checkpoint_path}")
    try:
        from utilities.checkpoint_utils import inspect_checkpoint

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

        raw = inspection.raw
        if isinstance(raw, dict) and state_dict is not raw:
            epoch = raw.get("epoch", "unknown")
            val_acc = raw.get("val_accuracy", "unknown")
            print(f"[Model] Checkpoint metadata: epoch={epoch}, val_accuracy={val_acc}")
        else:
            print("[Model] Checkpoint format: raw state_dict")

        load_state_dict_with_report(model, state_dict, label="Model")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {e}") from e

    # Safety: disable modality dropout and token subsampling for ALL eval runs
    # so no stochastic dropping of tiles/glyphs can affect these diagnostics.
    if hasattr(model, "modality_dropout_enabled"):
        model.modality_dropout_enabled = False
    if hasattr(model, "token_subsample_enabled"):
        model.token_subsample_enabled = False
    print(
        "[Model] Eval-time regularization disabled: "
        f"modality_dropout_enabled={getattr(model, 'modality_dropout_enabled', None)}, "
        f"token_subsample_enabled={getattr(model, 'token_subsample_enabled', None)}"
    )

    return model.to(device)


def _select_indices(dataset_len: int, index_a: int | None, index_b: int | None) -> Tuple[int, int]:
    """
    Resolve and validate two indices into the dataset.
    """
    if index_a is None:
        index_a = 0
    if index_b is None:
        index_b = 1 if dataset_len > 1 else 0

    if not (0 <= index_a < dataset_len):
        raise IndexError(f"index-a={index_a} out of range [0, {dataset_len - 1}]")
    if not (0 <= index_b < dataset_len):
        raise IndexError(f"index-b={index_b} out of range [0, {dataset_len - 1}]")
    if index_a == index_b:
        print(
            f"[Warn] index-a ({index_a}) == index-b ({index_b}); "
            "results will be degenerate. Consider choosing different indices."
        )
    return index_a, index_b


def _compute_delta(e_orig: torch.Tensor, e_swapped: torch.Tensor) -> float:
    """
    Compute Δ = 1 − cos(e_orig, e_swapped) for 1×D tensors.
    """
    e_orig = F.normalize(e_orig, dim=1, eps=1e-8)
    e_swapped = F.normalize(e_swapped, dim=1, eps=1e-8)
    cos_sim = F.cosine_similarity(e_orig, e_swapped, dim=1).item()
    return float(1.0 - cos_sim)


_MOD_NAMES = ("tiles", "glyphs", "words")


def _nonempty_modal_subsets() -> List[FrozenSet[str]]:
    subs: List[FrozenSet[str]] = []
    for r in range(1, len(_MOD_NAMES) + 1):
        for c in combinations(_MOD_NAMES, r):
            subs.append(frozenset(c))
    return subs


def _subset_label(sub: FrozenSet[str]) -> str:
    """Compact tag, e.g. T, W, TG, TGW (tiles/glyphs/words order)."""
    return "".join(x[0].upper() for x in _MOD_NAMES if x in sub)


@torch.no_grad()
def _hybrid_latent_single(
    model: MultiModal,
    device: torch.device,
    *,
    src: Dict[str, int],
    tiles: torch.Tensor,
    tile_valid_mask: torch.Tensor,
    coords: torch.Tensor,
    tile_page_segments: torch.Tensor,
    char_patches: torch.Tensor,
    char_valid_mask: torch.Tensor,
    glyph_coords: torch.Tensor,
    glyph_page_segments: torch.Tensor,
    char_class_ids: torch.Tensor,
    words: List[Any],
    word_metadata: List[Any],
    path_line: str,
) -> torch.Tensor:
    """One-sample forward: per-modality batch index in ``src`` (0=A, 1=B)."""
    ti, gi, wi = src["tiles"], src["glyphs"], src["words"]
    _, lat, _ = model(
        tiles=tiles[ti : ti + 1],
        tile_coords=coords[ti : ti + 1],
        tile_valid_mask=tile_valid_mask[ti : ti + 1],
        tile_page_segments=tile_page_segments[ti : ti + 1],
        glyph_patches=char_patches[gi : gi + 1],
        glyph_coords=glyph_coords[gi : gi + 1],
        glyph_valid_mask=char_valid_mask[gi : gi + 1],
        glyph_page_segments=glyph_page_segments[gi : gi + 1],
        char_class_ids=char_class_ids[gi : gi + 1],
        words=[words[wi]],
        word_metadata=[word_metadata[wi]],
        paths=[path_line],
    )
    return lat


def _make_test_dir(
    base_dir: str,
    test_index: int,
    label_name_a: str,
    label_name_b: str,
) -> str:
    """Create and return per-test folder path."""

    def _sanitize(s: str) -> str:
        return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)

    lab_a = _sanitize(str(label_name_a))
    lab_b = _sanitize(str(label_name_b))
    test_dir = os.path.join(base_dir, f"test_{test_index:02d}_A-{lab_a}_B-{lab_b}")
    os.makedirs(test_dir, exist_ok=True)
    return test_dir


def _save_pair_visualization(
    *,
    tiles: torch.Tensor,
    tile_valid_mask: torch.Tensor,
    char_patches: torch.Tensor,
    char_valid_mask: torch.Tensor,
    paths: Tuple[str, str],
    test_dir: str,
    max_tiles_vis: int,
    max_glyphs_vis: int,
) -> None:
    """
    Save two images visualizing what was extracted for a pair:
    - tiles.png  : tiles from A (left) and B (right)
    - glyphs.png : glyphs from A (left) and B (right)
    """
    # Move everything to CPU for visualization
    tiles_cpu = tiles.detach().cpu()
    tile_valid_cpu = tile_valid_mask.detach().cpu().bool()
    chars_cpu = char_patches.detach().cpu()
    chars_valid_cpu = char_valid_mask.detach().cpu().bool()

    to_pil = transforms.ToPILImage()

    # De-normalize back to [0,1] for colorful visualization
    inv_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(NORMALIZE_MEAN, NORMALIZE_STD)],
        std=[1.0 / s for s in NORMALIZE_STD],
    )

    def _make_pair_grid(
        feats_a: torch.Tensor,
        feats_b: torch.Tensor,
        mask_a: torch.Tensor,
        mask_b: torch.Tensor,
        max_items: int,
        fallback_size: Tuple[int, int] = (128, 128),
    ) -> Image.Image:
        if feats_a.ndim != 4 or feats_b.ndim != 4:
            raise ValueError(
                f"Expected 4D tensors for visualization, got shapes={tuple(feats_a.shape)}, {tuple(feats_b.shape)}"
            )
        idx_a = mask_a.nonzero(as_tuple=False).squeeze(-1)
        idx_b = mask_b.nonzero(as_tuple=False).squeeze(-1)
        if idx_a.numel() == 0 and idx_b.numel() == 0:
            return Image.new("RGB", fallback_size, color=(0, 0, 0))

        half = max_items // 2 if max_items > 1 else max_items
        sel_a = feats_a[idx_a][:half]
        sel_b = feats_b[idx_b][:max_items - sel_a.shape[0]]
        if sel_a.numel() == 0 and sel_b.numel() == 0:
            return Image.new("RGB", fallback_size, color=(0, 0, 0))
        subset = torch.cat([sel_a, sel_b], dim=0)

        # De-normalize tiles back to RGB if needed
        if subset.shape[1] == 3:
            subset = torch.stack([inv_normalize(p) for p in subset], dim=0)
        subset = torch.clamp(subset, 0.0, 1.0)
        nrow = min(max_items, subset.shape[0])
        grid = make_grid(subset, nrow=nrow, padding=2, normalize=False)
        return to_pil(grid)

    # Sample A (index 0) and B (index 1)
    tiles_a = tiles_cpu[0]  # [Na, C, H, W]
    tiles_b = tiles_cpu[1]
    tiles_valid_a = tile_valid_cpu[0]
    tiles_valid_b = tile_valid_cpu[1]

    glyphs_a = chars_cpu[0]
    glyphs_b = chars_cpu[1]
    glyphs_valid_a = chars_valid_cpu[0]
    glyphs_valid_b = chars_valid_cpu[1]

    tiles_img = _make_pair_grid(tiles_a, tiles_b, tiles_valid_a, tiles_valid_b, max_tiles_vis)
    tiles_path = os.path.join(test_dir, "tiles.png")
    tiles_img.save(tiles_path)
    print(f"[Viz] Saved tiles visualization grid to: {tiles_path}")

    glyphs_img = _make_pair_grid(glyphs_a, glyphs_b, glyphs_valid_a, glyphs_valid_b, max_glyphs_vis)
    glyphs_path = os.path.join(test_dir, "glyphs.png")
    glyphs_img.save(glyphs_path)
    print(f"[Viz] Saved glyphs visualization grid to: {glyphs_path}")


def _oriental_str(val: bool | None) -> str:
    """Format is_oriental value for display."""
    if val is True:
        return "oriental"
    if val is False:
        return "non-oriental"
    return "unknown"


def _write_summary_file(
    *,
    test_dir: str,
    label_name_a: str,
    label_name_b: str,
    label_idx_a: int,
    label_idx_b: int,
    path_a: str,
    path_b: str,
    is_oriental_a: bool | None,
    is_oriental_b: bool | None,
    delta_glyph_a: float,
    delta_tile_a: float,
    delta_glyph_b: float,
    delta_tile_b: float,
    delta_tiles_only: float,
    delta_glyphs_only: float,
    delta_words_only: float,
    a_valid_tiles: int,
    a_total_tiles: int,
    b_valid_tiles: int,
    b_total_tiles: int,
    a_valid_glyphs: int,
    a_total_glyphs: int,
    b_valid_glyphs: int,
    b_total_glyphs: int,
    a_valid_words: int,
    b_valid_words: int,
    cos_tiles_only: float,
    cos_glyphs_only: float,
    cos_words_only: float,
    cos_pooled_tiles: float | None,
    delta_pooled_tiles: float | None,
    cos_pooled_glyphs: float | None,
    delta_pooled_glyphs: float | None,
    cos_pooled_words: float | None,
    delta_pooled_words: float | None,
    modswap_delta_a: Dict[str, float] | None = None,
    modswap_delta_b: Dict[str, float] | None = None,
) -> None:
    """Write a human-readable text summary of the swap test results."""
    summary_path = os.path.join(test_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Glyph vs. Tile Swap Sensitivity Test\n")
        f.write("====================================\n\n")
        f.write(f"Sample A:\n")
        f.write(f"  Manuscript label: {label_name_a}\n")
        f.write(f"  Label index     : {label_idx_a}\n")
        f.write(f"  Is oriental     : {_oriental_str(is_oriental_a)}\n")
        f.write(f"  Image path      : {path_a}\n\n")
        f.write(f"Sample B:\n")
        f.write(f"  Manuscript label: {label_name_b}\n")
        f.write(f"  Label index     : {label_idx_b}\n")
        f.write(f"  Is oriental     : {_oriental_str(is_oriental_b)}\n")
        f.write(f"  Image path      : {path_b}\n\n")
        f.write(f"Pair type       : {_oriental_str(is_oriental_a)} vs {_oriental_str(is_oriental_b)}\n\n")
        f.write("Token statistics:\n")
        f.write(
            f"  Tiles  A: {a_valid_tiles} / {a_total_tiles} valid\n"
            f"  Tiles  B: {b_valid_tiles} / {b_total_tiles} valid\n"
        )
        f.write(
            f"  Glyphs A: {a_valid_glyphs} / {a_total_glyphs} valid\n"
            f"  Glyphs B: {b_valid_glyphs} / {b_total_glyphs} valid\n"
            f"  Words  A: {a_valid_words} valid\n"
            f"  Words  B: {b_valid_words} valid\n\n"
        )
        f.write("Cosine deltas (Δ = 1 - cos(e_orig, e_swapped)):\n")
        f.write(f"  A, glyph swap (keep tiles A, glyphs B): Δ_glyph = {delta_glyph_a:.6f}\n")
        f.write(f"  A, tile  swap (keep glyphs A, tiles B): Δ_tile  = {delta_tile_a:.6f}\n")
        f.write(f"  B, glyph swap (keep tiles B, glyphs A): Δ_glyph = {delta_glyph_b:.6f}\n")
        f.write(f"  B, tile  swap (keep glyphs B, tiles A): Δ_tile  = {delta_tile_b:.6f}\n")
        f.write("\nSingle-modality sanity check:\n")
        f.write(
            f"  Tiles-only  (glyphs disabled): "
            f"cos = {cos_tiles_only:.6f}, Δ_tiles_only   = {delta_tiles_only:.6f}\n"
        )
        f.write(
            f"  Glyphs-only (tiles disabled): "
            f"cos = {cos_glyphs_only:.6f}, Δ_glyphs_only = {delta_glyphs_only:.6f}\n"
        )
        f.write(
            f"  Words-only (tiles & glyphs disabled): "
            f"cos = {cos_words_only:.6f}, Δ_words_only = {delta_words_only:.6f}\n"
        )
        if cos_pooled_tiles is not None and delta_pooled_tiles is not None:
            f.write("\nPre-fusion tile-branch pooled embedding (DINOv2 + tile branch only):\n")
            f.write(
                f"  cos(pooled_tile_A, pooled_tile_B) = {cos_pooled_tiles:.6f}, "
                f"Δ_pooled_tiles = {delta_pooled_tiles:.6f}\n"
            )
        if cos_pooled_glyphs is not None and delta_pooled_glyphs is not None:
            f.write("\nPre-fusion glyph-branch pooled embedding (encoder + summarizer, before fusion):\n")
            f.write(
                f"  cos(pooled_glyph_A, pooled_glyph_B) = {cos_pooled_glyphs:.6f}, "
                f"Δ_pooled_glyphs = {delta_pooled_glyphs:.6f}\n"
            )
        if cos_pooled_words is not None and delta_pooled_words is not None:
            f.write("\nPre-fusion word-branch pooled embedding (before fusion):\n")
            f.write(
                f"  cos(pooled_word_A, pooled_word_B) = {cos_pooled_words:.6f}, "
                f"Δ_pooled_words = {delta_pooled_words:.6f}\n"
            )
        if modswap_delta_a:
            f.write("\nAll-modality hybrid swaps vs baseline (Δ = 1 - cos), A-centric:\n")
            f.write("  Tag letters: T=tiles, G=glyphs, W=words; modalities in tag taken from B.\n")
            for tag in sorted(modswap_delta_a.keys(), key=lambda k: (len(k), k)):
                f.write(f"  From B [{tag}]: Δ = {modswap_delta_a[tag]:.6f}\n")
        if modswap_delta_b:
            f.write("\nAll-modality hybrid swaps vs baseline, B-centric:\n")
            f.write("  Modalities in tag taken from A; rest from B.\n")
            for tag in sorted(modswap_delta_b.keys(), key=lambda k: (len(k), k)):
                f.write(f"  From A [{tag}]: Δ = {modswap_delta_b[tag]:.6f}\n")
    print(f"[Summary] Saved text summary to: {summary_path}")


def run_glyph_and_tile_swap_test(
    model: MultiModal,
    dataset: ManuscriptDataset,
    idx_a: int,
    idx_b: int,
    device: torch.device,
    *,
    out_dir: str,
    test_index: int,
    max_tiles_vis: int,
    max_glyphs_vis: int,
    oriental_map: dict | None = None,
) -> None:
    """
    Core experiment:
      - Build two samples (A, B)
      - Baseline logits/latents
      - Glyph swap: tiles(A) + glyphs(B)
      - Tile swap: tiles(B) + glyphs(A)
      - Report cosine deltas and basic stats.
    """
    print(f"[Experiment] Using indices A={idx_a}, B={idx_b}")

    # 1) Fetch raw samples and collate once to build masks/coords consistently.
    sample_cache = getattr(dataset, "_swap_sample_cache", {})
    sample_a = sample_cache.pop(idx_a, None)
    sample_b = sample_cache.pop(idx_b, None)
    if sample_a is None:
        sample_a = dataset[idx_a]
    if sample_b is None:
        sample_b = dataset[idx_b]

    batch = tile_collate_with_padding([sample_a, sample_b])
    if batch is None or len(batch[-1]) < 2:
        raise RuntimeError("Selected swap-sensitivity pair includes an unreadable image; skipping is not valid for this two-sample comparison.")
    (
        tiles,
        tile_valid_mask,
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
        labels,
        paths,
    ) = batch

    # Quick mask sanity checks: ensure we actually have valid tiles/glyphs.
    a_valid_tiles = int(tile_valid_mask[0].sum().item())
    a_total_tiles = int(tile_valid_mask[0].numel())
    b_valid_tiles = int(tile_valid_mask[1].sum().item())
    b_total_tiles = int(tile_valid_mask[1].numel())
    a_valid_glyphs = int(char_valid_mask[0].sum().item())
    a_total_glyphs = int(char_valid_mask[0].numel())
    b_valid_glyphs = int(char_valid_mask[1].sum().item())
    b_total_glyphs = int(char_valid_mask[1].numel())
    a_valid_words = len(words[0])
    b_valid_words = len(words[1])

    print("A valid tiles:", a_valid_tiles, "/", a_total_tiles)
    print("B valid tiles:", b_valid_tiles, "/", b_total_tiles)
    print("A valid glyphs:", a_valid_glyphs, "/", a_total_glyphs)
    print("B valid glyphs:", b_valid_glyphs, "/", b_total_glyphs)
    print("A valid words:", a_valid_words)
    print("B valid words:", b_valid_words)

    required_counts = getattr(dataset, "required_feature_counts", None)
    if required_counts:
        actual_counts = {
            "A": {
                "tiles": a_valid_tiles,
                "glyphs": a_valid_glyphs,
                "words": a_valid_words,
            },
            "B": {
                "tiles": b_valid_tiles,
                "glyphs": b_valid_glyphs,
                "words": b_valid_words,
            },
        }
        failures = [
            f"{side}.{modality}={counts[modality]}<{minimum}"
            for side, counts in actual_counts.items()
            for modality, minimum in required_counts.items()
            if counts[modality] < minimum
        ]
        if failures:
            raise RuntimeError(
                "Selected Geniza rows passed stored feature-count filtering but failed "
                f"runtime validation: {', '.join(failures)}"
            )
        print(
            "[Features] Runtime full-modality check passed for both images: "
            + ", ".join(f"{name}>={value}" for name, value in required_counts.items())
        )

    # Move tensor components to device
    tiles = tiles.to(device)
    tile_valid_mask = tile_valid_mask.to(device)
    coords = coords.to(device)
    char_patches = char_patches.to(device)
    char_valid_mask = char_valid_mask.to(device)
    glyph_coords = glyph_coords.to(device)
    glyph_page_segments = glyph_page_segments.to(device)
    char_class_ids = char_class_ids.to(device)
    tile_page_segments = tile_page_segments.to(device)

    path_a = paths[0]
    path_b = paths[1]
    label_idx_a = labels[0].item()
    label_idx_b = labels[1].item()
    # Original label strings (manuscript IDs) from dataset
    label_name_a = str(dataset.labels[idx_a])
    label_name_b = str(dataset.labels[idx_b])

    # Resolve is_oriental for each manuscript (None if map unavailable)
    om = oriental_map or {}
    is_oriental_a: bool | None = om.get(label_name_a, None)
    is_oriental_b: bool | None = om.get(label_name_b, None)

    print(f"[Sample A] path={path_a}, label={label_name_a} (idx={label_idx_a}), oriental={_oriental_str(is_oriental_a)}")
    print(f"[Sample B] path={path_b}, label={label_name_b} (idx={label_idx_b}), oriental={_oriental_str(is_oriental_b)}")

    # 2) Baseline forward for both A and B (no swaps)
    with torch.no_grad():
        logits_base, latent_base, _ = model(
            tiles=tiles,
            tile_coords=coords,
            tile_valid_mask=tile_valid_mask,
            tile_page_segments=tile_page_segments,
            glyph_patches=char_patches,
            glyph_coords=glyph_coords,
            glyph_valid_mask=char_valid_mask,
            glyph_page_segments=glyph_page_segments,
            char_class_ids=char_class_ids,
            words=words,
            word_metadata=word_metadata,
            paths=list(paths),
        )

        # Tiles-only sanity check (glyphs disabled)
        _, latent_tiles_only, _ = model(
            tiles=tiles,
            tile_coords=coords,
            tile_valid_mask=tile_valid_mask,
            tile_page_segments=tile_page_segments,
            glyph_patches=None,
            glyph_coords=None,
            glyph_valid_mask=None,
            char_class_ids=None,
            words=None,
            word_metadata=None,
            paths=list(paths),
        )

        # Glyphs-only sanity check (tiles disabled)
        _, latent_glyphs_only, _ = model(
            tiles=None,
            tile_coords=None,
            tile_valid_mask=None,
            tile_page_segments=None,
            glyph_patches=char_patches,
            glyph_coords=glyph_coords,
            glyph_valid_mask=char_valid_mask,
            glyph_page_segments=glyph_page_segments,
            char_class_ids=char_class_ids,
            words=None,
            word_metadata=None,
            paths=list(paths),
        )

        # Words-only sanity check (visual + glyph branches disabled)
        _, latent_words_only, _ = model(
            tiles=None,
            tile_coords=None,
            tile_valid_mask=None,
            tile_page_segments=None,
            glyph_patches=None,
            glyph_coords=None,
            glyph_valid_mask=None,
            char_class_ids=None,
            words=words,
            word_metadata=word_metadata,
            paths=list(paths),
        )

    # Helper to pretty-print a short summary
    def _print_summary(tag: str, logits: torch.Tensor, latent: torch.Tensor) -> None:
        logits_np = logits.cpu().numpy().ravel()
        latent_np = latent.cpu().numpy().ravel()
        print(f"{tag}: logits.shape={tuple(logits.shape)}, latent.shape={tuple(latent.shape)}")
        print(f"  logits (first 5): {logits_np[:5]}")
        print(f"  latent (first 8): {latent_np[:8]}")

    # 3) Build swapped batches and run forwards for A-centric view
    # A_orig: index 0, B_orig: index 1 in baseline batch
    base_logits_a = logits_base[0:1]
    base_logits_b = logits_base[1:2]
    base_latent_a = latent_base[0:1]
    base_latent_b = latent_base[1:2]

    print("\n=== Baseline ===")
    _print_summary("[A_orig]", base_logits_a, base_latent_a)
    _print_summary("[B_orig]", base_logits_b, base_latent_b)

    # Tiles-only cosine between A and B
    tiles_latent_a = latent_tiles_only[0:1]
    tiles_latent_b = latent_tiles_only[1:2]
    delta_tiles_only = _compute_delta(tiles_latent_a, tiles_latent_b)
    cos_tiles_only = 1.0 - delta_tiles_only
    print("\n=== Tiles-only sanity check (glyphs disabled) ===")
    print(f"  cos(eA_tile, eB_tile) = {cos_tiles_only:.6f}")
    print(f"  Δ_tiles_only (1 - cos) = {delta_tiles_only:.6f}")

    # Glyphs-only cosine between A and B
    glyphs_latent_a = latent_glyphs_only[0:1]
    glyphs_latent_b = latent_glyphs_only[1:2]
    delta_glyphs_only = _compute_delta(glyphs_latent_a, glyphs_latent_b)
    cos_glyphs_only = 1.0 - delta_glyphs_only
    print("\n=== Glyphs-only sanity check (tiles disabled) ===")
    print(f"  cos(eA_glyph, eB_glyph) = {cos_glyphs_only:.6f}")
    print(f"  Δ_glyphs_only (1 - cos) = {delta_glyphs_only:.6f}")

    words_latent_a = latent_words_only[0:1]
    words_latent_b = latent_words_only[1:2]
    delta_words_only = _compute_delta(words_latent_a, words_latent_b)
    cos_words_only = 1.0 - delta_words_only
    print("\n=== Words-only sanity check (tiles & glyphs disabled) ===")
    print(f"  cos(eA_word, eB_word) = {cos_words_only:.6f}")
    print(f"  Δ_words_only (1 - cos) = {delta_words_only:.6f}")

    # Pre-fusion tile-branch diagnostic: pooled tile tokens (before fusion/head).
    base_model = model.module if hasattr(model, "module") else model
    cos_pooled_tiles: float | None = None
    delta_pooled_tiles: float | None = None
    if getattr(base_model, "use_visual_mod", False) and hasattr(base_model, "tile_branch"):
        with torch.no_grad():
            tile_tokens, tile_summary_mask = base_model.tile_branch(
                tiles=tiles,
                tile_coords=coords,
                valid_mask=tile_valid_mask,
                page_segments=tile_page_segments,
            )  # [B, Q, d_model], [B, Q]
            mask = tile_summary_mask.bool().unsqueeze(-1)  # [B, Q, 1]
            # Avoid division by zero: if a row has no valid tiles, keep pooled vector at zero.
            masked_tokens = tile_tokens * mask
            counts = mask.sum(dim=1).clamp(min=1)
            pooled_tiles = masked_tokens.sum(dim=1) / counts  # [B, d_model]
            pooled_a = pooled_tiles[0:1]
            pooled_b = pooled_tiles[1:2]
            delta_pooled_tiles = _compute_delta(pooled_a, pooled_b)
            cos_pooled_tiles = 1.0 - delta_pooled_tiles
        print("\n=== Pre-fusion tile-branch pooled embedding (DINOv2+tile branch only) ===")
        print(f"  cos(pooled_tile_A, pooled_tile_B) = {cos_pooled_tiles:.6f}")
        print(f"  Δ_pooled_tiles (1 - cos) = {delta_pooled_tiles:.6f}")

    cos_pooled_glyphs: float | None = None
    delta_pooled_glyphs: float | None = None
    if getattr(base_model, "use_char_mod", False) and hasattr(base_model, "glyph_branch"):
        with torch.no_grad():
            glyph_tokens, g_mask = base_model.glyph_branch(
                glyph_patches=char_patches,
                glyph_coords=glyph_coords,
                valid_mask=char_valid_mask,
                page_segments=glyph_page_segments,
                char_class_ids=char_class_ids,
                debug_paths=list(paths),
            )
            mf = g_mask.unsqueeze(-1).to(dtype=glyph_tokens.dtype)
            pooled_glyphs = (glyph_tokens * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1)
            pooled_ga = pooled_glyphs[0:1]
            pooled_gb = pooled_glyphs[1:2]
            delta_pooled_glyphs = _compute_delta(pooled_ga, pooled_gb)
            cos_pooled_glyphs = 1.0 - delta_pooled_glyphs
        print("\n=== Pre-fusion glyph-branch pooled embedding (before fusion) ===")
        print(f"  cos(pooled_glyph_A, pooled_glyph_B) = {cos_pooled_glyphs:.6f}")
        print(f"  Δ_pooled_glyphs (1 - cos) = {delta_pooled_glyphs:.6f}")

    cos_pooled_words: float | None = None
    delta_pooled_words: float | None = None
    if getattr(base_model, "use_word_mod", False) and hasattr(base_model, "word_branch"):
        with torch.no_grad():
            word_tokens, word_valid = base_model.word_branch(
                words=words,
                word_metadata=word_metadata,
                device=device,
                page_segments=None,
            )
            wf = word_valid.unsqueeze(-1).to(dtype=word_tokens.dtype)
            pooled_words = (word_tokens * wf).sum(dim=1) / wf.sum(dim=1).clamp(min=1)
            pooled_wa = pooled_words[0:1]
            pooled_wb = pooled_words[1:2]
            delta_pooled_words = _compute_delta(pooled_wa, pooled_wb)
            cos_pooled_words = 1.0 - delta_pooled_words
        print("\n=== Pre-fusion word-branch pooled embedding (before fusion) ===")
        print(f"  cos(pooled_word_A, pooled_word_B) = {cos_pooled_words:.6f}")
        print(f"  Δ_pooled_words (1 - cos) = {delta_pooled_words:.6f}")

    # --- Glyph swap: tiles(A) + glyphs(B) ---
    tiles_a = tiles[0:1]
    tile_valid_a = tile_valid_mask[0:1]
    coords_a = coords[0:1]

    glyph_patches_b = char_patches[1:2]
    glyph_valid_b = char_valid_mask[1:2]
    glyph_coords_b = glyph_coords[1:2]
    glyph_page_segments_b = glyph_page_segments[1:2]
    char_class_ids_b = char_class_ids[1:2]

    # Words are list-of-lists; passed through when USE_WORD_MOD is enabled.
    words_a = [words[0]]
    word_metadata_a = [word_metadata[0]]

    with torch.no_grad():
        logits_a_glyphswap, latent_a_glyphswap, _ = model(
            tiles=tiles_a,
            tile_coords=coords_a,
            tile_valid_mask=tile_valid_a,
            tile_page_segments=tile_page_segments[0:1],
            glyph_patches=glyph_patches_b,
            glyph_coords=glyph_coords_b,
            glyph_valid_mask=glyph_valid_b,
            glyph_page_segments=glyph_page_segments_b,
            char_class_ids=char_class_ids_b,
            words=words_a,
            word_metadata=word_metadata_a,
            paths=[f"{path_a} (tiles) + {path_b} (glyphs)"],
        )

    delta_glyph_a = _compute_delta(base_latent_a, latent_a_glyphswap)

    print("\n=== Glyph Swap (A: keep tiles, swap glyphs from B) ===")
    _print_summary("[A_glyphswap]", logits_a_glyphswap, latent_a_glyphswap)
    print(f"[A] Δ_glyph (1 - cos) = {delta_glyph_a:.6f}")

    # --- Tile swap: tiles(B) + glyphs(A) ---
    tiles_b = tiles[1:2]
    tile_valid_b = tile_valid_mask[1:2]
    coords_b = coords[1:2]

    glyph_patches_a = char_patches[0:1]
    glyph_valid_a = char_valid_mask[0:1]
    glyph_coords_a = glyph_coords[0:1]
    glyph_page_segments_a = glyph_page_segments[0:1]
    char_class_ids_a = char_class_ids[0:1]

    words_b = [words[1]]
    word_metadata_b = [word_metadata[1]]

    with torch.no_grad():
        logits_a_tileswap, latent_a_tileswap, _ = model(
            tiles=tiles_b,
            tile_coords=coords_b,
            tile_valid_mask=tile_valid_b,
            tile_page_segments=tile_page_segments[1:2],
            glyph_patches=glyph_patches_a,
            glyph_coords=glyph_coords_a,
            glyph_valid_mask=glyph_valid_a,
            glyph_page_segments=glyph_page_segments_a,
            char_class_ids=char_class_ids_a,
            words=words_b,
            word_metadata=word_metadata_b,
            paths=[f"{path_b} (tiles) + {path_a} (glyphs)"],
        )

    delta_tile_a = _compute_delta(base_latent_a, latent_a_tileswap)

    print("\n=== Tile Swap (A: keep glyphs, swap tiles from B) ===")
    _print_summary("[A_tileswap]", logits_a_tileswap, latent_a_tileswap)
    print(f"[A] Δ_tile (1 - cos) = {delta_tile_a:.6f}")

    # Optionally: also report B-centric deltas (symmetry)
    # Glyph swap for B: tiles(B) + glyphs(A)
    with torch.no_grad():
        logits_b_glyphswap, latent_b_glyphswap, _ = model(
            tiles=tiles_b,
            tile_coords=coords_b,
            tile_valid_mask=tile_valid_b,
            tile_page_segments=tile_page_segments[1:2],
            glyph_patches=glyph_patches_a,
            glyph_coords=glyph_coords_a,
            glyph_valid_mask=glyph_valid_a,
            glyph_page_segments=glyph_page_segments_a,
            char_class_ids=char_class_ids_a,
            words=words_b,
            word_metadata=word_metadata_b,
            paths=[f"{path_b} (tiles) + {path_a} (glyphs)"],
        )

        logits_b_tileswap, latent_b_tileswap, _ = model(
            tiles=tiles_a,
            tile_coords=coords_a,
            tile_valid_mask=tile_valid_a,
            tile_page_segments=tile_page_segments[0:1],
            glyph_patches=glyph_patches_b,
            glyph_coords=glyph_coords_b,
            glyph_valid_mask=glyph_valid_b,
            glyph_page_segments=glyph_page_segments_b,
            char_class_ids=char_class_ids_b,
            words=words_a,
            word_metadata=word_metadata_a,
            paths=[f"{path_a} (tiles) + {path_b} (glyphs)"],
        )

    delta_glyph_b = _compute_delta(base_latent_b, latent_b_glyphswap)
    delta_tile_b = _compute_delta(base_latent_b, latent_b_tileswap)

    print("\n=== Symmetric view from B ===")
    _print_summary("[B_glyphswap]", logits_b_glyphswap, latent_b_glyphswap)
    print(f"[B] Δ_glyph (1 - cos) = {delta_glyph_b:.6f}")
    _print_summary("[B_tileswap]", logits_b_tileswap, latent_b_tileswap)
    print(f"[B] Δ_tile (1 - cos) = {delta_tile_b:.6f}")

    print("\n=== Summary (cosine deltas) ===")
    print(f"A: label={label_name_a}, label_idx={label_idx_a}, path={path_a}")
    print(f"  Δ_glyph = {delta_glyph_a:.6f}")
    print(f"  Δ_tile  = {delta_tile_a:.6f}")
    print(f"B: label={label_name_b}, label_idx={label_idx_b}, path={path_b}")
    print(f"  Δ_glyph = {delta_glyph_b:.6f}")
    print(f"  Δ_tile  = {delta_tile_b:.6f}")

    modswap_delta_a: Dict[str, float] = {}
    modswap_delta_b: Dict[str, float] = {}
    print("\n=== All-modality hybrid swaps (Δ vs baseline; T/G/W = tiles/glyphs/words) ===")
    print("  A-centric: modalities listed came from B; rest from A. B-centric: inverse.")
    for sub in _nonempty_modal_subsets():
        tag = _subset_label(sub)
        src_a = {m: (1 if m in sub else 0) for m in _MOD_NAMES}
        src_b = {m: (0 if m in sub else 1) for m in _MOD_NAMES}
        lat_ha = _hybrid_latent_single(
            model,
            device,
            src=src_a,
            tiles=tiles,
            tile_valid_mask=tile_valid_mask,
            coords=coords,
            tile_page_segments=tile_page_segments,
            char_patches=char_patches,
            char_valid_mask=char_valid_mask,
            glyph_coords=glyph_coords,
            glyph_page_segments=glyph_page_segments,
            char_class_ids=char_class_ids,
            words=words,
            word_metadata=word_metadata,
            path_line=f"hybrid A-slot [{tag}] from B | {path_a} / {path_b}",
        )
        modswap_delta_a[tag] = _compute_delta(base_latent_a, lat_ha)
        lat_hb = _hybrid_latent_single(
            model,
            device,
            src=src_b,
            tiles=tiles,
            tile_valid_mask=tile_valid_mask,
            coords=coords,
            tile_page_segments=tile_page_segments,
            char_patches=char_patches,
            char_valid_mask=char_valid_mask,
            glyph_coords=glyph_coords,
            glyph_page_segments=glyph_page_segments,
            char_class_ids=char_class_ids,
            words=words,
            word_metadata=word_metadata,
            path_line=f"hybrid B-slot [{tag}] from A | {path_a} / {path_b}",
        )
        modswap_delta_b[tag] = _compute_delta(base_latent_b, lat_hb)

    for tag in sorted(modswap_delta_a.keys(), key=lambda k: (len(k), k)):
        print(f"  A vs hybrid (from B {tag}): Δ = {modswap_delta_a[tag]:.6f}")
    for tag in sorted(modswap_delta_b.keys(), key=lambda k: (len(k), k)):
        print(f"  B vs hybrid (from A {tag}): Δ = {modswap_delta_b[tag]:.6f}")

    # Per-test output folder
    test_dir = _make_test_dir(out_dir, test_index, label_name_a, label_name_b)

    # Save visualization of tiles and glyphs for this pair
    _save_pair_visualization(
        tiles=tiles,
        tile_valid_mask=tile_valid_mask,
        char_patches=char_patches,
        char_valid_mask=char_valid_mask,
        paths=paths,
        test_dir=test_dir,
        max_tiles_vis=max_tiles_vis,
        max_glyphs_vis=max_glyphs_vis,
    )

    # Save text summary
    _write_summary_file(
        test_dir=test_dir,
        label_name_a=label_name_a,
        label_name_b=label_name_b,
        label_idx_a=label_idx_a,
        label_idx_b=label_idx_b,
        path_a=path_a,
        path_b=path_b,
        is_oriental_a=is_oriental_a,
        is_oriental_b=is_oriental_b,
        delta_glyph_a=delta_glyph_a,
        delta_tile_a=delta_tile_a,
        delta_glyph_b=delta_glyph_b,
        delta_tile_b=delta_tile_b,
        delta_tiles_only=delta_tiles_only,
        delta_glyphs_only=delta_glyphs_only,
        delta_words_only=delta_words_only,
        a_valid_tiles=a_valid_tiles,
        a_total_tiles=a_total_tiles,
        b_valid_tiles=b_valid_tiles,
        b_total_tiles=b_total_tiles,
        a_valid_glyphs=a_valid_glyphs,
        a_total_glyphs=a_total_glyphs,
        b_valid_glyphs=b_valid_glyphs,
        b_total_glyphs=b_total_glyphs,
        a_valid_words=a_valid_words,
        b_valid_words=b_valid_words,
        cos_tiles_only=cos_tiles_only,
        cos_glyphs_only=cos_glyphs_only,
        cos_words_only=cos_words_only,
        cos_pooled_tiles=cos_pooled_tiles,
        delta_pooled_tiles=delta_pooled_tiles,
        cos_pooled_glyphs=cos_pooled_glyphs,
        delta_pooled_glyphs=delta_pooled_glyphs,
        cos_pooled_words=cos_pooled_words,
        delta_pooled_words=delta_pooled_words,
        modswap_delta_a=modswap_delta_a,
        modswap_delta_b=modswap_delta_b,
    )


def _auto_pairs_for_tests(
    dataset: ManuscriptDataset,
    num_tests: int,
    oriental_map: dict | None = None,
    pair_type: str = "all",
    pair_seed: int | None = None,
) -> List[Tuple[int, int]]:
    """
    Randomly pick up to `num_tests` pairs from distinct manuscript IDs.

    Manuscripts are sampled without replacement across the complete run. Thus,
    when enough eligible manuscripts exist, every returned index belongs to a
    different manuscript. One page index is then sampled randomly per manuscript.

    Args:
        dataset:      The ManuscriptDataset to draw samples from.
        num_tests:    Maximum number of pairs to return.
        oriental_map: Optional dict mapping manuscript_id (str) -> is_oriental (bool | None).
                      Required when pair_type is not "all".
        pair_type:    One of:
                        "all" - any cross-manuscript pair (default)
                        "oo"  - both oriental
                        "nn"  - both non-oriental
                        "on"  - one oriental, one non-oriental (order doesn't matter)
        pair_seed:    Optional random seed. If omitted, a fresh seed is generated
                      and printed so the selection can be reproduced later.
    """
    n = len(dataset)
    if n < 2:
        raise RuntimeError("Need at least 2 images in the dataset to run swap tests.")

    labels = getattr(dataset, "labels", None)
    if labels is None or not labels:
        raise RuntimeError("Dataset does not expose 'labels'; cannot auto-build manuscript pairs.")

    if num_tests <= 0:
        raise ValueError(f"num_tests must be positive, got {num_tests}.")

    indices_by_manuscript: Dict[str, List[int]] = {}
    for idx, label in enumerate(labels):
        indices_by_manuscript.setdefault(str(label), []).append(idx)

    if len(indices_by_manuscript) < 2:
        raise RuntimeError("Need at least 2 manuscripts in the dataset to run swap tests.")

    om = oriental_map or {}
    if pair_type != "all" and not om:
        raise RuntimeError(
            f"pair_type='{pair_type}' requires is_oriental metadata, but no oriental map is available."
        )

    if pair_seed is None:
        pair_seed = random.SystemRandom().randrange(2**63)
    rng = random.Random(pair_seed)
    print(f"[Pairs] Random manuscript/page selection seed: {pair_seed}")

    manuscript_ids = list(indices_by_manuscript)
    required_counts = getattr(dataset, "required_feature_counts", None)
    sample_cache: Dict[int, Any] = {}
    rejected_pages = 0

    def _choose_page(manuscript_id: str) -> int | None:
        """Choose a random page and, when requested, verify its live feature counts."""
        nonlocal rejected_pages
        candidates = list(indices_by_manuscript[manuscript_id])
        rng.shuffle(candidates)
        for idx in candidates:
            if not required_counts:
                return idx
            sample = dataset[idx]
            if sample is None:
                rejected_pages += 1
                continue
            actual = {
                "tiles": int(sample[0].shape[0]),
                "glyphs": int(sample[3].shape[0]),
                "words": len(sample[5]),
            }
            if all(actual[name] >= minimum for name, minimum in required_counts.items()):
                sample_cache[idx] = sample
                return idx
            rejected_pages += 1
        return None

    def _select_pages(manuscripts: List[str], count: int) -> List[int]:
        selected: List[int] = []
        for manuscript_id in manuscripts:
            idx = _choose_page(manuscript_id)
            if idx is not None:
                selected.append(idx)
                if len(selected) >= count:
                    break
        return selected

    pairs: List[Tuple[int, int]] = []

    if pair_type == "on":
        oriental_ids = [mid for mid in manuscript_ids if om.get(mid) is True]
        non_oriental_ids = [mid for mid in manuscript_ids if om.get(mid) is False]
        rng.shuffle(oriental_ids)
        rng.shuffle(non_oriental_ids)
        oriental_pages = _select_pages(oriental_ids, num_tests)
        non_oriental_pages = _select_pages(non_oriental_ids, num_tests)
        pairs = list(zip(oriental_pages, non_oriental_pages))
    else:
        if pair_type == "oo":
            eligible_ids = [mid for mid in manuscript_ids if om.get(mid) is True]
        elif pair_type == "nn":
            eligible_ids = [mid for mid in manuscript_ids if om.get(mid) is False]
        else:
            eligible_ids = manuscript_ids

        rng.shuffle(eligible_ids)
        selected_pages = _select_pages(eligible_ids, 2 * num_tests)
        pairs = [
            (selected_pages[2 * k], selected_pages[2 * k + 1])
            for k in range(len(selected_pages) // 2)
        ]

    if required_counts:
        dataset._swap_sample_cache = sample_cache
        print(
            f"[Pairs] Runtime feature precheck selected {2 * len(pairs)} full-capacity "
            f"pages and rejected {rejected_pages} stale/incomplete candidate page(s)."
        )

    if len(pairs) < num_tests:
        print(
            f"[Pairs] Requested {num_tests} pairs, but only {len(pairs)} "
            f"non-reusing pair(s) are available for pair_type='{pair_type}'."
        )

    if not pairs:
        raise RuntimeError(
            f"Could not find any cross-manuscript pairs matching pair_type='{pair_type}' in the dataset. "
            "Try '--pair-type all' or check your DB is_oriental values."
        )

    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description="Glyph vs Tile swap sensitivity test.")
    ap.add_argument(
        "--checkpoint",
        type=str,
        default=system.BEST_MODEL_PATH,
        help="Path to model checkpoint (.pth). Defaults to system.BEST_MODEL_PATH.",
    )
    ap.add_argument(
        "--source",
        type=str,
        default="classification",
        choices=["classification", "geniza"],
        help=(
            "Image source: the configured classification split or all usable rows "
            "from the Geniza image-information table (default: classification)."
        ),
    )
    ap.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Classification dataset split to sample from; ignored for --source geniza (default: val).",
    )
    ap.add_argument(
        "--index-a",
        type=int,
        default=None,
        help="Index of first image in the chosen split (default: 0).",
    )
    ap.add_argument(
        "--index-b",
        type=int,
        default=None,
        help="Index of second image in the chosen split (default: 1).",
    )
    ap.add_argument(
        "--num-tests",
        type=int,
        default=10,
        help="Number of auto-selected cross-manuscript pairs to test when indices are not provided.",
    )
    ap.add_argument(
        "--vis-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "swap_sensitivity"),
        help="Directory to save visualization PNGs for each swap test.",
    )
    ap.add_argument(
        "--max-tiles-vis",
        type=int,
        default=24,
        help="Maximum number of tiles to visualize per image.",
    )
    ap.add_argument(
        "--max-glyphs-vis",
        type=int,
        default=104,
        help="Maximum number of glyph patches to visualize per image.",
    )
    ap.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to use (if available). Set to -1 for CPU-only.",
    )
    ap.add_argument(
        "--pair-type",
        type=str,
        default="all",
        choices=["all", "oo", "nn", "on"],
        help=(
            "Pair type filter (only used for auto-selected pairs, not --index-a/b):\n"
            "  all - any cross-manuscript pair (default)\n"
            "  oo  - both manuscripts are oriental\n"
            "  nn  - both manuscripts are non-oriental\n"
            "  on  - one oriental, one non-oriental"
        ),
    )
    ap.add_argument(
        "--pair-seed",
        type=int,
        default=None,
        help=(
            "Seed for random manuscript/page selection. If omitted, a fresh seed "
            "is generated and printed for reproducibility."
        ),
    )
    ap.add_argument(
        "--model-d-model",
        type=int,
        default=None,
        help="Override fusion d_model for old checkpoints (default: infer from weights).",
    )
    ap.add_argument(
        "--model-latent-dim",
        type=int,
        default=None,
        help="Override exported latent_dim for old checkpoints (default: infer from weights).",
    )
    ap.add_argument(
        "--model-glyph-summary-tokens",
        type=int,
        default=None,
        help="Override glyph summary-token count for old checkpoints (default: infer from weights).",
    )
    args = ap.parse_args()

    # Device selection
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"[Device] Using device: {device}")

    if args.source == "geniza":
        if args.pair_type != "all":
            ap.error("--source geniza currently supports only --pair-type all.")
        oriental_map = {}
        dataset, label2idx, idx2label = _build_geniza_dataset()
    else:
        # Fetch is_oriental map from DB (best-effort; keeps working if DB unavailable)
        oriental_map = _fetch_oriental_map()
        dataset, label2idx, idx2label = _build_dataset(args.split)

    dataset_num_classes = len(label2idx)
    checkpoint_num_classes = (
        _checkpoint_inspection.num_classes
        if _checkpoint_inspection is not None
        else None
    )
    num_classes = int(checkpoint_num_classes or dataset_num_classes)
    if num_classes != dataset_num_classes:
        print(
            f"[Model] Dataset has {dataset_num_classes} manuscript labels; initializing "
            f"the classifier with the checkpoint's {num_classes} classes. Dataset labels "
            "are used only for pair selection in this diagnostic."
        )

    # Load model + checkpoint
    model = _load_model(args.checkpoint, num_classes=num_classes, device=device)

    os.makedirs(args.vis_dir, exist_ok=True)

    # If explicit indices are provided, run a single test on that pair.
    if args.index_a is not None or args.index_b is not None:
        idx_a, idx_b = _select_indices(len(dataset), args.index_a, args.index_b)
        # Enforce cross-manuscript swap when running with explicit indices
        labels = getattr(dataset, "labels", None)
        if labels is not None and str(labels[idx_a]) == str(labels[idx_b]):
            raise RuntimeError(
                f"Explicit indices point to the same manuscript "
                f"(label={labels[idx_a]!r}). Please choose images from different manuscripts."
            )
        run_glyph_and_tile_swap_test(
            model,
            dataset,
            idx_a,
            idx_b,
            device,
            out_dir=args.vis_dir,
            test_index=0,
            max_tiles_vis=args.max_tiles_vis,
            max_glyphs_vis=args.max_glyphs_vis,
            oriental_map=oriental_map,
        )
    else:
        # Auto-select up to num-tests cross-manuscript pairs, respecting pair-type filter
        pairs = _auto_pairs_for_tests(
            dataset,
            args.num_tests,
            oriental_map=oriental_map,
            pair_type=args.pair_type,
            pair_seed=args.pair_seed,
        )
        print(f"[Main] Running {len(pairs)} swap tests (requested={args.num_tests}, pair_type='{args.pair_type}').")
        for k, (idx_a, idx_b) in enumerate(pairs):
            print("\n" + "=" * 80)
            print(f"[Main] Swap test {k + 1}/{len(pairs)}: A={idx_a}, B={idx_b}")
            print("=" * 80)
            run_glyph_and_tile_swap_test(
                model,
                dataset,
                idx_a,
                idx_b,
                device,
                out_dir=args.vis_dir,
                test_index=k,
                max_tiles_vis=args.max_tiles_vis,
                max_glyphs_vis=args.max_glyphs_vis,
                oriental_map=oriental_map,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
