"""
Tile Discriminability Calibration.

Answers the question: "Are tile embeddings useful at all for discriminating
manuscripts, or do they produce generic near-identical embeddings for every pair?"

Method:
  1. Sample N_SAME   pairs of images from the SAME manuscript  → should have high cosine sim
  2. Sample N_CROSS  pairs of images from DIFFERENT manuscripts → should have lower cosine sim
     (optionally split cross-pairs by oriental type: oo / nn / on)
  3. Collect ALL unique indices.
  4. Run the model on every unique sample in BATCHES (fast GPU path).
  5. Compute pair cosine similarities from the embedding cache — no redundant forward passes or I/O.

Embedding modes reported:
    tile_branch_clean  : model aux latent for tiles; after tile branch transformer, before fusion
    glyph_branch_clean : model aux latent for glyphs; after glyph summarizer, before fusion
    word_branch_clean  : model aux latent for words; after word summarizer, before fusion
    fuse_*        : full-model latent for each non-empty subset of {tiles, glyphs, words}
                    (fuse_t, fuse_g, fuse_w, fuse_tg, fuse_tw, fuse_gw, fuse_tgw).
                    With --skip-glyphs, only subsets that omit glyphs are run
                    (fuse_t, fuse_w, fuse_tw).

Usage:
    python Drafts/GlyphsDebugs/tile_discriminability_calibration.py --checkpoint PATH
    python Drafts/GlyphsDebugs/tile_discriminability_calibration.py --checkpoint PATH --skip-glyphs --batch-size 16
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from itertools import combinations
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Early arg parse + checkpoint inspection: must run BEFORE importing model
# modules so we can patch architecture-related system constants while they are
# still being bound by downstream imports.
_EARLY_PARSER = argparse.ArgumentParser(add_help=False)
_EARLY_PARSER.add_argument("--checkpoint", type=str, default="")
_EARLY_ARGS, _ = _EARLY_PARSER.parse_known_args()


def _early_resolve_path(path: str) -> str:
    if not path:
        return ""
    return path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)


import system  # noqa: E402
from utilities.checkpoint_utils import (  # noqa: E402
    inspect_checkpoint,
    apply_inspection_to_system,
    load_state_dict_with_report,
    warn_if_num_classes_mismatch,
    warn_if_tile_size_mismatch,
)

# Try to resolve the checkpoint path against system.BEST_MODEL_PATH when the
# user didn't pass --checkpoint, so the inspection still patches the right
# architecture knobs by default.
_INSPECTION_PATH = _early_resolve_path(_EARLY_ARGS.checkpoint) or _early_resolve_path(
    getattr(system, "BEST_MODEL_PATH", "") or ""
)
_CHECKPOINT_INSPECTION = inspect_checkpoint(_INSPECTION_PATH)
apply_inspection_to_system(_CHECKPOINT_INSPECTION, system)

from system import (  # noqa: E402
    BASE_DIR,
    MAX_TILES_EVAL,
    LABEL_HEAD,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    GENIZA_CONTRASTIVE_TABLE,
)

from models import MultiModal  # noqa: E402
from train.split_data import build_splits  # noqa: E402
from train.dataset import ManuscriptDataset, tile_collate_with_padding  # noqa: E402
from train.db_loader import get_db_connection, get_table_as_df  # noqa: E402
from tasks.label_heads.factory import create_label_head  # noqa: E402
from tasks.label_heads.base import build_label_maps  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TRAINING_MODE_ALIASES = {
    "pretrain": "stage1",
    "stage_1": "stage1",
    "1": "stage1",
    "finetune": "stage2",
    "fine-tune": "stage2",
    "stage_2": "stage2",
    "2": "stage2",
}
_TRAINING_MODE_CHOICES = ("stage1", "stage2", "demo", "pretrain", "finetune")
_SUPPORTED_TRAINING_MODES = {"stage1", "stage2", "demo"}


def _normalize_training_mode(mode: str) -> str:
    mode = str(mode or "stage1").strip().lower()
    return _TRAINING_MODE_ALIASES.get(mode, mode)

def _fetch_oriental_map() -> Dict[str, Optional[bool]]:
    try:
        from system import PRETRAIN_TABLE_NAME
        conn = get_db_connection()
        try:
            df = get_table_as_df(conn, PRETRAIN_TABLE_NAME)
        finally:
            conn.close()
        if "is_oriental" not in df.columns or "manuscript_id" not in df.columns:
            return {}
        result: Dict[str, Optional[bool]] = {}
        for _, row in df[["manuscript_id", "is_oriental"]].drop_duplicates("manuscript_id").iterrows():
            mid = str(row["manuscript_id"])
            val = row["is_oriental"]
            if val is None or (isinstance(val, float) and val != val):
                result[mid] = None
            elif isinstance(val, bool):
                result[mid] = val
            elif isinstance(val, str):
                result[mid] = val.strip().lower() in ("true", "t", "1", "yes")
            else:
                result[mid] = bool(val)
        return result
    except Exception:
        return {}


def _build_dataset(split: str) -> Tuple[ManuscriptDataset, dict, dict]:
    splits, _ = build_splits(BASE_DIR)
    label_head = create_label_head(name=LABEL_HEAD)
    flat = label_head.flatten_splits(splits)
    if split == "train":
        p, l, x = flat.train_paths, flat.train_labels, flat.train_xmls
    elif split == "val":
        p, l, x = flat.val_paths, flat.val_labels, flat.val_xmls
    else:
        p, l, x = flat.test_paths, flat.test_labels, flat.test_xmls
    all_labels = flat.train_labels + flat.val_labels + flat.test_labels
    l2i, i2l = build_label_maps(all_labels=all_labels)
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])
    dataset = ManuscriptDataset(p, l, trans, l2i, xml_paths=x, max_tiles_per_image=MAX_TILES_EVAL, split=split)
    return dataset, l2i, i2l


def _build_dataset_from_testset_csv(csv_path: str) -> Tuple[ManuscriptDataset, dict, dict, Any]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    required = ["image_path", "xml_path", "manuscript_id", "cluster_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in test-set CSV: {missing}")

    paths = df["image_path"].astype(str).tolist()
    labels = df["manuscript_id"].astype(str).tolist()
    xml_paths = [
        (str(v).strip() if pd.notna(v) and str(v).strip() else None)
        for v in df["xml_path"].tolist()
    ]
    label2idx, idx2label = build_label_maps(all_labels=labels)
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])
    dataset = ManuscriptDataset(
        paths,
        labels,
        trans,
        label2idx,
        xml_paths=xml_paths,
        max_tiles_per_image=MAX_TILES_EVAL,
        split="test",
    )
    return dataset, label2idx, idx2label, df


def _build_dataset_from_geniza_table(geniza_set: str) -> Tuple[ManuscriptDataset, dict, dict, Any]:
    """Build dataset from the Geniza contrastive table.

    Args:
        geniza_set:
            - "geniza-val": only rows where dataset_split == "val"
            - "geniza-all": all rows
    """
    conn = get_db_connection()
    try:
        df = get_table_as_df(conn, GENIZA_CONTRASTIVE_TABLE)
    finally:
        conn.close()
    if df is None or len(df) == 0:
        raise RuntimeError(f"Geniza table {GENIZA_CONTRASTIVE_TABLE!r} is empty or unavailable.")

    required = ["image_path", "manuscript_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {GENIZA_CONTRASTIVE_TABLE}: {missing}. "
            f"Available: {list(df.columns)}"
        )

    work = df.copy()
    if geniza_set == "geniza-val":
        if "dataset_split" not in work.columns:
            raise ValueError(
                f"{GENIZA_CONTRASTIVE_TABLE} has no 'dataset_split' column; "
                "cannot select geniza-val."
            )
        work["dataset_split"] = work["dataset_split"].astype(str).str.strip().str.lower()
        work = work[work["dataset_split"] == "val"].copy()
        if work.empty:
            raise RuntimeError(
                f"No rows with dataset_split='val' found in {GENIZA_CONTRASTIVE_TABLE}."
            )

    paths = work["image_path"].astype(str).tolist()
    labels = work["manuscript_id"].astype(str).tolist()
    if "xml_path" in work.columns:
        xml_paths = [
            (str(v).strip() if v is not None and str(v).strip() else None)
            for v in work["xml_path"].tolist()
        ]
    else:
        xml_paths = [None] * len(paths)

    label2idx, idx2label = build_label_maps(all_labels=labels)
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])
    dataset = ManuscriptDataset(
        paths,
        labels,
        trans,
        label2idx,
        xml_paths=xml_paths,
        max_tiles_per_image=MAX_TILES_EVAL,
        split="val" if geniza_set == "geniza-val" else "test",
    )
    return dataset, label2idx, idx2label, work


def _load_model(
    checkpoint_path: str,
    num_classes: int,
    device: torch.device,
    *,
    inspection: Optional[Any] = None,
) -> MultiModal:
    """Load a MultiModal checkpoint for offline analysis.

    ``num_classes`` is the local-split class count, used to size the classifier
    head. The actual training-time class count comes from the inspection (if
    available) and is only used for a sanity warning — the head is irrelevant
    for the latent-cosine analyses this script runs.
    """
    if inspection is None or inspection.path != checkpoint_path or not inspection.state_dict:
        inspection = inspect_checkpoint(checkpoint_path)
    if not inspection.state_dict:
        raise RuntimeError(f"Could not extract a state_dict from checkpoint: {checkpoint_path}")

    head_classes = inspection.num_classes if inspection.num_classes is not None else num_classes
    warn_if_num_classes_mismatch(inspection, num_classes, label="Checkpoint")
    if hasattr(system, "TILE_SIZE"):
        warn_if_tile_size_mismatch(inspection, system.TILE_SIZE, label="Checkpoint")

    model = MultiModal(num_classes=head_classes).to(device)
    model.eval()
    load_state_dict_with_report(model, inspection.state_dict, label="Checkpoint")
    if hasattr(model, "modality_dropout_enabled"):
        model.modality_dropout_enabled = False
    if hasattr(model, "token_subsample_enabled"):
        model.token_subsample_enabled = False
    return model


# ---------------------------------------------------------------------------
# Batched Embedding logic
# ---------------------------------------------------------------------------

class _IndexedSubset(torch.utils.data.Dataset):
    def __init__(self, ds, idxs):
        self.ds = ds
        self.idxs = idxs
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i): return self.ds[self.idxs[i]], self.idxs[i]

def _collate_indexed(batch):
    kept = [(item, idx) for item, idx in batch if item is not None]
    items = [item for item, _idx in kept]
    idxs = [idx for _item, idx in kept]
    return tile_collate_with_padding(items), idxs


_MODALITY_ORDER = ("tiles", "glyphs", "words")


def _fusion_subsets(*, enabled_modalities: FrozenSet[str]) -> List[FrozenSet[str]]:
    subs: List[FrozenSet[str]] = []
    keys = tuple(k for k in _MODALITY_ORDER if k in enabled_modalities)
    for r in range(1, len(keys) + 1):
        for comb in combinations(keys, r):
            subs.append(frozenset(comb))
    return subs


def _fusion_mode_name(sub: FrozenSet[str]) -> str:
    abbrev = "".join(k[0] for k in _MODALITY_ORDER if k in sub)
    return f"fuse_{abbrev}"


@torch.no_grad()
def _forward_fusion_subset(
    collated: Tuple[Any, ...],
    call_model: Any,
    device: torch.device,
    sub: FrozenSet[str],
) -> torch.Tensor:
    """Single forward for a batch; only modalities in ``sub`` are passed (others None)."""
    (t, tm, tc, tps, p, pm, pc, gps, cid, _m, w, wm, _l, paths) = collated
    t, tm, tc, tps = t.to(device), tm.to(device), tc.to(device), tps.to(device)
    use_t, use_g, use_w = "tiles" in sub, "glyphs" in sub, "words" in sub
    if use_t:
        tiles, tile_mask, tile_coords, tile_ps = t, tm, tc, tps
    else:
        tiles = tile_coords = tile_mask = tile_ps = None
    if use_g and p is not None:
        gp, gv, gc, gps, gcid = p.to(device), pm.to(device), pc.to(device), gps.to(device), cid.to(device)
    else:
        gp = gv = gc = gps = gcid = None
    if use_w:
        words, word_meta = w, wm
    else:
        words, word_meta = None, None
    _, lat, _ = call_model(
        tiles=tiles,
        tile_coords=tile_coords,
        tile_valid_mask=tile_mask,
        tile_page_segments=tile_ps,
        glyph_patches=gp,
        glyph_coords=gc,
        glyph_valid_mask=gv,
        glyph_page_segments=gps,
        char_class_ids=gcid,
        words=words,
        word_metadata=word_meta,
        paths=list(paths),
    )
    return lat


@torch.no_grad()
def _compute_all_embeddings(
    dataset: ManuscriptDataset,
    indices: List[int],
    model: MultiModal,
    modes: List[str],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    *,
    fusion_mode_to_subset: Dict[str, FrozenSet[str]],
) -> Dict[int, Dict[str, torch.Tensor]]:
    loader = DataLoader(
        _IndexedSubset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_indexed,
        # pin_memory can cause issues with multi-GPU / device-specific pinning
        # and is not critical for this offline analysis script.
        pin_memory=False,
    )
    base_model = model.module if hasattr(model, "module") else model
    emb_cache: Dict[int, Dict[str, torch.Tensor]] = {}
    total = len(indices)
    
    print(f"[Run] Embedding {total} samples in batches of {batch_size}...")
    for b_idx, (collated, orig_idxs) in enumerate(loader):
        if device.type == "cuda":
            # Tile/glyph/word pooled paths run multiple forwards per DataLoader batch;
            # this helps reduce fragmentation between batches.
            torch.cuda.empty_cache()
        n = min((b_idx + 1) * batch_size, total)
        if b_idx % 10 == 0 or n == total:
            print(f"  [Embed] {n}/{total} processed...")
        if collated is None:
            continue

        (t, tm, tc, tps, p, pm, pc, gps, cid, _m, w, wm, _l, paths) = collated
        t, tm, tc, tps = t.to(device), tm.to(device), tc.to(device), tps.to(device)
        if p is not None:
            p, pm, pc, gps, cid = p.to(device), pm.to(device), pc.to(device), gps.to(device), cid.to(device)

        batch_res: Dict[str, torch.Tensor] = {}

        branch_clean_modes = frozenset({"tile_branch_clean", "glyph_branch_clean", "word_branch_clean"})
        if branch_clean_modes.intersection(modes):
            _, _, aux_latents = base_model(
                tiles=t,
                tile_coords=tc,
                tile_valid_mask=tm,
                tile_page_segments=tps,
                glyph_patches=p,
                glyph_coords=pc,
                glyph_valid_mask=pm,
                glyph_page_segments=gps,
                char_class_ids=cid,
                words=w,
                word_metadata=wm,
                paths=list(paths),
                device=device,
                return_aux_latents=True,
            )
            if "tile_branch_clean" in modes and "tile" in aux_latents:
                batch_res["tile_branch_clean"] = aux_latents["tile"]
            if "glyph_branch_clean" in modes and "glyph" in aux_latents:
                batch_res["glyph_branch_clean"] = aux_latents["glyph"]
            if "word_branch_clean" in modes and "word" in aux_latents:
                batch_res["word_branch_clean"] = aux_latents["word"]

        for mode in modes:
            if mode in branch_clean_modes or not mode.startswith("fuse_"):
                continue
            sub = fusion_mode_to_subset.get(mode)
            if sub is None:
                continue
            # WordBranch (AlephBERT) is not reliably DataParallel-safe in this script.
            # For subsets that include `words`, run on the underlying single module
            # (typically GPU0). For subsets without `words`, allow DataParallel.
            call_target = base_model if "words" in sub else model
            batch_res[mode] = _forward_fusion_subset(collated, call_target, device, sub)

        batch_len = len(orig_idxs)
        bad_shapes = {
            m: tuple(tens.shape)
            for m, tens in batch_res.items()
            if not torch.is_tensor(tens) or tens.shape[0] != batch_len
        }
        if bad_shapes:
            raise RuntimeError(
                f"Embedding mode(s) returned invalid batch shapes at loader batch {b_idx}: "
                f"{bad_shapes}; expected first dimension {batch_len}. "
                f"paths={list(paths)[:min(4, len(paths))]}"
            )

        for i, idx in enumerate(orig_idxs):
            emb_cache[idx] = {m: tens[i].detach().cpu() for m, tens in batch_res.items()}

    return emb_cache


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a_n = F.normalize(a.float().unsqueeze(0), dim=-1)
    b_n = F.normalize(b.float().unsqueeze(0), dim=-1)
    return float((a_n * b_n).sum().item())


# ---------------------------------------------------------------------------
# Stats & Plotting
# ---------------------------------------------------------------------------

def _percentile(vals: List[float], p: float) -> float:
    s = _finite_vals(vals)
    if not s: return float("nan")
    s = sorted(s)
    idx = (len(s) - 1) * p / 100.0
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)

def _finite_vals(vals: List[float]) -> List[float]:
    out: List[float] = []
    for v in vals:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv):
            out.append(fv)
    return out

def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")

def _stdev(vals: List[float]) -> float:
    if len(vals) <= 1:
        return 0.0
    mu = _mean(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))

def _print_stats(label: str, vals: List[float]) -> None:
    if not vals: return
    finite = _finite_vals(vals)
    skipped = len(vals) - len(finite)
    skipped_msg = f" skipped_nonfinite={skipped}" if skipped else ""
    print(
        f"  {label}: n={len(finite):4d} mean={_mean(finite):.4f} "
        f"std={_stdev(finite):.4f} p50={_percentile(finite, 50):.4f} "
        f"p5={_percentile(finite, 5):.4f} p95={_percentile(finite, 95):.4f}"
        f"{skipped_msg}"
    )

def _try_plot(same_data, cross_data, out_path, pair_type, *, mode_filter=None, title_suffix=""):
    try:
        import matplotlib.pyplot as plt
        modes = [
            k for k in same_data
            if (_finite_vals(same_data[k]) or _finite_vals(cross_data.get(k, [])))
            and (mode_filter(k) if mode_filter else True)
        ]
        if not modes:
            return
        fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 4), squeeze=False)
        for ax, mode in zip(axes[0], modes):
            sv, cv = _finite_vals(same_data[mode]), _finite_vals(cross_data[mode])
            if not sv or not cv:
                continue
            lo, hi = min(sv+cv)-0.005, max(sv+cv)+0.005
            ax.hist(sv, bins=40, range=(lo, hi), density=True, alpha=0.5, label="same", color="steelblue")
            ax.hist(cv, bins=40, range=(lo, hi), density=True, alpha=0.5, label=f"cross({pair_type})", color="tomato")
            ax.set_title(mode); ax.legend()
        if title_suffix:
            fig.suptitle(title_suffix)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    except Exception: pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=system.BEST_MODEL_PATH)
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument(
        "--test-set-csv",
        type=str,
        default="",
        help="Optional CSV in the compare_clusters_pairs.py format. When set, build the dataset from this fixed test set and sample same/cross pairs by cluster_id instead of manuscript_id.",
    )
    ap.add_argument(
        "--training-mode",
        type=str,
        default=None,
        choices=_TRAINING_MODE_CHOICES,
        help="Which dataset split strategy to use for sampling pairs. "
             "Defaults to current system.TRAINING_MODE. Current modes are "
             "stage1/stage2/demo; pretrain/finetune are accepted as legacy aliases. "
             "If not set and the checkpoint filename contains 'demo', uses 'demo' "
             "to match the demo eval subset.",
    )
    ap.add_argument("--n-same", type=int, default=200)
    ap.add_argument("--n-cross", type=int, default=500)
    ap.add_argument("--pair-type", type=str, default="all", choices=["all", "oo", "nn", "on"])
    ap.add_argument(
        "--geniza-set",
        type=str,
        default="none",
        choices=["none", "geniza-val", "geniza-all"],
        help=(
            "Optional Geniza sampling source. "
            "'geniza-val' uses only dataset_split='val' from GENIZA_CONTRASTIVE_TABLE; "
            "'geniza-all' uses all rows in that table. "
            "When set, this overrides --split/--training-mode."
        ),
    )
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out-dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "discriminability_calibration"))
    ap.add_argument("--skip-glyphs", action="store_true")
    args = ap.parse_args()

    rng = random.Random(42)
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    test_set_mode = bool(args.test_set_csv)
    geniza_mode = args.geniza_set != "none"

    # Reuse the early inspection when the user didn't change --checkpoint, so
    # we don't load the .pth twice.
    # Note: any system-constants patch must already have happened during the
    # early-parse phase above; by this point models / train.dataset have been
    # imported and snapshotted those constants. We re-inspect only to feed
    # warnings (num_classes / tile_size) accurately.
    resolved_checkpoint = _early_resolve_path(args.checkpoint)
    if (
        _CHECKPOINT_INSPECTION is not None
        and _CHECKPOINT_INSPECTION.path == resolved_checkpoint
        and _CHECKPOINT_INSPECTION.state_dict
    ):
        inspection = _CHECKPOINT_INSPECTION
    else:
        inspection = inspect_checkpoint(resolved_checkpoint)
        if inspection.state_dict:
            print(
                "[Checkpoint] Note: --checkpoint differs from the path inspected at "
                "import time; architecture knobs from this checkpoint cannot be applied "
                "retroactively. Re-run with --checkpoint set on the command line if "
                "shapes mismatch."
            )

    oriental_map = _fetch_oriental_map()
    testset_df = None
    if test_set_mode:
        test_set_csv = args.test_set_csv
        if not os.path.isabs(test_set_csv):
            test_set_csv = os.path.join(_PROJECT_ROOT, test_set_csv)
        print(f"[Config] Using fixed test-set CSV: {test_set_csv}")
        dataset, l2i, _, testset_df = _build_dataset_from_testset_csv(test_set_csv)
    elif geniza_mode:
        print(
            f"[Config] Using {args.geniza_set} from table '{GENIZA_CONTRASTIVE_TABLE}' "
            f"(overrides --split/--training-mode)."
        )
        dataset, l2i, _, testset_df = _build_dataset_from_geniza_table(args.geniza_set)
    else:
        # Decide which dataset split regime to use (stage1 / stage2 / demo).
        # Precedence: explicit --training-mode CLI > checkpoint's saved
        # training_mode > filename heuristic > current system.TRAINING_MODE.
        default_training_mode = getattr(system, "TRAINING_MODE", "stage1")
        if args.training_mode is not None:
            effective_training_mode = _normalize_training_mode(args.training_mode)
            source = "--training-mode CLI"
        elif inspection.training_mode:
            effective_training_mode = _normalize_training_mode(inspection.training_mode)
            source = "checkpoint's saved training_mode"
        else:
            effective_training_mode = _normalize_training_mode(default_training_mode)
            ckpt_name = os.path.basename(args.checkpoint).lower()
            if "demo" in ckpt_name and default_training_mode != "demo":
                effective_training_mode = "demo"
                source = "checkpoint filename contains 'demo'"
            else:
                source = "system.TRAINING_MODE default"

        if effective_training_mode not in _SUPPORTED_TRAINING_MODES:
            raise ValueError(f"Unsupported training mode for analysis: {effective_training_mode}")

        print(
            f"[Config] Using training_mode='{effective_training_mode}' for dataset splits "
            f"(source: {source}; system.TRAINING_MODE was '{default_training_mode}')."
        )

        original_training_mode = getattr(system, "TRAINING_MODE", "pretrain")
        system.TRAINING_MODE = effective_training_mode
        try:
            dataset, l2i, _ = _build_dataset(args.split)
        finally:
            system.TRAINING_MODE = original_training_mode

    model = _load_model(args.checkpoint, len(l2i), device, inspection=inspection)

    # Enable simple multi-GPU usage when available.
    # Note: word-including subsets are still routed through the single module
    # at call sites to avoid AlephBERT device-mismatch.
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"[Config] Detected {torch.cuda.device_count()} GPUs. Using DataParallel.")
        model = torch.nn.DataParallel(model)

    base_for_flags = model.module if hasattr(model, "module") else model

    # Sample pairs
    if test_set_mode:
        cluster_idx: Dict[str, List[int]] = {}
        for i, row in testset_df.iterrows():
            cluster_idx.setdefault(str(row["cluster_id"]), []).append(i)

        eligible = [v for v in cluster_idx.values() if len(v) >= 2]
        if not eligible:
            raise RuntimeError("Test-set CSV has no clusters with >= 2 images; cannot sample same-cluster pairs.")
        same_pairs = [tuple(rng.sample(rng.choice(eligible), 2)) for _ in range(args.n_same)]

        cross_pairs = []
        cluster_list = list(cluster_idx.keys())
        cluster_to_manuscripts = {
            str(cid): [str(testset_df.iloc[i]["manuscript_id"]) for i in idxs]
            for cid, idxs in cluster_idx.items()
        }
        while len(cross_pairs) < args.n_cross:
            ca, cb = rng.sample(cluster_list, 2)
            ia = rng.choice(cluster_idx[ca])
            ib = rng.choice(cluster_idx[cb])
            ma = str(testset_df.iloc[ia]["manuscript_id"])
            mb = str(testset_df.iloc[ib]["manuscript_id"])
            oa, ob = oriental_map.get(ma), oriental_map.get(mb)
            ok = (args.pair_type == "all") or (oa is not None and ob is not None and (
                (args.pair_type == "oo" and oa and ob) or
                (args.pair_type == "nn" and not oa and not ob) or
                (args.pair_type == "on" and oa != ob)
            ))
            if ok:
                cross_pairs.append((ia, ib))
    else:
        lbl_idx = {}
        for i, l in enumerate(dataset.labels):
            lbl_idx.setdefault(str(l), []).append(i)

        eligible = [v for v in lbl_idx.values() if len(v) >= 2]
        same_pairs = [tuple(rng.sample(rng.choice(eligible), 2)) for _ in range(args.n_same)]

        cross_pairs = []
        ms_list = list(lbl_idx.keys())
        while len(cross_pairs) < args.n_cross:
            ma, mb = rng.sample(ms_list, 2)
            oa, ob = oriental_map.get(ma), oriental_map.get(mb)
            ok = (args.pair_type == "all") or (oa is not None and ob is not None and (
                (args.pair_type=="oo" and oa and ob) or (args.pair_type=="nn" and not oa and not ob) or (args.pair_type=="on" and oa!=ob)
            ))
            if ok:
                cross_pairs.append((rng.choice(lbl_idx[ma]), rng.choice(lbl_idx[mb])))

    # Embed unique
    unique_idxs = sorted(set(i for p in (same_pairs + cross_pairs) for i in p))
    enabled_modalities: List[str] = []
    if getattr(base_for_flags, "use_visual_mod", False):
        enabled_modalities.append("tiles")
    if not args.skip_glyphs and getattr(base_for_flags, "use_char_mod", False):
        enabled_modalities.append("glyphs")
    if getattr(base_for_flags, "use_word_mod", False):
        enabled_modalities.append("words")
    fusion_sets = _fusion_subsets(enabled_modalities=frozenset(enabled_modalities))
    fusion_mode_to_subset = {_fusion_mode_name(s): s for s in fusion_sets}
    branch_modes: List[str] = []
    if "tiles" in enabled_modalities:
        branch_modes.append("tile_branch_clean")
    if "glyphs" in enabled_modalities:
        branch_modes.append("glyph_branch_clean")
    if "words" in enabled_modalities:
        branch_modes.append("word_branch_clean")
    modes = branch_modes + list(fusion_mode_to_subset.keys())
    emb_cache = _compute_all_embeddings(
        dataset,
        unique_idxs,
        model,
        modes,
        device,
        args.batch_size,
        args.num_workers,
        fusion_mode_to_subset=fusion_mode_to_subset,
    )

    # Results
    same_cos = {m: [] for m in modes}
    cross_cos = {m: [] for m in modes}
    rows = []
    for split, pairs, bucket in [("same", same_pairs, same_cos), ("cross", cross_pairs, cross_cos)]:
        for ia, ib in pairs:
            va, vb = emb_cache[ia], emb_cache[ib]
            row = {"split": split, "label_a": dataset.labels[ia], "label_b": dataset.labels[ib]}
            if test_set_mode:
                row["cluster_id_a"] = testset_df.iloc[ia]["cluster_id"]
                row["cluster_id_b"] = testset_df.iloc[ib]["cluster_id"]
                row["same_cluster"] = bool(testset_df.iloc[ia]["cluster_id"] == testset_df.iloc[ib]["cluster_id"])
            for m in modes:
                c = _cos(va[m], vb[m])
                bucket[m].append(c)
                row[f"cos_{m}"] = c
            rows.append(row)

    # Stats
    print("\n" + "="*50 + "\nRESULTS\n" + "="*50)
    for m in modes:
        print(f"\n[{m}]")
        _print_stats("same ", same_cos[m])
        _print_stats("cross", cross_cos[m])
        same_finite = _finite_vals(same_cos[m])
        cross_finite = _finite_vals(cross_cos[m])
        sep = _mean(same_finite) - _mean(cross_finite)
        p25 = _percentile(same_finite, 25)
        frac = (
            sum(1 for v in cross_finite if v < p25) / len(cross_finite)
            if cross_finite and math.isfinite(p25)
            else float("nan")
        )
        print(f"  Separation: {sep:+.4f} | Frac correctly ranked: {frac:.2%}")

    # Files
    os.makedirs(args.out_dir, exist_ok=True)
    suffix = f"{args.split}_{args.pair_type}"
    with open(f"{args.out_dir}/calib_{suffix}.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    # Single output plot: one grid with every relevant mode.
    _try_plot(
        same_cos,
        cross_cos,
        f"{args.out_dir}/plot_{suffix}.png",
        args.pair_type,
        title_suffix="All relevant modes (branch-clean + fusion subsets)",
    )
    return 0

if __name__ == "__main__": main()
