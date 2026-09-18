"""
Raw ConvNeXt tile/glyph discriminability histograms.

This is a focused version of tile_discriminability_calibration.py:
- no checkpoint
- no fusion model
- no word branch
- ConvNeXt only
- two image-level embeddings per image:
  1. mean-pooled tile ConvNeXt features
  2. mean-pooled glyph ConvNeXt features

The script samples same-label and cross-label image pairs, computes cosine
similarities, and writes separate histograms for tiles and glyphs.

Examples:
  python Debugs/ConvNeXt/convnext_tile_glyph_histograms.py \
      --split val --n-same 200 --n-cross 500

  python Debugs/ConvNeXt/convnext_tile_glyph_histograms.py \
      --test-set-csv results_analysis/test_set/clusters_images_metadata.csv \
      --batch-size 4 --device cpu
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import statistics
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "system.py").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import system  # noqa: E402

# Force the dataset to load only the modalities this script needs. These must be
# set before importing train.dataset because that module binds the constants.
system.USE_VISUAL_MOD = True
system.USE_CHAR_MOD = True
system.USE_WORD_MOD = False

try:
    import psycopg2  # noqa: F401
except ModuleNotFoundError:
    psycopg2_stub = types.ModuleType("psycopg2")

    def _missing_psycopg2_connect(*_args: Any, **_kwargs: Any) -> None:
        raise ModuleNotFoundError("psycopg2 is required for DB-backed split mode. Use --test-set-csv or install psycopg2.")

    psycopg2_stub.connect = _missing_psycopg2_connect  # type: ignore[attr-defined]
    sys.modules["psycopg2"] = psycopg2_stub

from models.glyph_branch import GlyphVisualEncoder  # noqa: E402
from models.tile_backbones import create_tile_backbone  # noqa: E402
from system import (  # noqa: E402
    BASE_DIR,
    CHAR_PATCH_SIZE,
    LABEL_HEAD,
    MAX_TILES_EVAL,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    TILE_CONVNEXT_MODEL_NAME,
    TILE_SIZE,
)
from tasks.label_heads.base import build_label_maps  # noqa: E402
from tasks.label_heads.factory import create_label_head  # noqa: E402
from train.dataset import ManuscriptDataset, tile_collate_with_padding  # noqa: E402
from train.split_data import build_splits  # noqa: E402


@dataclass(frozen=True)
class PairSpec:
    split: str
    index_a: int
    index_b: int
    label_a: str
    label_b: str


class IndexedSubset(torch.utils.data.Dataset):
    def __init__(self, dataset: ManuscriptDataset, indices: List[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Tuple[Any, int]:
        original_idx = self.indices[item]
        return self.dataset[original_idx], original_idx


def collate_indexed(batch: List[Tuple[Any, int]]) -> Tuple[Optional[Tuple[Any, ...]], List[int]]:
    kept = [(item, idx) for item, idx in batch if item is not None]
    items = [item for item, _idx in kept]
    idxs = [idx for _item, idx in kept]
    return tile_collate_with_padding(items), idxs


def build_dataset(split: str) -> Tuple[ManuscriptDataset, Dict[str, int]]:
    splits, _ = build_splits(BASE_DIR)
    label_head = create_label_head(name=LABEL_HEAD)
    flat = label_head.flatten_splits(splits)
    if split == "train":
        paths, labels, xmls = flat.train_paths, flat.train_labels, flat.train_xmls
    elif split == "val":
        paths, labels, xmls = flat.val_paths, flat.val_labels, flat.val_xmls
    elif split == "test":
        paths, labels, xmls = flat.test_paths, flat.test_labels, flat.test_xmls
    else:
        raise ValueError(f"Unsupported split: {split}")

    all_labels = flat.train_labels + flat.val_labels + flat.test_labels
    label2idx, _ = build_label_maps(all_labels=all_labels)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )
    dataset = ManuscriptDataset(
        paths,
        labels,
        transform,
        label2idx,
        xml_paths=xmls,
        max_tiles_per_image=MAX_TILES_EVAL,
        split=split,
    )
    return dataset, label2idx


def build_dataset_from_testset_csv(csv_path: str) -> Tuple[ManuscriptDataset, Dict[str, int], Any]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    required = ["image_path", "xml_path", "manuscript_id", "cluster_id"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in test-set CSV: {missing}")

    paths = df["image_path"].astype(str).tolist()
    labels = df["manuscript_id"].astype(str).tolist()
    xml_paths = [
        str(value).strip() if pd.notna(value) and str(value).strip() else None
        for value in df["xml_path"].tolist()
    ]
    label2idx, _ = build_label_maps(all_labels=labels)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )
    dataset = ManuscriptDataset(
        paths,
        labels,
        transform,
        label2idx,
        xml_paths=xml_paths,
        max_tiles_per_image=MAX_TILES_EVAL,
        split="test",
    )
    return dataset, label2idx, df


def build_convnext_models(device: torch.device) -> Tuple[torch.nn.Module, torch.nn.Module]:
    tile_spec = create_tile_backbone(
        encoder_type="convnext",
        model_name=TILE_CONVNEXT_MODEL_NAME,
        tile_size=TILE_SIZE,
        pretrained=True,
    )
    tile_model = tile_spec.model.eval().to(device)
    glyph_model = GlyphVisualEncoder(char_patch_size=CHAR_PATCH_SIZE, encoder_type="convnext_tiny").encoder
    glyph_model = glyph_model.eval().to(device)
    for model in (tile_model, glyph_model):
        for param in model.parameters():
            param.requires_grad_(False)
    return tile_model, glyph_model


def sample_pairs(
    dataset: ManuscriptDataset,
    *,
    n_same: int,
    n_cross: int,
    rng: random.Random,
    testset_df: Optional[Any] = None,
) -> Tuple[List[PairSpec], List[PairSpec]]:
    if testset_df is not None:
        buckets: Dict[str, List[int]] = {}
        for idx, row in testset_df.iterrows():
            buckets.setdefault(str(row["cluster_id"]), []).append(int(idx))
        same_eligible = [idxs for idxs in buckets.values() if len(idxs) >= 2]
        if not same_eligible:
            raise RuntimeError("No clusters with at least two images; cannot sample same pairs.")
        same_pairs = [tuple(rng.sample(rng.choice(same_eligible), 2)) for _ in range(n_same)]

        cluster_ids = list(buckets.keys())
        cross_pairs = []
        while len(cross_pairs) < n_cross:
            ca, cb = rng.sample(cluster_ids, 2)
            cross_pairs.append((rng.choice(buckets[ca]), rng.choice(buckets[cb])))
    else:
        buckets = {}
        for idx, label in enumerate(dataset.labels):
            buckets.setdefault(str(label), []).append(idx)
        same_eligible = [idxs for idxs in buckets.values() if len(idxs) >= 2]
        if not same_eligible:
            raise RuntimeError("No labels with at least two images; cannot sample same pairs.")
        same_pairs = [tuple(rng.sample(rng.choice(same_eligible), 2)) for _ in range(n_same)]

        labels = list(buckets.keys())
        cross_pairs = []
        while len(cross_pairs) < n_cross:
            la, lb = rng.sample(labels, 2)
            cross_pairs.append((rng.choice(buckets[la]), rng.choice(buckets[lb])))

    def make_pair(split_name: str, ia: int, ib: int) -> PairSpec:
        return PairSpec(
            split=split_name,
            index_a=ia,
            index_b=ib,
            label_a=str(dataset.labels[ia]),
            label_b=str(dataset.labels[ib]),
        )

    return (
        [make_pair("same", ia, ib) for ia, ib in same_pairs],
        [make_pair("cross", ia, ib) for ia, ib in cross_pairs],
    )


def mean_pool_valid_features(
    model: torch.nn.Module,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    device: torch.device,
    encode_chunk_size: int,
) -> Dict[int, torch.Tensor]:
    """Encode [B,N,3,H,W], mean-pooling valid ConvNeXt features per batch row."""
    batch_size, num_items = valid_mask.shape
    flat_valid = valid_mask.reshape(-1)
    if not bool(flat_valid.any()):
        return {}

    flat_x = x.reshape(batch_size * num_items, *x.shape[2:])[flat_valid].to(device)
    flat_owner = torch.arange(batch_size).repeat_interleave(num_items)[flat_valid]

    features = []
    with torch.no_grad():
        for start in range(0, flat_x.shape[0], encode_chunk_size):
            features.append(model(flat_x[start : start + encode_chunk_size]))
    flat_features = torch.cat(features, dim=0).detach().cpu()

    pooled: Dict[int, torch.Tensor] = {}
    for batch_idx in range(batch_size):
        rows = flat_owner == batch_idx
        if bool(rows.any()):
            pooled[batch_idx] = F.normalize(flat_features[rows].mean(dim=0), dim=0).cpu()
    return pooled


def compute_embeddings(
    dataset: ManuscriptDataset,
    indices: List[int],
    tile_model: torch.nn.Module,
    glyph_model: torch.nn.Module,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    encode_chunk_size: int,
) -> Dict[int, Dict[str, torch.Tensor]]:
    loader = DataLoader(
        IndexedSubset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_indexed,
        pin_memory=False,
    )

    cache: Dict[int, Dict[str, torch.Tensor]] = {}
    total = len(indices)
    for batch_idx, (collated, original_idxs) in enumerate(loader):
        done = min((batch_idx + 1) * batch_size, total)
        if batch_idx % 10 == 0 or done == total:
            print(f"[Embed] {done}/{total} images")
        if collated is None:
            continue

        tiles, tile_mask, _coords, _tile_segments, glyphs, glyph_mask, *_rest = collated
        tile_embs = mean_pool_valid_features(
            tile_model,
            tiles,
            tile_mask,
            device=device,
            encode_chunk_size=encode_chunk_size,
        )
        glyph_embs = mean_pool_valid_features(
            glyph_model,
            glyphs,
            glyph_mask,
            device=device,
            encode_chunk_size=encode_chunk_size,
        )

        for local_idx, original_idx in enumerate(original_idxs):
            row: Dict[str, torch.Tensor] = {}
            if local_idx in tile_embs:
                row["tile_convnext"] = tile_embs[local_idx]
            if local_idx in glyph_embs:
                row["glyph_convnext"] = glyph_embs[local_idx]
            cache[original_idx] = row
    return cache


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((F.normalize(a.float(), dim=0) * F.normalize(b.float(), dim=0)).sum().item())


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    idx = (len(sorted_values) - 1) * p / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (idx - lo)


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "p5": float("nan"), "p50": float("nan"), "p95": float("nan")}
    return {
        "n": float(len(values)),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p5": percentile(values, 5),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
    }


def save_histogram(same: List[float], cross: List[float], title: str, out_path: str) -> None:
    import matplotlib.pyplot as plt

    if not same and not cross:
        return
    values = same + cross
    lo = max(-1.0, min(values) - 0.01)
    hi = min(1.0, max(values) + 0.01)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(same, bins=40, range=(lo, hi), density=True, alpha=0.55, label="same", color="steelblue")
    ax.hist(cross, bins=40, range=(lo, hi), density=True, alpha=0.55, label="cross", color="tomato")
    ax.set_title(title)
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_outputs(
    *,
    out_dir: str,
    rows: List[Dict[str, Any]],
    same_cos: Dict[str, List[float]],
    cross_cos: Dict[str, List[float]],
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "pair_cosines.csv")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_lines = ["# Raw ConvNeXt Tile/Glyph Histograms", ""]
    for mode in ("tile_convnext", "glyph_convnext"):
        same_stats = summarize(same_cos[mode])
        cross_stats = summarize(cross_cos[mode])
        separation = same_stats["mean"] - cross_stats["mean"]
        summary_lines.extend(
            [
                f"## {mode}",
                "",
                f"- same: n={int(same_stats['n'])}, mean={same_stats['mean']:.4f}, std={same_stats['std']:.4f}, p50={same_stats['p50']:.4f}",
                f"- cross: n={int(cross_stats['n'])}, mean={cross_stats['mean']:.4f}, std={cross_stats['std']:.4f}, p50={cross_stats['p50']:.4f}",
                f"- separation: {separation:+.4f}",
                "",
            ]
        )
        save_histogram(
            same_cos[mode],
            cross_cos[mode],
            mode,
            os.path.join(out_dir, f"{mode}_histogram.png"),
        )

    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--test-set-csv", default="", help="Optional fixed test-set CSV with image_path/xml_path/manuscript_id/cluster_id.")
    parser.add_argument("--training-mode", choices=["pretrain", "finetune", "demo"], default=None)
    parser.add_argument("--n-same", type=int, default=200)
    parser.add_argument("--n-cross", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--encode-chunk-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "outputs" / "convnext_tile_glyph_histograms"))
    args = parser.parse_args()

    if args.training_mode is not None:
        print(f"[Config] Overriding system.TRAINING_MODE={getattr(system, 'TRAINING_MODE', None)!r} -> {args.training_mode!r}")
        system.TRAINING_MODE = args.training_mode

    testset_df = None
    if args.test_set_csv:
        test_set_csv = args.test_set_csv
        if not os.path.isabs(test_set_csv):
            test_set_csv = str(PROJECT_ROOT / test_set_csv)
        print(f"[Config] Building dataset from fixed test-set CSV: {test_set_csv}")
        dataset, _label2idx, testset_df = build_dataset_from_testset_csv(test_set_csv)
        suffix = "testset_csv"
    else:
        print(f"[Config] Building {args.split!r} split from system/build_splits.")
        dataset, _label2idx = build_dataset(args.split)
        suffix = args.split

    rng = random.Random(args.seed)
    same_pairs, cross_pairs = sample_pairs(
        dataset,
        n_same=args.n_same,
        n_cross=args.n_cross,
        rng=rng,
        testset_df=testset_df,
    )
    unique_indices = sorted({idx for pair in same_pairs + cross_pairs for idx in (pair.index_a, pair.index_b)})
    print(f"[Pairs] same={len(same_pairs)}, cross={len(cross_pairs)}, unique_images={len(unique_indices)}")

    device = torch.device(args.device)
    tile_model, glyph_model = build_convnext_models(device)
    embeddings = compute_embeddings(
        dataset,
        unique_indices,
        tile_model,
        glyph_model,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        encode_chunk_size=args.encode_chunk_size,
    )

    same_cos = {"tile_convnext": [], "glyph_convnext": []}
    cross_cos = {"tile_convnext": [], "glyph_convnext": []}
    rows: List[Dict[str, Any]] = []
    for pair in same_pairs + cross_pairs:
        row: Dict[str, Any] = {
            "split": pair.split,
            "index_a": pair.index_a,
            "index_b": pair.index_b,
            "label_a": pair.label_a,
            "label_b": pair.label_b,
        }
        for mode in ("tile_convnext", "glyph_convnext"):
            emb_a = embeddings.get(pair.index_a, {}).get(mode)
            emb_b = embeddings.get(pair.index_b, {}).get(mode)
            if emb_a is None or emb_b is None:
                row[f"cos_{mode}"] = ""
                continue
            value = cosine(emb_a, emb_b)
            row[f"cos_{mode}"] = value
            (same_cos if pair.split == "same" else cross_cos)[mode].append(value)
        rows.append(row)

    out_dir = os.path.join(args.out_dir, suffix)
    write_outputs(out_dir=out_dir, rows=rows, same_cos=same_cos, cross_cos=cross_cos)

    print(f"[Done] Wrote outputs to {out_dir}")
    print(f"  - {os.path.join(out_dir, 'tile_convnext_histogram.png')}")
    print(f"  - {os.path.join(out_dir, 'glyph_convnext_histogram.png')}")
    print(f"  - {os.path.join(out_dir, 'summary.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
