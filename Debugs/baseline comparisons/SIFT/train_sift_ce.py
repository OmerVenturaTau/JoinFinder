#!/usr/bin/env python3
"""Train a CE projection/classifier on fixed whole-page RootSIFT features."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.auto import tqdm


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "system.py").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from system import (  # noqa: E402
    BASE_DIR,
    BETAS,
    CLUSTER_PAIRS_CSV_PATH,
    GRADIENT_ACCUMULATION_STEPS,
    GRADIENT_CLIP_NORM,
    LEARNING_RATE_STAGE1,
    NUM_EPOCHS,
    PER_GPU_BATCH_SIZE,
    SCHEDULER_ETA_MIN,
    STAGE1_TABLE_NAME,
    STAGE2_TABLE_NAME,
    TRAINING_SEED,
    WEIGHT_DECAY,
)
from train.metric_learning import compute_retrieval_metrics  # noqa: E402
from train.split_data import build_splits  # noqa: E402


Image.MAX_IMAGE_PIXELS = None
cv2.setNumThreads(1)
FEATURE_DIM = 256
MAIN_CHECKPOINT_LEARNING_RATE = 1e-4
DEFAULT_OUTPUT = PROJECT_ROOT / "Debugs/SIFT/outputs/sift_ce"
_WORKER_SIFT = None


@dataclass(frozen=True)
class PageRecord:
    path: str
    label: str


def _sift_instance(max_keypoints: int):
    global _WORKER_SIFT
    if _WORKER_SIFT is None:
        _WORKER_SIFT = cv2.SIFT_create(nfeatures=max_keypoints)
    return _WORKER_SIFT


def extract_rootsift_page(path: str, max_side: int, max_keypoints: int) -> np.ndarray:
    """Return L2-normalized [mean, std] pooled RootSIFT for a complete page."""
    with Image.open(path) as image:
        image.draft("L", (max_side, max_side))
        image = image.convert("L")
        scale = min(1.0, max_side / max(image.size))
        if scale < 1.0:
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
        pixels = np.asarray(image)
    _keypoints, descriptors = _sift_instance(max_keypoints).detectAndCompute(pixels, None)
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(FEATURE_DIM, dtype=np.float32)
    descriptors = descriptors.astype(np.float32, copy=False)
    descriptors /= descriptors.sum(axis=1, keepdims=True) + 1e-7
    descriptors = np.sqrt(descriptors)
    descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-7
    feature = np.concatenate((descriptors.mean(0), descriptors.std(0)), axis=0)
    norm = float(np.linalg.norm(feature))
    if norm > 0:
        feature /= norm
    return feature.astype(np.float32, copy=False)


class SiftExtractionDataset(Dataset):
    def __init__(self, records: Sequence[PageRecord], max_side: int, max_keypoints: int):
        self.records = list(records)
        self.max_side = int(max_side)
        self.max_keypoints = int(max_keypoints)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        feature = extract_rootsift_page(record.path, self.max_side, self.max_keypoints)
        return torch.from_numpy(feature), record.label, record.path


class SiftCEModel(nn.Module):
    """Small learned representation over fixed RootSIFT page statistics."""

    def __init__(self, num_classes: int, embedding_dim: int = 256):
        super().__init__()
        self.project = nn.Sequential(
            nn.LayerNorm(FEATURE_DIM),
            nn.Linear(FEATURE_DIM, embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, features: torch.Tensor, return_embedding: bool = False):
        embedding = self.project(features)
        return embedding if return_embedding else self.classifier(embedding)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage1", "stage2"), default="stage1")
    parser.add_argument("--table-name", default=None)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation", type=int, default=GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--gradient-clip", type=float, default=GRADIENT_CLIP_NORM)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--max-keypoints", type=int, default=4096)
    parser.add_argument("--resize-max-side", type=int, default=1600)
    parser.add_argument("--extract-workers", type=int, default=4)
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Visible CUDA device IDs used by DataParallel. By default, use every "
            "GPU exposed to this process (CUDA_VISIBLE_DEVICES remaps them to 0..N-1)."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=TRAINING_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--geniza-test", type=Path, default=PROJECT_ROOT / CLUSTER_PAIRS_CSV_PATH)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    if args.gpu_ids is None:
        args.gpu_ids = (
            []
            if args.device == "cpu" or not torch.cuda.is_available()
            else list(range(torch.cuda.device_count()))
        )
    if args.batch_size is None:
        args.batch_size = int(PER_GPU_BATCH_SIZE) * max(1, len(args.gpu_ids))
    if args.resume and args.init_checkpoint:
        parser.error("--resume and --init-checkpoint are mutually exclusive")
    if min(args.epochs, args.batch_size, args.gradient_accumulation, args.embedding_dim) <= 0:
        parser.error("epochs, batch size, accumulation, and embedding dim must be positive")
    return args


def flatten_split(split: Mapping[str, Sequence[Sequence[object]]]) -> list[PageRecord]:
    return [
        PageRecord(str(item[0]), str(label))
        for label, items in split.items()
        for item in items
        if item
    ]


def load_geniza(path: Path) -> list[PageRecord]:
    path = path if path.is_absolute() else PROJECT_ROOT / path
    frame = pd.read_csv(path)
    if not {"image_path", "cluster_id"}.issubset(frame.columns):
        raise ValueError("Geniza CSV requires image_path and cluster_id columns")
    return [
        PageRecord(str(row.image_path).strip(), str(row.cluster_id).strip())
        for row in frame.itertuples(index=False)
    ]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str, gpu_ids: Sequence[int]) -> torch.device:
    if value == "cpu" or (value == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not gpu_ids or max(gpu_ids) >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested GPUs {list(gpu_ids)} but only {torch.cuda.device_count()} are visible"
        )
    device = torch.device(f"cuda:{gpu_ids[0]}")
    torch.cuda.set_device(device)
    return device


def load_checkpoint(path: Path | None) -> dict | None:
    if path is None:
        return None
    return torch.load(path.expanduser(), map_location="cpu", weights_only=True)


def cached_features(
    records: Sequence[PageRecord],
    cache_path: Path,
    *,
    max_side: int,
    max_keypoints: int,
    workers: int,
    rebuild: bool,
) -> tuple[torch.Tensor, list[str]]:
    expected_paths = [record.path for record in records]
    if cache_path.exists() and not rebuild:
        saved = np.load(cache_path, allow_pickle=False)
        if saved["paths"].tolist() == expected_paths:
            return torch.from_numpy(saved["features"].astype(np.float32)), saved["labels"].tolist()
        print(f"cache path list changed; rebuilding {cache_path}", flush=True)

    dataset = SiftExtractionDataset(records, max_side, max_keypoints)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )
    feature_rows: list[torch.Tensor] = []
    labels: list[str] = []
    paths: list[str] = []
    for features, batch_labels, batch_paths in tqdm(
        loader, desc=f"extract SIFT {cache_path.stem}", dynamic_ncols=True
    ):
        feature_rows.append(features.float())
        labels.extend(batch_labels)
        paths.extend(batch_paths)
    matrix = torch.cat(feature_rows, dim=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=matrix.numpy(),
        labels=np.asarray(labels, dtype=str),
        paths=np.asarray(paths, dtype=str),
    )
    return matrix, labels


def label_map(records_by_split: Mapping[str, Sequence[PageRecord]], checkpoint: dict | None):
    observed = sorted({record.label for records in records_by_split.values() for record in records})
    if checkpoint is None:
        return {label: index for index, label in enumerate(observed)}
    mapping = {str(key): int(value) for key, value in checkpoint["label_to_index"].items()}
    missing = set(observed) - set(mapping)
    if missing:
        raise RuntimeError(f"Checkpoint lacks {len(missing)} dataset labels")
    return mapping


def make_tensor_dataset(features: torch.Tensor, labels: Sequence[str], mapping: Mapping[str, int]):
    targets = torch.tensor([mapping[label] for label in labels], dtype=torch.long)
    return TensorDataset(features, targets)


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def run_classification(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer=None,
    scaler=None,
    accumulation: int = 1,
    clip: float = 1.0,
    description: str,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    loss_sum = correct = count = 0
    progress = tqdm(loader, desc=description, dynamic_ncols=True, mininterval=1.0)
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for index, (features, targets) in enumerate(progress):
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(features)
                loss = criterion(logits, targets)
            if training:
                scaler.scale(loss / accumulation).backward()
                if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            batch = targets.shape[0]
            loss_sum += float(loss) * batch
            correct += int((logits.argmax(1) == targets).sum())
            count += batch
            progress.set_postfix(loss=f"{loss_sum/count:.4f}", acc=f"{correct/count:.4f}")
    return loss_sum / count, correct / count


@torch.no_grad()
def retrieval_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for sift, cluster_ids in tqdm(loader, desc="Geniza test", dynamic_ncols=True):
        sift = sift.to(device, non_blocking=True)
        embedding = model(sift, return_embedding=True)
        features.append(F.normalize(embedding.float(), dim=1).cpu())
        labels.append(cluster_ids)
    result = compute_retrieval_metrics(torch.cat(features), torch.cat(labels))
    return {
        "mAP": result.mean_average_precision,
        "KNN@1": result.knn_at_1,
        "KNN@5": result.knn_at_5,
        "KNN@10": result.knn_at_10,
    }


def write_history(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device, args.gpu_ids)
    args.table_name = args.table_name or (STAGE2_TABLE_NAME if args.stage == "stage2" else STAGE1_TABLE_NAME)
    if args.learning_rate is None:
        args.learning_rate = MAIN_CHECKPOINT_LEARNING_RATE if args.init_checkpoint else LEARNING_RATE_STAGE1
    checkpoint = load_checkpoint(args.resume or args.init_checkpoint)

    raw_splits, split_stats = build_splits(BASE_DIR, table_name=args.table_name, dataset_stage=args.stage)
    records = {name: flatten_split(raw_splits[name]) for name in ("train", "val", "test")}
    records["geniza"] = load_geniza(args.geniza_test)
    mapping = label_map({key: records[key] for key in ("train", "val", "test")}, checkpoint)
    geniza_mapping = {label: index for index, label in enumerate(sorted({r.label for r in records["geniza"]}))}

    output = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output = output / args.stage
    cache = output / "feature_cache"
    extracted = {}
    for name in ("train", "val", "geniza"):
        extracted[name] = cached_features(
            records[name],
            cache / f"{name}_k{args.max_keypoints}_s{args.resize_max_side}.npz",
            max_side=args.resize_max_side, max_keypoints=args.max_keypoints,
            workers=args.extract_workers, rebuild=args.rebuild_cache,
        )

    datasets = {
        "train": make_tensor_dataset(*extracted["train"], mapping),
        "val": make_tensor_dataset(*extracted["val"], mapping),
        "geniza": make_tensor_dataset(*extracted["geniza"], geniza_mapping),
    }
    loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=name == "train", pin_memory=device.type == "cuda")
        for name, ds in datasets.items()
    }
    model = SiftCEModel(len(mapping), args.embedding_dim)
    if checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    if device.type == "cuda" and len(args.gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=args.gpu_ids, output_device=args.gpu_ids[0])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=BETAS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=SCHEDULER_ETA_MIN)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 0
    best_val = -1.0
    if args.resume and checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint["best_val_accuracy"])

    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(json.dumps({
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "split_stats": split_stats,
        "split_sizes": {k: len(v) for k, v in records.items()},
        "num_classes": len(mapping),
    }, indent=2), encoding="utf-8")
    print(
        f"SIFT CE stage={args.stage} device={device} visible-GPUs={args.gpu_ids} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r} "
        f"global microbatch={args.batch_size} accumulation={args.gradient_accumulation} "
        f"effective batch={args.batch_size * args.gradient_accumulation}", flush=True,
    )

    history: list[dict] = []
    for epoch in range(start_epoch, args.epochs):
        started = time.monotonic()
        train_loss, train_acc = run_classification(
            model, loaders["train"], criterion, device, optimizer=optimizer, scaler=scaler,
            accumulation=args.gradient_accumulation, clip=args.gradient_clip,
            description=f"epoch {epoch + 1}/{args.epochs} train",
        )
        val_loss, val_acc = run_classification(
            model, loaders["val"], criterion, device,
            description=f"epoch {epoch + 1}/{args.epochs} val",
        )
        test = retrieval_metrics(model, loaders["geniza"], device)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
               "val_loss": val_loss, "val_accuracy": val_acc, **test,
               "seconds": time.monotonic() - started}
        history.append(row)
        is_best = val_acc > best_val
        best_val = max(best_val, val_acc)
        payload = {
            "format": "sift_ce_v1", "epoch": epoch,
            "model_state": unwrap(model).state_dict(),
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(), "label_to_index": mapping,
            "best_val_accuracy": best_val, "embedding_dim": args.embedding_dim,
        }
        torch.save(payload, output / "last.pth")
        if is_best:
            torch.save(payload, output / "best.pth")
        write_history(output / "history.csv", history)
        print(
            f"epoch {epoch + 1:03d}/{args.epochs:03d} train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f} | Geniza test "
            f"mAP={test['mAP']:.4f} KNN@1={test['KNN@1']:.4f} "
            f"KNN@5={test['KNN@5']:.4f} KNN@10={test['KNN@10']:.4f}", flush=True,
        )
    print(f"outputs: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
