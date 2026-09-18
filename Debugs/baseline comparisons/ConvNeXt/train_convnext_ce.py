#!/usr/bin/env python3
"""Train a whole-page ConvNeXt or DINOv2 classifier with cross-entropy.

The script deliberately contains no JoinsFinder fusion, tile, glyph, word,
ArcFace, or contrastive components. It reuses ``train.split_data.build_splits``
so stage1/stage2 select and sample the same classification datasets as main.py.

Typical two-stage use::

    python Debugs/ConvNeXt/train_convnext_ce.py --stage stage1
    python Debugs/ConvNeXt/train_convnext_ce.py --stage stage2 \
        --init-checkpoint Debugs/ConvNeXt/outputs/convnext_ce/stage1/best.pth
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
from timm.data import create_transform as create_timm_transform
from timm.data import resolve_model_data_config
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
    MAIN_DATAPARALLEL_NUM_WORKERS,
    NUM_EPOCHS,
    PER_GPU_BATCH_SIZE,
    PERSISTENT_WORKERS,
    PIN_MEMORY,
    SCHEDULER_ETA_MIN,
    STAGE1_TABLE_NAME,
    STAGE2_TABLE_NAME,
    TRAINING_SEED,
    WEIGHT_DECAY,
)
from train.split_data import build_splits  # noqa: E402
from train.metric_learning import compute_retrieval_metrics  # noqa: E402


MODEL_PRESETS = {
    "convnext": "convnext_tiny.fb_in22k_ft_in1k",
    "dinov2": "vit_base_patch14_dinov2.lvd142m",
}
DEFAULT_MODEL = MODEL_PRESETS["convnext"]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "Debugs/ConvNeXt/outputs"
# main.py uses this LR when initialization comes from a stage-1 checkpoint.
MAIN_CHECKPOINT_LEARNING_RATE = 1e-4


@dataclass(frozen=True)
class PageRecord:
    path: str
    label: str


class FullPageDataset(Dataset):
    """Load a page and apply the selected whole-page or center-crop transform."""

    def __init__(
        self,
        records: Sequence[PageRecord],
        label_to_index: Mapping[str, int],
        transform,
    ) -> None:
        self.records = list(records)
        self.label_to_index = dict(label_to_index)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        record = self.records[index]
        with Image.open(record.path) as image:
            # Most source pages are very large JPEGs. Let the JPEG decoder read
            # a lower-resolution version before the final 224/518px transform;
            # this avoids allocating the full-resolution decoded page.
            image.draft("RGB", (1024, 1024))
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.label_to_index[record.label], record.path


class LetterboxWholePage:
    """Fit the complete page inside a square without cropping or distortion."""

    def __init__(self, size: int, fill: tuple[int, int, int] = (255, 255, 255)) -> None:
        self.size = int(size)
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = min(self.size / width, self.size / height)
        resized = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            resample=Image.Resampling.BICUBIC,
        )
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(
            resized,
            ((self.size - resized.width) // 2, (self.size - resized.height) // 2),
        )
        return canvas


class PrelogitExtractor(nn.Module):
    """Expose a timm backbone's native representation as a forward call."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model.forward_features(images)
        return self.model.forward_head(features, pre_logits=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage1", "stage2"), default="stage1")
    parser.add_argument(
        "--table-name",
        default=None,
        help="Override the stage table. Defaults to STAGE1_TABLE_NAME/STAGE2_TABLE_NAME.",
    )
    parser.add_argument("--backbone", choices=tuple(MODEL_PRESETS), default="convnext")
    parser.add_argument(
        "--preprocessing",
        choices=("full-page", "center-crop"),
        default="full-page",
        help=(
            "Page geometry used for train, validation, and Geniza test. "
            "Default: preserve the full page with letterboxing."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional timm model override; normally selected by --backbone.",
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Global microbatch; defaults to main.py's per-GPU batch times GPU count.",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=GRADIENT_ACCUMULATION_STEPS,
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=GRADIENT_CLIP_NORM)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(4, int(MAIN_DATAPARALLEL_NUM_WORKERS)),
        help="Whole-page NAS loader workers; 4 avoids saturating remote JPEG reads.",
    )
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=TRAINING_SEED)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Defaults to Debugs/ConvNeXt/outputs/<backbone>_ce for full-page "
            "runs or <backbone>_ce_center_crop for center-crop runs."
        ),
    )
    parser.add_argument(
        "--geniza-test",
        type=Path,
        default=PROJECT_ROOT / CLUSTER_PAIRS_CSV_PATH,
        help="Fixed Geniza member CSV containing image_path and cluster_id.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Initialize model weights and label map, normally stage1 best.pth for stage2.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume model, optimizer, scheduler, scaler, and epoch from this script's checkpoint.",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the timm pretrained checkpoint when not initializing/resuming from a file.",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--limit-train",
        type=int,
        default=0,
        help="Optional smoke-test sample limit; 0 uses the complete training split.",
    )
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--limit-geniza-test", type=int, default=0)
    args = parser.parse_args()

    args.model = args.model or MODEL_PRESETS[args.backbone]
    if args.gpu_ids is None:
        args.gpu_ids = (
            []
            if args.device == "cpu" or not torch.cuda.is_available()
            else list(range(torch.cuda.device_count()))
        )
    if args.batch_size is None:
        args.batch_size = int(PER_GPU_BATCH_SIZE) * max(1, len(args.gpu_ids))
    output_name = f"{args.backbone}_ce"
    if args.preprocessing == "center-crop":
        output_name += "_center_crop"
    args.output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / output_name)

    if args.epochs <= 0 or args.batch_size <= 0 or args.gradient_accumulation <= 0:
        parser.error("epochs, batch-size, and gradient-accumulation must be positive")
    if not 0.0 <= args.label_smoothing < 1.0:
        parser.error("label-smoothing must be in [0, 1)")
    if args.num_workers < 0 or args.prefetch_factor <= 0:
        parser.error("num-workers must be non-negative and prefetch-factor must be positive")
    if args.resume is not None and args.init_checkpoint is not None:
        parser.error("--resume and --init-checkpoint are mutually exclusive")
    return args


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {value}")
    return device


def configure_cuda_device(device: torch.device, gpu_ids: Sequence[int]) -> torch.device:
    """Validate requested GPUs and return the primary CUDA device."""
    if device.type != "cuda":
        return device
    if not gpu_ids:
        raise ValueError("At least one --gpu-ids value is required for CUDA")
    if len(set(gpu_ids)) != len(gpu_ids) or min(gpu_ids) < 0:
        raise ValueError(f"Invalid --gpu-ids: {list(gpu_ids)}")
    if device.index is not None and device.index != gpu_ids[0]:
        raise ValueError(
            f"Primary --device is cuda:{device.index}, but the first --gpu-ids "
            f"value is {gpu_ids[0]}"
        )
    available = torch.cuda.device_count()
    unavailable = [gpu_id for gpu_id in gpu_ids if gpu_id >= available]
    if unavailable:
        raise RuntimeError(
            f"Requested CUDA devices {list(gpu_ids)}, but only {available} are visible"
        )
    primary = torch.device(f"cuda:{gpu_ids[0]}")
    torch.cuda.set_device(primary)
    return primary


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_split(split: Mapping[str, Sequence[Sequence[object]]]) -> list[PageRecord]:
    records: list[PageRecord] = []
    for raw_label, items in split.items():
        label = str(raw_label)
        for item in items:
            if not item:
                continue
            records.append(PageRecord(path=str(item[0]), label=label))
    return records


def load_geniza_test(path: Path, limit: int = 0) -> list[PageRecord]:
    """Load the canonical member-level Geniza retrieval test."""
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    frame = pd.read_csv(resolved)
    required = {"image_path", "cluster_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Geniza test is missing columns: {sorted(missing)}")
    if frame["image_path"].duplicated().any():
        raise ValueError("Geniza test contains duplicate image paths")
    cluster_sizes = frame.groupby("cluster_id").size()
    if (cluster_sizes < 2).any():
        raise ValueError("Every Geniza test cluster must contain at least two pages")
    records = [
        PageRecord(str(row.image_path).strip(), str(row.cluster_id).strip())
        for row in frame.itertuples(index=False)
    ]
    return apply_limit(records, limit)


def validate_splits(splits: Mapping[str, Sequence[PageRecord]]) -> None:
    if not splits["train"]:
        raise RuntimeError("The selected dataset has no training pages")
    if not splits["val"]:
        raise RuntimeError("The selected dataset has no validation pages")

    path_sets: dict[str, set[str]] = {}
    for name, records in splits.items():
        paths = [str(Path(record.path)) for record in records]
        if len(paths) != len(set(paths)):
            raise RuntimeError(f"Duplicate image paths found inside the {name} split")
        path_sets[name] = set(paths)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = path_sets[left] & path_sets[right]
        if overlap:
            raise RuntimeError(
                f"Image leakage between {left} and {right}: {next(iter(overlap))}"
            )


def apply_limit(records: list[PageRecord], limit: int) -> list[PageRecord]:
    return records if limit <= 0 else records[:limit]


def make_label_map(
    splits: Mapping[str, Sequence[PageRecord]],
    checkpoint: dict | None,
) -> dict[str, int]:
    observed = sorted({record.label for records in splits.values() for record in records})
    if checkpoint is None:
        return {label: index for index, label in enumerate(observed)}

    stored = checkpoint.get("label_to_index")
    if not isinstance(stored, dict) or not stored:
        raise RuntimeError("Checkpoint has no usable label_to_index mapping")
    label_to_index = {str(label): int(index) for label, index in stored.items()}
    missing = sorted(set(observed) - set(label_to_index))
    if missing:
        raise RuntimeError(
            "The checkpoint classifier has no output for dataset labels: "
            + ", ".join(missing[:10])
        )
    expected_indices = set(range(len(label_to_index)))
    if set(label_to_index.values()) != expected_indices:
        raise RuntimeError("Checkpoint label indices are not contiguous from zero")
    return label_to_index


def load_checkpoint(path: Path | None) -> dict | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    return torch.load(resolved, map_location="cpu", weights_only=True)


def create_transforms(model: nn.Module, preprocessing: str):
    config = dict(resolve_model_data_config(model))
    image_size = int(config["input_size"][-1])
    normalize = transforms.Normalize(mean=config["mean"], std=config["std"])
    if preprocessing == "center-crop":
        # Use the checkpoint's native deterministic evaluation geometry. For
        # ConvNeXt this is resize-shorter-side then 224px center crop; DINOv2
        # uses its own 518px data configuration. Training keeps the same crop
        # geometry and adds only the color augmentation used by the full-page run.
        eval_transform = create_timm_transform(**config, is_training=False)
        train_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            eval_transform,
        ])
    elif preprocessing == "full-page":
        eval_transform = transforms.Compose([
            LetterboxWholePage(image_size),
            transforms.ToTensor(),
            normalize,
        ])
        train_transform = transforms.Compose([
            LetterboxWholePage(image_size),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        raise ValueError(f"Unsupported preprocessing mode: {preprocessing!r}")
    config["preprocessing"] = preprocessing
    return train_transform, eval_transform, config


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    seed: int,
    prefetch_factor: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=bool(PIN_MEMORY) and device.type == "cuda",
        persistent_workers=bool(PERSISTENT_WORKERS) and num_workers > 0,
        generator=generator,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(**loader_kwargs)


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    use_amp: bool,
    accumulation_steps: int,
    gradient_clip: float,
    description: str,
) -> tuple[float, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    correct = 0
    count = 0

    progress = tqdm(loader, desc=description, dynamic_ncols=True, mininterval=2.0)
    for batch_index, (images, targets, _paths) in enumerate(progress):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast_context(device, use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
            scaled_loss = loss / accumulation_steps
        scaler.scale(scaled_loss).backward()

        should_step = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader)
        if should_step:
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_size = targets.shape[0]
        loss_sum += float(loss.detach()) * batch_size
        correct += int((logits.argmax(dim=1) == targets).sum())
        count += batch_size
        progress.set_postfix(
            loss=f"{loss_sum / max(1, count):.4f}",
            acc=f"{correct / max(1, count):.4f}",
        )

    return loss_sum / max(1, count), correct / max(1, count)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    use_amp: bool,
    description: str,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    progress = tqdm(loader, desc=description, dynamic_ncols=True, mininterval=2.0)
    for images, targets, _paths in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast_context(device, use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        batch_size = targets.shape[0]
        loss_sum += float(loss) * batch_size
        correct += int((logits.argmax(dim=1) == targets).sum())
        count += batch_size
        progress.set_postfix(
            loss=f"{loss_sum / max(1, count):.4f}",
            acc=f"{correct / max(1, count):.4f}",
        )
    return loss_sum / max(1, count), correct / max(1, count)


@torch.inference_mode()
def evaluate_geniza_retrieval(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_amp: bool,
    description: str,
) -> dict[str, float]:
    """Evaluate native pre-classifier embeddings with cosine retrieval.

    For ConvNeXt this is its global-average-pooled representation. For DINOv2,
    timm's native ``forward_head(..., pre_logits=True)`` returns the CLS token.
    """
    model.eval()
    base_model = unwrap_model(model)
    embedding_model: nn.Module = PrelogitExtractor(base_model)
    if isinstance(model, nn.DataParallel):
        embedding_model = nn.DataParallel(
            embedding_model,
            device_ids=model.device_ids,
            output_device=model.output_device,
        )
    embedding_model.eval()
    feature_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    for images, cluster_ids, _paths in tqdm(
        loader, desc=description, dynamic_ncols=True, mininterval=1.0
    ):
        images = images.to(device, non_blocking=True)
        with autocast_context(device, use_amp):
            embeddings = embedding_model(images)
        feature_rows.append(F.normalize(embeddings.float(), dim=1).cpu())
        label_rows.append(cluster_ids.long())
    metrics = compute_retrieval_metrics(
        torch.cat(feature_rows, dim=0), torch.cat(label_rows, dim=0)
    )
    return {
        "map": float(metrics.mean_average_precision),
        "knn1": float(metrics.knn_at_1),
        "knn5": float(metrics.knn_at_5),
        "knn10": float(metrics.knn_at_10),
    }


def checkpoint_payload(
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    label_to_index: Mapping[str, int],
    best_val_accuracy: float,
    args: argparse.Namespace,
    data_config: Mapping[str, object],
) -> dict:
    return {
        "format": "whole_page_backbone_ce_v1",
        "epoch": int(epoch),
        "model_name": args.model,
        "backbone": args.backbone,
        "model_state": unwrap_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "label_to_index": dict(label_to_index),
        "best_val_accuracy": float(best_val_accuracy),
        "stage": args.stage,
        "table_name": args.table_name,
        "data_config": dict(data_config),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }


def write_history(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    device = configure_cuda_device(device, args.gpu_ids)
    args.table_name = args.table_name or (
        STAGE2_TABLE_NAME if args.stage == "stage2" else STAGE1_TABLE_NAME
    )
    if args.learning_rate is None:
        args.learning_rate = (
            MAIN_CHECKPOINT_LEARNING_RATE
            if args.init_checkpoint is not None
            else LEARNING_RATE_STAGE1
        )

    checkpoint_path = args.resume or args.init_checkpoint
    loaded_checkpoint = load_checkpoint(checkpoint_path)
    if loaded_checkpoint is not None:
        checkpoint_model = loaded_checkpoint.get("model_name")
        if checkpoint_model != args.model:
            raise RuntimeError(
                f"Checkpoint model is {checkpoint_model!r}, but --model is {args.model!r}"
            )
        checkpoint_args = loaded_checkpoint.get("args", {})
        checkpoint_preprocessing = checkpoint_args.get("preprocessing", "full-page")
        if checkpoint_preprocessing != args.preprocessing:
            raise RuntimeError(
                f"Checkpoint preprocessing is {checkpoint_preprocessing!r}, but "
                f"--preprocessing is {args.preprocessing!r}"
            )

    raw_splits, split_stats = build_splits(
        BASE_DIR,
        table_name=args.table_name,
        dataset_stage=args.stage,
    )
    splits = {name: flatten_split(raw_splits[name]) for name in ("train", "val", "test")}
    validate_splits(splits)
    splits["train"] = apply_limit(splits["train"], args.limit_train)
    splits["val"] = apply_limit(splits["val"], args.limit_val)
    label_to_index = make_label_map(splits, loaded_checkpoint)
    geniza_records = load_geniza_test(args.geniza_test, args.limit_geniza_test)
    geniza_labels = sorted({record.label for record in geniza_records})
    geniza_label_to_index = {
        label: index for index, label in enumerate(geniza_labels)
    }

    model = timm.create_model(
        args.model,
        pretrained=args.pretrained and loaded_checkpoint is None,
        num_classes=len(label_to_index),
    )
    if loaded_checkpoint is not None:
        model.load_state_dict(loaded_checkpoint["model_state"], strict=True)
    train_transform, eval_transform, data_config = create_transforms(
        model, args.preprocessing
    )
    model.to(device)
    if device.type == "cuda" and len(args.gpu_ids) > 1:
        model = nn.DataParallel(
            model,
            device_ids=args.gpu_ids,
            output_device=args.gpu_ids[0],
        )

    datasets = {
        "train": FullPageDataset(splits["train"], label_to_index, train_transform),
        "val": FullPageDataset(splits["val"], label_to_index, eval_transform),
        "geniza_test": FullPageDataset(
            geniza_records, geniza_label_to_index, eval_transform
        ),
    }
    loaders = {
        name: make_loader(
            dataset,
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=args.num_workers,
            device=device,
            seed=args.seed + index,
            prefetch_factor=args.prefetch_factor,
        )
        for index, (name, dataset) in enumerate(datasets.items())
        if len(dataset) > 0
    }

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=BETAS,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=SCHEDULER_ETA_MIN
    )
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 0
    best_val_accuracy = -math.inf
    if args.resume is not None and loaded_checkpoint is not None:
        optimizer.load_state_dict(loaded_checkpoint["optimizer_state"])
        scheduler.load_state_dict(loaded_checkpoint["scheduler_state"])
        scaler.load_state_dict(loaded_checkpoint.get("scaler_state", {}))
        start_epoch = int(loaded_checkpoint["epoch"]) + 1
        best_val_accuracy = float(loaded_checkpoint.get("best_val_accuracy", -math.inf))

    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = output_dir / args.stage
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "args": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "split_stats": split_stats,
                "split_sizes": {
                    "train": len(splits["train"]),
                    "val": len(splits["val"]),
                    "geniza_test": len(geniza_records),
                },
                "num_classes": len(label_to_index),
                "data_config": data_config,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        f"stage={args.stage} table={args.table_name} backbone={args.backbone} "
        f"model={args.model} preprocessing={args.preprocessing} device={device} "
        f"visible-gpus={args.gpu_ids if device.type == 'cuda' else 'CPU'} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}\n"
        f"pages: train={len(splits['train'])} val={len(splits['val'])} "
        f"Geniza-test={len(geniza_records)}; classes={len(label_to_index)}; "
        f"lr={args.learning_rate:g}; global microbatch={args.batch_size}; "
        f"per-GPU={args.batch_size // max(1, len(args.gpu_ids))}; "
        f"accumulation={args.gradient_accumulation}; "
        f"effective batch={args.batch_size * args.gradient_accumulation}",
        flush=True,
    )

    history: list[dict[str, object]] = []
    for epoch in range(start_epoch, args.epochs):
        started = time.monotonic()
        train_loss, train_accuracy = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            scaler,
            device,
            use_amp=amp_enabled,
            accumulation_steps=args.gradient_accumulation,
            gradient_clip=args.gradient_clip,
            description=f"epoch {epoch + 1}/{args.epochs} train",
        )
        val_loss, val_accuracy = evaluate(
            model,
            loaders["val"],
            criterion,
            device,
            use_amp=amp_enabled,
            description=f"epoch {epoch + 1}/{args.epochs} val",
        )
        test_metrics = evaluate_geniza_retrieval(
            model,
            loaders["geniza_test"],
            device,
            use_amp=amp_enabled,
            description=f"epoch {epoch + 1}/{args.epochs} Geniza test",
        )
        scheduler.step()
        elapsed = time.monotonic() - started
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "test_map": test_metrics["map"],
            "test_knn1": test_metrics["knn1"],
            "test_knn5": test_metrics["knn5"],
            "test_knn10": test_metrics["knn10"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": elapsed,
        }
        history.append(row)
        is_best = val_accuracy > best_val_accuracy
        if is_best:
            best_val_accuracy = val_accuracy
        payload = checkpoint_payload(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            label_to_index=label_to_index,
            best_val_accuracy=best_val_accuracy,
            args=args,
            data_config=data_config,
        )
        torch.save(payload, output_dir / "last.pth")
        if is_best:
            torch.save(payload, output_dir / "best.pth")
        write_history(output_dir / "history.csv", history)
        print(
            f"epoch {epoch + 1:03d}/{args.epochs:03d} "
            f"train loss={train_loss:.4f} acc={train_accuracy:.4f} | "
            f"val loss={val_loss:.4f} acc={val_accuracy:.4f} | "
            f"Geniza test mAP={test_metrics['map']:.4f} "
            f"KNN@1={test_metrics['knn1']:.4f} "
            f"KNN@5={test_metrics['knn5']:.4f} "
            f"KNN@10={test_metrics['knn10']:.4f} | "
            f"{elapsed:.1f}s{' [best val]' if is_best else ''}",
            flush=True,
        )

    best = load_checkpoint(output_dir / "best.pth")
    if best is None:
        raise RuntimeError("Training completed without a best checkpoint")
    unwrap_model(model).load_state_dict(best["model_state"], strict=True)
    test_metrics = evaluate_geniza_retrieval(
        model,
        loaders["geniza_test"],
        device,
        use_amp=amp_enabled,
        description="best checkpoint Geniza test",
    )
    test_result = {
        "checkpoint_epoch": int(best["epoch"]),
        "mAP": test_metrics["map"],
        "KNN@1": test_metrics["knn1"],
        "KNN@5": test_metrics["knn5"],
        "KNN@10": test_metrics["knn10"],
        "num_test_pages": len(datasets["geniza_test"]),
    }
    (output_dir / "test_metrics.json").write_text(
        json.dumps(test_result, indent=2), encoding="utf-8"
    )
    print(
        f"best validation checkpoint | Geniza test mAP={test_metrics['map']:.4f} "
        f"KNN@1={test_metrics['knn1']:.4f} KNN@5={test_metrics['knn5']:.4f} "
        f"KNN@10={test_metrics['knn10']:.4f}",
        flush=True,
    )
    print(f"outputs: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
