#!/usr/bin/env python3
"""Evaluate ConvNeXt-Tiny and DINOv2 on OCR polygon-masked fragments."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import timm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from timm.data import resolve_model_data_config
from tqdm.auto import tqdm


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "system.py").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from debug_tools.NeoMME.evaluate_neomme_geniza import (  # noqa: E402
    PageInput,
    load_ocr_fragment_page,
    retrieval_summary,
)
from system import CLUSTER_PAIRS_CSV_PATH  # noqa: E402


MODELS = {
    "convnext": "convnext_tiny.fb_in22k_ft_in1k",
    "dinov2": "vit_base_patch14_dinov2.lvd142m",
}
BATCH_SIZES = {"convnext": 32, "dinov2": 8}
TRAINED_CHECKPOINTS = {
    "convnext": PROJECT_ROOT / "debug_tools/ConvNeXt/outputs/convnext_ce/stage1/best.pth",
    "dinov2": PROJECT_ROOT / "debug_tools/ConvNeXt/outputs/dinov2_ce/stage1/best.pth",
}


class Letterbox:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        scale = min(self.size / image.width, self.size / image.height)
        resized = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.BICUBIC,
        )
        canvas = Image.new("RGB", (self.size, self.size), "white")
        canvas.paste(
            resized,
            ((self.size - resized.width) // 2, (self.size - resized.height) // 2),
        )
        return canvas


class FragmentDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform, margin: float) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.margin = margin

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        page = PageInput(str(row.image_path), str(row.xml_path), str(row.cluster_id))
        image, _text, crop_box = load_ocr_fragment_page(page, self.margin)
        return self.transform(image), str(row.cluster_id), str(row.image_path), crop_box


class NativeEmbedding(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model.forward_features(images)
        return self.model.forward_head(features, pre_logits=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backbones", nargs="+", choices=tuple(MODELS), default=list(MODELS)
    )
    parser.add_argument(
        "--test-csv", type=Path, default=PROJECT_ROOT / CLUSTER_PAIRS_CSV_PATH
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to a separate off-the-shelf or CE-trained output directory.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--crop-margin-ratio", type=float, default=0.03)
    parser.add_argument(
        "--trained",
        action="store_true",
        help="Load the best stage-1 CE-trained checkpoint for each backbone.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(
        "cuda:0" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    frame = pd.read_csv(args.test_csv)
    required = {"image_path", "xml_path", "cluster_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Test CSV missing columns: {sorted(missing)}")
    labels_raw = frame["cluster_id"].astype(str).tolist()
    label_map = {label: i for i, label in enumerate(sorted(set(labels_raw)))}
    labels = torch.tensor([label_map[label] for label in labels_raw])
    output_dir = args.output_dir or (
        PROJECT_ROOT
        / "debug_tools/ConvNeXt/outputs"
        / ("ocr_polygon_trained" if args.trained else "ocr_polygon_off_the_shelf")
    )
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for backbone in args.backbones:
        model_name = MODELS[backbone]
        checkpoint = None
        if args.trained:
            checkpoint_path = TRAINED_CHECKPOINTS[backbone]
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            checkpoint_model = checkpoint.get("model_name")
            if checkpoint_model != model_name:
                raise RuntimeError(
                    f"Checkpoint model {checkpoint_model!r} does not match {model_name!r}"
                )
            model = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=len(checkpoint["label_to_index"]),
            )
            model.load_state_dict(checkpoint["model_state"], strict=True)
        else:
            checkpoint_path = None
            model = timm.create_model(model_name, pretrained=True)
        config = dict(resolve_model_data_config(model))
        size = int(config["input_size"][-1])
        transform = transforms.Compose(
            [
                Letterbox(size),
                transforms.ToTensor(),
                transforms.Normalize(config["mean"], config["std"]),
            ]
        )
        dataset = FragmentDataset(frame, transform, args.crop_margin_ratio)
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZES[backbone],
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        encoder = NativeEmbedding(model).to(device).eval()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        rows = []
        paths = []
        boxes = []
        started = time.monotonic()
        for images, _cluster_ids, batch_paths, batch_boxes in tqdm(
            loader, desc=f"{backbone} OCR-fragment inference", dynamic_ncols=True
        ):
            images = images.to(device, non_blocking=True)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                features = encoder(images)
            rows.append(F.normalize(features.float(), dim=1).cpu())
            paths.extend(batch_paths)
            boxes.extend(torch.stack(batch_boxes, dim=1).tolist())
        features = torch.cat(rows)
        result = retrieval_summary(features, labels)
        metrics = {
            "model": model_name,
            "frozen_off_the_shelf": not args.trained,
            "ce_trained": args.trained,
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "checkpoint_epoch": checkpoint.get("epoch") if checkpoint else None,
            "checkpoint_best_val_accuracy": (
                checkpoint.get("best_val_accuracy") if checkpoint else None
            ),
            "training_preprocessing": (
                checkpoint.get("args", {}).get("preprocessing", "full-page")
                if checkpoint
                else None
            ),
            "input": "ALTO TextBlock polygon-masked fragment; 3% margin; white exterior; letterboxed",
            "num_test_pages": len(frame),
            "num_clusters": len(label_map),
            "input_size": size,
            "embedding_dim": int(features.shape[1]),
            "metrics": result,
            "peak_gpu_allocated_GB": torch.cuda.max_memory_allocated(device) / 1e9
            if device.type == "cuda"
            else 0.0,
            "seconds": time.monotonic() - started,
        }
        np.savez_compressed(
            output_dir / f"{backbone}_embeddings.npz",
            features=features.numpy(),
            image_paths=np.asarray(paths, dtype=str),
            cluster_ids=np.asarray(labels_raw, dtype=str),
            crop_boxes=np.asarray(boxes, dtype=np.int32),
        )
        (output_dir / f"{backbone}_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        all_metrics[backbone] = metrics
        print(json.dumps({backbone: metrics}, indent=2), flush=True)
        del encoder, model, loader, dataset, features
        if device.type == "cuda":
            torch.cuda.empty_cache()
    (output_dir / "metrics.json").write_text(
        json.dumps(all_metrics, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
