#!/usr/bin/env python3
"""Evaluate native and CE-trained RootSIFT on ALTO TextBlock fragments."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
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
from debug_tools.SIFT.train_sift_ce import FEATURE_DIM, SiftCEModel  # noqa: E402
from system import CLUSTER_PAIRS_CSV_PATH  # noqa: E402


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "debug_tools/SIFT/outputs/sift_ce/stage1/best.pth"
)


def rootsift_fragment(image, max_side: int, max_keypoints: int) -> np.ndarray:
    image = image.convert("L")
    scale = min(1.0, max_side / max(image.size))
    if scale < 1.0:
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    sift = cv2.SIFT_create(nfeatures=max_keypoints)
    _keypoints, descriptors = sift.detectAndCompute(np.asarray(image), None)
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(FEATURE_DIM, dtype=np.float32)
    descriptors = descriptors.astype(np.float32, copy=False)
    descriptors /= descriptors.sum(axis=1, keepdims=True) + 1e-7
    descriptors = np.sqrt(descriptors)
    descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-7
    feature = np.concatenate((descriptors.mean(0), descriptors.std(0)))
    norm = float(np.linalg.norm(feature))
    if norm > 0:
        feature /= norm
    return feature.astype(np.float32, copy=False)


class FragmentSiftDataset(Dataset):
    def __init__(self, frame, margin: float, max_side: int, max_keypoints: int):
        self.frame = frame.reset_index(drop=True)
        self.margin = margin
        self.max_side = max_side
        self.max_keypoints = max_keypoints

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        page = PageInput(str(row.image_path), str(row.xml_path), str(row.cluster_id))
        image, _text, crop_box = load_ocr_fragment_page(page, self.margin)
        feature = rootsift_fragment(image, self.max_side, self.max_keypoints)
        return torch.from_numpy(feature), str(row.image_path), crop_box


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-csv", type=Path, default=PROJECT_ROOT / CLUSTER_PAIRS_CSV_PATH
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "debug_tools/SIFT/outputs/ocr_polygon",
    )
    parser.add_argument("--crop-margin-ratio", type=float, default=0.03)
    parser.add_argument("--resize-max-side", type=int, default=1600)
    parser.add_argument("--max-keypoints", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.test_csv)
    required = {"image_path", "xml_path", "cluster_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Test CSV missing columns: {sorted(missing)}")
    raw_labels = frame.cluster_id.astype(str).tolist()
    label_map = {label: i for i, label in enumerate(sorted(set(raw_labels)))}
    labels = torch.tensor([label_map[label] for label in raw_labels])

    dataset = FragmentSiftDataset(
        frame, args.crop_margin_ratio, args.resize_max_side, args.max_keypoints
    )
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    rows, paths, boxes = [], [], []
    started = time.monotonic()
    for features, batch_paths, batch_boxes in tqdm(
        loader, desc="RootSIFT TextBlock extraction", dynamic_ncols=True
    ):
        rows.append(features.float())
        paths.extend(batch_paths)
        boxes.extend(torch.stack(batch_boxes, dim=1).tolist())
    native = torch.cat(rows)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = SiftCEModel(
        len(checkpoint["label_to_index"]), int(checkpoint["embedding_dim"])
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    with torch.inference_mode():
        trained = F.normalize(model(native, return_embedding=True), dim=1)

    output = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "input": "ALTO TextBlock polygon-masked fragment; 3% margin; white exterior",
        "num_test_pages": len(frame),
        "num_clusters": len(label_map),
        "max_keypoints": args.max_keypoints,
        "resize_max_side": args.resize_max_side,
        "native_rootsift": retrieval_summary(native, labels),
        "ce_trained": retrieval_summary(trained, labels),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_val_accuracy": checkpoint.get("best_val_accuracy"),
        "seconds": time.monotonic() - started,
    }
    np.savez_compressed(
        output / "embeddings.npz",
        native_features=native.numpy(),
        trained_features=trained.numpy(),
        image_paths=np.asarray(paths, dtype=str),
        cluster_ids=np.asarray(raw_labels, dtype=str),
        crop_boxes=np.asarray(boxes, dtype=np.int32),
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
