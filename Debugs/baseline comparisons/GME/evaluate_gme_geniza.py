#!/usr/bin/env python3
"""Evaluate GME-Qwen2-VL-2B on OCR-cropped Geniza fragments.

The frozen model produces three representations per page: fragment image only,
raw OCR only, and a native joint image+OCR embedding. No metadata is encoded.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers

# GME's official remote wrapper uses the Transformers 4.x class name. It was
# renamed in Transformers 5.x; both names dispatch to the same auto-model type.
if int(transformers.__version__.split(".", 1)[0]) >= 5:
    transformers.AutoModelForVision2Seq = transformers.AutoModelForImageTextToText

from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "system.py").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Debugs.NeoMME.evaluate_neomme_geniza import (  # noqa: E402
    load_cropped_page,
    load_pages,
    retrieval_summary,
)
from system import CLUSTER_PAIRS_CSV_PATH  # noqa: E402


DEFAULT_MODEL = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
DEFAULT_OUTPUT = PROJECT_ROOT / "Debugs/GME/outputs/gme_qwen2_vl_2b_fragment_crop"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--test-csv", type=Path, default=PROJECT_ROOT / CLUSTER_PAIRS_CSV_PATH
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float16",), default="float16")
    parser.add_argument("--crop-margin-ratio", type=float, default=0.03)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()
    if args.crop_margin_ratio < 0 or args.limit < 0 or args.save_every <= 0:
        parser.error("crop margin/limit must be non-negative and save-every positive")
    return args


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def save_cache(
    path: Path,
    image_rows: list[np.ndarray],
    text_rows: list[np.ndarray],
    joint_rows: list[np.ndarray],
    image_paths: list[str],
    cluster_ids: list[str],
    crop_boxes: list[tuple[int, int, int, int]],
) -> None:
    count = len(image_rows)
    empty = np.empty((0, 0), dtype=np.float32)
    np.savez_compressed(
        path,
        image_features=np.stack(image_rows).astype(np.float32) if count else empty,
        text_features=np.stack(text_rows).astype(np.float32) if count else empty,
        joint_features=np.stack(joint_rows).astype(np.float32) if count else empty,
        image_paths=np.asarray(image_paths[:count], dtype=str),
        cluster_ids=np.asarray(cluster_ids[:count], dtype=str),
        crop_boxes=np.asarray(crop_boxes, dtype=np.int32),
    )


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else PROJECT_ROOT / args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pages = load_pages(args.test_csv, args.limit)

    model = SentenceTransformer(
        args.model,
        device=str(device),
        trust_remote_code=True,
    )
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    image_rows: list[np.ndarray] = []
    text_rows: list[np.ndarray] = []
    joint_rows: list[np.ndarray] = []
    crop_boxes: list[tuple[int, int, int, int]] = []
    image_paths = [page.image_path for page in pages]
    cluster_ids = [page.cluster_id for page in pages]
    partial_path = output_dir / "embeddings_partial.npz"
    started = time.monotonic()

    progress = tqdm(pages, desc="GME-Qwen2-VL-2B Geniza inference", dynamic_ncols=True)
    for index, page in enumerate(progress):
        fragment, ocr_text, crop_box = load_cropped_page(
            page, args.crop_margin_ratio
        )
        # The three entries use GME's official single-modal and fused-modal
        # input formats. Labels and every metadata field remain out of input.
        embeddings = model.encode(
            [
                {"image": fragment},
                {"text": ocr_text},
                {"image": fragment, "text": ocr_text},
            ],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        image_rows.append(np.asarray(embeddings[0], dtype=np.float32))
        text_rows.append(np.asarray(embeddings[1], dtype=np.float32))
        joint_rows.append(np.asarray(embeddings[2], dtype=np.float32))
        crop_boxes.append(crop_box)
        if (index + 1) % args.save_every == 0:
            save_cache(
                partial_path,
                image_rows,
                text_rows,
                joint_rows,
                image_paths,
                cluster_ids,
                crop_boxes,
            )
        if device.type == "cuda":
            progress.set_postfix(
                peak_GB=f"{torch.cuda.max_memory_allocated(device) / 1e9:.2f}"
            )

    save_cache(
        partial_path,
        image_rows,
        text_rows,
        joint_rows,
        image_paths,
        cluster_ids,
        crop_boxes,
    )
    representations = {
        "image_crop_only": F.normalize(torch.from_numpy(np.stack(image_rows)), dim=1),
        "ocr_only": F.normalize(torch.from_numpy(np.stack(text_rows)), dim=1),
        "native_joint_image_ocr": F.normalize(
            torch.from_numpy(np.stack(joint_rows)), dim=1
        ),
    }
    label_map = {label: i for i, label in enumerate(sorted(set(cluster_ids)))}
    labels = torch.tensor([label_map[label] for label in cluster_ids])
    metrics = {
        "model": args.model,
        "input": "ALTO text-line union crop and raw OCR; no metadata",
        "num_test_pages": len(pages),
        "num_clusters": len(label_map),
        "embedding_dim": int(representations["native_joint_image_ocr"].shape[1]),
        "representations": {
            name: retrieval_summary(features, labels)
            for name, features in representations.items()
        },
        "crop_margin_ratio": args.crop_margin_ratio,
        "peak_gpu_allocated_GB": (
            torch.cuda.max_memory_allocated(device) / 1e9
            if device.type == "cuda"
            else 0.0
        ),
        "peak_gpu_reserved_GB": (
            torch.cuda.max_memory_reserved(device) / 1e9
            if device.type == "cuda"
            else 0.0
        ),
        "seconds": time.monotonic() - started,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    final_path = output_dir / "embeddings.npz"
    partial_path.replace(final_path)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"embeddings: {final_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
