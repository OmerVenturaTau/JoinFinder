#!/usr/bin/env python3
"""Evaluate NeoMME-800M-Retriever on OCR-masked Geniza fragments plus OCR.

This is inference-only. NeoMME-Retriever does not permit pixels and text in one
conversation, so each page is encoded twice: once from the ALTO TextBlock
polygon-masked fragment and once from raw ALTO OCR. Their dense vectors are
with equal weight for late fusion. Identifiers, cluster labels, file names, page
numbers, and other metadata are never included in either model input.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "system.py").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from system import CLUSTER_PAIRS_CSV_PATH  # noqa: E402
from train.metric_learning import compute_retrieval_metrics  # noqa: E402


DEFAULT_MODEL = "Hcompany/NeoMME-800M-Retriever"
DEFAULT_OUTPUT = PROJECT_ROOT / "Debugs/NeoMME/outputs/neomme_800m_ocr_crop"


@dataclass(frozen=True)
class PageInput:
    image_path: str
    xml_path: str
    cluster_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=PROJECT_ROOT / CLUSTER_PAIRS_CSV_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--crop-margin-ratio",
        type=float,
        default=0.03,
        help="Margin around the ALTO TextBlock fragment polygon bounding box.",
    )
    parser.add_argument(
        "--attention-implementation",
        choices=("sdpa", "eager"),
        default="sdpa",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates all 242 pages.")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.crop_margin_ratio < 0:
        parser.error("--crop-margin-ratio must be non-negative")
    if args.limit < 0 or args.save_every <= 0:
        parser.error("--limit must be non-negative and --save-every must be positive")
    return args


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 CPU inference is unsupported; use bfloat16 or float32")
    return dtype


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_alto_crop_and_text(xml_path: str) -> tuple[tuple[float, float, float, float], tuple[int, int], str]:
    """Return TextLine union, ALTO page size, and raw OCR in XML line order."""
    root = ET.parse(xml_path).getroot()
    page_width = page_height = 0
    for element in root.iter():
        if local_name(element.tag) == "Page":
            page_width = int(float(element.attrib.get("WIDTH", 0)))
            page_height = int(float(element.attrib.get("HEIGHT", 0)))
            break
    if page_width <= 0 or page_height <= 0:
        raise ValueError(f"ALTO has no valid page dimensions: {xml_path}")

    boxes: list[tuple[float, float, float, float]] = []
    lines: list[str] = []
    for line in root.iter():
        if local_name(line.tag) != "TextLine":
            continue
        words = [
            child.attrib.get("CONTENT", "").strip()
            for child in line.iter()
            if local_name(child.tag) == "String"
            and child.attrib.get("CONTENT", "").strip()
        ]
        if words:
            lines.append(" ".join(words))
        try:
            x = float(line.attrib["HPOS"])
            y = float(line.attrib["VPOS"])
            width = float(line.attrib["WIDTH"])
            height = float(line.attrib["HEIGHT"])
        except (KeyError, TypeError, ValueError):
            continue
        if width > 0 and height > 0:
            boxes.append((x, y, x + width, y + height))

    if not boxes:
        raise ValueError(f"ALTO has no valid TextLine boxes: {xml_path}")
    ocr_text = "\n".join(lines).strip()
    if not ocr_text:
        raise ValueError(f"ALTO has no OCR String content: {xml_path}")
    bounds = (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )
    return bounds, (page_width, page_height), ocr_text


def load_cropped_page(page: PageInput, margin_ratio: float) -> tuple[Image.Image, str, tuple[int, int, int, int]]:
    bounds, xml_size, ocr_text = parse_alto_crop_and_text(page.xml_path)
    with Image.open(page.image_path) as source:
        image = source.convert("RGB")
    scale_x = image.width / xml_size[0]
    scale_y = image.height / xml_size[1]
    left, top, right, bottom = (
        bounds[0] * scale_x,
        bounds[1] * scale_y,
        bounds[2] * scale_x,
        bounds[3] * scale_y,
    )
    margin_x = (right - left) * margin_ratio
    margin_y = (bottom - top) * margin_ratio
    crop_box = (
        max(0, math.floor(left - margin_x)),
        max(0, math.floor(top - margin_y)),
        min(image.width, math.ceil(right + margin_x)),
        min(image.height, math.ceil(bottom + margin_y)),
    )
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        raise ValueError(f"Invalid crop {crop_box} for {page.image_path}")
    return image.crop(crop_box), ocr_text, crop_box


def load_ocr_fragment_page(
    page: PageInput, margin_ratio: float
) -> tuple[Image.Image, str, tuple[int, int, int, int]]:
    """Crop and mask to ALTO TextBlock polygons, falling back to text lines."""
    root = ET.parse(page.xml_path).getroot()
    _, xml_size, ocr_text = parse_alto_crop_and_text(page.xml_path)
    polygons: list[list[tuple[float, float]]] = []
    for block in root.iter():
        if local_name(block.tag) != "TextBlock":
            continue
        for element in block.iter():
            if local_name(element.tag) != "Polygon":
                continue
            points: list[tuple[float, float]] = []
            try:
                coordinates = [
                    float(value)
                    for value in element.attrib.get("POINTS", "").replace(",", " ").split()
                ]
                if len(coordinates) % 2:
                    coordinates = []
                points = list(zip(coordinates[0::2], coordinates[1::2]))
            except ValueError:
                points = []
            if len(points) >= 3:
                polygons.append(points)
                break
    if not polygons:
        return load_cropped_page(page, margin_ratio)

    with Image.open(page.image_path) as source:
        image = source.convert("RGB")
    scale_x = image.width / xml_size[0]
    scale_y = image.height / xml_size[1]
    scaled = [[(x * scale_x, y * scale_y) for x, y in poly] for poly in polygons]
    all_points = [point for poly in scaled for point in poly]
    left = min(point[0] for point in all_points)
    top = min(point[1] for point in all_points)
    right = max(point[0] for point in all_points)
    bottom = max(point[1] for point in all_points)
    margin_x = (right - left) * margin_ratio
    margin_y = (bottom - top) * margin_ratio
    crop_box = (
        max(0, math.floor(left - margin_x)),
        max(0, math.floor(top - margin_y)),
        min(image.width, math.ceil(right + margin_x)),
        min(image.height, math.ceil(bottom + margin_y)),
    )
    crop = image.crop(crop_box)
    mask = Image.new("L", crop.size, 0)
    draw = ImageDraw.Draw(mask)
    for poly in scaled:
        draw.polygon(
            [(x - crop_box[0], y - crop_box[1]) for x, y in poly], fill=255
        )
    white = Image.new("RGB", crop.size, "white")
    white.paste(crop, mask=mask)
    return white, ocr_text, crop_box


def load_pages(path: Path, limit: int) -> list[PageInput]:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    frame = pd.read_csv(resolved)
    required = {"image_path", "xml_path", "cluster_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Test CSV is missing columns: {sorted(missing)}")
    if frame["image_path"].duplicated().any():
        raise ValueError("Test CSV contains duplicate image paths")
    pages = [
        PageInput(str(row.image_path), str(row.xml_path), str(row.cluster_id))
        for row in frame.itertuples(index=False)
    ]
    pages = pages if limit == 0 else pages[:limit]
    for page in pages:
        if not Path(page.image_path).is_file():
            raise FileNotFoundError(page.image_path)
        if not Path(page.xml_path).is_file():
            raise FileNotFoundError(page.xml_path)
    return pages


def save_embedding_cache(
    path: Path,
    image_features: list[np.ndarray],
    text_features: list[np.ndarray],
    pages: list[PageInput],
    image_token_counts: list[int],
    text_token_counts: list[int],
    crop_boxes: list[tuple[int, int, int, int]],
) -> None:
    if len(image_features) != len(text_features):
        raise ValueError("Image/text embedding cache lengths differ")
    count = len(image_features)
    image_matrix = np.stack(image_features).astype(np.float32) if count else np.empty((0, 0), np.float32)
    text_matrix = np.stack(text_features).astype(np.float32) if count else np.empty((0, 0), np.float32)
    np.savez_compressed(
        path,
        image_features=image_matrix,
        text_features=text_matrix,
        image_paths=np.asarray([p.image_path for p in pages[:count]], dtype=str),
        cluster_ids=np.asarray([p.cluster_id for p in pages[:count]], dtype=str),
        image_token_counts=np.asarray(image_token_counts, dtype=np.int32),
        text_token_counts=np.asarray(text_token_counts, dtype=np.int32),
        crop_boxes=np.asarray(crop_boxes, dtype=np.int32),
    )


def load_embedding_cache(path: Path, pages: list[PageInput]):
    if not path.exists():
        return [], [], [], [], []
    saved = np.load(path, allow_pickle=False)
    saved_paths = saved["image_paths"].tolist()
    expected = [page.image_path for page in pages[: len(saved_paths)]]
    if saved_paths != expected:
        raise RuntimeError("Partial cache page order does not match this test set")
    return (
        [row.copy() for row in saved["image_features"]],
        [row.copy() for row in saved["text_features"]],
        saved["image_token_counts"].astype(int).tolist(),
        saved["text_token_counts"].astype(int).tolist(),
        [tuple(map(int, row)) for row in saved["crop_boxes"]],
    )


def retrieval_summary(features: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    result = compute_retrieval_metrics(features, labels)
    return {
        "mAP": float(result.mean_average_precision),
        "KNN@1": float(result.knn_at_1),
        "KNN@5": float(result.knn_at_5),
        "KNN@10": float(result.knn_at_10),
    }


def main() -> int:
    from transformers import NeoMMEForRetrieval, NeoMMEProcessor

    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "embeddings_partial.npz"
    pages = load_pages(args.test_csv, args.limit)

    processor = NeoMMEProcessor.from_pretrained(args.model, cache_dir=args.cache_dir)
    model = NeoMMEForRetrieval.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        dtype=dtype,
        attn_implementation=args.attention_implementation,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    if args.resume:
        (
            image_feature_rows,
            text_feature_rows,
            image_token_counts,
            text_token_counts,
            crop_boxes,
        ) = load_embedding_cache(partial_path, pages)
    else:
        image_feature_rows = []
        text_feature_rows = []
        image_token_counts = []
        text_token_counts = []
        crop_boxes = []
    start_index = len(image_feature_rows)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started = time.monotonic()
    progress = tqdm(
        range(start_index, len(pages)),
        initial=start_index,
        total=len(pages),
        desc="NeoMME-800M Geniza inference",
        dynamic_ncols=True,
    )
    for index in progress:
        page = pages[index]
        cropped_image, ocr_text, crop_box = load_ocr_fragment_page(
            page, args.crop_margin_ratio
        )
        # Deliberately exclude every metadata field. NeoMME-Retriever requires
        # image and text to be separate conversations, so encode both through
        # the shared document space and fuse only their final normalized vectors.
        image_messages = [[{
            "role": "user",
            "content": [{"type": "image", "image": cropped_image}],
        }]]
        text_messages = [[{
            "role": "user",
            "content": [{"type": "text", "text": ocr_text}],
        }]]
        image_inputs = processor.apply_chat_template(
            image_messages,
            task="document",
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": "longest"},
        ).to(device)
        text_inputs = processor.apply_chat_template(
            text_messages,
            task="document",
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": "longest"},
        ).to(device)
        context = (
            torch.autocast(device_type="cuda", dtype=dtype)
            if device.type == "cuda" and dtype != torch.float32
            else nullcontext()
        )
        with torch.inference_mode(), context:
            image_outputs = model(
                **image_inputs,
                output_multivector=False,
                output_dense=True,
            )
            text_outputs = model(
                **text_inputs,
                output_multivector=False,
                output_dense=True,
            )
        image_dense = F.normalize(image_outputs.dense_embeddings.float(), dim=-1)
        text_dense = F.normalize(text_outputs.dense_embeddings.float(), dim=-1)
        image_feature_rows.append(image_dense[0].cpu().numpy())
        text_feature_rows.append(text_dense[0].cpu().numpy())
        image_token_counts.append(int(image_inputs["attention_mask"].sum().item()))
        text_token_counts.append(int(text_inputs["attention_mask"].sum().item()))
        crop_boxes.append(crop_box)
        if (index + 1) % args.save_every == 0:
            save_embedding_cache(
                partial_path,
                image_feature_rows,
                text_feature_rows,
                pages,
                image_token_counts,
                text_token_counts,
                crop_boxes,
            )
        if device.type == "cuda":
            progress.set_postfix(
                peak_GB=f"{torch.cuda.max_memory_allocated(device) / 1e9:.2f}",
                image_tokens=image_token_counts[-1],
                text_tokens=text_token_counts[-1],
            )

    save_embedding_cache(
        partial_path,
        image_feature_rows,
        text_feature_rows,
        pages,
        image_token_counts,
        text_token_counts,
        crop_boxes,
    )
    image_features = F.normalize(torch.from_numpy(np.stack(image_feature_rows)).float(), dim=1)
    text_features = F.normalize(torch.from_numpy(np.stack(text_feature_rows)).float(), dim=1)
    # Two simple inference-only fusion baselines. The normalized arithmetic
    # mean is the literal average of the two modality embeddings. Concatenation
    # keeps the modalities separate and averages their cosine similarities.
    averaged_features = F.normalize(image_features + text_features, dim=1)
    fused_features = torch.cat((image_features, text_features), dim=1) / math.sqrt(2.0)
    label_values = [page.cluster_id for page in pages]
    label_map = {label: i for i, label in enumerate(sorted(set(label_values)))}
    labels = torch.tensor([label_map[label] for label in label_values])
    peak_allocated = (
        torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(device) / 1e9 if device.type == "cuda" else 0.0
    )
    metrics = {
        "model": args.model,
        "input": "separate ALTO TextBlock polygon-masked fragment and raw OCR encodings; no metadata",
        "fusion": (
            "reports both normalized vector averaging and equal-weight "
            "concatenation of normalized image/OCR dense vectors"
        ),
        "num_test_pages": len(pages),
        "single_modality_embedding_dim": int(image_features.shape[1]),
        "fused_embedding_dim": int(fused_features.shape[1]),
        "representations": {
            "image_crop_only": retrieval_summary(image_features, labels),
            "ocr_only": retrieval_summary(text_features, labels),
            "image_ocr_normalized_vector_average": retrieval_summary(averaged_features, labels),
            "image_ocr_equal_weight_fusion": retrieval_summary(fused_features, labels),
        },
        "crop_margin_ratio": args.crop_margin_ratio,
        "image_tokens_min": min(image_token_counts),
        "image_tokens_median": float(np.median(image_token_counts)),
        "image_tokens_max": max(image_token_counts),
        "text_tokens_min": min(text_token_counts),
        "text_tokens_median": float(np.median(text_token_counts)),
        "text_tokens_max": max(text_token_counts),
        "peak_gpu_allocated_GB": peak_allocated,
        "peak_gpu_reserved_GB": peak_reserved,
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
