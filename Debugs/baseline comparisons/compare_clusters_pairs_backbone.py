#!/usr/bin/env python3
"""Evaluate frozen ConvNeXt, DINOv2, or AlephBERT on the cluster test.

This is an off-the-shelf visual-backbone baseline: it does not load a trained
JoinsFinder checkpoint and does not use its learned 512-D branch adapters.

For visual encoders, each complete image is passed once through the native
pretrained model preprocessing and the backbone's raw 768-D output is used as
the page embedding. AlephBERT reads the OCR XML paired with each image and
mean-pools its native token states over the complete page (chunking only when
the model context limit requires it). An explicit ``--input-mode tiles`` option
is retained for studying pooled local visual features. Page vectors are
L2-normalized and evaluated with cosine similarity using the same metrics as
the SIFT baseline.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Dict, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import timm
from timm.data import create_transform, resolve_model_data_config
from transformers import AutoModel, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "system.py").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_clusters_pairs_sift import (  # noqa: E402
    cluster_retrieval_metrics,
    pair_separation_metrics,
)
from models.tile_backbones import create_tile_backbone  # noqa: E402
from system import (  # noqa: E402
    CLUSTERING_MAX_TILES_EVAL,
    CLUSTERING_TILE_SIZE,
    CLUSTERING_TILE_STRIDE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
)
from train.dataset import ManuscriptDataset, tile_collate_with_padding  # noqa: E402
from utilities.ContextModule.xml_word_extraction import extract_words_from_alto  # noqa: E402
from system import OCR_STRING_CONFIDENCE_THRESHOLD  # noqa: E402


DEFAULT_INPUT = PROJECT_ROOT / "results_analysis/test_set/clusters_images_metadata.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_MODELS = {
    "convnext": "convnext_tiny.fb_in22k_ft_in1k",
    "dinov2": "vit_base_patch14_dinov2.lvd142m",
    "alephbert": "onlplab/alephbert-base",
}


def pool_tile_descriptors(descriptors: np.ndarray, pooling: str) -> np.ndarray:
    """Pool a non-empty [num_tiles, feature_dim] descriptor matrix."""
    values = np.asarray(descriptors, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"Expected non-empty tile descriptors [N,D], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Tile descriptors contain non-finite values")

    mean = values.mean(axis=0)
    if pooling == "mean":
        page = mean
    elif pooling == "mean_std":
        # Population std is deterministic and remains well-defined for one tile.
        page = np.concatenate((mean, values.std(axis=0, ddof=0)), axis=0)
    else:
        raise ValueError(f"Unknown pooling mode: {pooling}")

    norm = float(np.linalg.norm(page))
    if norm < 1e-12:
        raise ValueError("Pooled page descriptor has zero norm")
    return (page / norm).astype(np.float32, copy=False)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_members(path: Path, image_limit: int) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls", ".xlsm"}:
        frame = pd.read_excel(path)
    else:
        frame = pd.read_csv(path)
    required = {"image_path", "cluster_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame["image_path"].astype(str).duplicated().any():
        raise ValueError("Input contains duplicate image_path rows; expected one row per image")
    if image_limit > 0:
        frame = frame.head(image_limit).copy()
    return frame.reset_index(drop=True)


def _device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return device


def _make_dataset(frame: pd.DataFrame, *, tile_size: int, stride: int, max_tiles: int) -> ManuscriptDataset:
    labels = frame["cluster_id"].astype(str).tolist()
    label2idx = {label: index for index, label in enumerate(dict.fromkeys(labels))}
    xml_paths = frame["xml_path"].tolist() if "xml_path" in frame.columns else None
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])
    return ManuscriptDataset(
        frame["image_path"].astype(str).tolist(),
        labels,
        transform,
        label2idx,
        xml_paths=xml_paths,
        patch_size=int(tile_size),
        stride=int(stride),
        max_tiles_per_image=int(max_tiles),
        split="test",
        patch_loading_method="extract",
        use_db_coordinates=False,
        use_xml_extraction=True,
        use_visual_mod=True,
        use_char_mod=False,
        use_word_mod=False,
    )


class FullImageDataset(torch.utils.data.Dataset):
    """Load one native-preprocessed tensor for each complete manuscript image."""

    def __init__(self, paths: Sequence[str], transform: Callable[[Image.Image], torch.Tensor]) -> None:
        self.paths = [str(path).strip() for path in paths]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, path


def _as_feature_matrix(output: object) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not torch.is_tensor(output):
        raise TypeError(f"Backbone returned unsupported output type: {type(output)}")
    if output.ndim > 2:
        output = output.flatten(start_dim=2).mean(dim=2)
    if output.ndim != 2:
        raise ValueError(f"Expected backbone output [N,D], got {tuple(output.shape)}")
    return output


def mean_pool_alephbert_chunks(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    special_tokens_mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Pool native token states, excluding padding and special tokens."""
    valid = attention_mask.bool() & ~special_tokens_mask.bool()
    token_count = int(valid.sum().item())
    if token_count == 0:
        raise ValueError("AlephBERT input has no non-special tokens")
    token_sum = (hidden * valid.unsqueeze(-1)).sum(dim=(0, 1))
    return token_sum / token_count, token_count


@torch.inference_mode()
def extract_alephbert_vectors(
    frame: pd.DataFrame,
    *,
    model_name: str,
    max_length: int,
    ocr_min_confidence: float,
    device: torch.device,
    amp: bool,
) -> tuple[Dict[str, np.ndarray], Dict[str, int], dict]:
    """Encode all usable OCR belonging to each image with native AlephBERT."""
    if "xml_path" not in frame.columns:
        raise ValueError("AlephBERT requires an xml_path column paired with each image")
    if max_length < 4:
        raise ValueError("AlephBERT max length must be at least 4 tokens")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # AlephBERT's checkpoint does not contain a trained BERT pooler. We do not
    # instantiate or use that randomly initialized layer; only native encoder
    # token states participate in the deterministic mean below.
    model = AutoModel.from_pretrained(model_name, add_pooling_layer=False).to(device).eval()
    feature_dim = int(model.config.hidden_size)
    native_max_length = int(model.config.max_position_embeddings)
    if max_length > native_max_length:
        raise ValueError(
            f"AlephBERT max length {max_length} exceeds the model limit {native_max_length}"
        )
    vectors: Dict[str, np.ndarray] = {}
    chunk_counts: Dict[str, int] = {}
    word_counts: Dict[str, int] = {}
    token_counts: Dict[str, int] = {}
    failures: list[dict[str, str]] = []

    amp_context: Callable[[], object]
    if amp and device.type == "cuda":
        amp_context = lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        amp_context = nullcontext

    for row_number, row in enumerate(frame.itertuples(index=False), start=1):
        image_path = str(row.image_path).strip()
        xml_path = str(row.xml_path).strip()
        try:
            words, _ = extract_words_from_alto(
                xml_path,
                string_conf_threshold=float(ocr_min_confidence),
                use_hebrew_dict=False,
                max_words=None,
            )
            if not words:
                raise ValueError("no usable Hebrew OCR words")
            # XML extraction already sorts pages, lines, and words in Hebrew
            # reading order. Overflow chunks prevent context-limit truncation.
            encoded = tokenizer(
                " ".join(words),
                padding=True,
                truncation=True,
                max_length=int(max_length),
                return_overflowing_tokens=True,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            special_tokens_mask = encoded.pop("special_tokens_mask")
            encoded.pop("overflow_to_sample_mapping", None)
            chunk_counts[image_path] = int(encoded["input_ids"].shape[0])
            model_inputs = {key: value.to(device) for key, value in encoded.items()}
            with amp_context():
                hidden = model(**model_inputs).last_hidden_state.float().cpu()
            page_embedding, token_count = mean_pool_alephbert_chunks(
                hidden,
                encoded["attention_mask"],
                special_tokens_mask,
            )
            embedding = page_embedding.numpy()
            norm = float(np.linalg.norm(embedding))
            if not np.isfinite(embedding).all() or norm < 1e-12:
                raise ValueError("AlephBERT produced an invalid page embedding")
            vectors[image_path] = (embedding / norm).astype(np.float32, copy=False)
            word_counts[image_path] = len(words)
            token_counts[image_path] = token_count
        except Exception as exc:
            failures.append({"image_path": image_path, "xml_path": xml_path, "error": str(exc)})
        if row_number == 1 or row_number % 25 == 0 or row_number == len(frame):
            chunks = chunk_counts.get(image_path, 0)
            print(f"[alephbert] {row_number}/{len(frame)} pages; {chunks} chunks for {image_path}")

    if not vectors:
        raise RuntimeError(f"AlephBERT found no usable OCR pages: {failures[:3]}")
    metadata = {
        "backbone": "alephbert",
        "modality": "ocr_text",
        "model_name": model_name,
        "pretrained": True,
        "input_mode": "ocr_xml",
        "adapter": None,
        "pooling": "mean_non_special_token_states_across_page_chunks",
        "page_vector_dim": feature_dim,
        "max_length": int(max_length),
        "ocr_min_confidence": float(ocr_min_confidence),
        "use_hebrew_dictionary_filter": False,
        "num_images_requested": int(len(frame)),
        "num_images_with_vectors": int(len(vectors)),
        "num_images_without_vectors": int(len(failures)),
        "total_ocr_words": int(sum(word_counts.values())),
        "total_encoded_tokens": int(sum(token_counts.values())),
        "total_chunks": int(sum(chunk_counts.values())),
        "failures": failures,
    }
    return vectors, chunk_counts, metadata


@torch.inference_mode()
def extract_full_image_vectors(
    frame: pd.DataFrame,
    *,
    backbone_name: str,
    model_name: str,
    image_batch_size: int,
    num_workers: int,
    device: torch.device,
    pretrained: bool,
    amp: bool,
) -> tuple[Dict[str, np.ndarray], Dict[str, int], dict]:
    """Use the unmodified model's native output for one complete image."""
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0).to(device).eval()
    feature_dim = int(getattr(model, "num_features", getattr(model, "embed_dim", 0)))
    if feature_dim <= 0:
        raise RuntimeError(f"Could not infer native feature dimension for {model_name}")
    data_config = resolve_model_data_config(model)
    native_transform = create_transform(**data_config, is_training=False)
    dataset = FullImageDataset(frame["image_path"].astype(str).tolist(), native_transform)
    loader = DataLoader(
        dataset,
        batch_size=int(image_batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
    )

    vectors: Dict[str, np.ndarray] = {}
    amp_context: Callable[[], object]
    if amp and device.type == "cuda":
        amp_context = lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        amp_context = nullcontext

    seen = 0
    for images, paths in loader:
        images = images.to(device, non_blocking=True)
        with amp_context():
            embeddings = _as_feature_matrix(model(images)).float().cpu().numpy()
        for embedding, raw_path in zip(embeddings, paths):
            path = str(raw_path).strip()
            norm = float(np.linalg.norm(embedding))
            if not np.isfinite(embedding).all() or norm < 1e-12:
                raise RuntimeError(f"Backbone produced an invalid embedding for {path}")
            vectors[path] = (embedding / norm).astype(np.float32, copy=False)
            seen += 1
            if seen == 1 or seen % 25 == 0 or seen == len(frame):
                print(f"[{backbone_name}] {seen}/{len(frame)} full images; {path}")

    if len(vectors) != len(frame):
        raise RuntimeError(f"Expected {len(frame)} full-image vectors, got {len(vectors)}")
    metadata = {
        "backbone": backbone_name,
        "model_name": model_name,
        "pretrained": bool(pretrained),
        "input_mode": "full_image",
        "adapter": None,
        "pooling": None,
        "page_vector_dim": feature_dim,
        "native_input_size": list(data_config.get("input_size", ())),
        "native_interpolation": str(data_config.get("interpolation", "")),
        "native_crop_pct": float(data_config.get("crop_pct", 1.0)),
        "num_images_requested": int(len(frame)),
        "num_images_with_vectors": int(len(vectors)),
        "failures": [],
    }
    # A value of one means one complete image generated each page embedding.
    return vectors, {path: 1 for path in vectors}, metadata


@torch.inference_mode()
def extract_page_vectors(
    frame: pd.DataFrame,
    *,
    backbone_name: str,
    model_name: str,
    pooling: str,
    tile_size: int,
    stride: int,
    max_tiles: int,
    image_batch_size: int,
    tile_batch_size: int,
    num_workers: int,
    device: torch.device,
    pretrained: bool,
    amp: bool,
) -> tuple[Dict[str, np.ndarray], Dict[str, int], dict]:
    spec = create_tile_backbone(
        encoder_type=backbone_name,
        model_name=model_name,
        tile_size=int(tile_size),
        pretrained=pretrained,
    )
    model = spec.model.to(device).eval()
    dataset = _make_dataset(frame, tile_size=tile_size, stride=stride, max_tiles=max_tiles)
    loader = DataLoader(
        dataset,
        batch_size=int(image_batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=tile_collate_with_padding,
    )

    vectors: Dict[str, np.ndarray] = {}
    tile_counts: Dict[str, int] = {}
    failures: list[dict[str, str]] = []
    amp_context: Callable[[], object]
    if amp and device.type == "cuda":
        amp_context = lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        amp_context = nullcontext

    seen = 0
    for batch in loader:
        if batch is None:
            continue
        tiles, valid_mask = batch[0], batch[1]
        paths: Sequence[str] = batch[-1]
        batch_size, max_batch_tiles = valid_mask.shape
        flat_tiles = tiles.reshape(batch_size * max_batch_tiles, *tiles.shape[2:])
        flat_valid = valid_mask.reshape(-1)
        valid_indices = flat_valid.nonzero(as_tuple=False).squeeze(1)
        encoded_parts = []
        for start in range(0, int(valid_indices.numel()), int(tile_batch_size)):
            indices = valid_indices[start : start + int(tile_batch_size)]
            tile_chunk = flat_tiles[indices].to(device, non_blocking=True)
            with amp_context():
                encoded_parts.append(_as_feature_matrix(model(tile_chunk)).float().cpu())
        encoded = torch.cat(encoded_parts, dim=0).numpy() if encoded_parts else np.empty((0, spec.feat_dim), np.float32)

        owner = (valid_indices.cpu().numpy() // max_batch_tiles) if valid_indices.numel() else np.empty(0, dtype=int)
        for image_index, raw_path in enumerate(paths):
            path = str(raw_path).strip()
            descriptors = encoded[owner == image_index]
            tile_counts[path] = int(descriptors.shape[0])
            try:
                vectors[path] = pool_tile_descriptors(descriptors, pooling)
            except Exception as exc:
                failures.append({"path": path, "error": str(exc)})
            seen += 1
            if seen == 1 or seen % 25 == 0 or seen == len(frame):
                print(f"[{backbone_name}] {seen}/{len(frame)} images; {descriptors.shape[0]} tiles for {path}")

    expected_dim = int(spec.feat_dim) * (2 if pooling == "mean_std" else 1)
    metadata = {
        "backbone": backbone_name,
        "model_name": model_name,
        "pretrained": bool(pretrained),
        "input_mode": "tiles",
        "adapter": None,
        "pooling": pooling,
        "backbone_feature_dim": int(spec.feat_dim),
        "page_vector_dim": expected_dim,
        "tile_size": int(tile_size),
        "stride": int(stride),
        "max_tiles": int(max_tiles),
        "num_images_requested": int(len(frame)),
        "num_images_with_vectors": int(len(vectors)),
        "failures": failures,
    }
    if failures:
        raise RuntimeError(f"Backbone extraction failed for {len(failures)} images: {failures[:3]}")
    if len(vectors) != len(frame):
        raise RuntimeError(f"Expected {len(frame)} page vectors, got {len(vectors)}")
    return vectors, tile_counts, metadata


def build_pair_table(
    frame: pd.DataFrame,
    vectors: Dict[str, np.ndarray],
    unit_counts: Dict[str, int],
    score_col: str,
    unit_name: str,
) -> pd.DataFrame:
    paths = frame["image_path"].astype(str).tolist()
    matrix = np.stack([vectors[path] for path in paths], axis=0)
    similarities = matrix @ matrix.T
    rows = []
    for i in range(len(frame)):
        for j in range(i + 1, len(frame)):
            row_i, row_j = frame.iloc[i], frame.iloc[j]
            out = {}
            for column in frame.columns:
                out[f"{column}_1"] = row_i[column]
                out[f"{column}_2"] = row_j[column]
            out["are_they_same_clusters"] = row_i["cluster_id"] == row_j["cluster_id"]
            out[score_col] = float(similarities[i, j])
            out["similarity_score"] = float(similarities[i, j])
            out[f"image_1_num_{unit_name}"] = unit_counts[paths[i]]
            out[f"image_2_num_{unit_name}"] = unit_counts[paths[j]]
            rows.append(out)
    return pd.DataFrame(rows)


def write_outputs(
    output_path: Path,
    pairs: pd.DataFrame,
    metrics: pd.DataFrame,
    metadata: dict,
    vectors: Dict[str, np.ndarray],
    unit_counts: Dict[str, int],
    unit_name: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        pairs.to_excel(writer, index=False, sheet_name="pairs")
        metrics.to_excel(writer, index=False, sheet_name="metrics")
        pd.DataFrame([metadata]).to_excel(writer, index=False, sheet_name="backbone_metadata")
    metrics.to_csv(output_path.with_name(output_path.stem + "_metrics.csv"), index=False)
    paths = list(vectors)
    np.savez_compressed(
        output_path.with_name(output_path.stem + "_latents.npz"),
        paths=np.asarray(paths, dtype=str),
        latents=np.stack([vectors[path] for path in paths]),
        num_encoding_units=np.asarray([unit_counts[path] for path in paths], dtype=np.int32),
        encoding_unit=np.asarray(unit_name),
        metadata=json.dumps(metadata, ensure_ascii=False),
    )


def evaluate_backbone(
    frame: pd.DataFrame,
    *,
    backbone_name: str,
    model_name: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Path:
    if backbone_name == "alephbert":
        vectors, unit_counts, metadata = extract_alephbert_vectors(
            frame,
            model_name=model_name,
            max_length=args.alephbert_max_length,
            ocr_min_confidence=args.ocr_min_confidence,
            device=device,
            amp=args.amp,
        )
        representation_name = "native"
        unit_name = "text_chunks"
    elif args.input_mode == "full_image":
        vectors, unit_counts, metadata = extract_full_image_vectors(
            frame,
            backbone_name=backbone_name,
            model_name=model_name,
            image_batch_size=args.image_batch_size,
            num_workers=args.num_workers,
            device=device,
            pretrained=args.pretrained,
            amp=args.amp,
        )
        representation_name = "native"
        unit_name = "full_images"
    else:
        vectors, unit_counts, metadata = extract_page_vectors(
            frame,
            backbone_name=backbone_name,
            model_name=model_name,
            pooling=args.pooling,
            tile_size=args.tile_size,
            stride=args.stride,
            max_tiles=args.max_tiles,
            image_batch_size=args.image_batch_size,
            tile_batch_size=args.tile_batch_size,
            num_workers=args.num_workers,
            device=device,
            pretrained=args.pretrained,
            amp=args.amp,
        )
        representation_name = f"tiles_{args.pooling}"
        unit_name = "tiles"
    evaluated_frame = frame[frame["image_path"].astype(str).isin(vectors)].reset_index(drop=True)
    if len(evaluated_frame) != len(frame):
        print(
            f"[{backbone_name}] scoring {len(evaluated_frame)}/{len(frame)} images; "
            f"{len(frame) - len(evaluated_frame)} have no usable encoder input"
        )
    score_col = f"{backbone_name}_similarity_score"
    pairs = build_pair_table(evaluated_frame, vectors, unit_counts, score_col, unit_name)

    def get_vector(path_value: object) -> np.ndarray | None:
        return vectors.get(str(path_value).strip())

    metric_rows = pair_separation_metrics(pairs, score_col)
    metric_rows.extend(cluster_retrieval_metrics(evaluated_frame, get_vector, score_col=score_col))
    metrics = pd.DataFrame(metric_rows)
    output = args.output_dir / f"clusters_images_metadata_pairs_{backbone_name}_{representation_name}.xlsx"
    write_outputs(output, pairs, metrics, metadata, vectors, unit_counts, unit_name)
    retrieval_set = f"{score_col}/cluster_retrieval"
    result = metrics[(metrics["metric_set"] == retrieval_set) & metrics["metric"].isin(["mAP", "hit@1", "hit@5", "hit@10"])]
    print(f"\n{backbone_name} ({model_name}, {metadata['page_vector_dim']}D)")
    for row in result.itertuples(index=False):
        print(f"  {row.metric}: {float(row.value):.6f}")
    print(f"  output: {output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_table", nargs="?", default=str(DEFAULT_INPUT))
    parser.add_argument("--backbone", choices=tuple(DEFAULT_MODELS), nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--convnext-model", default=DEFAULT_MODELS["convnext"])
    parser.add_argument("--dinov2-model", default=DEFAULT_MODELS["dinov2"])
    parser.add_argument("--alephbert-model", default=DEFAULT_MODELS["alephbert"])
    parser.add_argument("--alephbert-max-length", type=int, default=512)
    parser.add_argument("--ocr-min-confidence", type=float, default=OCR_STRING_CONFIDENCE_THRESHOLD)
    parser.add_argument(
        "--input-mode",
        choices=("full_image", "tiles"),
        default="full_image",
        help="full_image uses the raw native backbone embedding; tiles is an optional pooled-tile baseline.",
    )
    parser.add_argument("--pooling", choices=("mean", "mean_std"), default="mean_std", help="Used only with --input-mode tiles.")
    parser.add_argument("--tile-size", type=int, default=CLUSTERING_TILE_SIZE)
    parser.add_argument("--stride", type=int, default=CLUSTERING_TILE_STRIDE)
    parser.add_argument("--max-tiles", type=int, default=CLUSTERING_MAX_TILES_EVAL)
    parser.add_argument("--image-batch-size", type=int, default=2)
    parser.add_argument("--tile-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-limit", type=int, default=0, help="Smoke-test limit; 0 evaluates all images.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir = _resolve_path(args.output_dir)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    input_path = _resolve_path(args.input_table)
    frame = _load_members(input_path, args.image_limit)
    device = _device_from_arg(args.device)
    print(f"Loaded {len(frame)} images from {input_path}; device={device}; input_mode={args.input_mode}")
    model_names = {
        "convnext": args.convnext_model,
        "dinov2": args.dinov2_model,
        "alephbert": args.alephbert_model,
    }
    for backbone_name in args.backbone:
        evaluate_backbone(
            frame,
            backbone_name=backbone_name,
            model_name=model_names[backbone_name],
            args=args,
            device=device,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
