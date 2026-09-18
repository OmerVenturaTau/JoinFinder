"""
Measure how sensitive the repo's ConvNeXt encoders are to blur and resolution
loss on real manuscript tiles and glyphs.

This script is designed to answer a practical question:
"Are fine details preserved in the ConvNeXt representation, or does the
embedding mostly ignore them?"

Method:
- extract real page tiles and/or real glyph crops
- create controlled degraded variants:
  - Gaussian blur with increasing sigma
  - downsample -> upsample (resolution loss)
- encode all variants with the same frozen ConvNeXt encoder
- compare each degraded embedding against the original embeddings:
  - cosine to its own original
  - whether it still retrieves itself as nearest neighbor
  - margin to the best non-self original

Outputs:
- per-sample preview grids with metrics overlaid
- CSV with all measurements
- summary Markdown
- aggregate plots of cosine vs degradation severity

Example:
  python debug_tools/ConvNeXt/convnext_blur_sensitivity.py \
      --image-path Drafts/DataMock/990000414120205171/IE13499687/IE13499687_P000001_FL13499689.jpg \
      --xml-path Drafts/DataMock/990000414120205171/IE13499687/IE13499687_P000001_FL13499689_improved_polys.xml
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "system.py").exists())
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torchvision import transforms

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.glyph_branch import GlyphHardQualityFilter, GlyphVisualEncoder
from models.tile_backbones import create_tile_backbone
from system import (
    CHAR_PATCH_SIZE,
    DISABLE_PIL_LIMIT,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    TILE_CONVNEXT_MODEL_NAME,
    TILE_SIZE,
)
from utilities.VisionModule.xml_character_extraction import extract_character_patches


if DISABLE_PIL_LIMIT:
    Image.MAX_IMAGE_PIXELS = None


IMAGENET_MEAN = torch.tensor(NORMALIZE_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor(NORMALIZE_STD, dtype=torch.float32).view(1, 3, 1, 1)


@dataclass(frozen=True)
class ProbeSample:
    modality: str
    name: str
    image: Image.Image
    source: str


@dataclass(frozen=True)
class EncoderSpec:
    key: str
    title: str
    modality: str
    image_size: int
    model: nn.Module


@dataclass(frozen=True)
class VariantSpec:
    family: str
    label: str
    severity: float
    image: Image.Image


@dataclass(frozen=True)
class Measurement:
    encoder_key: str
    modality: str
    sample_name: str
    variant_family: str
    variant_label: str
    severity: float
    self_cosine: float
    best_original: str
    best_cosine: float
    best_nonself_cosine: float
    self_margin: float
    self_rank: int
    self_is_top1: int
    output_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-path", required=True, help="Path to the manuscript page image.")
    parser.add_argument("--xml-path", default=None, help="Optional XML path for extracting glyphs.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "outputs" / "convnext_blur_sensitivity"),
        help="Directory for figures and summaries.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use.",
    )
    parser.add_argument(
        "--encoders",
        nargs="+",
        choices=["tile", "glyph"],
        default=["tile", "glyph"],
        help="Which repo ConvNeXt encoders to probe.",
    )
    parser.add_argument(
        "--max-real-tiles",
        type=int,
        default=6,
        help="Max real page tiles to sample.",
    )
    parser.add_argument(
        "--max-real-glyphs",
        type=int,
        default=12,
        help="Max real glyphs to sample from XML.",
    )
    parser.add_argument(
        "--blur-sigmas",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 1.5, 2.0, 3.0],
        help="Gaussian blur sigma ladder.",
    )
    parser.add_argument(
        "--downsample-scales",
        nargs="+",
        type=float,
        default=[0.85, 0.65, 0.5, 0.35],
        help="Downsample factors before resizing back up.",
    )
    return parser.parse_args()


def ensure_dir(path: str | Path) -> None:
    os.makedirs(path, exist_ok=True)


def pil_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    pipeline = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    return pipeline(image.convert("RGB"))


def tensor_to_numpy_image(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=x.device, dtype=x.dtype)
    std = IMAGENET_STD.to(device=x.device, dtype=x.dtype)
    return (x - mean) / std


def build_tile_encoder(device: torch.device) -> EncoderSpec:
    spec = create_tile_backbone(
        encoder_type="convnext",
        model_name=TILE_CONVNEXT_MODEL_NAME,
        tile_size=TILE_SIZE,
        pretrained=True,
    )
    model = spec.model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return EncoderSpec(
        key="tile_convnext",
        title=f"Tile ConvNeXt ({TILE_CONVNEXT_MODEL_NAME})",
        modality="patch",
        image_size=TILE_SIZE,
        model=model,
    )


def build_glyph_encoder(device: torch.device) -> EncoderSpec:
    glyph_encoder = GlyphVisualEncoder(char_patch_size=CHAR_PATCH_SIZE, encoder_type="convnext_tiny")
    model = glyph_encoder.encoder.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return EncoderSpec(
        key="glyph_convnext",
        title="Glyph ConvNeXt (convnext_tiny)",
        modality="glyph",
        image_size=CHAR_PATCH_SIZE,
        model=model,
    )


def sample_real_tiles(image_path: str, limit: int) -> list[ProbeSample]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    patch_size = TILE_SIZE

    centers = [
        ("center", width / 2, height / 2),
        ("upper_left", width * 0.28, height * 0.28),
        ("upper_right", width * 0.72, height * 0.28),
        ("lower_center", width / 2, height * 0.72),
        ("mid_left", width * 0.22, height / 2),
        ("mid_right", width * 0.78, height / 2),
        ("lower_left", width * 0.28, height * 0.8),
        ("lower_right", width * 0.72, height * 0.8),
    ]

    samples: list[ProbeSample] = []
    for name, cx, cy in centers[:limit]:
        left = int(round(cx - patch_size / 2))
        top = int(round(cy - patch_size / 2))
        left = max(0, min(left, max(0, width - patch_size)))
        top = max(0, min(top, max(0, height - patch_size)))
        crop = image.crop((left, top, min(width, left + patch_size), min(height, top + patch_size)))
        if crop.size != (patch_size, patch_size):
            crop = crop.resize((patch_size, patch_size), resample=Image.Resampling.BICUBIC)
        samples.append(ProbeSample("patch", f"real_tile_{name}", crop, image_path))
    return samples


def sample_real_glyphs(image_path: str, xml_path: str, limit: int) -> list[ProbeSample]:
    image = Image.open(image_path).convert("RGB")
    patches, metadata = extract_character_patches(
        image=image,
        image_path=image_path,
        xml_path=xml_path,
        char_patch_size=CHAR_PATCH_SIZE,
        max_chars=None,
    )
    patch_tensors = [transforms.ToTensor()(patch) for patch in patches]
    quality_filter = GlyphHardQualityFilter()
    filtered_patches, filtered_metadata = quality_filter.filter_glyphs(patch_tensors, metadata)
    diverse_patches, diverse_metadata = quality_filter.sample_diverse_glyphs(
        filtered_patches,
        filtered_metadata,
        max_glyphs=limit,
        prefer_diverse_chars=True,
        min_confidence=0.0,
    )

    to_pil = transforms.ToPILImage()
    samples: list[ProbeSample] = []
    for idx, (patch, meta) in enumerate(zip(diverse_patches, diverse_metadata)):
        char = str(meta.get("char") or "?").strip() or "unknown"
        gc = float(meta.get("gc") or 0.0)
        pil_patch = to_pil(patch.clamp(0.0, 1.0))
        samples.append(ProbeSample("glyph", f"real_glyph_{idx:02d}_{char}_gc{gc:.2f}", pil_patch, image_path))
    return samples


def build_variants(
    image: Image.Image,
    *,
    blur_sigmas: Sequence[float],
    downsample_scales: Sequence[float],
) -> list[VariantSpec]:
    image = image.convert("RGB")
    width, height = image.size
    variants = [VariantSpec("identity", "orig", 0.0, image)]

    for sigma in blur_sigmas:
        variants.append(
            VariantSpec(
                "gaussian_blur",
                f"blur_sigma_{sigma:g}",
                float(sigma),
                image.filter(ImageFilter.GaussianBlur(radius=float(sigma))),
            )
        )

    for scale in downsample_scales:
        down_w = max(4, int(round(width * float(scale))))
        down_h = max(4, int(round(height * float(scale))))
        degraded = image.resize((down_w, down_h), resample=Image.Resampling.BICUBIC).resize(
            (width, height), resample=Image.Resampling.BICUBIC
        )
        variants.append(
            VariantSpec(
                "down_up",
                f"down_up_{scale:g}x",
                1.0 - float(scale),
                degraded,
            )
        )
    return variants


def encode_images(model: nn.Module, images: Sequence[Image.Image], image_size: int, device: torch.device) -> torch.Tensor:
    batch = torch.stack([pil_to_tensor(img, image_size) for img in images]).to(device)
    with torch.no_grad():
        feats = model(normalize_batch(batch))
    return F.normalize(feats, dim=1, eps=1e-8)


def measure_variants(
    *,
    encoder: EncoderSpec,
    samples: Sequence[ProbeSample],
    blur_sigmas: Sequence[float],
    downsample_scales: Sequence[float],
    output_dir: str,
    device: torch.device,
) -> list[Measurement]:
    originals = [sample.image for sample in samples]
    original_emb = encode_images(encoder.model, originals, encoder.image_size, device)
    results: list[Measurement] = []

    preview_dir = os.path.join(output_dir, encoder.key, encoder.modality, "previews")
    overview_dir = os.path.join(output_dir, "degradation_overviews")
    ensure_dir(preview_dir)
    ensure_dir(overview_dir)

    sim_matrix_original = original_emb @ original_emb.T

    for sample_idx, sample in enumerate(samples):
        variants = build_variants(
            sample.image,
            blur_sigmas=blur_sigmas,
            downsample_scales=downsample_scales,
        )
        variant_emb = encode_images(encoder.model, [variant.image for variant in variants], encoder.image_size, device)
        similarity = variant_emb @ original_emb.T

        preview_images: list[torch.Tensor] = []
        preview_titles: list[str] = []

        for variant_idx, variant in enumerate(variants):
            sims = similarity[variant_idx]
            sorted_idx = torch.argsort(sims, descending=True)
            self_cosine = float(sims[sample_idx].item())
            best_idx = int(sorted_idx[0].item())
            best_cosine = float(sims[best_idx].item())

            nonself_mask = torch.ones_like(sims, dtype=torch.bool)
            nonself_mask[sample_idx] = False
            if bool(nonself_mask.any()):
                best_nonself_cosine = float(sims[nonself_mask].max().item())
            else:
                best_nonself_cosine = float("nan")

            self_rank = int((sorted_idx == sample_idx).nonzero(as_tuple=False)[0].item()) + 1
            self_is_top1 = int(best_idx == sample_idx)
            self_margin = self_cosine - best_nonself_cosine if nonself_mask.any() else float("nan")

            out_path = os.path.join(preview_dir, f"{sample.name}.png")
            results.append(
                Measurement(
                    encoder_key=encoder.key,
                    modality=encoder.modality,
                    sample_name=sample.name,
                    variant_family=variant.family,
                    variant_label=variant.label,
                    severity=float(variant.severity),
                    self_cosine=self_cosine,
                    best_original=samples[best_idx].name,
                    best_cosine=best_cosine,
                    best_nonself_cosine=best_nonself_cosine,
                    self_margin=self_margin,
                    self_rank=self_rank,
                    self_is_top1=self_is_top1,
                    output_path=out_path,
                )
            )

            preview_images.append(pil_to_tensor(variant.image, encoder.image_size))
            preview_titles.append(
                f"{variant.label}\nself={self_cosine:.3f} rank={self_rank} top1={self_is_top1}"
            )

        save_sample_preview(
            encoder=encoder,
            sample=sample,
            tensors=preview_images,
            titles=preview_titles,
            output_path=os.path.join(preview_dir, f"{sample.name}.png"),
        )
        if sample_idx == 0:
            save_degradation_overview(
                encoder=encoder,
                sample=sample,
                tensors=preview_images,
                titles=preview_titles,
                output_path=os.path.join(overview_dir, f"{encoder.key}_{encoder.modality}_{sample.name}.png"),
            )

    _ = sim_matrix_original
    return results


def save_sample_preview(
    *,
    encoder: EncoderSpec,
    sample: ProbeSample,
    tensors: Sequence[torch.Tensor],
    titles: Sequence[str],
    output_path: str,
) -> None:
    n = len(tensors)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.6))
    if n == 1:
        axes = [axes]
    for ax, tensor, title in zip(axes, tensors, titles):
        ax.imshow(tensor_to_numpy_image(tensor))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{encoder.title}\n{sample.name}")
    fig.tight_layout()
    ensure_dir(os.path.dirname(output_path))
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_degradation_overview(
    *,
    encoder: EncoderSpec,
    sample: ProbeSample,
    tensors: Sequence[torch.Tensor],
    titles: Sequence[str],
    output_path: str,
) -> None:
    n = len(tensors)
    fig, axes = plt.subplots(2, n, figsize=(2.8 * n, 5.8), height_ratios=[4.0, 1.0])
    image_axes = axes[0]
    label_axes = axes[1]

    for ax, tensor in zip(image_axes, tensors):
        ax.imshow(tensor_to_numpy_image(tensor))
        ax.axis("off")

    for ax, title in zip(label_axes, titles):
        ax.text(0.5, 0.55, title, ha="center", va="center", fontsize=9, wrap=True)
        ax.axis("off")

    fig.suptitle(
        f"Blur and Resolution Degradation\n{encoder.title} | {sample.name}",
        fontsize=13,
    )
    fig.tight_layout()
    ensure_dir(os.path.dirname(output_path))
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_aggregate_plots(results: Sequence[Measurement], output_dir: str) -> None:
    grouped: dict[tuple[str, str, str], list[Measurement]] = {}
    for row in results:
        if row.variant_family == "identity":
            continue
        grouped.setdefault((row.encoder_key, row.modality, row.variant_family), []).append(row)

    plot_dir = os.path.join(output_dir, "aggregate_plots")
    ensure_dir(plot_dir)

    for (encoder_key, modality, variant_family), rows in grouped.items():
        severity_values = sorted({row.severity for row in rows})
        mean_cosines = []
        top1_rates = []
        mean_margins = []

        for severity in severity_values:
            subset = [row for row in rows if row.severity == severity]
            mean_cosines.append(float(np.mean([row.self_cosine for row in subset])))
            top1_rates.append(float(np.mean([row.self_is_top1 for row in subset])))
            mean_margins.append(float(np.mean([row.self_margin for row in subset])))

        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        axes[0].plot(severity_values, mean_cosines, marker="o")
        axes[0].set_title("Mean Self Cosine")
        axes[0].set_xlabel("Severity")
        axes[0].set_ylabel("Cosine")
        axes[0].set_ylim(0.0, 1.01)

        axes[1].plot(severity_values, top1_rates, marker="o")
        axes[1].set_title("Self Top-1 Rate")
        axes[1].set_xlabel("Severity")
        axes[1].set_ylabel("Rate")
        axes[1].set_ylim(0.0, 1.01)

        axes[2].plot(severity_values, mean_margins, marker="o")
        axes[2].set_title("Mean Self Margin")
        axes[2].set_xlabel("Severity")
        axes[2].set_ylabel("Cosine Margin")

        fig.suptitle(f"{encoder_key} | {modality} | {variant_family}")
        fig.tight_layout()
        plot_path = os.path.join(plot_dir, f"{encoder_key}_{modality}_{variant_family}.png")
        fig.savefig(plot_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def write_summary(results: Sequence[Measurement], output_dir: str) -> None:
    ensure_dir(output_dir)
    csv_path = os.path.join(output_dir, "measurements.csv")
    md_path = os.path.join(output_dir, "summary.md")

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Measurement.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(row.__dict__)

    grouped: dict[tuple[str, str, str], list[Measurement]] = {}
    for row in results:
        grouped.setdefault((row.encoder_key, row.modality, row.variant_family), []).append(row)

    lines = [
        "# ConvNeXt Blur Sensitivity Summary",
        "",
        "This experiment measures embedding drift under blur and resolution loss.",
        "",
        "Interpretation:",
        "- High self cosine plus high self top-1 under strong degradation suggests the embedding ignores that detail.",
        "- Fast cosine drop or self top-1 collapse suggests the embedding depends on that detail.",
        "- Margin shows whether the degraded sample remains distinctly itself or drifts toward other originals.",
        "",
    ]

    for key in sorted(grouped.keys()):
        encoder_key, modality, variant_family = key
        rows = [row for row in grouped[key] if row.variant_family != "identity"]
        lines.append(f"## {encoder_key} | {modality} | {variant_family}")
        lines.append("")
        for severity in sorted({row.severity for row in rows}):
            subset = [row for row in rows if row.severity == severity]
            mean_cos = float(np.mean([row.self_cosine for row in subset]))
            top1_rate = float(np.mean([row.self_is_top1 for row in subset]))
            mean_margin = float(np.mean([row.self_margin for row in subset]))
            lines.append(
                f"- severity={severity:.3f}: mean_self_cos={mean_cos:.4f}, "
                f"self_top1_rate={top1_rate:.4f}, mean_self_margin={mean_margin:.4f}"
            )
        lines.append("")

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    ensure_dir(output_dir)
    device = torch.device(args.device)

    encoders: list[EncoderSpec] = []
    if "tile" in args.encoders:
        encoders.append(build_tile_encoder(device))
    if "glyph" in args.encoders:
        encoders.append(build_glyph_encoder(device))

    all_samples: list[ProbeSample] = []
    all_samples.extend(sample_real_tiles(args.image_path, args.max_real_tiles))
    if args.xml_path:
        all_samples.extend(sample_real_glyphs(args.image_path, args.xml_path, args.max_real_glyphs))

    if not all_samples:
        raise RuntimeError("No samples were collected.")

    results: list[Measurement] = []
    for encoder in encoders:
        samples = [sample for sample in all_samples if sample.modality == encoder.modality]
        if not samples:
            continue
        results.extend(
            measure_variants(
                encoder=encoder,
                samples=samples,
                blur_sigmas=args.blur_sigmas,
                downsample_scales=args.downsample_scales,
                output_dir=output_dir,
                device=device,
            )
        )

    if not results:
        raise RuntimeError("No measurements were produced.")

    save_aggregate_plots(results, output_dir)
    write_summary(results, output_dir)

    print(f"Saved blur sensitivity outputs to {output_dir}")
    print(f"Summary: {os.path.join(output_dir, 'summary.md')}")
    print(f"Measurements: {os.path.join(output_dir, 'measurements.csv')}")


if __name__ == "__main__":
    main()
