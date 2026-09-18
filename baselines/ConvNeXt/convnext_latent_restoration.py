"""
Probe how much fine detail the repo's ConvNeXt encoders preserve by inverting
their pooled latent representations back into image space.

The script supports two input modes:
1. Synthetic probes (default): high-frequency patches and rendered glyphs.
2. Real manuscript inputs: tiles from a page image and glyph crops from XML.

For each selected ConvNeXt encoder, we:
- encode the source image to the final pooled feature vector used by the backbone
- optimize a new image whose pooled feature matches the source feature
- save side-by-side comparisons plus a CSV/Markdown summary of reconstruction loss

This is not a trained decoder. It is feature inversion. If fine details are
preserved in the latent space, they should be recoverable with limited blur and
without collapsing into generic shapes.

Examples:
  python debug_tools/ConvNeXt/convnext_latent_restoration.py

  python debug_tools/ConvNeXt/convnext_latent_restoration.py \
      --image-path Drafts/DataMock/.../page.jpg \
      --xml-path Drafts/DataMock/.../page_improved_polys.xml \
      --max-real-tiles 4 \
      --max-real-glyphs 8
"""

from __future__ import annotations

import argparse
import csv
import math
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
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torchvision import transforms


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.glyph_branch import GlyphVisualEncoder, GlyphHardQualityFilter
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
class ReconstructionResult:
    encoder_key: str
    modality: str
    sample_name: str
    source: str
    cosine: float
    l2: float
    pixel_l1: float
    pixel_l2: float
    output_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "outputs" / "convnext_latent_restoration"),
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
        "--steps",
        type=int,
        default=250,
        help="Optimization steps per inversion.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.08,
        help="Learning rate for image optimization.",
    )
    parser.add_argument(
        "--tv-weight",
        type=float,
        default=2e-4,
        help="Total-variation regularization weight.",
    )
    parser.add_argument(
        "--pixel-weight",
        type=float,
        default=0.02,
        help="Weak image-space anchor toward the original, to stabilize inversion.",
    )
    parser.add_argument(
        "--init",
        choices=["noise", "blur", "gray"],
        default="blur",
        help="Initialization for the reconstruction image.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed.",
    )
    parser.add_argument(
        "--image-path",
        default=None,
        help="Optional real manuscript page image for real tile/glyph probes.",
    )
    parser.add_argument(
        "--xml-path",
        default=None,
        help="Optional ALTO/XML path used when extracting real glyphs.",
    )
    parser.add_argument(
        "--max-real-tiles",
        type=int,
        default=4,
        help="Max real page tiles to sample if --image-path is provided.",
    )
    parser.add_argument(
        "--max-real-glyphs",
        type=int,
        default=8,
        help="Max real glyph probes to use if --image-path and --xml-path are provided.",
    )
    parser.add_argument(
        "--max-synthetic-patches",
        type=int,
        default=6,
        help="Synthetic patch probes to generate.",
    )
    parser.add_argument(
        "--max-synthetic-glyphs",
        type=int,
        default=6,
        help="Synthetic glyph probes to generate.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> None:
    os.makedirs(path, exist_ok=True)


def pil_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    return transform(image.convert("RGB"))


def tensor_to_numpy_image(t: torch.Tensor) -> np.ndarray:
    chw = t.detach().cpu().clamp(0.0, 1.0)
    return chw.permute(1, 2, 0).numpy()


def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=x.device, dtype=x.dtype)
    std = IMAGENET_STD.to(device=x.device, dtype=x.dtype)
    return (x - mean) / std


def total_variation(x: torch.Tensor) -> torch.Tensor:
    tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w


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


def encode_image(model: nn.Module, image_tensor: torch.Tensor) -> torch.Tensor:
    batch = normalize_batch(image_tensor.unsqueeze(0))
    return model(batch)


def make_init_tensor(
    source_tensor: torch.Tensor,
    mode: str,
    generator: torch.Generator,
) -> torch.Tensor:
    if mode == "gray":
        init = torch.full_like(source_tensor, 0.5)
    elif mode == "noise":
        init = torch.rand(source_tensor.shape, generator=generator, device=source_tensor.device)
    elif mode == "blur":
        blurred = transforms.GaussianBlur(kernel_size=11, sigma=4.0)(source_tensor.cpu())
        init = blurred.to(source_tensor.device)
    else:
        raise ValueError(f"Unknown init mode: {mode}")
    init = init.clamp(0.0, 1.0)
    return torch.logit(init.mul(0.98).add(0.01))


def invert_feature(
    model: nn.Module,
    source_tensor: torch.Tensor,
    *,
    steps: int,
    lr: float,
    tv_weight: float,
    pixel_weight: float,
    init_mode: str,
    seed: int,
) -> tuple[torch.Tensor, float, float]:
    device = source_tensor.device
    target = encode_image(model, source_tensor).detach()

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    logits = make_init_tensor(source_tensor, init_mode, generator).unsqueeze(0).detach().clone()
    logits.requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=lr)

    final_cosine = math.nan
    final_l2 = math.nan

    for _ in range(steps):
        recon = torch.sigmoid(logits)
        recon_feat = model(normalize_batch(recon))
        feat_mse = F.mse_loss(recon_feat, target)
        feat_cos = 1.0 - F.cosine_similarity(recon_feat, target, dim=1).mean()
        pixel_l1 = F.l1_loss(recon, source_tensor.unsqueeze(0))
        tv = total_variation(recon)
        loss = feat_mse + 0.1 * feat_cos + pixel_weight * pixel_l1 + tv_weight * tv

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        final_cosine = float(1.0 - feat_cos.detach().item())
        final_l2 = float(F.mse_loss(recon_feat, target).detach().item())

    result = torch.sigmoid(logits.detach()).squeeze(0)
    return result, final_cosine, final_l2


def save_comparison_figure(
    *,
    encoder: EncoderSpec,
    sample: ProbeSample,
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    cosine: float,
    l2: float,
    output_path: str,
) -> tuple[float, float]:
    original_np = tensor_to_numpy_image(original)
    reconstruction_np = tensor_to_numpy_image(reconstruction)
    abs_diff = np.abs(original_np - reconstruction_np)

    pixel_l1 = float(abs_diff.mean())
    pixel_l2 = float(np.mean((original_np - reconstruction_np) ** 2))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original_np)
    axes[0].set_title("Original")
    axes[1].imshow(reconstruction_np)
    axes[1].set_title("Reconstruction")
    axes[2].imshow(abs_diff, cmap="magma")
    axes[2].set_title("Absolute Diff")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{encoder.title}\n{sample.name} | cosine={cosine:.4f} | feat_mse={l2:.6f} | pixel_l1={pixel_l1:.4f}"
    )
    fig.tight_layout()
    ensure_dir(os.path.dirname(output_path))
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return pixel_l1, pixel_l2


def render_text_image(text: str, image_size: int, font_size: int, rotate_deg: float = 0.0) -> Image.Image:
    image = Image.new("RGB", (image_size, image_size), (250, 247, 240))
    draw = ImageDraw.Draw(image)
    font = load_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    xy = ((image_size - text_w) / 2, (image_size - text_h) / 2 - bbox[1])
    draw.text(xy, text, fill=(20, 20, 20), font=font)
    if rotate_deg:
        image = image.rotate(rotate_deg, resample=Image.Resampling.BICUBIC, fillcolor=(250, 247, 240))
    return image


def load_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansHebrew-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansHebrew-Regular.ttf",
    ]
    for candidate in font_candidates:
        if os.path.exists(candidate):
            return ImageFont.truetype(candidate, font_size)
    return ImageFont.load_default()


def make_synthetic_patch_samples(limit: int) -> list[ProbeSample]:
    patch_size = TILE_SIZE
    background = (245, 240, 228)
    ink = (20, 18, 15)
    samples: list[ProbeSample] = []

    def blank_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
        canvas = Image.new("RGB", (patch_size, patch_size), background)
        return canvas, ImageDraw.Draw(canvas)

    canvas, draw = blank_canvas()
    for y in range(20, patch_size, 24):
        draw.line((0, y, patch_size, y), fill=ink, width=1)
    samples.append(ProbeSample("patch", "parallel_1px_lines", canvas, "synthetic"))

    canvas, draw = blank_canvas()
    step = 12
    for y in range(0, patch_size, step):
        for x in range(0, patch_size, step):
            if ((x // step) + (y // step)) % 2 == 0:
                draw.rectangle((x, y, x + step - 1, y + step - 1), fill=ink)
    samples.append(ProbeSample("patch", "checkerboard_12px", canvas, "synthetic"))

    canvas, draw = blank_canvas()
    center = patch_size // 2
    for radius in range(25, center - 10, 20):
        draw.ellipse((center - radius, center - radius, center + radius, center + radius), outline=ink, width=2)
    samples.append(ProbeSample("patch", "concentric_rings", canvas, "synthetic"))

    canvas, draw = blank_canvas()
    rng = np.random.default_rng(7)
    for _ in range(120):
        x = int(rng.integers(8, patch_size - 8))
        y = int(rng.integers(8, patch_size - 8))
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=ink)
    samples.append(ProbeSample("patch", "micro_dots", canvas, "synthetic"))

    canvas, draw = blank_canvas()
    for offset in range(-patch_size // 2, patch_size // 2, 18):
        draw.line((0, center + offset, center + offset, 0), fill=ink, width=2)
        draw.line((patch_size - 1, center + offset, center + offset, patch_size - 1), fill=ink, width=2)
    samples.append(ProbeSample("patch", "diagonal_fan", canvas, "synthetic"))

    canvas, draw = blank_canvas()
    for y in range(40, patch_size - 40, 42):
        draw.line((40, y, patch_size - 40, y), fill=ink, width=5)
        draw.line((40, y + 9, patch_size - 40, y + 9), fill=ink, width=1)
    samples.append(ProbeSample("patch", "mixed_thick_thin", canvas, "synthetic"))

    return samples[:limit]


def make_synthetic_glyph_samples(limit: int) -> list[ProbeSample]:
    glyph_specs = [
        ("aleph_clean", "א", 86, 0),
        ("shin_clean", "ש", 86, 0),
        ("mem_rotated", "מ", 86, -8),
        ("latin_B", "B", 88, 0),
        ("digit_8", "8", 92, 0),
        ("aleph_blurred", "א", 86, 0),
    ]
    samples: list[ProbeSample] = []
    for name, glyph, font_size, rotation in glyph_specs[:limit]:
        image = render_text_image(glyph, CHAR_PATCH_SIZE, font_size, rotate_deg=rotation)
        if name == "aleph_blurred":
            image = image.filter(ImageFilter.GaussianBlur(radius=1.1))
        samples.append(ProbeSample("glyph", name, image, "synthetic"))
    return samples


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
        name = f"real_glyph_{idx:02d}_{char}_gc{gc:.2f}"
        samples.append(ProbeSample("glyph", name, pil_patch, image_path))
    return samples


def gather_samples(args: argparse.Namespace) -> list[ProbeSample]:
    samples: list[ProbeSample] = []
    samples.extend(make_synthetic_patch_samples(args.max_synthetic_patches))
    samples.extend(make_synthetic_glyph_samples(args.max_synthetic_glyphs))

    if args.image_path:
        samples.extend(sample_real_tiles(args.image_path, args.max_real_tiles))
        if args.xml_path:
            samples.extend(sample_real_glyphs(args.image_path, args.xml_path, args.max_real_glyphs))

    return samples


def filter_samples_for_encoder(samples: Sequence[ProbeSample], encoder: EncoderSpec) -> list[ProbeSample]:
    return [sample for sample in samples if sample.modality == encoder.modality]


def write_summary(results: Sequence[ReconstructionResult], output_dir: str) -> None:
    csv_path = os.path.join(output_dir, "summary.csv")
    md_path = os.path.join(output_dir, "summary.md")
    ensure_dir(output_dir)

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "encoder_key",
                "modality",
                "sample_name",
                "source",
                "cosine",
                "l2",
                "pixel_l1",
                "pixel_l2",
                "output_path",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row.__dict__)

    grouped: dict[str, list[ReconstructionResult]] = {}
    for result in results:
        grouped.setdefault(result.encoder_key, []).append(result)

    lines = [
        "# ConvNeXt Latent Restoration Summary",
        "",
        "Feature inversion from the repo's ConvNeXt pooled embeddings.",
        "",
    ]
    for encoder_key, rows in grouped.items():
        mean_cos = sum(r.cosine for r in rows) / max(1, len(rows))
        mean_l1 = sum(r.pixel_l1 for r in rows) / max(1, len(rows))
        lines.append(f"## {encoder_key}")
        lines.append("")
        lines.append(f"- samples: {len(rows)}")
        lines.append(f"- mean feature cosine: {mean_cos:.4f}")
        lines.append(f"- mean pixel L1: {mean_l1:.4f}")
        lines.append("")
        rows_sorted = sorted(rows, key=lambda r: (r.modality, -r.cosine, r.sample_name))
        for row in rows_sorted:
            lines.append(
                f"- `{row.sample_name}` ({row.modality}): cosine={row.cosine:.4f}, "
                f"pixel_l1={row.pixel_l1:.4f}, figure=`{row.output_path}`"
            )
        lines.append("")

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device = torch.device(args.device)
    encoders: list[EncoderSpec] = []
    if "tile" in args.encoders:
        encoders.append(build_tile_encoder(device))
    if "glyph" in args.encoders:
        encoders.append(build_glyph_encoder(device))

    samples = gather_samples(args)
    if not samples:
        raise RuntimeError("No probe samples were collected.")

    results: list[ReconstructionResult] = []

    for encoder_idx, encoder in enumerate(encoders):
        encoder_samples = filter_samples_for_encoder(samples, encoder)
        if not encoder_samples:
            continue

        for sample_idx, sample in enumerate(encoder_samples):
            source_tensor = pil_to_tensor(sample.image, encoder.image_size).to(device)
            reconstruction, cosine, l2 = invert_feature(
                encoder.model,
                source_tensor,
                steps=args.steps,
                lr=args.lr,
                tv_weight=args.tv_weight,
                pixel_weight=args.pixel_weight,
                init_mode=args.init,
                seed=args.seed + 1000 * encoder_idx + sample_idx,
            )

            sample_dir = os.path.join(args.output_dir, encoder.key, sample.modality)
            ensure_dir(sample_dir)
            output_path = os.path.join(sample_dir, f"{sample.name}.png")
            pixel_l1, pixel_l2 = save_comparison_figure(
                encoder=encoder,
                sample=sample,
                original=source_tensor,
                reconstruction=reconstruction,
                cosine=cosine,
                l2=l2,
                output_path=output_path,
            )
            results.append(
                ReconstructionResult(
                    encoder_key=encoder.key,
                    modality=sample.modality,
                    sample_name=sample.name,
                    source=sample.source,
                    cosine=cosine,
                    l2=l2,
                    pixel_l1=pixel_l1,
                    pixel_l2=pixel_l2,
                    output_path=output_path,
                )
            )

    if not results:
        raise RuntimeError("No reconstructions were produced.")

    write_summary(results, args.output_dir)
    print(f"Saved {len(results)} reconstructions to {args.output_dir}")
    print(f"Summary: {os.path.join(args.output_dir, 'summary.md')}")


if __name__ == "__main__":
    main()
