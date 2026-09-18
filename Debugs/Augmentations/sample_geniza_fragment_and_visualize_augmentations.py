#!/usr/bin/env python3
"""
Pick a Geniza fragment, sample one tile + one glyph, and visualize the current
training augmentations for both modalities. The glyph is chosen **uniformly at
random** from the same diverse pool training uses (not always the first letter).

Outputs:
  Debugs/Augmentations/outputs/sample_geniza_fragment_and_visualize_augmentations/
    tile_aug_grid.png
    glyph_aug_grid.png
    selected_geniza_row.json
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
import sys
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib-cache")

import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import functional as TVF

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "system.py").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from system import (  # noqa: E402
    DB_CONFIG_PATH,
    GENIZA_IMAGE_LATENTS_TABLE,
    GENIZA_IMAGE_INFORMATION_TABLE,
    GENIZA_CONTRASTIVE_TABLE,
    PRETRAIN_TABLE_NAME,
    TILE_SIZE,
    TILE_STRIDE,
    MAX_TILES_EVAL,
    CHAR_PATCH_SIZE,
    MAX_CHARS_PER_IMAGE,
    DISABLE_PIL_LIMIT,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    AUGMENT_COLOR_JITTER,
    AUGMENT_APPLY_PROB,
    AUGMENT_COLOR_JITTER_BRIGHTNESS,
    AUGMENT_COLOR_JITTER_CONTRAST,
    AUGMENT_COLOR_JITTER_SATURATION,
    AUGMENT_COLOR_JITTER_HUE,
    AUGMENT_COLOR_JITTER_PROB,
    AUGMENT_RANDOM_GRAYSCALE_PROB,
    AUGMENT_GAUSSIAN_BLUR_PROB,
    AUGMENT_GAUSSIAN_BLUR_KERNEL,
    GLYPH_GAUSSIAN_BLUR_KERNEL,
    AUGMENT_BACKGROUND_RANDOM_LIBRARY_PROB,
    AUGMENT_PARCHMENT_STAINS_PROB,
    AUGMENT_WHITE_BACKGROUND_PROB,
    AUGMENT_BACKGROUND_SAMPLING_ALPHA,
    AUGMENT_BACKGROUND_SAMPLING_FLOOR,
    AUGMENT_LOCAL_TEXTURE_PROB,
    AUGMENT_TONE_CONTRAST_PROB,
    AUGMENT_BORDER_CROP_PROB,
    AUGMENT_RANDAUGMENT_PROB,
    AUGMENT_ZOOM_PROB,
    AUGMENT_ZOOM_SCALE_RANGE,
    AUGMENT_TILT_PROB,
    GLYPH_AUG_APPLY_PROB,
    GLYPH_AUG_PARCHMENT_STAINS_PROB,
    GLYPH_AUG_WHITE_BACKGROUND_PROB,
    AUGMENT_RESOLUTION_JITTER_PROB,
    AUGMENT_LOCAL_TEXTURE_ALPHA_RANGE,
    AUGMENT_BORDER_MAX_FRAC,
    AUGMENT_EDGE_CROP_MAX_FRAC,
    CHAR_PATCH_APPLY_RANDAUGMENT,
    CHAR_PATCH_APPLY_IMAGENET_NORM,
)
from utilities.VisionModule.xml_patch_extraction import extract_patches_with_xml  # noqa: E402
from utilities.VisionModule.xml_character_extraction import extract_character_patches  # noqa: E402
from models.glyph_branch import GlyphHardQualityFilter  # noqa: E402
from utilities.augmentations.background_overlays import RandomLibraryBackgroundOverlay  # noqa: E402
from utilities.augmentations.manuscript_augmentations import (  # noqa: E402
    RandomResolutionJitter,
    RandomZoomJitter,
    RandomLocalTexturePerturbation,
    RandomToneAndContrastJitter,
    RandomBorderMaskAndEdgeCrop,
    RandomWhiteBackground,
)


def _glyph_background_pattern_types() -> list[str]:
    return ["random_library_background"]


def _viz_prob(key: str, default: float = 0.0, *, branch: str = "tile", effective: bool = False) -> float:
    # Normalized one-of probabilities used in training. By default this is
    # conditional on the 80% apply gate firing, so values sum to 1.0 across
    # active candidates. Set effective=True to include the apply gate.
    tile_weights = {
        "jitter_prob": float(AUGMENT_COLOR_JITTER_PROB) if AUGMENT_COLOR_JITTER else 0.0,
        "gray_prob": float(AUGMENT_RANDOM_GRAYSCALE_PROB),
        "blur_strong_prob": float(AUGMENT_GAUSSIAN_BLUR_PROB),
        "background_prob": float(AUGMENT_BACKGROUND_RANDOM_LIBRARY_PROB),
        "parchment_stains_prob": float(AUGMENT_PARCHMENT_STAINS_PROB),
        "white_background_prob": float(AUGMENT_WHITE_BACKGROUND_PROB),
        "local_texture_prob": float(AUGMENT_LOCAL_TEXTURE_PROB),
        "tone_contrast_prob": float(AUGMENT_TONE_CONTRAST_PROB),
        "tilt_prob": float(AUGMENT_TILT_PROB),
        "border_crop_prob": float(AUGMENT_BORDER_CROP_PROB),
        "randaugment_prob": float(AUGMENT_RANDAUGMENT_PROB),
        "zoom_prob": float(AUGMENT_ZOOM_PROB),
        "resolution_prob_hidden": float(AUGMENT_RESOLUTION_JITTER_PROB),
    }
    glyph_weights = {
        "jitter_prob": float(AUGMENT_COLOR_JITTER_PROB) if AUGMENT_COLOR_JITTER else 0.0,
        "gray_prob": float(AUGMENT_RANDOM_GRAYSCALE_PROB),
        "blur_strong_prob": float(AUGMENT_GAUSSIAN_BLUR_PROB),
        "background_prob": float(AUGMENT_BACKGROUND_RANDOM_LIBRARY_PROB),
        "parchment_stains_prob": float(GLYPH_AUG_PARCHMENT_STAINS_PROB),
        "white_background_prob": float(GLYPH_AUG_WHITE_BACKGROUND_PROB),
        "local_texture_prob": float(AUGMENT_LOCAL_TEXTURE_PROB),
        "tone_contrast_prob": float(AUGMENT_TONE_CONTRAST_PROB),
        "tilt_prob": float(AUGMENT_TILT_PROB),
        "border_crop_prob": float(AUGMENT_BORDER_CROP_PROB),
        "randaugment_prob": float(AUGMENT_RANDAUGMENT_PROB) if bool(CHAR_PATCH_APPLY_RANDAUGMENT) else 0.0,
        "zoom_prob": float(AUGMENT_ZOOM_PROB),
    }
    weights = glyph_weights if branch == "glyph" else tile_weights
    total = sum(max(0.0, v) for v in weights.values())
    scale = float(GLYPH_AUG_APPLY_PROB if branch == "glyph" else AUGMENT_APPLY_PROB) if effective else 1.0
    probs = {
        k: (scale * max(0.0, v) / total) if total > 0 else 0.0
        for k, v in weights.items()
    }
    apply_prob = float(GLYPH_AUG_APPLY_PROB if branch == "glyph" else AUGMENT_APPLY_PROB)
    probs["no_aug_prob"] = max(0.0, 1.0 - apply_prob)
    probs["zoom_in_prob"] = probs["zoom_prob"]
    probs["zoom_out_prob"] = probs["zoom_prob"]
    return float(probs.get(key, default))


def _load_db_config(db_config_path: str) -> configparser.SectionProxy:
    cfg_path = db_config_path
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(PROJECT_ROOT, cfg_path)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"DB config not found: {cfg_path}")
    config = configparser.ConfigParser()
    config.read(cfg_path)
    if "postgresql" not in config:
        raise ValueError(f"Missing [postgresql] section in DB config: {cfg_path}")
    return config["postgresql"]


def _get_db_connection(db_config_path: str):
    db = _load_db_config(db_config_path)
    from psycopg2 import connect as pg_connect

    return pg_connect(
        host=db["host"],
        database=db["database"],
        user=db["user"],
        password=db["password"],
        port=db.get("port", 5432),
    )


def _denormalize(img_tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(NORMALIZE_MEAN, dtype=img_tensor.dtype, device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor(NORMALIZE_STD, dtype=img_tensor.dtype, device=img_tensor.device).view(3, 1, 1)
    return img_tensor * std + mean


def _build_tile_transform_variant(*, do_jitter: bool, do_grayscale: bool, do_blur: bool, background_pattern: str | None) -> transforms.Compose:
    ops: list[Any] = []
    if do_jitter and AUGMENT_COLOR_JITTER:
        ops.append(
            transforms.ColorJitter(
                brightness=AUGMENT_COLOR_JITTER_BRIGHTNESS,
                contrast=AUGMENT_COLOR_JITTER_CONTRAST,
                saturation=AUGMENT_COLOR_JITTER_SATURATION,
                hue=AUGMENT_COLOR_JITTER_HUE,
            )
        )
    if do_grayscale and AUGMENT_RANDOM_GRAYSCALE_PROB > 0:
        ops.append(transforms.RandomGrayscale(p=1.0))
    if do_blur and AUGMENT_GAUSSIAN_BLUR_PROB > 0:
        ops.append(transforms.GaussianBlur(kernel_size=AUGMENT_GAUSSIAN_BLUR_KERNEL))
    if background_pattern:
        ops.append(
            RandomLibraryBackgroundOverlay(
                p=1.0,
                pattern_types=[background_pattern],
                mask_profile="tile",
                background_sampling_alpha=AUGMENT_BACKGROUND_SAMPLING_ALPHA,
                background_sampling_floor=AUGMENT_BACKGROUND_SAMPLING_FLOOR,
            )
        )
    ops.append(transforms.Resize((TILE_SIZE, TILE_SIZE)))
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD))
    return transforms.Compose(ops)


def _build_tile_custom_variant(*, extra_ops: list[Any]) -> transforms.Compose:
    ops: list[Any] = [*extra_ops, transforms.Resize((TILE_SIZE, TILE_SIZE)), transforms.ToTensor(), transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)]
    return transforms.Compose(ops)


def _build_tile_white_background_variant() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((TILE_SIZE, TILE_SIZE)),
            RandomWhiteBackground(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )


def _edge_fill_color(image: Image.Image) -> tuple[int, int, int]:
    img = image.convert("RGB")
    w, h = img.size
    pts = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1), (w // 2, 0), (w // 2, h - 1), (0, h // 2), (w - 1, h // 2)]
    vals = [img.getpixel(p) for p in pts]
    r = int(sum(v[0] for v in vals) / len(vals))
    g = int(sum(v[1] for v in vals) / len(vals))
    b = int(sum(v[2] for v in vals) / len(vals))
    return (r, g, b)


def _apply_zoom(image: Image.Image, *, size: int, scale: float) -> Image.Image:
    img = image.convert("RGB")
    if scale > 1.0:
        # Zoom in: enlarge then center-crop.
        new_size = max(size, int(round(size * scale)))
        enlarged = img.resize((new_size, new_size), Image.Resampling.BICUBIC)
        left = max(0, (new_size - size) // 2)
        top = max(0, (new_size - size) // 2)
        return enlarged.crop((left, top, left + size, top + size))
    # Zoom out: shrink then paste centered on background color.
    new_size = max(4, int(round(size * scale)))
    shrunk = img.resize((new_size, new_size), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (size, size), _edge_fill_color(img))
    left = (size - new_size) // 2
    top = (size - new_size) // 2
    canvas.paste(shrunk, (left, top))
    return canvas


def _apply_tilt(image: Image.Image, *, size: int, angle: float) -> Image.Image:
    img = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    fill = _edge_fill_color(img)
    return TVF.rotate(img, angle=angle, interpolation=transforms.InterpolationMode.BILINEAR, fill=fill)


def _build_tile_geometric_variant(*, zoom_scale: float | None = None, tilt_angle: float | None = None) -> transforms.Compose:
    ops: list[Any] = [transforms.Resize((TILE_SIZE, TILE_SIZE))]
    if zoom_scale is not None:
        ops.append(transforms.Lambda(lambda im, s=zoom_scale: _apply_zoom(im, size=TILE_SIZE, scale=s)))
    if tilt_angle is not None:
        ops.append(transforms.Lambda(lambda im, a=tilt_angle: _apply_tilt(im, size=TILE_SIZE, angle=a)))
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD))
    return transforms.Compose(ops)


def _strong_blur_kernel() -> int:
    base = AUGMENT_GAUSSIAN_BLUR_KERNEL
    if isinstance(base, (tuple, list)):
        k = int(max(base))
    else:
        k = int(base)
    k = max(3, k)
    k = min(31, (k * 2) + 1)
    if k % 2 == 0:
        k += 1
    return k


def _very_strong_blur_kernel() -> int:
    k = _strong_blur_kernel()
    k = min(63, (k * 2) + 1)
    if k % 2 == 0:
        k += 1
    return k


def _build_glyph_transform_variant(*, do_randaugment: bool, do_jitter: bool, do_grayscale: bool, do_blur: bool, background_pattern: str | None, blur_kernel: int | None = None) -> transforms.Compose:
    ops: list[Any] = []
    if do_jitter and AUGMENT_COLOR_JITTER:
        ops.append(
            transforms.ColorJitter(
                brightness=AUGMENT_COLOR_JITTER_BRIGHTNESS,
                contrast=AUGMENT_COLOR_JITTER_CONTRAST,
                saturation=AUGMENT_COLOR_JITTER_SATURATION,
                hue=AUGMENT_COLOR_JITTER_HUE,
            )
        )
    if do_grayscale and AUGMENT_RANDOM_GRAYSCALE_PROB > 0:
        ops.append(transforms.RandomGrayscale(p=1.0))
    if do_blur and AUGMENT_GAUSSIAN_BLUR_PROB > 0:
        ops.append(transforms.GaussianBlur(kernel_size=(blur_kernel or AUGMENT_GAUSSIAN_BLUR_KERNEL)))
    if do_randaugment and bool(CHAR_PATCH_APPLY_RANDAUGMENT):
        ops.append(transforms.RandAugment(num_ops=2, magnitude=10))
    # For glyphs, resize before background compositing so edge replacement scale
    # is controlled at the final patch resolution.
    ops.append(transforms.Resize((CHAR_PATCH_SIZE, CHAR_PATCH_SIZE)))
    if background_pattern:
        ops.append(
            RandomLibraryBackgroundOverlay(
                p=1.0,
                pattern_types=[background_pattern],
                allow_support_backing=True,
                mask_profile="glyph",
                background_sampling_alpha=AUGMENT_BACKGROUND_SAMPLING_ALPHA,
                background_sampling_floor=AUGMENT_BACKGROUND_SAMPLING_FLOOR,
            )
        )
    ops.append(transforms.ToTensor())
    if bool(CHAR_PATCH_APPLY_IMAGENET_NORM):
        ops.append(transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD))
    return transforms.Compose(ops)


def _build_glyph_custom_variant(*, extra_ops: list[Any]) -> transforms.Compose:
    ops: list[Any] = [*extra_ops, transforms.Resize((CHAR_PATCH_SIZE, CHAR_PATCH_SIZE)), transforms.ToTensor()]
    if bool(CHAR_PATCH_APPLY_IMAGENET_NORM):
        ops.append(transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD))
    return transforms.Compose(ops)


def _build_glyph_white_background_variant() -> transforms.Compose:
    ops: list[Any] = [
        transforms.Resize((CHAR_PATCH_SIZE, CHAR_PATCH_SIZE)),
        RandomWhiteBackground(p=1.0),
        transforms.ToTensor(),
    ]
    if bool(CHAR_PATCH_APPLY_IMAGENET_NORM):
        ops.append(transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD))
    return transforms.Compose(ops)


def _build_glyph_geometric_variant(*, zoom_scale: float | None = None, tilt_angle: float | None = None) -> transforms.Compose:
    ops: list[Any] = [transforms.Resize((CHAR_PATCH_SIZE, CHAR_PATCH_SIZE))]
    if zoom_scale is not None:
        ops.append(transforms.Lambda(lambda im, s=zoom_scale: _apply_zoom(im, size=CHAR_PATCH_SIZE, scale=s)))
    if tilt_angle is not None:
        ops.append(transforms.Lambda(lambda im, a=tilt_angle: _apply_tilt(im, size=CHAR_PATCH_SIZE, angle=a)))
    ops.append(transforms.ToTensor())
    if bool(CHAR_PATCH_APPLY_IMAGENET_NORM):
        ops.append(transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD))
    return transforms.Compose(ops)


def _build_glyph_tilting_variant(*, max_degrees: float = 10.0) -> transforms.Compose:
    """
    Single visualization slot for rotation: one random angle in ``[-max_degrees, max_degrees]`` per apply.
    (Avoids listing left/right as two separate variants with doubled effective weight in the grid.)
    """
    ops: list[Any] = [transforms.Resize((CHAR_PATCH_SIZE, CHAR_PATCH_SIZE))]
    ops.append(
        transforms.Lambda(
            lambda im, m=max_degrees: _apply_tilt(
                im, size=CHAR_PATCH_SIZE, angle=float(random.uniform(-m, m))
            )
        )
    )
    ops.append(transforms.ToTensor())
    if bool(CHAR_PATCH_APPLY_IMAGENET_NORM):
        ops.append(transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD))
    return transforms.Compose(ops)


def _grid_plot(images: list[Image.Image], titles: list[str], rows: int, cols: int, suptitle: str, output_path: str) -> None:
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten()
    for ax, im, title in zip(axes, images, titles):
        ax.imshow(im)
        display_title = title.replace(" (p|aug=", "\n(p|aug=", 1)
        ax.set_title(display_title, fontsize=14, fontweight="bold")
        ax.axis("off")
    for ax in axes[len(images):]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=16, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _title_with_state(title: str, *, enabled: bool) -> str:
    return title if enabled else f"{title} (disabled)"


def _apply_fixed_board_background(image: Image.Image, *, background_filename: str) -> Image.Image:
    """Apply a specific background image using board-style fragment compositing."""
    bg_path = PROJECT_ROOT / "Backgrounds" / background_filename
    if not bg_path.exists():
        return image
    img = image.convert("RGB")
    overlay = RandomLibraryBackgroundOverlay(
        p=1.0,
        pattern_types=["random_library_background"],
        mask_profile="tile",
        background_sampling_alpha=AUGMENT_BACKGROUND_SAMPLING_ALPHA,
        background_sampling_floor=AUGMENT_BACKGROUND_SAMPLING_FLOOR,
    )
    asset = Image.open(bg_path).convert("RGB")
    # Keep source background fixed, but randomize placement/mask each run to
    # simulate naturally varying torn-page geometry.
    backdrop = overlay._make_support_backdrop(asset, img.size, "random_library_background")
    fragment_mask, _ = overlay._make_fragment_mask(img, "random_library_background")
    composited = Image.composite(img, backdrop, fragment_mask)
    composited = overlay._add_fragment_shadow(composited, backdrop, fragment_mask)
    return composited


def _select_geniza_row(conn, *, random_pick: bool) -> Tuple[Dict[str, Any], str]:
    exact_where = """
      num_visual_patches = %s
      AND num_glyphs = %s
      AND image_path IS NOT NULL
      AND xml_path IS NOT NULL
    """
    fallback_where = """
      num_visual_patches IS NOT NULL
      AND num_glyphs IS NOT NULL
      AND num_visual_patches > 0
      AND num_glyphs > 0
      AND image_path IS NOT NULL
      AND xml_path IS NOT NULL
    """
    candidates_sql = f"SELECT COUNT(*) FROM {GENIZA_IMAGE_LATENTS_TABLE} WHERE {{where}}"
    pick_sql = f"""
      SELECT id, manuscript_id, picture_id, page_number, page_id,
             image_path, xml_path, num_visual_patches, num_glyphs, num_words
      FROM {GENIZA_IMAGE_LATENTS_TABLE}
      WHERE {{where}}
      ORDER BY id
      LIMIT 1 OFFSET %s
    """
    with conn.cursor() as cur:
        cur.execute(candidates_sql.format(where=exact_where), (MAX_TILES_EVAL, MAX_CHARS_PER_IMAGE))
        n_exact = cur.fetchone()[0]
        if n_exact and n_exact > 0:
            mode = "exact"
            offset = int(torch.randint(low=0, high=n_exact, size=(1,)).item()) if random_pick else 0
            cur.execute(pick_sql.format(where=exact_where), (MAX_TILES_EVAL, MAX_CHARS_PER_IMAGE, offset))
            row = cur.fetchone()
            colnames = [d.name for d in cur.description]  # type: ignore[union-attr]
            return dict(zip(colnames, row)), mode
        cur.execute(candidates_sql.format(where=fallback_where))
        n_fb = cur.fetchone()[0]
        if not n_fb or n_fb <= 0:
            raise RuntimeError("No geniza_image_latents rows match fallback (>0 tiles & glyphs).")
        mode = "fallback"
        offset = int(torch.randint(low=0, high=n_fb, size=(1,)).item()) if random_pick else 0
        cur.execute(pick_sql.format(where=fallback_where), (offset,))
        row = cur.fetchone()
        colnames = [d.name for d in cur.description]  # type: ignore[union-attr]
        return dict(zip(colnames, row)), mode


def _pick_one_row(conn, *, base_sql: str, random_pick: bool) -> Dict[str, Any]:
    count_sql = f"SELECT COUNT(*) FROM ({base_sql}) q"
    pick_sql = (
        f"SELECT * FROM ({base_sql}) q "
        "ORDER BY manuscript_id, picture_id, page_number "
        "LIMIT 1 OFFSET %s"
    )
    with conn.cursor() as cur:
        cur.execute(count_sql)
        n_rows = cur.fetchone()[0]
        if not n_rows or n_rows <= 0:
            raise RuntimeError("No rows found for requested dataset source.")
        offset = int(torch.randint(low=0, high=n_rows, size=(1,)).item()) if random_pick else 0
        cur.execute(pick_sql, (offset,))
        row = cur.fetchone()
        colnames = [d.name for d in cur.description]  # type: ignore[union-attr]
    return dict(zip(colnames, row))


def _select_source_row(
    conn,
    *,
    dataset_source: str,
    random_pick: bool,
) -> Tuple[Dict[str, Any], str]:
    source = dataset_source.strip().lower()
    if source == "geniza-full":
        sql = f"""
            SELECT
                gii.manuscript_id,
                gii.picture_id,
                gii.page_number,
                gii.page_id,
                gii.image_path,
                gii.xml_path
            FROM {GENIZA_IMAGE_INFORMATION_TABLE} gii
            WHERE gii.image_path IS NOT NULL
              AND gii.xml_path IS NOT NULL
        """
        return _pick_one_row(conn, base_sql=sql, random_pick=random_pick), "geniza-full"
    if source == "geniza-test":
        sql = f"""
            SELECT
                gii.manuscript_id,
                gii.picture_id,
                gii.page_number,
                gii.page_id,
                gii.image_path,
                gii.xml_path
            FROM (
                SELECT manuscript_id1 AS manuscript_id, page_id1 AS picture_id FROM geniza_joins
                UNION
                SELECT manuscript_id2 AS manuscript_id, page_id2 AS picture_id FROM geniza_joins
            ) gj
            JOIN {GENIZA_IMAGE_INFORMATION_TABLE} gii
              ON gii.manuscript_id = gj.manuscript_id
             AND gii.picture_id = gj.picture_id
            WHERE gii.image_path IS NOT NULL
              AND gii.xml_path IS NOT NULL
        """
        return _pick_one_row(conn, base_sql=sql, random_pick=random_pick), "geniza-test"
    if source == "geniza-contrastive":
        sql = f"""
            SELECT
                gts.manuscript_id,
                gts.picture_id,
                gts.page_number,
                gts.image_path,
                gts.xml_path
            FROM {GENIZA_CONTRASTIVE_TABLE} gts
            WHERE gts.image_path IS NOT NULL
              AND gts.xml_path IS NOT NULL
        """
        return _pick_one_row(conn, base_sql=sql, random_pick=random_pick), "geniza-contrastive"
    if source in {"train", "val"}:
        sql = f"""
            SELECT
                pts.manuscript_id,
                pts.picture_id,
                pts.page_number,
                pts.page_id,
                pts.image_path,
                pts.xml_path
            FROM {PRETRAIN_TABLE_NAME} pts
            WHERE LOWER(pts.dataset_split) = %s
              AND pts.image_path IS NOT NULL
              AND pts.xml_path IS NOT NULL
        """
        with conn.cursor() as cur:
            count_sql = f"SELECT COUNT(*) FROM ({sql}) q"
            cur.execute(count_sql, (source,))
            n_rows = cur.fetchone()[0]
            if not n_rows or n_rows <= 0:
                raise RuntimeError(f"No rows found for dataset source '{source}'.")
            offset = int(torch.randint(low=0, high=n_rows, size=(1,)).item()) if random_pick else 0
            pick_sql = (
                f"SELECT * FROM ({sql}) q "
                "ORDER BY manuscript_id, picture_id, page_number "
                "LIMIT 1 OFFSET %s"
            )
            cur.execute(pick_sql, (source, offset))
            row = cur.fetchone()
            colnames = [d.name for d in cur.description]  # type: ignore[union-attr]
        return dict(zip(colnames, row)), source

    raise ValueError(f"Unsupported dataset source: {dataset_source}")


def _augmentation_values_payload() -> Dict[str, Any]:
    return {
        "tile": {
            "color_jitter_enabled": bool(AUGMENT_COLOR_JITTER),
            "apply_prob": float(AUGMENT_APPLY_PROB),
            "color_jitter_brightness": AUGMENT_COLOR_JITTER_BRIGHTNESS,
            "color_jitter_contrast": AUGMENT_COLOR_JITTER_CONTRAST,
            "color_jitter_saturation": AUGMENT_COLOR_JITTER_SATURATION,
            "color_jitter_hue": AUGMENT_COLOR_JITTER_HUE,
            "random_grayscale_prob": AUGMENT_RANDOM_GRAYSCALE_PROB,
            "gaussian_blur_prob": AUGMENT_GAUSSIAN_BLUR_PROB,
            "gaussian_blur_kernel": AUGMENT_GAUSSIAN_BLUR_KERNEL,
            "background_random_library_prob": float(AUGMENT_BACKGROUND_RANDOM_LIBRARY_PROB),
            "white_background_weight": float(AUGMENT_WHITE_BACKGROUND_PROB),
        },
        "glyph": {
            "apply_prob": float(GLYPH_AUG_APPLY_PROB),
            "char_patch_apply_randaugment": bool(CHAR_PATCH_APPLY_RANDAUGMENT),
            "char_patch_apply_imagenet_norm": bool(CHAR_PATCH_APPLY_IMAGENET_NORM),
            "background_pattern_types_weighted": _glyph_background_pattern_types(),
            "white_background_weight": float(GLYPH_AUG_WHITE_BACKGROUND_PROB),
        },
    }


def _extract_one_tile(image: Image.Image, *, image_path: str, xml_path: str) -> Image.Image:
    patches, _, _, _ = extract_patches_with_xml(
        image=image,
        image_path=image_path,
        patch_size=TILE_SIZE,
        stride=TILE_STRIDE,
        max_patches=MAX_TILES_EVAL,
        xml_path=xml_path,
    )
    if not patches:
        raise RuntimeError("Tile extraction returned zero patches.")
    tile = patches[0]
    return tile.resize((TILE_SIZE, TILE_SIZE)) if tile.size != (TILE_SIZE, TILE_SIZE) else tile


def _pick_glyph_index(pool_len: int, rng: Optional[random.Random]) -> int:
    if pool_len <= 0:
        raise ValueError("pool_len must be positive")
    return rng.randrange(pool_len) if rng is not None else random.randrange(pool_len)


def _extract_one_glyph(
    image: Image.Image,
    *,
    image_path: str,
    xml_path: str,
    rng: Optional[random.Random] = None,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Same pipeline as training (extract → hard filter → diverse sample), then pick **one**
    glyph uniformly at random from that pool so the debug grid is not always the first
    character in reading/alphabet order (often aleph).
    """
    glyph_patches, glyph_meta = extract_character_patches(
        image=image,
        image_path=image_path,
        char_patch_size=CHAR_PATCH_SIZE,
        max_chars=None,
        xml_path=xml_path,
    )
    if not glyph_patches:
        raise RuntimeError("Glyph extraction returned zero patches.")
    quality_filter = GlyphHardQualityFilter()
    patch_tensors = [transforms.ToTensor()(p) for p in glyph_patches]
    filtered_patches, filtered_meta = quality_filter.filter_glyphs(patch_tensors, glyph_meta)
    sampled_patches, sampled_meta = quality_filter.sample_diverse_glyphs(
        filtered_patches,
        filtered_meta,
        max_glyphs=MAX_CHARS_PER_IMAGE,
    )
    if sampled_patches:
        idx = _pick_glyph_index(len(sampled_patches), rng)
        glyph_tensor = sampled_patches[idx].detach().cpu().clamp(0.0, 1.0)
        meta = sampled_meta[idx] if idx < len(sampled_meta) else {}
        return transforms.ToPILImage()(glyph_tensor), {**dict(meta), "pick_pool": "sampled_diverse", "pick_index": idx}

    # Fallback for debug visualization: when hard filtering removes all glyphs,
    # still show one raw extracted glyph so the script can complete.
    if filtered_patches:
        idx = _pick_glyph_index(len(filtered_patches), rng)
        glyph_tensor = filtered_patches[idx].detach().cpu().clamp(0.0, 1.0)
        meta = filtered_meta[idx] if idx < len(filtered_meta) else {}
        return transforms.ToPILImage()(glyph_tensor), {**dict(meta), "pick_pool": "filtered_only", "pick_index": idx}
    if patch_tensors:
        idx = _pick_glyph_index(len(patch_tensors), rng)
        glyph_tensor = patch_tensors[idx].detach().cpu().clamp(0.0, 1.0)
        meta = glyph_meta[idx] if idx < len(glyph_meta) else {}
        return transforms.ToPILImage()(glyph_tensor), {**dict(meta), "pick_pool": "raw_extract", "pick_index": idx}

    raise RuntimeError("After filtering/sampling, no glyphs remain.")


def _to_pil_vis(t: torch.Tensor, *, normalized: bool) -> Image.Image:
    if normalized:
        t = _denormalize(t).clamp(0.0, 1.0)
    else:
        t = t.clamp(0.0, 1.0)
    return transforms.ToPILImage()(t)


def _visualize_augmentations_for_tile(tile: Image.Image, *, rows: int, cols: int, out_path: str) -> None:
    num_cells = rows * cols
    images: list[Image.Image] = [tile.resize((TILE_SIZE, TILE_SIZE))]
    titles: list[str] = ["original"]

    jitter_enabled = bool(AUGMENT_COLOR_JITTER)
    grayscale_enabled = AUGMENT_RANDOM_GRAYSCALE_PROB > 0
    blur_enabled = AUGMENT_GAUSSIAN_BLUR_PROB > 0

    forced_variants: list[tuple[str, transforms.Compose]] = []
    if _viz_prob("jitter_prob") > 0:
        forced_variants.append((
            _title_with_state(f"jitter (p|aug={_viz_prob('jitter_prob'):.2f})", enabled=jitter_enabled),
            _build_tile_transform_variant(do_jitter=True, do_grayscale=False, do_blur=False, background_pattern=None),
        ))
    if _viz_prob("gray_prob") > 0:
        forced_variants.append((
            _title_with_state(f"gray (p|aug={_viz_prob('gray_prob'):.2f})", enabled=grayscale_enabled),
            _build_tile_transform_variant(do_jitter=False, do_grayscale=True, do_blur=False, background_pattern=None),
        ))
    if _viz_prob("zoom_prob") > 0:
        forced_variants.append((f"zoom (p|aug={_viz_prob('zoom_prob'):.2f})", _build_tile_custom_variant(extra_ops=[RandomZoomJitter(scale_range=AUGMENT_ZOOM_SCALE_RANGE, p=1.0)])))
    if _viz_prob("blur_strong_prob") > 0:
        forced_variants.append((
            _title_with_state(f"blur (p|aug={_viz_prob('blur_strong_prob'):.2f})", enabled=blur_enabled),
            _build_tile_transform_variant(do_jitter=False, do_grayscale=False, do_blur=True, background_pattern=None),
        ))
    if _viz_prob("background_prob") > 0:
        forced_variants.append((f"random_library_background (p|aug={_viz_prob('background_prob'):.2f})", _build_tile_transform_variant(do_jitter=False, do_grayscale=False, do_blur=False, background_pattern="random_library_background")))
    if _viz_prob("parchment_stains_prob") > 0:
        forced_variants.append((f"parchment_stains (p|aug={_viz_prob('parchment_stains_prob'):.2f})", _build_tile_transform_variant(do_jitter=False, do_grayscale=False, do_blur=False, background_pattern="parchment_stains")))
    if _viz_prob("white_background_prob") > 0:
        forced_variants.append((f"white_background (p|aug={_viz_prob('white_background_prob'):.2f})", _build_tile_white_background_variant()))
    if _viz_prob("local_texture_prob") > 0:
        forced_variants.append((
            f"local_texture (p|aug={_viz_prob('local_texture_prob'):.2f})",
            _build_tile_custom_variant(extra_ops=[RandomLocalTexturePerturbation(p=1.0, alpha_range=AUGMENT_LOCAL_TEXTURE_ALPHA_RANGE)]),
        ))
    if _viz_prob("tone_contrast_prob") > 0:
        forced_variants.append((
            f"tone_contrast_shift (p|aug={_viz_prob('tone_contrast_prob'):.2f})",
            _build_tile_custom_variant(extra_ops=[RandomToneAndContrastJitter(p=1.0)]),
        ))
    if _viz_prob("tilt_prob") > 0:
        forced_variants.append((f"tilting (p|aug={_viz_prob('tilt_prob'):.2f})", _build_tile_geometric_variant(tilt_angle=10.0)))
    if _viz_prob("border_crop_prob") > 0:
        forced_variants.append((
            f"border_edge_crop (p|aug={_viz_prob('border_crop_prob'):.2f})",
            _build_tile_custom_variant(
                extra_ops=[
                    RandomBorderMaskAndEdgeCrop(
                        p=1.0,
                        max_border_frac=AUGMENT_BORDER_MAX_FRAC,
                        max_crop_frac=AUGMENT_EDGE_CROP_MAX_FRAC,
                    )
                ]
            ),
        ))
    if _viz_prob("resolution_prob_hidden") > 0:
        forced_variants.append((f"resolution_jitter (p|aug={_viz_prob('resolution_prob_hidden'):.2f})", _build_tile_custom_variant(extra_ops=[RandomResolutionJitter(scale_range=(0.5, 0.8), p=1.0)])))

    for name, tfm in forced_variants[: max(0, num_cells - 1)]:
        images.append(_to_pil_vis(tfm(tile), normalized=True))
        titles.append(name)

    _grid_plot(images, titles, rows, cols, suptitle="Tile augmentations", output_path=out_path)


def _visualize_augmentations_for_glyph(
    glyph: Image.Image,
    *,
    rows: int,
    cols: int,
    out_path: str,
    suptitle: str = "Glyph augmentations",
) -> None:
    num_cells = rows * cols
    images: list[Image.Image] = [glyph.resize((CHAR_PATCH_SIZE, CHAR_PATCH_SIZE))]
    titles: list[str] = ["original"]

    randaugment_enabled = bool(CHAR_PATCH_APPLY_RANDAUGMENT)
    jitter_enabled = bool(AUGMENT_COLOR_JITTER)
    grayscale_enabled = AUGMENT_RANDOM_GRAYSCALE_PROB > 0
    blur_enabled = AUGMENT_GAUSSIAN_BLUR_PROB > 0

    def gp(key: str) -> float:
        return _viz_prob(key, branch="glyph")

    forced_variants: list[tuple[str, transforms.Compose]] = []
    if gp("randaugment_prob") > 0:
        forced_variants.append((
            _title_with_state(f"randaug (p|aug={gp('randaugment_prob'):.2f})", enabled=randaugment_enabled),
            _build_glyph_transform_variant(do_randaugment=True, do_jitter=False, do_grayscale=False, do_blur=False, background_pattern=None),
        ))
    if gp("jitter_prob") > 0:
        forced_variants.append((
            _title_with_state(f"jitter (p|aug={gp('jitter_prob'):.2f})", enabled=jitter_enabled),
            _build_glyph_transform_variant(do_randaugment=False, do_jitter=True, do_grayscale=False, do_blur=False, background_pattern=None),
        ))
    if gp("gray_prob") > 0:
        forced_variants.append((
            _title_with_state(f"gray (p|aug={gp('gray_prob'):.2f})", enabled=grayscale_enabled),
            _build_glyph_transform_variant(do_randaugment=False, do_jitter=False, do_grayscale=True, do_blur=False, background_pattern=None),
        ))
    if gp("zoom_prob") > 0:
        forced_variants.append((f"zoom (p|aug={gp('zoom_prob'):.2f})", _build_glyph_custom_variant(extra_ops=[RandomZoomJitter(scale_range=AUGMENT_ZOOM_SCALE_RANGE, p=1.0)])))
    if gp("tilt_prob") > 0:
        forced_variants.append((f"tilting (p|aug={gp('tilt_prob'):.2f})", _build_glyph_tilting_variant(max_degrees=10.0)))
    if gp("blur_strong_prob") > 0:
        forced_variants.append((
            _title_with_state(f"blur (p|aug={gp('blur_strong_prob'):.2f})", enabled=blur_enabled),
            _build_glyph_transform_variant(
                do_randaugment=False,
                do_jitter=False,
                do_grayscale=False,
                do_blur=True,
                background_pattern=None,
                blur_kernel=GLYPH_GAUSSIAN_BLUR_KERNEL,
            ),
        ))
    if gp("local_texture_prob") > 0:
        forced_variants.append((
            f"local_texture (p|aug={gp('local_texture_prob'):.2f})",
            _build_glyph_custom_variant(extra_ops=[RandomLocalTexturePerturbation(p=1.0, alpha_range=AUGMENT_LOCAL_TEXTURE_ALPHA_RANGE)]),
        ))
    if gp("tone_contrast_prob") > 0:
        forced_variants.append((
            f"tone_contrast_shift (p|aug={gp('tone_contrast_prob'):.2f})",
            _build_glyph_custom_variant(extra_ops=[RandomToneAndContrastJitter(p=1.0)]),
        ))
    if gp("border_crop_prob") > 0:
        forced_variants.append((
            f"border_edge_crop (p|aug={gp('border_crop_prob'):.2f})",
            _build_glyph_custom_variant(
                extra_ops=[
                    RandomBorderMaskAndEdgeCrop(
                        p=1.0,
                        max_border_frac=min(0.08, float(AUGMENT_BORDER_MAX_FRAC)),
                        max_crop_frac=min(0.1, float(AUGMENT_EDGE_CROP_MAX_FRAC)),
                    )
                ]
            ),
        ))
    if gp("background_prob") > 0:
        forced_variants.append(
            (
                f"random_library_background (p|aug={gp('background_prob'):.2f})",
                _build_glyph_transform_variant(do_randaugment=False, do_jitter=False, do_grayscale=False, do_blur=False, background_pattern="random_library_background"),
            )
        )
    if gp("parchment_stains_prob") > 0:
        forced_variants.append(
            (
                f"parchment_stains (p|aug={gp('parchment_stains_prob'):.2f})",
                _build_glyph_transform_variant(do_randaugment=False, do_jitter=False, do_grayscale=False, do_blur=False, background_pattern="parchment_stains"),
            )
        )
    if gp("white_background_prob") > 0:
        forced_variants.append(
            (
                f"white_background (p|aug={gp('white_background_prob'):.2f})",
                _build_glyph_white_background_variant(),
            )
        )

    for name, tfm in forced_variants[: max(0, num_cells - 1)]:
        images.append(_to_pil_vis(tfm(glyph), normalized=bool(CHAR_PATCH_APPLY_IMAGENET_NORM)))
        titles.append(name)

    _grid_plot(images, titles, rows, cols, suptitle=suptitle, output_path=out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db-config", type=str, default=DB_CONFIG_PATH)
    ap.add_argument("--out-dir", type=str, default=str(Path(__file__).resolve().parent / "outputs" / "sample_geniza_fragment_and_visualize_augmentations"))
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--random", action="store_true")
    ap.add_argument(
        "--glyph-seed",
        type=int,
        default=None,
        help="Optional RNG seed for which glyph is picked among the sampled pool (default: nondeterministic).",
    )
    ap.add_argument(
        "--dataset-source",
        choices=["geniza-full", "geniza-test", "geniza-contrastive", "train", "val"],
        default="geniza-full",
        help=(
            "Source dataset for selecting the sampled image: "
            "geniza-full (geniza_image_information), "
            "geniza-test (geniza_joins), "
            "geniza-contrastive (geniza_train_set), "
            "train/val (pretrain_finetune_oriental_non_oriental_train_val_test_split)."
        ),
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if DISABLE_PIL_LIMIT:
        Image.MAX_IMAGE_PIXELS = None

    conn = _get_db_connection(args.db_config)
    try:
        if args.dataset_source == "geniza-full":
            row, mode = _select_source_row(conn, dataset_source=args.dataset_source, random_pick=args.random)
            # Keep the original exact/fallback flow for geniza-full when possible.
            try:
                row_exact, mode_exact = _select_geniza_row(conn, random_pick=args.random)
                row, mode = row_exact, f"geniza-full-{mode_exact}"
            except Exception:
                pass
        else:
            row, mode = _select_source_row(conn, dataset_source=args.dataset_source, random_pick=args.random)
    finally:
        conn.close()

    manuscript_name = str(row.get("manuscript_id") or "unknown")
    manuscript_out_dir = out_dir / manuscript_name
    manuscript_out_dir.mkdir(parents=True, exist_ok=True)
    tile_out = str(manuscript_out_dir / "tile_aug_grid.png")
    glyph_out = str(manuscript_out_dir / "glyph_aug_grid.png")
    meta_out = str(manuscript_out_dir / "selected_geniza_row.json")
    meta_payload = {
        "dataset_source": args.dataset_source,
        "selection_mode": mode,
        "manuscript_name": manuscript_name,
        "selected_row": row,
        "glyph_seed": args.glyph_seed,
        "augmentation_values": {
            manuscript_name: _augmentation_values_payload(),
        },
    }

    print(
        f"Selected row: source={args.dataset_source} mode={mode} id={row.get('id')} "
        f"ms={manuscript_name} picture={row.get('picture_id')}"
    )
    image_path = str(row["image_path"])
    xml_path = str(row["xml_path"])
    if not os.path.exists(image_path):
        raise RuntimeError(f"image_path not found: {image_path}")
    if not os.path.exists(xml_path):
        raise RuntimeError(f"xml_path not found: {xml_path}")

    img = Image.open(image_path).convert("RGB")
    tile = _extract_one_tile(img, image_path=image_path, xml_path=xml_path)
    glyph_rng = random.Random(int(args.glyph_seed)) if args.glyph_seed is not None else None
    glyph, glyph_pick = _extract_one_glyph(img, image_path=image_path, xml_path=xml_path, rng=glyph_rng)
    meta_payload["selected_glyph"] = glyph_pick
    print(
        f"Glyph for aug grid: char={glyph_pick.get('char')!r} "
        f"pool={glyph_pick.get('pick_pool')} index={glyph_pick.get('pick_index')}"
    )

    ch = str(glyph_pick.get("char") or "").strip()
    glyph_suptitle = "Glyph augmentations"
    parts = [f'pool={glyph_pick.get("pick_pool")}', f'idx={glyph_pick.get("pick_index")}']
    if ch:
        parts.insert(0, f"char={ch!r}")
    glyph_suptitle = f"{glyph_suptitle} ({', '.join(parts)})"

    _visualize_augmentations_for_tile(tile, rows=args.rows, cols=args.cols, out_path=tile_out)
    _visualize_augmentations_for_glyph(
        glyph, rows=args.rows, cols=args.cols, out_path=glyph_out, suptitle=glyph_suptitle
    )

    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
