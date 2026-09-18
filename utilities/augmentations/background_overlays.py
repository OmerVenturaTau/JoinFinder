from __future__ import annotations

import math
import random
import re
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps, ImageStat


class RandomLibraryBackgroundOverlay:
    """
    Replace border regions with realistic library-colored backing.

    The goal is to mimic photographed fragments on library backings:
    backing shows up at torn edges, corners, or padded margins, not as a soft
    synthetic blob over the middle of the text.
    """

    _asset_cache: list[dict[str, object]] | None = None
    _pattern_aliases = {
        "library_blue_board": "library_blue_board",
        "library_blue_grid": "library_blue_grid",
        "library_neutral_mount": "library_neutral_mount",
        "library_warm_mount": "library_warm_mount",
        "parchment_stains": "parchment_stains",
        "padding_edge_artifacts": "padding_edge_artifacts",
        "blue_wash": "blue_board",
        "blue_mottle": "blue_board",
        "edge_tint": "neutral_mount",
        "blue_board": "library_blue_board",
        "blue_grid": "library_blue_grid",
        "neutral_mount": "library_neutral_mount",
        "warm_mount": "library_warm_mount",
        "paper_stains": "parchment_stains",
        "random_library_background": "random_library_background",
    }
    _support_patterns = {"library_blue_board", "library_blue_grid", "library_neutral_mount", "library_warm_mount", "random_library_background"}
    _surface_patterns = {"parchment_stains", "padding_edge_artifacts"}
    _display_names = {
        "library_blue_board": "library_blue_board",
        "library_blue_grid": "library_blue_grid",
        "library_neutral_mount": "library_neutral_mount",
        "library_warm_mount": "library_warm_mount",
        "parchment_stains": "parchment_stains",
        "padding_edge_artifacts": "padding_edge_artifacts",
        "random_library_background": "random_library_background",
    }

    def __init__(
        self,
        p: float = 0.0,
        pattern_types: Sequence[str] = (
            "library_blue_board",
            "library_blue_grid",
            "library_neutral_mount",
            "library_warm_mount",
            "parchment_stains",
            "padding_edge_artifacts",
        ),
        alpha_range: tuple[float, float] = (0.08, 0.28),
        grid_spacing_range: tuple[int, int] = (8, 22),
        grid_line_width: int = 1,
        blur_radius_range: tuple[float, float] = (0.5, 2.5),
        allow_support_backing: bool = True,
        mask_profile: str = "auto",
        background_sampling_alpha: float = 0.72,
        background_sampling_floor: float = 80.0,
    ) -> None:
        self.p = float(p)
        self.pattern_types = tuple(pattern_types)
        self.alpha_range = alpha_range
        self.grid_spacing_range = grid_spacing_range
        self.grid_line_width = int(grid_line_width)
        self.blur_radius_range = blur_radius_range
        self.allow_support_backing = bool(allow_support_backing)
        self.mask_profile = str(mask_profile or "auto").strip().lower()
        self.background_sampling_alpha = float(background_sampling_alpha)
        self.background_sampling_floor = float(background_sampling_floor)

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.p <= 0 or random.random() >= self.p:
            return image
        pattern_pool = self._eligible_pattern_types()
        if not pattern_pool:
            return image

        img = image.convert("RGB")
        pattern_name = self._canonical_pattern(random.choice(pattern_pool))
        if pattern_name == "padding_edge_artifacts":
            return self._apply_padding_edge_artifacts(img)
        if pattern_name in self._surface_patterns:
            asset = self._pick_asset(pattern_name)
            return self._apply_surface_aging(img=img, asset=asset, pattern_name=pattern_name)

        asset = self._pick_asset(pattern_name)
        if asset is None and pattern_name in self._support_patterns:
            asset = self._make_flat_support_asset(pattern_name, img.size)
        if asset is None:
            overlay = self._make_legacy_overlay(pattern_name, img.size)
            alpha = random.uniform(*self.alpha_range)
            return Image.blend(img, overlay, alpha=max(0.0, min(1.0, alpha)))
        return self._make_fragment_backing_overlay(img=img, asset=asset, pattern_name=pattern_name)

    def _eligible_pattern_types(self) -> tuple[str, ...]:
        patterns = tuple(self._canonical_pattern(p) for p in self.pattern_types)
        if self.allow_support_backing:
            return patterns
        return tuple(p for p in patterns if p in self._surface_patterns)

    def _canonical_pattern(self, pattern_name: str) -> str:
        return self._pattern_aliases.get(pattern_name, pattern_name)

    @classmethod
    def canonical_pattern_name(cls, pattern_name: str) -> str:
        return cls._pattern_aliases.get(pattern_name, pattern_name)

    @classmethod
    def display_pattern_name(cls, pattern_name: str) -> str:
        canonical = cls.canonical_pattern_name(pattern_name)
        return cls._display_names.get(canonical, canonical)

    @classmethod
    def _background_dir(cls) -> Path:
        return Path(__file__).resolve().parents[2] / "Backgrounds"

    @classmethod
    def _asset_family(cls, mean_rgb: tuple[int, int, int]) -> str:
        r, g, b = mean_rgb
        if b > r + 20 and b > g + 10:
            return "blue"
        if r >= b and g >= b:
            return "warm"
        return "neutral"

    @classmethod
    def _load_assets(cls) -> list[dict[str, object]]:
        if cls._asset_cache is not None:
            return cls._asset_cache

        assets: list[dict[str, object]] = []
        for path in sorted(cls._background_dir().glob("*.png")):
            try:
                image = Image.open(path).convert("RGB")
            except Exception:
                continue
            mean = image.resize((1, 1), Image.Resampling.BOX).getpixel((0, 0))
            assets.append(
                {
                    "image": image,
                    "mean": mean,
                    "family": cls._asset_family(mean),
                    "name": path.name,
                }
            )
        cls._asset_cache = assets
        return assets

    def _target_family(self, pattern_name: str) -> tuple[str, ...]:
        if pattern_name in {"library_blue_board", "library_blue_grid"}:
            return ("blue",)
        if pattern_name == "library_warm_mount":
            return ("warm",)
        if pattern_name == "library_neutral_mount":
            return ("neutral", "warm")
        if pattern_name in {"parchment_stains", "padding_edge_artifacts"}:
            return ("warm", "neutral")
        return ("neutral", "warm")

    def _pick_asset(self, pattern_name: str) -> Image.Image | None:
        assets = self._load_assets()
        if not assets:
            return None
        if pattern_name == "random_library_background":
            chosen = self._pick_weighted_library_asset(assets)
            return Image.Image.copy(chosen["image"])  # type: ignore[arg-type]
        families = self._target_family(pattern_name)
        filtered = [a for a in assets if a["family"] in families]
        pool = filtered or assets
        chosen = random.choice(pool)
        return Image.Image.copy(chosen["image"])  # type: ignore[arg-type]

    @staticmethod
    def _normalize_library_name(name: str) -> str:
        cleaned = str(name or "").strip()
        cleaned = cleaned.removesuffix(".png").strip()
        cleaned = re.sub(r"\s*\(\d+\)$", "", cleaned).strip()
        cleaned = re.sub(r"(\d+)$", "", cleaned).strip()
        return cleaned

    def _load_library_sampling_weights(self, assets: list[dict[str, object]]) -> list[float]:
        xlsx_path = self._background_dir() / "background_appearances.xlsx"
        base_names = []
        for asset in assets:
            filename = str(asset.get("name", ""))
            stem = Path(filename).stem
            base_names.append(self._normalize_library_name(stem))
        variant_count_by_library: dict[str, int] = {}
        for lib in base_names:
            variant_count_by_library[lib] = variant_count_by_library.get(lib, 0) + 1
        available = set(base_names)
        counts: dict[str, float] = {}
        if xlsx_path.exists():
            try:
                import pandas as pd
                df = pd.read_excel(xlsx_path)
                required = {"normalized_library", "manuscripts_with_images_count"}
                if required.issubset(set(df.columns)):
                    for _, row in df.iterrows():
                        lib = self._normalize_library_name(str(row["normalized_library"]))
                        if lib in available:
                            try:
                                counts[lib] = float(row["manuscripts_with_images_count"])
                            except Exception:
                                continue
            except Exception:
                counts = {}

        floor = max(0.0, self.background_sampling_floor)
        alpha = max(0.05, self.background_sampling_alpha)
        weights = []
        for lib in base_names:
            c = float(counts.get(lib, 0.0))
            # Split a library's probability mass across its filename variants
            # so "...", "...2", "...3" share one library-level weight.
            variants = max(1, int(variant_count_by_library.get(lib, 1)))
            w = ((c + floor) ** alpha) / float(variants)
            weights.append(max(1e-8, w))
        return weights

    def _pick_weighted_library_asset(self, assets: list[dict[str, object]]) -> dict[str, object]:
        weights = self._load_library_sampling_weights(assets)
        total = sum(weights)
        if total <= 0:
            return random.choice(assets)
        return random.choices(assets, weights=weights, k=1)[0]

    def _make_fragment_backing_overlay(self, *, img: Image.Image, asset: Image.Image, pattern_name: str) -> Image.Image:
        size = img.size
        backdrop = self._make_support_backdrop(asset, size, pattern_name)
        fragment_mask, _inferred_edge_background = self._make_fragment_mask(img, pattern_name)

        composited = Image.composite(img, backdrop, fragment_mask)
        composited = self._add_fragment_shadow(composited, backdrop, fragment_mask)
        return composited

    def _make_support_backdrop(self, asset: Image.Image, size: tuple[int, int], pattern_name: str) -> Image.Image:
        w, h = size
        texture = asset.copy()
        if random.random() < 0.5:
            texture = ImageOps.mirror(texture)
        if random.random() < 0.5:
            texture = ImageOps.flip(texture)

        # Preserve real background texture details at target resolution:
        # fit/crop the source asset to output size first, then apply subtle
        # support modeling (noise/seams/fading), instead of aggressively
        # downsampling and re-upsampling.
        src_w, src_h = texture.size
        if src_w > 0 and src_h > 0:
            if src_w >= w and src_h >= h:
                # Prefer random crop at native resolution (no zoom-in).
                left = random.randint(0, src_w - w) if src_w > w else 0
                top = random.randint(0, src_h - h) if src_h > h else 0
                texture = texture.crop((left, top, left + w, top + h))
            else:
                # Avoid enlarging small assets; tile them to fill instead.
                tiled = Image.new("RGB", (w, h))
                for x in range(0, w, src_w):
                    for y in range(0, h, src_h):
                        tiled.paste(texture, (x, y))
                texture = tiled
        else:
            texture = texture.resize(size, Image.Resampling.BICUBIC)

        stats = ImageStat.Stat(texture)
        mean = tuple(int(v) for v in stats.mean[:3])
        backdrop = Image.new("RGB", size, mean)
        backdrop = Image.blend(backdrop, texture, alpha=0.55)
        backdrop = self._add_support_noise(backdrop, mean)

        if pattern_name == "library_blue_grid":
            backdrop = self._add_backing_seams(backdrop, mean, blue_bias=True)
        elif pattern_name == "library_neutral_mount":
            backdrop = self._add_edge_vignette(backdrop, mean, heavier=True)
            backdrop = self._add_mount_shading(backdrop, mean, warm=False)
        elif pattern_name == "library_warm_mount":
            backdrop = self._add_edge_vignette(backdrop, mean, heavier=True)
            backdrop = self._add_mount_shading(backdrop, mean, warm=True)
        else:
            backdrop = self._add_support_fading(backdrop, mean, intensity=0.35)

        return backdrop.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.08, 0.35)))

    def _apply_surface_aging(self, *, img: Image.Image, asset: Image.Image | None, pattern_name: str) -> Image.Image:
        aged = img.copy()
        small_patch = min(img.size) <= 128
        mean = tuple(int(v) for v in ImageStat.Stat(aged).mean[:3])
        if asset is not None and not (small_patch and pattern_name == "parchment_stains"):
            stain_texture = asset.copy().resize(img.size, Image.Resampling.BICUBIC)
            stain_texture = stain_texture.convert("RGB")
            if pattern_name == "parchment_stains":
                texture_alpha = 0.12 if small_patch else 0.30
                apply_alpha = 0.07 if small_patch else 0.24
            else:
                texture_alpha = 0.18
                apply_alpha = 0.14
            stain_texture = Image.blend(Image.new("RGB", img.size, mean), stain_texture, texture_alpha)
            aged = Image.blend(aged, stain_texture, apply_alpha)
        aged = self._add_edge_aging(aged, stronger=pattern_name == "parchment_stains")
        aged = self._add_foxing_and_grime(aged, mean, small_patch=small_patch)
        aged = self._add_stain_blooms(aged, mean, small_patch=small_patch)
        if pattern_name == "parchment_stains":
            # Keep stains visible on tiles without softening text too much.
            final_blur = random.uniform(0.02, 0.08) if small_patch else random.uniform(0.03, 0.12)
        else:
            final_blur = random.uniform(0.02, 0.12) if small_patch else random.uniform(0.1, 0.28)
        return aged.filter(ImageFilter.GaussianBlur(radius=final_blur))

    def _apply_padding_edge_artifacts(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        out = img.copy()
        arr = np.asarray(out, dtype=np.int16)
        border = np.concatenate([arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]], axis=0)
        border_color = tuple(int(v) for v in np.median(border, axis=0))

        draw = ImageDraw.Draw(out, "RGBA")
        max_band = max(1, min(w, h) // 16)
        for side in ("left", "right", "top", "bottom"):
            if random.random() < 0.75:
                band = random.randint(1, max_band)
                alpha = random.randint(24, 54)
                tint = (
                    max(0, min(255, border_color[0] + random.randint(-12, 12))),
                    max(0, min(255, border_color[1] + random.randint(-12, 12))),
                    max(0, min(255, border_color[2] + random.randint(-12, 12))),
                    alpha,
                )
                if side == "left":
                    draw.rectangle([0, 0, band, h], fill=tint)
                elif side == "right":
                    draw.rectangle([w - band, 0, w, h], fill=tint)
                elif side == "top":
                    draw.rectangle([0, 0, w, band], fill=tint)
                else:
                    draw.rectangle([0, h - band, w, h], fill=tint)

        for _ in range(random.randint(1, 3)):
            side = random.choice(("left", "right", "top", "bottom"))
            length = random.randint(max(8, min(w, h) // 8), max(12, min(w, h) // 2))
            thickness = random.randint(1, max(2, min(w, h) // 20))
            color = (35, 28, 18, random.randint(20, 54))
            if side in {"left", "right"}:
                y0 = random.randint(0, max(0, h - length))
                x = random.randint(0, max_band) if side == "left" else random.randint(max(0, w - max_band), w - 1)
                draw.rectangle([x, y0, min(w, x + thickness), y0 + length], fill=color)
            else:
                x0 = random.randint(0, max(0, w - length))
                y = random.randint(0, max_band) if side == "top" else random.randint(max(0, h - max_band), h - 1)
                draw.rectangle([x0, y, x0 + length, min(h, y + thickness)], fill=color)

        out = out.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.05, 0.22)))
        out = self._add_scan_noise(out, edge_only=True)
        return out

    def _make_flat_support_asset(self, pattern_name: str, size: tuple[int, int]) -> Image.Image:
        if pattern_name in {"library_blue_board", "library_blue_grid"}:
            base = (
                random.randint(18, 42),
                random.randint(78, 110),
                random.randint(135, 175),
            )
        elif pattern_name == "library_warm_mount":
            base = (
                random.randint(185, 214),
                random.randint(174, 204),
                random.randint(150, 188),
            )
        else:
            base = (
                random.randint(180, 210),
                random.randint(180, 210),
                random.randint(175, 205),
            )
        return Image.new("RGB", size, base)

    def _make_fragment_mask(self, img: Image.Image, pattern_name: str) -> tuple[Image.Image, bool]:
        w, h = img.size
        edge_mask = self._make_edge_fragment_mask((w, h), pattern_name)
        inferred_background = self._detect_connected_edge_background(img)
        if inferred_background is None:
            return edge_mask, False
        return ImageChops.multiply(edge_mask, inferred_background), True

    def _make_edge_fragment_mask(self, size: tuple[int, int], pattern_name: str) -> Image.Image:
        w, h = size
        mask = Image.new("L", size, 255)
        draw = ImageDraw.Draw(mask)
        min_side = min(w, h)
        if self.mask_profile == "glyph":
            is_glyph_scale = True
            is_tile_scale = False
        elif self.mask_profile == "tile":
            is_glyph_scale = False
            is_tile_scale = True
        else:
            is_glyph_scale = min_side <= 96
            is_tile_scale = 96 < min_side <= 160

        def _draw_glyph_edge_chip(edge_name: str) -> None:
            """Small jagged chip near an edge/corner (non-circular tear)."""
            if edge_name in {"left", "right"}:
                depth = random.randint(1, max(1, w // 24))
                span = random.randint(max(2, h // 18), max(4, h // 8))
                y0 = random.randint(0, max(0, h - span))
                # Bias chips toward corners to mimic torn parchment corners.
                if random.random() < 0.7:
                    corner_band = max(1, h // 6)
                    y0 = random.randint(0, corner_band) if random.random() < 0.5 else random.randint(max(0, h - span - corner_band), max(0, h - span))
                n = random.randint(3, 4)
                pts: list[tuple[float, float]] = []
                if edge_name == "left":
                    pts.append((0, y0))
                    for i in range(n):
                        yy = y0 + (span * (i + 1) / (n + 1))
                        xx = depth * random.uniform(0.35, 1.0)
                        pts.append((xx, yy))
                    pts.append((0, y0 + span))
                else:
                    pts.append((w, y0))
                    for i in range(n):
                        yy = y0 + (span * (i + 1) / (n + 1))
                        xx = w - depth * random.uniform(0.35, 1.0)
                        pts.append((xx, yy))
                    pts.append((w, y0 + span))
                draw.polygon(pts, fill=0)
            else:
                depth = random.randint(1, max(1, h // 24))
                span = random.randint(max(2, w // 18), max(4, w // 8))
                x0 = random.randint(0, max(0, w - span))
                if random.random() < 0.7:
                    corner_band = max(1, w // 6)
                    x0 = random.randint(0, corner_band) if random.random() < 0.5 else random.randint(max(0, w - span - corner_band), max(0, w - span))
                n = random.randint(3, 4)
                pts = []
                if edge_name == "top":
                    pts.append((x0, 0))
                    for i in range(n):
                        xx = x0 + (span * (i + 1) / (n + 1))
                        yy = depth * random.uniform(0.35, 1.0)
                        pts.append((xx, yy))
                    pts.append((x0 + span, 0))
                else:
                    pts.append((x0, h))
                    for i in range(n):
                        xx = x0 + (span * (i + 1) / (n + 1))
                        yy = h - depth * random.uniform(0.35, 1.0)
                        pts.append((xx, yy))
                    pts.append((x0 + span, h))
                draw.polygon(pts, fill=0)

        major_count = {
            "library_blue_board": (1, 3),
            "library_blue_grid": (1, 3),
            "library_neutral_mount": (1, 2),
            "library_warm_mount": (1, 2),
            "parchment_stains": (2, 4),
            # Slightly stronger edge loss for sampled real library backgrounds.
            "random_library_background": (3, 5),
        }.get(pattern_name, (2, 4))
        if is_glyph_scale and pattern_name in {"library_blue_board", "library_blue_grid"}:
            # Need a visibly torn fragment shape on glyph patches; the previous
            # (0, 1) made the augmentation invisible on its own and forced the
            # connected-edge detector to do all the work.
            major_count = (2, 4)
        if is_tile_scale and pattern_name in {"library_blue_board", "library_blue_grid"}:
            # Previously (4, 8) produced backgrounds that consumed roughly
            # half of the tile area. Halve it so the parchment dominates.
            major_count = (2, 4)
        n_bites = random.randint(*major_count)
        use_connected_tears = pattern_name == "random_library_background"
        connected_edge: str | None = None
        connected_anchor: int | None = None
        if use_connected_tears:
            connected_edge = random.choice(("left", "right", "top", "bottom"))

        for _ in range(n_bites):
            if not use_connected_tears:
                edge = random.choice(("left", "right", "top", "bottom"))
            else:
                # Keep most bites on one edge (or an adjacent one) so removed
                # background area is contiguous like a real torn fragment.
                assert connected_edge is not None
                if random.random() < 0.72:
                    edge = connected_edge
                elif connected_edge in {"left", "right"}:
                    edge = random.choice(("top", "bottom"))
                else:
                    edge = random.choice(("left", "right"))
            if edge in {"left", "right"}:
                if is_glyph_scale and pattern_name in {"library_blue_board", "library_blue_grid"}:
                    # Previously (w/28, w/12) was barely visible on small
                    # patches; widen so the torn shape actually reads.
                    depth = random.randint(max(3, w // 18), max(9, w // 6))
                elif is_tile_scale:
                    # Previously max bite depth was w/3 (~186px on TILE_SIZE=560);
                    # cap at w/7 so each bite eats ~80px instead of ~190px.
                    # For random_library_background, nudge depth a bit so more
                    # visible backing appears (user-requested "bit more eaten").
                    depth_max_div = 6 if pattern_name == "random_library_background" else 7
                    depth = random.randint(max(6, w // 18), max(14, w // depth_max_div))
                else:
                    depth = random.randint(max(7, w // 14), max(16, w // 7))
                if is_tile_scale:
                    span = random.randint(max(8, h // 6), max(14, int(h * 0.35)))
                else:
                    span = random.randint(max(8, h // 5), max(14, int(h * 0.6)))
                y0 = random.randint(-span // 6, max(0, h - span + span // 6))
                if use_connected_tears:
                    if connected_anchor is None:
                        connected_anchor = y0
                    jitter = max(3, h // (10 if is_tile_scale else 14))
                    y0 = max(-span // 6, min(max(0, h - span + span // 6), connected_anchor + random.randint(-jitter, jitter)))
                steps = random.randint(7, 13)
                if is_tile_scale and random.random() < 0.5:
                    corner_bias = max(0, h // 5)
                    if random.random() < 0.5:
                        y0 = random.randint(-span // 8, corner_bias)
                    else:
                        y0 = random.randint(max(0, h - span - corner_bias), max(0, h - span + span // 8))
                if edge == "left":
                    points = [(0, y0)]
                    for i in range(steps + 1):
                        yy = y0 + (span * i) / steps
                        xx = depth * random.uniform(0.25, 1.0)
                        points.append((xx, yy))
                    points.append((0, y0 + span))
                else:
                    points = [(w, y0)]
                    for i in range(steps + 1):
                        yy = y0 + (span * i) / steps
                        xx = w - depth * random.uniform(0.25, 1.0)
                        points.append((xx, yy))
                    points.append((w, y0 + span))
            else:
                if is_glyph_scale and pattern_name in {"library_blue_board", "library_blue_grid"}:
                    depth = random.randint(max(3, h // 18), max(9, h // 6))
                elif is_tile_scale:
                    depth_max_div = 6 if pattern_name == "random_library_background" else 7
                    depth = random.randint(max(6, h // 18), max(14, h // depth_max_div))
                else:
                    depth = random.randint(max(7, h // 14), max(16, h // 7))
                if is_tile_scale:
                    span = random.randint(max(8, w // 6), max(14, int(w * 0.35)))
                else:
                    span = random.randint(max(8, w // 5), max(14, int(w * 0.6)))
                x0 = random.randint(-span // 6, max(0, w - span + span // 6))
                if use_connected_tears:
                    if connected_anchor is None:
                        connected_anchor = x0
                    jitter = max(3, w // (10 if is_tile_scale else 14))
                    x0 = max(-span // 6, min(max(0, w - span + span // 6), connected_anchor + random.randint(-jitter, jitter)))
                steps = random.randint(7, 13)
                if is_tile_scale and random.random() < 0.5:
                    corner_bias = max(0, w // 5)
                    if random.random() < 0.5:
                        x0 = random.randint(-span // 8, corner_bias)
                    else:
                        x0 = random.randint(max(0, w - span - corner_bias), max(0, w - span + span // 8))
                if edge == "top":
                    points = [(x0, 0)]
                    for i in range(steps + 1):
                        xx = x0 + (span * i) / steps
                        yy = depth * random.uniform(0.25, 1.0)
                        points.append((xx, yy))
                    points.append((x0 + span, 0))
                else:
                    points = [(x0, h)]
                    for i in range(steps + 1):
                        xx = x0 + (span * i) / steps
                        yy = h - depth * random.uniform(0.25, 1.0)
                        points.append((xx, yy))
                    points.append((x0 + span, h))
            draw.polygon(points, fill=0)

        # Corner loss: jagged chips for glyph scale; rounded loss for larger patches.
        if is_glyph_scale:
            # Previously this loop ran zero times. A few small jagged chips
            # near a corner are what sells the torn-fragment effect at glyph
            # resolution, complementing the bigger edge bites above.
            for _ in range(random.randint(1, 2)):
                _draw_glyph_edge_chip(random.choice(("left", "right", "top", "bottom")))
        else:
            corner_range = (
                (0, 2) if is_tile_scale else
                (0, 2)
            )
            for _ in range(random.randint(*corner_range)):
                corner = random.choice(("tl", "tr", "bl", "br"))
                # Tile corners now scale with w/12..w/6 (was w/10..w/4) so a
                # single corner doesn't cover ~25% of the tile.
                rx = random.randint(max(6, w // (12 if is_tile_scale else 16)), max(14, w // (6 if is_tile_scale else 7)))
                ry = random.randint(max(6, h // (12 if is_tile_scale else 16)), max(14, h // (6 if is_tile_scale else 7)))
                if corner == "tl":
                    box = (-rx // 2, -ry // 2, rx * 2, ry * 2)
                elif corner == "tr":
                    box = (w - rx * 2, -ry // 2, w + rx // 2, ry * 2)
                elif corner == "bl":
                    box = (-rx // 2, h - ry * 2, rx * 2, h + ry // 2)
                else:
                    box = (w - rx * 2, h - ry * 2, w + rx // 2, h + ry // 2)
                draw.ellipse(box, fill=0)

        # Occasional tears/holes near the edges.
        tear_range = (
            (0, 1) if is_glyph_scale else
            (0, 1) if is_tile_scale else
            (0, 1)
        )
        for _ in range(random.randint(*tear_range)):
            if is_glyph_scale:
                _draw_glyph_edge_chip(random.choice(("left", "right", "top", "bottom")))
            else:
                tear_w = random.randint(max(3, w // (18 if is_tile_scale else 24)), max(7, w // (8 if is_tile_scale else 14)))
                tear_h = random.randint(max(3, h // (18 if is_tile_scale else 24)), max(7, h // (8 if is_tile_scale else 14)))
                margin_x = max(1, w // 8)
                margin_y = max(1, h // 8)
                side = random.choice(("left", "right", "top", "bottom"))
                if side == "left":
                    x0 = random.randint(0, margin_x)
                    y0 = random.randint(0, max(0, h - tear_h))
                elif side == "right":
                    x0 = random.randint(max(0, w - margin_x - tear_w), max(0, w - tear_w))
                    y0 = random.randint(0, max(0, h - tear_h))
                elif side == "top":
                    x0 = random.randint(0, max(0, w - tear_w))
                    y0 = random.randint(0, margin_y)
                else:
                    x0 = random.randint(0, max(0, w - tear_w))
                    y0 = random.randint(max(0, h - margin_y - tear_h), max(0, h - tear_h))
                draw.ellipse([x0, y0, x0 + tear_w, y0 + tear_h], fill=0)

        if is_glyph_scale:
            return mask.filter(ImageFilter.GaussianBlur(radius=max(0.24, min(w, h) / 220.0)))
        if is_tile_scale:
            return mask.filter(ImageFilter.GaussianBlur(radius=max(0.22, min(w, h) / 220.0)))
        return mask.filter(ImageFilter.GaussianBlur(radius=max(0.35, min(w, h) / 150.0)))

    def _detect_connected_edge_background(self, img: Image.Image) -> Image.Image | None:
        """
        Try to identify the uniform padding bars that surround a glyph patch
        (artifacts of the aspect-preserving square-pad step) so they can be
        replaced with the library backing instead of looking like parchment.

        This is intentionally conservative: false positives flood-fill across
        the real parchment and erase the glyph's surroundings, which is the
        single biggest visual failure mode for the glyph augmentation.
        """
        w, h = img.size
        if max(w, h) > 192:
            return None

        arr = np.asarray(img.convert("RGB"), dtype=np.int16)
        border = np.concatenate(
            [
                arr[0, :, :],
                arr[-1, :, :],
                arr[:, 0, :],
                arr[:, -1, :],
            ],
            axis=0,
        )
        border_std = float(border.std(axis=0).mean())
        border_color = np.median(border, axis=0)

        # Parchment-likely color: take pixels brighter than the median (i.e.
        # the lighter half of the patch, which on a glyph is the parchment
        # not the ink stroke). This avoids the previous bug where a small
        # crop's center landed on dark ink, so color_gap blew up and the
        # flood-fill swept across all real parchment.
        gray = arr.mean(axis=2)
        brightness_threshold = float(np.percentile(gray, 60))
        bright_mask = gray >= brightness_threshold
        if bright_mask.sum() >= 16:
            parchment_color = np.median(arr[bright_mask].reshape(-1, 3), axis=0)
        else:
            parchment_color = border_color
        color_gap = float(np.abs(border_color - parchment_color).mean())

        # Require a uniform border AND a real color separation between the
        # padding strip and the actual parchment surface. If padding was
        # sampled from parchment edges the two will be close and we should
        # skip the inference (edge bites will handle the torn-fragment look).
        if border_std > 14 or color_gap < 22:
            return None

        # Tight per-pixel threshold tied to how uniform the border itself
        # is, not to color_gap. This keeps the flood-fill confined to the
        # quasi-uniform padding bar instead of bleeding into real parchment.
        threshold = max(6, int(border_std * 3.0) + 2)
        diff = np.abs(arr - border_color).max(axis=2)
        candidate = diff <= threshold

        visited = np.zeros((h, w), dtype=bool)
        stack: list[tuple[int, int]] = []
        for x in range(w):
            stack.append((0, x))
            stack.append((h - 1, x))
        for y in range(h):
            stack.append((y, 0))
            stack.append((y, w - 1))

        while stack:
            y, x = stack.pop()
            if y < 0 or y >= h or x < 0 or x >= w:
                continue
            if visited[y, x] or not candidate[y, x]:
                continue
            visited[y, x] = True
            stack.append((y - 1, x))
            stack.append((y + 1, x))
            stack.append((y, x - 1))
            stack.append((y, x + 1))

        # Sanity bail-out: if the flood-fill thinks more than ~55% of the
        # patch is "edge background", it has almost certainly leaked into
        # the parchment. Drop the inference and let the edge bites handle
        # the torn-fragment effect on their own.
        coverage = float(visited.sum()) / float(max(1, h * w))
        if coverage > 0.55:
            return None

        foreground = np.where(visited, 0, 255).astype(np.uint8)
        return Image.fromarray(foreground, mode="L").filter(ImageFilter.GaussianBlur(radius=1.0))

    def _add_fragment_shadow(self, composited: Image.Image, backdrop: Image.Image, fragment_mask: Image.Image) -> Image.Image:
        offset = max(1, min(composited.size) // 80)
        shadow_mask = ImageChops.offset(fragment_mask, offset, offset).filter(
            ImageFilter.GaussianBlur(radius=max(1.0, min(composited.size) / 55.0))
        )
        shadow_alpha = shadow_mask.point(lambda v: int(v * 0.12))
        dark_backdrop = Image.blend(backdrop, Image.new("RGB", composited.size, (0, 0, 0)), 0.25)
        shadowed_backdrop = Image.composite(dark_backdrop, backdrop, shadow_alpha)
        return Image.composite(composited, shadowed_backdrop, fragment_mask)

    def _add_backing_seams(self, image: Image.Image, mean: tuple[int, int, int], blue_bias: bool = False) -> Image.Image:
        w, h = image.size
        draw = ImageDraw.Draw(image, "RGBA")
        spacing = random.randint(*self.grid_spacing_range)
        base = (
            min(255, mean[0] + (8 if blue_bias else 16)),
            min(255, mean[1] + (10 if blue_bias else 16)),
            min(255, mean[2] + (18 if blue_bias else 16)),
            18,
        )
        for idx, x in enumerate(range(0, w + spacing, spacing)):
            if idx % random.choice((4, 5, 6)) == 0:
                draw.line([(x, 0), (x, h)], fill=base, width=self.grid_line_width)
        for idx, y in enumerate(range(0, h + spacing, spacing)):
            if idx % random.choice((4, 5, 6)) == 0:
                draw.line([(0, y), (w, y)], fill=base, width=self.grid_line_width)
        return image

    def _add_support_noise(self, image: Image.Image, mean: tuple[int, int, int]) -> Image.Image:
        w, h = image.size
        arr = np.asarray(image, dtype=np.int16)
        noise = np.random.normal(0.0, 4.0, size=(h, w, 1))
        banding = np.random.normal(0.0, 2.0, size=(h, 1, 1))
        arr = np.clip(arr + noise + banding, 0, 255).astype(np.uint8)
        out = Image.fromarray(arr, mode="RGB")
        draw = ImageDraw.Draw(out, "RGBA")
        if random.random() < 0.6:
            band = max(6, min(w, h) // random.randint(12, 20))
            shade = (
                max(0, mean[0] - random.randint(4, 12)),
                max(0, mean[1] - random.randint(4, 12)),
                max(0, mean[2] - random.randint(4, 12)),
                random.randint(18, 36),
            )
            edge = random.choice(("left", "right", "top", "bottom"))
            if edge == "left":
                draw.rectangle([0, 0, band, h], fill=shade)
            elif edge == "right":
                draw.rectangle([w - band, 0, w, h], fill=shade)
            elif edge == "top":
                draw.rectangle([0, 0, w, band], fill=shade)
            else:
                draw.rectangle([0, h - band, w, h], fill=shade)
        return out

    def _add_scan_noise(self, image: Image.Image, *, edge_only: bool = False) -> Image.Image:
        w, h = image.size
        arr = np.asarray(image, dtype=np.int16)
        noise = np.random.normal(0.0, 3.0, size=(h, w, 1))
        if edge_only:
            yy, xx = np.mgrid[0:h, 0:w]
            dist = np.minimum.reduce([xx, yy, w - 1 - xx, h - 1 - yy]).astype(np.float32)
            edge_width = max(2.0, min(w, h) / 10.0)
            edge_weight = np.clip(1.0 - (dist / edge_width), 0.0, 1.0)[..., None]
            noise *= edge_weight
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def _add_support_fading(self, image: Image.Image, mean: tuple[int, int, int], intensity: float = 0.6) -> Image.Image:
        w, h = image.size
        draw = ImageDraw.Draw(image, "RGBA")
        for _ in range(random.randint(2, 5)):
            span_w = random.randint(max(18, w // 5), max(22, int(w * 0.65)))
            span_h = random.randint(max(18, h // 5), max(22, int(h * 0.65)))
            x0 = random.randint(-span_w // 3, max(0, w - (2 * span_w) // 3))
            y0 = random.randint(-span_h // 3, max(0, h - (2 * span_h) // 3))
            delta = random.randint(5, 14)
            color = (
                max(0, min(255, mean[0] + random.randint(-delta, delta))),
                max(0, min(255, mean[1] + random.randint(-delta, delta))),
                max(0, min(255, mean[2] + random.randint(-delta, delta))),
                int(random.randint(10, 22) * intensity),
            )
            draw.rounded_rectangle([x0, y0, x0 + span_w, y0 + span_h], radius=max(8, min(span_w, span_h) // 7), fill=color)
        return image

    def _add_edge_aging(self, image: Image.Image, *, stronger: bool = False) -> Image.Image:
        w, h = image.size
        draw = ImageDraw.Draw(image, "RGBA")
        bands = [
            (0, 0, w, max(6, h // random.randint(10, 16))),
            (0, h - max(6, h // random.randint(10, 16)), w, h),
            (0, 0, max(6, w // random.randint(10, 16)), h),
            (w - max(6, w // random.randint(10, 16)), 0, w, h),
        ]
        for box in bands:
            draw.rectangle(box, fill=(60, 45, 28, random.randint(14, 34) if stronger else random.randint(10, 28)))
        return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.8, 1.6)))

    def _add_foxing_and_grime(self, image: Image.Image, mean: tuple[int, int, int], *, small_patch: bool = False) -> Image.Image:
        w, h = image.size
        draw = ImageDraw.Draw(image, "RGBA")
        for _ in range(random.randint(8, 16) if small_patch else random.randint(14, 28)):
            rx = random.randint(max(4, w // 18), max(8, w // 6))
            ry = random.randint(max(4, h // 18), max(8, h // 6))
            x0 = random.randint(-rx // 2, w)
            y0 = random.randint(-ry // 2, h)
            color = (
                max(0, min(255, mean[0] - random.randint(10, 55))),
                max(0, min(255, mean[1] - random.randint(12, 58))),
                max(0, min(255, mean[2] - random.randint(18, 65))),
                random.randint(18, 58),
            )
            draw.ellipse([x0, y0, x0 + rx, y0 + ry], fill=color)
        # Horizontal banding can look like synthetic lines on tiny glyph patches.
        if not small_patch:
            for _ in range(random.randint(1, 3)):
                band_h = random.randint(max(6, h // 20), max(10, h // 9))
                y0 = random.randint(0, max(0, h - band_h))
                draw.rectangle([0, y0, w, y0 + band_h], fill=(45, 35, 22, random.randint(12, 24)))
        blur_radius = random.uniform(0.15, 0.45) if small_patch else random.uniform(0.8, 1.5)
        return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    def _add_stain_blooms(self, image: Image.Image, mean: tuple[int, int, int], *, small_patch: bool = False) -> Image.Image:
        w, h = image.size
        draw = ImageDraw.Draw(image, "RGBA")
        n_main = random.randint(4, 7) if not small_patch else random.randint(1, 2)
        for _ in range(n_main):
            cx = random.randint(0, max(0, w - 1))
            cy = random.randint(0, max(0, h - 1))
            base_r = (
                random.randint(max(8, min(w, h) // 10), max(16, min(w, h) // 3))
                if not small_patch
                else random.randint(max(6, min(w, h) // 12), max(10, min(w, h) // 5))
            )
            lobes = random.randint(5, 9) if not small_patch else random.randint(3, 5)
            for _lobe in range(lobes):
                ox = random.randint(-base_r, base_r)
                oy = random.randint(-base_r, base_r)
                px = cx + ox
                py = cy + oy
                poly_points = []
                n_pts = random.randint(6, 10)
                for i in range(n_pts):
                    theta = (2.0 * math.pi * i / n_pts) + random.uniform(-0.22, 0.22)
                    radial = base_r * random.uniform(0.25, 0.95)
                    ex = int(round(px + radial * math.cos(theta)))
                    ey = int(round(py + radial * math.sin(theta) * random.uniform(0.7, 1.3)))
                    poly_points.append((ex, ey))
                color = (
                    max(0, min(255, mean[0] - random.randint(20, 78))),
                    max(0, min(255, mean[1] - random.randint(20, 80))),
                    max(0, min(255, mean[2] - random.randint(26, 90))),
                    random.randint(10, 24) if small_patch else random.randint(20, 46),
                )
                draw.polygon(poly_points, fill=color)

            # Add irregular dark cores/splashes so stains are less uniform.
            for _core in range(random.randint(0, 2) if small_patch else random.randint(2, 5)):
                cr = random.randint(max(2, base_r // 10), max(3, base_r // 5))
                cox = random.randint(-base_r // 2, base_r // 2)
                coy = random.randint(-base_r // 2, base_r // 2)
                color_core = (
                    max(0, min(255, mean[0] - random.randint(30, 90))),
                    max(0, min(255, mean[1] - random.randint(30, 94))),
                    max(0, min(255, mean[2] - random.randint(36, 106))),
                    random.randint(12, 28) if small_patch else random.randint(26, 56),
                )
                # Small asymmetric splashes around the stain center.
                pts = []
                n_core_pts = random.randint(5, 8)
                for i in range(n_core_pts):
                    theta = (2.0 * math.pi * i / n_core_pts) + random.uniform(-0.35, 0.35)
                    rr = cr * random.uniform(0.5, 1.3)
                    pts.append(
                        (
                            int(round(cx + cox + rr * math.cos(theta))),
                            int(round(cy + coy + rr * math.sin(theta) * random.uniform(0.75, 1.25))),
                        )
                    )
                draw.polygon(pts, fill=color_core)

        blur_radius = random.uniform(0.20, 0.55) if small_patch else random.uniform(0.9, 1.8)
        return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    def _add_soft_clouds(
        self,
        image: Image.Image,
        mean: tuple[int, int, int],
        *,
        count_scale: float,
        neutral_bias: bool = False,
    ) -> Image.Image:
        w, h = image.size
        draw = ImageDraw.Draw(image, "RGBA")
        n = max(4, int((min(w, h) / 70.0) * 8 * count_scale))
        for _ in range(n):
            rx = random.randint(max(8, w // 12), max(12, w // 4))
            ry = random.randint(max(8, h // 12), max(12, h // 4))
            x0 = random.randint(-rx // 2, w)
            y0 = random.randint(-ry // 2, h)
            delta = random.randint(6, 18) if neutral_bias else random.randint(8, 24)
            color = (
                max(0, min(255, mean[0] + random.randint(-delta, delta))),
                max(0, min(255, mean[1] + random.randint(-delta, delta))),
                max(0, min(255, mean[2] + random.randint(-delta, delta // (2 if neutral_bias else 1)))),
                random.randint(10, 22),
            )
            draw.ellipse([x0, y0, x0 + rx, y0 + ry], fill=color)
        return image

    def _add_edge_vignette(self, image: Image.Image, mean: tuple[int, int, int], heavier: bool = False) -> Image.Image:
        w, h = image.size
        draw = ImageDraw.Draw(image, "RGBA")
        band = max(4, min(w, h) // random.randint(10, 18))
        edge_color = (
            max(0, mean[0] - random.randint(10, 18)),
            max(0, mean[1] - random.randint(10, 18)),
            max(0, mean[2] - random.randint(10, 18)),
            random.randint(28, 42) if heavier else random.randint(16, 28),
        )
        draw.rectangle([0, 0, w, band], fill=edge_color)
        draw.rectangle([0, h - band, w, h], fill=edge_color)
        draw.rectangle([0, 0, band, h], fill=edge_color)
        draw.rectangle([w - band, 0, w, h], fill=edge_color)
        return image

    def _add_mount_shading(self, image: Image.Image, mean: tuple[int, int, int], *, warm: bool) -> Image.Image:
        w, h = image.size
        draw = ImageDraw.Draw(image, "RGBA")
        for _ in range(random.randint(2, 4)):
            span_w = random.randint(max(20, w // 4), max(24, int(w * 0.75)))
            span_h = random.randint(max(16, h // 6), max(20, int(h * 0.28)))
            x0 = random.randint(-span_w // 5, max(0, w - (4 * span_w) // 5))
            y0 = random.randint(-span_h // 4, max(0, h - (3 * span_h) // 4))
            if warm:
                color = (
                    max(0, min(255, mean[0] + random.randint(4, 14))),
                    max(0, min(255, mean[1] + random.randint(0, 10))),
                    max(0, min(255, mean[2] - random.randint(4, 16))),
                    random.randint(14, 28),
                )
            else:
                color = (
                    max(0, min(255, mean[0] + random.randint(-8, 8))),
                    max(0, min(255, mean[1] + random.randint(-6, 8))),
                    max(0, min(255, mean[2] + random.randint(-6, 10))),
                    random.randint(12, 24),
                )
            draw.rounded_rectangle([x0, y0, x0 + span_w, y0 + span_h], radius=max(8, span_h // 3), fill=color)
        return image

    def _make_legacy_overlay(self, pattern_name: str, size: tuple[int, int]) -> Image.Image:
        pattern_name = self._canonical_pattern(pattern_name)
        if pattern_name == "library_blue_grid":
            return self._make_blue_grid(size)
        if pattern_name == "library_neutral_mount":
            return self._make_edge_tint(size)
        if pattern_name == "library_warm_mount":
            return self._make_paper_stains(size)
        if pattern_name == "parchment_stains":
            return self._make_paper_stains(size)
        return self._make_blue_wash(size)

    def _make_blue_wash(self, size: tuple[int, int]) -> Image.Image:
        w, h = size
        base = Image.new(
            "RGB",
            size,
            color=(random.randint(25, 60), random.randint(90, 145), random.randint(150, 215)),
        )
        draw = ImageDraw.Draw(base, "RGBA")
        for _ in range(random.randint(8, 18)):
            x0 = random.randint(-w // 4, w)
            y0 = random.randint(-h // 4, h)
            rx = random.randint(max(6, w // 12), max(12, w // 3))
            ry = random.randint(max(6, h // 12), max(12, h // 3))
            draw.ellipse(
                [x0, y0, x0 + rx, y0 + ry],
                fill=(random.randint(35, 80), random.randint(100, 155), random.randint(165, 225), random.randint(18, 45)),
            )
        return base.filter(ImageFilter.GaussianBlur(radius=random.uniform(*self.blur_radius_range)))

    def _make_blue_grid(self, size: tuple[int, int]) -> Image.Image:
        w, h = size
        base = self._make_blue_wash(size)
        draw = ImageDraw.Draw(base, "RGBA")
        spacing = random.randint(*self.grid_spacing_range)
        major_every = random.choice((4, 5, 6))
        for idx, x in enumerate(range(0, w + spacing, spacing)):
            is_major = (idx % major_every) == 0
            color = (120, 165, 235, 120 if is_major else 70)
            draw.line([(x, 0), (x, h)], fill=color, width=self.grid_line_width + (1 if is_major else 0))
        for idx, y in enumerate(range(0, h + spacing, spacing)):
            is_major = (idx % major_every) == 0
            color = (120, 165, 235, 120 if is_major else 70)
            draw.line([(0, y), (w, y)], fill=color, width=self.grid_line_width + (1 if is_major else 0))
        return base.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.0)))

    def _make_blue_mottle(self, size: tuple[int, int]) -> Image.Image:
        w, h = size
        base = Image.new(
            "RGB",
            size,
            color=(random.randint(185, 225), random.randint(205, 240), random.randint(220, 250)),
        )
        draw = ImageDraw.Draw(base, "RGBA")
        for _ in range(random.randint(18, 36)):
            x0 = random.randint(-w // 3, w)
            y0 = random.randint(-h // 3, h)
            r = random.randint(max(4, min(w, h) // 16), max(10, min(w, h) // 4))
            draw.ellipse(
                [x0, y0, x0 + r, y0 + r],
                fill=(random.randint(70, 115), random.randint(120, 170), random.randint(180, 235), random.randint(14, 42)),
            )
        return base.filter(ImageFilter.GaussianBlur(radius=random.uniform(*self.blur_radius_range)))

    def _make_edge_tint(self, size: tuple[int, int]) -> Image.Image:
        w, h = size
        base = Image.new(
            "RGB",
            size,
            color=(random.randint(220, 240), random.randint(225, 245), random.randint(228, 248)),
        )
        draw = ImageDraw.Draw(base, "RGBA")
        edge_color = (
            random.randint(75, 120),
            random.randint(115, 165),
            random.randint(175, 235),
            random.randint(35, 65),
        )
        band = max(4, min(w, h) // random.randint(8, 14))
        draw.rectangle([0, 0, w, band], fill=edge_color)
        draw.rectangle([0, h - band, w, h], fill=edge_color)
        draw.rectangle([0, 0, band, h], fill=edge_color)
        draw.rectangle([w - band, 0, w, h], fill=edge_color)
        return base.filter(ImageFilter.GaussianBlur(radius=random.uniform(*self.blur_radius_range)))

    def _make_paper_stains(self, size: tuple[int, int]) -> Image.Image:
        w, h = size
        base = Image.new(
            "RGB",
            size,
            color=(random.randint(228, 244), random.randint(227, 242), random.randint(216, 236)),
        )
        draw = ImageDraw.Draw(base, "RGBA")
        for _ in range(random.randint(12, 28)):
            x0 = random.randint(-w // 4, w)
            y0 = random.randint(-h // 4, h)
            rx = random.randint(max(6, w // 14), max(12, w // 3))
            ry = random.randint(max(6, h // 14), max(12, h // 3))
            draw.ellipse(
                [x0, y0, x0 + rx, y0 + ry],
                fill=(random.randint(170, 215), random.randint(175, 220), random.randint(150, 205), random.randint(10, 28)),
            )
        return base.filter(ImageFilter.GaussianBlur(radius=random.uniform(*self.blur_radius_range)))
