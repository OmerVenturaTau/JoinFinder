"""Domain-specific augmentations for manuscript images.

RandomGammaCorrection: simulates ink fading / over-exposure variation across
    different manuscript preservation states and scanning conditions.

RandomResolutionJitter: simulates variable scan DPI by down-scaling then
    up-scaling the image, teaching the model to handle low-quality inputs.
"""

from __future__ import annotations

import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
from typing import Sequence


class RandomGammaCorrection:
    """Apply random gamma correction to simulate ink fading and exposure variation.

    Gamma < 1 brightens the image (simulates faded ink / washed-out scans).
    Gamma > 1 darkens the image (simulates high-contrast / fresh ink).

    Args:
        gamma_range: (lo, hi) range for the gamma exponent.
        p: probability of applying the transform.
    """

    def __init__(self, gamma_range: tuple[float, float] = (0.7, 1.5), p: float = 0.3):
        if gamma_range[0] <= 0 or gamma_range[1] <= 0:
            raise ValueError(f"gamma_range values must be > 0, got {gamma_range}")
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return image
        gamma = random.uniform(*self.gamma_range)
        lut = [int(((i / 255.0) ** gamma) * 255) for i in range(256)]
        if image.mode == "RGB":
            lut = lut * 3
        elif image.mode == "L":
            pass
        else:
            return image
        return image.point(lut)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(gamma_range={self.gamma_range}, p={self.p})"


class RandomResolutionJitter:
    """Simulate variable scan DPI by down-scaling then up-scaling.

    The image is resized to a fraction of its original size using LANCZOS
    (high-quality downscale), then scaled back up using BILINEAR (simulates
    the loss of detail from a lower-DPI scan).

    Args:
        scale_range: (lo, hi) range for the intermediate scale factor.
            E.g. (0.5, 0.8) means the image is first shrunk to 50-80% of its
            size, then scaled back to original.
        p: probability of applying the transform.
    """

    def __init__(self, scale_range: tuple[float, float] = (0.5, 0.8), p: float = 0.2):
        if scale_range[0] <= 0 or scale_range[1] > 1:
            raise ValueError(f"scale_range must be in (0, 1], got {scale_range}")
        self.scale_range = scale_range
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return image
        w, h = image.size
        scale = random.uniform(*self.scale_range)
        small_w = max(1, int(w * scale))
        small_h = max(1, int(h * scale))
        small = image.resize((small_w, small_h), Image.Resampling.LANCZOS)
        return small.resize((w, h), Image.Resampling.BILINEAR)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(scale_range={self.scale_range}, p={self.p})"


class RandomZoomJitter:
    """Randomly zoom in/out and return to original size."""

    def __init__(self, scale_range: tuple[float, float] = (0.82, 1.25), p: float = 0.2):
        if scale_range[0] <= 0 or scale_range[1] <= 0:
            raise ValueError(f"scale_range values must be > 0, got {scale_range}")
        self.scale_range = scale_range
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return image
        img = image.convert("RGB")
        w, h = img.size
        scale = random.uniform(*self.scale_range)
        if abs(scale - 1.0) < 1e-3:
            return img
        if scale > 1.0:
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))
            enlarged = img.resize((nw, nh), Image.Resampling.BICUBIC)
            left = max(0, (nw - w) // 2)
            top = max(0, (nh - h) // 2)
            return enlarged.crop((left, top, left + w, top + h))
        nw = max(2, int(round(w * scale)))
        nh = max(2, int(round(h * scale)))
        shrunk = img.resize((nw, nh), Image.Resampling.BICUBIC)
        edge = img.resize((1, 1), Image.Resampling.BOX).getpixel((0, 0))
        canvas = Image.new("RGB", (w, h), edge)
        left = (w - nw) // 2
        top = (h - nh) // 2
        canvas.paste(shrunk, (left, top))
        return canvas


class RandomTiltJitter:
    """Random small-angle tilt with edge-color fill."""

    def __init__(self, degrees: float = 10.0, p: float = 0.1):
        self.degrees = float(degrees)
        self.p = float(p)

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return image
        img = image.convert("RGB")
        w, h = img.size
        angle = random.uniform(-self.degrees, self.degrees)
        fill = img.resize((1, 1), Image.Resampling.BOX).getpixel((0, 0))
        return img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=fill)


class RandomLocalTexturePerturbation:
    """Inject low-frequency texture/noise fields to suppress library style leakage."""

    def __init__(
        self,
        p: float = 0.25,
        alpha_range: tuple[float, float] = (0.06, 0.20),
        blur_radius_range: tuple[float, float] = (4.0, 14.0),
    ) -> None:
        self.p = p
        self.alpha_range = alpha_range
        self.blur_radius_range = blur_radius_range

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.p or image.mode != "RGB":
            return image
        w, h = image.size
        noise = Image.effect_noise((w, h), random.uniform(20.0, 65.0)).convert("L")
        noise = noise.filter(ImageFilter.GaussianBlur(radius=random.uniform(*self.blur_radius_range)))
        tint_rgb = tuple(random.randint(90, 170) for _ in range(3))
        texture = ImageOps.colorize(noise, black=(0, 0, 0), white=tint_rgb).convert("RGB")
        alpha = random.uniform(*self.alpha_range)
        return Image.blend(image, texture, alpha)


class RandomToneAndContrastJitter:
    """Channel/tone perturbations beyond standard ColorJitter."""

    def __init__(
        self,
        p: float = 0.3,
        gamma_range: tuple[float, float] = (0.8, 1.25),
        channel_scale_range: tuple[float, float] = (0.9, 1.12),
        local_contrast_range: tuple[float, float] = (0.75, 1.3),
    ) -> None:
        self.p = p
        self.gamma_range = gamma_range
        self.channel_scale_range = channel_scale_range
        self.local_contrast_range = local_contrast_range

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.p or image.mode != "RGB":
            return image

        gamma = random.uniform(*self.gamma_range)
        lut = [int(((i / 255.0) ** gamma) * 255) for i in range(256)] * 3
        out = image.point(lut)

        channels = out.split()
        scaled_channels = []
        for ch in channels:
            scale = random.uniform(*self.channel_scale_range)
            scaled_channels.append(ch.point(lambda px, s=scale: int(max(0, min(255, px * s)))))
        out = Image.merge("RGB", tuple(scaled_channels))
        out = ImageEnhance.Contrast(out).enhance(random.uniform(*self.local_contrast_range))
        return out


class RandomInkDegradation:
    """Simulate scanner/ink degradation and compression artifacts."""

    def __init__(
        self,
        p: float = 0.25,
        jpeg_quality_range: tuple[int, int] = (28, 65),
        blur_radius_range: tuple[float, float] = (0.35, 1.1),
        unsharp_prob: float = 0.25,
    ) -> None:
        self.p = p
        self.jpeg_quality_range = jpeg_quality_range
        self.blur_radius_range = blur_radius_range
        self.unsharp_prob = unsharp_prob

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.p or image.mode != "RGB":
            return image

        out = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(*self.blur_radius_range)))
        if random.random() < self.unsharp_prob:
            out = out.filter(ImageFilter.UnsharpMask(radius=1.2, percent=125, threshold=2))

        # PIL JPEG roundtrip without filesystem IO.
        from io import BytesIO
        buffer = BytesIO()
        out.save(buffer, format="JPEG", quality=random.randint(*self.jpeg_quality_range))
        buffer.seek(0)
        out = Image.open(buffer).convert("RGB")
        return out


class RandomSpeckleAndMorphology:
    """Approximate ink stroke erosion/dilation + dust speckle artifacts."""

    def __init__(
        self,
        p: float = 0.22,
        speckle_prob_range: tuple[float, float] = (0.002, 0.01),
        morphology_prob: float = 0.5,
    ) -> None:
        self.p = p
        self.speckle_prob_range = speckle_prob_range
        self.morphology_prob = morphology_prob

    def _inject_speckles(self, image: Image.Image) -> Image.Image:
        out = image.copy()
        draw = ImageDraw.Draw(out)
        w, h = out.size
        density = random.uniform(*self.speckle_prob_range)
        n_points = max(1, int(w * h * density))
        for _ in range(n_points):
            x = random.randint(0, max(0, w - 1))
            y = random.randint(0, max(0, h - 1))
            val = random.randint(0, 255)
            draw.point((x, y), fill=(val, val, val))
        return out

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.p or image.mode != "RGB":
            return image
        out = image
        if random.random() < self.morphology_prob:
            gray = ImageOps.grayscale(out)
            if random.random() < 0.5:
                gray = gray.filter(ImageFilter.MinFilter(3))
            else:
                gray = gray.filter(ImageFilter.MaxFilter(3))
            out = Image.merge("RGB", (gray, gray, gray))
        out = self._inject_speckles(out)
        return out


class RandomBorderMaskAndEdgeCrop:
    """Suppress layout/frame signatures and simulate fragment truncation."""

    def __init__(
        self,
        p: float = 0.2,
        max_border_frac: float = 0.13,
        max_crop_frac: float = 0.18,
    ) -> None:
        self.p = p
        self.max_border_frac = max_border_frac
        self.max_crop_frac = max_crop_frac

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.p or image.mode != "RGB":
            return image
        w, h = image.size
        out = image.copy()
        draw = ImageDraw.Draw(out)
        border_color = tuple(random.randint(160, 220) for _ in range(3))
        bw = max(1, int(w * random.uniform(0.03, self.max_border_frac)))
        bh = max(1, int(h * random.uniform(0.03, self.max_border_frac)))
        if random.random() < 0.5:
            draw.rectangle((0, 0, bw, h), fill=border_color)
        if random.random() < 0.5:
            draw.rectangle((w - bw, 0, w, h), fill=border_color)
        if random.random() < 0.5:
            draw.rectangle((0, 0, w, bh), fill=border_color)
        if random.random() < 0.5:
            draw.rectangle((0, h - bh, w, h), fill=border_color)

        if random.random() < 0.65:
            crop_side = random.choice(["left", "right", "top", "bottom"])
            frac = random.uniform(0.05, self.max_crop_frac)
            if crop_side in {"left", "right"}:
                cw = max(1, int(w * frac))
                if crop_side == "left":
                    draw.rectangle((0, 0, cw, h), fill=border_color)
                else:
                    draw.rectangle((w - cw, 0, w, h), fill=border_color)
            else:
                ch = max(1, int(h * frac))
                if crop_side == "top":
                    draw.rectangle((0, 0, w, ch), fill=border_color)
                else:
                    draw.rectangle((0, h - ch, w, h), fill=border_color)
        return out


class RandomWhiteBackground:
    """Remove parchment/library background cues by compositing likely ink on white."""

    def __init__(
        self,
        p: float = 0.3,
        min_contrast_from_background: float = 28.0,
        ink_max_luminance: float = 188.0,
        max_color_spread: float = 100.0,
        edge_softness: float = 18.0,
    ) -> None:
        self.p = float(p)
        self.min_contrast_from_background = float(min_contrast_from_background)
        self.ink_max_luminance = float(ink_max_luminance)
        self.max_color_spread = float(max_color_spread)
        self.edge_softness = max(1.0, float(edge_softness))

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return image

        img = image.convert("RGB")
        arr = np.asarray(img, dtype=np.float32)
        if arr.size == 0:
            return img

        luminance = (0.299 * arr[:, :, 0]) + (0.587 * arr[:, :, 1]) + (0.114 * arr[:, :, 2])
        color_spread = arr.max(axis=2) - arr.min(axis=2)

        bright_cutoff = np.percentile(luminance, 60)
        bright_pixels = luminance >= bright_cutoff
        if np.any(bright_pixels):
            background_luminance = float(np.median(luminance[bright_pixels]))
        else:
            background_luminance = float(np.median(luminance))

        ink_cutoff = min(
            self.ink_max_luminance,
            max(0.0, background_luminance - self.min_contrast_from_background),
        )
        neutral_enough = color_spread <= self.max_color_spread
        darkness_alpha = np.clip((ink_cutoff + self.edge_softness - luminance) / self.edge_softness, 0.0, 1.0)
        alpha = np.where(neutral_enough, darkness_alpha, 0.0)[:, :, None]

        white = np.full_like(arr, 255.0)
        out = (arr * alpha) + (white * (1.0 - alpha))
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(p={self.p}, "
            f"min_contrast_from_background={self.min_contrast_from_background}, "
            f"ink_max_luminance={self.ink_max_luminance})"
        )


class WeightedOneOf:
    """Apply one normalized weighted augmentation with an optional no-op chance."""

    def __init__(self, candidates: Sequence[tuple[str, object, float]], p: float = 1.0) -> None:
        valid: list[tuple[str, object, float]] = []
        for name, transform, weight in candidates:
            w = float(weight)
            if transform is None or w <= 0:
                continue
            valid.append((name, transform, w))
        self.candidates = valid
        self.p = max(0.0, min(1.0, float(p)))

    def __call__(self, image: Image.Image) -> Image.Image:
        if not self.candidates or random.random() >= self.p:
            return image
        names = [c[0] for c in self.candidates]
        transforms = [c[1] for c in self.candidates]
        weights = [c[2] for c in self.candidates]
        _ = names  # reserved for debug hooks if needed
        chosen = random.choices(transforms, weights=weights, k=1)[0]
        return chosen(image)

    def normalized_probabilities(self) -> dict[str, float]:
        total = sum(weight for _name, _transform, weight in self.candidates)
        if total <= 0:
            return {}
        return {
            name: weight / total
            for name, _transform, weight in self.candidates
        }

    def effective_probabilities(self) -> dict[str, float]:
        return {
            name: self.p * prob
            for name, prob in self.normalized_probabilities().items()
        }
