#!/usr/bin/env python3
"""
Create fixed-size image latent vectors from SIFT descriptors.

This is a lightweight baseline for comparing learned model latents against a
classical local-feature representation. Each image is represented by pooled
RootSIFT descriptors:

  - mean:     128-dimensional latent
  - mean_std: 256-dimensional latent (default)

Outputs are saved to a compressed NPZ with:
  paths:          string array of image paths
  latents:        float32 array [N, D]
  num_keypoints:  int32 array [N]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

try:
    import cv2
except ImportError as exc:
    raise SystemExit(
        "OpenCV is required for SIFT. Install opencv-contrib-python or use the "
        "project environment that already includes cv2."
    ) from exc


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
}


def _read_path_list(path: Path) -> List[Path]:
    paths: List[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        paths.append(Path(line).expanduser())
    return paths


def _iter_image_dir(path: Path, recursive: bool) -> Iterable[Path]:
    iterator = path.rglob("*") if recursive else path.glob("*")
    for candidate in iterator:
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
            yield candidate


def collect_image_paths(args: argparse.Namespace) -> List[Path]:
    paths: List[Path] = []
    for image in args.image or []:
        paths.append(Path(image).expanduser())
    for list_path in args.image_list or []:
        paths.extend(_read_path_list(Path(list_path).expanduser()))
    for image_dir in args.image_dir or []:
        paths.extend(_iter_image_dir(Path(image_dir).expanduser(), args.recursive))

    seen = set()
    unique: List[Path] = []
    for path in paths:
        resolved_key = str(path.resolve()) if path.exists() else str(path)
        if resolved_key in seen:
            continue
        seen.add(resolved_key)
        unique.append(path)
    return sorted(unique, key=lambda p: str(p))


def read_image_gray(path: Path) -> np.ndarray:
    # cv2.imread can fail on some unicode paths; imdecode handles those paths reliably.
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def resize_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return image
    height, width = image.shape[:2]
    current_max = max(height, width)
    if current_max <= max_side:
        return image
    scale = float(max_side) / float(current_max)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def rootsift(descriptors: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    descriptors = descriptors.astype(np.float32, copy=False)
    descriptors /= descriptors.sum(axis=1, keepdims=True) + eps
    descriptors = np.sqrt(descriptors)
    descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True) + eps
    return descriptors


def create_sift() -> cv2.SIFT:
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError(
            "This OpenCV build does not expose cv2.SIFT_create(). Install "
            "opencv-contrib-python or a newer opencv-python package."
        )
    return cv2.SIFT_create


def pool_descriptors(descriptors: np.ndarray | None, pooling: str, use_rootsift: bool) -> np.ndarray:
    if pooling == "mean":
        dim = 128
    elif pooling == "mean_std":
        dim = 256
    else:
        raise ValueError(f"Unknown pooling mode: {pooling}")

    if descriptors is None or descriptors.shape[0] == 0:
        return np.zeros(dim, dtype=np.float32)

    descriptors = descriptors.astype(np.float32, copy=False)
    if use_rootsift:
        descriptors = rootsift(descriptors)

    mean = descriptors.mean(axis=0)
    if pooling == "mean":
        latent = mean
    else:
        std = descriptors.std(axis=0)
        latent = np.concatenate([mean, std], axis=0)

    norm = float(np.linalg.norm(latent))
    if norm > 0:
        latent = latent / norm
    return latent.astype(np.float32, copy=False)


def extract_sift_latent(
    path: Path,
    sift,
    *,
    pooling: str,
    use_rootsift: bool,
    resize_side: int,
) -> tuple[np.ndarray, int]:
    image = read_image_gray(path)
    image = resize_max_side(image, resize_side)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    latent = pool_descriptors(descriptors, pooling=pooling, use_rootsift=use_rootsift)
    return latent, len(keypoints or [])


def save_csv(path: Path, image_paths: Sequence[Path], latents: np.ndarray, num_keypoints: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "num_keypoints", *[f"z{i:04d}" for i in range(latents.shape[1])]])
        for image_path, latent, count in zip(image_paths, latents, num_keypoints):
            writer.writerow([str(image_path), int(count), *latent.tolist()])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create fixed-size SIFT latent vectors for images.")
    parser.add_argument("--image", nargs="*", default=[], help="One or more image paths.")
    parser.add_argument("--image-list", nargs="*", default=[], help="Text files containing one image path per line.")
    parser.add_argument("--image-dir", nargs="*", default=[], help="Directories containing images.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan --image-dir paths.")
    parser.add_argument(
        "--output",
        default="debug_tools/SIFT/outputs/sift_latents.npz",
        help="Output NPZ path.",
    )
    parser.add_argument("--csv-output", default="", help="Optional CSV output path.")
    parser.add_argument("--max-keypoints", type=int, default=4096, help="Maximum SIFT keypoints per image.")
    parser.add_argument("--resize-max-side", type=int, default=1600, help="Resize image long side before SIFT; <=0 disables.")
    parser.add_argument("--pooling", choices=("mean", "mean_std"), default="mean_std")
    parser.add_argument("--no-rootsift", action="store_true", help="Use raw SIFT descriptors instead of RootSIFT.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first unreadable image.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_paths = collect_image_paths(args)
    if not image_paths:
        print("No images found. Pass --image, --image-list, or --image-dir.", file=sys.stderr)
        return 2

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sift_factory = create_sift()
    sift = sift_factory(nfeatures=max(0, int(args.max_keypoints)))
    use_rootsift = not args.no_rootsift

    latents: List[np.ndarray] = []
    kept_paths: List[Path] = []
    num_keypoints: List[int] = []
    failures: List[dict] = []

    for idx, image_path in enumerate(image_paths, start=1):
        try:
            latent, count = extract_sift_latent(
                image_path,
                sift,
                pooling=args.pooling,
                use_rootsift=use_rootsift,
                resize_side=args.resize_max_side,
            )
        except Exception as exc:
            if args.fail_fast:
                raise
            failures.append({"path": str(image_path), "error": str(exc)})
            print(f"[{idx}/{len(image_paths)}] skipped {image_path}: {exc}", file=sys.stderr)
            continue

        latents.append(latent)
        kept_paths.append(image_path)
        num_keypoints.append(count)
        if idx == 1 or idx % 100 == 0 or idx == len(image_paths):
            print(f"[{idx}/{len(image_paths)}] processed {image_path} ({count} keypoints)")

    if not latents:
        print("No latents were created; all images failed.", file=sys.stderr)
        return 1

    latents_arr = np.stack(latents, axis=0).astype(np.float32, copy=False)
    keypoints_arr = np.asarray(num_keypoints, dtype=np.int32)
    paths_arr = np.asarray([str(path) for path in kept_paths], dtype=str)

    metadata = {
        "pooling": args.pooling,
        "rootsift": use_rootsift,
        "max_keypoints": int(args.max_keypoints),
        "resize_max_side": int(args.resize_max_side),
        "num_images_requested": len(image_paths),
        "num_images_written": len(kept_paths),
        "latent_dim": int(latents_arr.shape[1]),
        "failures": failures,
    }

    np.savez_compressed(
        output_path,
        paths=paths_arr,
        latents=latents_arr,
        num_keypoints=keypoints_arr,
        metadata=json.dumps(metadata, ensure_ascii=False),
    )

    if args.csv_output:
        save_csv(Path(args.csv_output), kept_paths, latents_arr, keypoints_arr)

    print(f"Saved {len(kept_paths)} SIFT latents to {output_path}")
    print(f"Latent shape: {latents_arr.shape}")
    if failures:
        print(f"Skipped {len(failures)} images; see metadata['failures'] in the NPZ.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
