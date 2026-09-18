#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path("/home/omerv/JoinsFinder")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utilities.augmentations.background_overlays import RandomLibraryBackgroundOverlay


def main() -> None:
    bg_dir = Path("/home/omerv/JoinsFinder/Backgrounds")
    out_dir = Path("/home/omerv/JoinsFinder/Debugs/Augmentations/outputs/background_probability_preview")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = PROJECT_ROOT / "config" / "model_regularization.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    aug_cfg = cfg.get("augmentation", {})
    sampling_alpha = float(aug_cfg.get("background_sampling_alpha", 0.72))
    sampling_floor = float(aug_cfg.get("background_sampling_floor", 80.0))

    overlay = RandomLibraryBackgroundOverlay(
        p=1.0,
        pattern_types=("random_library_background",),
        background_sampling_alpha=sampling_alpha,
        background_sampling_floor=sampling_floor,
    )
    assets = overlay._load_assets()
    weights = overlay._load_library_sampling_weights(assets)
    total = sum(weights) if weights else 1.0

    base_names = []
    for asset in assets:
        name = str(asset.get("name", ""))
        base_names.append(overlay._normalize_library_name(Path(name).stem))
    variant_count_by_library: dict[str, int] = {}
    for lib in base_names:
        variant_count_by_library[lib] = variant_count_by_library.get(lib, 0) + 1

    xlsx_path = bg_dir / "background_appearances.xlsx"
    lib_counts: dict[str, float] = {}
    if xlsx_path.exists():
        df = pd.read_excel(xlsx_path)
        required = {"normalized_library", "manuscripts_with_images_count"}
        if required.issubset(set(df.columns)):
            available = set(base_names)
            for _, row in df.iterrows():
                lib = overlay._normalize_library_name(str(row["normalized_library"]))
                if lib in available:
                    try:
                        lib_counts[lib] = float(row["manuscripts_with_images_count"])
                    except Exception:
                        continue
    total_raw_count = sum(lib_counts.values())

    rows = []
    for asset, w, lib in zip(assets, weights, base_names):
        name = str(asset.get("name", ""))
        variants = max(1, variant_count_by_library.get(lib, 1))
        raw_library_prob = (lib_counts.get(lib, 0.0) / total_raw_count) if total_raw_count > 0 else 0.0
        # "Supposed" count-based per-image probability:
        # library count share, split across image variants of that same library.
        supposed_prob = raw_library_prob / float(variants)
        # Smoothed probability actually used by training sampler.
        training_prob = float(w) / float(total) if total > 0 else 0.0
        rows.append((name, lib, supposed_prob, training_prob, asset))
    rows.sort(key=lambda x: x[3], reverse=True)

    print("Background sampling probabilities (UTF-8 safe):")
    print(f"  background_sampling_alpha: {sampling_alpha:.6f}")
    print(f"  background_sampling_floor: {sampling_floor:.6f}")
    print("  Per-asset probabilities:")
    print("    supposed_probability = raw manuscripts_with_images_count ratio (split by same-library variants)")
    print("    training_probability = smoothed sampling probability used by training")
    for name, _lib, supposed_prob, training_prob, _ in rows:
        print(
            f"    {name}: supposed_probability={supposed_prob:.6f} ({supposed_prob*100:.2f}%), "
            f"training_probability={training_prob:.6f} ({training_prob*100:.2f}%)"
        )

    n = len(rows)
    cols = 4
    n_rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(n_rows, cols, figsize=(5 * cols, 3.8 * n_rows))
    axes_flat = list(np.array(axes).ravel())

    for ax, (name, _lib, supposed_prob, training_prob, asset) in zip(axes_flat, rows):
        image = asset["image"] if isinstance(asset, dict) else None
        if image is None:
            ax.axis("off")
            continue
        if isinstance(image, Image.Image):
            ax.imshow(image)
        ax.set_title(
            f"{name}\nsupposed={supposed_prob:.4f} | train={training_prob:.4f}",
            fontsize=8,
        )
        ax.axis("off")

    for ax in axes_flat[len(rows):]:
        ax.axis("off")

    fig.tight_layout()
    out_path = out_dir / "backgrounds_with_probabilities.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved preview: {out_path}")


if __name__ == "__main__":
    main()
