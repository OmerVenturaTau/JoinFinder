"""
Find branch-only label flips caused by tile/glyph augmentations.

For each validation image:
- run the checkpoint with only the tile branch input
- run the checkpoint with only the glyph branch input
- apply deterministic augmentations to all tiles or all glyphs of that image at once
- rerun the same branch-only prediction
- print and save every case where the predicted label changes

The augmentation is applied consistently to all items in the branch stream:
all tiles get the same augmentation spec together, and all glyphs get the same
augmentation spec together.

Example:
  python debug_tools/Augmentations/branch_augmentation_label_flips.py \
      --checkpoint /path/to/best_model.pth \
      --split val \
      --batch-size 2
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-matplotlib-cache")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TVF


PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "system.py").exists())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import system  # noqa: E402

# Keep this debug focused on the two visual branches.
system.USE_VISUAL_MOD = True
system.USE_CHAR_MOD = True
system.USE_WORD_MOD = False

try:
    import psycopg2  # noqa: F401
except ModuleNotFoundError:
    psycopg2_stub = types.ModuleType("psycopg2")

    def _missing_psycopg2_connect(*_args: Any, **_kwargs: Any) -> None:
        raise ModuleNotFoundError("psycopg2 is required for DB-backed split mode. Install psycopg2 in this environment.")

    psycopg2_stub.connect = _missing_psycopg2_connect  # type: ignore[attr-defined]
    sys.modules["psycopg2"] = psycopg2_stub

from models import MultiModal  # noqa: E402
from losses.combined_loss import CombinedLoss  # noqa: E402
from system import (  # noqa: E402
    ARCFACE_MARGIN,
    ARCFACE_SCALE,
    ARCFACE_WEIGHT,
    BASE_DIR,
    CE_WEIGHT,
    D_MODEL,
    GLYPH_AUX_LOSS_WEIGHT,
    LABEL_HEAD,
    LATENT_DIM,
    LATENT_SPARSITY_WEIGHT,
    MAX_TILES_EVAL,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    TILE_AUX_LOSS_WEIGHT,
    WORD_AUX_LOSS_WEIGHT,
)
from tasks.label_heads.base import build_label_maps  # noqa: E402
from tasks.label_heads.factory import create_label_head  # noqa: E402
from train.dataset import ManuscriptDataset, tile_collate_with_padding  # noqa: E402
from train.split_data import build_splits  # noqa: E402


MEAN = torch.tensor(NORMALIZE_MEAN, dtype=torch.float32).view(1, 1, 3, 1, 1)
STD = torch.tensor(NORMALIZE_STD, dtype=torch.float32).view(1, 1, 3, 1, 1)


@dataclass(frozen=True)
class AugmentationSpec:
    name: str
    fn: Callable[[torch.Tensor], torch.Tensor]


def build_dataset(split: str) -> Tuple[ManuscriptDataset, Dict[str, int], Dict[int, str], Dict[str, Any]]:
    splits, split_stats = build_splits(BASE_DIR)
    label_head = create_label_head(name=LABEL_HEAD)
    flat = label_head.flatten_splits(splits)
    if split == "train":
        paths, labels, xmls = flat.train_paths, flat.train_labels, flat.train_xmls
    elif split == "val":
        paths, labels, xmls = flat.val_paths, flat.val_labels, flat.val_xmls
    elif split == "test":
        paths, labels, xmls = flat.test_paths, flat.test_labels, flat.test_xmls
    else:
        raise ValueError(f"Unsupported split: {split}")

    all_labels = flat.train_labels + flat.val_labels + flat.test_labels
    label2idx, idx2label = build_label_maps(all_labels=all_labels)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )
    dataset = ManuscriptDataset(
        paths,
        labels,
        transform,
        label2idx,
        xml_paths=xmls,
        max_tiles_per_image=MAX_TILES_EVAL,
        split=split,
    )
    return dataset, label2idx, idx2label, split_stats


def read_checkpoint_training_mode(checkpoint_path: str) -> Optional[str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        mode = checkpoint.get("training_mode")
        if mode in {"pretrain", "finetune", "demo"}:
            return str(mode)
    return None


def load_model_and_loss(
    checkpoint_path: str,
    num_classes: int,
    device: torch.device,
    *,
    allow_raw_logits: bool,
) -> Tuple[MultiModal, Optional[CombinedLoss]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}
    model = MultiModal(
        num_classes=int(config.get("num_classes", num_classes)),
        use_visual_mod=bool(config.get("use_visual_mod", True)),
        use_char_mod=bool(config.get("use_char_mod", True)),
        # Match the checkpoint architecture exactly. Branch-only evaluation is
        # done by passing None for inactive modalities at forward time.
        use_word_mod=bool(config.get("use_word_mod", False)),
    ).to(device)
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
    if state_dict and all(str(key).startswith("module.") for key in state_dict.keys()):
        state_dict = {str(key)[7:]: value for key, value in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()
    if hasattr(model, "modality_dropout_enabled"):
        model.modality_dropout_enabled = False
    if hasattr(model, "token_subsample_enabled"):
        model.token_subsample_enabled = False
    print(f"[Model] Loaded checkpoint: {checkpoint_path}")
    print(f"[Model] missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")

    loss_state = None
    if isinstance(checkpoint, dict):
        loss_state = (
            checkpoint.get("combined_loss_state_dict")
            or checkpoint.get("loss_state_dict")
            or checkpoint.get("criterion_state_dict")
        )
    combined_loss = None
    if loss_state is not None:
        combined_loss = CombinedLoss(
            num_classes=int(config.get("num_classes", num_classes)),
            embedding_dim=LATENT_DIM,
            arcface_weight=ARCFACE_WEIGHT,
            ce_weight=CE_WEIGHT,
            arcface_margin=ARCFACE_MARGIN,
            arcface_scale=ARCFACE_SCALE,
            sparsity_weight=LATENT_SPARSITY_WEIGHT,
            tile_aux_weight=TILE_AUX_LOSS_WEIGHT,
            glyph_aux_weight=GLYPH_AUX_LOSS_WEIGHT,
            word_aux_weight=WORD_AUX_LOSS_WEIGHT,
            aux_embedding_dim=D_MODEL,
        ).to(device)
        loss_missing, loss_unexpected = combined_loss.load_state_dict(loss_state, strict=False)
        combined_loss.eval()
        print(f"[Loss] Loaded loss state: missing_keys={len(loss_missing)}, unexpected_keys={len(loss_unexpected)}")
    elif ARCFACE_WEIGHT > 0:
        message = (
            "This checkpoint does not contain CombinedLoss/ArcFace weights. "
            "Training validation accuracy used ArcFace effective logits, but this checkpoint only has model weights. "
            "With CE_WEIGHT=0, raw model logits are not the trained classifier and may be near-random. "
            "Use a checkpoint saved after train/trainer.py was patched to include combined_loss_state_dict, "
            "or pass --allow-raw-logits for the old diagnostic behavior."
        )
        if not allow_raw_logits:
            raise RuntimeError(message)
        print(f"[Warning] {message}")
    return model, combined_loss


def to_pixel(x: torch.Tensor) -> torch.Tensor:
    mean = MEAN.to(device=x.device, dtype=x.dtype)
    std = STD.to(device=x.device, dtype=x.dtype)
    return (x * std + mean).clamp(0.0, 1.0)


def to_normalized(x: torch.Tensor) -> torch.Tensor:
    mean = MEAN.to(device=x.device, dtype=x.dtype)
    std = STD.to(device=x.device, dtype=x.dtype)
    return (x.clamp(0.0, 1.0) - mean) / std


def flatten_stream(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    bsz, n_items = x.shape[:2]
    return x.reshape(bsz * n_items, *x.shape[2:]), (bsz, n_items)


def unflatten_stream(x: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    bsz, n_items = shape
    return x.reshape(bsz, n_items, *x.shape[1:])


def tensor_gaussian_blur(radius: float) -> Callable[[torch.Tensor], torch.Tensor]:
    def apply(x: torch.Tensor) -> torch.Tensor:
        flat, shape = flatten_stream(x)
        kernel = max(3, int(round(radius * 6)) | 1)
        return unflatten_stream(TVF.gaussian_blur(flat, kernel_size=[kernel, kernel], sigma=[radius, radius]), shape)

    return apply


def tensor_rotate(degrees: float) -> Callable[[torch.Tensor], torch.Tensor]:
    def apply(x: torch.Tensor) -> torch.Tensor:
        flat, shape = flatten_stream(x)
        rotated = TVF.rotate(flat, angle=degrees, interpolation=transforms.InterpolationMode.BILINEAR, fill=1.0)
        return unflatten_stream(rotated, shape)

    return apply


def tensor_down_up(scale: float) -> Callable[[torch.Tensor], torch.Tensor]:
    def apply(x: torch.Tensor) -> torch.Tensor:
        flat, shape = flatten_stream(x)
        height, width = flat.shape[-2:]
        small = (max(4, int(round(height * scale))), max(4, int(round(width * scale))))
        y = F.interpolate(flat, size=small, mode="bicubic", align_corners=False)
        y = F.interpolate(y, size=(height, width), mode="bicubic", align_corners=False)
        return unflatten_stream(y, shape)

    return apply


def tensor_noise(std: float) -> Callable[[torch.Tensor], torch.Tensor]:
    def apply(x: torch.Tensor) -> torch.Tensor:
        return (x + torch.randn_like(x) * std).clamp(0.0, 1.0)

    return apply


def tensor_center_erase(frac: float) -> Callable[[torch.Tensor], torch.Tensor]:
    def apply(x: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        height, width = y.shape[-2:]
        erase_h = max(1, int(round(height * frac)))
        erase_w = max(1, int(round(width * frac)))
        top = (height - erase_h) // 2
        left = (width - erase_w) // 2
        y[..., top : top + erase_h, left : left + erase_w] = 1.0
        return y

    return apply


def tensor_sharpen(x: torch.Tensor) -> torch.Tensor:
    blurred = tensor_gaussian_blur(1.0)(x)
    return (x + 0.8 * (x - blurred)).clamp(0.0, 1.0)


def build_augmentations() -> List[AugmentationSpec]:
    return [
        AugmentationSpec("blur_sigma_0.5", tensor_gaussian_blur(0.5)),
        AugmentationSpec("blur_sigma_1.0", tensor_gaussian_blur(1.0)),
        AugmentationSpec("blur_sigma_2.0", tensor_gaussian_blur(2.0)),
        AugmentationSpec("blur_sigma_3.0", tensor_gaussian_blur(3.0)),
        AugmentationSpec("down_up_0.85x", tensor_down_up(0.85)),
        AugmentationSpec("down_up_0.65x", tensor_down_up(0.65)),
        AugmentationSpec("down_up_0.50x", tensor_down_up(0.50)),
        AugmentationSpec("brightness_0.75", lambda x: TVF.adjust_brightness(x, 0.75)),
        AugmentationSpec("brightness_1.25", lambda x: TVF.adjust_brightness(x, 1.25)),
        AugmentationSpec("contrast_0.70", lambda x: TVF.adjust_contrast(x, 0.70)),
        AugmentationSpec("contrast_1.40", lambda x: TVF.adjust_contrast(x, 1.40)),
        AugmentationSpec("grayscale", lambda x: TVF.rgb_to_grayscale(x, num_output_channels=3)),
        AugmentationSpec("rotate_minus_5deg", tensor_rotate(-5.0)),
        AugmentationSpec("rotate_plus_5deg", tensor_rotate(5.0)),
        AugmentationSpec("gaussian_noise_0.03", tensor_noise(0.03)),
        AugmentationSpec("gaussian_noise_0.07", tensor_noise(0.07)),
        AugmentationSpec("center_erase_0.18", tensor_center_erase(0.18)),
        AugmentationSpec("invert", lambda x: 1.0 - x),
        AugmentationSpec("sharpen", tensor_sharpen),
    ]


def apply_augmentation(x_normalized: torch.Tensor, spec: AugmentationSpec) -> torch.Tensor:
    pixels = to_pixel(x_normalized)
    augmented = spec.fn(pixels).clamp(0.0, 1.0)
    return to_normalized(augmented)


@torch.no_grad()
def predict_branch(
    model: MultiModal,
    combined_loss: Optional[CombinedLoss],
    *,
    branch: str,
    tiles: torch.Tensor,
    tile_mask: torch.Tensor,
    tile_coords: torch.Tensor,
    tile_page_segments: torch.Tensor,
    glyphs: torch.Tensor,
    glyph_mask: torch.Tensor,
    glyph_coords: torch.Tensor,
    char_class_ids: torch.Tensor,
    paths: List[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_indices = torch.arange(tiles.shape[0], dtype=torch.long, device=device)
    use_amp = device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        if branch == "tile":
            logits, latent, aux_latents = model(
                tiles=tiles,
                tile_coords=tile_coords,
                tile_valid_mask=tile_mask,
                tile_page_segments=tile_page_segments,
                glyph_patches=None,
                glyph_coords=None,
                glyph_valid_mask=None,
                char_class_ids=None,
                words=None,
                word_metadata=None,
                paths=paths,
                batch_element_indices=batch_indices,
                return_aux_latents=False,
            )
        elif branch == "glyph":
            logits, latent, aux_latents = model(
                tiles=None,
                tile_coords=None,
                tile_valid_mask=None,
                tile_page_segments=None,
                glyph_patches=glyphs,
                glyph_coords=glyph_coords,
                glyph_valid_mask=glyph_mask,
                char_class_ids=char_class_ids,
                words=None,
                word_metadata=None,
                paths=paths,
                batch_element_indices=batch_indices,
                return_aux_latents=False,
            )
        else:
            raise ValueError(f"Unknown branch: {branch}")
    if combined_loss is not None and combined_loss.arcface is not None:
        normalized_latent = F.normalize(latent.float(), dim=1, eps=1e-8)
        weight = F.normalize(combined_loss.arcface.weight.float(), p=2, dim=1)
        effective_logits = F.linear(normalized_latent, weight) * float(combined_loss.arcface.scale)
    else:
        effective_logits = logits.float()
    probs = torch.softmax(effective_logits, dim=1)
    conf, pred = probs.max(dim=1)
    return pred.detach().cpu(), conf.detach().cpu()


def select_batch(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return x.index_select(0, indices.to(device=x.device))


def decode_label(idx2label: Dict[int, str], label_idx: int) -> str:
    return str(idx2label.get(int(label_idx), int(label_idx)))


def write_rows(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_manifest_row(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def stable_seed_offset(*parts: str) -> int:
    text = "::".join(parts)
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(text)) % 1_000_003


def safe_filename(text: str, max_len: int = 90) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return cleaned[:max_len] if cleaned else "sample"


def image_np(tensor_chw: torch.Tensor) -> Any:
    return tensor_chw.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def save_flip_sample(
    *,
    output_dir: str,
    sample_idx: int,
    branch: str,
    augmentation: str,
    path: str,
    true_label: str,
    before: str,
    after: str,
    before_conf: float,
    after_conf: float,
    original_stream: torch.Tensor,
    augmented_stream: torch.Tensor,
    valid_mask: torch.Tensor,
    local_idx: int,
    max_items: int,
) -> str:
    import matplotlib.pyplot as plt

    sample_dir = os.path.join(output_dir, "flip_samples", branch, safe_filename(augmentation))
    os.makedirs(sample_dir, exist_ok=True)

    valid_indices = valid_mask[local_idx].detach().cpu().nonzero(as_tuple=True)[0].tolist()
    valid_indices = valid_indices[:max_items]
    if not valid_indices:
        return ""

    orig_pixels = to_pixel(original_stream[local_idx : local_idx + 1]).detach().cpu()[0]
    aug_pixels = to_pixel(augmented_stream[local_idx : local_idx + 1]).detach().cpu()[0]

    n = len(valid_indices)
    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 4.5), squeeze=False)
    for col, item_idx in enumerate(valid_indices):
        axes[0][col].imshow(image_np(orig_pixels[item_idx]))
        axes[0][col].axis("off")
        axes[0][col].set_title(f"{branch} {item_idx}", fontsize=8)
        axes[1][col].imshow(image_np(aug_pixels[item_idx]))
        axes[1][col].axis("off")

    axes[0][0].set_ylabel("original", fontsize=9)
    axes[1][0].set_ylabel("augmented", fontsize=9)
    fig.suptitle(
        f"{branch} | {augmentation}\n"
        f"true={true_label} | pred_before={before} (conf={before_conf:.3f})"
        f" -> pred_after={after} (conf={after_conf:.3f})\n"
        f"{path}",
        fontsize=10,
    )
    fig.tight_layout()

    stem = safe_filename(f"{sample_idx:03d}_{branch}_{augmentation}_{Path(path).stem}")
    out_path = os.path.join(sample_dir, f"{stem}.png")
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--training-mode", choices=["pretrain", "finetune", "demo"], default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-images", type=int, default=0, help="Optional cap for quick debugging.")
    parser.add_argument("--branches", nargs="+", choices=["tile", "glyph"], default=["tile", "glyph"])
    parser.add_argument(
        "--only-correct-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only report/save cases where the original branch-only prediction was correct before augmentation.",
    )
    parser.add_argument(
        "--save-flip-samples",
        type=int,
        default=10,
        help="Save this many successful flip previews per branch, using distinct augmentation names when possible.",
    )
    parser.add_argument("--sample-items", type=int, default=8, help="Max tiles/glyphs shown per saved flip preview.")
    parser.add_argument(
        "--stop-after-samples",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop the run once every requested branch has saved --save-flip-samples diverse previews.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--allow-raw-logits",
        action="store_true",
        help="Allow using raw model logits if the checkpoint lacks ArcFace/CombinedLoss weights. Usually misleading when CE_WEIGHT=0.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "outputs" / "branch_augmentation_label_flips"))
    args = parser.parse_args()

    checkpoint_mode = read_checkpoint_training_mode(args.checkpoint)
    effective_training_mode = args.training_mode or checkpoint_mode or getattr(system, "TRAINING_MODE", "pretrain")
    if effective_training_mode not in {"pretrain", "finetune", "demo"}:
        raise ValueError(f"Unsupported training mode: {effective_training_mode}")
    print(
        f"[Config] Using training_mode={effective_training_mode!r} "
        f"(system was {getattr(system, 'TRAINING_MODE', None)!r}, checkpoint says {checkpoint_mode!r})"
    )
    system.TRAINING_MODE = effective_training_mode

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dataset, label2idx, idx2label, split_stats = build_dataset(args.split)
    print(
        f"[Data] split={args.split} images={len(dataset)} classes={len(label2idx)} "
        f"stats={split_stats}"
    )
    if args.limit_images > 0:
        indices = list(range(min(args.limit_images, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, indices)
        dataset.labels = [dataset.dataset.labels[i] for i in indices]  # type: ignore[attr-defined]

    model, combined_loss = load_model_and_loss(
        args.checkpoint,
        num_classes=len(label2idx),
        device=device,
        allow_raw_logits=args.allow_raw_logits,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=tile_collate_with_padding,
        pin_memory=False,
    )
    augmentations = build_augmentations()

    rows: List[Dict[str, Any]] = []
    total_checked = {branch: 0 for branch in args.branches}
    total_flips = {branch: 0 for branch in args.branches}
    saved_flip_samples = {branch: 0 for branch in args.branches}
    saved_flip_augmentations = {branch: set() for branch in args.branches}
    manifest_path = os.path.join(args.out_dir, f"flip_samples_manifest_{args.split}.csv")

    for batch_idx, batch in enumerate(loader):
        if batch is None:
            continue
        (
            tiles,
            tile_mask,
            tile_coords,
            tile_page_segments,
            glyphs,
            glyph_mask,
            glyph_coords,
            _glyph_page_segments,
            char_class_ids,
            _char_metadata,
            _words,
            _word_metadata,
            labels,
            paths,
        ) = batch

        tiles = tiles.to(device)
        tile_mask = tile_mask.to(device)
        tile_coords = tile_coords.to(device)
        tile_page_segments = tile_page_segments.to(device)
        glyphs = glyphs.to(device)
        glyph_mask = glyph_mask.to(device)
        glyph_coords = glyph_coords.to(device)
        char_class_ids = char_class_ids.to(device)
        labels_cpu = labels.detach().cpu()
        paths_list = list(paths)

        if batch_idx % 10 == 0:
            print(f"[Batch] {batch_idx}: processing {len(paths_list)} images")

        for branch in args.branches:
            if branch == "tile" and not bool(tile_mask.any()):
                continue
            if branch == "glyph" and not bool(glyph_mask.any()):
                continue

            base_pred, base_conf = predict_branch(
                model,
                combined_loss,
                branch=branch,
                tiles=tiles,
                tile_mask=tile_mask,
                tile_coords=tile_coords,
                tile_page_segments=tile_page_segments,
                glyphs=glyphs,
                glyph_mask=glyph_mask,
                glyph_coords=glyph_coords,
                char_class_ids=char_class_ids,
                paths=paths_list,
                device=device,
            )
            if args.only_correct_baseline:
                eligible_local = (base_pred == labels_cpu).nonzero(as_tuple=True)[0]
                print(
                    f"[Baseline] batch={batch_idx} branch={branch} "
                    f"correct_before_aug={int(eligible_local.numel())}/{len(paths_list)}"
                )
                if eligible_local.numel() == 0:
                    continue
            else:
                eligible_local = torch.arange(len(paths_list), dtype=torch.long)
                print(
                    f"[Baseline] batch={batch_idx} branch={branch} "
                    f"checking_all={len(paths_list)} correct_before_aug={int((base_pred == labels_cpu).sum().item())}/{len(paths_list)}"
                )

            eligible_device = eligible_local.to(device=device)
            sub_paths = [paths_list[int(i)] for i in eligible_local.tolist()]
            sub_labels_cpu = labels_cpu.index_select(0, eligible_local)
            sub_base_pred = base_pred.index_select(0, eligible_local)
            sub_base_conf = base_conf.index_select(0, eligible_local)
            sub_tiles = select_batch(tiles, eligible_device)
            sub_tile_mask = select_batch(tile_mask, eligible_device)
            sub_tile_coords = select_batch(tile_coords, eligible_device)
            sub_tile_page_segments = select_batch(tile_page_segments, eligible_device)
            sub_glyphs = select_batch(glyphs, eligible_device)
            sub_glyph_mask = select_batch(glyph_mask, eligible_device)
            sub_glyph_coords = select_batch(glyph_coords, eligible_device)
            sub_char_class_ids = select_batch(char_class_ids, eligible_device)

            for spec in augmentations:
                torch.manual_seed(args.seed + batch_idx * 1000 + stable_seed_offset(branch, spec.name))
                if branch == "tile":
                    aug_tiles = apply_augmentation(sub_tiles, spec)
                    aug_glyphs = sub_glyphs
                else:
                    aug_tiles = sub_tiles
                    aug_glyphs = apply_augmentation(sub_glyphs, spec)

                aug_pred, aug_conf = predict_branch(
                    model,
                    combined_loss,
                    branch=branch,
                    tiles=aug_tiles,
                    tile_mask=sub_tile_mask,
                    tile_coords=sub_tile_coords,
                    tile_page_segments=sub_tile_page_segments,
                    glyphs=aug_glyphs,
                    glyph_mask=sub_glyph_mask,
                    glyph_coords=sub_glyph_coords,
                    char_class_ids=sub_char_class_ids,
                    paths=sub_paths,
                    device=device,
                )
                total_checked[branch] += len(sub_paths)

                changed = aug_pred != sub_base_pred
                for sub_idx in changed.nonzero(as_tuple=True)[0].tolist():
                    local_idx = int(eligible_local[sub_idx])
                    true_label = decode_label(idx2label, int(sub_labels_cpu[sub_idx]))
                    before = decode_label(idx2label, int(sub_base_pred[sub_idx]))
                    after = decode_label(idx2label, int(aug_pred[sub_idx]))
                    sample_path = ""
                    should_save_sample = (
                        saved_flip_samples[branch] < args.save_flip_samples
                        and spec.name not in saved_flip_augmentations[branch]
                    )
                    if should_save_sample:
                        if branch == "tile":
                            sample_path = save_flip_sample(
                                output_dir=args.out_dir,
                                sample_idx=saved_flip_samples[branch],
                                branch=branch,
                                augmentation=spec.name,
                                path=paths_list[local_idx],
                                true_label=true_label,
                                before=before,
                                after=after,
                                before_conf=float(sub_base_conf[sub_idx]),
                                after_conf=float(aug_conf[sub_idx]),
                                original_stream=sub_tiles,
                                augmented_stream=aug_tiles,
                                valid_mask=sub_tile_mask,
                                local_idx=sub_idx,
                                max_items=args.sample_items,
                            )
                        else:
                            sample_path = save_flip_sample(
                                output_dir=args.out_dir,
                                sample_idx=saved_flip_samples[branch],
                                branch=branch,
                                augmentation=spec.name,
                                path=paths_list[local_idx],
                                true_label=true_label,
                                before=before,
                                after=after,
                                before_conf=float(sub_base_conf[sub_idx]),
                                after_conf=float(aug_conf[sub_idx]),
                                original_stream=sub_glyphs,
                                augmented_stream=aug_glyphs,
                                valid_mask=sub_glyph_mask,
                                local_idx=sub_idx,
                                max_items=args.sample_items,
                            )
                        if sample_path:
                            manifest_row = {
                                "branch_sample_index": saved_flip_samples[branch],
                                "sample_png": sample_path,
                                "branch": branch,
                                "augmentation": spec.name,
                                "path": paths_list[local_idx],
                                "true_label": true_label,
                                "before_pred": before,
                                "after_pred": after,
                                "before_conf": float(sub_base_conf[sub_idx]),
                                "after_conf": float(aug_conf[sub_idx]),
                            }
                            append_manifest_row(manifest_path, manifest_row)
                            saved_flip_samples[branch] += 1
                            saved_flip_augmentations[branch].add(spec.name)
                            if saved_flip_samples[branch] == args.save_flip_samples:
                                print(
                                    f"[Samples] Reached --save-flip-samples={args.save_flip_samples} for branch={branch}. "
                                    f"Saved {len(saved_flip_augmentations[branch])} distinct successful augmentations. "
                                    f"Preview images are available now under {os.path.join(args.out_dir, 'flip_samples', branch)}"
                                )
                    row = {
                        "branch": branch,
                        "augmentation": spec.name,
                        "path": paths_list[local_idx],
                        "sample_png": sample_path,
                        "true_label": true_label,
                        "before_pred": before,
                        "after_pred": after,
                        "before_conf": float(sub_base_conf[sub_idx]),
                        "after_conf": float(aug_conf[sub_idx]),
                        "before_correct": before == true_label,
                        "after_correct": after == true_label,
                    }
                    rows.append(row)
                    total_flips[branch] += 1
                    print(
                        "[Flip] "
                        f"branch={branch} aug={spec.name} "
                        f"path={paths_list[local_idx]} "
                        f"true={true_label} "
                        f"pred_before={before}(conf={float(sub_base_conf[sub_idx]):.3f})"
                        f" -> pred_after={after}(conf={float(aug_conf[sub_idx]):.3f})"
                        + (f" sample={sample_path}" if sample_path else "")
                    )

            if args.stop_after_samples and all(
                saved_flip_samples[branch_name] >= args.save_flip_samples for branch_name in args.branches
            ):
                print("[Stop] Requested preview sample quota reached for all selected branches.")
                break

        if args.stop_after_samples and all(
            saved_flip_samples[branch_name] >= args.save_flip_samples for branch_name in args.branches
        ):
            break

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"label_flips_{args.split}.csv")
    write_rows(csv_path, rows)
    summary_path = os.path.join(args.out_dir, f"summary_{args.split}.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        for branch in args.branches:
            handle.write(
                f"{branch}: checked={total_checked[branch]}, flips={total_flips[branch]}, "
                f"flip_rate={(total_flips[branch] / max(1, total_checked[branch])):.6f}\n"
            )

    print("[Done]")
    print(f"  flips_csv: {csv_path}")
    print(f"  summary: {summary_path}")
    print(f"  sample_manifest: {manifest_path}")
    print(f"  saved_flip_samples: {saved_flip_samples}")
    for branch in args.branches:
        print(
            f"  {branch}: checked={total_checked[branch]}, flips={total_flips[branch]}, "
            f"flip_rate={(total_flips[branch] / max(1, total_checked[branch])):.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
