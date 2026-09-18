"""
Tile backbone factory.

We keep tile-level encoders behind a small factory so the rest of the tile branch
can stay stable while we experiment with different visual backbones (e.g. DINOv2
ViT vs ConvNeXt).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import timm
import torch.nn as nn


@dataclass(frozen=True)
class TileBackboneSpec:
    model: nn.Module
    feat_dim: int
    name: str
    encoder_type: str


def _infer_feat_dim(model: nn.Module, default: int = 768) -> int:
    # timm usually exposes num_features; ViTs sometimes have embed_dim.
    feat_dim = getattr(model, "num_features", None)
    if feat_dim is None:
        feat_dim = getattr(model, "embed_dim", None)
    return int(feat_dim) if feat_dim is not None else int(default)


def create_tile_backbone(
    *,
    encoder_type: str,
    model_name: str,
    tile_size: int,
    pretrained: bool = True,
) -> TileBackboneSpec:
    """
    Create a timm backbone that returns a single feature vector per tile.

    Notes:
    - We set num_classes=0 so timm returns pre-logits / pooled features.
    - Only pass img_size for ViT-like models; some CNNs don't accept it.
    """
    et = str(encoder_type).lower().strip()
    kwargs: Dict[str, Any] = dict(pretrained=pretrained, num_classes=0)

    if et in {"dinov2", "dino", "vit", "vit_dinov2"}:
        kwargs["img_size"] = tile_size

    model = timm.create_model(model_name, **kwargs)
    feat_dim = _infer_feat_dim(model, default=768)
    return TileBackboneSpec(model=model, feat_dim=feat_dim, name=model_name, encoder_type=et)

