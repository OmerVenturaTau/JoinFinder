"""Checkpoint introspection helpers for offline analysis scripts.

These helpers must run BEFORE importing ``models`` or ``train.dataset`` so the
inferred architecture knobs can be patched into ``system`` while module-level
constants are still being bound. Importing this module must therefore not pull
in any project module that snapshots ``system`` constants at import time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch


_TILE_BRANCH_PREFIXES = ("tile_branch.", "module.tile_branch.")
_GLYPH_BRANCH_PREFIXES = ("glyph_branch.", "module.glyph_branch.")
_WORD_BRANCH_PREFIXES = ("word_branch.", "module.word_branch.")
_TILE_QUERY_TOKENS_SUFFIX = "tile_branch.set_summarizer.query_tokens"
_GLYPH_QUERY_TOKENS_SUFFIX = "glyph_branch.set_summarizer.query_tokens"
_WORD_QUERY_TOKENS_SUFFIX = "word_branch.word_set_summarizer.query_tokens"


@dataclass
class CheckpointInspection:
    """Authoritative summary of a saved training checkpoint.

    All fields are optional so the caller can decide what to do with missing
    information; the helpers below tolerate raw state dicts saved without a
    ``model_config`` wrapper.
    """

    path: str
    raw: Any = None
    state_dict: Dict[str, Any] = field(default_factory=dict)
    model_config: Dict[str, Any] = field(default_factory=dict)
    training_mode: Optional[str] = None
    num_classes: Optional[int] = None
    tile_size: Optional[int] = None
    use_visual_mod: Optional[bool] = None
    use_char_mod: Optional[bool] = None
    use_word_mod: Optional[bool] = None
    tile_summary_tokens: Optional[int] = None
    glyph_summary_tokens: Optional[int] = None
    word_summary_tokens: Optional[int] = None
    d_model: Optional[int] = None
    latent_dim: Optional[int] = None
    symmetric_branch_dim: Optional[int] = None
    use_branch_adapters_and_summarizers: Optional[bool] = None
    symmetric_reliability_hidden_dim: Optional[int] = None
    fusion_dim_feedforward: Optional[int] = None
    tile_dim_feedforward: Optional[int] = None
    fusion_method: Optional[str] = None
    fusion_num_layers: Optional[int] = None
    tile_branch_transformer_layers: Optional[int] = None
    tile_summarizer_cross_attn_layers: Optional[int] = None
    glyph_summarizer_cross_attn_layers: Optional[int] = None


def _state_dict_from_raw(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        for key in ("state_dict", "model_state_dict"):
            sd = raw.get(key)
            if isinstance(sd, dict) and sd:
                return sd
        if all(hasattr(v, "shape") or torch.is_tensor(v) for v in raw.values() if v is not None):
            return raw
        return {}
    if hasattr(raw, "state_dict") and callable(raw.state_dict):
        return raw.state_dict()
    return {}


def load_training_checkpoint_for_evaluation(
    model: torch.nn.Module,
    combined_loss: torch.nn.Module,
    checkpoint_path: str,
    *,
    device: Optional[torch.device] = None,
    logger: Any = None,
) -> Dict[str, Any]:
    """Restore model and loss-module weights from a training checkpoint."""
    map_location = device if device is not None else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint {checkpoint_path!r} does not contain a 'state_dict'.")

    base_model = model.module if hasattr(model, "module") else model
    base_model.load_state_dict(checkpoint["state_dict"])

    loss_state = checkpoint.get("combined_loss_state_dict")
    if loss_state is None:
        if logger is not None:
            logger.warning(
                "[Checkpoint] %s has no combined_loss_state_dict; final loss/ArcFace "
                "metrics will use the current loss-module weights.",
                checkpoint_path,
            )
    else:
        incompatible = combined_loss.load_state_dict(loss_state, strict=False)
        if logger is not None and (incompatible.missing_keys or incompatible.unexpected_keys):
            logger.warning(
                "[Checkpoint] loaded combined_loss_state_dict with missing keys=%s, unexpected keys=%s",
                incompatible.missing_keys,
                incompatible.unexpected_keys,
            )

    return checkpoint


def _has_branch(state_dict: Dict[str, Any], prefixes: Tuple[str, ...]) -> bool:
    return any(k.startswith(p) for k in state_dict for p in prefixes)


def _infer_glyph_summary_tokens(state_dict: Dict[str, Any]) -> Optional[int]:
    for key, value in state_dict.items():
        if key.endswith(_GLYPH_QUERY_TOKENS_SUFFIX) and hasattr(value, "shape") and len(value.shape) >= 2:
            return int(value.shape[1])
    return None


def _infer_tile_summary_tokens(state_dict: Dict[str, Any]) -> Optional[int]:
    for key, value in state_dict.items():
        if key.endswith(_TILE_QUERY_TOKENS_SUFFIX) and hasattr(value, "shape") and len(value.shape) >= 2:
            return int(value.shape[1])
    # Recreate the old direct-tile path for checkpoints that contain a tile
    # branch but predate learned summary queries.
    if state_dict and _has_branch(state_dict, _TILE_BRANCH_PREFIXES):
        return 0
    return None


def _infer_word_summary_tokens(state_dict: Dict[str, Any]) -> Optional[int]:
    for key, value in state_dict.items():
        if key.endswith(_WORD_QUERY_TOKENS_SUFFIX) and hasattr(value, "shape") and len(value.shape) >= 2:
            return int(value.shape[1])
    return None


def _tensor_by_key_suffix(state_dict: Dict[str, Any], suffix: str) -> Optional[torch.Tensor]:
    for key, value in state_dict.items():
        if key.endswith(suffix) and hasattr(value, "shape"):
            return value
    return None


def _infer_d_model(state_dict: Dict[str, Any]) -> Optional[int]:
    """Infer fusion width from CLS token or tile token-enrichment projection."""
    cls = _tensor_by_key_suffix(state_dict, "fusion.cls_token")
    if cls is not None and len(cls.shape) >= 1:
        return int(cls.shape[-1])
    proj = _tensor_by_key_suffix(state_dict, "tile_branch.token_enrichment.proj.weight")
    if proj is not None and len(proj.shape) == 2:
        return int(proj.shape[0])
    return None


def _infer_latent_dim(state_dict: Dict[str, Any]) -> Optional[int]:
    """Infer exported embedding size from the last linear in PerceiverHead.latent_proj."""
    out = _tensor_by_key_suffix(state_dict, "head.latent_proj.3.weight")
    if out is not None and len(out.shape) >= 1:
        return int(out.shape[0])
    # SymmetricRetrievalHead does not transform the retrieval embedding; its
    # classifier input width is therefore the exported latent dimension.
    classifier = _tensor_by_key_suffix(state_dict, "head.classifier.0.weight")
    if classifier is not None and len(classifier.shape) == 2:
        return int(classifier.shape[1])
    return None


def _infer_symmetric_branch_dim(state_dict: Dict[str, Any]) -> Optional[int]:
    adapter = _tensor_by_key_suffix(state_dict, "fusion.adapters.tile.1.weight")
    if adapter is None:
        adapter = _tensor_by_key_suffix(state_dict, "fusion.adapters.glyph.1.weight")
    if adapter is None:
        adapter = _tensor_by_key_suffix(state_dict, "fusion.adapters.word.1.weight")
    if adapter is not None and len(adapter.shape) == 2:
        return int(adapter.shape[0])
    return None


def _infer_branch_adapters_and_summarizers(state_dict: Dict[str, Any]) -> bool:
    """Detect the optional compressed/summarized symmetric branch path."""
    return any(
        key.startswith(("fusion.adapters.", "module.fusion.adapters."))
        or key.endswith(_TILE_QUERY_TOKENS_SUFFIX)
        or key.endswith(_GLYPH_QUERY_TOKENS_SUFFIX)
        or key.endswith(_WORD_QUERY_TOKENS_SUFFIX)
        for key in state_dict
    )


def _infer_symmetric_reliability_hidden_dim(state_dict: Dict[str, Any]) -> Optional[int]:
    hidden = _tensor_by_key_suffix(state_dict, "fusion.reliability.0.weight")
    if hidden is not None and len(hidden.shape) == 2:
        return int(hidden.shape[0])
    return None


def _infer_transformer_num_layers(state_dict: Dict[str, Any], *, branch: str) -> Optional[int]:
    """Count TransformerEncoder layers from ``*.layers.N.*`` key suffixes."""
    if branch == "fusion":
        prefix = "fusion.transformer.layers."
    elif branch == "tile":
        prefix = "tile_branch.set_transformer.transformer.layers."
    else:
        return None
    layer_indices: list[int] = []
    for key in state_dict:
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix) :]
        layer_idx = rest.split(".", 1)[0]
        if layer_idx.isdigit():
            layer_indices.append(int(layer_idx))
    return (max(layer_indices) + 1) if layer_indices else None


def _infer_fusion_method(state_dict: Dict[str, Any]) -> Optional[str]:
    if any(
        k.startswith("fusion.adapters.") or k.startswith("fusion.reliability.")
        for k in state_dict
    ):
        return "symmetric"
    if any(k.startswith("fusion.latent_queries") for k in state_dict):
        return "perceiver"
    if any(k.startswith("fusion.transformer.") or k.endswith("fusion.cls_token") for k in state_dict):
        return "transformer"
    if any(k.startswith("fusion.vlad.") for k in state_dict):
        return "vlad"
    return None


def _infer_glyph_summarizer_cross_attn_layers(state_dict: Dict[str, Any]) -> Optional[int]:
    if any(k.startswith("glyph_branch.set_summarizer.cross_attn.") for k in state_dict):
        return 1
    layer_indices: list[int] = []
    prefix = "glyph_branch.set_summarizer.cross_attn_layers."
    for key in state_dict:
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix) :]
        layer_idx = rest.split(".", 1)[0]
        if layer_idx.isdigit():
            layer_indices.append(int(layer_idx))
    return (max(layer_indices) + 1) if layer_indices else None


def _infer_tile_summarizer_cross_attn_layers(state_dict: Dict[str, Any]) -> Optional[int]:
    layer_indices: list[int] = []
    prefix = "tile_branch.set_summarizer.cross_attn_layers."
    for key in state_dict:
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix) :]
        layer_idx = rest.split(".", 1)[0]
        if layer_idx.isdigit():
            layer_indices.append(int(layer_idx))
    return (max(layer_indices) + 1) if layer_indices else None


def remap_legacy_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Rename a few legacy glyph-summarizer keys so older checkpoints load cleanly."""
    remapped = dict(state_dict)
    legacy_cross_attn = "glyph_branch.set_summarizer.cross_attn."
    new_cross_attn = "glyph_branch.set_summarizer.cross_attn_layers.0."
    for key in list(state_dict):
        if key.startswith(legacy_cross_attn):
            remapped[new_cross_attn + key[len(legacy_cross_attn) :]] = state_dict[key]
            del remapped[key]
    legacy_norm = "glyph_branch.set_summarizer.norm."
    new_norm = "glyph_branch.set_summarizer.out_norm."
    for key in list(remapped):
        if key.startswith(legacy_norm):
            remapped[new_norm + key[len(legacy_norm) :]] = remapped[key]
            del remapped[key]
    return remapped


def without_classification_head(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Drop task-specific classifier weights before transfer learning.

    ``strict=False`` does not ignore tensor shape mismatches.  A checkpoint
    trained with a different number of manuscript classes therefore cannot be
    loaded safely unless the final classifier is removed first.
    """
    classifier_prefixes = (
        "head.classifier.",
        "module.head.classifier.",
        "head.arcface_head.",
        "module.head.arcface_head.",
    )
    return {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(classifier_prefixes)
    }


def _infer_transformer_dim_feedforward(state_dict: Dict[str, Any], *, branch: str) -> Optional[int]:
    """Infer TransformerEncoderLayer FFN width from the first layer's linear1."""
    if branch == "fusion":
        suffix = "fusion.transformer.layers.0.linear1.weight"
    elif branch == "tile":
        suffix = "tile_branch.set_transformer.transformer.layers.0.linear1.weight"
    else:
        return None
    linear1 = _tensor_by_key_suffix(state_dict, suffix)
    if linear1 is not None and len(linear1.shape) == 2:
        return int(linear1.shape[0])
    return None
    return None


def inspect_checkpoint(checkpoint_path: str) -> CheckpointInspection:
    """Load a checkpoint on CPU and infer architecture knobs.

    Inference precedence: ``model_config`` from the saved dict wins; we fall
    back to scanning the state dict only when the field is missing. This keeps
    new checkpoints authoritative while still supporting older ones that
    predated the ``model_config`` wrapper.
    """
    info = CheckpointInspection(path=checkpoint_path)
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return info

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    info.raw = raw
    info.state_dict = _state_dict_from_raw(raw)

    if isinstance(raw, dict):
        cfg = raw.get("model_config")
        if isinstance(cfg, dict):
            info.model_config = cfg
            if "num_classes" in cfg:
                info.num_classes = int(cfg["num_classes"])
            if "tile_size" in cfg:
                info.tile_size = int(cfg["tile_size"])
            if "use_visual_mod" in cfg:
                info.use_visual_mod = bool(cfg["use_visual_mod"])
            if "use_char_mod" in cfg:
                info.use_char_mod = bool(cfg["use_char_mod"])
            if "use_word_mod" in cfg:
                info.use_word_mod = bool(cfg["use_word_mod"])
            if "d_model" in cfg:
                info.d_model = int(cfg["d_model"])
            if "latent_dim" in cfg:
                info.latent_dim = int(cfg["latent_dim"])
            if "tile_summary_tokens" in cfg:
                info.tile_summary_tokens = int(cfg["tile_summary_tokens"])
            if "glyph_summary_tokens" in cfg:
                info.glyph_summary_tokens = int(cfg["glyph_summary_tokens"])
            if "word_summary_tokens" in cfg:
                info.word_summary_tokens = int(cfg["word_summary_tokens"])
            if "symmetric_branch_dim" in cfg and cfg["symmetric_branch_dim"] is not None:
                info.symmetric_branch_dim = int(cfg["symmetric_branch_dim"])
            if "use_branch_adapters_and_summarizers" in cfg:
                info.use_branch_adapters_and_summarizers = bool(
                    cfg["use_branch_adapters_and_summarizers"]
                )
            if "tile_summarizer_cross_attn_layers" in cfg:
                info.tile_summarizer_cross_attn_layers = int(
                    cfg["tile_summarizer_cross_attn_layers"]
                )
            fusion_cfg = cfg.get("fusion")
            if isinstance(fusion_cfg, dict):
                if info.d_model is None and "d_model" in fusion_cfg:
                    info.d_model = int(fusion_cfg["d_model"])
                if info.latent_dim is None and "latent_dim" in fusion_cfg:
                    info.latent_dim = int(fusion_cfg["latent_dim"])
                if (
                    info.symmetric_branch_dim is None
                    and "symmetric_branch_dim" in fusion_cfg
                ):
                    info.symmetric_branch_dim = int(fusion_cfg["symmetric_branch_dim"])
        tm = raw.get("training_mode")
        if isinstance(tm, str) and tm:
            info.training_mode = tm

    if info.use_visual_mod is None:
        info.use_visual_mod = _has_branch(info.state_dict, _TILE_BRANCH_PREFIXES) or None
    if info.use_char_mod is None:
        info.use_char_mod = _has_branch(info.state_dict, _GLYPH_BRANCH_PREFIXES) or None
    if info.use_word_mod is None:
        info.use_word_mod = _has_branch(info.state_dict, _WORD_BRANCH_PREFIXES) or None

    if info.tile_summary_tokens is None:
        info.tile_summary_tokens = _infer_tile_summary_tokens(info.state_dict)
    if info.glyph_summary_tokens is None:
        info.glyph_summary_tokens = _infer_glyph_summary_tokens(info.state_dict)
    if info.word_summary_tokens is None:
        info.word_summary_tokens = _infer_word_summary_tokens(info.state_dict)
    if info.d_model is None:
        info.d_model = _infer_d_model(info.state_dict)
    if info.latent_dim is None:
        info.latent_dim = _infer_latent_dim(info.state_dict)
    if info.symmetric_branch_dim is None:
        info.symmetric_branch_dim = _infer_symmetric_branch_dim(info.state_dict)
    if info.use_branch_adapters_and_summarizers is None:
        info.use_branch_adapters_and_summarizers = (
            _infer_branch_adapters_and_summarizers(info.state_dict)
        )
    info.symmetric_reliability_hidden_dim = _infer_symmetric_reliability_hidden_dim(info.state_dict)
    if info.fusion_dim_feedforward is None:
        info.fusion_dim_feedforward = _infer_transformer_dim_feedforward(info.state_dict, branch="fusion")
    if info.tile_dim_feedforward is None:
        info.tile_dim_feedforward = _infer_transformer_dim_feedforward(info.state_dict, branch="tile")
    info.fusion_method = _infer_fusion_method(info.state_dict)
    info.fusion_num_layers = _infer_transformer_num_layers(info.state_dict, branch="fusion")
    info.tile_branch_transformer_layers = _infer_transformer_num_layers(info.state_dict, branch="tile")
    if info.tile_summarizer_cross_attn_layers is None:
        info.tile_summarizer_cross_attn_layers = _infer_tile_summarizer_cross_attn_layers(
            info.state_dict
        )
    info.glyph_summarizer_cross_attn_layers = _infer_glyph_summarizer_cross_attn_layers(info.state_dict)
    info.state_dict = remap_legacy_state_dict(info.state_dict)
    return info


def apply_inspection_to_system(
    inspection: CheckpointInspection,
    system_module: Any,
    *,
    quiet: bool = False,
) -> Dict[str, Tuple[Any, Any]]:
    """Patch ``system_module`` constants from an inspection result.

    Returns a dict of ``{name: (old_value, new_value)}`` for every constant we
    actually changed, so callers can log or revert. Only fields the inspection
    successfully resolved are touched; unresolved fields are left intact.
    """
    changes: Dict[str, Tuple[Any, Any]] = {}

    def patch(name: str, value: Any) -> None:
        if value is None or not hasattr(system_module, name):
            return
        old = getattr(system_module, name)
        if old == value:
            return
        setattr(system_module, name, value)
        changes[name] = (old, value)

    patch("USE_VISUAL_MOD", inspection.use_visual_mod)
    patch("USE_CHAR_MOD", inspection.use_char_mod)
    patch("USE_WORD_MOD", inspection.use_word_mod)
    patch("TILE_NUM_SUMMARY_TOKENS", inspection.tile_summary_tokens)
    patch("GLYPH_NUM_SUMMARY_TOKENS", inspection.glyph_summary_tokens)
    patch("WORD_NUM_SUMMARY_TOKENS", inspection.word_summary_tokens)
    patch("D_MODEL", inspection.d_model)
    patch("LATENT_DIM", inspection.latent_dim)
    patch("SYMMETRIC_BRANCH_DIM", inspection.symmetric_branch_dim)
    patch(
        "USE_BRANCH_ADAPTERS_AND_SUMMARIZERS",
        inspection.use_branch_adapters_and_summarizers,
    )
    patch("SYMMETRIC_RELIABILITY_HIDDEN_DIM", inspection.symmetric_reliability_hidden_dim)
    patch("TRANSFORMER_DIM_FEEDFORWARD", inspection.fusion_dim_feedforward)
    patch("TILE_BRANCH_TRANSFORMER_DIM_FEEDFORWARD", inspection.tile_dim_feedforward)
    patch("FUSION_METHOD", inspection.fusion_method)
    patch("TRANSFORMER_NUM_LAYERS", inspection.fusion_num_layers)
    patch("TILE_BRANCH_TRANSFORMER_LAYERS", inspection.tile_branch_transformer_layers)
    patch("TILE_SUMMARIZER_CROSS_ATTN_LAYERS", inspection.tile_summarizer_cross_attn_layers)
    patch("GLYPH_SUMMARIZER_CROSS_ATTN_LAYERS", inspection.glyph_summarizer_cross_attn_layers)

    if changes and not quiet:
        joined = ", ".join(f"{k}: {old!r} -> {new!r}" for k, (old, new) in changes.items())
        print(f"[Checkpoint] Patched system constants from checkpoint: {joined}")
    return changes


def load_state_dict_with_report(
    model: torch.nn.Module,
    state_dict: Dict[str, Any],
    *,
    label: str = "Checkpoint",
    head_prefix: str = "head.",
    log_first_n: int = 10,
) -> Tuple[list, list]:
    """Wrap ``load_state_dict(strict=False)`` and surface key mismatches.

    Returns the ``(missing_keys, unexpected_keys)`` tuple from PyTorch. Keys
    under ``head_prefix`` are reported separately because the classifier head
    is routinely re-initialized when ``num_classes`` differs between the
    checkpoint and the current evaluation set; surfacing only non-head misses
    keeps the noise floor low while still flagging real architecture drift.
    """
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = list(missing)
    unexpected = list(unexpected)
    if not missing and not unexpected:
        print(f"[{label}] state_dict loaded with no missing/unexpected keys.")
        return missing, unexpected

    non_head_missing = [k for k in missing if not k.startswith(head_prefix)]
    head_missing = [k for k in missing if k.startswith(head_prefix)]
    print(
        f"[{label}] load_state_dict(strict=False): "
        f"missing={len(missing)} (non-head {len(non_head_missing)}, head {len(head_missing)}), "
        f"unexpected={len(unexpected)}."
    )
    if non_head_missing:
        print(f"[{label}] Non-head missing first{log_first_n}: {non_head_missing[:log_first_n]}")
    if unexpected:
        print(f"[{label}] Unexpected first{log_first_n}: {unexpected[:log_first_n]}")
    return missing, unexpected


def warn_if_num_classes_mismatch(
    inspection: CheckpointInspection,
    local_num_classes: int,
    *,
    label: str = "Checkpoint",
) -> None:
    """Print a warning if local class count disagrees with the checkpoint.

    The classifier head is irrelevant for latent/embedding-only analyses, but
    a silent mismatch is a strong hint that the wrong checkpoint was selected
    for the wrong split — worth flagging.
    """
    if inspection.num_classes is None:
        return
    if int(inspection.num_classes) == int(local_num_classes):
        return
    print(
        f"[{label}] num_classes mismatch: checkpoint={inspection.num_classes} "
        f"vs local split={local_num_classes}. Classifier head will be re-initialized "
        f"(latents are unaffected, but verify this is the intended checkpoint)."
    )


def warn_if_tile_size_mismatch(
    inspection: CheckpointInspection,
    runtime_tile_size: int,
    *,
    label: str = "Checkpoint",
) -> None:
    """Print a warning if runtime tile size differs from the trained tile size."""
    if inspection.tile_size is None:
        return
    if int(inspection.tile_size) == int(runtime_tile_size):
        return
    print(
        f"[{label}] tile_size mismatch: checkpoint trained at {inspection.tile_size}px "
        f"but runtime tiling is {runtime_tile_size}px. Distribution shift expected."
    )
