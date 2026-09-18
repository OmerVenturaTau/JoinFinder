"""Optimizer grouping for pretrained and newly initialized word components."""

from __future__ import annotations

import math

import torch


def build_optimizer_param_groups(
    model: torch.nn.Module,
    loss_module: torch.nn.Module,
    *,
    base_lr: float,
    alephbert_lr: float,
    new_word_lr: float,
) -> list[dict]:
    groups: dict[str, dict] = {
        "base": {"params": [], "lr": float(base_lr), "name": "base"},
        "alephbert": {"params": [], "lr": float(alephbert_lr), "name": "alephbert"},
        "new_word": {"params": [], "lr": float(new_word_lr), "name": "new_word"},
    }
    seen: set[int] = set()

    for name, parameter in model.named_parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        canonical = name.removeprefix("module.")
        if canonical.startswith("word_branch.line_encoder.encoder.transformer."):
            group = "alephbert"
        elif canonical.startswith("word_branch."):
            group = "new_word"
        else:
            group = "base"
        groups[group]["params"].append(parameter)

    for name, parameter in loss_module.named_parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        group = "new_word" if name.startswith("word_arcface.") else "base"
        groups[group]["params"].append(parameter)

    return [group for group in groups.values() if group["params"]]


def build_ratio_preserving_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_epochs: int,
    base_lr: float,
    eta_min: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Apply one cosine multiplier to every LR group, preserving LR ratios."""
    floor = max(0.0, min(1.0, float(eta_min) / max(float(base_lr), 1e-12)))

    def multiplier(epoch: int) -> float:
        progress = min(max(float(epoch) / max(1, int(total_epochs)), 0.0), 1.0)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)
