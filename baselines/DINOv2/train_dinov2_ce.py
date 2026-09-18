#!/usr/bin/env python3
"""Convenience entry point for the whole-page DINOv2 CE baseline."""

from __future__ import annotations

import sys
from pathlib import Path


SHARED_TRAINER_DIR = Path(__file__).resolve().parents[1] / "ConvNeXt"
if str(SHARED_TRAINER_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_TRAINER_DIR))

import train_convnext_ce  # noqa: E402


if __name__ == "__main__":
    if "--backbone" not in sys.argv:
        sys.argv[1:1] = ["--backbone", "dinov2"]
    raise SystemExit(train_convnext_ce.main())
