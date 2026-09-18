from __future__ import annotations

from typing import Optional

from .base import LabelHead
from .manuscript_id import ManuscriptIdLabelHead
from .decade import DecadeLabelHead


def create_label_head(*, name: str, period_mapping_csv: Optional[str] = None) -> LabelHead:
    """Factory for task heads.

    Currently supported:
      - manuscript_id
      - decade
    """
    name = (name or "").strip().lower()
    if name in ("manuscript_id", "manuscript", "mid"):
        return ManuscriptIdLabelHead()
    if name in ("decade", "dating"):
        # Uses the default dating table created by the preprocessing scripts.
        return DecadeLabelHead()
    raise ValueError(
        f"Unknown LABEL_HEAD='{name}'. Supported: manuscript_id, decade"
    )


