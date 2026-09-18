from __future__ import annotations

from .base import LabelHead


class ManuscriptIdLabelHead(LabelHead):
    name = "manuscript_id"

    def label_for(self, *, image_path: str, manuscript_id: str) -> str:
        # Current behavior: class = manuscript_id from split dict key.
        return str(manuscript_id)


