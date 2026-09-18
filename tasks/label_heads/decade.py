from __future__ import annotations

"""Decade-based dating head.

This label head maps each `manuscript_id` to a coarse dating bucket
(`decade`) and uses that bucket as the class label.

The mapping is loaded from the aggregated dating table
`sfar_data_manuscripts_dating`, which is created by
`preprocess/dating/create_manuscript_dating_db.py`.
"""

from typing import Dict

import pandas as pd

from .base import LabelHead
from train.db_loader import get_db_connection, get_table_as_df
from system import DATING_MANUSCRIPT_TABLE


class DecadeLabelHead(LabelHead):
    """Classify manuscripts by 50-year dating bucket (`decade`)."""

    name = "decade"

    def __init__(self, *, table_name: str | None = None) -> None:
        # Default to the global dating table configured in system.py
        self._table_name = table_name or DATING_MANUSCRIPT_TABLE
        self._mid_to_decade: Dict[str, str] = {}
        self._load_mapping()

    def _load_mapping(self) -> None:
        """Load manuscript_id → decade mapping from the dating table."""
        conn = get_db_connection()
        try:
            df = get_table_as_df(conn, self._table_name)
        finally:
            conn.close()

        if "manuscript_id" not in df.columns or "decade" not in df.columns:
            raise ValueError(
                f"DecadeLabelHead expected columns 'manuscript_id' and 'decade' "
                f"in table '{self._table_name}', but found: {list(df.columns)}"
            )

        # Keep unique (manuscript_id, decade) pairs and build mapping.
        sub = (
            df[["manuscript_id", "decade"]]
            .dropna(subset=["manuscript_id", "decade"])
            .drop_duplicates()
        )
        if sub.empty:
            raise ValueError(
                f"DecadeLabelHead: table '{self._table_name}' produced an empty "
                f"manuscript_id→decade mapping."
            )

        # Normalize to strings for label stability.
        for _, row in sub.iterrows():
            mid = str(row["manuscript_id"]).strip()
            decade = str(int(row["decade"]))
            if mid:
                self._mid_to_decade[mid] = decade

        if not self._mid_to_decade:
            raise ValueError(
                f"DecadeLabelHead: no valid manuscript_id→decade mappings "
                f"could be constructed from table '{self._table_name}'."
            )

    def label_for(self, *, image_path: str, manuscript_id: str) -> str:
        """Return the decade label for a given manuscript_id."""
        mid = str(manuscript_id)
        if mid not in self._mid_to_decade:
            raise KeyError(
                f"Missing decade mapping for manuscript_id='{mid}' in table "
                f"'{self._table_name}'. Ensure the dating preprocess scripts "
                f"have been run and the table is up to date."
            )
        return self._mid_to_decade[mid]

