import os
import random
import pandas as pd
from typing import Dict, List, Tuple
import numpy as np
import re
import logging

from train.db_loader import get_db_connection, get_table_as_df

# Set up logger
logger = logging.getLogger(__name__)
from system import (
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    FINETUNE_ORIENTAL_RATIO,
    DEMO_MANUSCRIPTS_PER_CLASS,
    DEMO_IMAGES_PER_MANUSCRIPT,
)


def _prepare_sorted_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* sorted in a stable, page-order aware way."""
    df = df.copy()
    if "width" in df.columns and "height" in df.columns:
        df["area"] = df["width"] * df["height"]

    # Sort within each manuscript by parent_directory and numeric page_number
    def to_page_num(x: str) -> int:
        # Extract digits from page_number like 'P000484' -> 484
        m = re.search(r"(\d+)", str(x))
        return int(m.group(1)) if m else 0

    df["page_num_int"] = df["page_number"].apply(to_page_num)
    sort_columns = ["manuscript_id"]
    if "parent_directory" in df.columns:
        sort_columns.append("parent_directory")
    sort_columns.append("page_num_int")
    df = df.sort_values(sort_columns)
    return df


def split_df_to_paths_labels(
    df: pd.DataFrame,
    base_dir: str,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> Tuple[
    List[str],
    List[str],
    List[str],
    List[str],
    List[str],
    List[str],
    List[str],
    List[str],
    List[str],
]:
    """
    Page-level split *within* each manuscript.

    Legacy helper retained for older scripts. build_splits() now uses
    manuscript-level splitting for normal training stages.
    """
    df = _prepare_sorted_df(df)
    grouped = df.groupby("manuscript_id")
    train_paths, train_labels, train_xmls = [], [], []
    val_paths, val_labels, val_xmls = [], [], []
    test_paths, test_labels, test_xmls = [], [], []

    for manuscript_id, group in grouped:
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train = group.iloc[:n_train]
        val = group.iloc[n_train : n_train + n_val]
        test = group.iloc[n_train + n_val :]

        for _, row in train.iterrows():
            path = (
                row["image_path"]
                if "image_path" in row
                else f"{base_dir}/{row['manuscript_id']}/{row['parent_directory']}/{row['picture_id']}"
            )
            train_paths.append(path)
            train_labels.append(row["manuscript_id"])
            train_xmls.append(row.get("xml_path"))

        for _, row in val.iterrows():
            path = (
                row["image_path"]
                if "image_path" in row
                else f"{base_dir}/{row['manuscript_id']}/{row['parent_directory']}/{row['picture_id']}"
            )
            val_paths.append(path)
            val_labels.append(row["manuscript_id"])
            val_xmls.append(row.get("xml_path"))

        for _, row in test.iterrows():
            path = (
                row["image_path"]
                if "image_path" in row
                else f"{base_dir}/{row['manuscript_id']}/{row['parent_directory']}/{row['picture_id']}"
            )
            test_paths.append(path)
            test_labels.append(row["manuscript_id"])
            test_xmls.append(row.get("xml_path"))

    return (
        train_paths,
        train_labels,
        train_xmls,
        val_paths,
        val_labels,
        val_xmls,
        test_paths,
        test_labels,
        test_xmls,
    )


def paths_labels_from_dataset_split_column(
    df: pd.DataFrame,
    base_dir: str,
) -> Tuple[
    List[str],
    List[str],
    List[str],
    List[str],
    List[str],
    List[str],
    List[str],
    List[str],
    List[str],
]:
    """
    Build train/val/test path lists from a table that has a 'dataset_split' column
    (e.g. pretrain_finetune_oriental_non_oriental_train_val_test_split).
    """
    base_dir = base_dir.rstrip("/")
    train_paths, train_labels, train_xmls = [], [], []
    val_paths, val_labels, val_xmls = [], [], []
    test_paths, test_labels, test_xmls = [], [], []

    for _, row in df.iterrows():
        path = (
            row["image_path"]
            if "image_path" in row and pd.notna(row.get("image_path"))
            else f"{base_dir}/{row['manuscript_id']}/{row['parent_directory']}/{row['picture_id']}"
        )
        label = str(row["manuscript_id"])
        xml = row.get("xml_path")
        split = str(row["dataset_split"]).strip().lower()
        if split == "train":
            train_paths.append(path)
            train_labels.append(label)
            train_xmls.append(xml)
        elif split == "val":
            val_paths.append(path)
            val_labels.append(label)
            val_xmls.append(xml)
        elif split == "test":
            test_paths.append(path)
            test_labels.append(label)
            test_xmls.append(xml)

    return (
        train_paths,
        train_labels,
        train_xmls,
        val_paths,
        val_labels,
        val_xmls,
        test_paths,
        test_labels,
        test_xmls,
    )


def build_page_level_classification_splits(
    df: pd.DataFrame,
    base_dir: str,
    *,
    train_fraction: float,
) -> Tuple[Dict[str, Dict[str, List]], Dict[str, int]]:
    """Build closed-set train/val splits by dividing pages within each class.

    This is used when Geniza manuscripts are added as ordinary manuscript-ID
    classification classes.  The Geniza table's stored split is deliberately
    manuscript-disjoint for contrastive/open-set training, which is unsuitable
    for a closed-set classifier: a validation-only manuscript would have no
    trained classifier weight.  Here every manuscript with at least two pages
    contributes pages to both train and validation.  No test rows are created,
    so the curated joins collection remains the downstream test set.
    """
    if not 0.0 < float(train_fraction) < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction!r}")

    required_columns = {"manuscript_id", "page_number"}
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise RuntimeError(
            f"Classification rows are missing required columns: {missing_columns}"
        )
    if "image_path" not in df.columns and not {
        "parent_directory",
        "picture_id",
    }.issubset(df.columns):
        raise RuntimeError(
            "Classification rows require image_path or both parent_directory and picture_id."
        )

    prepared = _prepare_sorted_df(df)
    split_dict: Dict[str, Dict[str, List]] = {"train": {}, "val": {}, "test": {}}

    for manuscript_id, group in prepared.groupby("manuscript_id", sort=True):
        manuscript_id = str(manuscript_id)
        n_rows = len(group)
        if n_rows >= 2:
            n_train = int(round(n_rows * float(train_fraction)))
            n_train = max(1, min(n_rows - 1, n_train))
        else:
            n_train = n_rows

        for row_index, (_, row) in enumerate(group.iterrows()):
            path = (
                row["image_path"]
                if "image_path" in row and pd.notna(row.get("image_path"))
                else f"{base_dir.rstrip('/')}/{row['manuscript_id']}/{row['parent_directory']}/{row['picture_id']}"
            )
            item = (path, row.get("xml_path"))
            split_name = "train" if row_index < n_train else "val"
            split_dict[split_name].setdefault(manuscript_id, []).append(item)

    stats = {
        "images_total": int(len(prepared)),
        "images_train": int(sum(len(items) for items in split_dict["train"].values())),
        "images_val": int(sum(len(items) for items in split_dict["val"].values())),
        "manuscripts_total": int(prepared["manuscript_id"].astype(str).nunique()),
        "manuscripts_train": int(len(split_dict["train"])),
        "manuscripts_val": int(len(split_dict["val"])),
    }
    return split_dict, stats


def merge_classification_splits(
    primary: Dict[str, Dict[str, List]],
    additional: Dict[str, Dict[str, List]],
) -> Dict[str, Dict[str, List]]:
    """Return a non-mutating merge of two train/val/test split dictionaries."""
    merged: Dict[str, Dict[str, List]] = {"train": {}, "val": {}, "test": {}}
    for split_name in merged:
        for source in (primary, additional):
            for label, items in source.get(split_name, {}).items():
                merged[split_name].setdefault(str(label), []).extend(list(items))
    return merged


def split_df_to_paths_labels_by_manuscript(
    df: pd.DataFrame,
    base_dir: str,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> Tuple[
    List[str],
    List[str],
    List[str],
    List[str],
    List[str],
    List[str],
    List[str],
    List[str],
    List[str],
]:
    """
    Manuscript-level split: each manuscript goes entirely into train, val, or test.

    This is used for dating experiments to avoid any leakage where pages from the
    same manuscript appear in both training and validation/test splits.
    """
    df = _prepare_sorted_df(df)

    manuscript_ids = df["manuscript_id"].astype(str).unique().tolist()
    rng = np.random.RandomState(42)
    rng.shuffle(manuscript_ids)

    n_total = len(manuscript_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_ms = set(manuscript_ids[:n_train])
    val_ms = set(manuscript_ids[n_train : n_train + n_val])
    test_ms = set(manuscript_ids[n_train + n_val :])

    train_paths, train_labels, train_xmls = [], [], []
    val_paths, val_labels, val_xmls = [], [], []
    test_paths, test_labels, test_xmls = [], [], []

    grouped = df.groupby("manuscript_id")
    for manuscript_id, group in grouped:
        mid = str(manuscript_id)
        if mid in train_ms:
            target = (train_paths, train_labels, train_xmls)
        elif mid in val_ms:
            target = (val_paths, val_labels, val_xmls)
        else:
            target = (test_paths, test_labels, test_xmls)

        paths_list, labels_list, xmls_list = target
        for _, row in group.iterrows():
            path = (
                row["image_path"]
                if "image_path" in row
                else f"{base_dir}/{row['manuscript_id']}/{row['parent_directory']}/{row['picture_id']}"
            )
            paths_list.append(path)
            labels_list.append(row["manuscript_id"])
            xmls_list.append(row.get("xml_path"))

    return (
        train_paths,
        train_labels,
        train_xmls,
        val_paths,
        val_labels,
        val_xmls,
        test_paths,
        test_labels,
        test_xmls,
    )


def _normalize_dataset_stage(stage: str) -> str:
    stage = str(stage or "stage1").strip().lower()
    aliases = {
        "pretrain": "stage1",
        "stage_1": "stage1",
        "1": "stage1",
        "finetune": "stage2",
        "fine-tune": "stage2",
        "stage_2": "stage2",
        "2": "stage2",
    }
    return aliases.get(stage, stage)


def build_splits(
    base_dir: str,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    table_name: str = None,
    dataset_stage: str = None,
) -> Tuple[Dict[str, Dict[str, List]], Dict[str, object]]:
    """
    Build dict splits {'train': {...}, 'val': {...}, 'test': {...}} using pre-populated DB tables.
    
    Determines the classification table based on dataset_stage/system.TRAINING_MODE
    or an explicit table_name. Stage affects only classification data/table
    selection and optional sampling. Geniza contrastive data is loaded separately.
    If the table has a dataset_split column, that column is authoritative.
    """
    from system import TRAINING_MODE, STAGE1_TABLE_NAME, STAGE2_TABLE_NAME

    dataset_stage = _normalize_dataset_stage(dataset_stage or TRAINING_MODE)
    if table_name is None:
        table_name = STAGE2_TABLE_NAME if dataset_stage == "stage2" else STAGE1_TABLE_NAME

    logger.info(f"Loading classification data from DB table: {table_name} (dataset_stage={dataset_stage})")
    
    conn = get_db_connection()
    try:
        filtered_df = get_table_as_df(conn, table_name)
    finally:
        conn.close()

    # Stage2 classification data: oriental-heavy mix (80-90% oriental,
    # 10-20% non-oriental rehearsal). Preserve any dataset_split values from
    # the table; they are the source of truth for classification splits.
    if dataset_stage == "stage2" and "is_oriental" in filtered_df.columns:
        oriental_mask = (filtered_df["is_oriental"] == True) | (filtered_df["is_oriental"].astype(str).str.lower() == "t")
        df_oriental = filtered_df[oriental_mask].copy()
        df_non_oriental = filtered_df[~oriental_mask].copy()
        non_oriental_target_frac = 1.0 - FINETUNE_ORIENTAL_RATIO  # e.g. 0.15 for 85% oriental

        n_oriental = len(df_oriental)
        n_non_target = int(round(n_oriental * non_oriental_target_frac / FINETUNE_ORIENTAL_RATIO)) if n_oriental else len(df_non_oriental)
        if 0 < n_non_target < len(df_non_oriental):
            frac = n_non_target / len(df_non_oriental)
            no_sampled = df_non_oriental.groupby("manuscript_id", group_keys=False).apply(
                lambda g: g.sample(frac=frac, random_state=42)
            )
        else:
            no_sampled = df_non_oriental

        filtered_df = pd.concat([df_oriental, no_sampled], ignore_index=True)
        n_total = len(filtered_df)
        logger.info(
            f"[DEBUG] build_splits: stage2 oriental-heavy: {n_oriental} oriental + {n_total - n_oriental} non-oriental rehearsal -> {n_total} rows"
        )

    # Demo: tiny, balanced subset sampled from the pretrain table.
    # We ignore any existing dataset_split assignments and re-split in code to
    # exercise the full train/val/test pipeline on a very small dataset.
    if dataset_stage == "demo":
        if "is_oriental" not in filtered_df.columns:
            logger.warning(
                "[DEMO] build_splits: 'is_oriental' column not found; "
                "cannot build balanced oriental/non-oriental demo subset. Returning empty splits."
            )
            return {'train': {}, 'val': {}, 'test': {}}, {}

        oriental_mask = (filtered_df["is_oriental"] == True) | (filtered_df["is_oriental"].astype(str).str.lower() == "t")
        df_oriental = filtered_df[oriental_mask].copy()
        df_non_oriental = filtered_df[~oriental_mask].copy()

        rng = np.random.RandomState(123)

        def _sample_demo_by_manuscript(df_group: pd.DataFrame) -> pd.DataFrame:
            """
            Sample a fixed number of manuscripts, then a fixed number of pages per manuscript.
            This better mimics the real problem where multiple pages come from each manuscript.
            """
            if df_group.empty:
                return df_group

            # Ensure stable page ordering within each manuscript
            df_sorted = _prepare_sorted_df(df_group)
            manuscript_ids = df_sorted["manuscript_id"].astype(str).unique().tolist()
            rng.shuffle(manuscript_ids)

            selected_ms = manuscript_ids[: min(DEMO_MANUSCRIPTS_PER_CLASS, len(manuscript_ids))]
            parts = []
            for mid in selected_ms:
                g = df_sorted[df_sorted["manuscript_id"].astype(str) == mid]
                # Take up to DEMO_IMAGES_PER_MANUSCRIPT pages per manuscript
                g = g.iloc[:DEMO_IMAGES_PER_MANUSCRIPT]
                parts.append(g)

            if not parts:
                return df_group.iloc[0:0]  # empty with same columns
            return pd.concat(parts, ignore_index=True)

        df_demo_oriental = _sample_demo_by_manuscript(df_oriental)
        df_demo_non_oriental = _sample_demo_by_manuscript(df_non_oriental)
        filtered_df = pd.concat([df_demo_oriental, df_demo_non_oriental], ignore_index=True)

        # Drop any pre-existing dataset_split so we re-split BY PAGES using the standard ratios.
        if "dataset_split" in filtered_df.columns:
            filtered_df = filtered_df.drop(columns=["dataset_split"])

        logger.info(
            "[DEMO] build_splits: demo subset created with "
            f"{len(df_demo_oriental)} oriental and {len(df_demo_non_oriental)} non-oriental rows "
            f"(total={len(filtered_df)}), "
            f"DEMO_MANUSCRIPTS_PER_CLASS={DEMO_MANUSCRIPTS_PER_CLASS}, "
            f"DEMO_IMAGES_PER_MANUSCRIPT={DEMO_IMAGES_PER_MANUSCRIPT}."
        )

    if filtered_df.empty:
        logger.warning(f"[DEBUG] build_splits: Table {table_name} is empty; returning empty splits")
        return {'train': {}, 'val': {}, 'test': {}}, {}

    # stats for logging
    stats = {
        'total_manuscripts': int(filtered_df['manuscript_id'].nunique()),
        'images_total_selected': int(len(filtered_df)),
    }

    if "dataset_split" in filtered_df.columns:
        split_values = set(filtered_df["dataset_split"].dropna().astype(str).str.strip().str.lower())
        missing_splits = {"train", "val"} - split_values
        if missing_splits:
            raise RuntimeError(
                f"Classification table {table_name!r} has dataset_split column but is missing "
                f"required split value(s): {sorted(missing_splits)}"
            )
        if "test" not in split_values:
            logger.warning(
                f"Classification table {table_name!r} has no dataset_split='test' rows; "
                "continuing with an empty test split."
            )
        logger.info("[DEBUG] build_splits: using dataset_split column from classification table.")
        (
            train_paths,
            train_labels,
            train_xmls,
            val_paths,
            val_labels,
            val_xmls,
            test_paths,
            test_labels,
            test_xmls,
        ) = paths_labels_from_dataset_split_column(filtered_df, base_dir.rstrip("/"))
        split_strategy = "dataset_split_column"
    else:
        logger.warning(
            "[DEBUG] build_splits: dataset_split column not found; falling back to "
            "legacy page-level split within each manuscript."
        )
        (
            train_paths,
            train_labels,
            train_xmls,
            val_paths,
            val_labels,
            val_xmls,
            test_paths,
            test_labels,
            test_xmls,
        ) = split_df_to_paths_labels(
            filtered_df,
            base_dir.rstrip("/"),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        split_strategy = "legacy_page_level"

    # DEBUG: Inspect manuscript_id distribution going into splits
    train_ids = sorted(set(str(l) for l in train_labels))
    val_ids = sorted(set(str(l) for l in val_labels))
    test_ids = sorted(set(str(l) for l in test_labels))
    all_ids = sorted(set(train_ids + val_ids + test_ids))
    logger.info(
        "[DEBUG] build_splits: "
        f"train_ms_count={len(train_ids)}, val_ms_count={len(val_ids)}, "
        f"test_ms_count={len(test_ids)}, total_unique_ms={len(all_ids)}"
    )
    stats["split_strategy"] = split_strategy

    def to_dict(paths: List[str], labels: List[str], xmls: List[str]) -> Dict[str, List[Tuple[str, str]]]:
        out: Dict[str, List[Tuple[str, str]]] = {}
        for p, l, x in zip(paths, labels, xmls):
            out.setdefault(l, []).append((p, x))
        return out

    splits = {
        'train': to_dict(train_paths, train_labels, train_xmls),
        'val': to_dict(val_paths, val_labels, val_xmls),
        'test': to_dict(test_paths, test_labels, test_xmls),
    }

    return splits, stats
