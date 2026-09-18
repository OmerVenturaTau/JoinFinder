"""Shared data, split, metric, and artifact helpers for interpretability probes."""

from __future__ import annotations

import configparser
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

_CACHE_ROOT = Path(tempfile.gettempdir()) / "joinsfinder-interpretability-cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPACES = ("tile", "glyph", "word", "shared")
VECTOR_COLUMNS = {space: f"{space}_vector" for space in SPACES}
DEFAULT_EXCLUDED_MANUSCRIPT_IDS = (
    "990000553830205171",
    "990000557720205171",
    "990000748850205171",
    "990000908640205171",
    "990000987680205171",
    "990001176180205171",
    "990001219080205171",
    "990001265360205171",
    "990001746540205171",
    "990001749030205171",
    "990001871010205171",
)


def safe_identifier(value: str) -> str:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(value)) is None:
        raise ValueError(f"Unsafe SQL identifier: {value!r}")
    return str(value)


def safe_path_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return cleaned or "run"


def get_connection(config_path: str):
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    config = configparser.ConfigParser()
    if not config.read(path):
        raise FileNotFoundError(f"Database config not found: {path}")
    db = config["postgresql"]
    return psycopg2.connect(
        host=db["host"],
        database=db["database"],
        user=db["user"],
        password=db["password"],
        port=db.get("port", 5432),
    )


def normalize_spaces(spaces: Iterable[str]) -> list[str]:
    requested = list(spaces)
    if not requested or "all" in requested:
        return list(SPACES)
    invalid = sorted(set(requested) - set(SPACES))
    if invalid:
        raise ValueError(f"Unknown latent spaces: {invalid}")
    return list(dict.fromkeys(requested))


def load_projection_rows(
    conn,
    *,
    projection_run: str,
    vectors_table: str,
    spaces: Sequence[str],
    excluded_manuscript_ids: Sequence[str] = DEFAULT_EXCLUDED_MANUSCRIPT_IDS,
) -> pd.DataFrame:
    table = safe_identifier(vectors_table)
    selected = normalize_spaces(spaces)
    vector_sql = ", ".join(
        f"{safe_identifier(VECTOR_COLUMNS[space])}::text AS {VECTOR_COLUMNS[space]}"
        for space in selected
    )
    excluded = sorted(
        {str(value).strip() for value in excluded_manuscript_ids if str(value).strip()}
    )
    exclusion_sql = " AND NOT (manuscript_id = ANY(%s))" if excluded else ""
    params: tuple[object, ...] = (projection_run, excluded) if excluded else (projection_run,)
    query = f"""
        SELECT manuscript_id, image_path, is_oriental, source_dataset_split,
               num_tiles, num_glyphs, num_words, {vector_sql}
        FROM {table}
        WHERE projection_run = %s
        {exclusion_sql}
        ORDER BY manuscript_id, image_path
    """
    return pd.read_sql_query(query, conn, params=params)


def parse_vector(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        result = value.astype(np.float32, copy=False).reshape(-1)
    elif isinstance(value, (list, tuple)):
        result = np.asarray(value, dtype=np.float32).reshape(-1)
    elif value is None or (isinstance(value, float) and math.isnan(value)):
        return np.empty(0, dtype=np.float32)
    else:
        text = str(value).strip()
        if not text or text.lower() == "none":
            return np.empty(0, dtype=np.float32)
        result = np.fromstring(text.strip("[]"), sep=",", dtype=np.float32)
    if result.size and not np.isfinite(result).all():
        raise ValueError("Stored vector contains non-finite values")
    return result


def space_matrix(frame: pd.DataFrame, space: str) -> tuple[pd.DataFrame, np.ndarray]:
    column = VECTOR_COLUMNS[space]
    if column not in frame:
        return frame.iloc[0:0].copy(), np.empty((0, 0), dtype=np.float32)
    parsed = frame[column].map(parse_vector)
    keep = parsed.map(len) > 0
    selected = frame.loc[keep].copy().reset_index(drop=True)
    vectors = list(parsed.loc[keep])
    if not vectors:
        return selected, np.empty((0, 0), dtype=np.float32)
    dimensions = {int(vector.size) for vector in vectors}
    if len(dimensions) != 1:
        raise ValueError(f"Projection run mixes {space} vector dimensions: {sorted(dimensions)}")
    matrix = np.stack(vectors).astype(np.float32, copy=False)
    return selected, matrix


def _unique_manuscript_labels(frame: pd.DataFrame, label_column: str) -> pd.DataFrame:
    counts = frame.groupby("manuscript_id")[label_column].nunique(dropna=False)
    bad = counts[counts != 1]
    if not bad.empty:
        raise ValueError(
            f"Labels are inconsistent within {len(bad)} manuscripts; first: {bad.index[:5].tolist()}"
        )
    return frame[["manuscript_id", label_column]].drop_duplicates().reset_index(drop=True)


def make_outer_split(
    frame: pd.DataFrame,
    *,
    label_column: str,
    seed: int,
    test_size: float = 0.30,
    rare_min_manuscripts: int | None = None,
) -> pd.DataFrame:
    manuscripts = _unique_manuscript_labels(frame, label_column)
    manuscripts["evaluable"] = True
    if rare_min_manuscripts is not None:
        label_counts = manuscripts[label_column].value_counts()
        rare_labels = set(label_counts[label_counts < rare_min_manuscripts].index.tolist())
        manuscripts.loc[manuscripts[label_column].isin(rare_labels), "evaluable"] = False
    common = manuscripts[manuscripts["evaluable"]].copy()
    rare = manuscripts[~manuscripts["evaluable"]].copy()
    if common[label_column].nunique() < 2:
        raise ValueError("At least two evaluable target classes are required")
    train_ids, test_ids = train_test_split(
        common["manuscript_id"].to_numpy(),
        test_size=test_size,
        random_state=seed,
        stratify=common[label_column].to_numpy(),
    )
    split_map = {str(value): "train" for value in train_ids}
    split_map.update({str(value): "test" for value in test_ids})
    split_map.update({str(value): "excluded" for value in rare["manuscript_id"]})
    manuscripts["split"] = manuscripts["manuscript_id"].astype(str).map(split_map)
    if manuscripts["split"].isna().any():
        raise AssertionError("Some manuscripts were not assigned to the outer split")
    return manuscripts.sort_values("manuscript_id").reset_index(drop=True)


def add_cv_folds(
    assignments: pd.DataFrame,
    *,
    label_column: str,
    seed: int,
    n_splits: int = 5,
) -> pd.DataFrame:
    result = assignments.copy()
    result["cv_fold"] = -1
    eligible = result[(result["split"] == "train") & result["evaluable"]].copy()
    minimum_class = int(eligible[label_column].value_counts().min())
    if minimum_class < n_splits:
        raise ValueError(
            f"Five-fold CV requires at least {n_splits} training manuscripts per class; "
            f"minimum is {minimum_class}"
        )
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, validation_indices) in enumerate(
        splitter.split(eligible["manuscript_id"], eligible[label_column])
    ):
        validation_ids = set(eligible.iloc[validation_indices]["manuscript_id"].astype(str))
        result.loc[
            result["manuscript_id"].astype(str).isin(validation_ids), "cv_fold"
        ] = fold
    return result


def assert_split_integrity(assignments: pd.DataFrame) -> None:
    if assignments["manuscript_id"].duplicated().any():
        raise AssertionError("Split assignments contain duplicate manuscript IDs")
    train = set(assignments.loc[assignments["split"] == "train", "manuscript_id"].astype(str))
    test = set(assignments.loc[assignments["split"] == "test", "manuscript_id"].astype(str))
    if train & test:
        raise AssertionError("Train and test manuscript IDs overlap")
    if (assignments.loc[~assignments["evaluable"], "split"] != "excluded").any():
        raise AssertionError("Rare/non-evaluable classes must be excluded")
    test_fraction = len(test) / max(1, len(train | test))
    if not 0.25 <= test_fraction <= 0.35:
        raise AssertionError(f"Unexpected evaluable test fraction: {test_fraction:.3f}")


def fold_masks(
    manuscript_ids: Sequence[str], assignments: pd.DataFrame, fold: int
) -> tuple[np.ndarray, np.ndarray]:
    lookup = assignments.set_index(assignments["manuscript_id"].astype(str))
    ids = pd.Index([str(value) for value in manuscript_ids])
    rows = lookup.reindex(ids)
    if rows["split"].isna().any():
        raise ValueError("Vector rows include manuscripts absent from split assignments")
    validation = (rows["split"].to_numpy() == "train") & (
        rows["cv_fold"].to_numpy(dtype=int) == fold
    )
    training = (rows["split"].to_numpy() == "train") & ~validation
    return training, validation


def outer_masks(
    manuscript_ids: Sequence[str], assignments: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    lookup = assignments.set_index(assignments["manuscript_id"].astype(str))
    rows = lookup.reindex(pd.Index([str(value) for value in manuscript_ids]))
    if rows["split"].isna().any():
        raise ValueError("Vector rows include manuscripts absent from split assignments")
    return rows["split"].to_numpy() == "train", rows["split"].to_numpy() == "test"


def manuscript_page_weights(manuscript_ids: Sequence[str]) -> np.ndarray:
    manuscripts = pd.Series([str(value) for value in manuscript_ids])
    per_manuscript = 1.0 / manuscripts.map(manuscripts.value_counts()).to_numpy(dtype=float)
    return (per_manuscript / per_manuscript.mean()).astype(np.float32)


def manuscript_class_balanced_page_weights(
    manuscript_ids: Sequence[str], labels: Sequence[int]
) -> np.ndarray:
    """Give each class equal total weight and each manuscript equal weight within it."""
    rows = pd.DataFrame(
        {
            "manuscript_id": [str(value) for value in manuscript_ids],
            "label": np.asarray(labels),
        }
    )
    label_counts = rows.groupby("manuscript_id")["label"].nunique(dropna=False)
    if (label_counts != 1).any():
        raise ValueError("A manuscript has multiple labels while computing sample weights")
    page_counts = rows["manuscript_id"].value_counts()
    manuscript_labels = rows.drop_duplicates("manuscript_id").set_index("manuscript_id")["label"]
    manuscripts_per_class = manuscript_labels.value_counts()
    weights = np.asarray(
        [
            1.0 / (page_counts[manuscript_id] * manuscripts_per_class[label])
            for manuscript_id, label in zip(rows["manuscript_id"], rows["label"])
        ],
        dtype=float,
    )
    return (weights / weights.mean()).astype(np.float32)


def page_sample_weights(manuscript_ids: Sequence[str], labels: Sequence[int]) -> np.ndarray:
    """Balance classes by manuscript while retaining every manuscript page.

    Multiplying independent page-level class weights by manuscript weights does
    not balance classes when their manuscripts have different mean page counts.
    Compute the joint weight directly instead.
    """
    return manuscript_class_balanced_page_weights(manuscript_ids, labels)


def aggregate_probabilities(
    manuscript_ids: Sequence[str],
    labels: Sequence[int],
    probabilities: np.ndarray,
    class_names: Sequence,
) -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "manuscript_id": [str(value) for value in manuscript_ids],
            "true_index": np.asarray(labels, dtype=int),
        }
    )
    for index, name in enumerate(class_names):
        data[f"probability_{name}"] = probabilities[:, index]
    if (data.groupby("manuscript_id")["true_index"].nunique() != 1).any():
        raise ValueError("A manuscript has multiple labels during probability aggregation")
    probability_columns = [f"probability_{name}" for name in class_names]
    grouped = data.groupby("manuscript_id", as_index=False).agg(
        {"true_index": "first", **{column: "mean" for column in probability_columns}}
    )
    matrix = grouped[probability_columns].to_numpy(dtype=float)
    grouped["predicted_index"] = matrix.argmax(axis=1)
    grouped["true_label"] = [class_names[index] for index in grouped["true_index"]]
    grouped["predicted_label"] = [class_names[index] for index in grouped["predicted_index"]]
    grouped["confidence"] = matrix.max(axis=1)
    return grouped


def image_prediction_frame(
    manuscript_ids: Sequence[str],
    image_paths: Sequence[str],
    labels: Sequence[int],
    probabilities: np.ndarray,
    class_names: Sequence,
) -> pd.DataFrame:
    predicted = probabilities.argmax(axis=1)
    result = pd.DataFrame(
        {
            "manuscript_id": [str(value) for value in manuscript_ids],
            "image_path": list(image_paths),
            "true_index": np.asarray(labels, dtype=int),
            "predicted_index": predicted,
            "true_label": [class_names[index] for index in labels],
            "predicted_label": [class_names[index] for index in predicted],
            "confidence": probabilities.max(axis=1),
        }
    )
    for index, name in enumerate(class_names):
        result[f"probability_{name}"] = probabilities[:, index]
    return result


def expected_calibration_error(
    labels: Sequence[int], probabilities: np.ndarray, bins: int = 10
) -> float:
    labels_array = np.asarray(labels, dtype=int)
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels_array
    total = max(1, len(labels_array))
    value = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            value += (mask.sum() / total) * abs(correct[mask].mean() - confidence[mask].mean())
    return float(value)


def multiclass_metrics(labels: Sequence[int], probabilities: np.ndarray) -> dict:
    y = np.asarray(labels, dtype=int)
    predicted = probabilities.argmax(axis=1)
    top_k = min(3, probabilities.shape[1])
    top_indices = np.argpartition(probabilities, -top_k, axis=1)[:, -top_k:]
    confidence = probabilities.max(axis=1)
    correct = predicted == y
    return {
        "accuracy_top1": float(accuracy_score(y, predicted)),
        "accuracy_top3": float(np.mean([label in row for label, row in zip(y, top_indices)])),
        "macro_f1": float(f1_score(y, predicted, labels=np.unique(y), average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, probabilities, labels=np.arange(probabilities.shape[1]))),
        "expected_calibration_error": expected_calibration_error(y, probabilities),
        "mean_confidence": float(confidence.mean()),
        "mean_confidence_correct": float(confidence[correct].mean()) if correct.any() else None,
        "mean_confidence_incorrect": float(confidence[~correct].mean()) if (~correct).any() else None,
        "num_examples": int(len(y)),
    }


def binary_metrics(labels: Sequence[int], probabilities: np.ndarray) -> dict:
    y = np.asarray(labels, dtype=int)
    positive = probabilities[:, 1]
    predicted = (positive >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "specificity": float(tn / max(1, tn + fp)),
        "sensitivity": float(tp / max(1, tp + fn)),
        "roc_auc": float(roc_auc_score(y, positive)),
        "pr_auc": float(average_precision_score(y, positive)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "num_examples": int(len(y)),
    }


def fit_temperature(logits: np.ndarray, labels: Sequence[int]) -> float:
    logits_array = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    if logits_array.size == 0:
        return 1.0

    def objective(log_temperature: float) -> float:
        temperature = math.exp(float(log_temperature))
        scaled = logits_array / temperature
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(y)), y]))

    result = minimize_scalar(
        objective,
        bounds=(math.log(0.05), math.log(10.0)),
        method="bounded",
    )
    return float(math.exp(result.x)) if result.success else 1.0


def logits_to_probabilities(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-6)
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def save_json(path: Path, value) -> None:
    def default(item):
        if isinstance(item, (np.integer, np.floating)):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(f"Cannot serialize {type(item).__name__}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=default) + "\n")


def save_confusion_plot(
    path: Path,
    labels: Sequence[int],
    predictions: Sequence[int],
    class_names: Sequence,
    title: str,
) -> None:
    matrix = confusion_matrix(labels, predictions, labels=np.arange(len(class_names)))
    size = max(5.0, min(14.0, 0.65 * len(class_names)))
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_xticks(np.arange(len(class_names)), labels=[str(value) for value in class_names], rotation=90)
    axis.set_yticks(np.arange(len(class_names)), labels=[str(value) for value in class_names])
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
