#!/usr/bin/env python3
"""
Compare clustered image pairs using SIFT baseline latents.

Default input:
  results_analysis/test_set/clusters_images_metadata.csv

Output:
  Debugs/baseline comparisons/outputs/clusters_images_metadata_pairs_sift.xlsx

The Excel file mirrors the pair-comparison shape used by
results_analysis/test_set/compare_clusters_pairs.py:
  - pairs sheet: one row per image pair with original columns suffixed _1/_2
  - metrics sheet: same/different separation and retrieval metrics, including mAP

SIFT latents are mean/std-pooled RootSIFT descriptors by default, producing
256-dimensional vectors.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "system.py").exists())
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sift_latents import create_sift, extract_sift_latent  # noqa: E402


DEFAULT_INPUT = PROJECT_ROOT / "results_analysis/test_set/clusters_images_metadata.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs/clusters_images_metadata_pairs_sift.xlsx"


def _resolve_project_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def _load_input_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _is_valid_vector(vec: np.ndarray | None) -> bool:
    if vec is None:
        return False
    arr = np.asarray(vec)
    return arr.ndim == 1 and arr.size > 0 and np.isfinite(arr).all()


def cosine_similarity(v1: np.ndarray | None, v2: np.ndarray | None) -> float:
    if not _is_valid_vector(v1) or not _is_valid_vector(v2):
        return np.nan
    return float(np.dot(v1, v2))


def _norm_key(path: str | Path) -> str:
    return os.path.normpath(str(path).strip())


def _image_file_path(path_value: object) -> Path:
    path = Path(str(path_value).strip()).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_latents_npz(path: Path) -> tuple[Dict[str, np.ndarray | None], Dict[str, int], dict]:
    data = np.load(path, allow_pickle=False)
    paths = [str(p) for p in data["paths"]]
    latents = data["latents"].astype(np.float32, copy=False)
    num_keypoints = data["num_keypoints"].astype(np.int32, copy=False)
    metadata_raw = str(data["metadata"]) if "metadata" in data.files else "{}"
    try:
        metadata = json.loads(metadata_raw)
    except json.JSONDecodeError:
        metadata = {}

    vecs: Dict[str, np.ndarray | None] = {}
    counts: Dict[str, int] = {}
    for p, vec, count in zip(paths, latents, num_keypoints):
        key = str(p).strip()
        norm = float(np.linalg.norm(vec)) if np.isfinite(vec).all() else 0.0
        stored_vec = vec / norm if norm >= 1e-12 else None
        vecs[key] = stored_vec
        vecs[_norm_key(key)] = stored_vec
        counts[key] = int(count)
        counts[_norm_key(key)] = int(count)
    return vecs, counts, metadata


def _save_latents_npz(
    path: Path,
    image_paths: Sequence[str],
    latents: np.ndarray,
    num_keypoints: np.ndarray,
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        paths=np.asarray(list(image_paths), dtype=str),
        latents=latents.astype(np.float32, copy=False),
        num_keypoints=num_keypoints.astype(np.int32, copy=False),
        metadata=json.dumps(metadata, ensure_ascii=False),
    )


def compute_sift_vectors(
    image_paths: Sequence[str],
    *,
    pooling: str,
    use_rootsift: bool,
    resize_side: int,
    max_keypoints: int,
    fail_fast: bool,
    latents_output: Path | None,
) -> tuple[Dict[str, np.ndarray | None], Dict[str, int], dict]:
    sift = create_sift()(nfeatures=max(0, int(max_keypoints)))
    vecs: Dict[str, np.ndarray | None] = {}
    counts: Dict[str, int] = {}
    written_paths: List[str] = []
    written_latents: List[np.ndarray] = []
    written_counts: List[int] = []
    failures: List[dict] = []

    for idx, path_str in enumerate(image_paths, start=1):
        image_path = _image_file_path(path_str)
        try:
            latent, count = extract_sift_latent(
                image_path,
                sift,
                pooling=pooling,
                use_rootsift=use_rootsift,
                resize_side=resize_side,
            )
        except Exception as exc:
            if fail_fast:
                raise
            failures.append({"path": str(path_str), "resolved_path": str(image_path), "error": str(exc)})
            vec = None
            count = 0
            print(f"[{idx}/{len(image_paths)}] skipped {path_str}: {exc}", file=sys.stderr)
        else:
            norm = float(np.linalg.norm(latent)) if np.isfinite(latent).all() else 0.0
            vec = latent / norm if norm >= 1e-12 else None
            written_paths.append(str(path_str))
            written_latents.append(latent)
            written_counts.append(count)
            if idx == 1 or idx % 25 == 0 or idx == len(image_paths):
                print(f"[{idx}/{len(image_paths)}] processed {path_str} ({count} keypoints)")

        key = str(path_str).strip()
        vecs[key] = vec
        vecs[_norm_key(key)] = vec
        counts[key] = int(count)
        counts[_norm_key(key)] = int(count)

    metadata = {
        "pooling": pooling,
        "rootsift": use_rootsift,
        "resize_max_side": int(resize_side),
        "max_keypoints": int(max_keypoints),
        "num_images_requested": len(image_paths),
        "num_images_with_latents": len(written_paths),
        "latent_dim": int(written_latents[0].shape[0]) if written_latents else 0,
        "failures": failures,
    }
    if latents_output and written_latents:
        _save_latents_npz(
            latents_output,
            written_paths,
            np.stack(written_latents, axis=0),
            np.asarray(written_counts, dtype=np.int32),
            metadata,
        )
        print(f"Wrote SIFT latents to {latents_output}")
    return vecs, counts, metadata


def _finite_floats(values) -> list[float]:
    out: list[float] = []
    for value in values:
        try:
            fv = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv):
            out.append(fv)
    return out


def _roc_auc_score_binary(y_true: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[y_true == 1]
    negatives = scores[y_true == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return np.nan
    # Mann-Whitney interpretation of ROC AUC, with half credit for ties.
    wins = 0.0
    for pos in positives:
        wins += float(np.sum(pos > negatives))
        wins += 0.5 * float(np.sum(pos == negatives))
    return wins / float(len(positives) * len(negatives))


def _average_precision_binary(y_true: np.ndarray, scores: np.ndarray) -> float:
    positives = int(np.sum(y_true == 1))
    if positives == 0:
        return np.nan
    order = np.argsort(-scores)
    sorted_true = y_true[order]
    hit_count = 0
    precision_sum = 0.0
    for rank, is_positive in enumerate(sorted_true, start=1):
        if int(is_positive) == 1:
            hit_count += 1
            precision_sum += hit_count / rank
    return precision_sum / positives


def _best_f1_at_threshold(y_true: np.ndarray, scores: np.ndarray) -> dict:
    best = {
        "threshold": np.nan,
        "f1": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "accuracy": np.nan,
    }
    thresholds = np.unique(scores)
    if thresholds.size == 0:
        return best

    for threshold in thresholds:
        pred = scores >= threshold
        tp = int(np.sum((pred == 1) & (y_true == 1)))
        fp = int(np.sum((pred == 1) & (y_true == 0)))
        fn = int(np.sum((pred == 0) & (y_true == 1)))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = (2.0 * precision * recall) / max(1e-12, precision + recall)
        accuracy = float(np.mean(pred.astype(int) == y_true))
        if not np.isfinite(best["f1"]) or f1 > best["f1"]:
            best = {
                "threshold": float(threshold),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "accuracy": float(accuracy),
            }
    return best


def pair_separation_metrics(
    pairs_df: pd.DataFrame,
    score_col: str,
    *,
    label_col: str = "are_they_same_clusters",
) -> list[dict[str, object]]:
    valid = pairs_df[[label_col, score_col]].dropna().copy()
    if valid.empty:
        return []

    valid[label_col] = valid[label_col].astype(bool)
    y_true = valid[label_col].astype(int).to_numpy()
    scores = valid[score_col].astype(float).to_numpy()
    same_scores = scores[y_true == 1]
    diff_scores = scores[y_true == 0]
    rows: list[dict[str, object]] = []

    def add(metric: str, value: object) -> None:
        rows.append({"metric_set": score_col, "metric": metric, "value": value})

    add("n_pairs", int(len(scores)))
    add("n_same_pairs", int(len(same_scores)))
    add("n_different_pairs", int(len(diff_scores)))
    for prefix, vals in (("same", same_scores), ("different", diff_scores)):
        if len(vals) == 0:
            continue
        add(f"{prefix}_mean", float(np.mean(vals)))
        add(f"{prefix}_std", float(np.std(vals)))
        add(f"{prefix}_median", float(np.median(vals)))
        add(f"{prefix}_p10", float(np.percentile(vals, 10)))
        add(f"{prefix}_p90", float(np.percentile(vals, 90)))
    if len(same_scores) > 0 and len(diff_scores) > 0:
        add("mean_gap_same_minus_different", float(np.mean(same_scores) - np.mean(diff_scores)))

    if len(np.unique(y_true)) >= 2:
        add("roc_auc", float(_roc_auc_score_binary(y_true, scores)))
        add("pr_auc_average_precision", float(_average_precision_binary(y_true, scores)))
        best = _best_f1_at_threshold(y_true, scores)
        add("best_f1_threshold", best["threshold"])
        add("best_f1", best["f1"])
        add("best_f1_precision", best["precision"])
        add("best_f1_recall", best["recall"])
        add("best_f1_accuracy", best["accuracy"])
    return rows


def cluster_retrieval_metrics(
    df: pd.DataFrame,
    get_vec: Callable[[object], np.ndarray | None],
    *,
    score_col: str,
    k_values: tuple[int, ...] = (1, 5, 10),
) -> list[dict[str, object]]:
    records = []
    for idx, row in df.iterrows():
        vec = get_vec(row["image_path"])
        if vec is None:
            continue
        records.append(
            {
                "idx": idx,
                "image_path": row["image_path"],
                "cluster_id": row["cluster_id"],
                "vec": vec,
            }
        )

    rows: list[dict[str, object]] = []
    metric_set = f"{score_col}/cluster_retrieval"

    def add(metric: str, value: object) -> None:
        rows.append({"metric_set": metric_set, "metric": metric, "value": value})

    add("n_images_with_scores", int(len(records)))
    if len(records) < 2:
        return rows

    aps: list[float] = []
    recalls_at_k = {k: [] for k in k_values}
    hits_at_k = {k: [] for k in k_values}
    evaluated_queries = 0

    for query in records:
        scored = []
        for candidate in records:
            if candidate["idx"] == query["idx"]:
                continue
            sim = cosine_similarity(query["vec"], candidate["vec"])
            scored.append((sim, candidate["cluster_id"] == query["cluster_id"]))
        scored.sort(key=lambda item: item[0], reverse=True)
        relevant_total = sum(1 for _sim, is_relevant in scored if is_relevant)
        if relevant_total == 0:
            continue

        evaluated_queries += 1
        hit_count = 0
        precision_sum = 0.0
        for rank, (_sim, is_relevant) in enumerate(scored, start=1):
            if is_relevant:
                hit_count += 1
                precision_sum += hit_count / rank
        aps.append(precision_sum / relevant_total)

        for k in k_values:
            top_k = scored[:k]
            rel_in_top_k = sum(1 for _sim, is_relevant in top_k if is_relevant)
            recalls_at_k[k].append(rel_in_top_k / relevant_total)
            hits_at_k[k].append(float(rel_in_top_k > 0))

    add("n_queries_with_relevant", int(evaluated_queries))
    if evaluated_queries == 0:
        return rows

    add("mAP", float(np.mean(aps)))
    for k in k_values:
        add(f"recall@{k}", float(np.mean(recalls_at_k[k])))
        add(f"hit@{k}", float(np.mean(hits_at_k[k])))
    return rows


def try_plot_histogram(pairs_df: pd.DataFrame, score_col: str, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    same_mask = pairs_df["are_they_same_clusters"] == True  # noqa: E712
    diff_mask = pairs_df["are_they_same_clusters"] == False  # noqa: E712
    same_vals = _finite_floats(pairs_df.loc[same_mask, score_col].tolist())
    diff_vals = _finite_floats(pairs_df.loc[diff_mask, score_col].tolist())
    if not same_vals and not diff_vals:
        return

    all_vals = same_vals + diff_vals
    lo = float(min(all_vals)) - 0.005
    hi = float(max(all_vals)) + 0.005
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    if same_vals:
        plt.hist(same_vals, bins=50, range=(lo, hi), density=True, alpha=0.55, label="same cluster", color="steelblue")
    if diff_vals:
        plt.hist(diff_vals, bins=50, range=(lo, hi), density=True, alpha=0.55, label="different clusters", color="tomato")
    plt.title("SIFT similarity score distribution")
    plt.xlabel("cosine similarity")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def write_outputs(output_path: Path, pairs_df: pd.DataFrame, metrics_df: pd.DataFrame, metadata: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ext = output_path.suffix.lower()
    if ext in {".xlsx", ".xlsm", ".xls"}:
        with pd.ExcelWriter(output_path) as writer:
            pairs_df.to_excel(writer, index=False, sheet_name="pairs")
            metrics_df.to_excel(writer, index=False, sheet_name="metrics")
            pd.DataFrame([metadata]).to_excel(writer, index=False, sheet_name="sift_metadata")
    else:
        pairs_df.to_csv(output_path, index=False)

    metrics_path = output_path.with_name(output_path.stem + "_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Wrote metrics to {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_table", nargs="?", default=str(DEFAULT_INPUT), help="CSV/XLSX cluster metadata table.")
    parser.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT), help="Output Excel/CSV path.")
    parser.add_argument("--latents-output", default="", help="Optional NPZ path to save computed SIFT latents.")
    parser.add_argument("--load-latents", default="", help="Reuse an existing SIFT latent NPZ instead of recomputing.")
    parser.add_argument("--limit", type=int, default=0, help="Max number of pairs to write/score (0 = all).")
    parser.add_argument("--image-limit", type=int, default=0, help="Debug/smoke-test limit on input images before pairing.")
    parser.add_argument("--max-keypoints", type=int, default=4096)
    parser.add_argument("--resize-max-side", type=int, default=1600)
    parser.add_argument("--pooling", choices=("mean", "mean_std"), default="mean_std")
    parser.add_argument("--no-rootsift", action="store_true", help="Use raw SIFT descriptors instead of RootSIFT.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first unreadable image.")
    parser.add_argument("--no-histogram", action="store_true", help="Do not write histogram PNG.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = _resolve_project_path(args.input_table)
    output_path = _resolve_project_path(args.output)
    latents_output = _resolve_project_path(args.latents_output) if args.latents_output else None
    load_latents = _resolve_project_path(args.load_latents) if args.load_latents else None

    if not input_path.exists():
        print(f"Input table not found: {input_path}", file=sys.stderr)
        return 2

    df = _load_input_table(input_path)
    required = ["manuscript_id", "picture_id", "image_path", "cluster_id"]
    for column in required:
        if column not in df.columns:
            print(f"Missing column {column!r}. Columns: {list(df.columns)}", file=sys.stderr)
            return 2
    if args.image_limit and args.image_limit > 0:
        df = df.head(args.image_limit).copy()

    image_paths = df["image_path"].dropna().astype(str).drop_duplicates().tolist()
    print(f"Loaded {len(df)} rows from {input_path}")
    print(f"Preparing SIFT latents for {len(image_paths)} unique images")

    if load_latents:
        if not load_latents.exists():
            print(f"Latent NPZ not found: {load_latents}", file=sys.stderr)
            return 2
        vecs, keypoint_counts, metadata = _load_latents_npz(load_latents)
        metadata = {**metadata, "loaded_latents": str(load_latents)}
        print(f"Loaded SIFT latents from {load_latents}")
    else:
        vecs, keypoint_counts, metadata = compute_sift_vectors(
            image_paths,
            pooling=args.pooling,
            use_rootsift=not args.no_rootsift,
            resize_side=args.resize_max_side,
            max_keypoints=args.max_keypoints,
            fail_fast=args.fail_fast,
            latents_output=latents_output,
        )

    def get_vec(path_value: object) -> np.ndarray | None:
        key = str(path_value).strip()
        vec = vecs.get(key)
        if _is_valid_vector(vec):
            return vec
        vec = vecs.get(_norm_key(key))
        return vec if _is_valid_vector(vec) else None

    def get_keypoint_count(path_value: object) -> int:
        key = str(path_value).strip()
        return int(keypoint_counts.get(key, keypoint_counts.get(_norm_key(key), 0)))

    n = len(df)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]
    print(f"Computing {len(pairs)} pairs")

    rows: list[dict] = []
    for pair_idx, (i, j) in enumerate(pairs, start=1):
        if pair_idx == 1 or pair_idx % 5000 == 0 or pair_idx == len(pairs):
            print(f"  Pair {pair_idx}/{len(pairs)}")
        row_i = df.iloc[i]
        row_j = df.iloc[j]
        path_i = row_i["image_path"]
        path_j = row_j["image_path"]
        v1 = get_vec(path_i)
        v2 = get_vec(path_j)
        sim = cosine_similarity(v1, v2)

        out = {}
        for column in df.columns:
            out[f"{column}_1"] = row_i[column]
            out[f"{column}_2"] = row_j[column]
        out["are_they_same_clusters"] = row_i["cluster_id"] == row_j["cluster_id"]
        out["sift_similarity_score"] = sim
        out["similarity_score"] = sim
        out["image_1_has_sift"] = v1 is not None
        out["image_2_has_sift"] = v2 is not None
        out["image_1_num_sift_keypoints"] = get_keypoint_count(path_i)
        out["image_2_num_sift_keypoints"] = get_keypoint_count(path_j)
        ms1 = row_i["manuscript_id"] if pd.notna(row_i["manuscript_id"]) else ""
        pic1 = row_i["picture_id"] if pd.notna(row_i["picture_id"]) else ""
        ms2 = row_j["manuscript_id"] if pd.notna(row_j["manuscript_id"]) else ""
        pic2 = row_j["picture_id"] if pd.notna(row_j["picture_id"]) else ""
        out["diagnose_cli"] = (
            f"python Debugs/impact_analysis/diagnose_similarity_issue.py "
            f"--ms1 {ms1} --pic1 {pic1} --ms2 {ms2} --pic2 {pic2}"
        )
        rows.append(out)

    pairs_df = pd.DataFrame(rows)
    score_col = "sift_similarity_score"
    metrics_rows = []
    metrics_rows.extend(pair_separation_metrics(pairs_df, score_col))
    metrics_rows.extend(cluster_retrieval_metrics(df, get_vec, score_col=score_col))
    metrics_df = pd.DataFrame(metrics_rows)

    write_outputs(output_path, pairs_df, metrics_df, metadata)
    print(f"Wrote {len(pairs_df)} pair rows to {output_path}")

    if not args.no_histogram:
        hist_path = output_path.with_name(output_path.stem + "_hist.png")
        try_plot_histogram(pairs_df, score_col, hist_path)
        if hist_path.exists():
            print(f"Wrote histogram to {hist_path}")

    if not metrics_df.empty:
        retrieval = metrics_df[metrics_df["metric_set"] == f"{score_col}/cluster_retrieval"]
        pair_metrics = metrics_df[metrics_df["metric_set"] == score_col]
        print("\n--- SIFT pair separation ---")
        for metric in ("n_pairs", "roc_auc", "pr_auc_average_precision", "same_mean", "different_mean", "mean_gap_same_minus_different"):
            values = pair_metrics.loc[pair_metrics["metric"] == metric, "value"]
            if len(values):
                value = values.iloc[0]
                print(f"  {metric}: {float(value):.6f}" if metric != "n_pairs" else f"  {metric}: {int(value)}")
        print("\n--- SIFT image retrieval ---")
        for metric in ("n_images_with_scores", "n_queries_with_relevant", "mAP", "recall@1", "recall@5", "recall@10", "hit@1", "hit@5", "hit@10"):
            values = retrieval.loc[retrieval["metric"] == metric, "value"]
            if len(values):
                value = values.iloc[0]
                print(f"  {metric}: {int(value)}" if metric.startswith("n_") else f"  {metric}: {float(value):.6f}")

    n_with_sift = sum(1 for path in image_paths if get_vec(path) is not None)
    print(f"\nSIFT vectors: {n_with_sift}/{len(image_paths)} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
