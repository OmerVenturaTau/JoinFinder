#!/usr/bin/env python3
"""Evaluate whether manuscript date is linearly encoded in each latent space."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from results_analysis.interpretability.probe.common import (  # noqa: E402
    DEFAULT_EXCLUDED_MANUSCRIPT_IDS,
    add_cv_folds,
    assert_split_integrity,
    fold_masks,
    get_connection,
    load_projection_rows,
    make_outer_split,
    manuscript_class_balanced_page_weights,
    normalize_spaces,
    outer_masks,
    safe_identifier,
    safe_path_component,
    save_json,
    space_matrix,
)
from system import (  # noqa: E402
    DATING_MANUSCRIPT_TABLE,
    DB_CONFIG_PATH,
    INTERPRETABILITY_EXPERIMENT_VECTORS_TABLE,
)


LOGGER = logging.getLogger("interpretability.dating_linear_regression")
ALPHA_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def aggregate_manuscript_vectors(
    frame: pd.DataFrame, matrix: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray]:
    """Mean-pool page vectors so every regression row is one manuscript."""
    labels_per_manuscript = frame.groupby("manuscript_id")["decade"].nunique()
    if (labels_per_manuscript != 1).any():
        raise ValueError("A manuscript has multiple dating labels")

    manuscript_ids = np.asarray(sorted(frame["manuscript_id"].astype(str).unique()))
    index_by_id = {value: index for index, value in enumerate(manuscript_ids)}
    row_indices = frame["manuscript_id"].astype(str).map(index_by_id).to_numpy(dtype=int)
    sums = np.zeros((len(manuscript_ids), matrix.shape[1]), dtype=np.float64)
    np.add.at(sums, row_indices, matrix)
    page_counts = np.bincount(row_indices, minlength=len(manuscript_ids))
    means = (sums / page_counts[:, None]).astype(np.float32)

    labels = (
        frame.assign(manuscript_id=frame["manuscript_id"].astype(str))
        .drop_duplicates("manuscript_id")
        .set_index("manuscript_id")["decade"]
        .reindex(manuscript_ids)
        .astype(int)
        .to_numpy()
    )
    manuscripts = pd.DataFrame(
        {
            "manuscript_id": manuscript_ids,
            "decade": labels,
            "num_pages": page_counts,
        }
    )
    return manuscripts, means


def nearest_bucket(values: np.ndarray, buckets: np.ndarray) -> np.ndarray:
    distances = np.abs(np.asarray(values)[:, None] - np.asarray(buckets)[None, :])
    return np.asarray(buckets)[distances.argmin(axis=1)]


def regression_metrics(
    labels: np.ndarray, predictions: np.ndarray, buckets: np.ndarray
) -> dict:
    labels = np.asarray(labels, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    absolute_error = np.abs(labels - predictions)
    correlation = (
        None
        if np.ptp(labels) == 0 or np.ptp(predictions) == 0
        else float(spearmanr(labels, predictions).statistic)
    )
    rounded = nearest_bucket(predictions, buckets)
    return {
        "mae_years": float(mean_absolute_error(labels, predictions)),
        "rmse_years": float(mean_squared_error(labels, predictions) ** 0.5),
        "r2": float(r2_score(labels, predictions)),
        "spearman_rho": correlation,
        "within_25_years": float(np.mean(absolute_error <= 25.0)),
        "within_50_years": float(np.mean(absolute_error <= 50.0)),
        "within_100_years": float(np.mean(absolute_error <= 100.0)),
        "nearest_bucket_accuracy": float(np.mean(rounded == labels)),
        "num_manuscripts": int(len(labels)),
    }


def fit_ridge(
    values: np.ndarray,
    labels: np.ndarray,
    manuscript_ids: np.ndarray,
    *,
    alpha: float,
) -> Ridge:
    model = Ridge(alpha=float(alpha), solver="lsqr")
    model.fit(
        values,
        labels,
        sample_weight=manuscript_class_balanced_page_weights(manuscript_ids, labels),
    )
    return model


def run_space(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    assignments: pd.DataFrame,
    *,
    space: str,
    output_dir: Path,
) -> dict:
    manuscripts_frame, manuscript_matrix = aggregate_manuscript_vectors(frame, matrix)
    manuscript_ids = manuscripts_frame["manuscript_id"].to_numpy(dtype=str)
    labels = manuscripts_frame["decade"].to_numpy(dtype=float)
    train_mask, test_mask = outer_masks(manuscript_ids, assignments)
    buckets = np.sort(assignments.loc[assignments["evaluable"], "decade"].unique())

    cv_rows = []
    for alpha in ALPHA_VALUES:
        fold_mae = []
        for fold in range(5):
            fold_train, fold_validation = fold_masks(manuscript_ids, assignments, fold)
            scaler = StandardScaler().fit(manuscript_matrix[fold_train])
            train_values = scaler.transform(manuscript_matrix[fold_train])
            validation_values = scaler.transform(manuscript_matrix[fold_validation])
            model = fit_ridge(
                train_values,
                labels[fold_train],
                manuscript_ids[fold_train],
                alpha=alpha,
            )
            fold_mae.append(
                float(mean_absolute_error(labels[fold_validation], model.predict(validation_values)))
            )
        cv_rows.append(
            {
                "alpha": float(alpha),
                "fold_manuscript_mae_years": fold_mae,
                "mean_manuscript_mae_years": float(np.mean(fold_mae)),
                "std_manuscript_mae_years": float(np.std(fold_mae)),
            }
        )
    best = min(cv_rows, key=lambda row: (row["mean_manuscript_mae_years"], row["alpha"]))

    scaler = StandardScaler().fit(manuscript_matrix[train_mask])
    model = fit_ridge(
        scaler.transform(manuscript_matrix[train_mask]),
        labels[train_mask],
        manuscript_ids[train_mask],
        alpha=best["alpha"],
    )
    predictions = model.predict(scaler.transform(manuscript_matrix[test_mask]))
    test = manuscripts_frame.loc[test_mask].reset_index(drop=True)
    test_labels = labels[test_mask]
    baseline_year = float(np.median(labels[train_mask]))
    baseline_predictions = np.full(len(test_labels), baseline_year)
    rounded = nearest_bucket(predictions, buckets)
    prediction_frame = test.assign(
        predicted_year=predictions,
        predicted_bucket=rounded.astype(int),
        absolute_error_years=np.abs(test_labels - predictions),
    )

    coefficients = np.asarray(model.coef_).reshape(-1)
    original_scale_coefficients = coefficients / scaler.scale_
    original_scale_intercept = float(
        model.intercept_ - np.dot(coefficients, scaler.mean_ / scaler.scale_)
    )
    coefficient_frame = pd.DataFrame(
        {
            "latent_dimension": np.arange(len(coefficients), dtype=int),
            "standardized_coefficient_years_per_sd": coefficients,
            "absolute_standardized_coefficient": np.abs(coefficients),
            "original_scale_coefficient": original_scale_coefficients,
        }
    ).sort_values("absolute_standardized_coefficient", ascending=False)
    coefficient_frame["absolute_rank"] = np.arange(1, len(coefficient_frame) + 1)
    metrics = {
        "space": space,
        "input_dimension": int(matrix.shape[1]),
        "selected_alpha": float(best["alpha"]),
        "test": regression_metrics(test_labels, predictions, buckets),
        "median_baseline_year": baseline_year,
        "median_baseline": regression_metrics(test_labels, baseline_predictions, buckets),
        "coefficient_norms": {
            "l1": float(np.linalg.norm(coefficients, ord=1)),
            "l2": float(np.linalg.norm(coefficients, ord=2)),
            "max_abs": float(np.abs(coefficients).max()),
        },
        "intercepts": {
            "standardized_space": float(model.intercept_),
            "original_space": original_scale_intercept,
        },
        "coverage": {
            "pages": int(len(frame)),
            "manuscripts": int(len(manuscripts_frame)),
            "test_manuscripts": int(test_mask.sum()),
        },
    }
    space_dir = output_dir / space
    space_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(space_dir / "manuscript_predictions.csv", index=False)
    coefficient_frame.to_csv(space_dir / "coefficients.csv", index=False)
    save_json(space_dir / "metrics.json", metrics)
    save_json(space_dir / "cv_summary.json", cv_rows)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection-run", required=True)
    parser.add_argument("--spaces", nargs="+", default=["all"])
    parser.add_argument("--vectors-table", default=INTERPRETABILITY_EXPERIMENT_VECTORS_TABLE)
    parser.add_argument("--dating-table", default=DATING_MANUSCRIPT_TABLE)
    parser.add_argument("--db-config", default=DB_CONFIG_PATH)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rare-min-manuscripts", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spaces = normalize_spaces(args.spaces)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(__file__).parent
        / "outputs"
        / "dating_linear_regression"
        / safe_path_component(args.projection_run)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = get_connection(args.db_config)
    try:
        vectors = load_projection_rows(
            conn,
            projection_run=args.projection_run,
            vectors_table=args.vectors_table,
            spaces=spaces,
        )
        dating_table = safe_identifier(args.dating_table)
        dating = pd.read_sql_query(
            f"""
            SELECT manuscript_id, decade::integer AS decade
            FROM {dating_table}
            WHERE is_decade = TRUE AND decade IS NOT NULL
            """,
            conn,
        )
    finally:
        conn.close()

    vectors["manuscript_id"] = vectors["manuscript_id"].astype(str)
    dating["manuscript_id"] = dating["manuscript_id"].astype(str)
    frame = vectors.merge(dating, on="manuscript_id", how="inner", validate="many_to_one")
    if frame.empty:
        raise RuntimeError("No projected manuscripts overlap the dating table")

    assignments = make_outer_split(
        frame,
        label_column="decade",
        seed=args.seed,
        rare_min_manuscripts=args.rare_min_manuscripts,
    )
    assignments = add_cv_folds(assignments, label_column="decade", seed=args.seed)
    assert_split_integrity(assignments)
    assignments.to_csv(output_dir / "split_assignments.csv", index=False)
    save_json(
        output_dir / "config.json",
        {
            "projection_run": args.projection_run,
            "spaces": spaces,
            "seed": args.seed,
            "model": "ridge regression on mean-pooled manuscript vectors",
            "alpha_values": list(ALPHA_VALUES),
            "cv_selection_metric": "manuscript MAE in years",
            "rare_min_manuscripts": args.rare_min_manuscripts,
            "rare_class_policy": "excluded from probe fitting and evaluation",
            "training_balance": "equal total weight per dating bucket and manuscript",
            "excluded_manuscript_ids": list(DEFAULT_EXCLUDED_MANUSCRIPT_IDS),
        },
    )

    evaluable_ids = set(
        assignments.loc[assignments["evaluable"], "manuscript_id"].astype(str)
    )
    probe_frame = frame[frame["manuscript_id"].isin(evaluable_ids)].copy()
    results = []
    for space in spaces:
        space_frame, matrix = space_matrix(probe_frame, space)
        if matrix.size == 0:
            LOGGER.warning("Skipping %s: no stored vectors", space)
            continue
        LOGGER.info("Running dating linear regression in %s space", space)
        results.append(
            run_space(
                space_frame,
                matrix,
                assignments,
                space=space,
                output_dir=output_dir,
            )
        )
    if not results:
        raise RuntimeError("No latent space had enough coverage to run the dating regression")
    pd.json_normalize(results).to_csv(output_dir / "metrics_by_space.csv", index=False)
    save_json(output_dir / "summary.json", results)
    LOGGER.info("Dating linear-regression outputs written to %s", output_dir)


if __name__ == "__main__":
    main()
