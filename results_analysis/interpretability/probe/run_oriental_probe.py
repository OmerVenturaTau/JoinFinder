#!/usr/bin/env python3
"""Evaluate oriental manuscript status with a linear logistic probe."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from results_analysis.interpretability.probe.common import (  # noqa: E402
    DEFAULT_EXCLUDED_MANUSCRIPT_IDS,
    add_cv_folds,
    aggregate_probabilities,
    assert_split_integrity,
    binary_metrics,
    fold_masks,
    get_connection,
    image_prediction_frame,
    load_projection_rows,
    make_outer_split,
    manuscript_class_balanced_page_weights,
    normalize_spaces,
    outer_masks,
    safe_path_component,
    save_confusion_plot,
    save_json,
    space_matrix,
)
from system import (  # noqa: E402
    DB_CONFIG_PATH,
    INTERPRETABILITY_EXPERIMENT_VECTORS_TABLE,
)


LOGGER = logging.getLogger("interpretability.oriental_probe")
C_VALUES = (0.01, 0.1, 1.0, 10.0)


def _probability_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[["probability_0", "probability_1"]].to_numpy(dtype=float)


def fit_logistic(
    train_values: np.ndarray,
    train_labels: np.ndarray,
    train_manuscripts: np.ndarray,
    *,
    c_value: float,
    seed: int,
) -> LogisticRegression:
    model = LogisticRegression(
        C=float(c_value),
        max_iter=5000,
        solver="lbfgs",
        random_state=seed,
    )
    model.fit(
        train_values,
        train_labels,
        sample_weight=manuscript_class_balanced_page_weights(
            train_manuscripts, train_labels
        ),
    )
    return model


def run_space(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    assignments: pd.DataFrame,
    *,
    space: str,
    output_dir: Path,
    seed: int,
) -> dict:
    labels = frame["is_oriental"].astype(bool).astype(int).to_numpy()
    manuscripts = frame["manuscript_id"].astype(str).to_numpy()
    train_mask, test_mask = outer_masks(manuscripts, assignments)
    if not train_mask.any() or not test_mask.any():
        raise ValueError(f"{space} does not cover both train and test manuscripts")

    cv_rows = []
    for c_value in C_VALUES:
        fold_scores = []
        for fold in range(5):
            fold_train, fold_validation = fold_masks(manuscripts, assignments, fold)
            if not fold_train.any() or not fold_validation.any():
                raise ValueError(f"{space} has no data for CV fold {fold}")
            scaler = StandardScaler().fit(matrix[fold_train])
            train_values = scaler.transform(matrix[fold_train])
            validation_values = scaler.transform(matrix[fold_validation])
            model = fit_logistic(
                train_values,
                labels[fold_train],
                manuscripts[fold_train],
                c_value=c_value,
                seed=seed + fold,
            )
            probabilities = model.predict_proba(validation_values)
            aggregated = aggregate_probabilities(
                manuscripts[fold_validation], labels[fold_validation], probabilities, [0, 1]
            )
            score = roc_auc_score(
                aggregated["true_index"], aggregated["probability_1"]
            )
            fold_scores.append(float(score))
        cv_rows.append(
            {
                "C": float(c_value),
                "fold_roc_auc": fold_scores,
                "mean_manuscript_roc_auc": float(np.mean(fold_scores)),
                "std_manuscript_roc_auc": float(np.std(fold_scores)),
            }
        )
    # Stable tie break: prefer the smaller C (stronger regularization).
    best = max(cv_rows, key=lambda row: (row["mean_manuscript_roc_auc"], -row["C"]))
    best_c = float(best["C"])

    scaler = StandardScaler().fit(matrix[train_mask])
    train_values = scaler.transform(matrix[train_mask])
    test_values = scaler.transform(matrix[test_mask])
    model = fit_logistic(
        train_values,
        labels[train_mask],
        manuscripts[train_mask],
        c_value=best_c,
        seed=seed,
    )
    probabilities = model.predict_proba(test_values)
    test_frame = frame.loc[test_mask].reset_index(drop=True)
    test_labels = labels[test_mask]
    image_predictions = image_prediction_frame(
        test_frame["manuscript_id"],
        test_frame["image_path"],
        test_labels,
        probabilities,
        [0, 1],
    )
    manuscript_predictions = aggregate_probabilities(
        test_frame["manuscript_id"], test_labels, probabilities, [0, 1]
    )
    manuscript_probabilities = _probability_matrix(manuscript_predictions)
    coefficients = model.coef_.reshape(-1)
    metrics = {
        "space": space,
        "input_dimension": int(matrix.shape[1]),
        "selected_C": best_c,
        "image": binary_metrics(test_labels, probabilities),
        "manuscript": binary_metrics(
            manuscript_predictions["true_index"], manuscript_probabilities
        ),
        "coefficient_norms": {
            "l1": float(np.linalg.norm(coefficients, ord=1)),
            "l2": float(np.linalg.norm(coefficients, ord=2)),
            "max_abs": float(np.abs(coefficients).max()),
        },
        "coverage": {
            "pages": int(len(frame)),
            "manuscripts": int(frame["manuscript_id"].nunique()),
            "test_pages": int(test_mask.sum()),
            "test_manuscripts": int(test_frame["manuscript_id"].nunique()),
        },
    }
    space_dir = output_dir / space
    space_dir.mkdir(parents=True, exist_ok=True)
    image_predictions.to_csv(space_dir / "image_predictions.csv", index=False)
    manuscript_predictions.to_csv(space_dir / "manuscript_predictions.csv", index=False)
    save_json(space_dir / "metrics.json", metrics)
    save_json(space_dir / "cv_summary.json", cv_rows)
    save_confusion_plot(
        space_dir / "confusion_matrix.png",
        manuscript_predictions["true_index"],
        manuscript_predictions["predicted_index"],
        ["non-oriental", "oriental"],
        f"Oriental logistic probe: {space}",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection-run", required=True)
    parser.add_argument("--spaces", nargs="+", default=["all"])
    parser.add_argument("--vectors-table", default=INTERPRETABILITY_EXPERIMENT_VECTORS_TABLE)
    parser.add_argument("--db-config", default=DB_CONFIG_PATH)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spaces = normalize_spaces(args.spaces)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(__file__).parent / "outputs" / "oriental" / safe_path_component(args.projection_run)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    conn = get_connection(args.db_config)
    try:
        frame = load_projection_rows(
            conn,
            projection_run=args.projection_run,
            vectors_table=args.vectors_table,
            spaces=spaces,
        )
    finally:
        conn.close()
    if frame.empty:
        raise RuntimeError(f"No vectors found for projection run {args.projection_run!r}")
    frame = frame.dropna(subset=["is_oriental"]).copy()
    frame["manuscript_id"] = frame["manuscript_id"].astype(str)
    frame["is_oriental"] = frame["is_oriental"].astype(bool).astype(int)
    assignments = make_outer_split(
        frame,
        label_column="is_oriental",
        seed=args.seed,
    )
    assignments = add_cv_folds(
        assignments, label_column="is_oriental", seed=args.seed
    )
    assert_split_integrity(assignments)
    assignments.to_csv(output_dir / "split_assignments.csv", index=False)
    save_json(
        output_dir / "config.json",
        {
            "projection_run": args.projection_run,
            "spaces": spaces,
            "seed": args.seed,
            "outer_train_fraction": 0.70,
            "outer_test_fraction": 0.30,
            "cv_folds": 5,
            "C_values": list(C_VALUES),
            "training_balance": (
                "equal class weight by unique manuscript; equal manuscript weight "
                "within class; all pages retained"
            ),
            "excluded_manuscript_ids": list(DEFAULT_EXCLUDED_MANUSCRIPT_IDS),
        },
    )

    results = []
    for space in spaces:
        space_frame, matrix = space_matrix(frame, space)
        if matrix.size == 0:
            LOGGER.warning("Skipping %s: no stored vectors", space)
            continue
        LOGGER.info("Running oriental probe in %s space on %d pages", space, len(space_frame))
        try:
            results.append(
                run_space(
                    space_frame,
                    matrix,
                    assignments,
                    space=space,
                    output_dir=output_dir,
                    seed=args.seed,
                )
            )
        except ValueError as exc:
            LOGGER.warning("Skipping %s: %s", space, exc)
    if not results:
        raise RuntimeError("No latent space had enough coverage to run the oriental probe")
    pd.json_normalize(results).to_csv(output_dir / "metrics_by_space.csv", index=False)
    save_json(output_dir / "summary.json", results)
    LOGGER.info("Oriental probe outputs written to %s", output_dir)


if __name__ == "__main__":
    main()
