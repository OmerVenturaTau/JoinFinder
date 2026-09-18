#!/usr/bin/env python3
"""Evaluate 50-year manuscript dating with a small MLP in each latent space."""

from __future__ import annotations

import argparse
import copy
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from results_analysis.interpretability.probe.common import (  # noqa: E402
    DEFAULT_EXCLUDED_MANUSCRIPT_IDS,
    add_cv_folds,
    aggregate_probabilities,
    assert_split_integrity,
    fit_temperature,
    fold_masks,
    get_connection,
    image_prediction_frame,
    load_projection_rows,
    logits_to_probabilities,
    make_outer_split,
    multiclass_metrics,
    normalize_spaces,
    outer_masks,
    page_sample_weights,
    safe_identifier,
    safe_path_component,
    save_confusion_plot,
    save_json,
    space_matrix,
)
from system import (  # noqa: E402
    DATING_MANUSCRIPT_TABLE,
    DB_CONFIG_PATH,
    INTERPRETABILITY_EXPERIMENT_VECTORS_TABLE,
)


LOGGER = logging.getLogger("interpretability.dating_probe")


class SmallDatingMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict_logits(
    model: nn.Module, values: np.ndarray, *, device: torch.device, batch_size: int
) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size]).to(device)
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 0))


def train_epoch_count(
    values: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    input_dim: int,
    num_classes: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> nn.Module:
    set_seed(seed)
    model = SmallDatingMLP(input_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dataset = TensorDataset(
        torch.from_numpy(values),
        torch.from_numpy(labels.astype(np.int64)),
        torch.from_numpy(weights.astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    for _ in range(max(1, int(epochs))):
        model.train()
        for batch_values, batch_labels, batch_weights in loader:
            batch_values = batch_values.to(device)
            batch_labels = batch_labels.to(device)
            batch_weights = batch_weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = nn.functional.cross_entropy(
                model(batch_values), batch_labels, reduction="none"
            )
            loss = (losses * batch_weights).sum() / batch_weights.sum().clamp_min(1e-8)
            loss.backward()
            optimizer.step()
    return model


def fit_fold_with_early_stopping(
    train_values: np.ndarray,
    train_labels: np.ndarray,
    train_weights: np.ndarray,
    validation_values: np.ndarray,
    validation_labels: np.ndarray,
    validation_manuscripts: np.ndarray,
    *,
    class_names: list[int],
    max_epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[int, np.ndarray, float]:
    set_seed(seed)
    model = SmallDatingMLP(train_values.shape[1], len(class_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dataset = TensorDataset(
        torch.from_numpy(train_values),
        torch.from_numpy(train_labels.astype(np.int64)),
        torch.from_numpy(train_weights.astype(np.float32)),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_epoch = 1
    best_score = float("-inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_values, batch_labels, batch_weights in loader:
            batch_values = batch_values.to(device)
            batch_labels = batch_labels.to(device)
            batch_weights = batch_weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = nn.functional.cross_entropy(
                model(batch_values), batch_labels, reduction="none"
            )
            loss = (losses * batch_weights).sum() / batch_weights.sum().clamp_min(1e-8)
            loss.backward()
            optimizer.step()
        logits = predict_logits(model, validation_values, device=device, batch_size=batch_size)
        probabilities = logits_to_probabilities(logits)
        manuscripts = aggregate_probabilities(
            validation_manuscripts,
            validation_labels,
            probabilities,
            class_names,
        )
        score = f1_score(
            manuscripts["true_index"],
            manuscripts["predicted_index"],
            labels=np.unique(manuscripts["true_index"]),
            average="macro",
            zero_division=0,
        )
        if score > best_score + 1e-8:
            best_score = float(score)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    model.load_state_dict(best_state)
    validation_logits = predict_logits(
        model, validation_values, device=device, batch_size=batch_size
    )
    return best_epoch, validation_logits, best_score


def _probability_matrix(frame: pd.DataFrame, class_names: list[int]) -> np.ndarray:
    return frame[[f"probability_{name}" for name in class_names]].to_numpy(dtype=float)


def run_space(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    assignments: pd.DataFrame,
    *,
    space: str,
    output_dir: Path,
    seed: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
) -> dict:
    class_names = sorted(
        assignments.loc[assignments["evaluable"], "decade"].astype(int).unique().tolist()
    )
    class_to_index = {label: index for index, label in enumerate(class_names)}
    labels = frame["decade"].astype(int).map(class_to_index).to_numpy(dtype=int)
    manuscripts = frame["manuscript_id"].astype(str).to_numpy()
    train_mask, test_mask = outer_masks(manuscripts, assignments)
    if not train_mask.any() or not test_mask.any():
        raise ValueError(f"{space} does not cover both train and test manuscripts")

    oof_logits = np.full((len(frame), len(class_names)), np.nan, dtype=np.float64)
    fold_results = []
    for fold in range(5):
        fold_train, fold_validation = fold_masks(manuscripts, assignments, fold)
        if not fold_train.any() or not fold_validation.any():
            raise ValueError(f"{space} has no data for CV fold {fold}")
        scaler = StandardScaler().fit(matrix[fold_train])
        train_values = scaler.transform(matrix[fold_train]).astype(np.float32)
        validation_values = scaler.transform(matrix[fold_validation]).astype(np.float32)
        weights = page_sample_weights(manuscripts[fold_train], labels[fold_train])
        best_epoch, validation_logits, score = fit_fold_with_early_stopping(
            train_values,
            labels[fold_train],
            weights,
            validation_values,
            labels[fold_validation],
            manuscripts[fold_validation],
            class_names=class_names,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            device=device,
            seed=seed + fold,
        )
        oof_logits[fold_validation] = validation_logits
        fold_results.append(
            {
                "fold": fold,
                "best_epoch": best_epoch,
                "validation_manuscript_macro_f1": score,
                "train_pages": int(fold_train.sum()),
                "validation_pages": int(fold_validation.sum()),
            }
        )

    calibrated = np.isfinite(oof_logits).all(axis=1)
    temperature = fit_temperature(oof_logits[calibrated], labels[calibrated])
    selected_epoch = max(1, int(round(np.median([row["best_epoch"] for row in fold_results]))))
    final_scaler = StandardScaler().fit(matrix[train_mask])
    final_train = final_scaler.transform(matrix[train_mask]).astype(np.float32)
    final_test = final_scaler.transform(matrix[test_mask]).astype(np.float32)
    final_weights = page_sample_weights(manuscripts[train_mask], labels[train_mask])
    model = train_epoch_count(
        final_train,
        labels[train_mask],
        final_weights,
        input_dim=matrix.shape[1],
        num_classes=len(class_names),
        epochs=selected_epoch,
        batch_size=batch_size,
        device=device,
        seed=seed,
    )
    test_logits = predict_logits(model, final_test, device=device, batch_size=batch_size)
    probabilities = logits_to_probabilities(test_logits, temperature)
    test_frame = frame.loc[test_mask].reset_index(drop=True)
    test_labels = labels[test_mask]
    image_predictions = image_prediction_frame(
        test_frame["manuscript_id"],
        test_frame["image_path"],
        test_labels,
        probabilities,
        class_names,
    )
    manuscript_predictions = aggregate_probabilities(
        test_frame["manuscript_id"], test_labels, probabilities, class_names
    )
    manuscript_probabilities = _probability_matrix(manuscript_predictions, class_names)

    training_manuscript_labels = assignments.loc[
        assignments["split"] == "train", "decade"
    ].astype(int)
    majority_label = int(training_manuscript_labels.mode().iloc[0])
    metrics = {
        "space": space,
        "input_dimension": int(matrix.shape[1]),
        "temperature": temperature,
        "selected_epoch": selected_epoch,
        "image": multiclass_metrics(test_labels, probabilities),
        "manuscript": multiclass_metrics(
            manuscript_predictions["true_index"], manuscript_probabilities
        ),
        "majority_label": majority_label,
        "majority_manuscript_accuracy": float(
            (manuscript_predictions["true_label"].astype(int) == majority_label).mean()
        ),
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
    save_json(space_dir / "cv_summary.json", fold_results)
    save_confusion_plot(
        space_dir / "confusion_matrix.png",
        manuscript_predictions["true_index"],
        manuscript_predictions["predicted_index"],
        class_names,
        f"Dating probe: {space}",
    )
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
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spaces = normalize_spaces(args.spaces)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(__file__).parent / "outputs" / "dating" / safe_path_component(args.projection_run)
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
    if vectors.empty:
        raise RuntimeError(f"No vectors found for projection run {args.projection_run!r}")
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
            "outer_train_fraction": 0.70,
            "outer_test_fraction": 0.30,
            "cv_folds": 5,
            "rare_min_manuscripts": args.rare_min_manuscripts,
            "rare_class_policy": "excluded from probe fitting and evaluation",
            "training_balance": (
                "equal class weight by unique manuscript; equal manuscript weight "
                "within class; all pages retained"
            ),
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "device": str(device),
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
        LOGGER.info("Running dating probe in %s space on %d pages", space, len(space_frame))
        try:
            results.append(
                run_space(
                    space_frame,
                    matrix,
                    assignments,
                    space=space,
                    output_dir=output_dir,
                    seed=args.seed,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    batch_size=args.batch_size,
                    device=device,
                )
            )
        except ValueError as exc:
            LOGGER.warning("Skipping %s: %s", space, exc)
    if not results:
        raise RuntimeError("No latent space had enough coverage to run the dating probe")
    pd.json_normalize(results).to_csv(output_dir / "metrics_by_space.csv", index=False)
    save_json(output_dir / "summary.json", results)
    LOGGER.info("Dating probe outputs written to %s", output_dir)


if __name__ == "__main__":
    main()
