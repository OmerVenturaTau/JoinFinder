from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from Drafts.create_interpretability_experiment_vectors_table import create_table_sql
from results_analysis.interpretability.probe.common import (
    DEFAULT_EXCLUDED_MANUSCRIPT_IDS,
    add_cv_folds,
    assert_split_integrity,
    load_projection_rows,
    make_outer_split,
    manuscript_class_balanced_page_weights,
    manuscript_page_weights,
    page_sample_weights,
    space_matrix,
)
from results_analysis.interpretability.probe.project_manuscripts import (
    _branch_value,
    read_manuscript_file,
    resolve_manuscript_ids,
)
from results_analysis.interpretability.probe.run_dating_probe import run_space as run_dating_space
from results_analysis.interpretability.probe.run_dating_linear_regression import (
    aggregate_manuscript_vectors,
    nearest_bucket,
    regression_metrics,
)
from results_analysis.interpretability.probe.run_oriental_probe import run_space as run_oriental_space
from system import INTERPRETABILITY_EXPERIMENT_VECTORS_TABLE


def test_interpretability_table_name_and_dimension_flexible_sql():
    assert INTERPRETABILITY_EXPERIMENT_VECTORS_TABLE == "interpretability_experiment_vectors"
    sql = create_table_sql(INTERPRETABILITY_EXPERIMENT_VECTORS_TABLE)
    assert "CREATE TABLE IF NOT EXISTS interpretability_experiment_vectors" in sql
    assert "shared_vector vector NOT NULL" in sql
    assert "vector(1536)" not in sql
    assert "PRIMARY KEY (projection_run, image_path)" in sql
    assert sql == create_table_sql(INTERPRETABILITY_EXPERIMENT_VECTORS_TABLE)


def test_manuscript_list_parsing_and_resolution(tmp_path):
    text_path = tmp_path / "manuscripts.txt"
    text_path.write_text("m2\n\nm1\nm2\n", encoding="utf-8")
    csv_path = tmp_path / "manuscripts.csv"
    csv_path.write_text("manuscript_id\nm4\nm3\n", encoding="utf-8")
    assert read_manuscript_file(str(text_path)) == ["m1", "m2"]
    assert read_manuscript_file(str(csv_path)) == ["m3", "m4"]
    assert resolve_manuscript_ids(["m0", "m2"], str(text_path)) == ["m0", "m1", "m2"]


def test_projection_loader_excludes_known_manuscripts_before_probes(monkeypatch):
    captured = {}

    def fake_read_sql_query(query, conn, params):
        captured.update(query=query, conn=conn, params=params)
        return pd.DataFrame()

    monkeypatch.setattr(pd, "read_sql_query", fake_read_sql_query)
    connection = object()
    load_projection_rows(
        connection,
        projection_run="experiment",
        vectors_table=INTERPRETABILITY_EXPERIMENT_VECTORS_TABLE,
        spaces=["shared"],
    )

    assert "NOT (manuscript_id = ANY(%s))" in captured["query"]
    assert captured["conn"] is connection
    assert captured["params"][0] == "experiment"
    assert set(captured["params"][1]) == set(DEFAULT_EXCLUDED_MANUSCRIPT_IDS)
    assert DEFAULT_EXCLUDED_MANUSCRIPT_IDS == (
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


def test_unavailable_branch_is_null_and_available_branch_is_serialized():
    aux = {"tile": torch.tensor([[3.0, 4.0]])}
    assert _branch_value(aux, "tile", 0, evidence_count=0, enabled=True) == (None, None)
    assert _branch_value(aux, "tile", 0, evidence_count=2, enabled=False) == (None, None)
    dimension, vector = _branch_value(aux, "tile", 0, evidence_count=2, enabled=True)
    assert dimension == 2
    assert vector == "[3,4]"


def _split_frame() -> pd.DataFrame:
    rows = []
    for label in (1200, 1250, 1300):
        for index in range(20):
            rows.append({"manuscript_id": f"{label}_{index}", "decade": label})
    for index in range(5):
        rows.append({"manuscript_id": f"rare_{index}", "decade": 1100})
    return pd.DataFrame(rows)


def test_split_is_70_30_by_manuscript_with_rare_classes_excluded():
    assignments = make_outer_split(
        _split_frame(),
        label_column="decade",
        seed=42,
        rare_min_manuscripts=10,
    )
    assignments = add_cv_folds(assignments, label_column="decade", seed=42)
    assert_split_integrity(assignments)
    rare = assignments[assignments["decade"] == 1100]
    assert set(rare["split"]) == {"excluded"}
    assert set(rare["cv_fold"]) == {-1}
    common = assignments[assignments["decade"] != 1100]
    assert (common["split"] == "test").sum() == 18
    common_train = common[common["split"] == "train"]
    assert set(common_train["cv_fold"]) == {0, 1, 2, 3, 4}
    assert not set(assignments.loc[assignments["split"] == "train", "manuscript_id"]) & set(
        assignments.loc[assignments["split"] == "test", "manuscript_id"]
    )


def test_space_matrix_filters_null_vectors_and_checks_dimension():
    frame = pd.DataFrame(
        {
            "manuscript_id": ["a", "b", "c"],
            "tile_vector": ["[1,2]", None, "[3,4]"],
        }
    )
    selected, matrix = space_matrix(frame, "tile")
    assert selected["manuscript_id"].tolist() == ["a", "c"]
    np.testing.assert_allclose(matrix, [[1, 2], [3, 4]])


def test_manuscript_page_weights_equalize_page_counts():
    manuscript_ids = np.array(["a", "a", "a", "b"])
    weights = manuscript_page_weights(manuscript_ids)
    assert np.isclose(weights[:3].sum(), weights[3:].sum())


def test_manuscript_class_balanced_weights_retain_rows_and_balance_classes():
    manuscript_ids = np.array(["negative_many", "negative_many", "negative_other", "positive"])
    labels = np.array([0, 0, 0, 1])
    weights = manuscript_class_balanced_page_weights(manuscript_ids, labels)

    assert len(weights) == len(manuscript_ids)
    assert np.isclose(weights[labels == 0].sum(), weights[labels == 1].sum())
    assert np.isclose(weights[:2].sum(), weights[2])
    np.testing.assert_allclose(page_sample_weights(manuscript_ids, labels), weights)


def _synthetic_probe_frame(label_column: str, labels: tuple[int, int]):
    rows = []
    vectors = []
    for label_index, label in enumerate(labels):
        center = -3.0 if label_index == 0 else 3.0
        for manuscript_index in range(10):
            manuscript_id = f"{label}_{manuscript_index}"
            for page in range(2):
                rows.append(
                    {
                        "manuscript_id": manuscript_id,
                        "image_path": f"/{manuscript_id}/{page}.jpg",
                        label_column: label,
                    }
                )
                vectors.append([center, center + 0.1 * page, float(page), 1.0])
    return pd.DataFrame(rows), np.asarray(vectors, dtype=np.float32)


def test_dating_probe_runs_on_synthetic_separable_vectors(tmp_path):
    frame, matrix = _synthetic_probe_frame("decade", (1200, 1250))
    assignments = make_outer_split(
        frame,
        label_column="decade",
        seed=42,
        rare_min_manuscripts=10,
    )
    assignments = add_cv_folds(assignments, label_column="decade", seed=42)
    result = run_dating_space(
        frame,
        matrix,
        assignments,
        space="shared",
        output_dir=tmp_path,
        seed=42,
        max_epochs=12,
        patience=3,
        batch_size=16,
        device=torch.device("cpu"),
    )
    assert result["manuscript"]["accuracy_top1"] >= 0.5
    assert (tmp_path / "shared" / "metrics.json").exists()
    assert (tmp_path / "shared" / "manuscript_predictions.csv").exists()


def test_dating_linear_regression_helpers_pool_pages_and_measure_year_error():
    frame, matrix = _synthetic_probe_frame("decade", (1200, 1250))
    manuscripts, pooled = aggregate_manuscript_vectors(frame, matrix)

    assert len(manuscripts) == 20
    assert pooled.shape == (20, 4)
    assert set(manuscripts["num_pages"]) == {2}
    np.testing.assert_array_equal(
        nearest_bucket(np.array([1210.0, 1240.0]), np.array([1200, 1250])),
        [1200, 1250],
    )
    metrics = regression_metrics(
        np.array([1200.0, 1250.0]),
        np.array([1210.0, 1240.0]),
        np.array([1200, 1250]),
    )
    assert metrics["mae_years"] == 10.0
    assert metrics["nearest_bucket_accuracy"] == 1.0


def test_oriental_probe_runs_on_synthetic_separable_vectors(tmp_path):
    frame, matrix = _synthetic_probe_frame("is_oriental", (0, 1))
    assignments = make_outer_split(
        frame,
        label_column="is_oriental",
        seed=42,
    )
    assignments = add_cv_folds(assignments, label_column="is_oriental", seed=42)
    result = run_oriental_space(
        frame,
        matrix,
        assignments,
        space="shared",
        output_dir=tmp_path,
        seed=42,
    )
    assert result["manuscript"]["roc_auc"] >= 0.9
    assert (tmp_path / "shared" / "cv_summary.json").exists()
    assert (tmp_path / "shared" / "confusion_matrix.png").exists()
