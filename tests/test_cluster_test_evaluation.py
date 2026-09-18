import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import train.trainer as trainer
import train.dataset as dataset_module
import system


def test_pca_diagnostic_preserves_raw_variance_geometry():
    dominant = np.linspace(-10.0, 10.0, 200, dtype=np.float32)
    small = np.tile(np.array([-0.1, 0.1], dtype=np.float32), 100)
    latents = np.column_stack((dominant, small))

    _, threshold_dim, _, explained, _, _, scaler = (
        trainer.compute_pca_elbow_dimension(latents)
    )

    assert threshold_dim == 1
    assert explained[0] > 0.99
    assert scaler is None


def _touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return str(path)


def test_csv_adapter_keeps_member_rows_and_validates_counts(tmp_path):
    image_paths = [_touch(tmp_path / "images" / f"p{i}.jpg") for i in range(4)]
    xml_paths = [_touch(tmp_path / "xml" / f"p{i}.xml") for i in range(4)]
    source = tmp_path / "members.csv"
    pd.DataFrame({
        "picture_id": [f"p{i}" for i in range(4)],
        "image_path": image_paths,
        "xml_path": xml_paths,
        "cluster_id": ["a", "a", "b", "b"],
    }).to_csv(source, index=False)

    members = trainer.load_cluster_test_members(
        "old", str(source), expected_rows=4, expected_clusters=2
    )

    assert list(members.columns) == ["image_path", "xml_path", "cluster_id"]
    assert members.attrs["num_rows"] == 4
    assert members.attrs["num_clusters"] == 2


def test_xlsx_adapter_derives_geniza_image_and_xml_paths(tmp_path, monkeypatch):
    image_base = tmp_path / "NLI_GNIZA_jpgs"
    xml_base = tmp_path / "xml"
    monkeypatch.setattr(trainer, "GENIZA_IMAGE_BASE", str(image_base))
    monkeypatch.setattr(trainer, "GENIZA_XML_BASE", str(xml_base))
    monkeypatch.setattr(trainer, "GENIZA_XML_FILENAME_SUFFIX", "—suffix.xml")

    rows = []
    for index, cluster_id in enumerate(("a", "a", "b", "b")):
        manuscript = f"ms{index // 2}"
        parent = f"scan{index // 2}"
        filename = f"page{index}.jpg"
        _touch(image_base / manuscript / parent / filename)
        _touch(xml_base / manuscript / parent / f"page{index}—suffix.xml")
        rows.append({
            "cluster_id": cluster_id,
            "image_id": f"image-{index}",
            "manuscript_id": manuscript,
            "relative_path": f"NLI_GNIZA_jpgs/{manuscript}/{parent}/{filename}",
        })
    source = tmp_path / "cluster_members.xlsx"
    pd.DataFrame(rows).to_excel(source, sheet_name="members", index=False)

    members = trainer.load_cluster_test_members(
        "new",
        str(source),
        sheet_name="members",
        expected_rows=4,
        expected_clusters=2,
    )

    assert members.iloc[0]["image_path"] == str(
        image_base / "ms0" / "scan0" / "page0.jpg"
    )
    assert members.iloc[0]["xml_path"] == str(
        xml_base / "ms0" / "scan0" / "page0—suffix.xml"
    )


def test_adapter_rejects_singleton_cluster(tmp_path):
    image_paths = [_touch(tmp_path / f"p{i}.jpg") for i in range(3)]
    xml_paths = [_touch(tmp_path / f"p{i}.xml") for i in range(3)]
    source = tmp_path / "bad.csv"
    pd.DataFrame({
        "picture_id": ["p0", "p1", "p2"],
        "image_path": image_paths,
        "xml_path": xml_paths,
        "cluster_id": ["a", "a", "singleton"],
    }).to_csv(source, index=False)

    with pytest.raises(ValueError, match="fewer than two"):
        trainer.load_cluster_test_members("bad", str(source))


def test_adapter_rejects_missing_image_or_xml(tmp_path):
    existing_image = _touch(tmp_path / "p0.jpg")
    existing_xml = _touch(tmp_path / "p0.xml")
    source = tmp_path / "missing.csv"
    pd.DataFrame({
        "picture_id": ["p0", "p1"],
        "image_path": [existing_image, str(tmp_path / "missing.jpg")],
        "xml_path": [existing_xml, str(tmp_path / "missing.xml")],
        "cluster_id": ["a", "a"],
    }).to_csv(source, index=False)

    with pytest.raises(FileNotFoundError, match="missing_images=1.*missing_xml=1"):
        trainer.load_cluster_test_members("missing", str(source))


def test_branch_retrieval_metrics_are_exactly_map_and_knn_fields_and_exclude_self():
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    perfect = torch.tensor([
        [1.0, 0.0],
        [0.99, 0.01],
        [0.0, 1.0],
        [0.01, 0.99],
    ])
    branches = {
        name: torch.nn.functional.normalize(perfect * (index + 1), dim=1)
        for index, name in enumerate(trainer.RETRIEVAL_BRANCH_NAMES)
    }

    metrics = trainer._compute_cluster_branch_maps(branches, labels, "fixture")

    assert set(metrics) == {
        f"test/fixture/{metric}/{branch}"
        for metric in ("mAP", "knn_at_1", "knn_at_5", "knn_at_10")
        for branch in ("fusion", "tile", "glyph", "word")
    }
    assert all(value == pytest.approx(1.0) for value in metrics.values())


def test_cluster_test_skips_unavailable_optional_branch_rows_and_logs(monkeypatch, caplog):
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    features = torch.tensor([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
        [0.1, 0.9],
    ])
    glyph = features.clone()
    glyph[0].zero_()
    paths = [f"image-{index}.jpg" for index in range(4)]

    def fake_forward(_model, _batch, *, device, return_aux_latents):
        assert return_aux_latents is True
        return None, features, {
            "tile": features,
            "glyph": glyph,
            "word": features,
        }, labels, paths

    class Loader:
        class Dataset:
            cluster_test_num_rows = 4
            cluster_test_num_clusters = 2

            def __len__(self):
                return 4

        dataset = Dataset()

        def __iter__(self):
            yield object()

    monkeypatch.setattr(trainer, "_forward_latent_batch", fake_forward)
    model = torch.nn.Linear(2, 2)

    with caplog.at_level(logging.WARNING):
        metrics = trainer.evaluate_cluster_test_map(
            model,
            Loader(),
            device=torch.device("cpu"),
            source_name="fixture",
        )

    assert len(metrics) == 16
    assert "reason=no_usable_modality path=image-0.jpg" in caplog.text
    assert "queries_with_relevant=2 embedded_rows=3" in caplog.text


@pytest.mark.parametrize(
    ("branch_name", "bad_value", "error_pattern"),
    [
        ("fusion", 0.0, "required_unavailable=.*fusion"),
        ("tile", 0.0, "required_unavailable=.*tile"),
        ("glyph", float("nan"), "nonfinite=.*glyph.*1"),
        ("word", float("inf"), "nonfinite=.*word.*1"),
    ],
)
def test_cluster_test_still_rejects_required_or_nonfinite_outputs(
    monkeypatch, branch_name, bad_value, error_pattern
):
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    good = torch.tensor([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
        [0.1, 0.9],
    ])
    latent = good.clone()
    branches = {name: good.clone() for name in ("tile", "glyph", "word")}
    target = latent if branch_name == "fusion" else branches[branch_name]
    target[0].fill_(bad_value)

    def fake_forward(_model, _batch, *, device, return_aux_latents):
        return None, latent, branches, labels, [f"image-{index}.jpg" for index in range(4)]

    class Loader:
        class Dataset:
            cluster_test_num_rows = 4
            cluster_test_num_clusters = 2

            def __len__(self):
                return 4

        dataset = Dataset()

        def __iter__(self):
            yield object()

    monkeypatch.setattr(trainer, "_forward_latent_batch", fake_forward)

    with pytest.raises(RuntimeError, match=error_pattern):
        trainer.evaluate_cluster_test_map(
            torch.nn.Linear(2, 2),
            Loader(),
            device=torch.device("cpu"),
            source_name="fixture",
        )


def test_word_ocr_threshold_is_095_and_shared_with_dataset():
    assert system.OCR_STRING_CONFIDENCE_THRESHOLD == pytest.approx(0.95)
    assert dataset_module.OCR_STRING_CONFIDENCE_THRESHOLD == pytest.approx(0.95)


def test_cluster_test_suite_runs_both_sources_once_and_returns_only_retrieval_metrics(monkeypatch):
    calls = []

    def fake_evaluate(_model, loader, *, device, source_name):
        calls.append((source_name, loader, device.type))
        return {
            f"test/{source_name}/{metric}/{branch}": 0.5
            for metric in trainer.CLUSTER_TEST_METRIC_NAMES
            for branch in trainer.RETRIEVAL_BRANCH_NAMES
        }

    monkeypatch.setattr(trainer, "evaluate_cluster_test_map", fake_evaluate)
    loaders = {"clusters_images_metadata": object(), "cluster_members": object()}

    metrics = trainer.evaluate_cluster_test_suite(
        object(), loaders, device=torch.device("cpu")
    )

    assert [call[0] for call in calls] == [
        "clusters_images_metadata",
        "cluster_members",
    ]
    assert len(metrics) == 32
    assert set(metrics) == trainer.CLUSTER_TEST_WANDB_KEYS


def test_cluster_test_wandb_guard_rejects_unapproved_test_fields():
    approved = {
        f"test/{source}/{metric}/{branch}": 0.5
        for source in ("clusters_images_metadata", "cluster_members")
        for metric in ("mAP", "knn_at_1", "knn_at_5", "knn_at_10")
        for branch in ("fusion", "tile", "glyph", "word")
    }
    trainer._validate_cluster_test_wandb_metrics(approved)

    with pytest.raises(RuntimeError, match="unexpected=.*similarity"):
        trainer._validate_cluster_test_wandb_metrics({
            **approved,
            "test/cluster_members/similarity/same_mean": 0.9,
        })
