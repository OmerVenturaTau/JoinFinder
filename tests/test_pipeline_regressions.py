import pandas as pd
import pytest
import torch
import torch.nn.functional as F
import sys
import types

from models.transformer_fusion import TransformerFusion

sys.modules.setdefault("psycopg2", types.SimpleNamespace(connect=None))
import train.split_data as split_data
from train.split_data import (
    build_page_level_classification_splits,
    build_splits,
    merge_classification_splits,
    paths_labels_from_dataset_split_column,
)


def test_dataset_split_column_keeps_test_rows():
    df = pd.DataFrame(
        [
            {
                "manuscript_id": "m1",
                "parent_directory": "p",
                "picture_id": "i1.jpg",
                "dataset_split": "train",
            },
            {
                "manuscript_id": "m2",
                "parent_directory": "p",
                "picture_id": "i2.jpg",
                "dataset_split": "val",
            },
            {
                "manuscript_id": "m3",
                "parent_directory": "p",
                "picture_id": "i3.jpg",
                "dataset_split": "test",
            },
        ]
    )

    (
        train_paths,
        train_labels,
        _train_xmls,
        val_paths,
        val_labels,
        _val_xmls,
        test_paths,
        test_labels,
        _test_xmls,
    ) = paths_labels_from_dataset_split_column(df, "/base")

    assert train_paths == ["/base/m1/p/i1.jpg"]
    assert train_labels == ["m1"]
    assert val_paths == ["/base/m2/p/i2.jpg"]
    assert val_labels == ["m2"]
    assert test_paths == ["/base/m3/p/i3.jpg"]
    assert test_labels == ["m3"]


def test_build_splits_uses_dataset_split_column(monkeypatch):
    df = pd.DataFrame(
        [
            {
                "manuscript_id": "m1",
                "parent_directory": "p",
                "picture_id": "i1.jpg",
                "page_number": "P000001",
                "dataset_split": "train",
            },
            {
                "manuscript_id": "m1",
                "parent_directory": "p",
                "picture_id": "i2.jpg",
                "page_number": "P000002",
                "dataset_split": "val",
            },
            {
                "manuscript_id": "m1",
                "parent_directory": "p",
                "picture_id": "i3.jpg",
                "page_number": "P000003",
                "dataset_split": "test",
            },
        ]
    )

    monkeypatch.setattr(split_data, "get_db_connection", lambda: types.SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(split_data, "get_table_as_df", lambda _conn, _table_name: df)

    splits, stats = build_splits("/base", table_name="classification_table", dataset_stage="stage2")

    assert stats["split_strategy"] == "dataset_split_column"
    assert splits["train"]["m1"] == [("/base/m1/p/i1.jpg", None)]
    assert splits["val"]["m1"] == [("/base/m1/p/i2.jpg", None)]
    assert splits["test"]["m1"] == [("/base/m1/p/i3.jpg", None)]


def test_build_splits_allows_missing_test_split(monkeypatch):
    df = pd.DataFrame(
        [
            {
                "manuscript_id": "m1",
                "parent_directory": "p",
                "picture_id": "i1.jpg",
                "page_number": "P000001",
                "dataset_split": "train",
            },
            {
                "manuscript_id": "m1",
                "parent_directory": "p",
                "picture_id": "i2.jpg",
                "page_number": "P000002",
                "dataset_split": "val",
            },
        ]
    )

    monkeypatch.setattr(split_data, "get_db_connection", lambda: types.SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(split_data, "get_table_as_df", lambda _conn, _table_name: df)

    splits, stats = build_splits("/base", table_name="classification_table", dataset_stage="stage2")

    assert stats["split_strategy"] == "dataset_split_column"
    assert splits["train"]["m1"] == [("/base/m1/p/i1.jpg", None)]
    assert splits["val"]["m1"] == [("/base/m1/p/i2.jpg", None)]
    assert splits["test"] == {}


def test_geniza_classification_page_splits_every_manuscript_into_train_and_val():
    df = pd.DataFrame(
        [
            {
                "manuscript_id": manuscript_id,
                "page_number": f"P{page_number:06d}",
                "image_path": f"/geniza/{manuscript_id}/{page_number}.jpg",
                "xml_path": f"/xml/{manuscript_id}/{page_number}.xml",
                # The source split is manuscript-disjoint and must not control
                # the closed-set classification split.
                "dataset_split": source_split,
            }
            for manuscript_id, source_split in (("g_train", "train"), ("g_val", "val"))
            for page_number in range(1, 6)
        ]
    )

    splits, stats = build_page_level_classification_splits(
        df,
        "/base",
        train_fraction=0.8,
    )

    assert set(splits["train"]) == {"g_train", "g_val"}
    assert set(splits["val"]) == {"g_train", "g_val"}
    assert splits["test"] == {}
    assert all(len(splits["train"][mid]) == 4 for mid in ("g_train", "g_val"))
    assert all(len(splits["val"][mid]) == 1 for mid in ("g_train", "g_val"))
    assert stats == {
        "images_total": 10,
        "images_train": 8,
        "images_val": 2,
        "manuscripts_total": 2,
        "manuscripts_train": 2,
        "manuscripts_val": 2,
    }

    train_paths = {path for items in splits["train"].values() for path, _xml in items}
    val_paths = {path for items in splits["val"].values() for path, _xml in items}
    assert train_paths.isdisjoint(val_paths)


def test_merge_classification_splits_preserves_sources_without_mutating_them():
    primary = {
        "train": {"base": [("/base/train.jpg", None)]},
        "val": {"base": [("/base/val.jpg", None)]},
        "test": {},
    }
    geniza = {
        "train": {"geniza": [("/geniza/train.jpg", "/geniza/train.xml")]},
        "val": {"geniza": [("/geniza/val.jpg", "/geniza/val.xml")]},
        "test": {},
    }

    merged = merge_classification_splits(primary, geniza)

    assert set(merged["train"]) == {"base", "geniza"}
    assert set(merged["val"]) == {"base", "geniza"}
    assert merged["train"]["geniza"] == [("/geniza/train.jpg", "/geniza/train.xml")]
    assert "geniza" not in primary["train"]


def test_transformer_fusion_empty_batch_shape():
    fusion = TransformerFusion(d_model=12, num_layers=1, nhead=3, dim_feedforward=24, dropout=0.0)
    tokens = torch.zeros(0, 2, 12)
    mask = torch.zeros(0, 2, dtype=torch.bool)

    out, attn = fusion(tile_tokens=tokens, tile_valid_mask=mask)

    assert out.shape == (0, 1, 12)
    assert attn is None


def test_transformer_fusion_rejects_mask_shape_mismatch():
    fusion = TransformerFusion(d_model=12, num_layers=1, nhead=3, dim_feedforward=24, dropout=0.0)
    tokens = torch.zeros(2, 3, 12)
    wrong_mask = torch.ones(2, 2, dtype=torch.bool)

    with pytest.raises(ValueError, match="tile_valid_mask must have shape"):
        fusion(tile_tokens=tokens, tile_valid_mask=wrong_mask)


def test_transformer_fusion_rejects_dtype_mismatch():
    fusion = TransformerFusion(d_model=12, num_layers=1, nhead=3, dim_feedforward=24, dropout=0.0)
    tile_tokens = torch.zeros(2, 3, 12, dtype=torch.float32)
    glyph_tokens = torch.zeros(2, 2, 12, dtype=torch.float64)

    with pytest.raises(ValueError, match="glyph_tokens dtype must match"):
        fusion(tile_tokens=tile_tokens, glyph_tokens=glyph_tokens)


def test_transformer_fusion_residual_gate_zero_uses_balanced_modality_pool():
    fusion = TransformerFusion(
        d_model=4,
        num_layers=1,
        nhead=2,
        dim_feedforward=8,
        dropout=0.0,
        use_residual_pool=True,
        residual_cls_gate=0.0,
        residual_tile_weight=1.0,
        residual_glyph_weight=1.0,
        residual_word_weight=1.0,
    )
    fusion.eval()
    tile_tokens = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]]])
    glyph_tokens = torch.tensor([[[0.0, 4.0, 0.0, 0.0], [0.0, 8.0, 0.0, 0.0]]])
    tile_mask = torch.tensor([[True, False]])
    glyph_mask = torch.tensor([[True, True]])

    out, _ = fusion(
        tile_tokens=tile_tokens,
        tile_valid_mask=tile_mask,
        glyph_tokens=glyph_tokens,
        glyph_valid_mask=glyph_mask,
    )

    tile_mean = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    glyph_mean = torch.tensor([[0.0, 6.0, 0.0, 0.0]])
    expected_raw = (tile_mean + glyph_mean) / 2.0
    expected = F.layer_norm(
        expected_raw,
        (4,),
        fusion.final_norm.weight,
        fusion.final_norm.bias,
        fusion.final_norm.eps,
    ).unsqueeze(1)
    assert torch.allclose(out, expected, atol=1e-6)


def test_transformer_fusion_residual_gate_one_matches_cls_only():
    torch.manual_seed(7)
    cls_only = TransformerFusion(
        d_model=8,
        num_layers=1,
        nhead=2,
        dim_feedforward=16,
        dropout=0.0,
        use_residual_pool=False,
    )
    residual = TransformerFusion(
        d_model=8,
        num_layers=1,
        nhead=2,
        dim_feedforward=16,
        dropout=0.0,
        use_residual_pool=True,
        residual_cls_gate=1.0,
    )
    residual.load_state_dict(cls_only.state_dict())
    cls_only.eval()
    residual.eval()

    tile_tokens = torch.randn(2, 3, 8)
    glyph_tokens = torch.randn(2, 4, 8)
    tile_mask = torch.tensor([[True, True, True], [True, False, False]])
    glyph_mask = torch.tensor([[True, True, False, False], [True, True, True, True]])

    out_cls, _ = cls_only(
        tile_tokens=tile_tokens,
        tile_valid_mask=tile_mask,
        glyph_tokens=glyph_tokens,
        glyph_valid_mask=glyph_mask,
    )
    out_residual, _ = residual(
        tile_tokens=tile_tokens,
        tile_valid_mask=tile_mask,
        glyph_tokens=glyph_tokens,
        glyph_valid_mask=glyph_mask,
    )

    assert torch.allclose(out_residual, out_cls, atol=1e-6)
