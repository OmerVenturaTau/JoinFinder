import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "debug_tools/compare_clusters_pairs_backbone.py"
SPEC = importlib.util.spec_from_file_location("compare_clusters_pairs_backbone", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_mean_pooling_returns_unit_vector():
    descriptors = np.array([[3.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    result = MODULE.pool_tile_descriptors(descriptors, "mean")
    expected = np.array([2.0, 1.0], dtype=np.float32)
    expected /= np.linalg.norm(expected)
    assert result.shape == (2,)
    assert np.allclose(result, expected)
    assert np.linalg.norm(result) == pytest.approx(1.0)


def test_mean_std_pooling_doubles_descriptor_dimension():
    descriptors = np.array([[1.0, 3.0], [3.0, 7.0]], dtype=np.float32)
    result = MODULE.pool_tile_descriptors(descriptors, "mean_std")
    expected = np.array([2.0, 5.0, 1.0, 2.0], dtype=np.float32)
    expected /= np.linalg.norm(expected)
    assert result.shape == (4,)
    assert np.allclose(result, expected)
    assert np.linalg.norm(result) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "descriptors",
    [np.empty((0, 2), dtype=np.float32), np.array([[np.nan, 1.0]], dtype=np.float32)],
)
def test_pooling_rejects_invalid_descriptors(descriptors):
    with pytest.raises(ValueError):
        MODULE.pool_tile_descriptors(descriptors, "mean_std")


def test_feature_matrix_keeps_native_two_dimensional_embedding():
    import torch

    native = torch.randn(3, 768)
    result = MODULE._as_feature_matrix(native)
    assert result is native
    assert result.shape == (3, 768)


def test_alephbert_pool_excludes_padding_and_special_tokens():
    import torch

    hidden = torch.tensor([[[100.0, 100.0], [1.0, 3.0], [3.0, 5.0], [50.0, 50.0]]])
    attention = torch.tensor([[1, 1, 1, 0]])
    special = torch.tensor([[1, 0, 0, 1]])
    pooled, count = MODULE.mean_pool_alephbert_chunks(hidden, attention, special)
    assert count == 2
    assert torch.equal(pooled, torch.tensor([2.0, 4.0]))
