"""Opt-in smoke test for the real cluster evaluation data on the server.

Run with:
    RUN_SERVER_CLUSTER_TESTS=1 pytest -q -m server_data \
        tests/test_cluster_test_server_data.py

The coverage model deliberately avoids loading pretrained model weights. It
still exercises the real CSV/XLSX adapters, image and ALTO extraction, collate
path, optional-modality skipping, retrieval metrics, and W&B metric-key guard.
"""

import logging
import os

import pytest
import torch

import system
import train.trainer as trainer
from models.word_branch import WordHardQualityFilter


pytestmark = [
    pytest.mark.server_data,
    pytest.mark.skipif(
        os.environ.get("RUN_SERVER_CLUSTER_TESTS") != "1",
        reason="set RUN_SERVER_CLUSTER_TESTS=1 to exercise NAS-backed cluster data",
    ),
]


class _CoverageOnlyModel(torch.nn.Module):
    """Emit deterministic vectors whose availability follows the real batch."""

    def __init__(self):
        super().__init__()
        self.modality_dropout_enabled = True
        self.token_subsample_enabled = True
        self.word_filter = WordHardQualityFilter()

    def _word_available(self, words, metadata, device):
        available = []
        for image_words, image_metadata in zip(words, metadata):
            entries = [
                (index, word, meta)
                for index, (word, meta) in enumerate(zip(image_words, image_metadata))
            ]
            available.append(bool(self.word_filter.filter_words_in_line(entries)))
        return torch.tensor(available, dtype=torch.bool, device=device)

    def forward(
        self,
        *,
        tiles,
        tile_valid_mask,
        glyph_valid_mask,
        words,
        word_metadata,
        return_aux_latents,
        **_kwargs,
    ):
        assert return_aux_latents is True
        batch_size = tiles.shape[0]
        device = tiles.device
        row = torch.arange(1, batch_size + 1, dtype=torch.float32, device=device)
        base = torch.stack((torch.ones_like(row), row, row.square(), row.reciprocal()), dim=1)

        tile_available = tile_valid_mask.any(dim=1)
        glyph_available = glyph_valid_mask.any(dim=1)
        word_available = self._word_available(words, word_metadata, device)

        def optional_vector(available):
            return torch.where(available.unsqueeze(1), base, torch.zeros_like(base))

        logits = torch.zeros(batch_size, 1, device=device)
        return logits, base, {
            "tile": optional_vector(tile_available),
            "glyph": optional_vector(glyph_available),
            "word": optional_vector(word_available),
        }


def _real_cluster_loaders():
    device = torch.device("cpu")
    specs = (
        ("clusters_images_metadata", system.CLUSTER_PAIRS_CSV_PATH, None),
        ("cluster_members", system.CLUSTER_MEMBERS_XLSX_PATH, system.CLUSTER_MEMBERS_XLSX_SHEET),
    )
    loaders = {}
    for source_name, source_path, sheet_name in specs:
        expected_rows, expected_clusters = system.CLUSTER_TEST_EXPECTED_COUNTS[source_name]
        members = trainer.load_cluster_test_members(
            source_name,
            source_path,
            sheet_name=sheet_name,
            expected_rows=expected_rows,
            expected_clusters=expected_clusters,
        )
        loaders[source_name] = trainer.build_cluster_test_loader(
            members,
            device=device,
            batch_size=2,
            num_workers=0,
        )
    return loaders


def test_real_server_cluster_sources_complete_without_optional_coverage_exception(caplog):
    with caplog.at_level(logging.INFO):
        metrics = trainer.evaluate_cluster_test_suite(
            _CoverageOnlyModel(),
            _real_cluster_loaders(),
            device=torch.device("cpu"),
        )

    assert set(metrics) == trainer.CLUSTER_TEST_WANDB_KEYS
    assert len(metrics) == 32
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    for source_name in trainer.CLUSTER_TEST_SOURCE_NAMES:
        assert f"source={source_name}" in caplog.text
    assert "[CLUSTER TEST][SKIP]" in caplog.text
    assert "[CLUSTER TEST][COVERAGE]" in caplog.text
