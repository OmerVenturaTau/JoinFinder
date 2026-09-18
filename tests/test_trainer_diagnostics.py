import torch

from train.trainer import (
    _batch_raw_evidence_counts,
    _compute_library_retrieval_metrics,
    _configure_trainable_phase,
)


class _PhaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.use_word_mod = True
        self.tile_branch = torch.nn.Linear(2, 2)
        self.glyph_branch = torch.nn.Linear(2, 2)
        self.word_branch = torch.nn.Linear(2, 2)
        self.fusion = torch.nn.Module()
        self.fusion.adapters = torch.nn.ModuleDict({
            "tile": torch.nn.Linear(2, 2),
            "glyph": torch.nn.Linear(2, 2),
            "word": torch.nn.Linear(2, 2),
        })
        self.head = torch.nn.Linear(6, 2)


class _PhaseLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.arcface = torch.nn.Linear(6, 2)
        self.tile_arcface = torch.nn.Linear(2, 2)
        self.glyph_arcface = torch.nn.Linear(2, 2)
        self.word_arcface = torch.nn.Linear(2, 2)


def test_batch_raw_evidence_counts_maps_fixed_column_order():
    metadata = {
        "raw_evidence_counts": torch.tensor([[16.0, 24.0, 80.0], [8.0, 5.0, 12.0]])
    }
    counts = _batch_raw_evidence_counts(metadata, torch.device("cpu"))
    assert counts is not None
    assert counts["tile"].tolist() == [16.0, 8.0]
    assert counts["glyph"].tolist() == [24.0, 5.0]
    assert counts["word"].tolist() == [80.0, 12.0]


def test_cross_library_metrics_use_all_other_library_candidates():
    features = torch.tensor([
        [1.0, 0.0], [0.99, 0.01],
        [0.0, 1.0], [0.01, 0.99],
    ])
    labels = torch.tensor([0, 0, 1, 1])
    libraries = ["A", "B", "A", "B"]

    metrics = _compute_library_retrieval_metrics(
        features, labels, libraries, "join_eval"
    )

    assert metrics["join_eval/library/cross_library_valid_queries"] == 4.0
    assert metrics["join_eval/library/cross_library_map"] == 1.0
    assert "join_eval/library/same_library_negative_bias" in metrics


def test_word_warmup_trains_only_word_path_and_private_head():
    model, loss = _PhaseModel(), _PhaseLoss()

    _configure_trainable_phase(model, loss, word_only=True)

    assert all(parameter.requires_grad for parameter in model.word_branch.parameters())
    assert all(parameter.requires_grad for parameter in model.fusion.adapters["word"].parameters())
    assert all(not parameter.requires_grad for parameter in model.tile_branch.parameters())
    assert all(not parameter.requires_grad for parameter in model.glyph_branch.parameters())
    assert all(not parameter.requires_grad for parameter in model.head.parameters())
    assert all(parameter.requires_grad for parameter in loss.word_arcface.parameters())
    assert all(not parameter.requires_grad for parameter in loss.arcface.parameters())


def test_multimodal_phase_restores_parameters_and_honors_branch_freeze():
    model, loss = _PhaseModel(), _PhaseLoss()
    _configure_trainable_phase(model, loss, word_only=True)

    _configure_trainable_phase(
        model,
        loss,
        word_only=False,
        freeze_tile=True,
        freeze_glyph=False,
        freeze_word=False,
    )

    assert all(not parameter.requires_grad for parameter in model.tile_branch.parameters())
    assert all(parameter.requires_grad for parameter in model.glyph_branch.parameters())
    assert all(parameter.requires_grad for parameter in model.word_branch.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())
    assert all(parameter.requires_grad for parameter in loss.parameters())
