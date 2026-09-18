import torch

from train.optimizer_groups import (
    build_optimizer_param_groups,
    build_ratio_preserving_cosine_scheduler,
)


class _LineEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.transformer = torch.nn.Linear(2, 2)


class _WordBranch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.line_encoder = _LineEncoder()
        self.adapter = torch.nn.Linear(2, 2)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.word_branch = _WordBranch()


class _Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.arcface = torch.nn.Linear(2, 2)
        self.word_arcface = torch.nn.Linear(2, 2)


def test_optimizer_groups_assign_expected_learning_rates_without_duplicates():
    model, loss = _Model(), _Loss()
    groups = build_optimizer_param_groups(
        model, loss, base_lr=2e-5, alephbert_lr=5e-6, new_word_lr=1e-4
    )
    by_name = {group["name"]: group for group in groups}
    assert {name: group["lr"] for name, group in by_name.items()} == {
        "base": 2e-5,
        "alephbert": 5e-6,
        "new_word": 1e-4,
    }
    parameter_ids = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(parameter_ids) == len(set(parameter_ids))
    assert id(model.word_branch.line_encoder.encoder.transformer.weight) in {
        id(parameter) for parameter in by_name["alephbert"]["params"]
    }
    assert id(loss.word_arcface.weight) in {
        id(parameter) for parameter in by_name["new_word"]["params"]
    }


def test_cosine_scheduler_preserves_group_lr_ratios():
    model, loss = _Model(), _Loss()
    groups = build_optimizer_param_groups(
        model, loss, base_lr=2e-5, alephbert_lr=5e-6, new_word_lr=1e-4
    )
    optimizer = torch.optim.AdamW(groups)
    scheduler = build_ratio_preserving_cosine_scheduler(
        optimizer, total_epochs=6, base_lr=2e-5, eta_min=1e-6
    )
    initial = [group["lr"] for group in optimizer.param_groups]
    for _ in range(3):
        optimizer.step()
        scheduler.step()
    current = [group["lr"] for group in optimizer.param_groups]
    ratios = [now / before for now, before in zip(current, initial)]
    torch.testing.assert_close(torch.tensor(ratios), torch.full((3,), ratios[0]))


def test_optimizer_groups_retain_temporarily_frozen_parameters():
    model, loss = _Model(), _Loss()
    frozen = model.word_branch.adapter.weight
    frozen.requires_grad = False

    groups = build_optimizer_param_groups(
        model, loss, base_lr=2e-5, alephbert_lr=5e-6, new_word_lr=1e-4
    )

    tracked = {id(parameter) for group in groups for parameter in group["params"]}
    assert id(frozen) in tracked
