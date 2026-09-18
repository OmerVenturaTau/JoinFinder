import torch
import torch.nn as nn
import pytest

from losses.combined_loss import CombinedLoss
from utilities.checkpoint_utils import load_training_checkpoint_for_evaluation


class TinyModule(nn.Module):
    def __init__(self, initial: float):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([initial], dtype=torch.float32))


def test_eval_checkpoint_restore_loads_model_and_combined_loss(tmp_path):
    model_to_save = TinyModule(1.5)
    loss_to_save = TinyModule(2.5)
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save(
        {
            "state_dict": model_to_save.state_dict(),
            "combined_loss_state_dict": loss_to_save.state_dict(),
        },
        checkpoint_path,
    )

    live_model = TinyModule(-10.0)
    live_loss = TinyModule(-20.0)

    load_training_checkpoint_for_evaluation(
        live_model,
        live_loss,
        str(checkpoint_path),
        device=torch.device("cpu"),
    )

    assert torch.equal(live_model.weight, model_to_save.weight)
    assert torch.equal(live_loss.weight, loss_to_save.weight)


def test_eval_checkpoint_restore_preserves_arcface_current_margin(tmp_path):
    num_classes = 3
    embedding_dim = 4
    model_to_save = TinyModule(1.5)
    loss_to_save = CombinedLoss(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        arcface_weight=1.0,
        tile_aux_weight=1.0,
        aux_embedding_dim=embedding_dim,
    )
    loss_to_save.set_arcface_margin(0.125)
    checkpoint_path = tmp_path / "checkpoint_with_margin.pth"
    torch.save(
        {
            "state_dict": model_to_save.state_dict(),
            "combined_loss_state_dict": loss_to_save.state_dict(),
        },
        checkpoint_path,
    )

    live_model = TinyModule(-10.0)
    live_loss = CombinedLoss(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        arcface_weight=1.0,
        tile_aux_weight=1.0,
        aux_embedding_dim=embedding_dim,
    )
    live_loss.set_arcface_margin(0.5)

    load_training_checkpoint_for_evaluation(
        live_model,
        live_loss,
        str(checkpoint_path),
        device=torch.device("cpu"),
    )

    assert live_loss.arcface.current_margin.item() == pytest.approx(0.125)
    assert live_loss.tile_arcface.current_margin.item() == pytest.approx(0.125)
