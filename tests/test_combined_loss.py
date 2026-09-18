"""
Tests for CombinedLoss including ArcFace and device handling.
"""

import torch
import torch.nn as nn
import pytest
from losses.combined_loss import CombinedLoss
from system import MAX_SELECTED_MANUSCRIPTS, LATENT_DIM


class ConstantArcFace(nn.Module):
    def __init__(self, value: float, num_classes: int):
        super().__init__()
        self.value = value
        self.num_classes = num_classes

    def forward(self, embeddings, labels):
        loss = embeddings.sum() * 0.0 + torch.tensor(
            self.value,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        logits = torch.zeros(
            embeddings.shape[0],
            self.num_classes,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        return loss, logits


def test_combined_loss_creation():
    """Test CombinedLoss creation with different configurations."""
    # Test with ArcFace enabled
    loss = CombinedLoss(
        num_classes=MAX_SELECTED_MANUSCRIPTS,
        embedding_dim=LATENT_DIM,
        arcface_weight=1.0,
    )
    assert loss.arcface is not None
    assert loss.arcface_weight == 1.0
    
    # Test with ArcFace disabled
    loss = CombinedLoss(
        num_classes=MAX_SELECTED_MANUSCRIPTS,
        embedding_dim=LATENT_DIM,
        arcface_weight=0.0,
    )
    assert loss.arcface is None
    assert loss.ce is not None  # Fallback CrossEntropy


def test_combined_loss_forward_arcface():
    """Test CombinedLoss forward pass with ArcFace."""
    B = 4
    num_classes = MAX_SELECTED_MANUSCRIPTS
    
    loss = CombinedLoss(
        num_classes=num_classes,
        embedding_dim=LATENT_DIM,
        arcface_weight=1.0,
    )
    
    # Create dummy inputs
    logits = torch.randn(B, num_classes)
    features = torch.randn(B, LATENT_DIM)
    labels = torch.randint(0, num_classes, (B,))
    
    total_loss, arcface_loss, sparsity_loss, effective_logits, aux_loss_dict = loss(logits, features, labels)
    
    assert isinstance(total_loss, torch.Tensor)
    assert isinstance(arcface_loss, torch.Tensor)
    assert isinstance(effective_logits, torch.Tensor)
    assert effective_logits.shape == (B, num_classes)
    assert total_loss.item() > 0
    assert arcface_loss.item() > 0


def test_combined_loss_device_handling():
    """Test that CombinedLoss can be moved to device and works correctly."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    B = 2
    num_classes = MAX_SELECTED_MANUSCRIPTS
    
    loss = CombinedLoss(
        num_classes=num_classes,
        embedding_dim=LATENT_DIM,
        arcface_weight=1.0,
    )
    
    # Move to CUDA
    loss = loss.cuda()
    
    # Verify weight matrix is on CUDA
    assert loss.arcface.weight.device.type == 'cuda'
    
    # Create inputs on CUDA
    logits = torch.randn(B, num_classes).cuda()
    features = torch.randn(B, LATENT_DIM).cuda()
    labels = torch.randint(0, num_classes, (B,)).cuda()
    
    # Should not raise device mismatch error
    total_loss, arcface_loss, sparsity_loss, effective_logits, aux_loss_dict = loss(logits, features, labels)
    
    assert total_loss.device.type == 'cuda'
    assert arcface_loss.device.type == 'cuda'
    assert effective_logits.device.type == 'cuda'


def test_combined_loss_gradient_flow():
    """Test that gradients flow through CombinedLoss."""
    B = 2
    num_classes = MAX_SELECTED_MANUSCRIPTS
    
    loss = CombinedLoss(
        num_classes=num_classes,
        embedding_dim=LATENT_DIM,
        arcface_weight=1.0,
    )
    
    # Create inputs with requires_grad
    logits = torch.randn(B, num_classes, requires_grad=True)
    features = torch.randn(B, LATENT_DIM, requires_grad=True)
    labels = torch.randint(0, num_classes, (B,))
    
    total_loss, arcface_loss, sparsity_loss, effective_logits, aux_loss_dict = loss(logits, features, labels)
    
    # Backward pass
    total_loss.backward()
    
    # Verify gradients exist
    assert features.grad is not None
    assert loss.arcface.weight.grad is not None


def test_combined_loss_weighted_total_arithmetic_with_aux_masks():
    B = 4
    num_classes = 3
    embedding_dim = 5
    aux_dim = 5

    loss = CombinedLoss(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        aux_embedding_dim=aux_dim,
        arcface_weight=2.0,
        sparsity_weight=0.25,
        tile_aux_weight=0.5,
        glyph_aux_weight=0.25,
    )
    loss.arcface = ConstantArcFace(5.0, num_classes)
    loss.tile_arcface = ConstantArcFace(7.0, num_classes)
    loss.glyph_arcface = ConstantArcFace(11.0, num_classes)

    logits = torch.randn(B, num_classes)
    features = torch.tensor(
        [
            [1.0, -2.0, 3.0, -4.0, 5.0],
            [-1.0, 2.0, -3.0, 4.0, -5.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 2.0],
        ]
    )
    labels = torch.tensor([0, 1, 2, 1])
    aux_latents = {
        "tile": torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        ),
        "glyph": torch.ones(B, aux_dim),
    }

    total_loss, classification_loss, sparsity_loss, _effective_logits, aux_loss_dict = loss(
        logits,
        features,
        labels,
        aux_latents=aux_latents,
    )

    expected_tile = torch.tensor(7.0 * (2 / 4))
    expected_glyph = torch.tensor(11.0)
    expected_total = (
        2.0 * torch.tensor(5.0)
        + 0.25 * torch.mean(torch.abs(features))
        + 0.5 * expected_tile
        + 0.25 * expected_glyph
    )

    assert classification_loss.item() == pytest.approx(5.0)
    assert sparsity_loss.item() == pytest.approx(torch.mean(torch.abs(features)).item())
    assert aux_loss_dict["tile"].item() == pytest.approx(expected_tile.item())
    assert aux_loss_dict["glyph"].item() == pytest.approx(expected_glyph.item())
    assert total_loss.item() == pytest.approx(expected_total.item())


def test_combined_loss_includes_fusion_aux_weight():
    B = 4
    num_classes = 3
    embedding_dim = 5

    loss = CombinedLoss(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        aux_embedding_dim=embedding_dim,
        arcface_weight=1.0,
        fusion_aux_weight=0.4,
    )
    loss.arcface = ConstantArcFace(3.0, num_classes)
    loss.fusion_arcface = ConstantArcFace(5.0, num_classes)

    logits = torch.randn(B, num_classes)
    features = torch.randn(B, embedding_dim)
    labels = torch.tensor([0, 1, 2, 1])
    aux_latents = {"fusion": torch.ones(B, embedding_dim)}

    total_loss, classification_loss, _sparsity_loss, _effective_logits, aux_loss_dict = loss(
        logits,
        features,
        labels,
        aux_latents=aux_latents,
    )

    assert classification_loss.item() == pytest.approx(3.0)
    assert aux_loss_dict["fusion"].item() == pytest.approx(5.0)
    assert total_loss.item() == pytest.approx(3.0 + 0.4 * 5.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
