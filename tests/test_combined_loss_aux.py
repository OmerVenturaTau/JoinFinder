"""
Tests for CombinedLoss with auxiliary signals (Tile, Glyph, Word).
Verifies dimension consistency and device handling for multi-branch supervision.
"""

import torch
import pytest
from losses.combined_loss import CombinedLoss
from system import MAX_SELECTED_MANUSCRIPTS, LATENT_DIM, D_MODEL

def test_combined_loss_aux_dims():
    """Verify that auxiliary heads can have different dimensions than the main head."""
    B = 2
    num_classes = 100
    main_dim = 2048
    aux_dim = 768
    
    loss = CombinedLoss(
        num_classes=num_classes,
        embedding_dim=main_dim,
        aux_embedding_dim=aux_dim,
        tile_aux_weight=1.0,
        glyph_aux_weight=1.0,
        word_aux_weight=1.0,
    )
    
    # Check head initialization
    assert loss.arcface.weight.shape == (num_classes, main_dim)
    assert loss.tile_arcface.weight.shape == (num_classes, aux_dim)
    assert loss.glyph_arcface.weight.shape == (num_classes, aux_dim)
    assert loss.word_arcface.weight.shape == (num_classes, aux_dim)


def test_shared_branch_arcface_reuses_class_prototypes():
    loss = CombinedLoss(
        num_classes=10,
        embedding_dim=12,
        aux_embedding_dim=6,
        tile_aux_weight=1.0,
        glyph_aux_weight=1.0,
        shared_branch_arcface=True,
    )
    assert loss.tile_arcface is loss.glyph_arcface


def test_gate_entropy_penalizes_collapsed_reliability_early_only():
    loss = CombinedLoss(
        num_classes=3,
        embedding_dim=6,
        arcface_weight=0.0,
        ce_weight=1.0,
        gate_entropy_weight=0.1,
        gate_entropy_decay_epochs=2,
    )
    logits = torch.randn(2, 3)
    features = torch.randn(2, 6)
    labels = torch.tensor([0, 1])
    aux = {
        "reliability_weights": torch.tensor([[0.99, 0.01], [0.98, 0.02]]),
        "modality_available": torch.ones(2, 2, dtype=torch.bool),
    }
    loss.set_training_epoch(0)
    *_, early = loss(logits, features, labels, aux)
    assert early["gate_entropy"] > 0
    loss.set_training_epoch(2)
    *_, late = loss(logits, features, labels, aux)
    assert late["gate_entropy"] == 0


def test_word_aux_weight_stays_full_then_decays_to_zero():
    loss = CombinedLoss(
        num_classes=3,
        embedding_dim=6,
        word_aux_weight=0.05,
        word_aux_full_weight_epochs=2,
        word_aux_decay_end_epoch=5,
    )

    expected = [0.05, 0.05, 0.05 * 2 / 3, 0.05 / 3, 0.0]
    observed = []
    for zero_based_epoch in range(5):
        loss.set_training_epoch(zero_based_epoch)
        observed.append(loss.current_word_aux_weight)

    torch.testing.assert_close(torch.tensor(observed), torch.tensor(expected))

def test_combined_loss_aux_forward():
    """Test forward pass with auxiliary latents of different dimensions."""
    B = 4
    num_classes = 10
    main_dim = 2048
    aux_dim = 768
    
    loss = CombinedLoss(
        num_classes=num_classes,
        embedding_dim=main_dim,
        aux_embedding_dim=aux_dim,
        tile_aux_weight=0.5,
        glyph_aux_weight=0.5,
    )
    
    logits = torch.randn(B, num_classes)
    features = torch.randn(B, main_dim)
    labels = torch.randint(0, num_classes, (B,))
    
    aux_latents = {
        'tile': torch.randn(B, aux_dim),
        'glyph': torch.randn(B, aux_dim),
        'word': torch.randn(B, aux_dim) # Should be ignored if weight=0
    }
    
    # This should not crash
    total_loss, classification_loss, sparsity_loss, effective_logits, aux_loss_dict = loss(
        logits, features, labels, aux_latents=aux_latents
    )
    
    assert isinstance(total_loss, torch.Tensor)
    assert 'tile' in aux_loss_dict
    assert 'glyph' in aux_loss_dict
    assert aux_loss_dict['tile'].item() > 0
    assert aux_loss_dict['glyph'].item() > 0

def test_combined_loss_aux_with_zeros():
    """Test that zero vectors (dropped modalities) don't cause NaNs."""
    B = 4
    num_classes = 10
    main_dim = 2048
    aux_dim = 768
    
    loss = CombinedLoss(
        num_classes=num_classes,
        embedding_dim=main_dim,
        aux_embedding_dim=aux_dim,
        tile_aux_weight=1.0,
        glyph_aux_weight=1.0,
    )
    
    logits = torch.randn(B, num_classes)
    features = torch.randn(B, main_dim)
    labels = torch.randint(0, num_classes, (B,))
    
    # Simulate: Tile is valid for all, Glyph is zero for all (dropped)
    aux_latents = {
        'tile': torch.randn(B, aux_dim),
        'glyph': torch.zeros(B, aux_dim)
    }
    
    total_loss, classification_loss, _, _, aux_loss_dict = loss(
        logits, features, labels, aux_latents=aux_latents
    )
    
    assert not torch.isnan(total_loss)
    assert aux_loss_dict['tile'].item() > 0
    assert aux_loss_dict['glyph'].item() == 0
    assert not torch.isnan(aux_loss_dict['glyph'])

def test_combined_loss_aux_partial_zeros():
    """Test that partially zeroed batches (some samples dropped) work correctly."""
    B = 4
    num_classes = 10
    main_dim = 2048
    aux_dim = 768
    
    loss = CombinedLoss(
        num_classes=num_classes,
        embedding_dim=main_dim,
        aux_embedding_dim=aux_dim,
        tile_aux_weight=1.0,
    )
    
    logits = torch.randn(B, num_classes)
    features = torch.randn(B, main_dim)
    labels = torch.randint(0, num_classes, (B,))
    
    # Simulate: Tile is zero for half the batch
    tile_l = torch.randn(B, aux_dim)
    tile_l[0:2] = 0.0 # First two samples dropped
    
    aux_latents = {'tile': tile_l}
    
    total_loss, _, _, _, aux_loss_dict = loss(
        logits, features, labels, aux_latents=aux_latents
    )
    
    assert not torch.isnan(total_loss)
    assert aux_loss_dict['tile'].item() > 0 
    assert not torch.isnan(aux_loss_dict['tile'])

def test_combined_loss_aux_device():
    """Test auxiliary loss forward pass on GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    device = torch.device("cuda:0")
    B = 4
    num_classes = 10
    main_dim = 2048
    aux_dim = 768
    
    loss = CombinedLoss(
        num_classes=num_classes,
        embedding_dim=main_dim,
        aux_embedding_dim=aux_dim,
        tile_aux_weight=0.5,
    ).to(device)
    
    logits = torch.randn(B, num_classes).to(device)
    features = torch.randn(B, main_dim).to(device)
    labels = torch.randint(0, num_classes, (B,)).to(device)
    
    aux_latents = {
        'tile': torch.randn(B, aux_dim).to(device)
    }
    
    total_loss, classification_loss, sparsity_loss, effective_logits, aux_loss_dict = loss(
        logits, features, labels, aux_latents=aux_latents
    )
    
    assert total_loss.device.type == 'cuda'
    assert aux_loss_dict['tile'].device.type == 'cuda'

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
