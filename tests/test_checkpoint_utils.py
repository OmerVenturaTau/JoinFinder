"""Tests for checkpoint architecture inference."""

import torch

from utilities.checkpoint_utils import (
    _infer_d_model,
    _infer_latent_dim,
    _infer_fusion_method,
    _infer_symmetric_branch_dim,
    _infer_branch_adapters_and_summarizers,
    _infer_symmetric_reliability_hidden_dim,
    _infer_tile_summary_tokens,
    _infer_tile_summarizer_cross_attn_layers,
    _infer_transformer_dim_feedforward,
    _infer_word_summary_tokens,
    apply_inspection_to_system,
    without_classification_head,
)


def test_infer_architecture_from_legacy_shapes():
    state_dict = {
        "fusion.cls_token": torch.zeros(1, 1, 1200),
        "tile_branch.token_enrichment.proj.weight": torch.zeros(1200, 1200),
        "head.latent_proj.3.weight": torch.zeros(2048, 1536),
        "fusion.transformer.layers.0.linear1.weight": torch.zeros(4800, 1200),
        "tile_branch.set_transformer.transformer.layers.0.linear1.weight": torch.zeros(4800, 1200),
    }
    assert _infer_d_model(state_dict) == 1200
    assert _infer_latent_dim(state_dict) == 2048
    assert _infer_transformer_dim_feedforward(state_dict, branch="fusion") == 4800
    assert _infer_transformer_dim_feedforward(state_dict, branch="tile") == 4800
    assert _infer_tile_summary_tokens(state_dict) == 0


def test_infer_tile_summarizer_architecture_from_query_parameters():
    state_dict = {
        "tile_branch.set_summarizer.query_tokens": torch.zeros(1, 4, 768),
        "tile_branch.set_summarizer.cross_attn_layers.0.in_proj_weight": torch.zeros(2304, 768),
        "tile_branch.set_summarizer.cross_attn_layers.1.in_proj_weight": torch.zeros(2304, 768),
    }

    assert _infer_tile_summary_tokens(state_dict) == 4
    assert _infer_tile_summarizer_cross_attn_layers(state_dict) == 2


def test_infer_symmetric_and_word_architecture_from_checkpoint_shapes():
    state_dict = {
        "word_branch.word_set_summarizer.query_tokens": torch.zeros(1, 24, 768),
        "fusion.adapters.tile.1.weight": torch.zeros(512, 768),
        "fusion.reliability.0.weight": torch.zeros(64, 3),
        "head.classifier.0.weight": torch.zeros(1212, 1536),
    }

    assert _infer_word_summary_tokens(state_dict) == 24
    assert _infer_fusion_method(state_dict) == "symmetric"
    assert _infer_symmetric_branch_dim(state_dict) == 512
    assert _infer_branch_adapters_and_summarizers(state_dict)
    assert _infer_symmetric_reliability_hidden_dim(state_dict) == 64
    assert _infer_latent_dim(state_dict) == 1536


def test_infer_adapter_free_symmetric_fusion_from_reliability_weights():
    state_dict = {
        "fusion.reliability.0.weight": torch.zeros(64, 3),
        "head.classifier.0.weight": torch.zeros(1212, 2304),
    }

    assert _infer_fusion_method(state_dict) == "symmetric"
    assert _infer_symmetric_branch_dim(state_dict) is None
    assert not _infer_branch_adapters_and_summarizers(state_dict)
    assert _infer_latent_dim(state_dict) == 2304


def test_infer_adapter_flag_from_data_parallel_checkpoint():
    state_dict = {
        "module.fusion.adapters.tile.1.weight": torch.zeros(512, 768),
    }

    assert _infer_branch_adapters_and_summarizers(state_dict)


def test_apply_inspection_patches_system_d_model():
    import system
    from utilities.checkpoint_utils import CheckpointInspection

    old_d_model = system.D_MODEL
    old_latent = system.LATENT_DIM
    old_ff = system.TRANSFORMER_DIM_FEEDFORWARD
    old_tile_ff = system.TILE_BRANCH_TRANSFORMER_DIM_FEEDFORWARD
    try:
        info = CheckpointInspection(
            path="synthetic",
            state_dict={
                "fusion.cls_token": torch.zeros(1, 1, 1200),
                "head.latent_proj.3.weight": torch.zeros(2048, 1536),
                "fusion.transformer.layers.0.linear1.weight": torch.zeros(4800, 1200),
                "tile_branch.set_transformer.transformer.layers.0.linear1.weight": torch.zeros(4800, 1200),
            },
            d_model=1200,
            latent_dim=2048,
            fusion_dim_feedforward=4800,
            tile_dim_feedforward=4800,
        )
        changes = apply_inspection_to_system(info, system, quiet=True)
        assert system.D_MODEL == 1200
        assert system.LATENT_DIM == 2048
        assert "D_MODEL" in changes
        assert system.TRANSFORMER_DIM_FEEDFORWARD == 4800
    finally:
        system.D_MODEL = old_d_model
        system.LATENT_DIM = old_latent
        system.TRANSFORMER_DIM_FEEDFORWARD = old_ff
        system.TILE_BRANCH_TRANSFORMER_DIM_FEEDFORWARD = old_tile_ff


def test_without_classification_head_keeps_transferable_weights():
    state_dict = {
        "tile_branch.encoder.weight": torch.ones(2, 2),
        "head.latent_proj.0.weight": torch.ones(2, 2),
        "head.classifier.0.weight": torch.ones(3, 2),
        "head.classifier.0.bias": torch.ones(3),
        "module.head.arcface_head.weight": torch.ones(3, 2),
    }

    transfer_state = without_classification_head(state_dict)

    assert set(transfer_state) == {
        "tile_branch.encoder.weight",
        "head.latent_proj.0.weight",
    }
