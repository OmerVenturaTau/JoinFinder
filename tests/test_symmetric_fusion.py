import torch

from models.symmetric_fusion import SymmetricRetrievalFusion
from models.tile_branch import TileSetSummarizer
from losses.combined_loss import CombinedLoss
from system import (
    D_MODEL,
    LATENT_DIM,
    SYMMETRIC_BRANCH_DIM,
    USE_BRANCH_ADAPTERS_AND_SUMMARIZERS,
)


def test_configured_symmetric_output_width_matches_architecture_flag():
    assert D_MODEL == 768
    expected_branch_dim = (
        SYMMETRIC_BRANCH_DIM
        if USE_BRANCH_ADAPTERS_AND_SUMMARIZERS
        else D_MODEL
    )
    assert LATENT_DIM == 3 * expected_branch_dim


def test_symmetric_fusion_has_no_projection_adapter_or_post_pool_transformer():
    module = _fusion()
    assert not hasattr(module, "adapters")
    assert not hasattr(module, "modality_transformer")


def test_optional_projection_adapters_change_branch_and_output_width():
    module = SymmetricRetrievalFusion(
        d_model=8,
        branch_dim=4,
        enabled_modalities=("tile", "glyph"),
        reliability_hidden_dim=8,
        use_adapters=True,
    ).eval()
    latent, details = module(
        torch.randn(2, 3, 8), None,
        torch.randn(2, 2, 8), None,
    )
    assert hasattr(module, "adapters")
    assert details["tile"].shape == (2, 4)
    assert details["glyph"].shape == (2, 4)
    assert latent.shape == (2, 8)


def _fusion():
    return SymmetricRetrievalFusion(
        d_model=8,
        enabled_modalities=("tile", "glyph"),
        reliability_hidden_dim=8,
    )


def test_symmetric_fusion_starts_equal_and_normalized():
    module = _fusion().eval()
    tile = torch.randn(3, 4, 8)
    glyph = torch.randn(3, 2, 8)
    latent, details = module(tile, torch.ones(3, 4, dtype=torch.bool),
                             glyph, torch.ones(3, 2, dtype=torch.bool))
    assert latent.shape == (3, 16)
    torch.testing.assert_close(latent.norm(dim=1), torch.ones(3), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(details["reliability_weights"], torch.full((3, 2), 0.5))


def test_symmetric_attention_is_exact_masked_pooling_contribution():
    module = SymmetricRetrievalFusion(
        d_model=8,
        enabled_modalities=("tile", "glyph", "word"),
        reliability_hidden_dim=8,
    ).eval()
    tile = torch.randn(1, 3, 8)
    glyph = torch.randn(1, 2, 8)
    word = torch.randn(1, 4, 8)
    tile_mask = torch.tensor([[True, True, False]])
    glyph_mask = torch.tensor([[True, False]])
    word_mask = torch.tensor([[True, True, True, True]])

    _, details = module(
        tile, tile_mask, glyph, glyph_mask, word, word_mask
    )

    # Reliability starts equal (1/3 each). Within a modality, symmetric
    # fusion uses a strict mask mean, so contribution is uniform over valid
    # tokens and exactly zero on padding.
    expected = torch.tensor([[1 / 6, 1 / 6, 0, 1 / 3, 0,
                              1 / 12, 1 / 12, 1 / 12, 1 / 12]])
    assert details["attention"].shape == (1, 1, 9)
    torch.testing.assert_close(details["attention"][:, 0], expected)
    torch.testing.assert_close(details["attention"].sum(dim=2), torch.ones(1, 1))


def test_symmetric_attention_renormalizes_when_modality_is_missing():
    module = _fusion().eval()
    tile = torch.randn(1, 2, 8)
    glyph = torch.randn(1, 3, 8)
    glyph_mask = torch.zeros(1, 3, dtype=torch.bool)
    _, details = module(tile, None, glyph, glyph_mask)
    expected = torch.tensor([[[0.5, 0.5, 0.0, 0.0, 0.0]]])
    torch.testing.assert_close(details["attention"], expected)


def test_summary_source_mask_flows_to_fusion_availability_scorer():
    """Mask source logits, retain all queries, then mask only empty modalities."""
    summarizer = TileSetSummarizer(
        d_model=8,
        num_queries=4,
        num_heads=2,
        num_cross_attn_layers=1,
        dropout=0.0,
    ).eval()
    source_tokens = torch.randn(2, 3, 8)
    source_mask = torch.tensor([
        [True, False, False],
        [False, False, False],
    ])
    summaries, summary_mask, query_to_source = summarizer(
        source_tokens,
        source_mask,
        return_attention=True,
    )

    assert summary_mask.tolist() == [[True] * 4, [False] * 4]
    torch.testing.assert_close(query_to_source[0, :, 1:], torch.zeros(4, 2))
    torch.testing.assert_close(query_to_source[1], torch.zeros(4, 3))

    fusion = SymmetricRetrievalFusion(
        d_model=8,
        enabled_modalities=("tile", "glyph"),
        reliability_hidden_dim=1,
    ).eval()
    with torch.no_grad():
        fusion.reliability[0].weight.zero_()
        fusion.reliability[0].bias.zero_()
        fusion.reliability[0].weight[0, 0] = 1.0
        fusion.reliability[2].weight.fill_(1.0)
    glyph = torch.randn(2, 2, 8)
    glyph_mask = torch.ones(2, 2, dtype=torch.bool)
    tile_evidence = source_mask.float().mean(dim=1)
    glyph_evidence = torch.ones(2)
    _, details = fusion(
        summaries,
        summary_mask,
        glyph,
        glyph_mask,
        tile_evidence_fraction=tile_evidence,
        glyph_evidence_fraction=glyph_evidence,
    )

    torch.testing.assert_close(
        details["evidence_fractions"],
        torch.tensor([[1 / 3, 1.0], [0.0, 1.0]]),
    )
    assert details["modality_available"].tolist() == [
        [True, True],
        [False, True],
    ]
    assert details["reliability_weights"][0, 1] > details["reliability_weights"][0, 0]
    torch.testing.assert_close(
        details["reliability_weights"][1], torch.tensor([0.0, 1.0])
    )


def test_all_modalities_mask_mean_pool_summaries_at_native_width():
    module = SymmetricRetrievalFusion(
        d_model=8,
        enabled_modalities=("tile", "glyph", "word"),
        reliability_hidden_dim=8,
    ).eval()
    tokens = {
        "tile": torch.randn(2, 3, 8),
        "glyph": torch.randn(2, 4, 8),
        "word": torch.randn(2, 5, 8),
    }
    masks = {
        "tile": torch.tensor([[True, True, False], [True, False, False]]),
        "glyph": torch.tensor([[True, False, True, False], [True, True, True, False]]),
        "word": torch.tensor([[True, True, False, False, False], [True, False, True, True, False]]),
    }
    latent, details = module(
        tokens["tile"], masks["tile"],
        tokens["glyph"], masks["glyph"],
        tokens["word"], masks["word"],
    )

    for name in ("tile", "glyph", "word"):
        weights = masks[name].unsqueeze(-1).to(tokens[name].dtype)
        pooled = (tokens[name] * weights).sum(dim=1) / weights.sum(dim=1)
        expected = torch.nn.functional.normalize(pooled, dim=1)
        torch.testing.assert_close(details[name], expected)
    assert latent.shape == (2, 24)
    torch.testing.assert_close(latent.norm(dim=1), torch.ones(2))


def test_weighted_concatenation_cosine_matches_weighted_branch_cosines():
    module = _fusion().eval()
    tile = torch.randn(2, 3, 8)
    glyph = torch.randn(2, 3, 8)
    _, details = module(tile, None, glyph, None)
    fused = details["fusion"]
    expected = 0.5 * (details["tile"][0] @ details["tile"][1])
    expected += 0.5 * (details["glyph"][0] @ details["glyph"][1])
    torch.testing.assert_close(fused[0] @ fused[1], expected, atol=1e-6, rtol=1e-6)


def test_missing_modality_is_masked_and_remaining_weight_is_one():
    module = _fusion().eval()
    tile = torch.randn(2, 3, 8)
    glyph = torch.randn(2, 3, 8)
    glyph_mask = torch.tensor([[False, False, False], [True, True, True]])
    _, details = module(tile, None, glyph, glyph_mask)
    torch.testing.assert_close(details["reliability_weights"][0], torch.tensor([1.0, 0.0]))
    assert torch.count_nonzero(details["glyph"][0]) == 0


def test_missing_modality_backward_has_only_finite_gradients():
    module = _fusion().train()
    tile = torch.randn(2, 3, 8, requires_grad=True)
    glyph = torch.randn(2, 3, 8, requires_grad=True)
    # Exercise both a per-sample missing branch and a fully present sample,
    # matching modality dropout during accumulated training.
    glyph_mask = torch.tensor([[False, False, False], [True, True, True]])
    latent, _ = module(tile, None, glyph, glyph_mask)
    latent.square().sum().backward()
    for parameter in module.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    assert torch.isfinite(tile.grad).all()
    assert torch.isfinite(glyph.grad).all()


def test_padding_values_do_not_change_output():
    module = _fusion().eval()
    tile = torch.randn(1, 3, 8)
    glyph = torch.randn(1, 3, 8)
    mask = torch.tensor([[True, True, False]])
    first, _ = module(tile, None, glyph, mask)
    glyph[:, 2] = 10000
    second, _ = module(tile, None, glyph, mask)
    torch.testing.assert_close(first, second)


def test_accumulated_optimizer_steps_remain_finite_with_modality_dropout():
    torch.manual_seed(7)
    module = _fusion().train()
    loss_fn = CombinedLoss(
        num_classes=5,
        embedding_dim=16,
        aux_embedding_dim=8,
        tile_aux_weight=0.2,
        glyph_aux_weight=0.2,
        shared_branch_arcface=True,
        gate_entropy_weight=0.01,
        gate_entropy_decay_epochs=2,
        arcface_margin=0.12,
        arcface_scale=24.0,
    )
    classifier = torch.nn.Linear(16, 5)
    parameters = list(module.parameters()) + list(classifier.parameters()) + list(loss_fn.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=2e-5)
    optimizer.zero_grad()
    masks = (
        torch.tensor([[True, True, True], [True, True, True]]),
        torch.tensor([[False, False, False], [True, True, True]]),
        torch.tensor([[True, True, True], [False, False, False]]),
        torch.tensor([[False, False, False], [False, False, False]]),
    )
    for step in range(12):
        tile = torch.randn(2, 3, 8)
        glyph = torch.randn(2, 3, 8)
        glyph_mask = masks[step % len(masks)]
        tile_mask = masks[(step + 2) % len(masks)]
        # Ensure the second sample always retains at least one modality, as the
        # production modality-dropout rescue does.
        both_empty = ~(tile_mask.any(dim=1) | glyph_mask.any(dim=1))
        tile_mask = tile_mask.clone()
        tile_mask[both_empty, 0] = True
        latent, details = module(tile, tile_mask, glyph, glyph_mask)
        logits = classifier(latent)
        labels = torch.tensor([step % 5, (step + 1) % 5])
        total, *_ = loss_fn(logits, latent, labels, details)
        assert torch.isfinite(total)
        (total / 4).backward()
        if (step + 1) % 4 == 0:
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad()
            for parameter in parameters:
                assert torch.isfinite(parameter).all()
