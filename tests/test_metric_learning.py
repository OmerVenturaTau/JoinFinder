import torch
import os
import sys
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train.metric_learning import (
    LabeledFeatureQueue,
    MemoryBankSupConLoss,
    _bounded_quantile_input,
    compute_retrieval_metrics,
)


def test_bounded_quantile_input_uses_in_bounds_integer_subsample():
    values = torch.arange(11)

    sampled = _bounded_quantile_input(values, max_scores=4)

    assert sampled.tolist() == [0, 2, 5, 8]
from train.pk_sampler import PKBatchSampler


def test_labeled_feature_queue_wraps_and_keeps_labels_aligned():
    queue = LabeledFeatureQueue(capacity=3, feature_dim=2, device=torch.device("cpu"))
    queue.enqueue(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([10, 20]))
    queue.enqueue(torch.tensor([[1.0, 1.0], [-1.0, 0.0]]), torch.tensor([30, 40]))

    features, labels = queue.get()
    assert len(queue) == 3
    assert features.shape == (3, 2)
    assert labels.tolist() == [40, 20, 30]


def test_memory_bank_supcon_uses_bank_positives_and_has_gradients():
    loss_fn = MemoryBankSupConLoss(temperature=0.1)
    anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    labels = torch.tensor([1, 2])
    bank_features = torch.tensor([[0.9, 0.1], [1.0, 0.0], [0.0, 1.0]])
    bank_labels = torch.tensor([1, 3, 2])

    loss, stats = loss_fn(anchors, labels, bank_features, bank_labels)
    loss.backward()

    assert loss.item() > 0
    assert anchors.grad is not None
    assert stats["valid_anchor_frac"] == 1.0
    assert stats["positive_pairs"] >= 2


def test_memory_bank_supcon_ignores_anchors_without_positives():
    loss_fn = MemoryBankSupConLoss(temperature=0.1)
    anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    labels = torch.tensor([1, 2])
    bank_features = torch.tensor([[0.5, 0.5]])
    bank_labels = torch.tensor([3])

    loss, stats = loss_fn(anchors, labels, bank_features, bank_labels)

    assert loss.item() == 0.0
    assert stats["valid_anchor_frac"] == 0.0


def test_pk_batch_sampler_produces_positive_pairs():
    labels = ["a", "a", "b", "b", "c", "c"]
    sampler = PKBatchSampler(labels, batch_size=4, k=2, seed=1)
    batch = next(iter(sampler))
    batch_labels = [labels[i] for i in batch]

    assert len(batch) == 4
    assert len(set(batch_labels)) == 2
    assert all(batch_labels.count(label) == 2 for label in set(batch_labels))


def test_pk_batch_sampler_rejects_non_divisible_k():
    labels = ["a", "a", "b", "b", "c", "c"]
    with pytest.raises(ValueError, match="divisible by K"):
        PKBatchSampler(labels, batch_size=4, k=3, seed=1)


def test_retrieval_metrics_perfect_clusters():
    features = torch.tensor([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
        [0.1, 0.9],
    ])
    labels = torch.tensor([1, 1, 2, 2])

    metrics = compute_retrieval_metrics(features, labels)

    assert metrics.knn_at_1 == 1.0
    assert metrics.knn_at_5 == 1.0
    assert metrics.mean_average_precision == 1.0
    assert metrics.same_cosine_mean > metrics.different_cosine_mean
    assert metrics.same_diff_cosine_gap > 0
    assert metrics.hard_negative_cosine_p95 >= metrics.different_cosine_mean
    assert metrics.num_queries == 4
