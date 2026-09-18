from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Iterable, Iterator, List, Sequence

from torch.utils.data import Sampler


class PKBatchSampler(Sampler[List[int]]):
    """Batch sampler that draws P labels and K examples per label."""

    def __init__(
        self,
        labels: Sequence[object],
        batch_size: int,
        k: int = 2,
        seed: int = 42,
        drop_last: bool = False,
    ):
        if int(batch_size) < 2:
            raise ValueError(f"PK contrastive batches require batch_size >= 2, got {batch_size}")
        if int(k) < 2:
            raise ValueError(f"K must be >= 2 so each batch has in-batch positives, got {k}")
        if int(k) > int(batch_size):
            raise ValueError(f"K must be <= batch_size, got K={k}, batch_size={batch_size}")
        if int(batch_size) % int(k) != 0:
            raise ValueError(
                f"PK contrastive batch_size must be divisible by K to avoid silently shrinking batches, "
                f"got batch_size={batch_size}, K={k}"
            )
        self.labels = [str(x) for x in labels]
        self.batch_size = int(batch_size)
        self.k = int(k)
        self.p = self.batch_size // self.k
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        self.indices_by_label: dict[str, list[int]] = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.indices_by_label[label].append(idx)
        if not self.indices_by_label:
            raise ValueError("PKBatchSampler received no labels")

        self.unique_labels = sorted(self.indices_by_label)
        self.num_batches = len(self.labels) // self.batch_size
        if not self.drop_last and len(self.labels) % self.batch_size:
            self.num_batches += 1
        self.num_batches = max(1, self.num_batches)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        labels = list(self.unique_labels)
        for _ in range(self.num_batches):
            if len(labels) >= self.p:
                chosen_labels = rng.sample(labels, self.p)
            else:
                chosen_labels = [rng.choice(labels) for _ in range(self.p)]

            batch: list[int] = []
            for label in chosen_labels:
                candidates = self.indices_by_label[label]
                if len(candidates) >= self.k:
                    batch.extend(rng.sample(candidates, self.k))
                else:
                    batch.extend(rng.choices(candidates, k=self.k))
            yield batch
