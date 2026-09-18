from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class SplitLabels:
    train_paths: List[str]
    train_labels: List[str]
    train_xmls: List[str]
    val_paths: List[str]
    val_labels: List[str]
    val_xmls: List[str]
    test_paths: List[str]
    test_labels: List[str]
    test_xmls: List[str]


class LabelHead(ABC):
    """Derive a single class label (string) per sample."""

    name: str

    @abstractmethod
    def label_for(self, *, image_path: str, manuscript_id: str) -> str:
        """Return the class label (string) for this sample."""

    def flatten_splits(self, splits: Dict[str, Dict[str, List[Tuple[str, str]]]]) -> SplitLabels:
        """Convert the `build_splits()` output into flat path/label lists."""
        all_train_paths, all_train_labels, all_train_xmls = [], [], []
        all_val_paths, all_val_labels, all_val_xmls = [], [], []
        all_test_paths, all_test_labels, all_test_xmls = [], [], []

        for mid, items in splits["train"].items():
            for p, x in items:
                all_train_paths.append(p)
                all_train_xmls.append(x)
                all_train_labels.append(self.label_for(image_path=p, manuscript_id=mid))

        for mid, items in splits["val"].items():
            for p, x in items:
                all_val_paths.append(p)
                all_val_xmls.append(x)
                all_val_labels.append(self.label_for(image_path=p, manuscript_id=mid))

        for mid, items in splits["test"].items():
            for p, x in items:
                all_test_paths.append(p)
                all_test_xmls.append(x)
                all_test_labels.append(self.label_for(image_path=p, manuscript_id=mid))

        return SplitLabels(
            train_paths=all_train_paths,
            train_labels=all_train_labels,
            train_xmls=all_train_xmls,
            val_paths=all_val_paths,
            val_labels=all_val_labels,
            val_xmls=all_val_xmls,
            test_paths=all_test_paths,
            test_labels=all_test_labels,
            test_xmls=all_test_xmls,
        )


def build_label_maps(*, all_labels: Iterable[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    uniq = sorted(set(all_labels))
    label2idx = {lab: i for i, lab in enumerate(uniq)}
    idx2label = {i: lab for lab, i in label2idx.items()}
    return label2idx, idx2label


