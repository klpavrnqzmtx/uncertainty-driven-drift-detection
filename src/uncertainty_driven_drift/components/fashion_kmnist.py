"""Fashion-MNIST and KMNIST known-vs-novel corruption streams.

Both datasets share the 28×28 grayscale format with MNIST, so the full
MNIST-C corruption library (``mnist_c_corruptions.py``) applies unchanged.
The ``MNISTKnownNovelStream`` pattern is reused by overriding ``_load_pool``
to swap the test-set pool without touching the schedule machinery.

Registered datasets
-------------------
* ``fashion_known_novel`` — Fashion-MNIST (10 clothing categories).
* ``kmnist_known_novel``  — Kuzushiji-MNIST (10 Japanese cursive characters).

For both, the LeNet MC-Dropout model is reused via ``lenet_mc_dropout``
with ``dataset_name`` set to ``fashion_mnist`` or ``kmnist`` respectively.
This ensures pretraining uses the correct source dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from uncertainty_driven_drift.components.mnist_c import MNISTKnownNovelStream
from uncertainty_driven_drift.registry import register


# ---------------------------------------------------------------------------
# Pool loaders
# ---------------------------------------------------------------------------

def _load_torchvision_pool(dataset_cls_name: str, cache_root: Path):
    """Load the test split of a 28×28 grayscale torchvision dataset as numpy arrays."""
    cache_root.mkdir(parents=True, exist_ok=True)
    try:
        from torchvision import datasets
    except ImportError as exc:
        raise ImportError(
            f"{dataset_cls_name} stream requires torchvision. "
            "Install: pip install torchvision"
        ) from exc
    ds_cls = getattr(datasets, dataset_cls_name)
    ds = ds_cls(root=str(cache_root), train=False, download=True, transform=None)
    x = ds.data.numpy().astype(np.float32) / 255.0  # (N, 28, 28) → float [0,1]
    x = x[:, None, :, :]                             # (N, 1, 28, 28)
    y = ds.targets.numpy().astype(np.int64)
    return x, y


# ---------------------------------------------------------------------------
# Fashion-MNIST known-vs-novel stream
# ---------------------------------------------------------------------------

@register("dataset", "fashion_known_novel")
class FashionKnownNovelStream(MNISTKnownNovelStream):
    """Known-vs-novel corruption stream on Fashion-MNIST.

    Identical semantics to :class:`MNISTKnownNovelStream` but the image
    pool is drawn from the Fashion-MNIST test set.  The LeNet backbone
    should be pretrained with ``dataset_name: fashion_mnist`` in the
    model config.

    Parameters mirror :class:`MNISTKnownNovelStream`.
    """

    def __init__(
        self,
        known_corruptions: Sequence[str],
        novel_corruptions: Sequence[str],
        n_batches_per_phase: int = 20,
        severity: int = 3,
        batch_size: int = 128,
        seed: int = 0,
        data_root: str = "./artifacts/fashion_mnist",
    ) -> None:
        super().__init__(
            known_corruptions=known_corruptions,
            novel_corruptions=novel_corruptions,
            n_batches_per_phase=n_batches_per_phase,
            severity=severity,
            batch_size=batch_size,
            seed=seed,
            data_root=data_root,
        )
        self.spec.name = "fashion_known_novel"

    def _load_pool(self):
        return _load_torchvision_pool("FashionMNIST", Path(self.data_root))


# ---------------------------------------------------------------------------
# KMNIST known-vs-novel stream
# ---------------------------------------------------------------------------

@register("dataset", "kmnist_known_novel")
class KMNISTKnownNovelStream(MNISTKnownNovelStream):
    """Known-vs-novel corruption stream on Kuzushiji-MNIST.

    Identical semantics to :class:`MNISTKnownNovelStream` but the image
    pool is drawn from the KMNIST test set.  The LeNet backbone should be
    pretrained with ``dataset_name: kmnist`` in the model config.

    Parameters mirror :class:`MNISTKnownNovelStream`.
    """

    def __init__(
        self,
        known_corruptions: Sequence[str],
        novel_corruptions: Sequence[str],
        n_batches_per_phase: int = 20,
        severity: int = 3,
        batch_size: int = 128,
        seed: int = 0,
        data_root: str = "./artifacts/kmnist",
    ) -> None:
        super().__init__(
            known_corruptions=known_corruptions,
            novel_corruptions=novel_corruptions,
            n_batches_per_phase=n_batches_per_phase,
            severity=severity,
            batch_size=batch_size,
            seed=seed,
            data_root=data_root,
        )
        self.spec.name = "kmnist_known_novel"

    def _load_pool(self):
        return _load_torchvision_pool("KMNIST", Path(self.data_root))
