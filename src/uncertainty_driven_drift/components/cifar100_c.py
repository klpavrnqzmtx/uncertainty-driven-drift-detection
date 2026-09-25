"""CIFAR-100 known-vs-novel corruption stream.

The 100-class analogue of :mod:`cifar_c`.  Streams CIFAR-100 test images with
on-the-fly corruptions drawn from the same CIFAR-10-C corruption family
(Hendrycks & Dietterich, 2019) — the corruptions operate on 3x32x32 RGB images
and are class-agnostic, so the exact same pipeline is reused; only the base
dataset (CIFAR-100, 100 classes) differs.  No large download is required beyond
the CIFAR-100 tarball itself; corruptions are generated in-process.

Registered components
---------------------
* ``cifar100_known_novel`` — DatasetStream, input_shape=(3, 32, 32), n_classes=100
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Sequence

import numpy as np

from uncertainty_driven_drift.components.cifar_c import _SHORT_NAME
from uncertainty_driven_drift.components.cifar_c_corruptions import (
    apply_corruption,
    available_corruptions,
)
from uncertainty_driven_drift.data.base import DatasetStream, StreamBatch, StreamSpec
from uncertainty_driven_drift.registry import register

_CIFAR100_CLASSES = 100
_INPUT_SHAPE = (3, 32, 32)


def _load_cifar100_test_pool(data_root: Path):
    """Return (x, y) for the CIFAR-100 test set as (N, 3, 32, 32) float32."""
    data_root.mkdir(parents=True, exist_ok=True)
    try:
        from torchvision import datasets
    except ImportError as exc:
        raise ImportError(
            "cifar100_known_novel requires torchvision. Install: pip install torchvision"
        ) from exc
    ds = datasets.CIFAR100(str(data_root), train=False, download=True)
    x = ds.data.astype(np.float32) / 255.0          # (10000, 32, 32, 3)
    x = x.transpose(0, 3, 1, 2).astype(np.float32)  # (10000, 3, 32, 32)
    y = np.array(ds.targets, dtype=np.int64)         # (10000,)
    return x, y


@register("dataset", "cifar100_known_novel")
class CIFAR100KnownNovelStream(DatasetStream):
    """CIFAR-100 stream with known-then-novel corruption phases (see :mod:`cifar_c`)."""

    def __init__(
        self,
        known_corruptions: Sequence[str],
        novel_corruptions: Sequence[str],
        n_batches_per_phase: int = 20,
        severity: int = 3,
        batch_size: int = 64,
        seed: int = 0,
        data_root: str = "./artifacts/cifar100",
    ) -> None:
        if not known_corruptions:
            raise ValueError("known_corruptions must not be empty")
        if not novel_corruptions:
            raise ValueError("novel_corruptions must not be empty")

        all_corruptions = list(known_corruptions) + list(novel_corruptions)
        avail = set(available_corruptions())
        bad = [c for c in all_corruptions if c not in avail]
        if bad:
            raise KeyError(f"Unknown corruptions {bad!r}; have {sorted(avail)}")

        self.known_corruptions = list(known_corruptions)
        self.novel_corruptions = list(novel_corruptions)
        self.n_batches_per_phase = int(n_batches_per_phase)
        self.severity = int(severity)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.data_root = str(data_root)

        n_known = len(self.known_corruptions)
        n_novel = len(self.novel_corruptions)
        total_batches = (n_known + n_novel) * self.n_batches_per_phase
        novel_start = n_known * self.n_batches_per_phase

        drift_indices: List[int] = [
            i * self.n_batches_per_phase for i in range(1, n_known + n_novel)
        ]

        self._pool_x, self._pool_y = _load_cifar100_test_pool(Path(self.data_root))

        channel_names = [
            _SHORT_NAME.get(c, c)
            for c in self.known_corruptions + self.novel_corruptions
        ]

        self.spec = StreamSpec(
            name="cifar100_known_novel",
            input_shape=_INPUT_SHAPE,
            n_classes=_CIFAR100_CLASSES,
            n_batches=total_batches,
            batch_size=self.batch_size,
            drift_indices=tuple(drift_indices),
            has_true_posterior=False,
            extras={
                "severity": self.severity,
                "known_corruptions": self.known_corruptions,
                "novel_corruptions": self.novel_corruptions,
                "novel_start_batch": novel_start,
                "channel_names": channel_names,
                "pool_size": int(self._pool_x.shape[0]),
                "available_corruptions": available_corruptions(),
            },
        )

    def __iter__(self) -> Iterator[StreamBatch]:
        sample_rng = np.random.default_rng(self.seed)
        corrupt_rng = np.random.default_rng(self.seed + 1)

        phase_corruptions = self.known_corruptions + self.novel_corruptions
        novel_start = self.spec.extras["novel_start_batch"]
        drift_set = set(self.spec.drift_indices)
        pool_n = self._pool_x.shape[0]

        t = 0
        for phase_i, corruption in enumerate(phase_corruptions):
            for _ in range(self.n_batches_per_phase):
                idx = sample_rng.integers(0, pool_n, size=self.batch_size)
                x_clean = self._pool_x[idx]
                y = self._pool_y[idx]
                x = apply_corruption(corruption, x_clean, self.severity, corrupt_rng)
                yield StreamBatch(
                    index=t,
                    x=x,
                    y=y,
                    concept_id=phase_i,
                    is_drift=(t in drift_set),
                    true_posterior=None,
                    extras={"corruption": corruption, "is_novel": t >= novel_start},
                )
                t += 1
