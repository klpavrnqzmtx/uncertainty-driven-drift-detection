"""CIFAR-10 severity-shift stream (known-vs-novel by *severity*, not type).

A variant of :mod:`cifar_c` where the novel phase re-uses the *same* corruptions
the model was trained on, only at a higher severity.  This isolates a different
kind of drift: the input distribution stays qualitatively familiar (same
corruption families) but intensifies, so the model degrades gracefully rather
than encountering an unseen corruption type.

* **Known phase**: each corruption at ``known_severity`` (what the model saw in
  training) — low error, low uncertainty.
* **Novel phase**: the same corruptions at ``novel_severity`` (harder) — the
  drift the detectors should catch.

This is a *new* component; it does not modify ``cifar_known_novel``.  It reuses
the corruption implementations and the CIFAR-10 test pool from :mod:`cifar_c`.

Registered components
---------------------
* ``cifar_severity_shift`` — DatasetStream, input_shape=(3, 32, 32), n_classes=10
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Sequence

import numpy as np

from uncertainty_driven_drift.components.cifar_c import _SHORT_NAME, _load_cifar10_test_pool
from uncertainty_driven_drift.components.cifar_c_corruptions import (
    apply_corruption,
    available_corruptions,
)
from uncertainty_driven_drift.data.base import DatasetStream, StreamBatch, StreamSpec
from uncertainty_driven_drift.registry import register

_CIFAR_CLASSES = 10
_INPUT_SHAPE = (3, 32, 32)


@register("dataset", "cifar_severity_shift")
class CIFARSeverityShiftStream(DatasetStream):
    """CIFAR-10 stream that drifts by corruption *severity* rather than type.

    Parameters
    ----------
    corruptions :
        Corruption types the model was trained on.  Presented once at
        ``known_severity`` (known phase) then again at ``novel_severity``
        (novel phase).
    known_severity, novel_severity :
        CIFAR-10-C severities (1–5) for the two phases.  ``novel_severity``
        should exceed ``known_severity`` for the intended "same corruption,
        harder" drift.
    n_batches_per_phase :
        Pure batches per corruption type within each phase.
    batch_size, seed, data_root :
        As in :class:`cifar_c.CIFARKnownNovelStream`.
    """

    def __init__(
        self,
        corruptions: Sequence[str],
        known_severity: int = 1,
        novel_severity: int = 3,
        n_batches_per_phase: int = 20,
        batch_size: int = 64,
        seed: int = 0,
        data_root: str = "./artifacts/cifar10",
    ) -> None:
        if not corruptions:
            raise ValueError("corruptions must not be empty")
        avail = set(available_corruptions())
        bad = [c for c in corruptions if c not in avail]
        if bad:
            raise KeyError(f"Unknown corruptions {bad!r}; have {sorted(avail)}")

        self.corruptions = list(corruptions)
        self.known_severity = int(known_severity)
        self.novel_severity = int(novel_severity)
        self.n_batches_per_phase = int(n_batches_per_phase)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.data_root = str(data_root)

        n = len(self.corruptions)
        total_batches = 2 * n * self.n_batches_per_phase
        novel_start = n * self.n_batches_per_phase
        drift_indices = [i * self.n_batches_per_phase for i in range(1, 2 * n)]

        self._pool_x, self._pool_y = _load_cifar10_test_pool(Path(self.data_root))

        # Label each block with its severity so the known/novel split reads
        # as "same corruption, harder" (e.g. "bright s1" ... "bright s3").
        channel_names = (
            [f"{_SHORT_NAME.get(c, c)} s{self.known_severity}" for c in self.corruptions]
            + [f"{_SHORT_NAME.get(c, c)} s{self.novel_severity}" for c in self.corruptions]
        )

        self.spec = StreamSpec(
            name="cifar_severity_shift",
            input_shape=_INPUT_SHAPE,
            n_classes=_CIFAR_CLASSES,
            n_batches=total_batches,
            batch_size=self.batch_size,
            drift_indices=tuple(drift_indices),
            has_true_posterior=False,
            extras={
                "known_severity": self.known_severity,
                "novel_severity": self.novel_severity,
                "known_corruptions": self.corruptions,
                # Novel = same corruption types, so record them for the plotting
                # code that colours the novel phase.
                "novel_corruptions": self.corruptions,
                "novel_start_batch": novel_start,
                "channel_names": channel_names,
                "pool_size": int(self._pool_x.shape[0]),
                "available_corruptions": available_corruptions(),
            },
        )

    def __iter__(self) -> Iterator[StreamBatch]:
        sample_rng = np.random.default_rng(self.seed)
        corrupt_rng = np.random.default_rng(self.seed + 1)

        # (corruption, severity) per phase block: known first, then novel.
        blocks = (
            [(c, self.known_severity) for c in self.corruptions]
            + [(c, self.novel_severity) for c in self.corruptions]
        )
        novel_start = self.spec.extras["novel_start_batch"]
        drift_set = set(self.spec.drift_indices)
        pool_n = self._pool_x.shape[0]

        t = 0
        for phase_i, (corruption, severity) in enumerate(blocks):
            for _ in range(self.n_batches_per_phase):
                idx = sample_rng.integers(0, pool_n, size=self.batch_size)
                x_clean = self._pool_x[idx]
                y = self._pool_y[idx]
                x = apply_corruption(corruption, x_clean, severity, corrupt_rng)
                yield StreamBatch(
                    index=t,
                    x=x,
                    y=y,
                    concept_id=phase_i,
                    is_drift=(t in drift_set),
                    true_posterior=None,
                    extras={
                        "corruption": corruption,
                        "severity": severity,
                        "is_novel": t >= novel_start,
                    },
                )
                t += 1
