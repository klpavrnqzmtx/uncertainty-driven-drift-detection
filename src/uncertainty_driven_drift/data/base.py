"""Dataset stream API.

A ``DatasetStream`` is an iterable of ``StreamBatch`` objects. Each batch
carries inputs and labels, plus per-batch metadata used by the evaluation:

* ``is_drift``: whether a drift boundary sits just before this batch.
* ``concept_id``: integer tag for the current generating concept.
* ``true_posterior``: optional oracle ``p*(y|x)`` for every sample in the
  batch; only synthetic streams with a known ground truth populate this.
  When present, it enables posterior-mismatch (TV) computation without
  labels.

Implementations are registered under ``kind='dataset'`` via
:mod:`uncertainty_driven_drift.registry`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional, Protocol, Tuple, runtime_checkable

import numpy as np


@dataclass
class StreamSpec:
    """Static description of a stream.

    ``input_shape`` is the per-sample shape, e.g. ``(2,)`` for SINE,
    ``(3, 32, 32)`` for CIFAR. ``drift_indices`` lists batch indices where
    a drift boundary is located (only populated for streams where this is
    known a priori).
    """

    name: str
    input_shape: Tuple[int, ...]
    n_classes: int
    n_batches: int
    batch_size: int
    drift_indices: Tuple[int, ...] = ()
    has_true_posterior: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamBatch:
    """One batch drawn from a stream."""

    index: int
    x: np.ndarray                      # (B, *input_shape)
    y: np.ndarray                      # (B,) int64
    concept_id: int = 0
    is_drift: bool = False
    true_posterior: Optional[np.ndarray] = None   # (B, n_classes), optional
    extras: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DatasetStream(Protocol):
    """Stream of ``StreamBatch`` objects.

    Iteration must be **deterministic** given the stream's seed so runs are
    reproducible. ``spec`` is consulted before iteration to size models and
    pre-allocate arrays.
    """

    spec: StreamSpec

    def __iter__(self) -> Iterator[StreamBatch]: ...
