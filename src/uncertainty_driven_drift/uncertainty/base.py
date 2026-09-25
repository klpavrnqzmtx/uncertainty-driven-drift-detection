"""Uncertainty API.

An ``UncertaintyEstimator`` returns **per-sample** scalar scores:

* ``total``: total predictive uncertainty (e.g. entropy of the mean).
* ``epistemic``: knowledge / model uncertainty (e.g. mutual information).
* ``aleatoric``: data noise (defaults to ``total - epistemic`` when the
  estimator does not provide it explicitly).

Estimators may keep internal state that depends on the model, in which case
``fit`` is called periodically by the runner with a reference buffer of
recent batches. Estimators that don't need fitting implement a no-op.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Protocol, runtime_checkable

import numpy as np

from uncertainty_driven_drift.data.base import StreamBatch
from uncertainty_driven_drift.models.base import Classifier, Prediction


@dataclass
class UncertaintyScores:
    """Per-sample uncertainty decomposition for one batch."""

    total: np.ndarray                              # (B,)
    epistemic: np.ndarray                          # (B,)
    aleatoric: Optional[np.ndarray] = None         # (B,) or None
    extras: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class UncertaintyEstimator(Protocol):
    def setup(self, model: Classifier) -> None: ...

    def fit(self, buffer: Iterable[StreamBatch]) -> None:
        """Refresh internal state from a reference buffer (no-op by default)."""

    def score(self, batch: StreamBatch, prediction: Prediction) -> UncertaintyScores: ...
