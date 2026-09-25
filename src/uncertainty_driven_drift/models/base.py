"""Model API.

A ``Classifier`` predicts class probabilities on a batch and optionally
updates itself when the labels arrive (prequential / test-then-train).
We keep the interface deliberately narrow: anything an uncertainty
estimator or detector needs should go through this surface, not through
framework-specific objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import numpy as np

from uncertainty_driven_drift.data.base import StreamBatch, StreamSpec


@dataclass
class Prediction:
    """Model output for one batch.

    ``probs`` is the predicted categorical posterior (softmaxed). ``features``
    is an optional per-sample penultimate representation, used by uncertainty
    estimators that need embeddings (e.g. last-layer Laplace).
    """

    probs: np.ndarray                                # (B, n_classes)
    features: Optional[np.ndarray] = None            # (B, D) or None
    extras: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Classifier(Protocol):
    """Classifier interface used by the runner.

    ``setup`` is called once before iteration begins with the stream spec,
    so models can adapt their output dimension or input shape to the stream
    without re-construction.
    """

    def setup(self, spec: StreamSpec) -> None: ...

    def predict(self, batch: StreamBatch) -> Prediction: ...

    def observe(self, batch: StreamBatch, prediction: Prediction) -> None:
        """Optional update after seeing the labels (no-op by default)."""
