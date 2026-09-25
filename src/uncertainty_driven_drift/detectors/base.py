"""Drift detector API.

A ``DriftDetector`` receives one batch's worth of signal per step and may
emit an **alarm**. To keep river, custom chi-square, and supervised
detectors behind the same interface, ``update`` accepts all channels the
runner has at that step and the detector picks what it needs:

* ``batch``: raw inputs and (optionally) labels.
* ``prediction``: model probabilities / features.
* ``scores``: uncertainty decomposition for the batch.

The detector returns a :class:`DetectorEvent` describing the alarm and
any diagnostic statistic (p-value, running mean, window size, ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, runtime_checkable

from uncertainty_driven_drift.data.base import StreamBatch
from uncertainty_driven_drift.models.base import Prediction
from uncertainty_driven_drift.uncertainty.base import UncertaintyScores


@dataclass
class DetectorEvent:
    """Result of one detector update."""

    alarm: bool
    statistic: float = 0.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DriftDetector(Protocol):
    name: str

    def update(
        self,
        batch: StreamBatch,
        prediction: Prediction,
        scores: UncertaintyScores,
    ) -> DetectorEvent: ...

    def reset(self) -> None: ...
