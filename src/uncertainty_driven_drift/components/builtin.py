"""Minimal built-in components used for smoke tests.

These exist so the infrastructure can be exercised end-to-end **without**
pulling torch / river / laplace-torch. Real implementations (synthetic
TV-ground-truth streams, Laplace uncertainty, river detector adapters)
are added in follow-up steps and registered in their own modules.
"""

from __future__ import annotations

from typing import Iterable, Iterator, List

import numpy as np

from uncertainty_driven_drift.data.base import DatasetStream, StreamBatch, StreamSpec
from uncertainty_driven_drift.detectors.base import DetectorEvent, DriftDetector
from uncertainty_driven_drift.models.base import Classifier, Prediction
from uncertainty_driven_drift.registry import register
from uncertainty_driven_drift.uncertainty.base import UncertaintyEstimator, UncertaintyScores


# ---------------------------------------------------------------------------
# Dataset: Bernoulli stream with an abrupt concept flip.
# ---------------------------------------------------------------------------

@register("dataset", "bernoulli_flip")
class BernoulliFlipStream:
    """A 1-D binary stream with one abrupt label flip at the mid-point.

    Purely for smoke testing the runner / writer / detector loop — it is
    the simplest stream that still has a well-defined drift boundary.
    """

    def __init__(
        self,
        n_batches: int = 20,
        batch_size: int = 64,
        drift_fraction: float = 0.5,
        seed: int = 0,
    ) -> None:
        self.n_batches = int(n_batches)
        self.batch_size = int(batch_size)
        self.drift_fraction = float(drift_fraction)
        self.seed = int(seed)
        self.spec = StreamSpec(
            name="bernoulli_flip",
            input_shape=(1,),
            n_classes=2,
            n_batches=self.n_batches,
            batch_size=self.batch_size,
            drift_indices=(int(self.n_batches * self.drift_fraction),),
            has_true_posterior=True,
        )

    def __iter__(self) -> Iterator[StreamBatch]:
        rng = np.random.default_rng(self.seed)
        drift_idx = self.spec.drift_indices[0]
        for i in range(self.n_batches):
            x = rng.normal(size=(self.batch_size, 1)).astype(np.float32)
            concept = 0 if i < drift_idx else 1
            base_prob = 0.3 if concept == 0 else 0.7
            probs = np.full((self.batch_size, 2), 0.0, dtype=np.float64)
            probs[:, 1] = base_prob
            probs[:, 0] = 1.0 - base_prob
            y = (rng.random(self.batch_size) < base_prob).astype(np.int64)
            yield StreamBatch(
                index=i,
                x=x,
                y=y,
                concept_id=concept,
                is_drift=(i == drift_idx),
                true_posterior=probs,
            )


# ---------------------------------------------------------------------------
# Model: prior-only classifier that estimates class frequencies online.
# ---------------------------------------------------------------------------

@register("model", "prior_classifier")
class PriorClassifier:
    """Predicts the running empirical class frequency seen in ``observe``."""

    def __init__(self, smoothing: float = 1.0) -> None:
        self.smoothing = float(smoothing)
        self._counts: np.ndarray | None = None
        self._n_classes = 0

    def setup(self, spec: StreamSpec) -> None:
        self._n_classes = spec.n_classes
        self._counts = np.full(self._n_classes, self.smoothing, dtype=np.float64)

    def predict(self, batch: StreamBatch) -> Prediction:
        assert self._counts is not None, "Call setup() first"
        probs_row = self._counts / self._counts.sum()
        probs = np.broadcast_to(probs_row, (batch.x.shape[0], self._n_classes)).copy()
        return Prediction(probs=probs)

    def observe(self, batch: StreamBatch, prediction: Prediction) -> None:
        assert self._counts is not None
        for c in range(self._n_classes):
            self._counts[c] += float(np.sum(batch.y == c))


# ---------------------------------------------------------------------------
# Uncertainty: predictive entropy only (total = epistemic for smoke test).
# ---------------------------------------------------------------------------

@register("uncertainty", "predictive_entropy")
class PredictiveEntropy:
    """Per-sample entropy of the predictive categorical distribution."""

    def setup(self, model: Classifier) -> None:
        del model  # not needed

    def fit(self, buffer: Iterable[StreamBatch]) -> None:
        del buffer  # stateless

    def score(self, batch: StreamBatch, prediction: Prediction) -> UncertaintyScores:
        eps = 1e-12
        ent = -np.sum(prediction.probs * np.log(prediction.probs + eps), axis=1)
        return UncertaintyScores(total=ent, epistemic=ent, aleatoric=None)


# ---------------------------------------------------------------------------
# Detector: running-mean z-score on chunk error rate (simple sanity check).
# ---------------------------------------------------------------------------

@register("detector", "running_mean_error")
class RunningMeanErrorDetector:
    """Alarm when the chunk error rate exceeds ``threshold`` times its
    running mean. Not meant to be competitive — just a smoke detector."""

    def __init__(self, threshold: float = 2.0, warmup: int = 3) -> None:
        self.name = "running_mean_error"
        self.threshold = float(threshold)
        self.warmup = int(warmup)
        self._errors: List[float] = []

    def update(self, batch, prediction, scores) -> DetectorEvent:
        preds = prediction.probs.argmax(axis=1)
        err = float((preds != batch.y).mean())
        self._errors.append(err)
        if len(self._errors) <= self.warmup:
            return DetectorEvent(alarm=False, statistic=err,
                                 diagnostics={"warmup": True})
        baseline = float(np.mean(self._errors[: -1])) + 1e-9
        ratio = err / baseline
        alarm = ratio > self.threshold
        return DetectorEvent(
            alarm=alarm,
            statistic=ratio,
            diagnostics={"error": err, "baseline_error": baseline},
        )

    def reset(self) -> None:
        self._errors = []
