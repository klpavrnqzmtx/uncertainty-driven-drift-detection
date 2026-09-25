"""Uncertainty decomposition from MC-Dropout samples.

Consumes the ``(S, B, n_classes)`` MC stack that
:class:`LeNetMCDropout` stashes on ``Prediction.extras['mc_probs']`` and
returns the standard decomposition::

    total_i     = H[ (1/S) Σ_s p_s(y|x_i) ]
    aleatoric_i = (1/S) Σ_s H[ p_s(y|x_i) ]
    epistemic_i = total_i - aleatoric_i           (≥ 0)

This is the Bayesian active learning decomposition (BALD / Gal 2016),
extended from binary to multi-class.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from uncertainty_driven_drift.data.base import StreamBatch
from uncertainty_driven_drift.models.base import Classifier, Prediction
from uncertainty_driven_drift.registry import register
from uncertainty_driven_drift.uncertainty.base import UncertaintyEstimator, UncertaintyScores


@register("uncertainty", "mc_dropout")
class MCDropoutUncertainty(UncertaintyEstimator):
    """Entropy-of-mean minus mean-of-entropy over MC-dropout samples."""

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = float(eps)
        self._model: Classifier | None = None

    def setup(self, model: Classifier) -> None:
        self._model = model

    def fit(self, buffer: Iterable[StreamBatch]) -> None:
        del buffer  # the model is frozen; nothing to refit here.

    def score(self, batch: StreamBatch, prediction: Prediction) -> UncertaintyScores:
        mc = prediction.extras.get("mc_probs")
        if mc is None:
            raise ValueError(
                "mc_dropout uncertainty requires the model to populate "
                "prediction.extras['mc_probs'] with a (S, B, K) stack"
            )
        mc = np.asarray(mc, dtype=np.float64)
        if mc.ndim != 3:
            raise ValueError(f"mc_probs must be (S, B, K); got shape {mc.shape}")

        mean_probs = mc.mean(axis=0)                             # (B, K)
        total = _entropy(mean_probs, self.eps)                   # (B,)

        sample_entropy = _entropy(
            mc.reshape(-1, mc.shape[-1]), self.eps,
        ).reshape(mc.shape[0], mc.shape[1])                      # (S, B)
        aleatoric = sample_entropy.mean(axis=0)                  # (B,)
        epistemic = np.clip(total - aleatoric, 0.0, None)

        return UncertaintyScores(
            total=total,
            epistemic=epistemic,
            aleatoric=aleatoric,
            extras={"n_samples": int(mc.shape[0])},
        )


def _entropy(probs: np.ndarray, eps: float) -> np.ndarray:
    p = np.clip(probs, eps, 1.0)
    return -np.sum(p * np.log(p), axis=-1)
