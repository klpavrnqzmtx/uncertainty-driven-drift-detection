"""Monte-Carlo uncertainty decomposition from a Laplace posterior.

Given a :class:`BayesianLogisticRegression` model that exposes a posterior
over weights, this estimator draws ``n_samples`` weight vectors, computes
per-sample class probabilities, and returns the standard BNN
decomposition::

    total_i     = H[ (1/S) Σ_s p_s(y|x_i) ]
    aleatoric_i = (1/S) Σ_s H[ p_s(y|x_i) ]
    epistemic_i = total_i - aleatoric_i           (≥ 0 by Jensen's inequality)

Before the underlying model has been fit for the first time, the
estimator falls back to predictive entropy from the prior-mean weights
and reports ``aleatoric = 0`` so the decomposition is still additive.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from uncertainty_driven_drift.components.bayes_logreg import _sigmoid
from uncertainty_driven_drift.data.base import StreamBatch
from uncertainty_driven_drift.models.base import Classifier, Prediction
from uncertainty_driven_drift.registry import register
from uncertainty_driven_drift.uncertainty.base import UncertaintyEstimator, UncertaintyScores


@register("uncertainty", "laplace_mc")
class LaplaceMCUncertainty(UncertaintyEstimator):
    def __init__(self, n_samples: int = 30, seed: int = 0) -> None:
        self.n_samples = int(n_samples)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._model: Classifier | None = None

    def setup(self, model: Classifier) -> None:
        # Duck-typed so wrappers that expose the BLR surface (feature
        # extractor + Laplace head, e.g. lenet_laplace) work without
        # inheritance.
        required = ("is_fitted", "posterior_samples", "map_weights", "d")
        missing = [attr for attr in required if not hasattr(model, attr)]
        if missing:
            raise TypeError(
                "laplace_mc requires a model exposing the bayes_logreg "
                f"surface (Laplace posterior over a linear head); missing "
                f"attributes: {missing} on {type(model).__name__}"
            )
        self._model = model
        self._rng = np.random.default_rng(self.seed)

    def fit(self, buffer: Iterable[StreamBatch]) -> None:
        # The model fits itself in ``observe``; nothing to do here.
        del buffer

    def score(self, batch: StreamBatch, prediction: Prediction) -> UncertaintyScores:
        assert self._model is not None
        X = prediction.features
        if X is None:
            raise ValueError("bayes_logreg must populate prediction.features")

        if not self._model.is_fitted:
            probs = prediction.probs
            ent = _entropy(probs)
            B = probs.shape[0]
            return UncertaintyScores(
                total=ent,
                epistemic=ent,
                aleatoric=np.zeros(B, dtype=np.float64),
                extras={"fitted": False},
            )

        samples = self._model.posterior_samples(self.n_samples, self._rng)  # (S, D+1)
        logits = X @ samples.T                                              # (B, S)
        p1 = _sigmoid(logits)
        probs_per_sample = np.stack([1.0 - p1, p1], axis=-1)                # (B, S, 2)

        mean_probs = probs_per_sample.mean(axis=1)                          # (B, 2)
        total = _entropy(mean_probs)
        sample_ent = _entropy(probs_per_sample.reshape(-1, 2)).reshape(
            probs_per_sample.shape[:2]
        )                                                                    # (B, S)
        aleatoric = sample_ent.mean(axis=1)
        epistemic = np.clip(total - aleatoric, 0.0, None)

        return UncertaintyScores(
            total=total,
            epistemic=epistemic,
            aleatoric=aleatoric,
            extras={"fitted": True, "n_samples": self.n_samples},
        )


def _entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise Shannon entropy, in nats."""
    p = np.clip(probs, eps, 1.0)
    return -np.sum(p * np.log(p), axis=-1)
