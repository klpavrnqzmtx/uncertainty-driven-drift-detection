"""Binary Bayesian logistic regression with a Laplace posterior.

Numpy-only implementation. Maintains a rolling buffer of recent samples
and, after warm-up, re-fits the MAP by Newton / IRLS and computes the
Gauss-Newton Hessian at the MAP to obtain ``Σ = H^{-1}``. The posterior
``N(w*, Σ)`` is exposed via :meth:`posterior_samples` so uncertainty
estimators can compute MC decompositions.

Design notes
------------
* Features are appended with a bias column, so ``w ∈ R^{D+1}``.
* We regularize with a Gaussian prior ``N(0, τ^{-1} I)``, giving a ridge
  term ``τ I`` added to the Hessian (guarantees PD and acts as Laplace
  precision).
* Newton iterations stop at ``newton_tol`` or ``max_newton_iter``.
* If the fit fails (e.g. only one class in the buffer), the MAP and
  covariance are left unchanged so predictions remain valid.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from uncertainty_driven_drift.data.base import StreamBatch, StreamSpec
from uncertainty_driven_drift.models.base import Classifier, Prediction
from uncertainty_driven_drift.registry import register


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable via tanh identity.
    return 0.5 * (1.0 + np.tanh(0.5 * z))


@register("model", "bayes_logreg")
class BayesianLogisticRegression(Classifier):
    """Binary Bayesian logistic regression + Laplace posterior over weights."""

    def __init__(
        self,
        prior_precision: float = 1.0,
        window_size: int = 500,
        warmup: int = 5,
        refit_every: int = 1,
        max_newton_iter: int = 25,
        newton_tol: float = 1e-6,
        jitter: float = 1e-8,
    ) -> None:
        self.prior_precision = float(prior_precision)
        self.window_size = int(window_size)
        self.warmup = int(warmup)
        self.refit_every = max(1, int(refit_every))
        self.max_newton_iter = int(max_newton_iter)
        self.newton_tol = float(newton_tol)
        self.jitter = float(jitter)

        self._d: int = 0
        self._w: Optional[np.ndarray] = None       # (D+1,)
        self._cov: Optional[np.ndarray] = None     # (D+1, D+1)
        self._chol: Optional[np.ndarray] = None    # lower-triangular of cov
        self._buffer_X: List[np.ndarray] = []
        self._buffer_y: List[np.ndarray] = []
        self._n_observed: int = 0
        self._fitted: bool = False

    # ---- API ------------------------------------------------------------

    def setup(self, spec: StreamSpec) -> None:
        if spec.n_classes != 2:
            raise ValueError(
                f"bayes_logreg is binary; got n_classes={spec.n_classes}"
            )
        d_in = int(np.prod(spec.input_shape))
        self._d = d_in + 1  # + bias
        self._w = np.zeros(self._d, dtype=np.float64)
        self._cov = np.eye(self._d, dtype=np.float64) / self.prior_precision
        self._chol = np.linalg.cholesky(self._cov + self.jitter * np.eye(self._d))
        self._buffer_X.clear()
        self._buffer_y.clear()
        self._n_observed = 0
        self._fitted = False

    def predict(self, batch: StreamBatch) -> Prediction:
        assert self._w is not None
        X = self._features(batch.x)
        p1 = _sigmoid(X @ self._w)
        probs = np.stack([1.0 - p1, p1], axis=1)
        return Prediction(probs=probs, features=X)

    def observe(self, batch: StreamBatch, prediction: Prediction) -> None:
        X = prediction.features if prediction.features is not None else self._features(batch.x)
        y = batch.y.astype(np.float64, copy=False)
        self._buffer_X.append(X)
        self._buffer_y.append(y)
        self._trim_buffer()

        self._n_observed += 1
        if self._n_observed < self.warmup:
            return
        if (self._n_observed - self.warmup) % self.refit_every != 0:
            return
        self._fit_laplace()

    # ---- Helpers --------------------------------------------------------

    def _features(self, x: np.ndarray) -> np.ndarray:
        flat = np.asarray(x, dtype=np.float64).reshape(x.shape[0], -1)
        return np.hstack([flat, np.ones((flat.shape[0], 1), dtype=np.float64)])

    def _trim_buffer(self) -> None:
        total = sum(X.shape[0] for X in self._buffer_X)
        while total > self.window_size and len(self._buffer_X) > 1:
            total -= self._buffer_X[0].shape[0]
            self._buffer_X.pop(0)
            self._buffer_y.pop(0)

    def _fit_laplace(self) -> None:
        assert self._w is not None
        X = np.concatenate(self._buffer_X, axis=0)
        y = np.concatenate(self._buffer_y, axis=0)
        if np.unique(y).size < 2:
            # Degenerate buffer — keep current posterior.
            return
        tau = self.prior_precision
        I = np.eye(self._d)
        w = self._w.copy()

        def neg_log_posterior(ww: np.ndarray) -> float:
            zz = X @ ww
            # log(1 + exp(z)) stably:
            log1pexp = np.where(zz >= 0, zz + np.log1p(np.exp(-zz)),
                                np.log1p(np.exp(zz)))
            nll = float(np.sum(log1pexp - y * zz))
            prior = 0.5 * tau * float(ww @ ww)
            return nll + prior

        f_curr = neg_log_posterior(w)
        for _ in range(self.max_newton_iter):
            z = X @ w
            p = _sigmoid(z)
            grad = X.T @ (p - y) + tau * w
            S = p * (1.0 - p) + 1e-8
            H = (X.T * S) @ X + tau * I
            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                return

            # Backtracking (Armijo with β=0.5, c=1e-4).
            grad_dot_step = float(grad @ step)
            alpha = 1.0
            while alpha > 1e-8:
                w_try = w - alpha * step
                f_try = neg_log_posterior(w_try)
                if f_try <= f_curr - 1e-4 * alpha * grad_dot_step:
                    break
                alpha *= 0.5
            if alpha <= 1e-8:
                break  # line search failed — keep current w
            w_new = w - alpha * step
            if np.max(np.abs(alpha * step)) < self.newton_tol:
                w = w_new
                break
            w = w_new
            f_curr = f_try

        # Final Hessian at MAP → Laplace covariance.
        z = X @ w
        p = _sigmoid(z)
        S = p * (1.0 - p) + 1e-8
        H = (X.T * S) @ X + tau * I
        try:
            cov = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return
        cov = 0.5 * (cov + cov.T)  # symmetrize
        try:
            chol = np.linalg.cholesky(cov + self.jitter * I)
        except np.linalg.LinAlgError:
            # Clamp eigenvalues before Cholesky.
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.clip(eigvals, self.jitter, None)
            cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
            chol = np.linalg.cholesky(cov + self.jitter * I)

        self._w = w
        self._cov = cov
        self._chol = chol
        self._fitted = True

    # ---- Public introspection for uncertainty estimators ----------------

    @property
    def d(self) -> int:
        return self._d

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def map_weights(self) -> np.ndarray:
        assert self._w is not None
        return self._w

    def posterior_samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Return ``n`` weight samples from ``N(w*, Σ)``, shape ``(n, D+1)``."""
        assert self._w is not None and self._chol is not None
        eps = rng.standard_normal(size=(n, self._d))
        return self._w[None, :] + eps @ self._chol.T
