"""Synthetic Gaussian-mixture streams with exact oracle posterior.

The stream is a sequence of batches drawn from a time-varying
class-conditional Gaussian mixture. Because parameters are known to the
generator, the oracle posterior ``p*(y|x) = π_k N(x; μ_k, Σ_k) / Σ_j ...``
is computed analytically per sample and attached to every
:class:`StreamBatch`. This is what enables exact TV-mismatch measurement
downstream.

Drift is described by a schedule of events ``(start, end, target_params)``.
``start == end`` yields abrupt drift; ``end > start`` yields gradual
drift via linear interpolation of priors / means / covariances.

Five presets are exposed through ``drift_mode``:

* ``virtual``  — translate both class means by a common vector; boundary
  geometry is rigidly shifted (P(X) changes, decision surface moves with it).
* ``real``     — rotate class means around their centroid; boundary
  geometry changes relative to the input axes (P(Y|X) changes as a
  function of x).
* ``mixed``    — one virtual event, then one real event, at separate times.
* ``abrupt``   — synonym for an abrupt real event.
* ``gradual``  — real event interpolated over ``gradual_span`` batches.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from uncertainty_driven_drift.data.base import DatasetStream, StreamBatch, StreamSpec
from uncertainty_driven_drift.registry import register


@dataclass
class GMMParams:
    """Parameters of a class-conditional Gaussian mixture."""

    priors: np.ndarray   # (K,)
    means: np.ndarray    # (K, D)
    covs: np.ndarray     # (K, D, D)

    def __post_init__(self) -> None:
        self.priors = np.asarray(self.priors, dtype=np.float64)
        self.means = np.asarray(self.means, dtype=np.float64)
        self.covs = np.asarray(self.covs, dtype=np.float64)
        if self.priors.ndim != 1:
            raise ValueError("priors must be 1-D")
        if not np.isclose(self.priors.sum(), 1.0):
            raise ValueError("priors must sum to 1")
        K = self.priors.shape[0]
        if self.means.shape[0] != K or self.covs.shape[0] != K:
            raise ValueError("priors / means / covs must share K")

    @property
    def K(self) -> int:
        return self.priors.shape[0]

    @property
    def D(self) -> int:
        return self.means.shape[1]

    def sample(self, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """Draw ``n`` samples from the mixture. Returns ``(x, y)``."""
        y = rng.choice(self.K, size=n, p=self.priors)
        x = np.empty((n, self.D), dtype=np.float64)
        for k in range(self.K):
            mask = y == k
            nk = int(mask.sum())
            if nk:
                x[mask] = rng.multivariate_normal(self.means[k], self.covs[k], size=nk)
        return x.astype(np.float32), y.astype(np.int64)

    def posterior(self, x: np.ndarray) -> np.ndarray:
        """Return the oracle posterior matrix ``p*(y|x)`` with shape ``(n, K)``."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.D:
            raise ValueError(f"x must be (n, {self.D}), got {x.shape}")
        log_joint = np.empty((x.shape[0], self.K), dtype=np.float64)
        for k in range(self.K):
            log_joint[:, k] = (
                np.log(self.priors[k] + 1e-300)
                + _log_mvn_density(x, self.means[k], self.covs[k])
            )
        m = log_joint.max(axis=1, keepdims=True)
        exp = np.exp(log_joint - m)
        return exp / exp.sum(axis=1, keepdims=True)


def _log_mvn_density(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Log density of N(mean, cov) at each row of x."""
    D = x.shape[1]
    diff = x - mean
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise np.linalg.LinAlgError("Covariance not positive definite")
    cov_inv = np.linalg.inv(cov)
    quad = np.einsum("ni,ij,nj->n", diff, cov_inv, diff)
    return -0.5 * (D * np.log(2.0 * np.pi) + logdet + quad)


def _interpolate(a: GMMParams, b: GMMParams, alpha: float) -> GMMParams:
    """Linear interpolation of priors / means / covs."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    priors = (1.0 - alpha) * a.priors + alpha * b.priors
    priors = priors / priors.sum()  # numerical clean-up
    return GMMParams(
        priors=priors,
        means=(1.0 - alpha) * a.means + alpha * b.means,
        covs=(1.0 - alpha) * a.covs + alpha * b.covs,
    )


@dataclass
class DriftEvent:
    start: int
    end: int                 # end == start → abrupt
    target: GMMParams


def _params_trajectory(
    initial: GMMParams, events: Sequence[DriftEvent], n_batches: int,
) -> List[GMMParams]:
    """Materialize params at each batch index, given ordered non-overlapping events."""
    sorted_events = sorted(events, key=lambda e: e.start)
    traj: List[GMMParams] = []
    base = initial
    finished_idx = 0  # number of completed events whose target is now the base

    for t in range(n_batches):
        # Advance base through any events that have fully completed by now.
        while (
            finished_idx < len(sorted_events)
            and t >= sorted_events[finished_idx].end
            and t >= sorted_events[finished_idx].start
        ):
            base = sorted_events[finished_idx].target
            finished_idx += 1

        active: Optional[DriftEvent] = None
        if finished_idx < len(sorted_events):
            nxt = sorted_events[finished_idx]
            if nxt.start <= t < max(nxt.end, nxt.start + 1):
                active = nxt

        if active is None or active.end == active.start:
            traj.append(base)
        else:
            span = active.end - active.start
            alpha = (t - active.start) / span
            traj.append(_interpolate(base, active.target, alpha))

    return traj


# ---------------------------------------------------------------------------
# Drift presets
# ---------------------------------------------------------------------------

def _base_gmm() -> GMMParams:
    return GMMParams(
        priors=np.array([0.5, 0.5]),
        means=np.array([[-1.5, 0.0], [1.5, 0.0]]),
        covs=np.stack([np.eye(2), np.eye(2)]),
    )


def _rotate_means(params: GMMParams, angle_deg: float) -> GMMParams:
    angle = np.deg2rad(angle_deg)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    centroid = params.means.mean(axis=0)
    new_means = (params.means - centroid) @ R.T + centroid
    return replace(params, means=new_means)


def _shift_means(params: GMMParams, shift: np.ndarray) -> GMMParams:
    return replace(params, means=params.means + shift)


def _preset(
    mode: str,
    n_batches: int,
    drift_fraction: float,
    gradual_span: int,
    shift: Optional[Sequence[float]],
    rotation_deg: float,
) -> Tuple[GMMParams, List[DriftEvent]]:
    base = _base_gmm()
    t_mid = int(n_batches * drift_fraction)
    v = np.array([3.0, 2.0]) if shift is None else np.asarray(shift, dtype=np.float64)

    if mode == "virtual":
        return base, [DriftEvent(t_mid, t_mid, _shift_means(base, v))]
    if mode == "real":
        return base, [DriftEvent(t_mid, t_mid, _rotate_means(base, rotation_deg))]
    if mode == "mixed":
        t1 = int(n_batches * 0.30)
        t2 = int(n_batches * 0.70)
        virt = _shift_means(base, v)
        real = _rotate_means(virt, rotation_deg)
        return base, [DriftEvent(t1, t1, virt), DriftEvent(t2, t2, real)]
    if mode == "abrupt":
        return base, [DriftEvent(t_mid, t_mid, _rotate_means(base, rotation_deg))]
    if mode == "gradual":
        return base, [DriftEvent(t_mid, t_mid + gradual_span,
                                 _rotate_means(base, rotation_deg))]
    raise ValueError(
        f"Unknown drift_mode={mode!r}; "
        "expected one of virtual/real/mixed/abrupt/gradual"
    )


# ---------------------------------------------------------------------------
# Stream
# ---------------------------------------------------------------------------

@register("dataset", "gaussian_mixture")
class GaussianMixtureDriftStream(DatasetStream):
    """Class-conditional Gaussian mixture with a drift schedule.

    Parameters
    ----------
    drift_mode :
        One of ``virtual``, ``real``, ``mixed``, ``abrupt``, ``gradual``.
    n_batches, batch_size :
        Total number of batches and batch size.
    seed :
        Stream sampling seed; fully determines (x, y) along with the schedule.
    drift_fraction :
        Position of the main drift event as a fraction of ``n_batches``
        (ignored by ``mixed``, which hardcodes 0.30 and 0.70).
    gradual_span :
        Width, in batches, of the interpolation window used by ``gradual``.
    shift :
        Optional 2-vector used as the common mean translation in ``virtual``
        and ``mixed``. Defaults to ``(3, 2)``.
    rotation_deg :
        Mean rotation angle for ``real``, ``abrupt``, ``gradual``, and the
        second event of ``mixed``. Defaults to 90°.
    """

    def __init__(
        self,
        drift_mode: str = "virtual",
        n_batches: int = 100,
        batch_size: int = 100,
        seed: int = 0,
        drift_fraction: float = 0.5,
        gradual_span: int = 20,
        shift: Optional[Sequence[float]] = None,
        rotation_deg: float = 90.0,
    ) -> None:
        self.drift_mode = drift_mode
        self.n_batches = int(n_batches)
        self.batch_size = int(batch_size)
        self.seed = int(seed)

        initial, events = _preset(
            drift_mode,
            n_batches=self.n_batches,
            drift_fraction=float(drift_fraction),
            gradual_span=int(gradual_span),
            shift=shift,
            rotation_deg=float(rotation_deg),
        )
        self._trajectory = _params_trajectory(initial, events, self.n_batches)
        self._drift_starts = tuple(e.start for e in sorted(events, key=lambda e: e.start))
        self._drift_ends = tuple(e.end for e in sorted(events, key=lambda e: e.start))

        self.spec = StreamSpec(
            name=f"gaussian_mixture:{drift_mode}",
            input_shape=(initial.D,),
            n_classes=initial.K,
            n_batches=self.n_batches,
            batch_size=self.batch_size,
            drift_indices=self._drift_starts,
            has_true_posterior=True,
            extras={
                "drift_mode": drift_mode,
                "drift_events": [
                    {"start": int(s), "end": int(e)}
                    for s, e in zip(self._drift_starts, self._drift_ends)
                ],
            },
        )

    def __iter__(self) -> Iterator[StreamBatch]:
        rng = np.random.default_rng(self.seed)
        drift_set = set(self._drift_starts)
        for t in range(self.n_batches):
            params = self._trajectory[t]
            x, y = params.sample(self.batch_size, rng)
            post = params.posterior(x.astype(np.float64)).astype(np.float32)
            concept_id = sum(1 for ds in self._drift_starts if t >= ds)
            yield StreamBatch(
                index=t,
                x=x,
                y=y,
                concept_id=concept_id,
                is_drift=(t in drift_set),
                true_posterior=post,
            )
