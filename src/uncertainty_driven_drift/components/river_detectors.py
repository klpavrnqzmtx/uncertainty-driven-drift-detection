"""River-backed drift detector adapters.

Wraps ``river.drift`` algorithms behind the framework's
:class:`DriftDetector` protocol. All adapters consume a single
per-step signal drawn from one of the channels the runner produces:

* ``mean_total``        — mean predictive entropy across the batch
  (default unsupervised channel; a model-aware signal in nats).
* ``mean_epistemic``    — mean MI across the batch (kept for
  compatibility with existing configs).
* ``mean_max_softmax``  — mean top-class softmax probability (kept
  for back-compat; new configs should avoid it).
* ``mean_input``        — mean pixel / feature intensity of the
  raw input batch (``batch.x.mean()``), i.e. a *model-free* summary
  that lets us compare drift detection in uncertainty space to
  drift detection directly on the input distribution using the same
  algorithms.
* ``error``             — batch error rate (needs ``batch.y``).

Two update modes are used:

* **per_batch_scalar** (``PageHinkley``, ``ADWIN``, ``KSWIN``) —
  the scalar batch signal is sent to ``update`` once per step.
* **per_sample_binary** (``DDM``, ``EDDM``) — river's supervised
  detectors operate on Bernoulli correctness signals, so we iterate
  over the batch and push each per-sample 0/1 error.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional

import numpy as np

from uncertainty_driven_drift.data.base import StreamBatch
from uncertainty_driven_drift.detectors.base import DetectorEvent, DriftDetector
from uncertainty_driven_drift.models.base import Prediction
from uncertainty_driven_drift.registry import register
from uncertainty_driven_drift.uncertainty.base import UncertaintyScores


# ---------------------------------------------------------------------------
# Signal extractors
# ---------------------------------------------------------------------------

def _batch_signal(
    channel: str,
    batch: StreamBatch,
    prediction: Prediction,
    scores: UncertaintyScores,
) -> float:
    if channel == "mean_epistemic":
        return float(np.mean(scores.epistemic))
    if channel == "mean_total":
        return float(np.mean(scores.total))
    if channel == "mean_max_softmax":
        return float(prediction.probs.max(axis=1).mean())
    if channel == "mean_input":
        # L2 norm of (per-sample mean, per-sample std) averaged over the
        # batch. Reacts to both shifts in input level (mean) and spread
        # (std/noise), works for tabular (B, D) and image (B, C, H, W).
        x = np.asarray(batch.x, dtype=np.float64).reshape(len(batch.x), -1)
        per_sample_mean = x.mean(axis=1)
        per_sample_std = x.std(axis=1)
        return float(np.sqrt(per_sample_mean ** 2 + per_sample_std ** 2).mean())
    if channel == "error":
        preds = prediction.probs.argmax(axis=1)
        return float((preds != batch.y).mean())
    raise ValueError(
        f"Unknown signal channel {channel!r}; expected one of "
        "{mean_total, mean_epistemic, mean_max_softmax, mean_input, error}"
    )


def _per_sample_errors(batch: StreamBatch, prediction: Prediction) -> np.ndarray:
    preds = prediction.probs.argmax(axis=1)
    return (preds != batch.y).astype(np.int64)


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------

class _RiverDetector(DriftDetector):
    """Shared scaffolding for all river adapters.

    Subclasses supply ``_make_detector`` and ``update_mode``.
    """

    update_mode: str = "per_batch_scalar"
    default_signal: str = "mean_total"

    def __init__(
        self,
        signal: Optional[str] = None,
        name: Optional[str] = None,
        **river_kwargs: Any,
    ) -> None:
        self.name = name or self._default_name()
        self.signal = signal or self.default_signal
        self.river_kwargs = dict(river_kwargs)
        self._detector = self._make_detector()

    def _default_name(self) -> str:
        return type(self).__name__.lower()

    def _make_detector(self):
        raise NotImplementedError

    def update(
        self,
        batch: StreamBatch,
        prediction: Prediction,
        scores: UncertaintyScores,
    ) -> DetectorEvent:
        if self.update_mode == "per_batch_scalar":
            value = _batch_signal(self.signal, batch, prediction, scores)
            self._detector.update(value)
            return DetectorEvent(
                alarm=bool(self._detector.drift_detected),
                statistic=value,
                diagnostics=self._diagnostics(),
            )
        elif self.update_mode == "per_sample_binary":
            errors = _per_sample_errors(batch, prediction)
            alarm = False
            for e in errors:
                self._detector.update(int(e))
                if bool(self._detector.drift_detected):
                    alarm = True
            err_rate = float(errors.mean())
            return DetectorEvent(
                alarm=alarm,
                statistic=err_rate,
                diagnostics={**self._diagnostics(), "error": err_rate},
            )
        raise ValueError(f"Unknown update_mode={self.update_mode!r}")

    def _diagnostics(self) -> Dict[str, Any]:
        return {}

    def reset(self) -> None:
        self._detector = self._make_detector()


# ---------------------------------------------------------------------------
# Concrete adapters
# ---------------------------------------------------------------------------

@register("detector", "page_hinkley")
class PageHinkleyDetector(_RiverDetector):
    """Unsupervised mean-shift detector on a scalar signal."""

    update_mode = "per_batch_scalar"

    def _default_name(self) -> str:
        return "page_hinkley"

    def _make_detector(self):
        from river.drift import PageHinkley

        return PageHinkley(**self.river_kwargs)


@register("detector", "adwin")
class ADWINDetector(_RiverDetector):
    """Adaptive windowing change detector — updated per sample for sensitivity.

    ADWIN receives one update *per sample* in the batch rather than one per
    batch. For epistemic/total signals this means each individual MI or
    entropy value; for the input signal each sample's L2(mean, std) scalar.
    This gives ADWIN ~batch_size× more data points per step, which is
    required for its adaptive window to accumulate enough evidence to detect
    real drift on streams of only ~100–200 batches.
    """

    update_mode = "per_batch_scalar"  # kept so parent machinery initialises

    def _default_name(self) -> str:
        return "adwin"

    def _make_detector(self):
        from river.drift import ADWIN

        return ADWIN(**self.river_kwargs)

    def _diagnostics(self) -> Dict[str, Any]:
        return {"width": int(getattr(self._detector, "width", 0))}

    def update(
        self,
        batch: StreamBatch,
        prediction: Prediction,
        scores: UncertaintyScores,
    ) -> DetectorEvent:
        """Per-sample ADWIN update for better sensitivity on short streams."""
        if self.signal == "mean_epistemic":
            values = np.asarray(scores.epistemic, dtype=np.float64).ravel()
        elif self.signal == "mean_total":
            values = np.asarray(scores.total, dtype=np.float64).ravel()
        elif self.signal == "mean_input":
            x = np.asarray(batch.x, dtype=np.float64).reshape(len(batch.x), -1)
            per_mean = x.mean(axis=1)
            per_std = x.std(axis=1)
            values = np.sqrt(per_mean ** 2 + per_std ** 2)
        elif self.signal == "error":
            preds = prediction.probs.argmax(axis=1)
            values = (preds != batch.y).astype(np.float64)
        else:
            v = _batch_signal(self.signal, batch, prediction, scores)
            values = np.array([v])

        alarm = False
        for v in values:
            self._detector.update(float(v))
            if self._detector.drift_detected:
                alarm = True

        return DetectorEvent(
            alarm=alarm,
            statistic=float(values.mean()),
            diagnostics=self._diagnostics(),
        )


@register("detector", "kswin")
class KSWINDetector(_RiverDetector):
    """Kolmogorov-Smirnov change detector on a scalar signal."""

    update_mode = "per_batch_scalar"

    def _default_name(self) -> str:
        return "kswin"

    def _make_detector(self):
        from river.drift import KSWIN

        return KSWIN(**self.river_kwargs)

    def _diagnostics(self) -> Dict[str, Any]:
        return {"p_value": float(getattr(self._detector, "p_value", 1.0))}


@register("detector", "chi_square")
class ChiSquareDetector(DriftDetector):
    """Sliding-window χ² goodness-of-fit drift detector.

    At every step we push the scalar ``signal`` value into a ring buffer
    of size ``2 * stat_size``. Once full, we split the buffer into:

    * **reference** — the older ``stat_size`` values,
    * **current**   — the most recent ``stat_size`` values.

    Both windows are binned into ``n_bins`` equal-width bins over the
    pooled ``[min, max]`` range, the reference histogram is rescaled to
    match the current total, and a Pearson χ² statistic is computed
    against the reference as the expected distribution. An alarm fires
    when the tail probability is below ``alpha`` (``df = n_bins - 1``).
    After an alarm we clear the buffer so the next test builds a fresh
    reference from the new regime.

    Unlike Page-Hinkley (mean shift) and KSWIN (CDF shift), χ² reacts
    to changes in *bin occupancy*, so it picks up shape changes —
    bimodality, variance swings, tail inflation — even when the mean
    barely moves. Useful as a complementary channel for uncertainty
    signals that don't drift monotonically (e.g. entropy distributions
    that widen before their mean moves).

    Parameters
    ----------
    signal:
        Channel name (see :func:`_batch_signal`). Defaults to
        ``mean_total``.
    name:
        Optional identifier for logging / plotting.
    stat_size:
        Size of each of the two sliding windows.
    n_bins:
        Number of equal-width histogram bins.
    alpha:
        Tail probability threshold; lower = less sensitive.
    min_expected:
        Floor for expected bin counts to keep χ² numerically stable
        when a bin is near-empty in the reference window.
    """

    update_mode: str = "per_batch_scalar"

    def __init__(
        self,
        signal: str = "mean_total",
        name: Optional[str] = None,
        stat_size: int = 16,
        n_bins: int = 4,
        alpha: float = 0.005,
        min_expected: float = 1.0,
    ) -> None:
        if stat_size < 2:
            raise ValueError("stat_size must be >= 2")
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.name = name or "chi_square"
        self.signal = signal
        self.stat_size = int(stat_size)
        self.n_bins = int(n_bins)
        self.alpha = float(alpha)
        self.min_expected = float(min_expected)
        self._buf: Deque[float] = deque(maxlen=2 * self.stat_size)
        self._last_pvalue: float = 1.0
        self._last_chi2: float = 0.0

    def update(
        self,
        batch: StreamBatch,
        prediction: Prediction,
        scores: UncertaintyScores,
    ) -> DetectorEvent:
        value = _batch_signal(self.signal, batch, prediction, scores)
        self._buf.append(value)

        alarm = False
        if len(self._buf) == self._buf.maxlen:
            arr = np.fromiter(self._buf, dtype=np.float64, count=len(self._buf))
            ref = arr[: self.stat_size]
            cur = arr[self.stat_size :]
            lo, hi = float(arr.min()), float(arr.max())
            if hi > lo:
                edges = np.linspace(lo, hi, self.n_bins + 1)
                ref_h, _ = np.histogram(ref, bins=edges)
                cur_h, _ = np.histogram(cur, bins=edges)
                # Rescale reference to match current window total so the
                # comparison is on identical sample sizes.
                ref_total = ref_h.sum()
                cur_total = cur_h.sum()
                if ref_total > 0 and cur_total > 0:
                    expected = ref_h.astype(np.float64) * (cur_total / ref_total)
                    expected = np.maximum(expected, self.min_expected)
                    chi2_stat = float(np.sum((cur_h - expected) ** 2 / expected))
                    df = self.n_bins - 1
                    # Survival function of χ²; use scipy if available,
                    # otherwise fall back to a Wilson-Hilferty approx so
                    # the component stays light on dependencies.
                    try:
                        from scipy.stats import chi2 as _chi2  # type: ignore

                        p_value = float(_chi2.sf(chi2_stat, df))
                    except Exception:
                        # Wilson-Hilferty: ((X/df)^(1/3) - (1 - 2/(9 df))) * sqrt(9 df / 2) ~ N(0,1)
                        z = ((chi2_stat / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) * np.sqrt(9.0 * df / 2.0)
                        # 1 - Phi(z) via erfc
                        from math import erfc, sqrt

                        p_value = 0.5 * erfc(z / sqrt(2.0))
                    self._last_chi2 = chi2_stat
                    self._last_pvalue = p_value
                    if p_value < self.alpha:
                        alarm = True
                        self._buf.clear()

        return DetectorEvent(
            alarm=alarm,
            statistic=value,
            diagnostics={"chi2": self._last_chi2, "p_value": self._last_pvalue},
        )

    def reset(self) -> None:
        self._buf.clear()
        self._last_pvalue = 1.0
        self._last_chi2 = 0.0


@register("detector", "ddm")
class DDMDetector(_RiverDetector):
    """Supervised DDM baseline: per-sample correctness."""

    update_mode = "per_sample_binary"
    default_signal = "error"

    def _default_name(self) -> str:
        return "ddm"

    def _make_detector(self):
        from river.drift.binary import DDM

        return DDM(**self.river_kwargs)

    def _diagnostics(self) -> Dict[str, Any]:
        return {"warning": bool(getattr(self._detector, "warning_detected", False))}


@register("detector", "eddm")
class EDDMDetector(_RiverDetector):
    """Supervised EDDM baseline: per-sample correctness, gradual-friendly."""

    update_mode = "per_sample_binary"
    default_signal = "error"

    def _default_name(self) -> str:
        return "eddm"

    def _make_detector(self):
        from river.drift.binary import EDDM

        return EDDM(**self.river_kwargs)

    def _diagnostics(self) -> Dict[str, Any]:
        return {"warning": bool(getattr(self._detector, "warning_detected", False))}
