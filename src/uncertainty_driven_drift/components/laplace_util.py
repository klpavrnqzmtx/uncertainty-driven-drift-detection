"""Shared helpers for diagonal last-layer Laplace posteriors.

The failure this module exists to prevent
----------------------------------------
A diagonal GGN posterior sets ``sigma^2 = 1 / (H + prior_precision)``.  ``H`` grows
with the fit-set size and the feature scale, so a *fixed* ``prior_precision`` means
the posterior width silently depends on the backbone, the number of classes and
how many images you happened to fit on.  When ``sigma`` ends up comparable to the
head weights themselves, every sampled head is noise: predictions collapse to
near-uniform, accuracy drops to chance and epistemic MI saturates.

That is exactly what happened to the CIFAR-100 ViT arm — ``prior_precision=1.0``
gave ~96% error (chance is 99%) and 2.69 nats of "epistemic" uncertainty that was
pure sampling noise rather than a signal about the data.

:func:`calibrate_prior_precision` removes the hand-tuning: it picks the *widest*
posterior (smallest precision) that still preserves the MAP model's predictions to
within a tolerance.  Uncertainty stays as large as the data supports, but never so
large that it destroys the predictions it is supposed to describe.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# Ascending precision = tightening posterior. Spans the range from "far too wide"
# to "essentially the MAP point estimate".
DEFAULT_PRIOR_GRID: Sequence[float] = (1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7)


def sample_head_probs(feats: np.ndarray, W_map: np.ndarray, b_map: np.ndarray,
                      W_var: np.ndarray, b_var: np.ndarray, n_samples: int,
                      rng: np.random.Generator) -> np.ndarray:
    """(S, B, K) softmax stack from ``n_samples`` draws of the last-layer posterior."""
    W_std, b_std = np.sqrt(W_var), np.sqrt(b_var)
    out = []
    for _ in range(n_samples):
        W_s = W_map + rng.standard_normal(W_map.shape) * W_std
        b_s = b_map + rng.standard_normal(b_map.shape) * b_std
        logits = feats @ W_s.T + b_s[None, :]
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        out.append(e / e.sum(axis=1, keepdims=True))
    return np.stack(out, axis=0)


def calibrate_prior_precision(
    feats: np.ndarray,
    labels: np.ndarray,
    W_map: np.ndarray,
    b_map: np.ndarray,
    H_W: np.ndarray,
    H_b: np.ndarray,
    grid: Optional[Sequence[float]] = None,
    n_samples: int = 8,
    tol: float = 0.02,
    seed: int = 0,
    label: str = "laplace",
) -> float:
    """Smallest prior precision whose posterior keeps accuracy within ``tol`` of MAP.

    Walking the grid upward from the widest posterior, the first precision whose
    sampled-head accuracy is within ``tol`` (absolute) of the MAP head's accuracy
    wins.  Returning early keeps the posterior as wide as the data allows, which is
    what makes epistemic uncertainty informative; the tolerance is what stops it
    from turning the predictions into noise.
    """
    grid = list(grid or DEFAULT_PRIOR_GRID)
    rng = np.random.default_rng(seed)

    map_acc = float((np.argmax(feats @ W_map.T + b_map[None, :], axis=1) == labels).mean())
    chosen, chosen_acc = grid[-1], None
    for lam in grid:
        probs = sample_head_probs(feats, W_map, b_map,
                                  1.0 / (H_W + lam), 1.0 / (H_b + lam),
                                  n_samples, rng)
        acc = float((np.argmax(probs.mean(axis=0), axis=1) == labels).mean())
        if map_acc - acc <= tol:
            chosen, chosen_acc = lam, acc
            break
    if chosen_acc is None:      # nothing met the tolerance — take the tightest
        probs = sample_head_probs(feats, W_map, b_map,
                                  1.0 / (H_W + chosen), 1.0 / (H_b + chosen),
                                  n_samples, rng)
        chosen_acc = float((np.argmax(probs.mean(axis=0), axis=1) == labels).mean())
    print(f"  [{label}] prior_precision={chosen:g} selected "
          f"(MAP acc {map_acc:.3f} -> sampled {chosen_acc:.3f}, tol {tol})", flush=True)
    return float(chosen)
