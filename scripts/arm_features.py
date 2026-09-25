"""Capture an arm's penultimate features without knowing how it preprocesses.

Every model family preprocesses differently: CIFAR normalises with dataset
channel statistics, MNIST with its own single-channel ones (and a different
function signature), the pretrained ViT resizes to 224 and normalises with the
checkpoint's own mean/std, audio builds log-mel spectrograms. Reimplementing
that per family is how both baseline_driftlens.py and laplace_diagnostics.py
broke on arms other than the one they were tested against.

So don't reimplement it. Run the arm's OWN predict(), which already does its
preprocessing, and intercept embed() -- the one call every family makes on its
already-prepared batch. Also times those calls, which gives the deterministic
backbone cost that the sampling predict is measured against.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

_EMBED_NAMES = ("embed", "embed_deterministic")


class _Capture:
    def __init__(self):
        self.feats: list = []
        self.seconds: float = 0.0

    @property
    def last(self):
        return self.feats[-1] if self.feats else None


@contextmanager
def capture_features(model):
    """Context manager yielding a _Capture that fills as predict() is called.

        with capture_features(model) as cap:
            model.predict(batch)
        feats = cap.last            # (B, D) float array, or None

    Wraps both embed() and embed_deterministic(): MC-Dropout arms on the
    pretrained ViT use the latter to run the backbone once and resample only
    the head, while the Laplace arms use the former.
    """
    inner = getattr(model, "_model", None)
    cap = _Capture()
    if inner is None:
        yield cap
        return

    patched = []
    for name in _EMBED_NAMES:
        fn = getattr(inner, name, None)
        if fn is None:
            continue

        def _make(f):
            def wrapped(x):
                t0 = time.perf_counter()
                out = f(x)
                cap.seconds += time.perf_counter() - t0
                try:
                    cap.feats.append(out.detach().cpu().numpy())
                except Exception:
                    pass
                return out
            return wrapped

        setattr(inner, name, _make(fn))
        patched.append(name)
    try:
        yield cap
    finally:
        # Drop the instance-level shadow so the class method is visible again.
        for name in patched:
            inner.__dict__.pop(name, None)
