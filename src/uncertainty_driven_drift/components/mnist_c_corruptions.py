"""In-process image corruptions modelled on MNIST-C.

MNIST-C (Mu & Gilmer, 2019) distributes a 650 MB zip of pre-computed
corrupted MNIST arrays. Downloading that on every machine is awkward,
so we implement a faithful subset of the corruptions directly in
numpy/scipy. Each corruption takes a batch of MNIST images in the
canonical ``(B, 1, 28, 28)`` float layout on ``[0, 1]`` and returns
another batch of the same shape, clipped to ``[0, 1]``.

Each corruption is parameterised by an integer ``severity`` on the
MNIST-C 1–5 scale, so dropping in the real MNIST-C arrays later (e.g.
for a reproducibility check) would not require any other changes
downstream.

The set below is chosen to cover the main failure modes relevant to
drift detection without pulling in extra graphics code:

* additive noise (gaussian, impulse/salt–pepper)
* blur (motion)
* geometric shift (rotate, translate)
* photometric shift (brightness, fog)

``identity`` is the clean baseline — useful as the first phase in a
stream so the pretrained backbone starts from a low-error regime.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np


CorruptionFn = Callable[[np.ndarray, int, np.random.Generator], np.ndarray]


def _check_severity(severity: int) -> int:
    s = int(severity)
    if not 1 <= s <= 5:
        raise ValueError(f"severity must be in 1..5, got {severity!r}")
    return s


def _check_image_batch(x: np.ndarray) -> None:
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError(
            f"expected (B, 1, H, W) images; got shape {x.shape}"
        )


# ---------------------------------------------------------------------------
# Individual corruptions
# ---------------------------------------------------------------------------

def identity(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Clean pass-through (severity is accepted for API uniformity)."""
    _check_image_batch(x)
    _check_severity(severity)
    del rng
    return x.astype(np.float32, copy=False)


def gaussian_noise(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Add i.i.d. N(0, σ) noise per pixel; σ grows with severity."""
    _check_image_batch(x)
    sigma = [0.08, 0.12, 0.18, 0.26, 0.38][_check_severity(severity) - 1]
    noise = rng.normal(0.0, sigma, size=x.shape).astype(np.float32)
    return np.clip(x + noise, 0.0, 1.0)


def impulse_noise(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Salt-and-pepper noise: flip each pixel to 0 or 1 with probability p."""
    _check_image_batch(x)
    p = [0.03, 0.06, 0.10, 0.17, 0.27][_check_severity(severity) - 1]
    out = x.astype(np.float32, copy=True)
    u = rng.random(size=x.shape)
    out[u < p / 2.0] = 0.0
    out[u > 1.0 - p / 2.0] = 1.0
    return out


def motion_blur(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Convolve each image with a randomly-angled line kernel."""
    from scipy.signal import convolve2d  # local import keeps scipy optional

    _check_image_batch(x)
    ksize = [3, 5, 7, 9, 11][_check_severity(severity) - 1]
    out = np.empty_like(x, dtype=np.float32)
    for i in range(x.shape[0]):
        angle = float(rng.uniform(-45.0, 45.0))
        kernel = _motion_kernel(ksize, angle)
        out[i, 0] = convolve2d(
            x[i, 0].astype(np.float32), kernel, mode="same", boundary="symm",
        )
    return np.clip(out, 0.0, 1.0)


def rotate(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Rotate each image by a uniform angle in ``[-max, +max]`` degrees."""
    from scipy.ndimage import rotate as ndrotate

    _check_image_batch(x)
    max_angle = [5.0, 10.0, 15.0, 22.0, 30.0][_check_severity(severity) - 1]
    angles = rng.uniform(-max_angle, max_angle, size=x.shape[0])
    out = np.empty_like(x, dtype=np.float32)
    for i in range(x.shape[0]):
        out[i, 0] = ndrotate(
            x[i, 0].astype(np.float32), float(angles[i]),
            reshape=False, order=1, mode="constant", cval=0.0,
        )
    return np.clip(out, 0.0, 1.0)


def translate(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Integer-pixel translation with zero-padding on the receding edge."""
    _check_image_batch(x)
    max_shift = [1, 2, 3, 4, 6][_check_severity(severity) - 1]
    shifts = rng.integers(-max_shift, max_shift + 1, size=(x.shape[0], 2))
    out = np.zeros_like(x, dtype=np.float32)
    h, w = x.shape[2], x.shape[3]
    for i in range(x.shape[0]):
        dy, dx = int(shifts[i, 0]), int(shifts[i, 1])
        src_y0 = max(0, -dy); src_y1 = min(h, h - dy)
        src_x0 = max(0, -dx); src_x1 = min(w, w - dx)
        dst_y0 = max(0, dy);  dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x0 = max(0, dx);  dst_x1 = dst_x0 + (src_x1 - src_x0)
        out[i, 0, dst_y0:dst_y1, dst_x0:dst_x1] = \
            x[i, 0, src_y0:src_y1, src_x0:src_x1]
    return out


def brightness(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Add a constant to every pixel (saturating at 1)."""
    _check_image_batch(x)
    delta = [0.10, 0.20, 0.30, 0.40, 0.50][_check_severity(severity) - 1]
    del rng
    return np.clip(x.astype(np.float32) + delta, 0.0, 1.0)


def fog(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Overlay a smooth low-frequency 'fog' field. Alpha grows with severity.

    The field is white Gaussian noise smoothed with a large Gaussian
    kernel and then re-normalised to ``[0, 1]``. The output image is
    ``(1 - α) x + α fog``.
    """
    from scipy.ndimage import gaussian_filter

    _check_image_batch(x)
    alpha = [0.15, 0.25, 0.35, 0.45, 0.55][_check_severity(severity) - 1]
    out = np.empty_like(x, dtype=np.float32)
    for i in range(x.shape[0]):
        field = rng.random(size=x.shape[2:]).astype(np.float32)
        field = gaussian_filter(field, sigma=4.0)
        lo, hi = float(field.min()), float(field.max())
        if hi > lo:
            field = (field - lo) / (hi - lo)
        out[i, 0] = (1.0 - alpha) * x[i, 0] + alpha * field
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_CORRUPTIONS: Dict[str, CorruptionFn] = {
    "identity": identity,
    "gaussian_noise": gaussian_noise,
    "impulse_noise": impulse_noise,
    "motion_blur": motion_blur,
    "rotate": rotate,
    "translate": translate,
    "brightness": brightness,
    "fog": fog,
}


def register_corruption(name: str, fn: CorruptionFn) -> None:
    if name in _CORRUPTIONS:
        raise ValueError(f"Corruption {name!r} already registered")
    _CORRUPTIONS[name] = fn


def available_corruptions() -> List[str]:
    return sorted(_CORRUPTIONS)


def apply_corruption(
    name: str,
    x: np.ndarray,
    severity: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Dispatch by name with validation."""
    if name not in _CORRUPTIONS:
        raise KeyError(
            f"Unknown corruption {name!r}; have {available_corruptions()}"
        )
    return _CORRUPTIONS[name](x, severity, rng)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _motion_kernel(ksize: int, angle_deg: float) -> np.ndarray:
    """Unit-mass line kernel of length ``ksize`` at ``angle_deg`` degrees.

    The kernel is constructed by tracing a line through the centre of a
    ``ksize × ksize`` square and accumulating mass onto the nearest
    integer cells along the path. The result is normalised to sum to 1.
    """
    if ksize < 1 or ksize % 2 == 0:
        raise ValueError(f"ksize must be a positive odd integer; got {ksize}")
    k = np.zeros((ksize, ksize), dtype=np.float32)
    c = (ksize - 1) / 2.0
    theta = np.deg2rad(angle_deg)
    steps = max(ksize * 4, 16)
    for s in np.linspace(-c, c, steps):
        yy = int(round(c + s * np.sin(theta)))
        xx = int(round(c + s * np.cos(theta)))
        if 0 <= yy < ksize and 0 <= xx < ksize:
            k[yy, xx] += 1.0
    total = float(k.sum())
    return k / total if total > 0 else np.ones_like(k) / (ksize * ksize)
