"""CIFAR-10-C–style image corruptions for RGB 32×32 images.

Implements the 15 corruption types from Hendrycks & Dietterich (2019)
"Benchmarking Neural Network Robustness to Common Corruptions and
Perturbations", adapted for on-the-fly generation from numpy so the
750 MB CIFAR-10-C download is not required.

Each function operates on batches in ``(B, 3, H, W)`` float32 ``[0, 1]``
layout and returns the same shape, clipped to ``[0, 1]``.  A ``severity``
argument on the 1–5 scale controls corruption strength.
"""

from __future__ import annotations

import io
from typing import Callable, Dict, List

import numpy as np


CorruptionFn = Callable[[np.ndarray, int, np.random.Generator], np.ndarray]


def _check_severity(severity: int) -> int:
    s = int(severity)
    if not 1 <= s <= 5:
        raise ValueError(f"severity must be in 1..5, got {severity!r}")
    return s


def _check_image_batch(x: np.ndarray) -> None:
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(f"expected (B, 3, H, W) images; got shape {x.shape}")


def _center_crop(arr: np.ndarray, H: int, W: int) -> np.ndarray:
    h, w = arr.shape
    top = (h - H) // 2
    left = (w - W) // 2
    return arr[top: top + H, left: left + W]


# ---------------------------------------------------------------------------
# 1. Identity
# ---------------------------------------------------------------------------

def identity(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    _check_image_batch(x)
    _check_severity(severity)
    return x.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# 2. Noise corruptions
# ---------------------------------------------------------------------------

def gaussian_noise(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    _check_image_batch(x)
    s = _check_severity(severity)
    sigma = [0.08, 0.12, 0.18, 0.26, 0.38][s - 1]
    return np.clip(x + rng.normal(0.0, sigma, x.shape).astype(np.float32), 0.0, 1.0).astype(np.float32)


def shot_noise(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    _check_image_batch(x)
    s = _check_severity(severity)
    rate = [500, 250, 100, 75, 50][s - 1]
    return np.clip(rng.poisson(x * rate).astype(np.float32) / rate, 0.0, 1.0).astype(np.float32)


def impulse_noise(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    _check_image_batch(x)
    s = _check_severity(severity)
    frac = [0.03, 0.06, 0.09, 0.17, 0.27][s - 1]
    out = x.copy()
    B, C, H, W = x.shape
    mask = rng.random((B, H, W)) < frac
    vals = rng.integers(0, 2, size=(B, H, W)).astype(np.float32)
    for c in range(C):
        out[:, c, :, :][mask] = vals[mask]
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Blur corruptions
# ---------------------------------------------------------------------------

def defocus_blur(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    _check_image_batch(x)
    s = _check_severity(severity)
    sigma = [1.0, 1.5, 2.0, 2.5, 3.0][s - 1]
    from scipy.ndimage import gaussian_filter
    B, C, H, W = x.shape
    out = np.empty_like(x)
    for b in range(B):
        for c in range(C):
            out[b, c] = gaussian_filter(x[b, c], sigma=sigma)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def glass_blur(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    _check_image_batch(x)
    s = _check_severity(severity)
    sigma, radius, iters = [(0.5, 1, 1), (0.7, 1, 2), (0.9, 2, 2), (1.1, 2, 3), (1.3, 3, 3)][s - 1]
    from scipy.ndimage import gaussian_filter
    B, C, H, W = x.shape
    out = np.empty_like(x)
    for b in range(B):
        img = np.stack([gaussian_filter(x[b, c], sigma=sigma) for c in range(C)])
        for _ in range(iters):
            dy = rng.integers(-radius, radius + 1, (H, W))
            dx = rng.integers(-radius, radius + 1, (H, W))
            yy = np.clip(np.arange(H)[:, None] + dy, 0, H - 1)
            xx = np.clip(np.arange(W)[None, :] + dx, 0, W - 1)
            img = img[:, yy, xx]
        out[b] = img
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _motion_kernel(size: int, angle_deg: float) -> np.ndarray:
    kernel = np.zeros((size, size), dtype=np.float32)
    mid = size // 2
    kernel[mid, :] = 1.0 / size
    from scipy.ndimage import rotate
    kernel = rotate(kernel, angle_deg, reshape=False)
    total = kernel.sum()
    return kernel / total if total > 0 else kernel


def motion_blur(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    _check_image_batch(x)
    s = _check_severity(severity)
    size = [5, 7, 9, 11, 13][s - 1]
    from scipy.ndimage import convolve
    B, C, H, W = x.shape
    out = np.empty_like(x)
    for b in range(B):
        angle = float(rng.uniform(0.0, 180.0))
        k = _motion_kernel(size, angle)
        for c in range(C):
            out[b, c] = convolve(x[b, c], k, mode="wrap")
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def zoom_blur(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    _check_image_batch(x)
    s = _check_severity(severity)
    max_zoom, n_levels = [(1.02, 3), (1.04, 5), (1.06, 7), (1.08, 9), (1.10, 11)][s - 1]
    from scipy.ndimage import zoom as spzoom
    B, C, H, W = x.shape
    out = np.zeros_like(x)
    factors = np.linspace(1.0, max_zoom, n_levels)
    for b in range(B):
        acc = np.zeros((C, H, W), dtype=np.float64)
        for f in factors:
            for c in range(C):
                z = spzoom(x[b, c], f, order=1)
                acc[c] += _center_crop(z, H, W)
        out[b] = (acc / len(factors)).astype(np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Weather corruptions
# ---------------------------------------------------------------------------

def snow(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    _check_image_batch(x)
    s = _check_severity(severity)
    frac, intensity = [(0.05, 0.50), (0.10, 0.55), (0.15, 0.60), (0.25, 0.65), (0.35, 0.70)][s - 1]
    B, C, H, W = x.shape
    out = x.copy()
    mask = rng.random((B, H, W)) < frac
    for c in range(C):
        ch = out[:, c, :, :]
        ch[mask] = np.maximum(ch[mask], intensity)
        out[:, c, :, :] = ch
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def frost(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    _check_image_batch(x)
    s = _check_severity(severity)
    alpha = [0.20, 0.30, 0.40, 0.50, 0.60][s - 1]
    from scipy.ndimage import gaussian_filter
    B, C, H, W = x.shape
    out = np.empty_like(x)
    for b in range(B):
        noise = rng.standard_normal((H, W)).astype(np.float32)
        noise = gaussian_filter(noise, sigma=3.0)
        lo, hi = noise.min(), noise.max()
        noise = (noise - lo) / (hi - lo + 1e-8)
        # Blue-cold tint: R=0.8, G=0.9, B=1.0
        frost_rgb = np.stack([noise * 0.80, noise * 0.90, noise * 1.00])
        out[b] = x[b] * (1.0 - alpha) + frost_rgb * alpha
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def fog(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    _check_image_batch(x)
    s = _check_severity(severity)
    fog_alpha = [0.30, 0.40, 0.50, 0.65, 0.75][s - 1]
    from scipy.ndimage import gaussian_filter
    B, C, H, W = x.shape
    out = np.empty_like(x)
    for b in range(B):
        depth = rng.standard_normal((H, W)).astype(np.float32)
        depth = gaussian_filter(depth, sigma=max(H // 6, 2))
        lo, hi = depth.min(), depth.max()
        depth = (depth - lo) / (hi - lo + 1e-8)
        alpha_map = fog_alpha * (0.5 + 0.5 * depth)  # (H, W)
        for c in range(C):
            out[b, c] = x[b, c] * (1.0 - alpha_map) + alpha_map
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 5. Digital corruptions
# ---------------------------------------------------------------------------

def brightness(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    _check_image_batch(x)
    s = _check_severity(severity)
    add = [0.05, 0.10, 0.15, 0.20, 0.25][s - 1]
    return np.clip(x + add, 0.0, 1.0).astype(np.float32)


def contrast(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    _check_image_batch(x)
    s = _check_severity(severity)
    factor = [0.70, 0.50, 0.40, 0.30, 0.20][s - 1]
    means = x.mean(axis=(2, 3), keepdims=True)  # (B, 3, 1, 1)
    return np.clip(x * factor + means * (1.0 - factor), 0.0, 1.0).astype(np.float32)


def elastic_transform(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    _check_image_batch(x)
    s = _check_severity(severity)
    alpha, sigma = [(30, 5), (50, 5), (70, 6), (90, 7), (110, 8)][s - 1]
    from scipy.ndimage import gaussian_filter, map_coordinates
    B, C, H, W = x.shape
    out = np.empty_like(x)
    for b in range(B):
        dy = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), sigma=sigma) * alpha
        dx = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), sigma=sigma) * alpha
        y_base, x_base = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        yy = np.clip(y_base + dy, 0, H - 1)
        xx = np.clip(x_base + dx, 0, W - 1)
        for c in range(C):
            out[b, c] = map_coordinates(x[b, c], [yy, xx], order=1, mode="nearest")
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def pixelate(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    _check_image_batch(x)
    s = _check_severity(severity)
    block = [2, 3, 4, 6, 8][s - 1]
    B, C, H, W = x.shape
    # Trim to block-divisible size, average, then repeat
    h2 = (H // block) * block
    w2 = (W // block) * block
    x_trim = x[:, :, :h2, :w2]
    small = x_trim.reshape(B, C, h2 // block, block, w2 // block, block).mean(axis=(3, 5))
    big = np.repeat(np.repeat(small, block, axis=2), block, axis=3)
    out = x.copy()
    out[:, :, :h2, :w2] = big
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def jpeg_compression(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    _check_image_batch(x)
    s = _check_severity(severity)
    quality = [75, 60, 40, 30, 20][s - 1]
    from PIL import Image
    B, C, H, W = x.shape
    out = np.empty_like(x)
    for b in range(B):
        img_np = (x[b].transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(img_np, mode="RGB")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        arr = np.array(Image.open(buf)).astype(np.float32) / 255.0
        out[b] = arr.transpose(2, 0, 1)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, CorruptionFn] = {
    "identity":         identity,
    "gaussian_noise":   gaussian_noise,
    "shot_noise":       shot_noise,
    "impulse_noise":    impulse_noise,
    "defocus_blur":     defocus_blur,
    "glass_blur":       glass_blur,
    "motion_blur":      motion_blur,
    "zoom_blur":        zoom_blur,
    "snow":             snow,
    "frost":            frost,
    "fog":              fog,
    "brightness":       brightness,
    "contrast":         contrast,
    "elastic_transform": elastic_transform,
    "pixelate":         pixelate,
    "jpeg_compression": jpeg_compression,
}


def available_corruptions() -> List[str]:
    return sorted(_REGISTRY.keys())


def apply_corruption(
    name: str,
    x: np.ndarray,
    severity: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown corruption {name!r}; have {available_corruptions()}")
    return _REGISTRY[name](x, severity, rng)
