"""Waveform I/O and log-mel front-end for the audio streams.

Deliberately depends on **numpy + scipy only** — no torchaudio, no librosa,
no soundfile.  The cluster venv is built from ``pyproject.toml`` on a login
node and then used offline; every extra wheel is another thing that can fail
to resolve there.  Everything here (WAV decode, polyphase resample, STFT,
mel filterbank) is a few dozen lines against ``scipy.signal``, so the trade
is a clear win.

Conventions shared by every audio component
-------------------------------------------
* Waveforms are ``float32`` in ``[-1, 1]``, mono, at :data:`SAMPLE_RATE`.
* A stream batch carries **waveforms**, shape ``(B, 1, n_samples)``, and
  ``StreamSpec.input_shape == (1, n_samples)``.  Corruptions are physical
  processes on a signal, so they must be applied in the waveform domain;
  featurisation therefore belongs to the *model*, exactly as CIFAR channel
  normalisation does in ``resnet_cifar.py``.
* Models call :func:`log_mel_batch` to get ``(B, 1, n_mels, n_frames)``,
  which is then a 1-channel image and can go into an ordinary 2-D CNN.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from uncertainty_driven_drift.data.base import StreamSpec

SAMPLE_RATE = 16_000

# 25 ms window / 10 ms hop / 64 mel bands — the standard keyword-spotting and
# audio-tagging front-end (Speech Commands baselines, AST, PANNs all sit here).
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 64
FMIN = 20.0
FMAX = 7600.0

# log(x + LOG_OFFSET): floors the log at ~-11.5 so silence is a finite,
# stable value rather than -inf drifting with the dtype epsilon.
LOG_OFFSET = 1e-5


# ---------------------------------------------------------------------------
# WAV decoding
# ---------------------------------------------------------------------------

def _to_float32(data: np.ndarray) -> np.ndarray:
    """Scale an integer PCM array to float32 in [-1, 1]; pass floats through."""
    if data.dtype == np.float32 or data.dtype == np.float64:
        return data.astype(np.float32)
    if data.dtype == np.uint8:            # 8-bit PCM is unsigned, midpoint 128
        return (data.astype(np.float32) - 128.0) / 128.0
    info = np.iinfo(data.dtype)
    return data.astype(np.float32) / float(-info.min)


def load_wav(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Read a WAV file as mono float32 at ``sample_rate``.

    Handles the formats the three audio datasets actually ship: 16-bit PCM
    (Speech Commands), 24/32-bit and float WAV at 44.1/48 kHz (ESC-50,
    UrbanSound8K), mono or stereo.
    """
    from scipy.io import wavfile

    sr, data = wavfile.read(str(path))
    x = _to_float32(np.asarray(data))
    if x.ndim > 1:                        # (n, channels) -> mono
        x = x.mean(axis=1)
    if sr != sample_rate:
        x = resample(x, sr, sample_rate)
    return np.ascontiguousarray(x, dtype=np.float32)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resample (anti-aliased) from ``sr_in`` to ``sr_out``."""
    if sr_in == sr_out:
        return x.astype(np.float32)
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(int(sr_in), int(sr_out))
    return resample_poly(x, int(sr_out) // g, int(sr_in) // g).astype(np.float32)


def fix_length(x: np.ndarray, n_samples: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Crop or zero-pad ``x`` to exactly ``n_samples``.

    Longer clips are cropped at a random offset when ``rng`` is given (train
    time augmentation) and centred otherwise (deterministic evaluation).
    """
    n = x.shape[-1]
    if n == n_samples:
        return x
    if n > n_samples:
        start = int(rng.integers(0, n - n_samples + 1)) if rng is not None else (n - n_samples) // 2
        return x[start:start + n_samples]
    pad = n_samples - n
    left = pad // 2
    return np.pad(x, (left, pad - left))


# ---------------------------------------------------------------------------
# Log-mel spectrogram
# ---------------------------------------------------------------------------

def _hz_to_mel(f: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


_FILTERBANK_CACHE: dict[tuple, np.ndarray] = {}


def mel_filterbank(
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    fmin: float = FMIN,
    fmax: float = FMAX,
) -> np.ndarray:
    """Triangular mel filterbank, shape ``(n_mels, n_fft // 2 + 1)``.

    HTK-style (the log10 mel scale above), slope-normalised so each filter
    integrates to 1 — matches ``librosa.filters.mel(norm=None)`` closely
    enough that checkpoints are interchangeable in practice, and is stable
    on its own terms since we always train and evaluate through this code.
    """
    key = (sample_rate, n_fft, n_mels, fmin, fmax)
    cached = _FILTERBANK_CACHE.get(key)
    if cached is not None:
        return cached

    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_bins)
    mel_edges = np.linspace(_hz_to_mel(fmin), _hz_to_mel(min(fmax, sample_rate / 2.0)), n_mels + 2)
    hz_edges = _mel_to_hz(mel_edges)

    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(n_mels):
        lo, ctr, hi = hz_edges[m], hz_edges[m + 1], hz_edges[m + 2]
        rising = (fft_freqs - lo) / max(ctr - lo, 1e-9)
        falling = (hi - fft_freqs) / max(hi - ctr, 1e-9)
        fb[m] = np.clip(np.minimum(rising, falling), 0.0, None)
        norm = fb[m].sum()
        if norm > 0:
            fb[m] /= norm
    _FILTERBANK_CACHE[key] = fb
    return fb


def _frame(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Center-padded framing -> ``(..., n_frames, n_fft)`` view."""
    pad = n_fft // 2
    xp = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)], mode="reflect")
    n_frames = 1 + (xp.shape[-1] - n_fft) // hop
    shape = xp.shape[:-1] + (n_frames, n_fft)
    strides = xp.strides[:-1] + (hop * xp.strides[-1], xp.strides[-1])
    return np.lib.stride_tricks.as_strided(xp, shape=shape, strides=strides)


def log_mel_batch(
    wav: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
) -> np.ndarray:
    """Log-mel spectrogram of a waveform batch.

    Parameters
    ----------
    wav :
        ``(B, n_samples)`` or ``(B, 1, n_samples)`` float32 in [-1, 1].

    Returns
    -------
    ``(B, 1, n_mels, n_frames)`` float32 — a 1-channel image.
    """
    x = np.asarray(wav, dtype=np.float32)
    if x.ndim == 3:
        if x.shape[1] != 1:
            raise ValueError(f"expected mono (B, 1, T); got {x.shape}")
        x = x[:, 0, :]
    elif x.ndim != 2:
        raise ValueError(f"expected (B, T) or (B, 1, T); got {x.shape}")

    frames = _frame(x, n_fft, hop_length)                    # (B, F, n_fft)
    window = np.hanning(n_fft + 1)[:-1].astype(np.float32)   # periodic Hann
    spec = np.fft.rfft(frames * window, axis=-1)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)   # (B, F, bins)

    fb = mel_filterbank(sample_rate, n_fft, n_mels)          # (n_mels, bins)
    mel = power @ fb.T                                       # (B, F, n_mels)
    logmel = np.log(mel + LOG_OFFSET)
    return np.ascontiguousarray(logmel.transpose(0, 2, 1)[:, None, :, :], dtype=np.float32)


def n_frames_for(n_samples: int, hop_length: int = HOP_LENGTH) -> int:
    """Frame count :func:`log_mel_batch` produces for ``n_samples`` (center-padded)."""
    return 1 + n_samples // hop_length


# ---------------------------------------------------------------------------
# Spec guard
# ---------------------------------------------------------------------------

def require_audio_spec(spec: StreamSpec, component: str, *, n_classes: int | None = None) -> Tuple[int, int]:
    """Assert ``spec`` describes a waveform stream; return ``(channels, n_samples)``.

    The audio backbones normalise, featurise and pool against a ``(1, T)``
    waveform layout.  Handing one an image stream ``(3, 32, 32)`` or a
    tabular stream ``(8,)`` would otherwise fail deep inside the front-end,
    or — worse — succeed on nonsense.  Mirrors
    ``components/image_spec.require_image_spec``.
    """
    shape = tuple(int(d) for d in spec.input_shape)
    if len(shape) != 2 or shape[0] != 1:
        raise ValueError(
            f"{component} is an audio model and needs a (1, n_samples) waveform stream; "
            f"stream {spec.name!r} has input_shape={shape}. "
            f"Audio streams in this repo: audio_known_novel."
        )
    if n_classes is not None and spec.n_classes != n_classes:
        raise ValueError(
            f"{component} is {n_classes}-way; stream {spec.name!r} has n_classes={spec.n_classes}."
        )
    return shape[0], shape[1]
