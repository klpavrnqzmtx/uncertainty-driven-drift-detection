"""Waveform corruption family — the audio analogue of CIFAR-10-C.

Fifteen corruption types on a 1-5 severity scale, generated in-process so
there is no ``*-C`` tarball to download, exactly as
``cifar_c_corruptions.py`` / ``mnist_c_corruptions.py`` do for images.

Calibration
-----------
Additive corruptions are **SNR-controlled at 40/30/20/10/0 dB** for
severities 1-5, and the quantiser steps through ``2**c`` levels with
``c in {24, 16, 8, 4, 2}``.  Both scales are taken from AVRobustBench
(Kinetics-2C / AudioSet-2C, arXiv:2506.00358) so severities here mean the
same thing they do in the published audio-robustness benchmarks rather
than being invented per corruption.  The non-additive families (filters,
reverb, time/pitch, packet loss) carry their own five-entry tables,
chosen so severity 1 is barely audible and severity 5 is severe but never
destroys the label — the same design rule Hendrycks & Dietterich use.

Every function is pure numpy/scipy, operates on ``(B, T)`` or ``(B, 1, T)``
float32 waveforms in [-1, 1], preserves shape, and takes its randomness
only from the passed-in ``Generator`` so a stream stays reproducible.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

# Severity -> SNR in dB for every additive-noise corruption.
_SNR_DB = (40.0, 30.0, 20.0, 10.0, 0.0)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _as_2d(x: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return ``(B, T)`` view plus a flag recording whether a channel axis was dropped."""
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 3:
        if a.shape[1] != 1:
            raise ValueError(f"audio corruptions expect mono (B, 1, T); got {a.shape}")
        return a[:, 0, :], True
    if a.ndim != 2:
        raise ValueError(f"audio corruptions expect (B, T) or (B, 1, T); got {a.shape}")
    return a, False


def _restore(y: np.ndarray, had_channel: bool) -> np.ndarray:
    out = y[:, None, :] if had_channel else y
    return np.ascontiguousarray(out, dtype=np.float32)


def _severity_index(severity: int) -> int:
    s = int(severity)
    if not 1 <= s <= 5:
        raise ValueError(f"severity must be in 1..5; got {severity}")
    return s - 1


def _signal_power(x: np.ndarray) -> np.ndarray:
    """Per-sample mean power, floored so silent clips do not divide by zero."""
    return np.maximum((x ** 2).mean(axis=-1, keepdims=True), 1e-10)


def _mix_at_snr(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Add ``noise`` to ``x`` scaled to the requested per-clip SNR."""
    noise_power = np.maximum((noise ** 2).mean(axis=-1, keepdims=True), 1e-20)
    target = _signal_power(x) / (10.0 ** (snr_db / 10.0))
    return x + noise * np.sqrt(target / noise_power)


def _lfilter(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    from scipy.signal import lfilter

    return lfilter(b, a, x, axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# additive noise
# ---------------------------------------------------------------------------

def gaussian_noise(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """White Gaussian noise — the microphone/electronics floor."""
    n = rng.standard_normal(x.shape).astype(np.float32)
    return _mix_at_snr(x, n, _SNR_DB[_severity_index(severity)])


def pink_noise(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise — traffic hum, HVAC, wind rumble; energy concentrated low."""
    n_fft = x.shape[-1]
    freqs = np.fft.rfftfreq(n_fft, d=1.0)
    scale = 1.0 / np.sqrt(np.maximum(freqs, 1.0 / n_fft))
    white = np.fft.rfft(rng.standard_normal(x.shape), axis=-1)
    n = np.fft.irfft(white * scale, n=n_fft, axis=-1).astype(np.float32)
    return _mix_at_snr(x, n, _SNR_DB[_severity_index(severity)])


def impulse_noise(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Sparse clicks / crackle — dropped packets, bad cable, vinyl pops."""
    density = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)[_severity_index(severity)]
    mask = rng.random(x.shape) < density
    amp = rng.uniform(0.5, 1.0, size=x.shape).astype(np.float32)
    sign = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=x.shape)
    peak = np.maximum(np.abs(x).max(axis=-1, keepdims=True), 1e-6)
    return np.clip(x + mask * amp * sign * peak, -1.0, 1.0)


def hum_noise(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Mains hum: a 50 Hz tone plus its first two harmonics."""
    t = np.arange(x.shape[-1], dtype=np.float32) / 16_000.0
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(x.shape[0], 1)).astype(np.float32)
    tone = np.zeros_like(x)
    for k, w in ((1, 1.0), (2, 0.5), (3, 0.25)):
        tone += w * np.sin(2.0 * np.pi * 50.0 * k * t[None, :] + phase)
    return _mix_at_snr(x, tone, _SNR_DB[_severity_index(severity)])


# ---------------------------------------------------------------------------
# amplitude / quantisation
# ---------------------------------------------------------------------------

def clipping(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Hard clipping — an overdriven preamp or too-hot input gain."""
    del rng
    q = (99.0, 95.0, 90.0, 80.0, 60.0)[_severity_index(severity)]
    thresh = np.percentile(np.abs(x), q, axis=-1, keepdims=True)
    thresh = np.maximum(thresh, 1e-6)
    return np.clip(x, -thresh, thresh).astype(np.float32)


def quantization(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Bit-depth reduction to ``2**c`` levels, ``c in {24, 16, 8, 4, 2}``.

    The compression axis of AVRobustBench: severity 5 is a 2-bit signal.
    """
    del rng
    bits = (24, 16, 8, 4, 2)[_severity_index(severity)]
    levels = float(2 ** bits)
    peak = np.maximum(np.abs(x).max(axis=-1, keepdims=True), 1e-6)
    return (np.round(x / peak * (levels / 2.0)) / (levels / 2.0) * peak).astype(np.float32)


def gain(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Level shift — a mis-set recording gain, no spectral change."""
    del rng
    db = (-3.0, -6.0, -12.0, -20.0, -30.0)[_severity_index(severity)]
    return (x * (10.0 ** (db / 20.0))).astype(np.float32)


# ---------------------------------------------------------------------------
# linear filtering
# ---------------------------------------------------------------------------

def _butter(kind: str, cutoff, order: int = 4):
    from scipy.signal import butter

    return butter(order, cutoff, btype=kind, fs=16_000.0)


def lowpass(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Band-limiting — telephone codec, muffled/occluded microphone."""
    del rng
    cutoff = (6000.0, 4000.0, 2500.0, 1500.0, 800.0)[_severity_index(severity)]
    b, a = _butter("low", cutoff)
    return _lfilter(b, a, x)


def highpass(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Bass roll-off — small-diaphragm mic, aggressive rumble filter."""
    del rng
    cutoff = (100.0, 300.0, 600.0, 1000.0, 1600.0)[_severity_index(severity)]
    b, a = _butter("high", cutoff)
    return _lfilter(b, a, x)


def band_stop(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Notch out a widening band around 1 kHz — comb filtering / dead driver."""
    del rng
    width = (0.15, 0.3, 0.5, 0.7, 0.85)[_severity_index(severity)]
    lo = 1000.0 * (1.0 - width)
    hi = 1000.0 * (1.0 + width * 3.0)
    b, a = _butter("bandstop", [max(lo, 30.0), min(hi, 7900.0)], order=4)
    return _lfilter(b, a, x)


# ---------------------------------------------------------------------------
# room / propagation
# ---------------------------------------------------------------------------

def reverb(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Convolution with a synthetic exponentially-decaying room response.

    Severity moves the RT60 (0.1 s -> 1.2 s, a booth up to a hall) *and* the
    wet/dry energy ratio (0.1 -> 0.9, i.e. how far from the source the mic is).

    The direct path is held at unit gain and only the reflections are scaled.
    Normalising the whole impulse response to unit energy instead — the
    obvious-looking alternative — makes the direct component vanish as RT60
    grows, so severity 1 came out *more* distorting than severity 3 and the
    scale stopped being a scale.
    """
    from scipy.signal import fftconvolve

    idx = _severity_index(severity)
    rt60 = (0.1, 0.25, 0.45, 0.8, 1.2)[idx]
    wet = (0.1, 0.2, 0.35, 0.6, 0.9)[idx]

    n = max(int(rt60 * 16_000.0), 2)
    t = np.arange(n, dtype=np.float32) / 16_000.0
    tail = rng.standard_normal(n).astype(np.float32) * np.exp(-6.9078 * t / rt60)
    tail[0] = 0.0
    tail /= np.sqrt(max((tail ** 2).sum(), 1e-12))    # unit-energy reflections
    ir = tail * np.sqrt(wet)
    ir[0] = 1.0                                      # direct path, unattenuated
    out = fftconvolve(x, ir[None, :], mode="full", axes=-1)[..., : x.shape[-1]]
    return out.astype(np.float32)


def echo(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """One discrete delayed copy — a hard reflecting wall, or comms echo."""
    del rng
    delay_ms, decay = ((20.0, 0.2), (40.0, 0.35), (80.0, 0.5), (150.0, 0.6), (250.0, 0.7))[
        _severity_index(severity)
    ]
    d = int(delay_ms * 16.0)
    y = x.copy()
    if d < x.shape[-1]:
        y[..., d:] += decay * x[..., : x.shape[-1] - d]
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def packet_loss(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Zero out random short spans — VoIP dropouts, buffer underruns."""
    frac, span_ms = ((0.02, 20.0), (0.05, 20.0), (0.10, 30.0), (0.20, 30.0), (0.35, 40.0))[
        _severity_index(severity)
    ]
    span = max(int(span_ms * 16.0), 1)
    y = x.copy()
    n_spans = max(int(frac * x.shape[-1] / span), 1)
    for i in range(x.shape[0]):
        starts = rng.integers(0, max(x.shape[-1] - span, 1), size=n_spans)
        for s in starts:
            y[i, s:s + span] = 0.0
    return y


# ---------------------------------------------------------------------------
# time / pitch
# ---------------------------------------------------------------------------

def _resample_keep_length(x: np.ndarray, factor: float) -> np.ndarray:
    """Resample by ``factor``, then crop/pad back to the original length.

    ``resample_poly`` on a rational approximation of ``factor``, not
    ``scipy.signal.resample``: the latter is a single FFT of the (arbitrary,
    badly-factorable) clip length and measured ~5x slower here for the same
    result.
    """
    from fractions import Fraction

    from scipy.signal import resample_poly

    ratio = Fraction(float(factor)).limit_denominator(100)
    y = resample_poly(x, ratio.numerator, ratio.denominator, axis=-1).astype(np.float32)
    return _pad_or_crop(y, x.shape[-1])


def speed(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Playback-rate change: tempo *and* pitch move together, as on tape.

    Clock drift between a recorder and a player, or a resampling bug in a
    capture pipeline.  Distinct from :func:`time_stretch` and
    :func:`pitch_shift`, which each move one of the two.
    """
    del rng
    factor = (1.05, 1.10, 1.20, 1.35, 1.50)[_severity_index(severity)]
    return _resample_keep_length(x, factor)


# -- phase vocoder ----------------------------------------------------------
#
# A pitch shift at fixed duration cannot be done by resampling alone: you have
# to stretch the signal in time first and then resample it back, and the stretch
# has to preserve pitch, which means operating on STFT phase.  The first version
# here tried two reciprocal resamples instead; they very nearly cancel, so
# severities 1-2 were numerically no-ops (log-mel delta 0.004 against 1.5 for
# `speed`).  A corruption that does nothing is worse than a missing one, so this
# is a real phase vocoder.

_PV_N_FFT = 512
_PV_HOP = 128


def _pv_window() -> np.ndarray:
    return np.hanning(_PV_N_FFT + 1)[:-1].astype(np.float32)


def _pv_stft(x: np.ndarray) -> np.ndarray:
    """``(B, T, n_fft//2+1)`` complex STFT with centre padding."""
    pad = _PV_N_FFT // 2
    xp = np.pad(x, ((0, 0), (pad, pad)), mode="reflect")
    n_frames = 1 + (xp.shape[-1] - _PV_N_FFT) // _PV_HOP
    idx = np.arange(_PV_N_FFT)[None, :] + _PV_HOP * np.arange(n_frames)[:, None]
    frames = xp[:, idx] * _pv_window()[None, None, :]
    return np.fft.rfft(frames, axis=-1)


def _pv_istft(spec: np.ndarray, length: int) -> np.ndarray:
    """Overlap-add inverse of :func:`_pv_stft`, cropped to ``length``."""
    window = _pv_window()
    frames = np.fft.irfft(spec, n=_PV_N_FFT, axis=-1) * window[None, None, :]
    b, n_frames, _ = frames.shape
    total = _PV_HOP * (n_frames - 1) + _PV_N_FFT
    out = np.zeros((b, total), dtype=np.float32)
    wsum = np.zeros(total, dtype=np.float32)
    for k in range(n_frames):
        s = k * _PV_HOP
        out[:, s:s + _PV_N_FFT] += frames[:, k, :]
        wsum[s:s + _PV_N_FFT] += window ** 2
    out /= np.maximum(wsum, 1e-8)[None, :]
    pad = _PV_N_FFT // 2
    return out[:, pad:pad + length].astype(np.float32)


def _time_stretch_pv(x: np.ndarray, rate: float) -> np.ndarray:
    """Phase-vocoder time stretch. ``rate > 1`` = faster (shorter) output."""
    spec = _pv_stft(x)                                  # (B, T, bins)
    n_frames = spec.shape[1]
    mag, phase = np.abs(spec), np.angle(spec)
    bins = spec.shape[-1]
    expected = 2.0 * np.pi * _PV_HOP * np.arange(bins) / _PV_N_FFT

    positions = np.arange(0.0, n_frames - 1, rate)
    out = np.empty((spec.shape[0], positions.size, bins), dtype=np.complex128)
    acc = phase[:, 0, :].copy()
    for k, pos in enumerate(positions):
        i = int(np.floor(pos))
        frac = float(pos - i)
        j = min(i + 1, n_frames - 1)
        out[:, k, :] = ((1.0 - frac) * mag[:, i, :] + frac * mag[:, j, :]) * np.exp(1j * acc)
        dphi = phase[:, j, :] - phase[:, i, :] - expected[None, :]
        dphi -= 2.0 * np.pi * np.round(dphi / (2.0 * np.pi))     # principal value
        acc = acc + expected[None, :] + dphi
    return _pv_istft(out, length=int(round(x.shape[-1] / rate)))


def time_stretch(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Tempo change at constant pitch — a slowed or hurried talker.

    Not used by the default known/novel split; kept because it is the third
    independent axis of the time/pitch family and useful for held-out sweeps.
    """
    del rng
    rate = (1.05, 1.10, 1.20, 1.35, 1.50)[_severity_index(severity)]
    # The stretched signal is shorter; the stream's clips are fixed length, so
    # pad back rather than resampling back (which would undo the tempo change).
    return _pad_or_crop(_time_stretch_pv(x, rate), x.shape[-1])


def pitch_shift(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Pitch shift at constant duration — resampling artefact, wrong-speaker timbre.

    Stretch by ``ratio`` with the vocoder (pitch unchanged, duration longer),
    then resample back to the original length, which multiplies every frequency
    by ``ratio``.  Severity is the shift in semitones.
    """
    del rng
    semitones = (1.0, 2.0, 3.5, 5.0, 7.0)[_severity_index(severity)]
    ratio = 2.0 ** (semitones / 12.0)
    stretched = _time_stretch_pv(x, 1.0 / ratio)         # length ~ n * ratio
    return _resample_to_length(stretched, x.shape[-1])


def _pad_or_crop(x: np.ndarray, n: int) -> np.ndarray:
    """Centre-pad with silence or centre-crop to exactly ``n`` samples."""
    m = x.shape[-1]
    if m == n:
        return x.astype(np.float32)
    if m > n:
        start = (m - n) // 2
        return x[..., start:start + n].astype(np.float32)
    pad = n - m
    return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad // 2, pad - pad // 2)]).astype(np.float32)


def _resample_to_length(x: np.ndarray, n: int) -> np.ndarray:
    from scipy.signal import resample

    if x.shape[-1] == n:
        return x.astype(np.float32)
    return resample(x, n, axis=-1).astype(np.float32)


def identity(x: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """No-op — the clean channel, kept so configs can name it explicitly."""
    del severity, rng
    return np.asarray(x, dtype=np.float32)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

_CORRUPTIONS: Dict[str, Callable[[np.ndarray, int, np.random.Generator], np.ndarray]] = {
    "gaussian_noise": gaussian_noise,
    "pink_noise": pink_noise,
    "impulse_noise": impulse_noise,
    "hum_noise": hum_noise,
    "clipping": clipping,
    "quantization": quantization,
    "gain": gain,
    "lowpass": lowpass,
    "highpass": highpass,
    "band_stop": band_stop,
    "reverb": reverb,
    "echo": echo,
    "packet_loss": packet_loss,
    "speed": speed,
    "time_stretch": time_stretch,
    "pitch_shift": pitch_shift,
    "identity": identity,
}

# Groupings used by the configs and quoted in the paper text.
CORRUPTION_GROUPS = {
    "additive": ["gaussian_noise", "pink_noise", "impulse_noise", "hum_noise"],
    "amplitude": ["clipping", "quantization", "gain"],
    "filter": ["lowpass", "highpass", "band_stop"],
    "propagation": ["reverb", "echo", "packet_loss"],
    "time_pitch": ["speed", "time_stretch", "pitch_shift"],
}


def available_corruptions() -> List[str]:
    return sorted(_CORRUPTIONS)


def apply_corruption(
    name: str,
    x: np.ndarray,
    severity: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply corruption ``name`` at ``severity`` to a waveform batch.

    Shape in == shape out, ``(B, T)`` or ``(B, 1, T)`` float32.  Output is
    left un-normalised on purpose: level *is* part of what several of these
    corruptions do, and rescaling here would silently undo ``gain`` and half
    of ``clipping``.
    """
    try:
        fn = _CORRUPTIONS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown audio corruption {name!r}; have {available_corruptions()}"
        ) from exc
    x2, had_channel = _as_2d(x)
    return _restore(fn(x2, severity, rng), had_channel)
