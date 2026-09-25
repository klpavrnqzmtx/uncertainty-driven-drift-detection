"""Tests for the audio modality: front-end, corruptions, pools, stream, model.

Everything here runs without the dataset tarballs except the two tests
marked ``needs_audio_cache``, which are skipped when the caches built by
``scripts/prepare_audio.py`` are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import uncertainty_driven_drift.components.phase3  # noqa: F401 — register components
from uncertainty_driven_drift.components.audio_corruptions import (
    CORRUPTION_GROUPS,
    apply_corruption,
    available_corruptions,
)
from uncertainty_driven_drift.components.audio_features import (
    N_MELS,
    SAMPLE_RATE,
    fix_length,
    log_mel_batch,
    mel_filterbank,
    n_frames_for,
    require_audio_spec,
)
from uncertainty_driven_drift.components.audio_pools import DATASETS, cache_paths, get_spec
from uncertainty_driven_drift.data.base import StreamSpec
from uncertainty_driven_drift.registry import build

AUDIO_ROOT = Path("artifacts/audio")


def _tone_batch(b: int = 4, n: int = SAMPLE_RATE, seed: int = 0) -> np.ndarray:
    """A 440 Hz tone plus mild noise — a signal with real spectral structure.

    White noise alone would make every filter-type corruption look identical.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    base = 0.3 * np.sin(2 * np.pi * 440.0 * t)
    x = base[None, :] + 0.05 * rng.standard_normal((b, n))
    return np.ascontiguousarray(x[:, None, :], dtype=np.float32)


# ---------------------------------------------------------------------------
# front-end
# ---------------------------------------------------------------------------

def test_log_mel_shape_and_finiteness() -> None:
    x = _tone_batch(b=3)
    lm = log_mel_batch(x)
    assert lm.shape == (3, 1, N_MELS, n_frames_for(SAMPLE_RATE))
    assert lm.dtype == np.float32
    assert np.isfinite(lm).all()


def test_log_mel_accepts_both_layouts() -> None:
    x = _tone_batch(b=2)
    np.testing.assert_allclose(log_mel_batch(x), log_mel_batch(x[:, 0, :]))


def test_log_mel_locates_the_tone() -> None:
    """A 440 Hz tone must peak in the mel band that contains 440 Hz."""
    lm = log_mel_batch(_tone_batch(b=1))[0, 0]          # (n_mels, frames)
    fb = mel_filterbank()
    bin_440 = int(round(440.0 / (SAMPLE_RATE / 2) * (fb.shape[1] - 1)))
    expected_band = int(np.argmax(fb[:, bin_440]))
    assert abs(int(np.argmax(lm.mean(axis=1))) - expected_band) <= 2


def test_fix_length_crops_and_pads() -> None:
    assert fix_length(np.zeros(100, dtype=np.float32), 40).shape == (40,)
    assert fix_length(np.zeros(30, dtype=np.float32), 40).shape == (40,)
    x = np.arange(10, dtype=np.float32)
    np.testing.assert_array_equal(fix_length(x, 10), x)


def test_require_audio_spec_rejects_image_streams() -> None:
    img = StreamSpec(name="cifar", input_shape=(3, 32, 32), n_classes=10,
                     n_batches=1, batch_size=1)
    with pytest.raises(ValueError, match="waveform stream"):
        require_audio_spec(img, "audio_cnn_mc_dropout")

    wav = StreamSpec(name="audio", input_shape=(1, SAMPLE_RATE), n_classes=35,
                     n_batches=1, batch_size=1)
    assert require_audio_spec(wav, "audio_cnn_mc_dropout") == (1, SAMPLE_RATE)


# ---------------------------------------------------------------------------
# corruptions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", available_corruptions())
def test_corruption_preserves_shape_dtype_and_is_finite(name: str) -> None:
    x = _tone_batch(b=3)
    out = apply_corruption(name, x, severity=3, rng=np.random.default_rng(1))
    assert out.shape == x.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_corruption_accepts_2d_and_returns_2d() -> None:
    x = _tone_batch(b=2)[:, 0, :]
    out = apply_corruption("gaussian_noise", x, severity=3, rng=np.random.default_rng(0))
    assert out.shape == x.shape


def test_identity_is_a_noop() -> None:
    x = _tone_batch()
    out = apply_corruption("identity", x, severity=3, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(out, x)


@pytest.mark.parametrize("name", ["gaussian_noise", "pink_noise", "hum_noise"])
def test_additive_severity_follows_the_snr_schedule(name: str) -> None:
    """Severity s must inject noise at 40, 30, 20, 10, 0 dB SNR."""
    x = _tone_batch(b=8, seed=3)
    for sev, snr_db in zip((1, 2, 3, 4, 5), (40.0, 30.0, 20.0, 10.0, 0.0)):
        out = apply_corruption(name, x, severity=sev, rng=np.random.default_rng(sev))
        noise = out - x
        measured = 10.0 * np.log10((x ** 2).mean() / max((noise ** 2).mean(), 1e-20))
        assert abs(measured - snr_db) < 1.5, (name, sev, measured)


# Time-warping corruptions are excluded from the *sample-domain* monotonicity
# check on purpose: shifting a 440 Hz tone by even 5% fully decorrelates it
# sample-by-sample, so |y - x| saturates at severity 1 and carries no ordering.
# They are checked spectrally instead, where the ordering is real.
_TIME_WARPING = {"speed", "time_stretch", "pitch_shift"}


def test_severity_is_monotone_in_distortion() -> None:
    x = _tone_batch(b=8, seed=5)
    for name in available_corruptions():
        if name == "identity" or name in _TIME_WARPING:
            continue
        dists = [
            float(np.abs(apply_corruption(name, x, s, np.random.default_rng(11)) - x).mean())
            for s in (1, 3, 5)
        ]
        assert dists[0] <= dists[2] + 1e-6, (name, dists)


@pytest.mark.parametrize("name", sorted(_TIME_WARPING))
def test_time_warping_severity_is_monotone_in_the_spectral_domain(name: str) -> None:
    x = _tone_batch(b=8, seed=5)
    lm0 = log_mel_batch(x)
    dists = [
        float(np.abs(log_mel_batch(apply_corruption(name, x, s, np.random.default_rng(11))) - lm0).mean())
        for s in (1, 3, 5)
    ]
    assert dists[0] < dists[1] < dists[2], (name, dists)
    # A corruption that barely moves the spectrogram is not a corruption. The
    # first pitch_shift implementation (two reciprocal resamples) sat at 0.004.
    assert dists[0] > 0.05, (name, dists)


def test_unknown_corruption_names_fail_loudly() -> None:
    with pytest.raises(KeyError, match="Unknown audio corruption"):
        apply_corruption("no_such_thing", _tone_batch(), 3, np.random.default_rng(0))


def test_corruption_groups_cover_every_non_identity_corruption() -> None:
    grouped = {c for names in CORRUPTION_GROUPS.values() for c in names}
    assert grouped == set(available_corruptions()) - {"identity"}


# ---------------------------------------------------------------------------
# stream construction
# ---------------------------------------------------------------------------

def _cache_present(dataset: str) -> bool:
    root = AUDIO_ROOT / dataset
    return all(p.exists() for split in ("train", "test")
               for p in cache_paths(dataset, split, root))


needs_audio_cache = pytest.mark.skipif(
    not _cache_present("esc50"),
    reason="ESC-50 cache absent; run scripts/prepare_audio.py --dataset esc50",
)


def test_dataset_specs_are_self_consistent() -> None:
    for name, spec in DATASETS.items():
        assert get_spec(name) is spec
        assert spec.n_samples == int(round(spec.clip_seconds * SAMPLE_RATE))
        assert spec.n_classes > 1


@needs_audio_cache
def test_stream_phases_labels_and_boundary() -> None:
    stream = build(
        "dataset", "audio_known_novel",
        dataset="esc50",
        known_corruptions=["gaussian_noise", "lowpass"],
        novel_corruptions=["packet_loss"],
        n_batches_per_phase=2, severity=3, batch_size=8, seed=0,
        data_root=str(AUDIO_ROOT / "esc50"),
    )
    spec = stream.spec
    assert spec.input_shape == (1, get_spec("esc50").n_samples)
    assert spec.n_classes == 50
    assert spec.n_batches == 6
    assert spec.extras["novel_start_batch"] == 4
    assert spec.drift_indices == (2, 4)

    batches = list(stream)
    assert len(batches) == spec.n_batches
    assert [b.extras["corruption"] for b in batches] == [
        "gaussian_noise", "gaussian_noise", "lowpass", "lowpass",
        "packet_loss", "packet_loss",
    ]
    assert [b.extras["is_novel"] for b in batches] == [False] * 4 + [True] * 2
    for b in batches:
        assert b.x.shape == (8, 1, spec.input_shape[1])
        assert b.x.dtype == np.float32
        assert b.y.shape == (8,) and b.y.max() < 50
        assert np.isfinite(b.x).all()


@needs_audio_cache
def test_stream_is_deterministic_given_the_seed() -> None:
    def _first_batch(seed: int):
        stream = build(
            "dataset", "audio_known_novel", dataset="esc50",
            known_corruptions=["gaussian_noise"], novel_corruptions=["packet_loss"],
            n_batches_per_phase=1, batch_size=4, seed=seed,
            data_root=str(AUDIO_ROOT / "esc50"),
        )
        return next(iter(stream)).x

    np.testing.assert_array_equal(_first_batch(0), _first_batch(0))
    assert not np.allclose(_first_batch(0), _first_batch(1))


def test_stream_rejects_a_corruption_listed_as_both_known_and_novel() -> None:
    with pytest.raises(ValueError, match="both known and novel"):
        build("dataset", "audio_known_novel", dataset="esc50",
              known_corruptions=["gaussian_noise"], novel_corruptions=["gaussian_noise"],
              data_root=str(AUDIO_ROOT / "esc50"))


def test_model_rejects_a_dataset_mismatch() -> None:
    """A checkpoint's log-mel statistics do not transfer across datasets."""
    model = build("model", "audio_cnn_mc_dropout", dataset="esc50")
    spec = StreamSpec(
        name="audio_known_novel", input_shape=(1, get_spec("esc50").n_samples),
        n_classes=50, n_batches=1, batch_size=1,
        extras={"dataset": "urbansound8k"},
    )
    with pytest.raises(ValueError, match="configured for 'esc50'"):
        model.setup(spec)


# ---------------------------------------------------------------------------
# pool builder (synthetic archive — no real dataset needed)
# ---------------------------------------------------------------------------

def _write_fake_speech_commands(tmp_path: Path) -> Path:
    """Build a Speech-Commands-shaped tarball: 2 words, official list files, 1 corrupt WAV."""
    import io
    import tarfile
    from scipy.io import wavfile

    from uncertainty_driven_drift.components.audio_pools import DATASETS

    root = tmp_path / "speech_commands"
    root.mkdir(parents=True)
    archive = root / DATASETS["speech_commands"].archive

    def _wav_bytes(freq: float, sr: int = 8000, n: int = 8000) -> bytes:
        t = np.arange(n) / sr
        x = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
        buf = io.BytesIO()
        wavfile.write(buf, sr, x)          # 8 kHz on purpose: exercises the resampler
        return buf.getvalue()

    files = {
        "yes/a_nohash_0.wav": _wav_bytes(300.0),
        "yes/b_nohash_0.wav": _wav_bytes(320.0),
        "yes/c_nohash_0.wav": _wav_bytes(340.0),
        "no/a_nohash_0.wav": _wav_bytes(600.0),
        "no/b_nohash_0.wav": _wav_bytes(620.0),
        "no/c_nohash_0.wav": b"this is not a wav file",     # must be dropped, not silenced
        "_background_noise_/hum.wav": _wav_bytes(50.0),     # must be ignored entirely
    }
    lists = {
        "validation_list.txt": "yes/b_nohash_0.wav\nno/b_nohash_0.wav\n",
        "testing_list.txt": "yes/c_nohash_0.wav\nno/c_nohash_0.wav\n",
    }

    with tarfile.open(archive, "w:gz") as tar:
        for name, payload in {**files, **{k: v.encode() for k, v in lists.items()}}.items():
            info = tarfile.TarInfo(f"./{name}")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return root


def test_build_cache_uses_official_splits_and_drops_undecodable(tmp_path: Path) -> None:
    from uncertainty_driven_drift.components.audio_pools import build_cache, load_pool, to_float

    root = _write_fake_speech_commands(tmp_path)
    build_cache("speech_commands", root, verbose=False)

    n_samples = get_spec("speech_commands").n_samples
    counts = {}
    for split in ("train", "val", "test"):
        x, y = load_pool("speech_commands", split, root, build_if_missing=False)
        counts[split] = x.shape[0]
        assert x.shape[1] == n_samples          # resampled 8 kHz -> 16 kHz, fixed length
        assert y.shape[0] == x.shape[0]         # labels stay aligned after compaction
        peak = np.abs(to_float(x)).max(axis=1)
        assert (peak > 0.1).all()               # no all-silent placeholder rows survived

    # yes/a + no/a train, yes/b + no/b val, yes/c test (no/c was the corrupt file).
    # _background_noise_ is excluded from every split.
    assert counts == {"train": 2, "val": 2, "test": 1}


@needs_audio_cache
def test_union_split_concatenates_pools() -> None:
    from uncertainty_driven_drift.components.audio_pools import load_pool

    root = AUDIO_ROOT / "esc50"
    xv, yv = load_pool("esc50", "val", root, build_if_missing=False)
    xt, yt = load_pool("esc50", "test", root, build_if_missing=False)
    xu, yu = load_pool("esc50", "val+test", root, build_if_missing=False)

    assert xu.shape == (xv.shape[0] + xt.shape[0], xv.shape[1])
    assert yu.shape[0] == xu.shape[0]
    np.testing.assert_array_equal(yu[: yv.shape[0]], yv)
    np.testing.assert_array_equal(yu[yv.shape[0]:], yt)


# ---------------------------------------------------------------------------
# backbone dispatch (capacity arms)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch,widths,expect_params,expect_embed", [
    ("cnn", None, 247_058, 128),
    ("cnn", [64, 128, 256, 512], 1_576_434, 512),
    ("resnet18", None, 11_195_890, 512),
])
def test_backbone_sizes_and_shapes(arch, widths, expect_params, expect_embed) -> None:
    """The capacity arms are the control for 'is this a small-network artifact?'.

    Pinning the parameter counts keeps that argument honest: the numbers are
    quoted in the config headers.
    """
    import torch

    from uncertainty_driven_drift.components.audio_cnn import build_backbone

    model = build_backbone(arch, 50, 0.3, widths)
    assert sum(p.numel() for p in model.parameters()) == expect_params
    with torch.no_grad():
        # 64 mels x 501 frames = a 5 s ESC-50 clip; GAP must absorb any length.
        assert model(torch.randn(2, 1, 64, 501)).shape == (2, 50)
        assert model.embed(torch.randn(2, 1, 64, 501)).shape == (2, expect_embed)


def test_unknown_arch_fails_loudly() -> None:
    from uncertainty_driven_drift.components.audio_cnn import build_backbone

    with pytest.raises(KeyError, match="Unknown audio arch"):
        build_backbone("wavenet", 10, 0.0)
