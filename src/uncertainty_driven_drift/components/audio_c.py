"""Audio known-vs-novel corruption stream — the audio arm of scenario S1.

Same protocol as ``cifar_c.py`` / ``mnist_c.py``, one modality over:

* **Known phase** — corruptions the backbone was trained on, each streamed
  for ``n_batches_per_phase`` batches.  The model has seen these, so
  epistemic uncertainty should stay low even though accuracy drops.
* **Novel phase** — corruptions absent from training.  This is where an
  epistemic detector is supposed to fire.

``spec.extras["novel_start_batch"]`` marks the boundary, so the existing
plotting and sweep code annotates these runs identically to the image ones.

Two layouts. ``blocked`` is the DEFAULT and is the protocol every reported audio
run uses — it is the image arms' protocol, so the modalities stay comparable.
``mixed`` exists because of a diagnosis, described below; no reported
run uses it.

* ``blocked`` (the image arms' protocol) — one corruption at a time, 20 batches
  each. The known phase then contains **six internal change points**, and they
  are real: measured on ESC-50, known corruptions sit at systematically different
  epistemic levels (band_stop 0.288 vs quantization 0.211) and the spread of
  block means (0.0245) exceeds the within-block noise (0.0173). A change detector
  fires on those, correctly, and the run reports them as known-phase false
  alarms. They are an artifact of streaming corruptions in blocks.
* ``mixed`` — every batch draws a fresh corruption per *sample* from the current
  phase's set. The known phase becomes stationary, so the only change point in
  the stream is the one under test, and a known-phase alarm is unambiguously a
  false alarm. This is also closer to how the backbone was trained (a random
  known corruption per mini-batch) and to deployment, where the corruption mix
  does not switch wholesale every 20 batches.

Batches carry **waveforms**, ``(B, 1, n_samples)`` float32 in [-1, 1]; the
model owns log-mel featurisation (see ``audio_features.py`` for why).

Registered components
---------------------
* ``audio_known_novel`` — DatasetStream over speech_commands / esc50 /
  urbansound8k, ``input_shape=(1, n_samples)``.
"""

from __future__ import annotations

from typing import Iterator, List, Sequence

import numpy as np

from uncertainty_driven_drift.components.audio_corruptions import (
    apply_corruption,
    available_corruptions,
)
from uncertainty_driven_drift.components.audio_pools import get_spec, load_pool, to_float
from uncertainty_driven_drift.data.base import DatasetStream, StreamBatch, StreamSpec
from uncertainty_driven_drift.registry import register

_SHORT_NAME = {
    "gaussian_noise": "gauss",
    "pink_noise": "pink",
    "impulse_noise": "impulse",
    "hum_noise": "hum",
    "clipping": "clip",
    "quantization": "quant",
    "gain": "gain",
    "lowpass": "lowpass",
    "highpass": "highpass",
    "band_stop": "notch",
    "reverb": "reverb",
    "echo": "echo",
    "packet_loss": "dropout",
    "speed": "speed",
    "pitch_shift": "pitch",
    "identity": "clean",
}


@register("dataset", "audio_known_novel")
class AudioKnownNovelStream(DatasetStream):
    """Waveform stream with known-then-novel corruption phases.

    Parameters
    ----------
    dataset :
        ``speech_commands`` | ``esc50`` | ``urbansound8k``.
    known_corruptions, novel_corruptions :
        Corruption names (see ``audio_corruptions.available_corruptions``).
        ``known_*`` must match the backbone's ``train_corruptions`` or the
        scenario is meaningless — the preflight check enforces this.
    n_batches_per_phase :
        Batches per corruption type.
    severity :
        1-5.  Additive corruptions are SNR-calibrated (40 dB ... 0 dB).
    split :
        Which pool the stream draws from.  ``test`` by default: the backbone
        trains on ``train``, so streaming ``test`` keeps the *examples* unseen
        while the *corruptions* are the variable under test.
    """

    def __init__(
        self,
        dataset: str = "speech_commands",
        known_corruptions: Sequence[str] = (),
        novel_corruptions: Sequence[str] = (),
        n_batches_per_phase: int = 20,
        severity: int = 3,
        batch_size: int = 64,
        seed: int = 0,
        split: str = "test",
        data_root: str = "./artifacts/audio/speech_commands",
        layout: str = "blocked",
    ) -> None:
        if not known_corruptions:
            raise ValueError("known_corruptions must not be empty")
        if not novel_corruptions:
            raise ValueError("novel_corruptions must not be empty")

        avail = set(available_corruptions())
        bad = [c for c in list(known_corruptions) + list(novel_corruptions) if c not in avail]
        if bad:
            raise KeyError(f"Unknown audio corruptions {bad!r}; have {sorted(avail)}")
        overlap = sorted(set(known_corruptions) & set(novel_corruptions))
        if overlap:
            # A corruption in both lists makes the "novel" phase partly known and
            # silently weakens whatever the run reports.
            raise ValueError(f"{overlap!r} listed as both known and novel")

        self.dataset = str(dataset)
        self.known_corruptions = list(known_corruptions)
        self.novel_corruptions = list(novel_corruptions)
        self.n_batches_per_phase = int(n_batches_per_phase)
        self.severity = int(severity)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.split = str(split)
        self.data_root = str(data_root)
        if layout not in ("blocked", "mixed"):
            raise ValueError(f"layout must be 'blocked' or 'mixed'; got {layout!r}")
        self.layout = str(layout)

        ds_spec = get_spec(self.dataset)
        self._pool_x, self._pool_y = load_pool(self.dataset, self.split, self.data_root)

        n_known = len(self.known_corruptions)
        n_novel = len(self.novel_corruptions)
        if self.layout == "blocked":
            total_batches = (n_known + n_novel) * self.n_batches_per_phase
            novel_start = n_known * self.n_batches_per_phase
            drift_indices: List[int] = [
                i * self.n_batches_per_phase for i in range(1, n_known + n_novel)
            ]
            channel_names = [
                _SHORT_NAME.get(c, c)
                for c in self.known_corruptions + self.novel_corruptions
            ]
        else:
            # Same batch budget as blocked, so the two layouts are comparable:
            # n_known blocks' worth of known batches, then n_novel blocks' worth
            # of novel ones. Only one change point exists.
            total_batches = (n_known + n_novel) * self.n_batches_per_phase
            novel_start = n_known * self.n_batches_per_phase
            drift_indices = [novel_start]
            channel_names = ["known mix", "novel mix"]

        self.spec = StreamSpec(
            name="audio_known_novel",
            input_shape=(1, ds_spec.n_samples),
            n_classes=ds_spec.n_classes,
            n_batches=total_batches,
            batch_size=self.batch_size,
            drift_indices=tuple(drift_indices),
            has_true_posterior=False,
            extras={
                "dataset": self.dataset,
                "split": self.split,
                "sample_rate": 16_000,
                "clip_seconds": ds_spec.clip_seconds,
                "severity": self.severity,
                "known_corruptions": self.known_corruptions,
                "novel_corruptions": self.novel_corruptions,
                "novel_start_batch": novel_start,
                "layout": self.layout,
                "channel_names": channel_names,
                "pool_size": int(self._pool_x.shape[0]),
                "available_corruptions": available_corruptions(),
                "citation": ds_spec.citation,
            },
        )

    def __iter__(self) -> Iterator[StreamBatch]:
        if self.layout == "blocked":
            yield from self._iter_blocked()
        else:
            yield from self._iter_mixed()

    def _iter_blocked(self) -> Iterator[StreamBatch]:
        sample_rng = np.random.default_rng(self.seed)
        corrupt_rng = np.random.default_rng(self.seed + 1)

        phase_corruptions = self.known_corruptions + self.novel_corruptions
        novel_start = self.spec.extras["novel_start_batch"]
        drift_set = set(self.spec.drift_indices)
        pool_n = self._pool_x.shape[0]

        t = 0
        for phase_i, corruption in enumerate(phase_corruptions):
            for _ in range(self.n_batches_per_phase):
                idx = np.sort(sample_rng.integers(0, pool_n, size=self.batch_size))
                x_clean = to_float(self._pool_x[idx])[:, None, :]   # (B, 1, T)
                y = np.asarray(self._pool_y[idx], dtype=np.int64)

                x = apply_corruption(corruption, x_clean, self.severity, corrupt_rng)

                yield StreamBatch(
                    index=t,
                    x=x,
                    y=y,
                    concept_id=phase_i,
                    is_drift=(t in drift_set),
                    true_posterior=None,
                    extras={
                        "corruption": corruption,
                        "is_novel": t >= novel_start,
                    },
                )
                t += 1

    def _iter_mixed(self) -> Iterator[StreamBatch]:
        """Every batch is a fresh per-sample mix of the current phase's corruptions.

        Each sample independently draws one corruption, so every batch has the
        same expected composition and the phase is stationary: the only change
        point in the stream is the known -> novel boundary. Corruptions are
        applied group-wise (one call per distinct corruption in the batch), which
        keeps the cost the same as the blocked layout.
        """
        sample_rng = np.random.default_rng(self.seed)
        corrupt_rng = np.random.default_rng(self.seed + 1)
        choice_rng = np.random.default_rng(self.seed + 2)

        novel_start = self.spec.extras["novel_start_batch"]
        pool_n = self._pool_x.shape[0]

        for t in range(self.spec.n_batches):
            is_novel = t >= novel_start
            menu = self.novel_corruptions if is_novel else self.known_corruptions

            idx = np.sort(sample_rng.integers(0, pool_n, size=self.batch_size))
            x = to_float(self._pool_x[idx])[:, None, :]
            y = np.asarray(self._pool_y[idx], dtype=np.int64)

            assign = choice_rng.integers(0, len(menu), size=self.batch_size)
            for j, corruption in enumerate(menu):
                sel = assign == j
                if not sel.any():
                    continue
                x[sel] = apply_corruption(corruption, x[sel], self.severity, corrupt_rng)

            yield StreamBatch(
                index=t,
                x=x,
                y=y,
                concept_id=int(is_novel),
                is_drift=(t == novel_start),
                true_posterior=None,
                extras={
                    "corruption": "novel_mix" if is_novel else "known_mix",
                    "corruption_counts": {
                        c: int((assign == j).sum()) for j, c in enumerate(menu)
                    },
                    "is_novel": is_novel,
                },
            )
