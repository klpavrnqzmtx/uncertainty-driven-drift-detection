"""Quadriga SISO QPSK channel known-vs-novel stream + RNN MC-Dropout model.

Four QuaDRiGa channel realizations are loaded from .mat files:
  channel_01_3GPP_37_885_Highway_LOS      → known (index 0)
  channel_02_3GPP_37_885_Highway_NLOSv    → known (index 1)
  channel_03_3GPP_37_885_Urban_LOS        → novel (index 2)
  channel_04_3GPP_37_885_Urban_NLOS       → novel (index 3)

Transmission model
------------------
QPSK modulation (4 Gray-coded symbols, unit-power constellation).
Each batch = one block of ``batch_size`` consecutive symbols drawn from a
single channel. The channel is flat-fading (single time-varying complex gain
h[t] taken from the ``h`` field of the .mat file). Channel gain is normalised
to unit RMS so SNR is well-defined independent of absolute path loss:

    h_norm[t] = h[t] / rms(h)
    rx[t]     = h_norm[t] * s[t] + n[t],   n[t] ~ CN(0, σ²)
    σ         = 1 / sqrt(10^(snr_db/10))

Model input per symbol: [Re(rx[t]), Im(rx[t])] → shape (batch_size, 2).
Label per symbol: QPSK index 0-3.

Uncertainty
-----------
The LSTM backbone is frozen after pretraining. MC-Dropout (dropout layer
after the LSTM output, forced to train-mode at inference) produces an
(n_samples, batch_size, 4) probability stack consumed by the shared
``mc_dropout`` UncertaintyEstimator.

Registered components
---------------------
* ``quadriga_known_novel``  — DatasetStream, input_shape=(2,), n_classes=4
* ``rnn_mc_dropout_qpsk``   — LSTM(2→hidden) + Dropout + Linear(hidden→4)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

import numpy as np

from uncertainty_driven_drift.data.base import DatasetStream, StreamBatch, StreamSpec
from uncertainty_driven_drift.detectors.base import DetectorEvent
from uncertainty_driven_drift.models.base import Classifier, Prediction
from uncertainty_driven_drift.registry import register
from uncertainty_driven_drift.uncertainty.base import UncertaintyScores

# ---------------------------------------------------------------------------
# QPSK constellation (Gray-coded, unit power)
# 0: +1+j, 1: -1+j, 2: -1-j, 3: +1-j  (all divided by √2)
# ---------------------------------------------------------------------------
_QPSK = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j], dtype=np.complex128) / np.sqrt(2)

# Gray-coded 2-bit representation of each QPSK symbol (MSB first)
# symbol 0 → 00,  1 → 01,  2 → 11,  3 → 10
_GRAY_BITS = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.int64)


# ---------------------------------------------------------------------------
# Channel helpers
# ---------------------------------------------------------------------------

def _load_channel(mat_path: Path) -> np.ndarray:
    """Load the ``h`` field from a Quadriga .mat file → shape (N,) complex128."""
    try:
        import scipy.io
    except ImportError as exc:
        raise ImportError(
            "quadriga_known_novel requires scipy. Install: pip install scipy"
        ) from exc
    mat = scipy.io.loadmat(str(mat_path))
    cd = mat["channel_data"]
    h = cd["h"][0, 0]   # structured array access → (5000, 1) complex
    return h[:, 0].astype(np.complex128)


def _channel_rms(h: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(h) ** 2)))


def _normalize_channel(h: np.ndarray, reference_rms: float | None = None) -> np.ndarray:
    """Normalise channel by ``reference_rms`` (or its own RMS if None).

    Using a shared ``reference_rms`` computed from the *known* channels
    preserves the true power difference between scenarios: novel channels
    with lower path gain will see proportionally lower effective SNR,
    causing the model to express genuine uncertainty on hard OOD channels.
    """
    rms = reference_rms if reference_rms is not None else _channel_rms(h)
    return h / rms


def _isi_cir(
    h_norm: np.ndarray,
    offset: int,
    isi_taps: int,
    isi_decay: float,
) -> np.ndarray:
    """Build a length-``isi_taps`` channel impulse response (CIR) for ISI.

    The taps are ``isi_taps`` consecutive Quadriga channel coefficients
    starting at ``offset``, weighted by an exponential power-delay profile
    ``isi_decay**k`` so the zeroth (main / current-symbol) tap dominates and
    later delayed taps are progressively attenuated — a physically typical
    decaying multipath profile.  The CIR is normalised to **unit energy**
    (``Σ_k |g_k|² = 1``) so the received signal power — and therefore the
    effective SNR — is identical to the flat-fading (single-tap) case.  The
    resulting drift is purely *structural* (inter-symbol interference), not a
    change in received power or SNR.
    """
    T = len(h_norm)
    taps = h_norm[np.arange(offset, offset + isi_taps) % T]     # (L,) complex
    decay = isi_decay ** np.arange(isi_taps)                    # (L,) real, tap0=1
    cir = taps * decay
    energy = float(np.sum(np.abs(cir) ** 2))
    if energy > 0:
        cir = cir / np.sqrt(energy)                            # unit-energy CIR
    return cir


def _generate_batch(
    h_norm: np.ndarray,
    offset: int,
    batch_size: int,
    snr_db: float,
    rng: np.random.Generator,
    coherence_len: int | None = None,
    isi_taps: int = 1,
    isi_decay: float = 0.6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one QPSK block through a flat-fading or multipath (ISI) channel.

    Parameters
    ----------
    coherence_len :
        If ``None`` or ``>= batch_size``, the batch reads ``batch_size``
        *consecutive* channel taps starting at ``offset`` (a long block that
        marches across the channel realisation, deep fading nulls included).
        If ``< batch_size``, the batch instead samples a short **coherent
        window** of ``coherence_len`` consecutive taps and tiles it up to
        ``batch_size`` symbols.  Every symbol then sits at the same channel
        operating point, so the block does not straddle deep fading nulls the
        way a long contiguous block would, while still drawing ``batch_size``
        independent symbol/noise realisations for a low-variance BER/uncertainty
        estimate.
    isi_taps :
        Number of channel-impulse-response taps.  ``1`` (default) is the
        single-tap flat-fading channel (no memory, existing behaviour).  When
        ``> 1`` the transmitted symbol stream is convolved with a fixed
        ``isi_taps``-tap CIR (see :func:`_isi_cir`), so each received sample
        mixes the current symbol with ``isi_taps - 1`` past symbols —
        inter-symbol interference.  A model trained on the flat channel has
        never seen this structure, so its errors here stem from *ignorance of
        the channel structure* (epistemic) rather than added noise.
    isi_decay :
        Per-tap attenuation of the ISI power-delay profile (only used when
        ``isi_taps > 1``); smaller → milder interference.

    Returns
    -------
    x    : (batch_size, 2) float32 — [Re(rx), Im(rx)] per symbol
    y    : (batch_size,) int64     — QPSK symbol indices 0-3
    bits : (batch_size, 2) int64   — Gray-coded transmitted bits per symbol
    """
    T = len(h_norm)
    snr_lin = 10.0 ** (snr_db / 10.0)
    sigma = 1.0 / np.sqrt(snr_lin)

    if isi_taps and isi_taps > 1:
        L = int(isi_taps)
        cir = _isi_cir(h_norm, offset, L, isi_decay)            # (L,) unit energy
        # Draw B + (L-1) symbols so every output sees a full ISI window; the
        # zeroth CIR tap carries the *current* symbol, later taps the past ones.
        sym_idx = rng.integers(0, 4, size=batch_size + L - 1)
        symbols = _QPSK[sym_idx]
        # conv[n] = Σ_k cir[k]·s[n + (L-1) - k]  (main tap cir[0] → current sym)
        conv = np.convolve(symbols, cir, mode="valid")          # (B,) complex
        noise = (sigma / np.sqrt(2.0)) * (
            rng.standard_normal(batch_size) + 1j * rng.standard_normal(batch_size)
        )
        rx = conv + noise
        cur_idx = sym_idx[L - 1:]                                # (B,) current symbols
        x = np.stack([rx.real, rx.imag], axis=-1).astype(np.float32)
        y = cur_idx.astype(np.int64)
        bits = _GRAY_BITS[cur_idx]
        return x, y, bits

    if coherence_len is None or coherence_len >= batch_size:
        idx = np.arange(offset, offset + batch_size) % T
        h_batch = h_norm[idx]                       # (B,) complex
    else:
        widx = np.arange(offset, offset + coherence_len) % T
        h_win = h_norm[widx]                        # (coherence_len,) complex
        reps = int(np.ceil(batch_size / coherence_len))
        h_batch = np.tile(h_win, reps)[:batch_size]  # (B,) complex

    sym_idx = rng.integers(0, 4, size=batch_size)
    symbols = _QPSK[sym_idx]                        # (B,) complex, |s|²=1

    # Complex AWGN: CN(0, σ²) → each component N(0, σ²/2)
    noise = (sigma / np.sqrt(2.0)) * (
        rng.standard_normal(batch_size) + 1j * rng.standard_normal(batch_size)
    )

    rx = h_batch * symbols + noise                  # (B,) complex
    x = np.stack([rx.real, rx.imag], axis=-1).astype(np.float32)  # (B, 2)
    y = sym_idx.astype(np.int64)
    bits = _GRAY_BITS[sym_idx]                      # (B, 2) — true bit pairs
    return x, y, bits


# ---------------------------------------------------------------------------
# Stream
# ---------------------------------------------------------------------------

@register("dataset", "quadriga_known_novel")
class QuadrigaKnownNovelStream:
    """QPSK stream over 4 Quadriga SISO channels with known/novel phase split.

    Batches are emitted in channel order: two known channels (Highway LOS,
    Highway NLOSv) each contribute ``n_batches_per_channel`` batches, then
    the two novel channels (Urban LOS, Urban NLOS) follow.
    ``spec.extras["novel_start_batch"]`` marks the boundary for detectors
    and plotting code.

    Parameters
    ----------
    data_root :
        Directory containing the four ``channel_0*.mat`` files.
    n_batches_per_channel :
        Number of batches emitted per channel (default 30).
    batch_size :
        QPSK symbols per batch (default 64).
    coherence_len :
        Length (in taps) of the coherent channel window each batch samples.
        When ``batch_size > coherence_len`` the batch tiles a ``coherence_len``
        window up to ``batch_size`` symbols (block-coherent sampling), so a
        large batch gains statistical power without a long contiguous block
        marching across the channel's deep fading nulls.  The window advances
        by ``coherence_len`` per batch, so ``n_batches_per_channel`` batches
        span the first ``coherence_len * n_batches_per_channel`` taps — the
        same fade-free coverage the default 64-symbol setup used.  Default 64
        makes ``batch_size=64`` runs identical to the historical behaviour.
    snr_db :
        Receiver SNR in dB after channel-gain normalisation.
    isi_taps :
        Channel-impulse-response length (inter-symbol interference).  Either a
        scalar applied to all four channels, or a per-channel list
        ``[ch0, ch1, ch2, ch3]``.  ``1`` = flat single-tap channel (default, no
        memory).  ``> 1`` convolves the symbol stream with a multi-tap CIR so
        received samples mix adjacent symbols.  Set e.g. ``[1, 1, 3, 3]`` to
        keep the known Highway channels flat and introduce a 3-tap multipath
        (ISI) drift only on the novel Urban channels — a *structural* drift at
        unchanged SNR, so errors reflect the model's ignorance of the new
        channel structure rather than added noise.
    isi_decay :
        Per-tap attenuation of the ISI power-delay profile (used when
        ``isi_taps > 1``); smaller → milder interference.  Default 0.6.
    seed :
        RNG seed for reproducible symbol generation.
    """

    def __init__(
        self,
        data_root: str = "./database/SISO_channel_quadriga",
        n_batches_per_channel: int = 30,
        batch_size: int = 64,
        coherence_len: int = 64,
        snr_db: float = 15.0,
        channel_snr_db: list | None = None,
        isi_taps: int | list | None = 1,
        isi_decay: float = 0.6,
        seed: int = 0,
    ) -> None:
        self.data_root = str(data_root)
        self.n_batches_per_channel = int(n_batches_per_channel)
        self.batch_size = int(batch_size)
        self.coherence_len = int(coherence_len)
        self.snr_db = float(snr_db)
        self.channel_snr_db = [float(v) for v in channel_snr_db] if channel_snr_db else None
        if isinstance(isi_taps, (list, tuple)):
            self.isi_taps = [int(v) for v in isi_taps]
        else:
            self.isi_taps = [int(isi_taps)] * 4
        self.isi_decay = float(isi_decay)
        self.seed = int(seed)

        mat_files = sorted(Path(self.data_root).glob("channel_*.mat"))
        if len(mat_files) < 4:
            raise FileNotFoundError(
                f"Expected ≥4 channel_*.mat files in {self.data_root}, "
                f"found {len(mat_files)}"
            )
        self._mat_files: List[Path] = list(mat_files[:4])

        n_known, n_novel = 2, 2
        novel_start = self.n_batches_per_channel * n_known
        total = self.n_batches_per_channel * (n_known + n_novel)
        drift_indices = tuple(
            self.n_batches_per_channel * (i + 1)
            for i in range(n_known + n_novel - 1)
        )

        self.spec = StreamSpec(
            name="quadriga_known_novel",
            input_shape=(2,),
            n_classes=4,
            n_batches=total,
            batch_size=self.batch_size,
            drift_indices=drift_indices,
            extras={
                "novel_start_batch": novel_start,
                "channel_names": [
                    "Highway LOS",
                    "Highway NLOSv",
                    "Urban LOS",
                    "Urban NLOS",
                ],
            },
        )

    def __iter__(self) -> Iterator[StreamBatch]:
        rng = np.random.default_rng(self.seed)
        raw = [_load_channel(p) for p in self._mat_files]

        if self.channel_snr_db is not None:
            # Per-channel SNR: normalise each channel by its own RMS so that
            # the noise level alone sets the effective SNR independently.
            channels = [_normalize_channel(h) for h in raw]
            snr_list = self.channel_snr_db
        else:
            # Shared reference RMS — preserves natural path-loss differences.
            ref_rms = float(np.mean([_channel_rms(raw[i]) for i in range(2)]))
            channels = [_normalize_channel(h, ref_rms) for h in raw]
            snr_list = [self.snr_db] * 4

        # Per-batch window advance: for block-coherent sampling step by the
        # coherence length (batches stay in the fade-free prefix); otherwise
        # march by the full block, the historical contiguous behaviour.
        step = (
            self.coherence_len
            if 0 < self.coherence_len < self.batch_size
            else self.batch_size
        )

        batch_idx = 0
        for ch_i, (h_norm, snr_i) in enumerate(zip(channels, snr_list)):
            concept_id = 0 if ch_i < 2 else 1
            isi_i = self.isi_taps[ch_i]
            offset = 0
            for b in range(self.n_batches_per_channel):
                x, y, bits = _generate_batch(
                    h_norm, offset, self.batch_size, snr_i, rng,
                    self.coherence_len,
                    isi_taps=isi_i, isi_decay=self.isi_decay,
                )
                offset = (offset + step) % len(h_norm)
                yield StreamBatch(
                    index=batch_idx,
                    x=x,
                    y=y,
                    concept_id=concept_id,
                    is_drift=(b == 0 and ch_i > 0),
                    extras={"bits": bits},
                )
                batch_idx += 1


# ---------------------------------------------------------------------------
# RNN model
# ---------------------------------------------------------------------------

def _build_rnn_dropout(
    hidden_size: int = 64,
    n_classes: int = 4,
    p_drop: float = 0.3,
):
    """Single-layer LSTM with a post-output dropout layer for MC-Dropout."""
    import torch.nn as nn

    class RNNDropout(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(2, hidden_size, num_layers=1, batch_first=True)
            self.dropout = nn.Dropout(p=p_drop)
            self.head = nn.Linear(hidden_size, n_classes)

        def forward(self, x):
            # x: (1, T, 2)
            out, _ = self.lstm(x)       # (1, T, hidden_size)
            out = self.dropout(out)     # dropout fires here in MC mode
            return self.head(out)       # (1, T, n_classes)

    return RNNDropout()


def _enable_mc_dropout(model) -> None:
    """Set all Dropout layers to train mode; leave everything else in eval."""
    import torch.nn as nn
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


def _pretrain_rnn(
    mat_files: List[Path],
    known_indices: List[int],
    ckpt_path: Path,
    hidden_size: int,
    n_classes: int,
    p_drop: float,
    snr_db: float,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> None:
    """Pretrain on known channels; save weights to ``ckpt_path``."""
    import torch
    from torch import optim

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    raw_known = [_load_channel(mat_files[i]) for i in known_indices]
    ref_rms = float(np.mean([_channel_rms(h) for h in raw_known]))
    channels = [_normalize_channel(h, ref_rms) for h in raw_known]

    model = _build_rnn_dropout(hidden_size, n_classes, p_drop)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()

    for _ in range(epochs):
        for h_norm in channels:
            T = len(h_norm)
            n_blocks = T // batch_size
            block_offsets = rng.permutation(n_blocks) * batch_size
            for offset in block_offsets:
                x_np, y_np, _ = _generate_batch(
                    h_norm, int(offset), batch_size, snr_db, rng
                )
                x_t = torch.from_numpy(x_np).unsqueeze(0)  # (1, T, 2)
                y_t = torch.from_numpy(y_np)                # (T,)
                opt.zero_grad()
                logits = model(x_t).squeeze(0)              # (T, n_classes)
                loss = loss_fn(logits, y_t)
                loss.backward()
                opt.step()

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(ckpt_path))


@register("model", "rnn_mc_dropout_qpsk")
class RNNMCDropoutQPSK(Classifier):
    """Frozen LSTM MC-Dropout model for QPSK detection over Quadriga channels.

    The LSTM is pretrained on the two known (highway) channels and frozen
    during streaming. At inference time the Dropout layer is kept in train
    mode so ``n_samples`` stochastic forward passes produce an
    ``(n_samples, batch_size, 4)`` MC stack consumed by ``mc_dropout``
    uncertainty.

    Parameters
    ----------
    data_root :
        Directory with ``channel_0*.mat`` files (used for pretraining).
    ckpt_path :
        Cache path for pretrained weights.
    hidden_size :
        LSTM hidden dimension.
    p_drop :
        Dropout probability (same at train and MC-inference time).
    n_samples :
        MC forward passes per batch at inference.
    snr_db :
        SNR used to generate pretraining data (should match the stream).
    pretrain_epochs :
        Full passes over the known-channel training blocks.
    pretrain_batch_size :
        Block length (symbols) per gradient step during pretraining.
    pretrain_lr :
        Adam learning rate for pretraining.
    pretrain_seed :
        Seed for pretraining RNG.
    seed :
        Seed for the inference-time MC-Dropout RNG.
    """

    def __init__(
        self,
        data_root: str = "./database/SISO_channel_quadriga",
        ckpt_path: str = "./artifacts/models/rnn_qpsk.pt",
        hidden_size: int = 64,
        p_drop: float = 0.3,
        n_samples: int = 20,
        snr_db: float = 15.0,
        pretrain_epochs: int = 10,
        pretrain_batch_size: int = 64,
        pretrain_lr: float = 1e-3,
        pretrain_seed: int = 0,
        seed: int = 0,
    ) -> None:
        self.data_root = str(data_root)
        self.ckpt_path = str(ckpt_path)
        self.hidden_size = int(hidden_size)
        self.p_drop = float(p_drop)
        self.n_samples = int(n_samples)
        self.snr_db = float(snr_db)
        self.pretrain_epochs = int(pretrain_epochs)
        self.pretrain_batch_size = int(pretrain_batch_size)
        self.pretrain_lr = float(pretrain_lr)
        self.pretrain_seed = int(pretrain_seed)
        self.seed = int(seed)

        self._rnn = None

    def setup(self, spec: StreamSpec) -> None:
        if spec.n_classes != 4:
            raise ValueError(
                f"rnn_mc_dropout_qpsk is 4-way (QPSK); got n_classes={spec.n_classes}"
            )
        if tuple(spec.input_shape) != (2,):
            raise ValueError(
                f"rnn_mc_dropout_qpsk expects input_shape=(2,); got {spec.input_shape}"
            )
        import torch

        mat_files = sorted(Path(self.data_root).glob("channel_*.mat"))
        if len(mat_files) < 4:
            raise FileNotFoundError(
                f"Expected ≥4 channel_*.mat files in {self.data_root}"
            )

        ckpt = Path(self.ckpt_path)
        if not ckpt.exists():
            print(
                f"[rnn_mc_dropout_qpsk] Pretraining on highway channels "
                f"({self.pretrain_epochs} epochs)…"
            )
            _pretrain_rnn(
                mat_files=list(mat_files[:4]),
                known_indices=[0, 1],
                ckpt_path=ckpt,
                hidden_size=self.hidden_size,
                n_classes=4,
                p_drop=self.p_drop,
                snr_db=self.snr_db,
                epochs=self.pretrain_epochs,
                batch_size=self.pretrain_batch_size,
                lr=self.pretrain_lr,
                seed=self.pretrain_seed,
            )

        torch.manual_seed(self.seed)
        model = _build_rnn_dropout(self.hidden_size, 4, self.p_drop)
        model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
        _enable_mc_dropout(model)
        self._rnn = model

    def predict(self, batch: StreamBatch) -> Prediction:
        import torch
        import torch.nn.functional as F

        assert self._rnn is not None
        x = torch.from_numpy(np.ascontiguousarray(batch.x)).float()
        x = x.unsqueeze(0)  # (1, T, 2)

        samples: list[np.ndarray] = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                logits = self._rnn(x).squeeze(0)               # (T, 4)
                probs = F.softmax(logits, dim=-1).numpy().astype(np.float64)
                samples.append(probs)

        mc_probs = np.stack(samples, axis=0)    # (S, B, 4)
        mean_probs = mc_probs.mean(axis=0)      # (B, 4)
        pred_sym = mean_probs.argmax(axis=1)    # (B,) — decoded symbol indices
        pred_bits = _GRAY_BITS[pred_sym]        # (B, 2) — decoded bits (Gray)
        return Prediction(
            probs=mean_probs,
            features=None,
            extras={"mc_probs": mc_probs, "pred_bits": pred_bits},
        )

    def observe(self, batch: StreamBatch, prediction: Prediction) -> None:
        """Frozen backbone — no online update."""
        del batch, prediction


# ---------------------------------------------------------------------------
# Pilot-based supervised drift detector
# ---------------------------------------------------------------------------

@register("detector", "pilot_ddm")
class PilotDDMDetector:
    """Supervised drift detector using pilot-symbol bit errors (DDM).

    Every ``pilot_spacing``-th symbol in the batch is treated as a known
    pilot whose transmitted bits are available at the receiver without
    requiring model-decoded labels.  DDM is fed the per-bit errors on those
    pilot symbols, mirroring the supervisory signal available in pilot-aided
    wireless receivers (e.g. LTE/5G reference symbols used for channel
    quality monitoring).

    Parameters
    ----------
    pilot_spacing :
        One pilot every ``pilot_spacing`` symbols (default 4 → 25% overhead,
        typical for LTE PDCCH density).
    name :
        Display name used in figures and tables.
    """

    def __init__(
        self,
        name: str = "pilot_ddm",
        pilot_spacing: int = 4,
        **river_kwargs,
    ) -> None:
        self.name = name
        self.pilot_spacing = int(pilot_spacing)
        self._river_kwargs = dict(river_kwargs)
        self._ddm = self._make_ddm()

    def _make_ddm(self):
        from river.drift.binary import DDM
        return DDM(**self._river_kwargs)

    def update(
        self,
        batch: StreamBatch,
        prediction: Prediction,
        scores: UncertaintyScores,
    ) -> DetectorEvent:
        pilot_idx = np.arange(0, len(batch.y), self.pilot_spacing)
        true_bits = _GRAY_BITS[batch.y[pilot_idx]]          # (P, 2)
        pred_sym = prediction.probs.argmax(axis=1)
        pred_bits = _GRAY_BITS[pred_sym[pilot_idx]]         # (P, 2)
        bit_errors = (true_bits != pred_bits).ravel()       # (2P,) bool

        alarm = False
        for e in bit_errors:
            self._ddm.update(int(e))
            if self._ddm.drift_detected:
                alarm = True

        pilot_ber = float(bit_errors.mean())
        return DetectorEvent(
            alarm=alarm,
            statistic=pilot_ber,
            diagnostics={"pilot_ber": pilot_ber, "n_pilots": int(len(pilot_idx))},
        )

    def reset(self) -> None:
        self._ddm = self._make_ddm()


@register("detector", "pilot_eddm")
class PilotEDDMDetector:
    """Supervised EDDM on pilot-symbol bit errors.

    Identical to :class:`PilotDDMDetector` but uses EDDM (Early Drift
    Detection Method) which monitors the *distance between errors* rather
    than the raw error rate.  EDDM is more sensitive to gradual drift:
    it alarms when errors start clustering more densely even before the
    mean error rate visibly rises.

    Parameters
    ----------
    pilot_spacing :
        One pilot every ``pilot_spacing`` symbols (default 4 → 25%).
    name :
        Display name for figures and tables.
    """

    def __init__(
        self,
        name: str = "pilot_eddm",
        pilot_spacing: int = 4,
        **river_kwargs,
    ) -> None:
        self.name = name
        self.pilot_spacing = int(pilot_spacing)
        self._river_kwargs = dict(river_kwargs)
        self._eddm = self._make_eddm()

    def _make_eddm(self):
        from river.drift.binary import EDDM
        return EDDM(**self._river_kwargs)

    def update(
        self,
        batch: StreamBatch,
        prediction: Prediction,
        scores: UncertaintyScores,
    ) -> DetectorEvent:
        pilot_idx = np.arange(0, len(batch.y), self.pilot_spacing)
        true_bits = _GRAY_BITS[batch.y[pilot_idx]]
        pred_sym  = prediction.probs.argmax(axis=1)
        pred_bits = _GRAY_BITS[pred_sym[pilot_idx]]
        bit_errors = (true_bits != pred_bits).ravel()

        alarm = False
        for e in bit_errors:
            self._eddm.update(int(e))
            if self._eddm.drift_detected:
                alarm = True

        pilot_ber = float(bit_errors.mean())
        return DetectorEvent(
            alarm=alarm,
            statistic=pilot_ber,
            diagnostics={"pilot_ber": pilot_ber, "n_pilots": int(len(pilot_idx))},
        )

    def reset(self) -> None:
        self._eddm = self._make_eddm()


# ---------------------------------------------------------------------------
# Last-layer Laplace approximation for QPSK
# ---------------------------------------------------------------------------

def _build_rnn_no_dropout(hidden_size: int = 64, n_classes: int = 4):
    """Deterministic LSTM backbone for last-layer Laplace (no dropout)."""
    import torch.nn as nn

    class RNNDeterministic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(2, hidden_size, num_layers=1, batch_first=True)
            self.head = nn.Linear(hidden_size, n_classes)

        def forward(self, x):
            out, _ = self.lstm(x)       # (1, T, hidden_size)
            return self.head(out)       # (1, T, n_classes)

        def embed(self, x):
            out, _ = self.lstm(x)
            return out.squeeze(0)       # (T, hidden_size)

    return RNNDeterministic()


def _pretrain_rnn_no_dropout(
    mat_files: List[Path],
    known_indices: List[int],
    ckpt_path: Path,
    hidden_size: int,
    n_classes: int,
    snr_db: float,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> None:
    """Pretrain deterministic LSTM on known channels (no dropout)."""
    import torch
    from torch import optim

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    raw_known = [_load_channel(mat_files[i]) for i in known_indices]
    ref_rms = float(np.mean([_channel_rms(h) for h in raw_known]))
    channels = [_normalize_channel(h, ref_rms) for h in raw_known]

    model = _build_rnn_no_dropout(hidden_size, n_classes)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()

    for _ in range(epochs):
        for h_norm in channels:
            T = len(h_norm)
            n_blocks = T // batch_size
            block_offsets = rng.permutation(n_blocks) * batch_size
            for offset in block_offsets:
                x_np, y_np, _ = _generate_batch(
                    h_norm, int(offset), batch_size, snr_db, rng
                )
                x_t = torch.from_numpy(x_np).unsqueeze(0)
                y_t = torch.from_numpy(y_np)
                opt.zero_grad()
                logits = model(x_t).squeeze(0)
                loss = loss_fn(logits, y_t)
                loss.backward()
                opt.step()

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(ckpt_path))


def _compute_laplace_posterior(
    model,
    mat_files: List[Path],
    known_indices: List[int],
    snr_db: float,
    batch_size: int,
    prior_precision: float,
    seed: int,
) -> dict:
    """Diagonal GGN Laplace posterior over the last Linear(hidden→K) layer.

    Uses the Generalised Gauss-Newton (GGN) diagonal as the Hessian of the
    softmax cross-entropy loss w.r.t. the head weights:

        H_W[k, j] = Σ_t h_j(t)² · p_k(t) · (1 − p_k(t))
        H_b[k]    = Σ_t p_k(t) · (1 − p_k(t))

    Posterior variance:
        σ²_W[k, j] = 1 / (H_W[k, j] + prior_precision)
        σ²_b[k]    = 1 / (H_b[k]    + prior_precision)

    Returns dict with keys: W_map, b_map, W_var, b_var.
    """
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(seed)
    raw_known = [_load_channel(mat_files[i]) for i in known_indices]
    ref_rms = float(np.mean([_channel_rms(h) for h in raw_known]))
    channels = [_normalize_channel(h, ref_rms) for h in raw_known]

    model.eval()
    n_classes  = model.head.out_features
    hidden_size = model.head.in_features

    H_W = np.zeros((n_classes, hidden_size), dtype=np.float64)
    H_b = np.zeros(n_classes, dtype=np.float64)

    with torch.no_grad():
        for h_norm in channels:
            T = len(h_norm)
            n_blocks = T // batch_size
            for block_i in range(n_blocks):
                x_np, _, _ = _generate_batch(
                    h_norm, block_i * batch_size, batch_size, snr_db, rng
                )
                x_t = torch.from_numpy(x_np).unsqueeze(0)   # (1, B, 2)
                feats  = model.embed(x_t)                    # (B, hidden)
                logits = model.head(feats)                   # (B, K)
                probs  = F.softmax(logits, dim=-1).numpy()   # (B, K)
                feats_np = feats.numpy()                     # (B, hidden)

                pk_1_pk  = probs * (1.0 - probs)            # (B, K)
                H_W     += pk_1_pk.T @ (feats_np ** 2)      # (K, hidden)
                H_b     += pk_1_pk.sum(axis=0)              # (K,)

    W_map = model.head.weight.detach().numpy().astype(np.float64)   # (K, hidden)
    b_map = model.head.bias.detach().numpy().astype(np.float64)     # (K,)

    return {
        "W_map": W_map,
        "b_map": b_map,
        "W_var": 1.0 / (H_W + prior_precision),
        "b_var": 1.0 / (H_b + prior_precision),
    }


@register("model", "rnn_laplace_qpsk")
class RNNLaplaceQPSK(Classifier):
    """Frozen LSTM + diagonal last-layer Laplace approximation for QPSK.

    Contrast with :class:`RNNMCDropoutQPSK`:
    * **MC-Dropout**: stochastic dropout masks at inference → uncertainty
      comes from randomly ablating hidden units.
    * **Last-layer Laplace**: deterministic backbone, Gaussian posterior
      over the final Linear(64→4) head weights fitted via the diagonal GGN
      approximation → uncertainty comes from posterior weight uncertainty.

    At inference, ``n_samples`` weight matrices W_s ~ N(W_MAP, diag(σ²_W))
    produce an (n_samples, B, 4) probability stack stored in
    ``prediction.extras['mc_probs']``.  The shared ``mc_dropout`` uncertainty
    estimator then decomposes it into total entropy and epistemic MI as usual.

    Parameters
    ----------
    prior_precision :
        Precision of the zero-mean isotropic Gaussian prior over head weights.
        Higher → tighter posterior → less uncertainty inflation.
    """

    def __init__(
        self,
        data_root: str = "./database/SISO_channel_quadriga",
        ckpt_path: str = "./artifacts/models/rnn_laplace_qpsk.pt",
        hidden_size: int = 64,
        n_samples: int = 20,
        prior_precision: float = 1.0,
        snr_db: float = 28.0,
        pretrain_epochs: int = 10,
        pretrain_batch_size: int = 64,
        pretrain_lr: float = 1e-3,
        pretrain_seed: int = 0,
        seed: int = 0,
    ) -> None:
        self.data_root = str(data_root)
        self.ckpt_path = str(ckpt_path)
        self.hidden_size = int(hidden_size)
        self.n_samples = int(n_samples)
        self.prior_precision = float(prior_precision)
        self.snr_db = float(snr_db)
        self.pretrain_epochs = int(pretrain_epochs)
        self.pretrain_batch_size = int(pretrain_batch_size)
        self.pretrain_lr = float(pretrain_lr)
        self.pretrain_seed = int(pretrain_seed)
        self.seed = int(seed)

        self._rnn = None
        self._posterior: dict | None = None

    def setup(self, spec: StreamSpec) -> None:
        if spec.n_classes != 4:
            raise ValueError(
                f"rnn_laplace_qpsk is 4-way (QPSK); got n_classes={spec.n_classes}"
            )
        if tuple(spec.input_shape) != (2,):
            raise ValueError(
                f"rnn_laplace_qpsk expects input_shape=(2,); got {spec.input_shape}"
            )
        import torch

        mat_files = sorted(Path(self.data_root).glob("channel_*.mat"))
        if len(mat_files) < 4:
            raise FileNotFoundError(
                f"Expected ≥4 channel_*.mat files in {self.data_root}"
            )

        ckpt = Path(self.ckpt_path)
        if not ckpt.exists():
            print(
                f"[rnn_laplace_qpsk] Pretraining deterministic LSTM on highway channels "
                f"({self.pretrain_epochs} epochs)…"
            )
            _pretrain_rnn_no_dropout(
                mat_files=list(mat_files[:4]),
                known_indices=[0, 1],
                ckpt_path=ckpt,
                hidden_size=self.hidden_size,
                n_classes=4,
                snr_db=self.snr_db,
                epochs=self.pretrain_epochs,
                batch_size=self.pretrain_batch_size,
                lr=self.pretrain_lr,
                seed=self.pretrain_seed,
            )

        torch.manual_seed(self.seed)
        model = _build_rnn_no_dropout(self.hidden_size, 4)
        model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
        model.eval()
        self._rnn = model

        print("[rnn_laplace_qpsk] Computing diagonal GGN Laplace posterior over head…")
        self._posterior = _compute_laplace_posterior(
            model=self._rnn,
            mat_files=list(mat_files[:4]),
            known_indices=[0, 1],
            snr_db=self.snr_db,
            batch_size=self.pretrain_batch_size,
            prior_precision=self.prior_precision,
            seed=self.pretrain_seed,
        )
        print("[rnn_laplace_qpsk] Laplace posterior ready.")

    def predict(self, batch: StreamBatch) -> Prediction:
        import torch

        assert self._rnn is not None and self._posterior is not None
        rng = np.random.default_rng(self.seed + batch.index)

        x = torch.from_numpy(np.ascontiguousarray(batch.x)).float().unsqueeze(0)
        with torch.no_grad():
            feats_np = self._rnn.embed(x).numpy().astype(np.float64)  # (B, hidden)

        W_map = self._posterior["W_map"]        # (K, hidden)
        b_map = self._posterior["b_map"]        # (K,)
        W_std = np.sqrt(self._posterior["W_var"])
        b_std = np.sqrt(self._posterior["b_var"])

        samples: list[np.ndarray] = []
        for _ in range(self.n_samples):
            W_s = W_map + rng.standard_normal(W_map.shape) * W_std
            b_s = b_map + rng.standard_normal(b_map.shape) * b_std
            logits = feats_np @ W_s.T + b_s    # (B, K)
            logits -= logits.max(axis=1, keepdims=True)
            exp_l  = np.exp(logits)
            probs  = exp_l / exp_l.sum(axis=1, keepdims=True)
            samples.append(probs.astype(np.float64))

        mc_probs   = np.stack(samples, axis=0)   # (S, B, 4)
        mean_probs = mc_probs.mean(axis=0)        # (B, 4)
        pred_sym   = mean_probs.argmax(axis=1)
        pred_bits  = _GRAY_BITS[pred_sym]

        return Prediction(
            probs=mean_probs,
            features=None,
            extras={"mc_probs": mc_probs, "pred_bits": pred_bits},
        )

    def observe(self, batch: StreamBatch, prediction: Prediction) -> None:
        """Frozen backbone and head — no online update."""
        del batch, prediction


# ---------------------------------------------------------------------------
# Full-Hessian Laplace approximation for QPSK
# ---------------------------------------------------------------------------

def _compute_full_hessian_posterior(
    model,
    mat_files: List[Path],
    known_indices: List[int],
    snr_db: float,
    batch_size: int,
    prior_precision: float,
    seed: int,
) -> dict:
    """Full GGN Hessian Laplace posterior over the last Linear(hidden→K) layer.

    For K=4, hidden=64 the parameter vector θ = [vec(W), b] has P=260 entries.
    The full (260×260) GGN matrix captures all cross-weight correlations ignored
    by the diagonal approximation, giving a richer posterior geometry.

    The GGN is accumulated in blocks using the Jacobian's block structure:

        H_WW[k₁·H+h₁, k₂·H+h₂] += Σ_i Λ_i[k₁,k₂] · f_i[h₁] · f_i[h₂]
        H_Wb[k₁·H+h₁, K·H+k₂]  += Σ_i Λ_i[k₁,k₂] · f_i[h₁]
        H_bb[K·H+k₁,  K·H+k₂]  += Σ_i Λ_i[k₁,k₂]

    where Λ_i = diag(p_i) − p_i pᵢᵀ is the softmax Hessian (4×4) and f_i is
    the LSTM feature vector (64-dim).

    Returns dict with keys: theta_map (P,), L (P×P Cholesky of posterior cov),
    K, H_size.
    """
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(seed)
    raw_known = [_load_channel(mat_files[i]) for i in known_indices]
    ref_rms = float(np.mean([_channel_rms(h) for h in raw_known]))
    channels = [_normalize_channel(h, ref_rms) for h in raw_known]

    model.eval()
    K = model.head.out_features     # 4
    H_size = model.head.in_features  # 64
    P = K * H_size + K              # 260

    H_full = np.zeros((P, P), dtype=np.float64)

    with torch.no_grad():
        for h_norm in channels:
            T = len(h_norm)
            n_blocks = T // batch_size
            for block_i in range(n_blocks):
                x_np, _, _ = _generate_batch(
                    h_norm, block_i * batch_size, batch_size, snr_db, rng
                )
                x_t = torch.from_numpy(x_np).unsqueeze(0)
                feats = model.embed(x_t)          # (B, H_size)
                logits = model.head(feats)        # (B, K)
                probs = F.softmax(logits, dim=-1).numpy().astype(np.float64)
                feats_np = feats.numpy().astype(np.float64)

                # Λ_batch[i, k1, k2] = p_{k1}(i)·(δ_{k1k2} − p_{k2}(i))
                Lambda = np.einsum("ik,kj->ikj", probs, np.eye(K)) \
                       - np.einsum("ik,ij->ikj", probs, probs)  # (B, K, K)

                # W-W block: (K, H, K, H) → (K·H, K·H)
                HWW = np.einsum("ikl,ih,ig->khlg", Lambda, feats_np, feats_np,
                                optimize=True)
                H_full[:K*H_size, :K*H_size] += HWW.reshape(K * H_size, K * H_size)

                # W-b block: (K, H, K) → (K·H, K); b-W is its transpose
                HWb = np.einsum("ikl,ih->khl", Lambda, feats_np)  # (K, H, K)
                Wb = HWb.reshape(K * H_size, K)
                H_full[:K*H_size, K*H_size:] += Wb
                H_full[K*H_size:, :K*H_size] += Wb.T

                # b-b block: (K, K)
                H_full[K*H_size:, K*H_size:] += Lambda.sum(axis=0)

    # Posterior precision = GGN + λI; invert to get covariance
    Sigma = np.linalg.inv(H_full + prior_precision * np.eye(P))

    # Cholesky for efficient posterior sampling: Sigma = L Lᵀ
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(Sigma + 1e-8 * np.eye(P))

    W_map = model.head.weight.detach().numpy().astype(np.float64)  # (K, H_size)
    b_map = model.head.bias.detach().numpy().astype(np.float64)    # (K,)
    theta_map = np.concatenate([W_map.ravel(), b_map])             # (P,)

    return {"theta_map": theta_map, "L": L, "K": K, "H_size": H_size}


@register("model", "rnn_full_hessian_qpsk")
class RNNFullHessianLaplaceQPSK(Classifier):
    """Frozen LSTM + full-Hessian last-layer Laplace approximation for QPSK.

    Contrast with :class:`RNNLaplaceQPSK` (diagonal GGN):
    * **Diagonal GGN**: stores 260 independent variances; ignores all
      off-diagonal weight correlations → tends to over-inflate uncertainty.
    * **Full Hessian GGN**: stores and inverts the full 260×260 matrix;
      captures all cross-weight correlations for a geometrically accurate
      ellipsoidal posterior over the head → tighter, better-calibrated samples.

    Shares the same deterministic backbone checkpoint as
    :class:`RNNLaplaceQPSK` (``rnn_laplace_qpsk.pt``).
    """

    def __init__(
        self,
        data_root: str = "./database/SISO_channel_quadriga",
        ckpt_path: str = "./artifacts/models/rnn_laplace_qpsk.pt",
        hidden_size: int = 64,
        n_samples: int = 20,
        prior_precision: float = 1.0,
        snr_db: float = 28.0,
        pretrain_epochs: int = 10,
        pretrain_batch_size: int = 64,
        pretrain_lr: float = 1e-3,
        pretrain_seed: int = 0,
        seed: int = 0,
    ) -> None:
        self.data_root = str(data_root)
        self.ckpt_path = str(ckpt_path)
        self.hidden_size = int(hidden_size)
        self.n_samples = int(n_samples)
        self.prior_precision = float(prior_precision)
        self.snr_db = float(snr_db)
        self.pretrain_epochs = int(pretrain_epochs)
        self.pretrain_batch_size = int(pretrain_batch_size)
        self.pretrain_lr = float(pretrain_lr)
        self.pretrain_seed = int(pretrain_seed)
        self.seed = int(seed)

        self._rnn = None
        self._posterior: dict | None = None

    def setup(self, spec: StreamSpec) -> None:
        if spec.n_classes != 4:
            raise ValueError(
                f"rnn_full_hessian_qpsk is 4-way (QPSK); got n_classes={spec.n_classes}"
            )
        if tuple(spec.input_shape) != (2,):
            raise ValueError(
                f"rnn_full_hessian_qpsk expects input_shape=(2,); got {spec.input_shape}"
            )
        import torch

        mat_files = sorted(Path(self.data_root).glob("channel_*.mat"))
        if len(mat_files) < 4:
            raise FileNotFoundError(
                f"Expected ≥4 channel_*.mat files in {self.data_root}"
            )

        ckpt = Path(self.ckpt_path)
        if not ckpt.exists():
            print(
                f"[rnn_full_hessian_qpsk] Pretraining deterministic LSTM "
                f"({self.pretrain_epochs} epochs)…"
            )
            _pretrain_rnn_no_dropout(
                mat_files=list(mat_files[:4]),
                known_indices=[0, 1],
                ckpt_path=ckpt,
                hidden_size=self.hidden_size,
                n_classes=4,
                snr_db=self.snr_db,
                epochs=self.pretrain_epochs,
                batch_size=self.pretrain_batch_size,
                lr=self.pretrain_lr,
                seed=self.pretrain_seed,
            )

        torch.manual_seed(self.seed)
        model = _build_rnn_no_dropout(self.hidden_size, 4)
        model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
        model.eval()
        self._rnn = model

        print("[rnn_full_hessian_qpsk] Computing full-Hessian GGN Laplace posterior…")
        self._posterior = _compute_full_hessian_posterior(
            model=self._rnn,
            mat_files=list(mat_files[:4]),
            known_indices=[0, 1],
            snr_db=self.snr_db,
            batch_size=self.pretrain_batch_size,
            prior_precision=self.prior_precision,
            seed=self.pretrain_seed,
        )
        print("[rnn_full_hessian_qpsk] Full-Hessian posterior ready.")

    def predict(self, batch: StreamBatch) -> Prediction:
        import torch

        assert self._rnn is not None and self._posterior is not None
        rng = np.random.default_rng(self.seed + batch.index)

        x = torch.from_numpy(np.ascontiguousarray(batch.x)).float().unsqueeze(0)
        with torch.no_grad():
            feats_np = self._rnn.embed(x).numpy().astype(np.float64)  # (B, H_size)

        theta_map = self._posterior["theta_map"]  # (P,)
        L = self._posterior["L"]                  # (P, P) Cholesky
        K = self._posterior["K"]
        H_size = self._posterior["H_size"]
        P = K * H_size + K

        samples: list[np.ndarray] = []
        for _ in range(self.n_samples):
            theta_s = theta_map + L @ rng.standard_normal(P)
            W_s = theta_s[:K * H_size].reshape(K, H_size)
            b_s = theta_s[K * H_size:]
            logits = feats_np @ W_s.T + b_s       # (B, K)
            logits -= logits.max(axis=1, keepdims=True)
            exp_l = np.exp(logits)
            probs = exp_l / exp_l.sum(axis=1, keepdims=True)
            samples.append(probs.astype(np.float64))

        mc_probs = np.stack(samples, axis=0)       # (S, B, 4)
        mean_probs = mc_probs.mean(axis=0)
        pred_sym = mean_probs.argmax(axis=1)
        pred_bits = _GRAY_BITS[pred_sym]

        return Prediction(
            probs=mean_probs,
            features=None,
            extras={"mc_probs": mc_probs, "pred_bits": pred_bits},
        )

    def observe(self, batch: StreamBatch, prediction: Prediction) -> None:
        """Frozen backbone and head — no online update."""
        del batch, prediction


# ---------------------------------------------------------------------------
# Deep-dropout Laplace: 2-layer LSTM body dropout + diagonal Laplace head
# ---------------------------------------------------------------------------

def _build_rnn_deep_dropout(
    hidden_size: int = 64,
    n_classes: int = 4,
    p_drop: float = 0.3,
):
    """Two-layer LSTM with inter-layer body dropout + post-LSTM head dropout.

    Dropout fires at two points:
    * **Between LSTM layers** (``num_layers=2, dropout=p_drop``) — stochastic
      feature extraction; each MC pass sees a different hidden trajectory.
    * **After the LSTM output** (``nn.Dropout``) — same as the single-layer
      MC-Dropout model; adds a second source of stochasticity before the head.

    ``embed()`` disables both dropout sources (call in eval mode) for clean
    Laplace Hessian accumulation.  ``embed_stochastic()`` requires the caller
    to put the model in train mode so both dropout paths are active.
    """
    import torch.nn as nn

    class RNNDeepDropout(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                2, hidden_size, num_layers=2,
                dropout=p_drop, batch_first=True,
            )
            self.dropout = nn.Dropout(p=p_drop)
            self.head = nn.Linear(hidden_size, n_classes)

        def forward(self, x):
            out, _ = self.lstm(x)       # inter-layer dropout when in train mode
            out = self.dropout(out)
            return self.head(out)

        def embed(self, x):
            """Deterministic features (eval mode): both dropout sources off."""
            out, _ = self.lstm(x)
            return out.squeeze(0)       # (T, hidden_size)

        def embed_stochastic(self, x):
            """Stochastic features (train mode): both dropout sources active."""
            out, _ = self.lstm(x)       # inter-layer dropout fires
            out = self.dropout(out)     # post-LSTM dropout fires
            return out.squeeze(0)       # (T, hidden_size)

    return RNNDeepDropout()


def _pretrain_rnn_deep_dropout(
    mat_files: List[Path],
    known_indices: List[int],
    ckpt_path: Path,
    hidden_size: int,
    n_classes: int,
    p_drop: float,
    snr_db: float,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> None:
    """Pretrain the 2-layer deep-dropout LSTM; save weights to ``ckpt_path``."""
    import torch
    from torch import optim

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    raw_known = [_load_channel(mat_files[i]) for i in known_indices]
    ref_rms = float(np.mean([_channel_rms(h) for h in raw_known]))
    channels = [_normalize_channel(h, ref_rms) for h in raw_known]

    model = _build_rnn_deep_dropout(hidden_size, n_classes, p_drop)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        for h_norm in channels:
            T = len(h_norm)
            n_blocks = T // batch_size
            block_offsets = rng.permutation(n_blocks) * batch_size
            for offset in block_offsets:
                x_np, y_np, _ = _generate_batch(
                    h_norm, int(offset), batch_size, snr_db, rng
                )
                x_t = torch.from_numpy(x_np).unsqueeze(0)
                y_t = torch.from_numpy(y_np)
                opt.zero_grad()
                logits = model(x_t).squeeze(0)
                loss = loss_fn(logits, y_t)
                loss.backward()
                opt.step()
        if (epoch + 1) % 5 == 0:
            print(f"  [deep_laplace pretrain] epoch {epoch + 1}/{epochs}")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(ckpt_path))
    print(f"  [deep_laplace pretrain] saved → {ckpt_path}")


@register("model", "rnn_deep_laplace_qpsk")
class RNNDeepDropoutLaplaceQPSK(Classifier):
    """Deep-dropout LSTM + diagonal last-layer Laplace for QPSK detection.

    Combines two sources of epistemic uncertainty that the individual methods
    lack:

    * **Body dropout** (inter-layer, 2-layer LSTM): each MC pass sees a
      different hidden-state trajectory → feature vectors ``h_s`` differ
      between samples even for the same input.  On novel channels the LSTM
      has no robust representation, so ``h_s`` vary *more* than on known
      channels.
    * **Head weight sampling** (diagonal GGN Laplace): each pass uses a
      weight draw ``W_s ~ N(W_MAP, diag(σ²_W))``.

    Together these two stochastic sources drive ``H[mean_p] - mean(H[p_s])``
    up on novel channels while keeping it low on known channels — exactly the
    signal the uncertainty-based drift detectors need.

    The Laplace posterior is fitted with dropout disabled (eval mode) so the
    curvature reflects the deterministic MAP, and inference re-enables all
    dropout so both uncertainty sources contribute to each MC sample.
    """

    def __init__(
        self,
        data_root: str = "./database/SISO_channel_quadriga",
        ckpt_path: str = "./artifacts/models/rnn_qpsk_deep_laplace.pt",
        hidden_size: int = 64,
        p_drop: float = 0.3,
        n_samples: int = 20,
        prior_precision: float = 1.0,
        snr_db: float = 15.0,
        pretrain_epochs: int = 10,
        pretrain_batch_size: int = 64,
        pretrain_lr: float = 1e-3,
        pretrain_seed: int = 0,
        seed: int = 0,
    ) -> None:
        self.data_root = str(data_root)
        self.ckpt_path = str(ckpt_path)
        self.hidden_size = int(hidden_size)
        self.p_drop = float(p_drop)
        self.n_samples = int(n_samples)
        self.prior_precision = float(prior_precision)
        self.snr_db = float(snr_db)
        self.pretrain_epochs = int(pretrain_epochs)
        self.pretrain_batch_size = int(pretrain_batch_size)
        self.pretrain_lr = float(pretrain_lr)
        self.pretrain_seed = int(pretrain_seed)
        self.seed = int(seed)

        self._rnn = None
        self._posterior: dict | None = None

    def setup(self, spec: StreamSpec) -> None:
        if spec.n_classes != 4:
            raise ValueError(
                f"rnn_deep_laplace_qpsk is 4-way (QPSK); got n_classes={spec.n_classes}"
            )
        if tuple(spec.input_shape) != (2,):
            raise ValueError(
                f"rnn_deep_laplace_qpsk expects input_shape=(2,); got {spec.input_shape}"
            )
        import torch

        mat_files = sorted(Path(self.data_root).glob("channel_*.mat"))
        if len(mat_files) < 4:
            raise FileNotFoundError(
                f"Expected ≥4 channel_*.mat files in {self.data_root}"
            )

        ckpt = Path(self.ckpt_path)
        if not ckpt.exists():
            print("[rnn_deep_laplace_qpsk] Pretraining 2-layer LSTM with body dropout…")
            _pretrain_rnn_deep_dropout(
                mat_files=list(mat_files[:4]),
                known_indices=[0, 1],
                ckpt_path=ckpt,
                hidden_size=self.hidden_size,
                n_classes=4,
                p_drop=self.p_drop,
                snr_db=self.snr_db,
                epochs=self.pretrain_epochs,
                batch_size=self.pretrain_batch_size,
                lr=self.pretrain_lr,
                seed=self.pretrain_seed,
            )

        torch.manual_seed(self.seed)
        model = _build_rnn_deep_dropout(self.hidden_size, 4, self.p_drop)
        model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
        model.eval()
        self._rnn = model

        print("[rnn_deep_laplace_qpsk] Computing diagonal GGN Laplace posterior over head…")
        self._posterior = _compute_laplace_posterior(
            model=self._rnn,
            mat_files=list(mat_files[:4]),
            known_indices=[0, 1],
            snr_db=self.snr_db,
            batch_size=self.pretrain_batch_size,
            prior_precision=self.prior_precision,
            seed=self.pretrain_seed,
        )
        print("[rnn_deep_laplace_qpsk] Posterior ready.")

    def predict(self, batch: StreamBatch) -> Prediction:
        import torch

        assert self._rnn is not None and self._posterior is not None
        rng = np.random.default_rng(self.seed + batch.index)

        x = torch.from_numpy(np.ascontiguousarray(batch.x)).float().unsqueeze(0)

        W_map = self._posterior["W_map"]        # (K, hidden)
        b_map = self._posterior["b_map"]        # (K,)
        W_std = np.sqrt(self._posterior["W_var"])
        b_std = np.sqrt(self._posterior["b_var"])

        # Enable all dropout for stochastic body features
        self._rnn.train()
        samples: list[np.ndarray] = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                feats_np = self._rnn.embed_stochastic(x).numpy().astype(np.float64)
                W_s = W_map + rng.standard_normal(W_map.shape) * W_std
                b_s = b_map + rng.standard_normal(b_map.shape) * b_std
                logits = feats_np @ W_s.T + b_s         # (B, K)
                logits -= logits.max(axis=1, keepdims=True)
                exp_l  = np.exp(logits)
                probs  = exp_l / exp_l.sum(axis=1, keepdims=True)
                samples.append(probs.astype(np.float64))

        mc_probs   = np.stack(samples, axis=0)   # (S, B, 4)
        mean_probs = mc_probs.mean(axis=0)
        pred_sym   = mean_probs.argmax(axis=1)
        pred_bits  = _GRAY_BITS[pred_sym]

        return Prediction(
            probs=mean_probs,
            features=None,
            extras={"mc_probs": mc_probs, "pred_bits": pred_bits},
        )

    def observe(self, batch: StreamBatch, prediction: Prediction) -> None:
        del batch, prediction
