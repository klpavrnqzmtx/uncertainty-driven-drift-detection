"""QuaDRiGa multi-tap channel-drift experiment.

Unlike :mod:`quadriga` (a flat-fading, single-gain-per-symbol stream), this
module models **true frequency-selective (ISI) channels**: each channel
realisation is a discrete-time complex impulse response ``h = [h0, h1, ...,
hL]`` and the transmitted QPSK stream is *convolved* with it, so each received
sample mixes the current symbol with up to ``L-1`` past symbols.

Data (``database/QuaDRiGa/<scenario>_taps.mat``)
-----------------------------------------------
Each file holds

* ``taps``        — complex ``[n_realizations, max_taps]``; row ``i`` is one
  independent QuaDRiGa channel realisation, zero-padded to ``max_taps``.
* ``tap_lengths`` — ``[n_realizations]``; the *valid* tap count of row ``i``,
  so the true CIR is ``h = taps[i, :tap_lengths[i]]``.

The taps are already power-normalised in MATLAB (``Σ|h|² ≈ 1`` per row), so we
do **not** renormalise per scenario — the drift we want to isolate is the
change in channel *structure* (delay spread / ISI), not received power or SNR.

Experiment
----------
The receiver is trained **only on Highway LOS**, then the *same frozen model*
is evaluated on a sequence of increasingly distant propagation scenarios:

    Highway LOS  →  Highway NLOSv  →  Urban LOS  →  Urban NLOS  →  Indoor NLOS
    (reference)     (intermediate)    (moderate)    (stronger)     (strong OOD)

ordered by growing delay spread.  Every scenario uses the **same SNR**
(``snr_db``, default 15 dB) so error rate and epistemic uncertainty reflect
channel-distribution drift alone.

Registered components
---------------------
* ``quadriga_channel_drift``     — DatasetStream, input_shape=(2,), n_classes=4
* ``rnn_equalizer_mc_dropout``   — LSTM equaliser, MC-Dropout uncertainty
* ``rnn_equalizer_laplace``      — LSTM equaliser, diagonal last-layer Laplace
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from uncertainty_driven_drift.components.quadriga import (
    _GRAY_BITS,
    _QPSK,
    _build_rnn_dropout,
    _build_rnn_no_dropout,
    _enable_mc_dropout,
)
from uncertainty_driven_drift.data.base import DatasetStream, StreamBatch, StreamSpec
from uncertainty_driven_drift.models.base import Classifier, Prediction
from uncertainty_driven_drift.registry import register

# Canonical drift order (near → far from the Highway-LOS training distribution)
# and the display names used by the plotting code.
_DRIFT_ORDER: List[Tuple[str, str]] = [
    ("highway_los", "Highway LOS"),
    ("highway_nlosv", "Highway NLOSv"),
    ("urban_los", "Urban LOS"),
    ("urban_nlos", "Urban NLOS"),
    ("indoor_los", "Indoor LOS"),
    ("indoor_nlos", "Indoor NLOS"),
]
_DISPLAY = dict(_DRIFT_ORDER)


# ---------------------------------------------------------------------------
# Data loading + transmission
# ---------------------------------------------------------------------------

def _load_taps(data_root: Path, scenario: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load ``taps`` (complex ``[N, Lmax]``) and ``tap_lengths`` (int ``[N]``)."""
    try:
        import scipy.io
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "quadriga_channel_drift requires scipy. Install: pip install scipy"
        ) from exc
    path = data_root / f"{scenario}_taps.mat"
    if not path.exists():
        raise FileNotFoundError(f"Missing QuaDRiGa scenario file: {path}")
    mat = scipy.io.loadmat(str(path))
    taps = np.asarray(mat["taps"], dtype=np.complex128)
    tap_lengths = np.asarray(mat["tap_lengths"]).ravel().astype(int)
    if taps.shape[0] != tap_lengths.shape[0]:
        raise ValueError(
            f"{path.name}: taps rows ({taps.shape[0]}) != tap_lengths "
            f"({tap_lengths.shape[0]})"
        )
    return taps, tap_lengths


def _cir(taps: np.ndarray, tap_lengths: np.ndarray, i: int) -> np.ndarray:
    """Recover the valid impulse response of realisation ``i`` (drop zero-pad)."""
    L = int(tap_lengths[i])
    L = max(1, L)
    return taps[i, :L]


def _transmit(
    h: np.ndarray,
    batch_size: int,
    snr_db: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Send ``batch_size`` **differential-QPSK** symbols through channel ``h``.

    Each channel realisation applies an unknown constant complex gain (random
    phase/magnitude), and absolute QPSK is 90°-ambiguous, so a receiver that
    must *generalise across realisations* cannot recover absolute symbols
    blindly.  We therefore encode information differentially: the info symbol
    ``d[n] ∈ {0,1,2,3}`` sets the phase *increment* between consecutive
    transmitted symbols, ``a[n] = (a[n-1] + d[n]) mod 4``, ``s[n] = QPSK[a[n]]``.
    A constant channel gain cancels in the phase difference, so a flat channel
    decodes ``d`` perfectly; multi-tap ISI (growing delay spread) progressively
    corrupts the differential relationship — the drift we want to isolate.

    The stream is passed through the channel by convolution,
    ``rx[n] = Σ_k h[k]·s[n-k] + w[n]``, with the channel used **as stored**
    (already ``Σ|h|²≈1``) and AWGN ``σ² = 10^(-snr_db/10)`` added after it, so the
    effective SNR is ``snr_db`` and identical across scenarios.  An ``L-1`` symbol
    ISI guard (full memory) plus one differential-reference sample precede the
    ``batch_size`` scored outputs, so there is no per-batch startup transient.

    Returns ``x`` ``(B, 2)`` scored received [Re, Im], ``y`` ``(B,)`` info-symbol
    indices, ``bits`` ``(B, 2)`` Gray-coded info bits, and ``ref`` ``(2,)`` the
    received sample immediately preceding the first scored output (the
    differential reference the receiver conditions on).
    """
    L = len(h)
    G = L - 1                              # ISI guard so every output has full memory
    n_abs = G + batch_size + 1             # + reference sample before first scored one
    d = rng.integers(0, 4, size=n_abs)     # differential info symbols (d[0] = random start)
    a = np.cumsum(d) % 4                   # absolute (differentially-encoded) symbols
    s = _QPSK[a]
    full = np.convolve(s, h, mode="full")  # (n_abs + L - 1,)

    # Received samples at times t = G .. G+B  (index 0 = differential reference,
    # indices 1..B = the scored outputs), each with full ISI history.
    rx = full[G: G + batch_size + 1].copy()               # (B + 1,)
    snr_lin = 10.0 ** (snr_db / 10.0)
    sigma = 1.0 / np.sqrt(snr_lin)
    rx = rx + (sigma / np.sqrt(2.0)) * (
        rng.standard_normal(batch_size + 1) + 1j * rng.standard_normal(batch_size + 1)
    )

    ref = np.array([rx[0].real, rx[0].imag], dtype=np.float32)   # (2,)
    rxs = rx[1:]                                                  # (B,) scored
    x = np.stack([rxs.real, rxs.imag], axis=-1).astype(np.float32)
    y = d[G + 1: G + batch_size + 1].astype(np.int64)            # info at scored times
    bits = _GRAY_BITS[y]
    return x, y, bits, ref


def _lstm_input(x_np: np.ndarray, ref: np.ndarray):
    """Build the LSTM input tensor ``(1, B+1, 2)`` = differential reference
    prepended to the scored received samples.  The model runs on ``B+1``
    steps and drops the first output (aligned to the reference)."""
    import torch
    arr = np.concatenate([ref[None, :], x_np], axis=0)          # (B+1, 2)
    return torch.from_numpy(np.ascontiguousarray(arr)).float().unsqueeze(0)


# ---------------------------------------------------------------------------
# Stream
# ---------------------------------------------------------------------------

@register("dataset", "quadriga_channel_drift")
class QuadrigaChannelDriftStream(DatasetStream):
    """QPSK stream over multi-tap QuaDRiGa channels with escalating drift.

    Batches are emitted scenario-by-scenario in ``scenarios`` order; each
    scenario contributes ``n_batches_per_scenario`` batches, and each batch
    transmits ``batch_size`` symbols through one randomly drawn channel
    realisation of that scenario.  ``spec.extras['novel_start_batch']`` marks
    the end of the (in-distribution) reference scenario.

    Parameters
    ----------
    data_root :
        Directory holding ``<scenario>_taps.mat`` files.
    scenarios :
        Scenario keys in the order to present them (default: the five-scenario
        escalating-drift order, Highway LOS first).
    train_scenario :
        Scenario(s) the receiver was trained on — a single name or a list.
        Recorded in ``spec.extras`` and used to place the in-distribution /
        drift boundary: the leading run of scenarios that are all in the
        training set is treated as in-distribution, and ``novel_start_batch``
        marks the first scenario beyond it (default ``highway_los``).
    n_batches_per_scenario, batch_size :
        Batches per scenario and symbols per batch.
    snr_db :
        Receiver SNR in dB, identical for every scenario.
    seed :
        RNG seed for realisation selection, symbols and noise.
    """

    def __init__(
        self,
        data_root: str = "./database/QuaDRiGa",
        scenarios: Optional[Sequence[str]] = None,
        train_scenario="highway_los",
        n_batches_per_scenario: int = 30,
        batch_size: int = 512,
        snr_db: float = 15.0,
        seed: int = 0,
    ) -> None:
        self.data_root = Path(data_root)
        self.scenarios = list(scenarios) if scenarios else [k for k, _ in _DRIFT_ORDER]
        self.train_scenarios = _as_scenario_list(train_scenario)
        self.n_batches_per_scenario = int(n_batches_per_scenario)
        self.batch_size = int(batch_size)
        self.snr_db = float(snr_db)
        self.seed = int(seed)

        self._taps = {}
        self._lens = {}
        for s in self.scenarios:
            self._taps[s], self._lens[s] = _load_taps(self.data_root, s)

        n_sec = len(self.scenarios)
        total = n_sec * self.n_batches_per_scenario
        drift_indices = tuple(
            self.n_batches_per_scenario * (i + 1) for i in range(n_sec - 1)
        )
        # In-distribution = the leading run of scenarios all in the training set;
        # the drift boundary is the first scenario beyond it.
        train_set = set(self.train_scenarios)
        n_ref = 0
        for s in self.scenarios:
            if s in train_set:
                n_ref += 1
            else:
                break
        n_ref = max(1, n_ref)
        novel_start = n_ref * self.n_batches_per_scenario

        self.spec = StreamSpec(
            name="quadriga_channel_drift",
            input_shape=(2,),
            n_classes=4,
            n_batches=total,
            batch_size=self.batch_size,
            drift_indices=drift_indices,
            has_true_posterior=False,
            extras={
                "novel_start_batch": novel_start,
                "channel_names": [_DISPLAY.get(s, s) for s in self.scenarios],
                "scenarios": list(self.scenarios),
                "train_scenarios": list(self.train_scenarios),
                "snr_db": self.snr_db,
                "delay_spread_max": {s: int(self._lens[s].max()) for s in self.scenarios},
            },
        )

    def __iter__(self) -> Iterator[StreamBatch]:
        rng = np.random.default_rng(self.seed)
        batch_idx = 0
        for sec_i, scenario in enumerate(self.scenarios):
            taps, lens = self._taps[scenario], self._lens[scenario]
            n_real = taps.shape[0]
            for b in range(self.n_batches_per_scenario):
                i = int(rng.integers(0, n_real))
                h = _cir(taps, lens, i)
                x, y, bits, ref = _transmit(h, self.batch_size, self.snr_db, rng)
                yield StreamBatch(
                    index=batch_idx,
                    x=x,
                    y=y,
                    concept_id=sec_i,
                    is_drift=(b == 0 and sec_i > 0),
                    true_posterior=None,
                    extras={"bits": bits, "ref_iq": ref, "scenario": scenario},
                )
                batch_idx += 1


# ---------------------------------------------------------------------------
# Pre-training (on the training scenario pool) + Laplace fit
# ---------------------------------------------------------------------------

Pool = Tuple[np.ndarray, np.ndarray]   # (taps, tap_lengths) for one scenario


def _sample_h(pools: List[Pool], rng: np.random.Generator) -> np.ndarray:
    """Draw one channel realisation: pick a scenario pool uniformly, then a
    realisation within it (equal weight per training scenario)."""
    taps, lens = pools[int(rng.integers(0, len(pools)))]
    return _cir(taps, lens, int(rng.integers(0, taps.shape[0])))


def _build_deep_lstm(hidden_size: int, n_classes: int, p_drop: float, num_layers: int):
    """Stacked LSTM: ``num_layers`` single-layer LSTMs with an explicit
    ``nn.Dropout`` after each (so MC-Dropout — which flips ``nn.Dropout`` to
    train mode — samples every layer, giving a second inter-layer stochastic
    source on top of the post-output one).  Exposes ``embed`` (penultimate
    features) so the Laplace fit works unchanged."""
    import torch.nn as nn

    class DeepLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstms = nn.ModuleList(
                nn.LSTM(2 if i == 0 else hidden_size, hidden_size,
                        num_layers=1, batch_first=True)
                for i in range(num_layers)
            )
            self.drops = nn.ModuleList(nn.Dropout(p_drop) for _ in range(num_layers))
            self.head = nn.Linear(hidden_size, n_classes)

        def _body(self, x):
            out = x
            for lstm, drop in zip(self.lstms, self.drops):
                out, _ = lstm(out)
                out = drop(out)
            return out                      # (1, T, hidden)

        def forward(self, x):
            return self.head(self._body(x))  # (1, T, n_classes)

        def embed(self, x):
            return self._body(x).squeeze(0)  # (T, hidden)

    return DeepLSTM()


def _build_eq_lstm(hidden_size: int, p_drop: float, num_layers: int, with_dropout: bool):
    """Select the equaliser LSTM: single-layer reuses the shared quadriga
    builders (so existing 1-layer checkpoints stay loadable); ``num_layers > 1``
    uses the stacked :func:`_build_deep_lstm`."""
    if num_layers and int(num_layers) > 1:
        return _build_deep_lstm(hidden_size, 4, p_drop if with_dropout else 0.0, int(num_layers))
    return _build_rnn_dropout(hidden_size, 4, p_drop) if with_dropout \
        else _build_rnn_no_dropout(hidden_size, 4)


def _pretrain_equalizer(
    pools: List[Pool],
    ckpt_path: Path,
    hidden_size: int,
    p_drop: float,
    snr_db: float,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    lr: float,
    seed: int,
    with_dropout: bool,
    num_layers: int = 1,
) -> None:
    """Train the LSTM equaliser on the pooled training scenarios' channels."""
    import torch
    from torch import optim

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = _build_eq_lstm(hidden_size, p_drop, num_layers, with_dropout)
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(epochs))
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()

    for epoch in range(int(epochs)):
        running, n = 0.0, 0
        for _ in range(int(steps_per_epoch)):
            h = _sample_h(pools, rng)
            x_np, y_np, _, ref = _transmit(h, batch_size, snr_db, rng)
            x_t = _lstm_input(x_np, ref)                        # (1, B+1, 2)
            y_t = torch.from_numpy(y_np)                        # (B,)
            opt.zero_grad()
            logits = model(x_t).squeeze(0)[1:]                  # (B, 4) drop reference step
            loss = loss_fn(logits, y_t)
            loss.backward()
            opt.step()
            running += loss.item()
            n += 1
        scheduler.step()                                        # cosine LR decay per epoch
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [equalizer pretrain] epoch {epoch+1}/{epochs}  "
                  f"loss={running/max(n,1):.4f}  lr={opt.param_groups[0]['lr']:.2e}")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.to("cpu").state_dict(), str(ckpt_path))
    print(f"  [equalizer pretrain] saved → {ckpt_path}")


def _fit_laplace_head(
    model,
    pools: List[Pool],
    snr_db: float,
    n_fit_batches: int,
    batch_size: int,
    prior_precision: float,
    seed: int,
) -> dict:
    """Diagonal GGN Laplace posterior over the ``Linear(hidden→4)`` head,
    accumulated on the pooled training scenarios' channels."""
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(seed)
    model.eval()
    K = model.head.out_features
    D = model.head.in_features
    H_W = np.zeros((K, D), dtype=np.float64)
    H_b = np.zeros(K, dtype=np.float64)

    with torch.no_grad():
        for _ in range(int(n_fit_batches)):
            h = _sample_h(pools, rng)
            x_np, _, _, ref = _transmit(h, batch_size, snr_db, rng)
            x_t = _lstm_input(x_np, ref)
            feats = model.embed(x_t)[1:]                        # (B, D) drop reference step
            probs = F.softmax(model.head(feats), dim=-1).numpy()
            feats_np = feats.numpy()
            pk1pk = probs * (1.0 - probs)
            H_W += pk1pk.T @ (feats_np ** 2)
            H_b += pk1pk.sum(axis=0)

    return {
        "W_map": model.head.weight.detach().numpy().astype(np.float64),
        "b_map": model.head.bias.detach().numpy().astype(np.float64),
        "W_var": 1.0 / (H_W + prior_precision),
        "b_var": 1.0 / (H_b + prior_precision),
    }


# ---------------------------------------------------------------------------
# MC-Dropout equaliser
# ---------------------------------------------------------------------------

def _as_scenario_list(train_scenario) -> List[str]:
    """Accept a single scenario name or a list of them → list of names."""
    if isinstance(train_scenario, (list, tuple)):
        return [str(s) for s in train_scenario]
    return [str(train_scenario)]


class _EqualizerBase(Classifier):
    def _validate(self, spec: StreamSpec, who: str) -> None:
        if spec.n_classes != 4:
            raise ValueError(f"{who} is 4-way (QPSK); got n_classes={spec.n_classes}")
        if tuple(spec.input_shape) != (2,):
            raise ValueError(f"{who} expects input_shape=(2,); got {spec.input_shape}")

    def _train_pools(self) -> List[Pool]:
        return [_load_taps(Path(self.data_root), s) for s in self.train_scenarios]


@register("model", "rnn_equalizer_mc_dropout")
class RNNEqualizerMCDropout(_EqualizerBase):
    """LSTM equaliser trained on one scenario, MC-Dropout uncertainty at test."""

    def __init__(
        self,
        data_root: str = "./database/QuaDRiGa",
        ckpt_path: str = "./artifacts/models/rnn_equalizer_highwaylos.pt",
        train_scenario="highway_los",
        hidden_size: int = 64,
        num_layers: int = 1,
        p_drop: float = 0.3,
        n_samples: int = 20,
        snr_db: float = 15.0,
        pretrain_epochs: int = 12,
        pretrain_steps_per_epoch: int = 150,
        pretrain_batch_size: int = 512,
        pretrain_lr: float = 1e-3,
        pretrain_seed: int = 0,
        seed: int = 0,
    ) -> None:
        self.data_root = str(data_root)
        self.ckpt_path = str(ckpt_path)
        self.train_scenarios = _as_scenario_list(train_scenario)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.p_drop = float(p_drop)
        self.n_samples = int(n_samples)
        self.snr_db = float(snr_db)
        self.pretrain_epochs = int(pretrain_epochs)
        self.pretrain_steps_per_epoch = int(pretrain_steps_per_epoch)
        self.pretrain_batch_size = int(pretrain_batch_size)
        self.pretrain_lr = float(pretrain_lr)
        self.pretrain_seed = int(pretrain_seed)
        self.seed = int(seed)
        self._rnn = None

    def setup(self, spec: StreamSpec) -> None:
        import torch
        self._validate(spec, "rnn_equalizer_mc_dropout")
        ckpt = Path(self.ckpt_path)
        if not ckpt.exists():
            print(f"[rnn_equalizer_mc_dropout] Pretraining on {self.train_scenarios} "
                  f"(L={self.num_layers}, {self.pretrain_epochs} epochs)…")
            _pretrain_equalizer(
                self._train_pools(), ckpt, self.hidden_size, self.p_drop, self.snr_db,
                self.pretrain_epochs, self.pretrain_steps_per_epoch,
                self.pretrain_batch_size, self.pretrain_lr, self.pretrain_seed,
                with_dropout=True, num_layers=self.num_layers,
            )
        torch.manual_seed(self.seed)
        model = _build_eq_lstm(self.hidden_size, self.p_drop, self.num_layers, with_dropout=True)
        model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
        _enable_mc_dropout(model)
        self._rnn = model

    def predict(self, batch: StreamBatch) -> Prediction:
        import torch
        import torch.nn.functional as F
        assert self._rnn is not None
        ref = np.asarray(batch.extras["ref_iq"], dtype=np.float32)
        x = _lstm_input(batch.x, ref)                           # (1, B+1, 2)
        samples: list = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                logits = self._rnn(x).squeeze(0)[1:]            # (B, 4) drop reference
                samples.append(F.softmax(logits, dim=-1).numpy().astype(np.float64))
        mc_probs = np.stack(samples, axis=0)                    # (S, B, 4)
        mean_probs = mc_probs.mean(axis=0)
        pred_bits = _GRAY_BITS[mean_probs.argmax(axis=1)]
        return Prediction(
            probs=mean_probs, features=None,
            extras={"mc_probs": mc_probs, "pred_bits": pred_bits},
        )

    def observe(self, batch: StreamBatch, prediction: Prediction) -> None:
        del batch, prediction


# ---------------------------------------------------------------------------
# Last-layer Laplace equaliser
# ---------------------------------------------------------------------------

@register("model", "rnn_equalizer_laplace")
class RNNEqualizerLaplace(_EqualizerBase):
    """LSTM equaliser trained on one scenario, diagonal last-layer Laplace."""

    def __init__(
        self,
        data_root: str = "./database/QuaDRiGa",
        ckpt_path: str = "./artifacts/models/rnn_equalizer_highwaylos_map.pt",
        train_scenario="highway_los",
        hidden_size: int = 64,
        num_layers: int = 1,
        n_samples: int = 20,
        prior_precision: float = 1.0,
        snr_db: float = 15.0,
        pretrain_epochs: int = 12,
        pretrain_steps_per_epoch: int = 150,
        pretrain_batch_size: int = 512,
        pretrain_lr: float = 1e-3,
        pretrain_seed: int = 0,
        laplace_fit_batches: int = 200,
        seed: int = 0,
    ) -> None:
        self.data_root = str(data_root)
        self.ckpt_path = str(ckpt_path)
        self.train_scenarios = _as_scenario_list(train_scenario)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.n_samples = int(n_samples)
        self.prior_precision = float(prior_precision)
        self.snr_db = float(snr_db)
        self.pretrain_epochs = int(pretrain_epochs)
        self.pretrain_steps_per_epoch = int(pretrain_steps_per_epoch)
        self.pretrain_batch_size = int(pretrain_batch_size)
        self.pretrain_lr = float(pretrain_lr)
        self.pretrain_seed = int(pretrain_seed)
        self.laplace_fit_batches = int(laplace_fit_batches)
        self.seed = int(seed)
        self._rnn = None
        self._posterior: dict | None = None

    def setup(self, spec: StreamSpec) -> None:
        import torch
        self._validate(spec, "rnn_equalizer_laplace")
        ckpt = Path(self.ckpt_path)
        pools = self._train_pools()
        if not ckpt.exists():
            print(f"[rnn_equalizer_laplace] Pretraining MAP backbone on "
                  f"{self.train_scenarios} (L={self.num_layers}, {self.pretrain_epochs} epochs)…")
            _pretrain_equalizer(
                pools, ckpt, self.hidden_size, 0.0, self.snr_db,
                self.pretrain_epochs, self.pretrain_steps_per_epoch,
                self.pretrain_batch_size, self.pretrain_lr, self.pretrain_seed,
                with_dropout=False, num_layers=self.num_layers,
            )
        model = _build_eq_lstm(self.hidden_size, 0.0, self.num_layers, with_dropout=False)
        model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
        model.eval()
        self._rnn = model
        print("[rnn_equalizer_laplace] Fitting diagonal GGN Laplace posterior over head…")
        self._posterior = _fit_laplace_head(
            model, pools, self.snr_db, self.laplace_fit_batches,
            self.pretrain_batch_size, self.prior_precision, self.pretrain_seed,
        )
        print("[rnn_equalizer_laplace] Laplace posterior ready.")

    def predict(self, batch: StreamBatch) -> Prediction:
        import torch
        assert self._rnn is not None and self._posterior is not None
        rng = np.random.default_rng(self.seed + batch.index)
        ref = np.asarray(batch.extras["ref_iq"], dtype=np.float32)
        x = _lstm_input(batch.x, ref)                                   # (1, B+1, 2)
        with torch.no_grad():
            feats_np = self._rnn.embed(x)[1:].numpy().astype(np.float64)  # (B, D) drop reference

        W_map = self._posterior["W_map"]
        b_map = self._posterior["b_map"]
        W_std = np.sqrt(self._posterior["W_var"])
        b_std = np.sqrt(self._posterior["b_var"])

        samples: list = []
        for _ in range(self.n_samples):
            W_s = W_map + rng.standard_normal(W_map.shape) * W_std
            b_s = b_map + rng.standard_normal(b_map.shape) * b_std
            logits = feats_np @ W_s.T + b_s[None, :]
            logits -= logits.max(axis=1, keepdims=True)
            e = np.exp(logits)
            samples.append((e / e.sum(axis=1, keepdims=True)).astype(np.float64))
        mc_probs = np.stack(samples, axis=0)                    # (S, B, 4)
        mean_probs = mc_probs.mean(axis=0)
        pred_bits = _GRAY_BITS[mean_probs.argmax(axis=1)]
        return Prediction(
            probs=mean_probs, features=None,
            extras={"mc_probs": mc_probs, "pred_bits": pred_bits},
        )

    def observe(self, batch: StreamBatch, prediction: Prediction) -> None:
        del batch, prediction


# ===========================================================================
# Windowed-MLP equaliser (tapped-delay-line): a non-causal ±W window of
# received samples per symbol -> small MLP.  Sees future taps (unlike the
# causal LSTM), so it actually equalises the multi-tap channel.
# ===========================================================================

def _window_features(x_np: np.ndarray, ref: np.ndarray, W: int) -> np.ndarray:
    """Tapped-delay-line features: for each of the ``B`` scored received
    samples, a length-``2W+1`` window centred on it (Re then Im), zero-padded at
    the batch edges.  The differential reference ``ref`` supplies the left
    neighbour of the first symbol.  Returns ``(B, 2*(2W+1))`` float32."""
    B = x_np.shape[0]
    seq = np.concatenate([np.asarray(ref, np.float32)[None, :], x_np], axis=0)   # (B+1, 2)
    padded = np.pad(seq, ((W, W), (0, 0)), mode="constant")                       # (B+1+2W, 2)
    sw = np.lib.stride_tricks.sliding_window_view(padded, 2 * W + 1, axis=0)      # (B+1, 2, 2W+1)
    return sw[1: B + 1].reshape(B, -1).astype(np.float32)                         # (B, 2*(2W+1))


def _diff_features(x_np: np.ndarray, ref: np.ndarray, W: int) -> np.ndarray:
    """Differential (rotation-invariant) features for DQPSK decoding.

    The discriminative quantity for differential QPSK is the phase rotation
    between consecutive received samples, ``g[n] = rx[n]·conj(rx[n-1])``, whose
    phase equals the transmitted differential increment and which is invariant
    to the unknown constant channel gain.  We build the lag-1 product stream and
    hand the MLP a ``2W+1`` window of it (centred on each scored symbol, Re then
    Im), so the network gets exactly the statistic DQPSK needs rather than
    having to learn the ``rx[n]·conj(rx[n-1])`` product from raw IQ.
    Returns ``(B, 2*(2W+1))`` float32 — same width as the raw window."""
    B = x_np.shape[0]
    c = x_np[:, 0].astype(np.float64) + 1j * x_np[:, 1].astype(np.float64)        # (B,)
    r = complex(float(ref[0]), float(ref[1]))
    seq = np.concatenate([[r], c])                                                # (B+1,)
    g = seq[1:] * np.conj(seq[:-1])                                               # (B,) lag-1 product
    gpad = np.pad(g, (W, W), mode="constant")
    sw = np.lib.stride_tricks.sliding_window_view(gpad, 2 * W + 1)                # (B, 2W+1)
    return np.concatenate([sw.real, sw.imag], axis=1).astype(np.float32)          # (B, 2*(2W+1))


def _features(x_np: np.ndarray, ref: np.ndarray, W: int, mode: str) -> np.ndarray:
    """Dispatch to the raw-IQ window (``mode='raw'``) or the differential-product
    window (``mode='diff'``)."""
    if mode == "diff":
        return _diff_features(x_np, ref, W)
    return _window_features(x_np, ref, W)


def _build_mlp(in_dim: int, hidden, p_drop: float, n_classes: int = 4):
    """Small MLP with a ``head`` linear layer and an ``embed`` (penultimate
    features) method, matching the interface the Laplace fit expects."""
    import torch.nn as nn

    dims = [int(in_dim)] + [int(h) for h in hidden]

    class MLPEqualizer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            body: list = []
            for a, b in zip(dims[:-1], dims[1:]):
                body += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(p_drop)]
            self.body = nn.Sequential(*body)
            self.head = nn.Linear(dims[-1], n_classes)

        def forward(self, x):
            return self.head(self.body(x))

        def embed(self, x):
            return self.body(x)

    return MLPEqualizer()


def _pretrain_mlp(
    pools: List[Pool],
    ckpt_path: Path,
    window_radius: int,
    hidden,
    p_drop: float,
    snr_db: float,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    lr: float,
    seed: int,
    feature_mode: str = "raw",
) -> None:
    """Train the windowed-MLP equaliser on the pooled training scenarios."""
    import torch
    from torch import optim

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    in_dim = 2 * (2 * window_radius + 1)
    model = _build_mlp(in_dim, hidden, p_drop)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()

    for epoch in range(int(epochs)):
        running, n = 0.0, 0
        for _ in range(int(steps_per_epoch)):
            h = _sample_h(pools, rng)
            x_np, y_np, _, ref = _transmit(h, batch_size, snr_db, rng)
            feats = _features(x_np, ref, window_radius, feature_mode)
            ft = torch.from_numpy(feats)
            y_t = torch.from_numpy(y_np)
            opt.zero_grad()
            loss = loss_fn(model(ft), y_t)
            loss.backward()
            opt.step()
            running += loss.item()
            n += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [mlp pretrain] epoch {epoch+1}/{epochs}  loss={running/max(n,1):.4f}")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.to("cpu").state_dict(), str(ckpt_path))
    print(f"  [mlp pretrain] saved → {ckpt_path}")


def _fit_laplace_mlp(
    model,
    pools: List[Pool],
    window_radius: int,
    snr_db: float,
    n_fit_batches: int,
    batch_size: int,
    prior_precision: float,
    seed: int,
    feature_mode: str = "raw",
) -> dict:
    """Diagonal GGN Laplace posterior over the MLP ``head`` (Linear→4)."""
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(seed)
    model.eval()
    K = model.head.out_features
    D = model.head.in_features
    H_W = np.zeros((K, D), dtype=np.float64)
    H_b = np.zeros(K, dtype=np.float64)

    with torch.no_grad():
        for _ in range(int(n_fit_batches)):
            h = _sample_h(pools, rng)
            x_np, _, _, ref = _transmit(h, batch_size, snr_db, rng)
            ft = torch.from_numpy(_features(x_np, ref, window_radius, feature_mode))
            feats = model.embed(ft)                             # (B, D)
            probs = F.softmax(model.head(feats), dim=-1).numpy()
            feats_np = feats.numpy()
            pk1pk = probs * (1.0 - probs)
            H_W += pk1pk.T @ (feats_np ** 2)
            H_b += pk1pk.sum(axis=0)

    return {
        "W_map": model.head.weight.detach().numpy().astype(np.float64),
        "b_map": model.head.bias.detach().numpy().astype(np.float64),
        "W_var": 1.0 / (H_W + prior_precision),
        "b_var": 1.0 / (H_b + prior_precision),
    }


@register("model", "mlp_equalizer_mc_dropout")
class MLPEqualizerMCDropout(_EqualizerBase):
    """Windowed-MLP equaliser trained on the training pool, MC-Dropout at test."""

    def __init__(
        self,
        data_root: str = "./database/QuaDRiGa",
        ckpt_path: str = "./artifacts/models/mlp_equalizer_highway.pt",
        train_scenario="highway_los",
        window_radius: int = 8,
        hidden=(128, 64),
        feature_mode: str = "raw",
        p_drop: float = 0.3,
        n_samples: int = 20,
        snr_db: float = 15.0,
        pretrain_epochs: int = 15,
        pretrain_steps_per_epoch: int = 200,
        pretrain_batch_size: int = 512,
        pretrain_lr: float = 1e-3,
        pretrain_seed: int = 0,
        seed: int = 0,
    ) -> None:
        self.data_root = str(data_root)
        self.ckpt_path = str(ckpt_path)
        self.train_scenarios = _as_scenario_list(train_scenario)
        self.window_radius = int(window_radius)
        self.hidden = list(hidden)
        self.feature_mode = str(feature_mode)
        self.p_drop = float(p_drop)
        self.n_samples = int(n_samples)
        self.snr_db = float(snr_db)
        self.pretrain_epochs = int(pretrain_epochs)
        self.pretrain_steps_per_epoch = int(pretrain_steps_per_epoch)
        self.pretrain_batch_size = int(pretrain_batch_size)
        self.pretrain_lr = float(pretrain_lr)
        self.pretrain_seed = int(pretrain_seed)
        self.seed = int(seed)
        self._model = None

    def setup(self, spec: StreamSpec) -> None:
        import torch
        self._validate(spec, "mlp_equalizer_mc_dropout")
        ckpt = Path(self.ckpt_path)
        if not ckpt.exists():
            print(f"[mlp_equalizer_mc_dropout] Pretraining on {self.train_scenarios} "
                  f"(W={self.window_radius}, feat={self.feature_mode}, "
                  f"{self.pretrain_epochs} epochs)…")
            _pretrain_mlp(
                self._train_pools(), ckpt, self.window_radius, self.hidden,
                self.p_drop, self.snr_db, self.pretrain_epochs,
                self.pretrain_steps_per_epoch, self.pretrain_batch_size,
                self.pretrain_lr, self.pretrain_seed, self.feature_mode,
            )
        torch.manual_seed(self.seed)
        in_dim = 2 * (2 * self.window_radius + 1)
        model = _build_mlp(in_dim, self.hidden, self.p_drop)
        model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
        _enable_mc_dropout(model)
        self._model = model

    def predict(self, batch: StreamBatch) -> Prediction:
        import torch
        import torch.nn.functional as F
        assert self._model is not None
        ref = np.asarray(batch.extras["ref_iq"], dtype=np.float32)
        ft = torch.from_numpy(_features(batch.x, ref, self.window_radius, self.feature_mode))
        samples: list = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                samples.append(F.softmax(self._model(ft), dim=-1).numpy().astype(np.float64))
        mc_probs = np.stack(samples, axis=0)                    # (S, B, 4)
        mean_probs = mc_probs.mean(axis=0)
        pred_bits = _GRAY_BITS[mean_probs.argmax(axis=1)]
        return Prediction(
            probs=mean_probs, features=None,
            extras={"mc_probs": mc_probs, "pred_bits": pred_bits},
        )

    def observe(self, batch: StreamBatch, prediction: Prediction) -> None:
        del batch, prediction


@register("model", "mlp_equalizer_laplace")
class MLPEqualizerLaplace(_EqualizerBase):
    """Windowed-MLP equaliser, diagonal last-layer Laplace over the head."""

    def __init__(
        self,
        data_root: str = "./database/QuaDRiGa",
        ckpt_path: str = "./artifacts/models/mlp_equalizer_highway_map.pt",
        train_scenario="highway_los",
        window_radius: int = 8,
        hidden=(128, 64),
        feature_mode: str = "raw",
        n_samples: int = 20,
        prior_precision: float = 1.0,
        snr_db: float = 15.0,
        pretrain_epochs: int = 15,
        pretrain_steps_per_epoch: int = 200,
        pretrain_batch_size: int = 512,
        pretrain_lr: float = 1e-3,
        pretrain_seed: int = 0,
        laplace_fit_batches: int = 200,
        seed: int = 0,
    ) -> None:
        self.data_root = str(data_root)
        self.ckpt_path = str(ckpt_path)
        self.train_scenarios = _as_scenario_list(train_scenario)
        self.window_radius = int(window_radius)
        self.hidden = list(hidden)
        self.feature_mode = str(feature_mode)
        self.n_samples = int(n_samples)
        self.prior_precision = float(prior_precision)
        self.snr_db = float(snr_db)
        self.pretrain_epochs = int(pretrain_epochs)
        self.pretrain_steps_per_epoch = int(pretrain_steps_per_epoch)
        self.pretrain_batch_size = int(pretrain_batch_size)
        self.pretrain_lr = float(pretrain_lr)
        self.pretrain_seed = int(pretrain_seed)
        self.laplace_fit_batches = int(laplace_fit_batches)
        self.seed = int(seed)
        self._model = None
        self._posterior: dict | None = None

    def setup(self, spec: StreamSpec) -> None:
        import torch
        self._validate(spec, "mlp_equalizer_laplace")
        ckpt = Path(self.ckpt_path)
        pools = self._train_pools()
        if not ckpt.exists():
            print(f"[mlp_equalizer_laplace] Pretraining MAP MLP on {self.train_scenarios} "
                  f"(W={self.window_radius}, feat={self.feature_mode}, "
                  f"{self.pretrain_epochs} epochs)…")
            _pretrain_mlp(
                pools, ckpt, self.window_radius, self.hidden, 0.0, self.snr_db,
                self.pretrain_epochs, self.pretrain_steps_per_epoch,
                self.pretrain_batch_size, self.pretrain_lr, self.pretrain_seed,
                self.feature_mode,
            )
        in_dim = 2 * (2 * self.window_radius + 1)
        model = _build_mlp(in_dim, self.hidden, 0.0)
        model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
        model.eval()
        self._model = model
        print("[mlp_equalizer_laplace] Fitting diagonal GGN Laplace posterior over head…")
        self._posterior = _fit_laplace_mlp(
            model, pools, self.window_radius, self.snr_db,
            self.laplace_fit_batches, self.pretrain_batch_size,
            self.prior_precision, self.pretrain_seed, self.feature_mode,
        )
        print("[mlp_equalizer_laplace] Laplace posterior ready.")

    def predict(self, batch: StreamBatch) -> Prediction:
        import torch
        assert self._model is not None and self._posterior is not None
        rng = np.random.default_rng(self.seed + batch.index)
        ref = np.asarray(batch.extras["ref_iq"], dtype=np.float32)
        ft = torch.from_numpy(_features(batch.x, ref, self.window_radius, self.feature_mode))
        with torch.no_grad():
            feats_np = self._model.embed(ft).numpy().astype(np.float64)    # (B, D)

        W_map = self._posterior["W_map"]
        b_map = self._posterior["b_map"]
        W_std = np.sqrt(self._posterior["W_var"])
        b_std = np.sqrt(self._posterior["b_var"])

        samples: list = []
        for _ in range(self.n_samples):
            W_s = W_map + rng.standard_normal(W_map.shape) * W_std
            b_s = b_map + rng.standard_normal(b_map.shape) * b_std
            logits = feats_np @ W_s.T + b_s[None, :]
            logits -= logits.max(axis=1, keepdims=True)
            e = np.exp(logits)
            samples.append((e / e.sum(axis=1, keepdims=True)).astype(np.float64))
        mc_probs = np.stack(samples, axis=0)
        mean_probs = mc_probs.mean(axis=0)
        pred_bits = _GRAY_BITS[mean_probs.argmax(axis=1)]
        return Prediction(
            probs=mean_probs, features=None,
            extras={"mc_probs": mc_probs, "pred_bits": pred_bits},
        )

    def observe(self, batch: StreamBatch, prediction: Prediction) -> None:
        del batch, prediction
