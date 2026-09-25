"""QuaDRiGa SNR-degradation drift experiment.

Trains a QPSK LSTM equaliser on two Urban channels at their *typical* operating
SNRs, sampled per batch:

* Urban LOS  at SNR ~ Uniform(20, 25) dB  (strong link)
* Urban NLOS at SNR ~ Uniform(5, 10) dB   (weak link)

The frozen receiver is then evaluated on a stream whose SNR is swept *downward,
out of the trained range*, following an explicit schedule — e.g.

* Urban LOS  starting at 23 dB, stepping down to 15 dB
* then Urban NLOS starting at 8 dB, stepping down to 3 dB

each SNR level held for ``blocks_per_snr`` batches.  This lets us watch the error
rate and epistemic uncertainty as the operating point leaves the training
distribution — an *SNR-distribution* drift, complementary to the channel-type
drift of :mod:`quadriga_drift`.  Channel taps are used as stored; only the AWGN
level changes.

Registered components
---------------------
* ``quadriga_snr_drift``            — DatasetStream, input_shape=(2,), n_classes=4
* ``rnn_equalizer_snr_mc_dropout``  — trained on the SNR-range mix, MC-Dropout
* ``rnn_equalizer_snr_laplace``     — same backbone, diagonal last-layer Laplace
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from uncertainty_driven_drift.components.quadriga_drift import (
    _DISPLAY,
    _GRAY_BITS,
    _build_eq_lstm,
    _cir,
    _enable_mc_dropout,
    _load_taps,
    _lstm_input,
    _transmit,
    RNNEqualizerLaplace,
    RNNEqualizerMCDropout,
)
from uncertainty_driven_drift.data.base import DatasetStream, StreamBatch, StreamSpec
from uncertainty_driven_drift.models.base import Prediction, StreamSpec as _S  # noqa: F401
from uncertainty_driven_drift.registry import register

# (taps, tap_lengths, snr_min, snr_max) for one training scenario
SNRPool = Tuple[np.ndarray, np.ndarray, float, float]


# ---------------------------------------------------------------------------
# Training with per-scenario SNR ranges
# ---------------------------------------------------------------------------

def _sample_h_snr(pools: List[SNRPool], rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    """Pick a training scenario uniformly, a realisation within it, and an SNR
    uniformly from that scenario's range."""
    taps, lens, smin, smax = pools[int(rng.integers(0, len(pools)))]
    h = _cir(taps, lens, int(rng.integers(0, taps.shape[0])))
    snr = float(rng.uniform(smin, smax))
    return h, snr


def _pretrain_snr(
    pools: List[SNRPool], ckpt_path: Path, hidden_size: int, p_drop: float,
    epochs: int, steps_per_epoch: int, batch_size: int, lr: float, seed: int,
    with_dropout: bool, num_layers: int = 1,
) -> None:
    """Train the equaliser, sampling a fresh (scenario, SNR) per batch."""
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
            h, snr = _sample_h_snr(pools, rng)
            x_np, y_np, _, ref = _transmit(h, batch_size, snr, rng)
            x_t = _lstm_input(x_np, ref)
            y_t = torch.from_numpy(y_np)
            opt.zero_grad()
            loss = loss_fn(model(x_t).squeeze(0)[1:], y_t)
            loss.backward()
            opt.step()
            running += loss.item()
            n += 1
        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [snr-equalizer pretrain] epoch {epoch+1}/{epochs}  "
                  f"loss={running/max(n,1):.4f}  lr={opt.param_groups[0]['lr']:.2e}")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.to("cpu").state_dict(), str(ckpt_path))
    print(f"  [snr-equalizer pretrain] saved → {ckpt_path}")


def _fit_laplace_snr(
    model, pools: List[SNRPool], n_fit_batches: int, batch_size: int,
    prior_precision: float, seed: int,
) -> dict:
    """Diagonal GGN Laplace posterior over the head, on the SNR-range training mix."""
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
            h, snr = _sample_h_snr(pools, rng)
            x_np, _, _, ref = _transmit(h, batch_size, snr, rng)
            feats = model.embed(_lstm_input(x_np, ref))[1:]
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


def _spec_to_pools(data_root: str, train_spec: Sequence[Dict]) -> List[SNRPool]:
    pools: List[SNRPool] = []
    for s in train_spec:
        taps, lens = _load_taps(Path(data_root), s["scenario"])
        pools.append((taps, lens, float(s["snr_min"]), float(s["snr_max"])))
    return pools


# ---------------------------------------------------------------------------
# SNR-schedule stream
# ---------------------------------------------------------------------------

def _expand_schedule(schedule: Sequence[Dict]) -> Tuple[List[Tuple[str, float]], List[dict]]:
    """Expand a schedule of segments into a flat list of (scenario, snr) blocks.

    Each segment sweeps SNR from ``snr_start`` down to ``snr_end`` in steps of
    ``snr_step`` (default 1), holding each level for ``blocks_per_snr`` batches
    (default 2).  The end value is always reached (a shorter final step is added
    if ``snr_step`` would overshoot it)."""
    blocks: List[Tuple[str, float]] = []
    segments: List[dict] = []
    for seg in schedule:
        scen = str(seg["scenario"])
        s0 = float(seg["snr_start"])
        s1 = float(seg["snr_end"])
        step = abs(float(seg.get("snr_step", 1.0)))
        rep = int(seg.get("blocks_per_snr", 2))
        levels: List[float] = []
        v = s0
        while v > s1 + 1e-9:
            levels.append(round(v, 3))
            v -= step
        if not levels or abs(levels[-1] - s1) > 1e-9:
            levels.append(round(s1, 3))
        start = len(blocks)
        for lv in levels:
            for _ in range(rep):
                blocks.append((scen, lv))
        segments.append({"scenario": scen, "snr_start": s0, "snr_end": s1,
                          "levels": levels, "start": start, "end": len(blocks)})
    return blocks, segments


@register("dataset", "quadriga_snr_drift")
class QuadrigaSNRDriftStream(DatasetStream):
    """QPSK stream over Urban channels with a scheduled downward SNR sweep."""

    def __init__(
        self,
        data_root: str = "./database/QuaDRiGa",
        schedule: Optional[Sequence[Dict]] = None,
        batch_size: int = 512,
        seed: int = 0,
    ) -> None:
        if not schedule:
            raise ValueError("quadriga_snr_drift requires a non-empty 'schedule'")
        self.data_root = Path(data_root)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self._blocks, self._segments = _expand_schedule(schedule)

        self._taps: Dict[str, np.ndarray] = {}
        self._lens: Dict[str, np.ndarray] = {}
        for scen in {b[0] for b in self._blocks}:
            self._taps[scen], self._lens[scen] = _load_taps(self.data_root, scen)

        total = len(self._blocks)
        self._drift_indices = tuple(
            seg["start"] for seg in self._segments if seg["start"] > 0
        )
        # Bold boundary = the channel-type transition (first block of 2nd segment).
        novel_start = self._segments[1]["start"] if len(self._segments) > 1 else 0
        channel_names = [
            f"{_DISPLAY.get(seg['scenario'], seg['scenario'])} "
            f"{int(seg['snr_start'])}-{int(seg['snr_end'])}dB"
            for seg in self._segments
        ]

        self.spec = StreamSpec(
            name="quadriga_snr_drift",
            input_shape=(2,),
            n_classes=4,
            n_batches=total,
            batch_size=self.batch_size,
            drift_indices=self._drift_indices,
            has_true_posterior=False,
            extras={
                "novel_start_batch": novel_start,
                "channel_names": channel_names,
                "block_snr": [b[1] for b in self._blocks],
                "block_scenario": [b[0] for b in self._blocks],
                "segments": self._segments,
            },
        )

    def _segment_of(self, i: int) -> int:
        for si, seg in enumerate(self._segments):
            if seg["start"] <= i < seg["end"]:
                return si
        return len(self._segments) - 1

    def __iter__(self) -> Iterator[StreamBatch]:
        rng = np.random.default_rng(self.seed)
        drift_set = set(self._drift_indices)
        for i, (scen, snr) in enumerate(self._blocks):
            taps, lens = self._taps[scen], self._lens[scen]
            h = _cir(taps, lens, int(rng.integers(0, taps.shape[0])))
            x, y, bits, ref = _transmit(h, self.batch_size, snr, rng)
            yield StreamBatch(
                index=i,
                x=x,
                y=y,
                concept_id=self._segment_of(i),
                is_drift=(i in drift_set),
                true_posterior=None,
                extras={"bits": bits, "ref_iq": ref, "scenario": scen, "snr_db": snr},
            )


# ---------------------------------------------------------------------------
# Models: single-layer LSTM equaliser trained on the SNR-range mix
# ---------------------------------------------------------------------------

@register("model", "rnn_equalizer_snr_mc_dropout")
class RNNEqualizerSNRMCDropout(RNNEqualizerMCDropout):
    """MC-Dropout equaliser trained across per-scenario SNR ranges."""

    def __init__(
        self,
        data_root: str = "./database/QuaDRiGa",
        ckpt_path: str = "./artifacts/models/rnn_equalizer_urban_snr.pt",
        train_spec: Optional[Sequence[Dict]] = None,
        hidden_size: int = 64,
        num_layers: int = 1,
        p_drop: float = 0.3,
        n_samples: int = 20,
        pretrain_epochs: int = 40,
        pretrain_steps_per_epoch: int = 300,
        pretrain_batch_size: int = 512,
        pretrain_lr: float = 1e-3,
        pretrain_seed: int = 0,
        seed: int = 0,
    ) -> None:
        self.data_root = str(data_root)
        self.ckpt_path = str(ckpt_path)
        self.train_spec = list(train_spec or [])
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.p_drop = float(p_drop)
        self.n_samples = int(n_samples)
        self.pretrain_epochs = int(pretrain_epochs)
        self.pretrain_steps_per_epoch = int(pretrain_steps_per_epoch)
        self.pretrain_batch_size = int(pretrain_batch_size)
        self.pretrain_lr = float(pretrain_lr)
        self.pretrain_seed = int(pretrain_seed)
        self.seed = int(seed)
        self._rnn = None

    def setup(self, spec: StreamSpec) -> None:
        import torch
        self._validate(spec, "rnn_equalizer_snr_mc_dropout")
        ckpt = Path(self.ckpt_path)
        if not ckpt.exists():
            print(f"[rnn_equalizer_snr_mc_dropout] Pretraining on {self.train_spec} "
                  f"(L={self.num_layers}, {self.pretrain_epochs} epochs)…")
            _pretrain_snr(
                _spec_to_pools(self.data_root, self.train_spec), ckpt,
                self.hidden_size, self.p_drop, self.pretrain_epochs,
                self.pretrain_steps_per_epoch, self.pretrain_batch_size,
                self.pretrain_lr, self.pretrain_seed,
                with_dropout=True, num_layers=self.num_layers,
            )
        torch.manual_seed(self.seed)
        model = _build_eq_lstm(self.hidden_size, self.p_drop, self.num_layers, with_dropout=True)
        model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
        _enable_mc_dropout(model)
        self._rnn = model

    # predict / observe inherited from RNNEqualizerMCDropout


@register("model", "rnn_equalizer_snr_laplace")
class RNNEqualizerSNRLaplace(RNNEqualizerLaplace):
    """Diagonal last-layer Laplace equaliser trained across per-scenario SNR ranges."""

    def __init__(
        self,
        data_root: str = "./database/QuaDRiGa",
        ckpt_path: str = "./artifacts/models/rnn_equalizer_urban_snr_map.pt",
        train_spec: Optional[Sequence[Dict]] = None,
        hidden_size: int = 64,
        num_layers: int = 1,
        n_samples: int = 20,
        prior_precision: float = 1.0,
        pretrain_epochs: int = 40,
        pretrain_steps_per_epoch: int = 300,
        pretrain_batch_size: int = 512,
        pretrain_lr: float = 1e-3,
        pretrain_seed: int = 0,
        laplace_fit_batches: int = 200,
        seed: int = 0,
    ) -> None:
        self.data_root = str(data_root)
        self.ckpt_path = str(ckpt_path)
        self.train_spec = list(train_spec or [])
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.n_samples = int(n_samples)
        self.prior_precision = float(prior_precision)
        self.pretrain_epochs = int(pretrain_epochs)
        self.pretrain_steps_per_epoch = int(pretrain_steps_per_epoch)
        self.pretrain_batch_size = int(pretrain_batch_size)
        self.pretrain_lr = float(pretrain_lr)
        self.pretrain_seed = int(pretrain_seed)
        self.laplace_fit_batches = int(laplace_fit_batches)
        self.seed = int(seed)
        self._rnn = None
        self._posterior = None

    def setup(self, spec: StreamSpec) -> None:
        import torch
        self._validate(spec, "rnn_equalizer_snr_laplace")
        ckpt = Path(self.ckpt_path)
        pools = _spec_to_pools(self.data_root, self.train_spec)
        if not ckpt.exists():
            print(f"[rnn_equalizer_snr_laplace] Pretraining MAP backbone on "
                  f"{self.train_spec} (L={self.num_layers}, {self.pretrain_epochs} epochs)…")
            _pretrain_snr(
                pools, ckpt, self.hidden_size, 0.0, self.pretrain_epochs,
                self.pretrain_steps_per_epoch, self.pretrain_batch_size,
                self.pretrain_lr, self.pretrain_seed,
                with_dropout=False, num_layers=self.num_layers,
            )
        model = _build_eq_lstm(self.hidden_size, 0.0, self.num_layers, with_dropout=False)
        model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
        model.eval()
        self._rnn = model
        print("[rnn_equalizer_snr_laplace] Fitting diagonal GGN Laplace posterior over head…")
        self._posterior = _fit_laplace_snr(
            model, pools, self.laplace_fit_batches, self.pretrain_batch_size,
            self.prior_precision, self.pretrain_seed,
        )
        print("[rnn_equalizer_snr_laplace] Laplace posterior ready.")

    # predict / observe inherited from RNNEqualizerLaplace
