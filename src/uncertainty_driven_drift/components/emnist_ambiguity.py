"""EMNIST label-prior drift — the benign false-alarm scenario (S2-benign).

What this scenario is for
-------------------------
S1 (novel corruptions) does not separate total entropy from epistemic: on the committed
CIFAR runs their AUROCs are 0.799 vs 0.802 and they correlate at 0.969. The blend-based
S2 separates them, but only for a mixup-trained ViT, and it raises the error rate.

This scenario separates them with **no mixup**, on **unmodified images**, and with the
**error rate held flat** — so nothing has actually gone wrong and any alarm is a pure
false alarm:

    total entropy   +3.35 sigma  -> a total-entropy detector (UDD) fires   [WRONG]
    epistemic       +0.17 sigma  -> the epistemic detector stays quiet     [RIGHT]
    accuracy        +0.06 sigma  -> unchanged, so DDM/EDDM also stay quiet [RIGHT]

Only the total-entropy detector gets it wrong.

Why EMNIST and not CIFAR
------------------------
The shift needs classes that are ALEATORICALLY ambiguous yet EPISTEMICALLY familiar.
EMNIST has them structurally: ``1`` collides with ``I``/``L`` and ``0`` with ``O``, so the
glyph genuinely underdetermines the label — while the images are utterly ordinary
handwriting the model saw thousands of times. Measured per-class:

    1   entropy 1.033   epistemic 0.026   accuracy 0.735     <- unsure but usually right
    9   entropy 0.737   epistemic 0.033   accuracy 0.828
    Y   entropy 0.296   epistemic 0.043   accuracy 0.958     <- confident and right

Per-class aleatoric/epistemic correlation is 0.359 on EMNIST against 0.996 on CIFAR-10,
0.989 on Fashion-MNIST and 0.957 on SVHN — natural-image datasets have no such structure,
because there a "hard" class is hard by virtue of ATYPICAL examples, which raises epistemic
in lockstep.

How accuracy is held flat
-------------------------
Per-class entropy and accuracy correlate at -0.885 — strongly, but not -1. That residual
slack is the room: mixing high-entropy-but-accurate classes (1, 9, 0) with high-accuracy
ones (N, B, G, R, Y) raises mean entropy while pinning mean accuracy. The mix is solved by
a small linear program, because every batch mean is linear in the class proportions.

The prior is derived on a HELD-OUT VALIDATION SPLIT by scripts/derive_benign_prior.py and
loaded from JSON here. It is never fitted on the stream the detectors see, which is what
keeps the design non-circular.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from uncertainty_driven_drift.components.image_spec import require_image_spec
from uncertainty_driven_drift.data.base import DatasetStream, StreamBatch, StreamSpec
from uncertainty_driven_drift.models.base import Classifier, Prediction
from uncertainty_driven_drift.registry import register

_N_CLASSES = 47
_INPUT_SHAPE = (1, 28, 28)
#: EMNIST-balanced label order.
LABELS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ") + list("abdefghnqrt")


def _emnist(data_root: Path, train: bool):
    """EMNIST-balanced as (N,1,28,28) float32 in [0,1] plus int64 labels."""
    from torchvision import datasets

    ds = datasets.EMNIST(str(data_root), split="balanced", train=train, download=True)
    x = (ds.data.numpy().astype(np.float32) / 255.0)[:, None]
    return np.ascontiguousarray(x), ds.targets.numpy().astype(np.int64)


def train_val_split(n: int, val_fraction: float, seed: int):
    """Shared split so the model and the prior-derivation see the same holdout.

    The prior MUST be solved on data the model did not train on; otherwise the
    scenario is fitted to the very predictions it is meant to probe.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(val_fraction * n))
    return perm[n_val:], perm[:n_val]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@register("model", "resnet_mc_dropout_emnist")
class ResNetMCDropoutEMNIST(Classifier):
    """ResNet-20 + MC-Dropout, 47-way, for the EMNIST streams.

    EMNIST is 1x28x28; the backbone is the same CIFAR ResNet-20, so inputs are padded to
    32x32 and the single channel is repeated to 3. Keeping one backbone means the
    architecture is not a variable across scenarios.

    ``val_fraction`` is held out of training and is what
    ``scripts/derive_benign_prior.py`` solves the class prior on.
    """

    def __init__(
        self,
        n_samples: int = 20,
        p_drop: float = 0.3,
        data_root: str = "./artifacts/emnist",
        ckpt_path: str = "./artifacts/models/resnet20_emnist_mc.pt",
        val_fraction: float = 0.1,
        split_seed: int = 0,
        pretrain_epochs: int = 20,
        pretrain_batch_size: int = 128,
        pretrain_lr: float = 0.1,
        pretrain_seed: int = 0,
        seed: int = 0,
    ) -> None:
        self.n_samples = int(n_samples)
        self.p_drop = float(p_drop)
        self.data_root = str(data_root)
        self.ckpt_path = str(ckpt_path)
        self.val_fraction = float(val_fraction)
        self.split_seed = int(split_seed)
        self.pretrain_epochs = int(pretrain_epochs)
        self.pretrain_batch_size = int(pretrain_batch_size)
        self.pretrain_lr = float(pretrain_lr)
        self.pretrain_seed = int(pretrain_seed)
        self.seed = int(seed)
        self._model = None
        self._device = None

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def to_backbone_input(x: np.ndarray):
        """(B,1,28,28) -> (B,3,32,32) torch tensor: pad to 32 and repeat the channel."""
        import torch

        x = np.repeat(x, 3, axis=1)
        p = (32 - x.shape[-1]) // 2
        x = np.pad(x, ((0, 0), (0, 0), (p, 32 - x.shape[-2] - p), (p, 32 - x.shape[-1] - p)))
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))

    def _build(self):
        from uncertainty_driven_drift.components.resnet_cifar import _build_resnet20
        return _build_resnet20(n_classes=_N_CLASSES, p_drop=self.p_drop)

    def _pretrain(self) -> None:
        import torch
        from torch import optim
        import torch.nn.functional as F

        print(f"[resnet_mc_dropout_emnist] Pretraining ({self.pretrain_epochs} epochs)…")
        x, y = _emnist(Path(self.data_root), train=True)
        tr_idx, _ = train_val_split(len(x), self.val_fraction, self.split_seed)
        x, y = x[tr_idx], y[tr_idx]

        torch.manual_seed(self.pretrain_seed)
        rng = np.random.default_rng(self.pretrain_seed)
        dev = self._select_device()
        m = self._build().to(dev)
        opt = optim.SGD(m.parameters(), lr=self.pretrain_lr, momentum=0.9, weight_decay=5e-4)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.pretrain_epochs)
        bs = self.pretrain_batch_size
        for ep in range(self.pretrain_epochs):
            m.train()
            perm = rng.permutation(len(x))
            tot, nb = 0.0, 0
            for i in range(0, len(x), bs):
                sel = perm[i:i + bs]
                xb = self.to_backbone_input(x[sel]).to(dev)
                yb = torch.from_numpy(y[sel]).long().to(dev)
                opt.zero_grad()
                loss = F.cross_entropy(m(xb), yb)
                loss.backward()
                opt.step()
                tot += loss.item()
                nb += 1
            sch.step()
            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"  epoch {ep+1}/{self.pretrain_epochs}  loss={tot/nb:.4f}")
        Path(self.ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(m.to("cpu").state_dict(), self.ckpt_path)
        print(f"  saved → {self.ckpt_path}")

    @staticmethod
    def _select_device():
        from uncertainty_driven_drift.components.vit import _select_device
        return _select_device()

    # -- Classifier protocol ---------------------------------------------
    def setup(self, spec: StreamSpec) -> None:
        import torch
        from uncertainty_driven_drift.components.vit import _enable_mc_dropout

        require_image_spec(
            spec, "resnet_mc_dropout_emnist",
            channels=(1,), spatial=((28, 28),), n_classes=_N_CLASSES,
        )
        if not Path(self.ckpt_path).exists():
            self._pretrain()
        torch.manual_seed(self.seed)
        m = self._build()
        m.load_state_dict(torch.load(self.ckpt_path, map_location="cpu"))
        _enable_mc_dropout(m)
        self._device = self._select_device()
        self._model = m.to(self._device)

    def predict(self, batch: StreamBatch) -> Prediction:
        import torch
        import torch.nn.functional as F

        assert self._model is not None
        x = self.to_backbone_input(batch.x).to(self._device)
        samples = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                samples.append(F.softmax(self._model(x), dim=-1).cpu().numpy().astype(np.float64))
        mc = np.stack(samples, axis=0)
        return Prediction(probs=mc.mean(axis=0), features=None, extras={"mc_probs": mc})

    def observe(self, batch: StreamBatch, prediction: Prediction) -> None:
        """Frozen backbone — observing labels is a no-op by design."""
        del batch, prediction


# ---------------------------------------------------------------------------
# Stream
# ---------------------------------------------------------------------------

@register("dataset", "emnist_label_prior")
class EMNISTLabelPriorStream(DatasetStream):
    """EMNIST with a benign label-prior shift: same images, different class mix.

    Phase A draws classes uniformly. Phase B ramps toward ``prior_path``'s solved prior in
    ``len(ramp)`` steps, so the response is a dose-response staircase rather than one jump.

    Nothing about the images changes — only WHICH classes arrive. The model is frozen. The
    prior is solved on a validation holdout to keep mean accuracy pinned, so the error rate
    should not move and any detector alarm is a false alarm.
    """

    def __init__(
        self,
        prior_path: str = "./artifacts/models/emnist_benign_prior.json",
        ramp: Sequence[float] = (0.34, 0.67, 1.0),
        n_uniform_batches: int = 40,
        n_shift_batches: int = 60,
        batch_size: int = 64,
        seed: int = 0,
        data_root: str = "./artifacts/emnist",
    ) -> None:
        p = Path(prior_path)
        if not p.exists():
            raise FileNotFoundError(
                f"{prior_path} not found. The benign class prior must be solved on a "
                f"validation holdout first:\n"
                f"    python scripts/derive_benign_prior.py "
                f"--config configs/experiments/emnist_label_prior/benign_shift.yaml"
            )
        blob = json.loads(p.read_text())
        self.target_prior = np.asarray(blob["prior"], dtype=np.float64)
        self.prior_meta = {k: v for k, v in blob.items() if k != "prior"}
        if len(self.target_prior) != _N_CLASSES:
            raise ValueError(f"prior has {len(self.target_prior)} entries, expected {_N_CLASSES}")

        self.ramp = [float(r) for r in ramp]
        self.n_uniform_batches = int(n_uniform_batches)
        self.n_shift_batches = int(n_shift_batches)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.data_root = str(data_root)

        # Stream only from the TEST split — disjoint from anything the model trained on
        # and from the validation split the prior was solved on.
        self._x, self._y = _emnist(Path(self.data_root), train=False)
        self._idx_by_class = [np.where(self._y == c)[0] for c in range(_N_CLASSES)]

        total = self.n_uniform_batches + self.n_shift_batches
        shift_start = self.n_uniform_batches

        # The ONLY real boundary is the onset of the shift. The uniform batches are
        # i.i.d. from one prior, and the ramp steps inside the shift region are a dose
        # rather than separate concepts, so subdividing either would draw boundaries
        # where nothing changes.
        drift_indices = [shift_start]
        channel_names = ["uniform", "skew towards ambiguous classes"]

        self.spec = StreamSpec(
            name="emnist_label_prior",
            input_shape=_INPUT_SHAPE,
            n_classes=_N_CLASSES,
            n_batches=total,
            batch_size=self.batch_size,
            drift_indices=tuple(drift_indices),
            has_true_posterior=False,
            extras={
                # Reused verbatim as the boundary key so the existing panels and tables
                # mark it. Here it is the onset of the BENIGN prior shift.
                "novel_start_batch": shift_start,
                "shift_kind": "label_prior_benign",
                "ramp": self.ramp,
                "n_uniform_batches": self.n_uniform_batches,
                "n_shift_batches": self.n_shift_batches,
                "channel_names": channel_names,
                "pool_size": int(len(self._x)),
                "prior_top_classes": [
                    f"{LABELS[c]}={self.target_prior[c]:.3f}"
                    for c in np.argsort(-self.target_prior)[:8]
                    if self.target_prior[c] > 0.01
                ],
                **self.prior_meta,
            },
        )

    def __iter__(self) -> Iterator[StreamBatch]:
        rng = np.random.default_rng(self.seed)
        uniform = np.full(_N_CLASSES, 1.0 / _N_CLASSES)
        shift_start = self.spec.extras["novel_start_batch"]
        drift_set = set(self.spec.drift_indices)

        # (prior, n_batches) per segment: one uniform block, then the ramp inside the
        # single shift region. Interpolating uniform -> target makes ambiguity arrive as
        # a dose, which shows up in the curve without needing labelled sub-regions.
        n_steps = len(self.ramp)
        step = self.n_shift_batches // n_steps
        counts = [step] * n_steps
        counts[-1] += self.n_shift_batches - sum(counts)   # absorb any remainder
        segments = [(uniform, self.n_uniform_batches)]
        segments += [((1.0 - r) * uniform + r * self.target_prior, c)
                     for r, c in zip(self.ramp, counts)]

        t = 0
        for phase_i, (prior, n_batches) in enumerate(segments):
            prior = np.asarray(prior, dtype=np.float64)
            prior = prior / prior.sum()
            for _ in range(n_batches):
                classes = rng.choice(_N_CLASSES, size=self.batch_size, p=prior)
                idx = np.array([rng.choice(self._idx_by_class[c]) for c in classes])
                yield StreamBatch(
                    index=t,
                    x=self._x[idx],
                    y=self._y[idx],
                    concept_id=phase_i,
                    is_drift=(t in drift_set),
                    true_posterior=None,
                    extras={
                        "phase": "uniform" if t < shift_start else "skewed",
                        "ramp": float(0.0 if phase_i == 0 else self.ramp[phase_i - 1]),
                        "is_novel": t >= shift_start,
                    },
                )
                t += 1
