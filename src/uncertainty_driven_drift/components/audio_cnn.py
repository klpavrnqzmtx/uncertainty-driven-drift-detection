"""Log-mel CNN for the audio streams, with MC-Dropout and Laplace variants.

Architecture
------------
A 4-stage VGG-style convnet on 64-band log-mel input: (32, 64, 128, 128)
channels, each stage 3x3 conv -> BN -> ReLU -> 2x2 max-pool, then global
average pooling, optional Dropout, and a Linear(128 -> n_classes) head.
245 K parameters for the 35-way head (247 K at 50-way), against the
272 K of the ResNet-20 used in the image arms — deliberately matched, so a
difference between the two modalities is not just a difference in capacity.

Global average pooling over the time axis is what lets one architecture
serve 1 s Speech Commands clips, 4 s UrbanSound8K excerpts and 5 s ESC-50
clips without reshaping anything.

Training
--------
Mirrors the image recipe exactly: each mini-batch is corrupted with a
random *known* corruption unless it falls in the ``clean_fraction`` kept
clean.  Corruption happens on the waveform, before featurisation, because
that is where the physical process lives.

Registered components
---------------------
* ``audio_cnn_mc_dropout`` — dropout left in train mode at inference,
  ``n_samples`` stochastic passes -> ``mc_probs``.
* ``audio_cnn_laplace``    — diagonal GGN last-layer Laplace over the
  Linear(128 -> K) head, ``n_samples`` weight draws -> ``mc_probs``.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from uncertainty_driven_drift.components.audio_features import (
    N_MELS,
    log_mel_batch,
    require_audio_spec,
)
from uncertainty_driven_drift.components.audio_pools import get_spec, load_pool, to_float
from uncertainty_driven_drift.data.base import StreamBatch, StreamSpec
from uncertainty_driven_drift.models.base import Classifier, Prediction
from uncertainty_driven_drift.registry import register


# ---------------------------------------------------------------------------
# architecture
# ---------------------------------------------------------------------------

def _build_audio_cnn(n_classes: int, p_drop: float = 0.0, widths=(32, 64, 128, 128)):
    import torch.nn as nn

    class AudioCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            blocks: List[nn.Module] = []
            in_ch = 1
            for w in widths:
                blocks += [
                    nn.Conv2d(in_ch, w, 3, padding=1, bias=False),
                    nn.BatchNorm2d(w),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
                in_ch = w
            self.features = nn.Sequential(*blocks)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(p=p_drop)
            self.head = nn.Linear(widths[-1], n_classes)

        def embed(self, x):
            """Return the (B, widths[-1]) representation the head reads."""
            return self.avgpool(self.features(x)).flatten(1)

        def forward(self, x):
            return self.head(self.dropout(self.embed(x)))

    return AudioCNN()


def _build_audio_resnet18(n_classes: int, p_drop: float = 0.0):
    """torchvision ResNet-18 adapted to 1-channel log-mel input (~11.2 M params).

    A ResNet-18 over log-mel is *the* standard audio-classification baseline
    (it is the backbone family PANNs and most ESC-50 / UrbanSound8K papers
    report), so this arm answers "does the result hold at a size someone would
    actually deploy?" — which the 247 K :func:`_build_audio_cnn` cannot, since
    that one is sized to match the image arms' ResNet-20 for cross-modality
    comparability.

    Two changes to the stock model: ``conv1`` takes 1 channel instead of 3, and
    a Dropout sits before the head so MC-Dropout has something to sample. No
    pretrained weights — "epistemically familiar" is defined relative to the
    training set, and an ImageNet- or AudioSet-pretrained backbone has already
    seen a huge range of conditions, which weakens the very notion of a novel
    corruption.
    """
    import torch.nn as nn
    from torchvision.models import resnet18

    net = resnet18(weights=None, num_classes=n_classes)
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    class AudioResNet18(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = net
            self.dropout = nn.Dropout(p=p_drop)
            self.head = net.fc
            self.net.fc = nn.Identity()

        def embed(self, x):
            """Return the 512-D pooled representation the head reads."""
            return self.net(x)

        def forward(self, x):
            return self.head(self.dropout(self.embed(x)))

    return AudioResNet18()


_ARCHS = {"cnn": _build_audio_cnn, "resnet18": _build_audio_resnet18}


def build_backbone(arch: str, n_classes: int, p_drop: float, widths=None):
    """Dispatch to an audio backbone by name; ``widths`` applies to ``cnn`` only."""
    try:
        fn = _ARCHS[arch]
    except KeyError as exc:
        raise KeyError(f"Unknown audio arch {arch!r}; have {sorted(_ARCHS)}") from exc
    if arch == "cnn":
        return fn(n_classes=n_classes, p_drop=p_drop,
                  widths=tuple(widths) if widths else (32, 64, 128, 128))
    return fn(n_classes=n_classes, p_drop=p_drop)


def _pick_device(allow_cuda: bool = True):
    import torch

    if allow_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# featurisation
# ---------------------------------------------------------------------------

def _features(wav: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Waveform batch -> normalised log-mel, ``(B, 1, n_mels, n_frames)``."""
    lm = log_mel_batch(wav)
    return ((lm - mean) / std).astype(np.float32)


def _estimate_norm_stats(
    pool_x: np.ndarray,
    rng: np.random.Generator,
    n_clips: int = 512,
) -> Tuple[float, float]:
    """Global (mean, std) of log-mel energy over a random sample of clean clips.

    One scalar pair rather than per-band statistics: the bands are already
    on a common log scale, and a single pair transfers unchanged to every
    corruption, which is what we want when the *point* of the experiment is
    that the input distribution moves.
    """
    idx = np.sort(rng.choice(pool_x.shape[0], size=min(n_clips, pool_x.shape[0]), replace=False))
    lm = log_mel_batch(to_float(pool_x[idx])[:, None, :])
    return float(lm.mean()), float(max(lm.std(), 1e-6))


# ---------------------------------------------------------------------------
# pre-training
# ---------------------------------------------------------------------------

def _pretrain_audio_cnn(
    dataset: str,
    data_root: Path,
    ckpt_path: Path,
    n_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    p_drop: float,
    train_corruptions: Optional[List[str]],
    corruption_severity: int,
    clean_fraction: float,
    allow_cuda: bool = True,
    max_clips: int = 0,
    widths: tuple = (32, 64, 128, 128),
    arch: str = "cnn",
) -> None:
    """Train the log-mel CNN on ``dataset``'s train split, then save the checkpoint.

    The checkpoint stores the log-mel normalisation statistics alongside the
    weights.  They are estimated from this dataset's train pool, so a
    checkpoint is only meaningful with the pool it was fitted on — keeping
    them together makes that impossible to get wrong.
    """
    import torch
    from torch import optim

    torch.manual_seed(seed)
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    device = _pick_device(allow_cuda)

    pool_x, pool_y = load_pool(dataset, "train", data_root)
    n_train = pool_x.shape[0]
    if max_clips and max_clips < n_train:
        # Smoke-test path only (preflight dry runs). A backbone trained on a
        # subsample is not a result; it exists so a config error surfaces in
        # seconds instead of after a full pretrain.
        keep = np.sort(np.random.default_rng(seed).choice(n_train, max_clips, replace=False))
        pool_x, pool_y = pool_x[keep], pool_y[keep]
        n_train = max_clips
        print(f"[audio_cnn] WARNING: pretrain_max_clips={max_clips} — smoke-test backbone, "
              f"not a result")
    mean, std = _estimate_norm_stats(pool_x, np.random.default_rng(seed + 99))
    print(f"[audio_cnn] {dataset}: {n_train} train clips, log-mel norm "
          f"mean={mean:.3f} std={std:.3f}, device={device}")

    model = build_backbone(arch, n_classes, p_drop, widths).to(device)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = torch.nn.CrossEntropyLoss()

    if train_corruptions:
        from uncertainty_driven_drift.components.audio_corruptions import apply_corruption

    model.train()
    for epoch in range(int(epochs)):
        order = np_rng.permutation(n_train)
        running_loss, running_correct, seen, n_batches = 0.0, 0, 0, 0
        for start in range(0, n_train - 1, batch_size):
            idx = np.sort(order[start:start + batch_size])
            if idx.size < 2:                     # BatchNorm needs >1 sample
                continue
            wav = to_float(pool_x[idx])[:, None, :]
            yb_np = np.asarray(pool_y[idx], dtype=np.int64)

            if train_corruptions and py_rng.random() >= clean_fraction:
                corruption = py_rng.choice(train_corruptions)
                wav = apply_corruption(corruption, wav, corruption_severity, np_rng)

            xb = torch.from_numpy(_features(wav, mean, std)).to(device)
            yb = torch.from_numpy(yb_np).to(device)

            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()

            running_loss += float(loss.item())
            running_correct += int((out.argmax(1) == yb).sum().item())
            seen += int(idx.size)
            n_batches += 1
        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{epochs}  loss={running_loss/max(n_batches,1):.4f}  "
                  f"train_acc={running_correct/max(seen,1):.4f}")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.to("cpu").state_dict(),
            "norm_mean": mean,
            "norm_std": std,
            "n_classes": int(n_classes),
            "n_mels": int(N_MELS),
            "dataset": dataset,
            "p_drop": float(p_drop),
            "widths": list(widths),
            "arch": str(arch),
            "train_corruptions": list(train_corruptions or []),
            "corruption_severity": int(corruption_severity),
        },
        str(ckpt_path),
    )
    print(f"  saved -> {ckpt_path}")


# ---------------------------------------------------------------------------
# Laplace posterior
# ---------------------------------------------------------------------------

def _compute_laplace_posterior(
    model,
    dataset: str,
    data_root: Path,
    mean: float,
    std: float,
    train_corruptions: Optional[List[str]],
    corruption_severity: int,
    batch_size: int,
    prior_precision: float,
    seed: int,
    allow_cuda: bool = True,
) -> dict:
    """Diagonal GGN Laplace posterior over the ``Linear(128 -> K)`` head.

    Identical estimator to ``resnet_cifar._compute_laplace_posterior`` — the
    same diagonal GGN, fitted on the train pool with the same known-corruption
    augmentation the backbone saw — so the audio and image Laplace arms are
    the same method and differ only in modality.
    """
    import torch
    import torch.nn.functional as F

    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    device = _pick_device(allow_cuda)
    model = model.to(device)

    pool_x, _ = load_pool(dataset, "train", data_root)
    n_train = pool_x.shape[0]

    K = model.head.out_features
    D = model.head.in_features
    H_W = np.zeros((K, D), dtype=np.float64)
    H_b = np.zeros(K, dtype=np.float64)

    if train_corruptions:
        from uncertainty_driven_drift.components.audio_corruptions import apply_corruption

    model.eval()
    with torch.no_grad():
        for start in range(0, n_train, batch_size):
            idx = np.arange(start, min(start + batch_size, n_train))
            wav = to_float(pool_x[idx])[:, None, :]
            if train_corruptions:
                corruption = py_rng.choice(train_corruptions)
                wav = apply_corruption(corruption, wav, corruption_severity, np_rng)
            xb = torch.from_numpy(_features(wav, mean, std)).to(device)
            feats = model.embed(xb)
            probs = F.softmax(model.head(feats), dim=-1).cpu().numpy().astype(np.float64)
            feats_np = feats.cpu().numpy().astype(np.float64)
            pk1pk = probs * (1.0 - probs)
            H_W += pk1pk.T @ (feats_np ** 2)
            H_b += pk1pk.sum(axis=0)

    return {
        "W_map": model.head.weight.detach().cpu().numpy().astype(np.float64),
        "b_map": model.head.bias.detach().cpu().numpy().astype(np.float64),
        "W_var": 1.0 / (H_W + prior_precision),
        "b_var": 1.0 / (H_b + prior_precision),
    }


# ---------------------------------------------------------------------------
# shared setup
# ---------------------------------------------------------------------------

class _AudioCNNBase(Classifier):
    """Checkpoint loading, pretraining trigger and spec validation.

    Subclasses differ only in how they turn one frozen backbone into a
    posterior predictive sample stack.
    """

    component_name = "audio_cnn"

    def __init__(
        self,
        dataset: str,
        data_root: str,
        ckpt_path: str,
        p_drop: float,
        n_samples: int,
        train_corruptions: Optional[List[str]],
        corruption_severity: int,
        clean_fraction: float,
        pretrain_epochs: int,
        pretrain_batch_size: int,
        pretrain_lr: float,
        pretrain_seed: int,
        seed: int,
        allow_cuda: bool,
        pretrain_max_clips: int = 0,
        widths: Optional[List[int]] = None,
        arch: str = "cnn",
    ) -> None:
        self.dataset = str(dataset)
        self.data_root = str(data_root)
        self.ckpt_path = str(ckpt_path)
        self.p_drop = float(p_drop)
        self.n_samples = int(n_samples)
        self.train_corruptions = list(train_corruptions) if train_corruptions else None
        self.corruption_severity = int(corruption_severity)
        self.clean_fraction = float(clean_fraction)
        self.pretrain_epochs = int(pretrain_epochs)
        self.pretrain_batch_size = int(pretrain_batch_size)
        self.pretrain_lr = float(pretrain_lr)
        self.pretrain_seed = int(pretrain_seed)
        self.seed = int(seed)
        self.allow_cuda = bool(allow_cuda)
        self.pretrain_max_clips = int(pretrain_max_clips)
        self.widths = tuple(int(w) for w in widths) if widths else (32, 64, 128, 128)
        self.arch = str(arch)
        self._model = None
        self._norm: Tuple[float, float] = (0.0, 1.0)
        self._device = None

    # -- shared plumbing ---------------------------------------------------

    def _load_or_train(self, spec: StreamSpec, p_drop: float):
        import torch

        require_audio_spec(spec, self.component_name)
        ds_spec = get_spec(self.dataset)
        if spec.n_classes != ds_spec.n_classes:
            raise ValueError(
                f"{self.component_name} configured for dataset {self.dataset!r} "
                f"({ds_spec.n_classes} classes) but the stream has {spec.n_classes}."
            )
        stream_ds = spec.extras.get("dataset")
        if stream_ds is not None and stream_ds != self.dataset:
            # Two different audio datasets with the same class count would load
            # silently and score nonsense; the norm stats alone make them
            # incompatible. Fail here instead.
            raise ValueError(
                f"{self.component_name} is configured for {self.dataset!r} but the "
                f"stream is {stream_ds!r}. Point model.params.dataset at the same one."
            )

        ckpt = Path(self.ckpt_path)
        if not ckpt.exists():
            print(f"[{self.component_name}] Pretraining on {self.dataset} "
                  f"({self.pretrain_epochs} epochs)…")
            _pretrain_audio_cnn(
                dataset=self.dataset,
                data_root=Path(self.data_root),
                ckpt_path=ckpt,
                n_classes=ds_spec.n_classes,
                epochs=self.pretrain_epochs,
                batch_size=self.pretrain_batch_size,
                lr=self.pretrain_lr,
                seed=self.pretrain_seed,
                p_drop=p_drop,
                train_corruptions=self.train_corruptions,
                corruption_severity=self.corruption_severity,
                clean_fraction=self.clean_fraction,
                allow_cuda=self.allow_cuda,
                max_clips=self.pretrain_max_clips,
                widths=self.widths,
                arch=self.arch,
            )

        blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        # Trust the checkpoint's own widths over the config's: a mismatch would
        # otherwise surface as an opaque state_dict shape error.
        ckpt_arch = blob.get("arch")
        if ckpt_arch and ckpt_arch != self.arch:
            print(f"[{self.component_name}] checkpoint arch {ckpt_arch!r} overrides "
                  f"config arch {self.arch!r}")
            self.arch = str(ckpt_arch)
        ckpt_widths = blob.get("widths")
        if ckpt_widths and tuple(ckpt_widths) != self.widths:
            print(f"[{self.component_name}] checkpoint widths {tuple(ckpt_widths)} override "
                  f"config widths {self.widths}")
            self.widths = tuple(int(w) for w in ckpt_widths)
        model = build_backbone(self.arch, ds_spec.n_classes, p_drop, self.widths)
        model.load_state_dict(blob["state_dict"])
        self._norm = (float(blob["norm_mean"]), float(blob["norm_std"]))
        if blob.get("dataset") not in (None, self.dataset):
            raise ValueError(
                f"checkpoint {ckpt} was trained on {blob.get('dataset')!r}, "
                f"not {self.dataset!r} — its log-mel normalisation does not transfer."
            )
        self._device = _pick_device(self.allow_cuda)
        return model.to(self._device)

    def _batch_features(self, batch: StreamBatch):
        import torch

        mean, std = self._norm
        return torch.from_numpy(_features(batch.x, mean, std)).to(self._device)

    def observe(self, batch: StreamBatch, prediction: Prediction) -> None:
        del batch, prediction


# ---------------------------------------------------------------------------
# MC-Dropout
# ---------------------------------------------------------------------------

@register("model", "audio_cnn_mc_dropout")
class AudioCNNMCDropout(_AudioCNNBase):
    """Frozen log-mel CNN with MC-Dropout over the pooled representation."""

    component_name = "audio_cnn_mc_dropout"

    def __init__(
        self,
        dataset: str = "speech_commands",
        n_samples: int = 20,
        p_drop: float = 0.3,
        data_root: str = "./artifacts/audio/speech_commands",
        ckpt_path: str = "./artifacts/models/audio_cnn_sc_kn_mc.pt",
        train_corruptions: Optional[List[str]] = None,
        corruption_severity: int = 3,
        clean_fraction: float = 0.25,
        pretrain_epochs: int = 30,
        pretrain_batch_size: int = 128,
        pretrain_lr: float = 0.05,
        pretrain_seed: int = 0,
        seed: int = 0,
        allow_cuda: bool = True,
        pretrain_max_clips: int = 0,
        widths: Optional[List[int]] = None,
        arch: str = "cnn",
    ) -> None:
        super().__init__(
            dataset=dataset, data_root=data_root, ckpt_path=ckpt_path, p_drop=p_drop,
            n_samples=n_samples, train_corruptions=train_corruptions,
            corruption_severity=corruption_severity, clean_fraction=clean_fraction,
            pretrain_epochs=pretrain_epochs, pretrain_batch_size=pretrain_batch_size,
            pretrain_lr=pretrain_lr, pretrain_seed=pretrain_seed, seed=seed,
            allow_cuda=allow_cuda,
            pretrain_max_clips=pretrain_max_clips, widths=widths, arch=arch,
        )

    def setup(self, spec: StreamSpec) -> None:
        import torch
        import torch.nn as nn

        model = self._load_or_train(spec, p_drop=self.p_drop)
        torch.manual_seed(self.seed)
        model.eval()
        for m in model.modules():          # dropout stays stochastic; BN does not
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()
        self._model = model

    def predict(self, batch: StreamBatch) -> Prediction:
        import torch
        import torch.nn.functional as F

        assert self._model is not None
        x = self._batch_features(batch)
        samples: list = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                probs = F.softmax(self._model(x), dim=-1)
                samples.append(probs.cpu().numpy().astype(np.float64))
        mc_probs = np.stack(samples, axis=0)      # (S, B, K)
        return Prediction(
            probs=mc_probs.mean(axis=0), features=None, extras={"mc_probs": mc_probs}
        )


# ---------------------------------------------------------------------------
# Last-layer Laplace
# ---------------------------------------------------------------------------

@register("model", "audio_cnn_laplace")
class AudioCNNLaplace(_AudioCNNBase):
    """Frozen log-mel CNN (no dropout) + diagonal last-layer Laplace."""

    component_name = "audio_cnn_laplace"

    def __init__(
        self,
        dataset: str = "speech_commands",
        n_samples: int = 20,
        prior_precision: float = 1.0,
        data_root: str = "./artifacts/audio/speech_commands",
        ckpt_path: str = "./artifacts/models/audio_cnn_sc_kn_laplace.pt",
        train_corruptions: Optional[List[str]] = None,
        corruption_severity: int = 3,
        clean_fraction: float = 0.25,
        pretrain_epochs: int = 30,
        pretrain_batch_size: int = 128,
        pretrain_lr: float = 0.05,
        pretrain_seed: int = 0,
        seed: int = 0,
        allow_cuda: bool = True,
        pretrain_max_clips: int = 0,
        widths: Optional[List[int]] = None,
        arch: str = "cnn",
    ) -> None:
        super().__init__(
            dataset=dataset, data_root=data_root, ckpt_path=ckpt_path, p_drop=0.0,
            n_samples=n_samples, train_corruptions=train_corruptions,
            corruption_severity=corruption_severity, clean_fraction=clean_fraction,
            pretrain_epochs=pretrain_epochs, pretrain_batch_size=pretrain_batch_size,
            pretrain_lr=pretrain_lr, pretrain_seed=pretrain_seed, seed=seed,
            allow_cuda=allow_cuda,
            pretrain_max_clips=pretrain_max_clips, widths=widths, arch=arch,
        )
        self.prior_precision = float(prior_precision)
        self._posterior: dict | None = None

    def setup(self, spec: StreamSpec) -> None:
        model = self._load_or_train(spec, p_drop=0.0)
        model.eval()
        self._model = model
        print(f"[{self.component_name}] Computing diagonal GGN posterior over head…")
        self._posterior = _compute_laplace_posterior(
            model=model,
            dataset=self.dataset,
            data_root=Path(self.data_root),
            mean=self._norm[0],
            std=self._norm[1],
            train_corruptions=self.train_corruptions,
            corruption_severity=self.corruption_severity,
            batch_size=self.pretrain_batch_size,
            prior_precision=self.prior_precision,
            seed=self.pretrain_seed,
            allow_cuda=self.allow_cuda,
        )
        print(f"[{self.component_name}] Laplace posterior ready.")

    def predict(self, batch: StreamBatch) -> Prediction:
        import torch

        assert self._model is not None and self._posterior is not None
        rng = np.random.default_rng(self.seed + batch.index)
        x = self._batch_features(batch)
        with torch.no_grad():
            feats = self._model.embed(x).cpu().numpy().astype(np.float64)

        W_map, b_map = self._posterior["W_map"], self._posterior["b_map"]
        W_std = np.sqrt(self._posterior["W_var"])
        b_std = np.sqrt(self._posterior["b_var"])

        samples: list = []
        for _ in range(self.n_samples):
            W_s = W_map + rng.standard_normal(W_map.shape) * W_std
            b_s = b_map + rng.standard_normal(b_map.shape) * b_std
            logits = feats @ W_s.T + b_s[None, :]
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            samples.append((e / e.sum(axis=1, keepdims=True)).astype(np.float64))

        mc_probs = np.stack(samples, axis=0)
        return Prediction(
            probs=mc_probs.mean(axis=0), features=None, extras={"mc_probs": mc_probs}
        )
