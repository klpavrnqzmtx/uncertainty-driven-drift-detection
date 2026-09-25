"""ResNet-20 for the MNIST-family streams, with MC-Dropout and Laplace variants.

This is the *same* small-image ResNet-20 (He et al. 2016) used for CIFAR-10 —
3×3 stem (16 filters, no 7×7 conv or max-pool), three residual stages at
16/32/64 filters, global average pool, ~270 K parameters — adapted to a single
grayscale channel (``in_channels=1``) and 28×28 inputs.  It exists so the
known-vs-novel MNIST panels can be reproduced under a *CNN depth* change
(LeNet-5 → ResNet-20) alongside the ViT arm, with the data, corruption split
and uncertainty machinery held fixed.

Training mirrors the CIFAR ResNet arm (SGD + cosine, known-corruption
augmentation) but on grayscale MNIST-C corruptions and MNIST normalisation.

Normalisation: **GroupNorm**, not BatchNorm.  At MNIST-C severity 3 the
brightness corruption shifts the input mean by a large amount; eval-mode
BatchNorm — whose running statistics are averaged over the training corruption
mix — then mismatches a pure single-corruption test batch and collapses to
~70% accuracy on brightness while training loss stays low (BN uses batch stats
during training).  GroupNorm normalises per sample, has no running statistics,
and removes the pathology (like the ViT's LayerNorm and the norm-free LeNet).
CIFAR-10 uses BatchNorm and severity 1, where the shift is small enough not to
trigger this, so its arm is unchanged.

Registered components
---------------------
* ``resnet_mc_dropout_mnist`` — MC-Dropout: 20+ stochastic passes → ``mc_probs``.
* ``resnet_laplace_mnist``    — Diagonal last-layer Laplace over the ``Linear(64
  → 10)`` head; weight samples → ``mc_probs``.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional

import numpy as np

from uncertainty_driven_drift.components.image_spec import default_norm_stats, require_image_spec
from uncertainty_driven_drift.components.resnet_cifar import _build_resnet20, _enable_mc_dropout
from uncertainty_driven_drift.data.base import StreamBatch, StreamSpec
from uncertainty_driven_drift.models.base import Classifier, Prediction
from uncertainty_driven_drift.registry import register

_MNIST_SHAPE = (1, 28, 28)
_MEAN, _STD = default_norm_stats(1)   # MNIST train-set grayscale stats


def _select_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _norm_tensors(device=None):
    import torch
    mean_t = torch.tensor(_MEAN).view(1, -1, 1, 1)
    std_t = torch.tensor(_STD).view(1, -1, 1, 1)
    if device is not None:
        mean_t, std_t = mean_t.to(device), std_t.to(device)
    return mean_t, std_t


# ---------------------------------------------------------------------------
# Pre-training
# ---------------------------------------------------------------------------

def _pretrain_resnet20_mnist(
    data_root: Path,
    ckpt_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    p_drop: float,
    train_corruptions: Optional[List[str]],
    corruption_severity: int,
    clean_fraction: float = 0.0,
) -> None:
    """Train the grayscale ResNet-20 on MNIST with known-corruption augmentation."""
    import torch
    from torch import optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    torch.manual_seed(seed)
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    device = _select_device()
    mean_t, std_t = _norm_tensors(device)

    # Digits are not left-right symmetric, so no horizontal flip; small pad-crop
    # jitter only. Corruptions are applied to the [0, 1] tensor after ToTensor.
    tfm = transforms.Compose([
        transforms.RandomCrop(28, padding=2),
        transforms.ToTensor(),
    ])
    ds = datasets.MNIST(str(data_root), train=True, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = _build_resnet20(n_classes=10, p_drop=p_drop, in_channels=1, norm="gn").to(device)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = torch.nn.CrossEntropyLoss()

    if train_corruptions:
        from uncertainty_driven_drift.components.mnist_c_corruptions import apply_corruption

    model.train()
    for epoch in range(int(epochs)):
        running_loss, n_batches = 0.0, 0
        for xb, yb in loader:
            if train_corruptions and py_rng.random() >= clean_fraction:
                corruption = py_rng.choice(train_corruptions)
                xb = torch.from_numpy(
                    apply_corruption(corruption, xb.numpy(), corruption_severity, np_rng)
                )
            xb = (xb.to(device) - mean_t) / std_t
            yb = yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            n_batches += 1
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{epochs}  loss={running_loss/n_batches:.4f}")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.to("cpu").state_dict(), str(ckpt_path))
    print(f"  saved → {ckpt_path}")


def _compute_laplace_posterior_mnist(
    model,
    data_root: Path,
    train_corruptions: Optional[List[str]],
    corruption_severity: int,
    batch_size: int,
    prior_precision: float,
    seed: int,
) -> dict:
    """Diagonal GGN Laplace posterior over the ``Linear(64 → 10)`` head.

        H_W[k, j] = Σ h_j² · p_k · (1 − p_k) ;  σ² = 1 / (H + prior_precision)
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    mean_t, std_t = _norm_tensors()

    ds = datasets.MNIST(str(data_root), train=True, download=True,
                        transform=transforms.ToTensor())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    K = model.head.out_features
    D = model.head.in_features
    H_W = np.zeros((K, D), dtype=np.float64)
    H_b = np.zeros(K, dtype=np.float64)

    if train_corruptions:
        from uncertainty_driven_drift.components.mnist_c_corruptions import apply_corruption

    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            if train_corruptions:
                corruption = py_rng.choice(train_corruptions)
                xb = torch.from_numpy(
                    apply_corruption(corruption, xb.numpy(), corruption_severity, np_rng)
                )
            xb = (xb - mean_t) / std_t
            feats = model.embed(xb)                          # (B, 64)
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


def _normalize_batch(x_np: np.ndarray):
    """(B, 1, 28, 28) float [0,1] → MNIST-normalised CPU tensor."""
    import torch
    x_t = torch.from_numpy(np.ascontiguousarray(x_np)).float()
    mean_t, std_t = _norm_tensors()
    return (x_t - mean_t) / std_t


# ---------------------------------------------------------------------------
# MC-Dropout classifier
# ---------------------------------------------------------------------------

@register("model", "resnet_mc_dropout_mnist")
class ResNetMCDropoutMNIST(Classifier):
    """Frozen grayscale ResNet-20 with MC-Dropout for MNIST-family streams."""

    def __init__(
        self,
        n_samples: int = 30,
        p_drop: float = 0.3,
        data_root: str = "./artifacts/mnist",
        ckpt_path: str = "./artifacts/models/resnet20_mnist_kn_mc.pt",
        train_corruptions: Optional[List[str]] = None,
        corruption_severity: int = 3,
        clean_fraction: float = 0.0,
        pretrain_epochs: int = 40,
        pretrain_batch_size: int = 128,
        pretrain_lr: float = 0.1,
        pretrain_seed: int = 0,
        seed: int = 0,
    ) -> None:
        self.n_samples = int(n_samples)
        self.p_drop = float(p_drop)
        self.data_root = str(data_root)
        self.ckpt_path = str(ckpt_path)
        self.train_corruptions = list(train_corruptions) if train_corruptions else None
        self.corruption_severity = int(corruption_severity)
        self.clean_fraction = float(clean_fraction)
        self.pretrain_epochs = int(pretrain_epochs)
        self.pretrain_batch_size = int(pretrain_batch_size)
        self.pretrain_lr = float(pretrain_lr)
        self.pretrain_seed = int(pretrain_seed)
        self.seed = int(seed)
        self._model = None

    def setup(self, spec: StreamSpec) -> None:
        import torch
        require_image_spec(
            spec, "resnet_mc_dropout_mnist",
            channels=(1,), spatial=(_MNIST_SHAPE[1:],), n_classes=10,
        )
        ckpt = Path(self.ckpt_path)
        if not ckpt.exists():
            print(f"[resnet_mc_dropout_mnist] Pretraining ResNet-20 ({self.pretrain_epochs} epochs)…")
            _pretrain_resnet20_mnist(
                data_root=Path(self.data_root), ckpt_path=ckpt,
                epochs=self.pretrain_epochs, batch_size=self.pretrain_batch_size,
                lr=self.pretrain_lr, seed=self.pretrain_seed, p_drop=self.p_drop,
                train_corruptions=self.train_corruptions,
                corruption_severity=self.corruption_severity,
                clean_fraction=self.clean_fraction,
            )
        torch.manual_seed(self.seed)
        model = _build_resnet20(n_classes=10, p_drop=self.p_drop, in_channels=1, norm="gn")
        model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
        _enable_mc_dropout(model)
        self._model = model

    def predict(self, batch: StreamBatch) -> Prediction:
        import torch
        import torch.nn.functional as F
        assert self._model is not None
        x = _normalize_batch(batch.x)
        samples: list = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                probs = F.softmax(self._model(x), dim=-1).numpy().astype(np.float64)
                samples.append(probs)
        mc_probs = np.stack(samples, axis=0)   # (S, B, 10)
        return Prediction(
            probs=mc_probs.mean(axis=0), features=None,
            extras={"mc_probs": mc_probs},
        )

    def observe(self, batch: StreamBatch, prediction: Prediction) -> None:
        del batch, prediction


# ---------------------------------------------------------------------------
# Diagonal Laplace classifier
# ---------------------------------------------------------------------------

@register("model", "resnet_laplace_mnist")
class ResNetLaplaceMNIST(Classifier):
    """Frozen grayscale ResNet-20 + diagonal last-layer Laplace for MNIST."""

    def __init__(
        self,
        n_samples: int = 30,
        prior_precision: float = 1.0,
        data_root: str = "./artifacts/mnist",
        ckpt_path: str = "./artifacts/models/resnet20_mnist_kn_laplace.pt",
        train_corruptions: Optional[List[str]] = None,
        corruption_severity: int = 3,
        clean_fraction: float = 0.0,
        pretrain_epochs: int = 40,
        pretrain_batch_size: int = 128,
        pretrain_lr: float = 0.1,
        pretrain_seed: int = 0,
        seed: int = 0,
    ) -> None:
        self.n_samples = int(n_samples)
        self.prior_precision = float(prior_precision)
        self.data_root = str(data_root)
        self.ckpt_path = str(ckpt_path)
        self.train_corruptions = list(train_corruptions) if train_corruptions else None
        self.corruption_severity = int(corruption_severity)
        self.clean_fraction = float(clean_fraction)
        self.pretrain_epochs = int(pretrain_epochs)
        self.pretrain_batch_size = int(pretrain_batch_size)
        self.pretrain_lr = float(pretrain_lr)
        self.pretrain_seed = int(pretrain_seed)
        self.seed = int(seed)
        self._model = None
        self._posterior: dict | None = None

    def setup(self, spec: StreamSpec) -> None:
        import torch
        require_image_spec(
            spec, "resnet_laplace_mnist",
            channels=(1,), spatial=(_MNIST_SHAPE[1:],), n_classes=10,
        )
        ckpt = Path(self.ckpt_path)
        if not ckpt.exists():
            print(f"[resnet_laplace_mnist] Pretraining ResNet-20 no-dropout ({self.pretrain_epochs} epochs)…")
            _pretrain_resnet20_mnist(
                data_root=Path(self.data_root), ckpt_path=ckpt,
                epochs=self.pretrain_epochs, batch_size=self.pretrain_batch_size,
                lr=self.pretrain_lr, seed=self.pretrain_seed, p_drop=0.0,
                train_corruptions=self.train_corruptions,
                corruption_severity=self.corruption_severity,
                clean_fraction=self.clean_fraction,
            )
        model = _build_resnet20(n_classes=10, p_drop=0.0, in_channels=1, norm="gn")
        model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
        model.eval()
        self._model = model

        print("[resnet_laplace_mnist] Computing diagonal GGN posterior over head…")
        self._posterior = _compute_laplace_posterior_mnist(
            model=self._model, data_root=Path(self.data_root),
            train_corruptions=self.train_corruptions,
            corruption_severity=self.corruption_severity,
            batch_size=self.pretrain_batch_size,
            prior_precision=self.prior_precision, seed=self.pretrain_seed,
        )
        print("[resnet_laplace_mnist] Laplace posterior ready.")

    def predict(self, batch: StreamBatch) -> Prediction:
        import torch
        assert self._model is not None and self._posterior is not None
        rng = np.random.default_rng(self.seed + batch.index)
        x = _normalize_batch(batch.x)
        with torch.no_grad():
            feats_np = self._model.embed(x).numpy().astype(np.float64)   # (B, 64)

        W_map = self._posterior["W_map"]
        b_map = self._posterior["b_map"]
        W_std = np.sqrt(self._posterior["W_var"])
        b_std = np.sqrt(self._posterior["b_var"])

        samples: list = []
        for _ in range(self.n_samples):
            W_s = W_map + rng.standard_normal(W_map.shape) * W_std
            b_s = b_map + rng.standard_normal(b_map.shape) * b_std
            logits = feats_np @ W_s.T + b_s[None, :]
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            samples.append((e / e.sum(axis=1, keepdims=True)).astype(np.float64))
        mc_probs = np.stack(samples, axis=0)   # (S, B, 10)
        return Prediction(
            probs=mc_probs.mean(axis=0), features=None,
            extras={"mc_probs": mc_probs},
        )

    def observe(self, batch: StreamBatch, prediction: Prediction) -> None:
        del batch, prediction
