"""Compact Vision Transformer with MC-Dropout and last-layer Laplace variants.

Why a small, from-scratch ViT
-----------------------------
This backbone exists so the known-vs-novel panels can be reproduced under an
*architecture* change while every other variable stays fixed.  It is therefore
trained exactly the way :mod:`uncertainty_driven_drift.components.resnet_cifar` trains
ResNet-20 — from scratch, on the same data, with the same known-corruption
augmentation — rather than fine-tuned from ImageNet weights.  Pretraining on a
much broader visual distribution would confound "ViT vs ResNet" with "saw
ImageNet vs did not", and an ImageNet backbone has plausibly already seen
blur/frost-like statistics, which weakens the "novel" in known-vs-novel.

Architecture
------------
A CIFAR-scale ViT, not a downscaled ImageNet one: ``patch_size=4`` (so a 32×32
image becomes 8×8 = 64 tokens and a 28×28 image becomes 7×7 = 49, keeping
spatial resolution that patch-16 would destroy at this input size), embedding
dim 192, depth 6, 3 heads, MLP ratio 2 — roughly 1.8 M parameters.  Pre-LN
blocks, a learnable class token, learnable position embeddings, and a
``Linear(dim → n_classes)`` head.

The backbone adapts to the stream's ``input_shape``, so the same component
serves both the RGB 32×32 CIFAR streams and the grayscale 28×28 MNIST-family
streams.  It is image-only by construction — see
:func:`uncertainty_driven_drift.components.image_spec.require_image_spec`.

Training
--------
ViTs need a different optimiser from ResNets: SGD at lr 0.1 diverges, so we use
AdamW (lr 1e-3, weight decay 0.05) with a linear warmup into cosine decay, plus
label smoothing 0.1.  As with the ResNet, each mini-batch is corrupted with a
random *known* corruption unless it falls in ``clean_fraction`` of batches left
clean — this is what keeps the model calibrated on known corruptions so that
epistemic uncertainty stays low there and spikes on novel ones.

Registered components
---------------------
* ``vit_mc_dropout`` — MC-Dropout: the MLP/head dropouts are kept in train mode
  at inference; ``n_samples`` stochastic passes → ``mc_probs`` in extras.
* ``vit_laplace``    — Diagonal last-layer Laplace: GGN posterior over the
  ``Linear(dim → n_classes)`` head; ``n_samples`` weight draws → ``mc_probs``.
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from uncertainty_driven_drift.components.image_spec import default_norm_stats, require_image_spec
from uncertainty_driven_drift.data.base import StreamBatch, StreamSpec
from uncertainty_driven_drift.models.base import Classifier, Prediction
from uncertainty_driven_drift.registry import register

# torchvision dataset class + expected (C, H, W) per supported source.
_DATASET_CLS = {
    "mnist": "MNIST",
    "fashion_mnist": "FashionMNIST",
    "kmnist": "KMNIST",
    "cifar10": "CIFAR10",
}
_DATASET_SHAPE = {
    "mnist": (1, 28, 28),
    "fashion_mnist": (1, 28, 28),
    "kmnist": (1, 28, 28),
    "cifar10": (3, 32, 32),
}


def _parse_sm(arch: str) -> Optional[Tuple[int, int]]:
    """``'sm_86'`` -> ``(8, 6)``; ``'sm_100'`` -> ``(10, 0)``. None if unparseable."""
    if not arch.startswith("sm_") or not arch[3:].isdigit():
        return None
    digits = arch[3:]
    return int(digits[:-1]), int(digits[-1])


def _cuda_arch_supported(capability: Tuple[int, int], arch_list: Sequence[str]) -> bool:
    """Can this torch build actually execute on a device of ``capability``?

    CUDA cubins are forward-compatible only within the same *major* compute
    capability, so a wheel built for ``sm_86`` runs on an sm_89 device (Ada) but
    a wheel whose lowest arch is ``sm_75`` cannot run on sm_70 (Volta) or sm_61
    (Pascal) at all.
    """
    major, minor = capability
    for arch in arch_list:
        parsed = _parse_sm(arch)
        if parsed and parsed[0] == major and parsed[1] <= minor:
            return True
    return False


def _select_device():
    """Best available device, refusing a GPU this torch build cannot run on.

    Recent PyTorch wheels dropped Pascal (sm_60/61) and Volta (sm_70): the binary
    contains no kernels for them, and every CUDA call fails with the fairly
    opaque ``no kernel image is available for execution on the device``. On a
    shared cluster you can land on such a card by chance, so name the problem —
    and the fix — instead of letting the training loop die on its first batch.
    """
    import torch

    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        arch_list = torch.cuda.get_arch_list()
        if _cuda_arch_supported(capability, arch_list):
            return torch.device("cuda")

        name = torch.cuda.get_device_name()
        sm = f"sm_{capability[0]}{capability[1]}"
        message = (
            f"GPU {name!r} has compute capability {sm}, which this PyTorch build "
            f"({torch.__version__}) has no kernels for — it ships {list(arch_list)}. "
            f"Every CUDA op would fail with 'no kernel image is available for execution "
            f"on the device'.\n"
            f"  Use a Turing-or-newer card (compute capability >= 7.5).\n"
            f"  (GTX 1080, TITAN Xp and Tesla P100 are Pascal and cannot work with this wheel.)\n"
            f"  To train on CPU anyway — far slower, but valid — set UB_ALLOW_CPU=1."
        )
        if os.environ.get("UB_ALLOW_CPU") == "1":
            print(f"[vit] WARNING: {message}\n[vit] UB_ALLOW_CPU=1 — falling back to CPU.")
            return torch.device("cpu")
        raise RuntimeError(message)

    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _apply_corruption_for(channels: int):
    """Return the corruption dispatcher matching the stream's channel count."""
    if channels == 3:
        from uncertainty_driven_drift.components.cifar_c_corruptions import apply_corruption
    else:
        from uncertainty_driven_drift.components.mnist_c_corruptions import apply_corruption
    return apply_corruption


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

def _build_vit(
    in_channels: int,
    image_size: Tuple[int, int],
    n_classes: int,
    patch_size: int = 4,
    dim: int = 192,
    depth: int = 6,
    n_heads: int = 3,
    mlp_ratio: float = 2.0,
    p_drop: float = 0.0,
):
    """CIFAR-scale ViT returning a module with ``embed()`` and ``forward()``.

    ``p_drop`` drives the MLP, position-embedding and pre-head dropouts — the
    layers MC-Dropout reactivates at inference.  Attention dropout is left at
    zero on purpose: ``nn.MultiheadAttention`` applies it via the module's own
    ``training`` flag rather than a child ``nn.Dropout``, so it could not be
    toggled surgically by :func:`_enable_mc_dropout`.
    """
    import torch
    import torch.nn as nn

    h, w = image_size
    if h % patch_size or w % patch_size:
        raise ValueError(
            f"patch_size={patch_size} does not tile a {h}×{w} image evenly."
        )
    n_patches = (h // patch_size) * (w // patch_size)
    hidden = int(dim * mlp_ratio)

    class Block(nn.Module):
        """Pre-LN transformer block."""

        def __init__(self) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, n_heads, dropout=0.0, batch_first=True)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.GELU(),
                nn.Dropout(p=p_drop),
                nn.Linear(hidden, dim),
                nn.Dropout(p=p_drop),
            )

        def forward(self, x):
            h_ = self.norm1(x)
            x = x + self.attn(h_, h_, h_, need_weights=False)[0]
            return x + self.mlp(self.norm2(x))

    class ViT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_embed = nn.Conv2d(
                in_channels, dim, kernel_size=patch_size, stride=patch_size
            )
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
            self.pos_drop = nn.Dropout(p=p_drop)
            self.blocks = nn.ModuleList([Block() for _ in range(depth)])
            self.norm = nn.LayerNorm(dim)
            self.dropout = nn.Dropout(p=p_drop)
            self.head = nn.Linear(dim, n_classes)

            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            self.apply(self._init_weights)

        @staticmethod
        def _init_weights(m) -> None:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        def embed(self, x):
            """Return the (B, dim) class-token feature, before dropout + head."""
            b = x.shape[0]
            t = self.patch_embed(x).flatten(2).transpose(1, 2)   # (B, N, dim)
            t = torch.cat([self.cls_token.expand(b, -1, -1), t], dim=1)
            t = self.pos_drop(t + self.pos_embed)
            for blk in self.blocks:
                t = blk(t)
            return self.norm(t)[:, 0]                            # (B, dim)

        def forward(self, x):
            return self.head(self.dropout(self.embed(x)))

    return ViT()


def _enable_mc_dropout(model) -> None:
    """Keep Dropout layers stochastic while the rest of the net stays in eval."""
    import torch.nn as nn

    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
            m.train()


# ---------------------------------------------------------------------------
# Data plumbing
# ---------------------------------------------------------------------------

def _train_loader(dataset_name: str, data_root: Path, batch_size: int, augment: bool):
    """Torchvision train loader for ``dataset_name``, images as float [0, 1]."""
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    cls_name = _DATASET_CLS.get(dataset_name)
    if cls_name is None:
        raise ValueError(
            f"Unsupported dataset_name={dataset_name!r}; "
            f"expected one of {sorted(_DATASET_CLS)}."
        )
    channels, h, _w = _DATASET_SHAPE[dataset_name]

    steps: list = []
    if augment:
        steps.append(transforms.RandomCrop(h, padding=4 if channels == 3 else 2))
        if channels == 3:
            # Digits and characters are not left-right symmetric; natural images are.
            steps.append(transforms.RandomHorizontalFlip())
    steps.append(transforms.ToTensor())

    ds = getattr(datasets, cls_name)(
        str(data_root), train=True, download=True, transform=transforms.Compose(steps)
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=augment, num_workers=0)


def _pretrain_vit(
    *,
    dataset_name: str,
    data_root: Path,
    ckpt_path: Path,
    arch: dict,
    n_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
    label_smoothing: float,
    seed: int,
    p_drop: float,
    train_corruptions: Optional[List[str]],
    corruption_severity: int,
    clean_fraction: float,
    mixup_alpha: float = 0.0,
) -> None:
    """Train the ViT from scratch with known-corruption augmentation.

    AdamW + linear-warmup-into-cosine, because ViTs do not tolerate the SGD
    lr=0.1 schedule the ResNet-20 backbone uses.

    ``mixup_alpha > 0`` additionally trains on blended image pairs with the
    correspondingly blended loss. For scenario S2 this is not a regularisation
    choice but the mechanism under test: it teaches the model to *report*
    ambiguity with a spread-but-confident posterior, which decouples aleatoric
    from epistemic uncertainty. Without it, blended inputs are simply
    out-of-distribution and epistemic rises alongside total entropy (measured:
    +7.96 vs +7.69 sigma, no separation); with it, only total entropy moves
    (+12.98 vs -0.11 sigma).
    """
    import torch
    from torch import optim

    torch.manual_seed(seed)
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    device = _select_device()

    channels, h, w = _DATASET_SHAPE[dataset_name]
    mean_np, std_np = default_norm_stats(channels)
    mean_t = torch.tensor(mean_np).view(1, -1, 1, 1).to(device)
    std_t = torch.tensor(std_np).view(1, -1, 1, 1).to(device)

    loader = _train_loader(dataset_name, data_root, batch_size, augment=True)
    model = _build_vit(
        in_channels=channels, image_size=(h, w), n_classes=n_classes,
        p_drop=p_drop, **arch,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ViT: {n_params/1e6:.2f}M params, device={device}")

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    steps_per_epoch = max(1, len(loader))
    total_steps = max(1, int(epochs) * steps_per_epoch)
    warmup_steps = min(int(warmup_epochs) * steps_per_epoch, total_steps - 1)

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_at)
    apply_corruption = _apply_corruption_for(channels) if train_corruptions else None

    model.train()
    for epoch in range(int(epochs)):
        running_loss, n_batches, n_correct, n_seen = 0.0, 0, 0, 0
        for xb, yb in loader:
            if apply_corruption is not None and py_rng.random() >= clean_fraction:
                corruption = py_rng.choice(train_corruptions)
                xb = torch.from_numpy(
                    apply_corruption(corruption, xb.numpy(), corruption_severity, np_rng)
                )
            xb = (xb.to(device) - mean_t) / std_t
            yb = yb.to(device)
            opt.zero_grad()
            if mixup_alpha > 0.0:
                lam = float(np_rng.beta(mixup_alpha, mixup_alpha))
                perm = torch.randperm(xb.shape[0], device=device)
                xb = lam * xb + (1.0 - lam) * xb[perm]
                logits = model(xb)
                loss = lam * loss_fn(logits, yb) + (1.0 - lam) * loss_fn(logits, yb[perm])
                # Accuracy against the dominant label; only a training-progress signal.
                target = yb if lam >= 0.5 else yb[perm]
            else:
                logits = model(xb)
                loss = loss_fn(logits, yb)
                target = yb
            loss.backward()
            opt.step()
            scheduler.step()
            running_loss += loss.item()
            n_batches += 1
            n_correct += (logits.argmax(dim=-1) == target).sum().item()
            n_seen += target.numel()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  epoch {epoch+1}/{epochs}  loss={running_loss/n_batches:.4f}  "
                f"train_acc={n_correct/max(1, n_seen):.4f}"
            )

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.to("cpu").state_dict(), str(ckpt_path))
    print(f"  saved → {ckpt_path}")


def _compute_laplace_posterior(
    *,
    model,
    dataset_name: str,
    data_root: Path,
    train_corruptions: Optional[List[str]],
    corruption_severity: int,
    batch_size: int,
    prior_precision: float,
    seed: int,
) -> dict:
    """Diagonal GGN Laplace posterior over the ``Linear(dim → K)`` head.

        H_W[k, j] = Σ h_j² · p_k · (1 − p_k)
        H_b[k]    = Σ        p_k · (1 − p_k)
        σ² = 1 / (H + prior_precision)

    Fitted on the training set with the same known-corruption augmentation the
    backbone saw, so the posterior width reflects the known-corruption regime.
    """
    import torch
    import torch.nn.functional as F

    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    device = _select_device()

    channels, _h, _w = _DATASET_SHAPE[dataset_name]
    mean_np, std_np = default_norm_stats(channels)
    mean_t = torch.tensor(mean_np).view(1, -1, 1, 1).to(device)
    std_t = torch.tensor(std_np).view(1, -1, 1, 1).to(device)

    loader = _train_loader(dataset_name, data_root, batch_size, augment=False)
    apply_corruption = _apply_corruption_for(channels) if train_corruptions else None

    K = model.head.out_features
    D = model.head.in_features
    H_W = np.zeros((K, D), dtype=np.float64)
    H_b = np.zeros(K, dtype=np.float64)

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            if apply_corruption is not None:
                corruption = py_rng.choice(train_corruptions)
                xb = torch.from_numpy(
                    apply_corruption(corruption, xb.numpy(), corruption_severity, np_rng)
                )
            xb = (xb.to(device) - mean_t) / std_t
            feats = model.embed(xb)                            # (B, D)
            probs = F.softmax(model.head(feats), dim=-1)       # (B, K)
            feats_np = feats.cpu().numpy().astype(np.float64)
            probs_np = probs.cpu().numpy().astype(np.float64)
            pk1pk = probs_np * (1.0 - probs_np)
            H_W += pk1pk.T @ (feats_np ** 2)
            H_b += pk1pk.sum(axis=0)

    model = model.to("cpu")
    return {
        "W_map": model.head.weight.detach().numpy().astype(np.float64),
        "b_map": model.head.bias.detach().numpy().astype(np.float64),
        "W_var": 1.0 / (H_W + prior_precision),
        "b_var": 1.0 / (H_b + prior_precision),
    }


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------

class _ViTBase(Classifier):
    """Config plumbing common to both ViT variants."""

    #: Registry name, used for guard messages and log lines.
    component = "vit"

    def __init__(
        self,
        *,
        dataset_name: str,
        data_root: str,
        ckpt_path: str,
        patch_size: int,
        dim: int,
        depth: int,
        n_heads: int,
        mlp_ratio: float,
        train_corruptions: Optional[List[str]],
        corruption_severity: int,
        clean_fraction: float,
        mixup_alpha: float,
        pretrain_epochs: int,
        pretrain_batch_size: int,
        pretrain_lr: float,
        pretrain_weight_decay: float,
        pretrain_warmup_epochs: int,
        label_smoothing: float,
        pretrain_seed: int,
        seed: int,
    ) -> None:
        self.dataset_name = str(dataset_name)
        self.data_root = str(data_root)
        self.ckpt_path = str(ckpt_path)
        self.patch_size = int(patch_size)
        self.dim = int(dim)
        self.depth = int(depth)
        self.n_heads = int(n_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.train_corruptions = list(train_corruptions) if train_corruptions else None
        self.corruption_severity = int(corruption_severity)
        self.clean_fraction = float(clean_fraction)
        self.mixup_alpha = float(mixup_alpha)
        self.pretrain_epochs = int(pretrain_epochs)
        self.pretrain_batch_size = int(pretrain_batch_size)
        self.pretrain_lr = float(pretrain_lr)
        self.pretrain_weight_decay = float(pretrain_weight_decay)
        self.pretrain_warmup_epochs = int(pretrain_warmup_epochs)
        self.label_smoothing = float(label_smoothing)
        self.pretrain_seed = int(pretrain_seed)
        self.seed = int(seed)
        self._model = None
        self._shape: Tuple[int, int, int] | None = None
        self._device = None      # resolved in setup(); predict() asserts on _model first

    @property
    def _arch(self) -> dict:
        return {
            "patch_size": self.patch_size,
            "dim": self.dim,
            "depth": self.depth,
            "n_heads": self.n_heads,
            "mlp_ratio": self.mlp_ratio,
        }

    def _validate(self, spec: StreamSpec) -> Tuple[int, int, int]:
        """Reject non-image streams and streams the checkpoint cannot serve."""
        expected = _DATASET_SHAPE.get(self.dataset_name)
        if expected is None:
            raise ValueError(
                f"{self.component}: unsupported dataset_name={self.dataset_name!r}; "
                f"expected one of {sorted(_DATASET_CLS)}."
            )
        shape = require_image_spec(
            spec,
            self.component,
            channels=(expected[0],),
            spatial=(expected[1:],),
        )
        self._shape = shape
        return shape

    def _load_backbone(self, spec: StreamSpec, p_drop: float):
        """Pretrain if the checkpoint is missing, then build and load it."""
        import torch

        channels, h, w = self._shape  # type: ignore[misc]
        ckpt = Path(self.ckpt_path)
        if not ckpt.exists():
            print(
                f"[{self.component}] Pretraining ViT on {self.dataset_name} "
                f"({self.pretrain_epochs} epochs)…"
            )
            _pretrain_vit(
                dataset_name=self.dataset_name,
                data_root=Path(self.data_root),
                ckpt_path=ckpt,
                arch=self._arch,
                n_classes=spec.n_classes,
                epochs=self.pretrain_epochs,
                batch_size=self.pretrain_batch_size,
                lr=self.pretrain_lr,
                weight_decay=self.pretrain_weight_decay,
                warmup_epochs=self.pretrain_warmup_epochs,
                label_smoothing=self.label_smoothing,
                seed=self.pretrain_seed,
                p_drop=p_drop,
                train_corruptions=self.train_corruptions,
                corruption_severity=self.corruption_severity,
                clean_fraction=self.clean_fraction,
                mixup_alpha=self.mixup_alpha,
            )
        model = _build_vit(
            in_channels=channels, image_size=(h, w), n_classes=spec.n_classes,
            p_drop=p_drop, **self._arch,
        )
        model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
        return model

    def _normalize(self, x_np: np.ndarray):
        """(B, C, H, W) float [0,1] → normalised tensor on the inference device."""
        import torch

        channels = self._shape[0]  # type: ignore[index]
        mean_np, std_np = default_norm_stats(channels)
        x_t = torch.from_numpy(np.ascontiguousarray(x_np)).float()
        mean_t = torch.tensor(mean_np).view(1, -1, 1, 1)
        std_t = torch.tensor(std_np).view(1, -1, 1, 1)
        return ((x_t - mean_t) / std_t).to(self._device)

    def observe(self, batch: StreamBatch, prediction: Prediction) -> None:
        """Frozen backbone — observing labels is a no-op by design."""
        del batch, prediction


# ---------------------------------------------------------------------------
# MC-Dropout variant
# ---------------------------------------------------------------------------

@register("model", "vit_mc_dropout")
class ViTMCDropout(_ViTBase):
    """Frozen from-scratch ViT with MC-Dropout, for image streams.

    Parameters
    ----------
    n_samples :
        Stochastic forward passes per batch at inference time.
    p_drop :
        Dropout probability in the MLP blocks, position embedding and pre-head,
        shared between training and MC inference.
    dataset_name :
        Which torchvision source to pretrain on: ``cifar10`` (3×32×32) or
        ``mnist`` / ``fashion_mnist`` / ``kmnist`` (1×28×28). Must match the
        stream's ``input_shape``.
    patch_size, dim, depth, n_heads, mlp_ratio :
        Backbone geometry. Defaults are the CIFAR-scale configuration.
    train_corruptions, corruption_severity, clean_fraction :
        Known-corruption augmentation — a random known corruption is applied to
        each mini-batch unless it falls in ``clean_fraction`` of clean batches.
    pretrain_* , label_smoothing :
        Pretraining hyperparameters; ignored when the checkpoint already exists.
    seed :
        Seeds the torch RNG driving dropout, so MC sampling is reproducible.
    """

    component = "vit_mc_dropout"

    def __init__(
        self,
        n_samples: int = 20,
        p_drop: float = 0.1,
        dataset_name: str = "cifar10",
        data_root: str = "./artifacts/cifar10",
        ckpt_path: str = "./artifacts/models/vit_cifar_kn_mc.pt",
        patch_size: int = 4,
        dim: int = 192,
        depth: int = 6,
        n_heads: int = 3,
        mlp_ratio: float = 2.0,
        train_corruptions: Optional[List[str]] = None,
        corruption_severity: int = 3,
        clean_fraction: float = 0.0,
        mixup_alpha: float = 0.0,
        pretrain_epochs: int = 200,
        pretrain_batch_size: int = 128,
        pretrain_lr: float = 1e-3,
        pretrain_weight_decay: float = 0.05,
        pretrain_warmup_epochs: int = 10,
        label_smoothing: float = 0.1,
        pretrain_seed: int = 0,
        seed: int = 0,
    ) -> None:
        super().__init__(
            dataset_name=dataset_name, data_root=data_root, ckpt_path=ckpt_path,
            patch_size=patch_size, dim=dim, depth=depth, n_heads=n_heads,
            mlp_ratio=mlp_ratio, train_corruptions=train_corruptions,
            corruption_severity=corruption_severity, clean_fraction=clean_fraction,
            mixup_alpha=mixup_alpha, pretrain_epochs=pretrain_epochs, pretrain_batch_size=pretrain_batch_size,
            pretrain_lr=pretrain_lr, pretrain_weight_decay=pretrain_weight_decay,
            pretrain_warmup_epochs=pretrain_warmup_epochs,
            label_smoothing=label_smoothing, pretrain_seed=pretrain_seed, seed=seed,
        )
        self.n_samples = int(n_samples)
        self.p_drop = float(p_drop)

    def setup(self, spec: StreamSpec) -> None:
        import torch

        self._validate(spec)
        torch.manual_seed(self.seed)
        model = self._load_backbone(spec, p_drop=self.p_drop)
        _enable_mc_dropout(model)
        self._device = _select_device()
        self._model = model.to(self._device)

    def predict(self, batch: StreamBatch) -> Prediction:
        import torch
        import torch.nn.functional as F

        assert self._model is not None
        x = self._normalize(batch.x)
        samples: list[np.ndarray] = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                probs = F.softmax(self._model(x), dim=-1)
                samples.append(probs.cpu().numpy().astype(np.float64))

        mc_probs = np.stack(samples, axis=0)          # (S, B, K)
        return Prediction(
            probs=mc_probs.mean(axis=0),
            features=None,
            extras={"mc_probs": mc_probs},
        )


# ---------------------------------------------------------------------------
# Laplace variant
# ---------------------------------------------------------------------------

@register("model", "vit_laplace")
class ViTLaplace(_ViTBase):
    """Frozen from-scratch ViT + diagonal last-layer Laplace, for image streams.

    Shares the backbone geometry of :class:`ViTMCDropout` but is trained with
    ``p_drop=0.0`` for a clean MAP estimate, after which a diagonal GGN
    posterior is fitted over the ``Linear(dim → n_classes)`` head.

    Parameters
    ----------
    prior_precision :
        Precision of the zero-mean isotropic Gaussian prior over head weights.
    n_samples :
        Head-weight samples drawn per batch at inference time.

    Remaining parameters match :class:`ViTMCDropout`.
    """

    component = "vit_laplace"

    def __init__(
        self,
        n_samples: int = 20,
        prior_precision: float = 1.0,
        dataset_name: str = "cifar10",
        data_root: str = "./artifacts/cifar10",
        ckpt_path: str = "./artifacts/models/vit_cifar_kn_laplace.pt",
        patch_size: int = 4,
        dim: int = 192,
        depth: int = 6,
        n_heads: int = 3,
        mlp_ratio: float = 2.0,
        train_corruptions: Optional[List[str]] = None,
        corruption_severity: int = 3,
        clean_fraction: float = 0.0,
        mixup_alpha: float = 0.0,
        pretrain_epochs: int = 200,
        pretrain_batch_size: int = 128,
        pretrain_lr: float = 1e-3,
        pretrain_weight_decay: float = 0.05,
        pretrain_warmup_epochs: int = 10,
        label_smoothing: float = 0.1,
        pretrain_seed: int = 0,
        seed: int = 0,
    ) -> None:
        super().__init__(
            dataset_name=dataset_name, data_root=data_root, ckpt_path=ckpt_path,
            patch_size=patch_size, dim=dim, depth=depth, n_heads=n_heads,
            mlp_ratio=mlp_ratio, train_corruptions=train_corruptions,
            corruption_severity=corruption_severity, clean_fraction=clean_fraction,
            mixup_alpha=mixup_alpha, pretrain_epochs=pretrain_epochs, pretrain_batch_size=pretrain_batch_size,
            pretrain_lr=pretrain_lr, pretrain_weight_decay=pretrain_weight_decay,
            pretrain_warmup_epochs=pretrain_warmup_epochs,
            label_smoothing=label_smoothing, pretrain_seed=pretrain_seed, seed=seed,
        )
        self.n_samples = int(n_samples)
        self.prior_precision = float(prior_precision)
        self._posterior: dict | None = None

    def setup(self, spec: StreamSpec) -> None:
        self._validate(spec)
        model = self._load_backbone(spec, p_drop=0.0)
        model.eval()
        self._device = _select_device()

        print(f"[{self.component}] Computing diagonal GGN posterior over head…")
        self._posterior = _compute_laplace_posterior(
            model=model,
            dataset_name=self.dataset_name,
            data_root=Path(self.data_root),
            train_corruptions=self.train_corruptions,
            corruption_severity=self.corruption_severity,
            batch_size=self.pretrain_batch_size,
            prior_precision=self.prior_precision,
            seed=self.pretrain_seed,
        )
        self._model = model.to(self._device)
        print(f"[{self.component}] Laplace posterior ready.")

    def predict(self, batch: StreamBatch) -> Prediction:
        import torch

        assert self._model is not None and self._posterior is not None
        rng = np.random.default_rng(self.seed + batch.index)

        x = self._normalize(batch.x)
        with torch.no_grad():
            feats_np = self._model.embed(x).cpu().numpy().astype(np.float64)  # (B, D)

        W_map = self._posterior["W_map"]           # (K, D)
        b_map = self._posterior["b_map"]           # (K,)
        W_std = np.sqrt(self._posterior["W_var"])  # (K, D)
        b_std = np.sqrt(self._posterior["b_var"])  # (K,)

        samples: list[np.ndarray] = []
        for _ in range(self.n_samples):
            W_s = W_map + rng.standard_normal(W_map.shape) * W_std
            b_s = b_map + rng.standard_normal(b_map.shape) * b_std
            logits = feats_np @ W_s.T + b_s[None, :]
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            samples.append((e / e.sum(axis=1, keepdims=True)).astype(np.float64))

        mc_probs = np.stack(samples, axis=0)       # (S, B, K)
        return Prediction(
            probs=mc_probs.mean(axis=0),
            features=None,
            extras={"mc_probs": mc_probs},
        )
