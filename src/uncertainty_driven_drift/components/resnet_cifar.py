"""ResNet-20 for CIFAR-10 with MC-Dropout and last-layer Laplace variants.

Architecture
------------
ResNet-20 (He et al. 2016): 3×3 first conv (no stride, no maxpool), three
residual stages at 16/32/64 filters with 3 BasicBlocks each, global average
pool, optional Dropout, then a Linear(64 → 10) head.  Total: ~270 K params.

Training
--------
Both variants share the same backbone architecture (p_drop controls the
dropout rate; set to 0.0 for the Laplace checkpoint).  Training mixes in
random known corruptions per mini-batch so the model stays calibrated when
those corruptions appear at inference time — same philosophy as the MNIST-C
LeNet backbone.

Registered components
---------------------
* ``resnet_mc_dropout_cifar`` — MC-Dropout: dropout kept in train mode at
  inference; 20 stochastic forward passes → ``mc_probs`` in extras.
* ``resnet_laplace_cifar``    — Diagonal last-layer Laplace: GGN posterior
  over the Linear(64 → 10) head; 20 weight samples → ``mc_probs`` in extras.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional

import numpy as np

from uncertainty_driven_drift.components.image_spec import require_image_spec
from uncertainty_driven_drift.data.base import StreamBatch, StreamSpec
from uncertainty_driven_drift.models.base import Classifier, Prediction
from uncertainty_driven_drift.registry import register

# CIFAR-10 channel statistics (computed on training set)
_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
_STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

# CIFAR-100 channel statistics — used by the resnet_*_cifar100 arms, which reuse
# the same ResNet-20 architecture and corruption pipeline via a `dataset` switch.
_MEAN100 = np.array([0.5071, 0.4865, 0.4409], dtype=np.float32)
_STD100  = np.array([0.2673, 0.2564, 0.2762], dtype=np.float32)


def _dataset_stats(dataset: str):
    """(mean, std) normalisation stats for ``cifar10`` (default) or ``cifar100``."""
    return (_MEAN100, _STD100) if dataset == "cifar100" else (_MEAN, _STD)


def _torchvision_cifar(datasets_mod, dataset: str):
    """torchvision dataset class for ``cifar10`` (default) or ``cifar100``."""
    return datasets_mod.CIFAR100 if dataset == "cifar100" else datasets_mod.CIFAR10


# ---------------------------------------------------------------------------
# ResNet-20 architecture
# ---------------------------------------------------------------------------

def _build_resnet20(n_classes: int = 10, p_drop: float = 0.0, in_channels: int = 3,
                    norm: str = "bn"):
    """Small-image ResNet-20 (He et al. 2016).

    ``in_channels`` defaults to 3 (RGB CIFAR-10); pass 1 for grayscale
    MNIST-family streams. The 3×3 stem + global average pool make the
    backbone size-agnostic, so the same definition serves 32×32 and 28×28.
    """
    import torch.nn as nn
    import torch.nn.functional as F

    def _norm(planes: int):
        """Normalisation layer. ``bn`` = BatchNorm (default, CIFAR); ``gn`` =
        GroupNorm — per-sample, so it has no running statistics to go stale
        under a large input-mean shift (e.g. severity-3 brightness on MNIST,
        where eval-mode BatchNorm mismatches a pure-single-corruption batch)."""
        if norm == "gn":
            groups = 8 if planes % 8 == 0 else (4 if planes % 4 == 0 else 1)
            return nn.GroupNorm(groups, planes)
        return nn.BatchNorm2d(planes)

    class BasicBlock(nn.Module):
        def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
            self.bn1 = _norm(planes)
            self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
            self.bn2 = _norm(planes)
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                    _norm(planes),
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return F.relu(out + self.shortcut(x))

    def _make_layer(in_planes: int, planes: int, n: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_planes, planes, stride=stride)]
        for _ in range(n - 1):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)

    class ResNet20(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1  = nn.Conv2d(in_channels, 16, 3, stride=1, padding=1, bias=False)
            self.bn1    = _norm(16)
            self.layer1 = _make_layer(16, 16, n=3, stride=1)
            self.layer2 = _make_layer(16, 32, n=3, stride=2)
            self.layer3 = _make_layer(32, 64, n=3, stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(p=p_drop)
            self.head   = nn.Linear(64, n_classes)

        def _backbone(self, x):
            import torch.nn.functional as _F
            out = _F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            return self.avgpool(out).flatten(1)   # (B, 64)

        def embed(self, x):
            """Return 64-D features (before dropout and head)."""
            return self._backbone(x)

        def forward(self, x):
            return self.head(self.dropout(self._backbone(x)))

    return ResNet20()


# ---------------------------------------------------------------------------
# Pre-training
# ---------------------------------------------------------------------------

def _pretrain_resnet20(
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
    mixup_alpha: float = 0.0,
    n_classes: int = 10,
    dataset: str = "cifar10",
    build_fn=None,
    label_smoothing: float = 0.0,
    lr_schedule: str = "cosine",
    lr_milestones_frac: tuple = (0.3, 0.6, 0.8),
    lr_drop: float = 0.2,
    warmup_epochs: int = 0,
    nesterov: bool = False,
    grad_clip: float = 0.0,
) -> None:
    """Train ResNet-20 on CIFAR-10 with optional known-corruption augmentation.

    When ``train_corruptions`` is set, each mini-batch is corrupted with a
    random known corruption *unless* it falls in the ``clean_fraction`` of
    batches left uncorrupted — mixing clean images in keeps the base
    representation strong so easy corruptions stay near clean accuracy.

    ``mixup_alpha > 0`` additionally trains on blended pairs against the
    convex-combination loss, so the model learns to *report* label ambiguity
    with a spread-but-confident posterior. For scenario S2 this is the
    mechanism under test, not a regularisation choice: it is what decouples
    aleatoric from epistemic uncertainty (see components/cifar_ambiguity.py).

    ``lr_schedule``, ``warmup_epochs``, ``nesterov`` and ``grad_clip`` default
    to the original ResNet-20 recipe (plain cosine decay, no warmup, no
    Nesterov, no clipping) so existing ResNet-20/ViT configs are untouched.
    They were added for WRN-28-10, on the theory that from-scratch training a
    36M-param net under heavy per-batch corruption augmentation plateaus under
    plain cosine annealing. MEASURED 12 Sep: that theory was wrong. Trained
    both ways and evaluated identically, plain cosine (v3) gives clean 0.760 /
    known-corruption 0.683 and this recipe (v4) gives 0.753 / 0.676 -- no
    improvement, marginally worse. 100 and 200 epochs agreed originally because
    ~0.68 is simply where this model lands on this task, not because of an
    optimisation pathology. The knobs are kept (they are the WRN paper's own
    recipe, and harmless) but they fix nothing; the defaults above still
    reproduce the original ResNet-20/ViT behaviour exactly.
    """
    import torch
    from torch import optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    torch.manual_seed(seed)
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    _mean, _std = _dataset_stats(dataset)
    mean_t = torch.tensor(_mean).view(1, 3, 1, 1).to(device)
    std_t  = torch.tensor(_std).view(1, 3, 1, 1).to(device)

    tfm = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])
    ds = _torchvision_cifar(datasets, dataset)(str(data_root), train=True, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    _mk = build_fn if build_fn is not None else _build_resnet20
    model = _mk(n_classes=n_classes, p_drop=p_drop).to(device)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4,
                    nesterov=nesterov)

    main_epochs = max(1, epochs - warmup_epochs)
    if lr_schedule == "step":
        milestones = [max(1, int(main_epochs * f)) for f in lr_milestones_frac]
        main_sched = optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=lr_drop)
    else:
        main_sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=main_epochs)
    if warmup_epochs > 0:
        warmup_sched = optim.lr_scheduler.LinearLR(
            opt, start_factor=1.0 / warmup_epochs, total_iters=warmup_epochs)
        scheduler = optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup_sched, main_sched], milestones=[warmup_epochs])
    else:
        scheduler = main_sched
    # Label smoothing curbs the over-confidence that comes with memorising the
    # training set — a WRN-28-10 driven to ~0.05 train loss on CIFAR-100 has no
    # epistemic signal left to give, because every MC sample agrees.
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))

    if train_corruptions:
        from uncertainty_driven_drift.components.cifar_c_corruptions import apply_corruption

    model.train()
    for epoch in range(int(epochs)):
        running_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            if train_corruptions and py_rng.random() >= clean_fraction:
                corruption = py_rng.choice(train_corruptions)
                xb_np = xb.numpy()               # (B, 3, 32, 32) float32 [0,1]
                xb_np = apply_corruption(corruption, xb_np, corruption_severity, np_rng)
                xb = torch.from_numpy(xb_np)
            xb = ((xb.to(device) - mean_t) / std_t)
            yb = yb.to(device)
            opt.zero_grad()
            if mixup_alpha > 0.0:
                lam = float(np_rng.beta(mixup_alpha, mixup_alpha))
                perm = torch.randperm(xb.shape[0], device=device)
                xb = lam * xb + (1.0 - lam) * xb[perm]
                out = model(xb)
                loss = lam * loss_fn(out, yb) + (1.0 - lam) * loss_fn(out, yb[perm])
            else:
                loss = loss_fn(model(xb), yb)
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            running_loss += loss.item()
            n_batches += 1
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{epochs}  loss={running_loss/n_batches:.4f}")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.to("cpu").state_dict(), str(ckpt_path))
    print(f"  saved → {ckpt_path}")


# ---------------------------------------------------------------------------
# Laplace posterior
# ---------------------------------------------------------------------------

def _compute_laplace_posterior(
    model,
    data_root: Path,
    train_corruptions: Optional[List[str]],
    corruption_severity: int,
    batch_size: int,
    prior_precision: float,
    seed: int,
    dataset: str = "cifar10",
    calibrate: bool = True,
) -> dict:
    """Diagonal GGN Laplace posterior over the Linear(64 → 10) head.

        H_W[k, j] = Σ h_j² · p_k · (1 − p_k)
        H_b[k]    = Σ        p_k · (1 − p_k)
        σ²_W = 1 / (H_W + prior_precision)
        σ²_b = 1 / (H_b + prior_precision)

    Fitted on the CIFAR-10 training set with the same known-corruption
    augmentation used during backbone training.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    ds = _torchvision_cifar(datasets, dataset)(str(data_root), train=True, download=True,
                           transform=transforms.ToTensor())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Run the GGN accumulation wherever the model already lives (the caller moves
    # it to the accelerator). On CPU this pass over the whole train set is the
    # dominant cost for a large backbone like WRN-28-10.
    device = next(model.parameters()).device
    _mean, _std = _dataset_stats(dataset)
    mean_t = torch.tensor(_mean).view(1, 3, 1, 1).to(device)
    std_t  = torch.tensor(_std).view(1, 3, 1, 1).to(device)

    K = model.head.out_features   # 10 (CIFAR-10) / 100 (CIFAR-100)
    H_size = model.head.in_features
    H_W = np.zeros((K, H_size), dtype=np.float64)
    H_b = np.zeros(K, dtype=np.float64)
    feat_cache, label_cache, cache_cap = [], [], 8192

    if train_corruptions:
        from uncertainty_driven_drift.components.cifar_c_corruptions import apply_corruption

    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            if train_corruptions:
                corruption = py_rng.choice(train_corruptions)
                xb_np = xb.numpy()
                xb_np = apply_corruption(corruption, xb_np, corruption_severity, np_rng)
                xb = torch.from_numpy(xb_np)
            xb = (xb.to(device) - mean_t) / std_t
            feats  = model.embed(xb)               # (B, D)
            logits = model.head(feats)             # (B, K)
            probs  = F.softmax(logits, dim=-1).cpu().numpy()
            feats_np = feats.cpu().numpy()
            pk1pk = probs * (1.0 - probs)          # (B, K)
            H_W += pk1pk.T @ (feats_np ** 2)       # (K, D)
            H_b += pk1pk.sum(axis=0)               # (K,)
            if sum(f.shape[0] for f in feat_cache) < cache_cap:
                feat_cache.append(feats_np); label_cache.append(yb.numpy())

    W_map = model.head.weight.detach().cpu().numpy().astype(np.float64)
    b_map = model.head.bias.detach().cpu().numpy().astype(np.float64)
    if calibrate and feat_cache:
        from uncertainty_driven_drift.components.laplace_util import calibrate_prior_precision
        f = np.concatenate(feat_cache, axis=0).astype(np.float64)
        lb = np.concatenate(label_cache, axis=0)
        prior_precision = calibrate_prior_precision(
            f, lb, W_map, b_map, H_W, H_b, seed=seed, label="resnet_laplace")
    return {
        "W_map": W_map,
        "b_map": b_map,
        "W_var": 1.0 / (H_W + prior_precision),
        "b_var": 1.0 / (H_b + prior_precision),
        "prior_precision": float(prior_precision),
    }


# ---------------------------------------------------------------------------
# Shared inference helpers
# ---------------------------------------------------------------------------

def _select_device():
    """cuda > mps > cpu, matching the pretraining loop.

    Inference used to stay on the CPU, which was tolerable for ResNet-20 (~0.28M
    params) but is not for the WRN-28-10 arm (~36.5M): 20 MC passes over 200
    batches is 4000 forward passes, and on CPU that turns minutes into hours.
    """
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _enable_mc_dropout(model) -> None:
    """Put the model in eval mode but reactivate the dropout MC-Dropout samples.

    When the backbone exposes a dedicated pre-head dropout (``head_drop``), ONLY
    that one is reactivated: deep block-internal dropout (WRN's regulariser) adds
    largely input-independent noise that drowns out the input-dependent
    disagreement the epistemic signal depends on.
    """
    import torch.nn as nn
    model.eval()
    head_drop = getattr(model, "head_drop", None)
    if isinstance(head_drop, nn.Dropout):
        head_drop.train()
        return
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


def _normalize_batch(x_np: np.ndarray, dataset: str = "cifar10"):
    """Normalise (B, 3, H, W) float32 [0,1] with the dataset's stats → torch tensor."""
    import torch
    x_t = torch.from_numpy(np.ascontiguousarray(x_np)).float()
    _mean, _std = _dataset_stats(dataset)
    mean_t = torch.tensor(_mean).view(1, 3, 1, 1)
    std_t  = torch.tensor(_std).view(1, 3, 1, 1)
    return (x_t - mean_t) / std_t


# ---------------------------------------------------------------------------
# MC-Dropout classifier
# ---------------------------------------------------------------------------

@register("model", "resnet_mc_dropout_cifar")
class ResNetMCDropoutCIFAR(Classifier):
    """Frozen ResNet-20 with MC-Dropout for CIFAR-10 known-vs-novel detection.

    Parameters
    ----------
    n_samples :
        Stochastic forward passes at inference time.
    p_drop :
        Dropout probability (applied after global avgpool, before head).
    data_root :
        Cache for CIFAR-10 train set used during pretraining.
    ckpt_path :
        Where to cache the pretrained weights.
    train_corruptions :
        List of known corruption names mixed in during training.
    corruption_severity :
        Severity level (1–5) for training corruptions.
    pretrain_epochs, pretrain_batch_size, pretrain_lr, pretrain_seed :
        Pretraining hyperparameters; ignored if checkpoint exists.
    seed :
        Seeds the MC sampling RNG.
    """

    # Dataset selector — the CIFAR-100 / WRN subclasses override these.
    _DATASET = "cifar10"
    _N_CLASSES = 10
    _ARCH = "ResNet-20"
    _BUILD_FN = staticmethod(_build_resnet20)

    def __init__(
        self,
        n_samples: int = 20,
        p_drop: float = 0.3,
        data_root: str = "./artifacts/cifar10",
        ckpt_path: str = "./artifacts/models/resnet20_cifar_kn_mc.pt",
        train_corruptions: Optional[List[str]] = None,
        corruption_severity: int = 3,
        clean_fraction: float = 0.0,
        mixup_alpha: float = 0.0,
        pretrain_epochs: int = 30,
        pretrain_batch_size: int = 128,
        pretrain_lr: float = 0.1,
        pretrain_seed: int = 0,
        label_smoothing: float = 0.0,
        lr_schedule: str = "cosine",
        warmup_epochs: int = 0,
        nesterov: bool = False,
        grad_clip: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.n_samples = int(n_samples)
        self.p_drop = float(p_drop)
        self.label_smoothing = float(label_smoothing)
        self.data_root = str(data_root)
        self.ckpt_path = str(ckpt_path)
        self.train_corruptions = list(train_corruptions) if train_corruptions else None
        self.corruption_severity = int(corruption_severity)
        self.clean_fraction = float(clean_fraction)
        self.mixup_alpha = float(mixup_alpha)
        self.pretrain_epochs = int(pretrain_epochs)
        self.pretrain_batch_size = int(pretrain_batch_size)
        self.pretrain_lr = float(pretrain_lr)
        self.pretrain_seed = int(pretrain_seed)
        self.lr_schedule = str(lr_schedule)
        self.warmup_epochs = int(warmup_epochs)
        self.nesterov = bool(nesterov)
        self.grad_clip = float(grad_clip)
        self.seed = int(seed)
        self._model = None
        self._device = None

    def setup(self, spec: StreamSpec) -> None:
        import torch
        # Backbone, normalisation stats and pretraining source are all CIFAR-10
        # specific; fail loudly rather than deep inside the forward pass.
        require_image_spec(
            spec, "resnet_mc_dropout_cifar",
            channels=(3,), spatial=((32, 32),), n_classes=self._N_CLASSES,
        )
        ckpt = Path(self.ckpt_path)
        if not ckpt.exists():
            print(f"[{type(self).__name__}] Pretraining {self._ARCH} on {self._DATASET} "
                  f"({self._N_CLASSES}-way, {self.pretrain_epochs} epochs)…", flush=True)
            _pretrain_resnet20(
                data_root=Path(self.data_root),
                ckpt_path=ckpt,
                epochs=self.pretrain_epochs,
                batch_size=self.pretrain_batch_size,
                lr=self.pretrain_lr,
                seed=self.pretrain_seed,
                p_drop=self.p_drop,
                train_corruptions=self.train_corruptions,
                corruption_severity=self.corruption_severity,
                clean_fraction=self.clean_fraction,
                mixup_alpha=self.mixup_alpha,
                n_classes=self._N_CLASSES,
                dataset=self._DATASET,
                build_fn=self._BUILD_FN,
                label_smoothing=self.label_smoothing,
                lr_schedule=self.lr_schedule,
                warmup_epochs=self.warmup_epochs,
                nesterov=self.nesterov,
                grad_clip=self.grad_clip,
            )
        torch.manual_seed(self.seed)
        model = self._BUILD_FN(n_classes=self._N_CLASSES, p_drop=self.p_drop)
        model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
        _enable_mc_dropout(model)
        self._device = _select_device()
        self._model = model.to(self._device)

    def predict(self, batch: StreamBatch) -> Prediction:
        import torch
        import torch.nn.functional as F
        assert self._model is not None
        x = _normalize_batch(batch.x, self._DATASET).to(self._device)
        # Only the pre-head dropout is stochastic (see _enable_mc_dropout), so the
        # backbone is deterministic: run it ONCE and resample just dropout+head.
        # Same distribution as N full passes, ~N times less compute — which is what
        # makes 10-seed evaluation of a 36M-param backbone affordable.
        samples: list = []
        with torch.no_grad():
            feats = self._model.embed(x)                     # (B, D), no dropout
            for _ in range(self.n_samples):
                logits = self._model.head(self._model.dropout(feats)) \
                    if hasattr(self._model, "dropout") \
                    else self._model.head(self._model.head_drop(feats))
                probs = F.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)
                samples.append(probs)
        mc_probs = np.stack(samples, axis=0)   # (S, B, 10)
        return Prediction(
            probs=mc_probs.mean(axis=0),
            features=None,
            extras={"mc_probs": mc_probs},
        )

    def observe(self, batch: StreamBatch, prediction: Prediction) -> None:
        del batch, prediction


# ---------------------------------------------------------------------------
# Diagonal Laplace classifier
# ---------------------------------------------------------------------------

@register("model", "resnet_laplace_cifar")
class ResNetLaplaceCIFAR(Classifier):
    """Frozen ResNet-20 + diagonal last-layer Laplace for CIFAR-10.

    Shares the same backbone architecture as :class:`ResNetMCDropoutCIFAR`
    but trained *without* dropout (p_drop=0.0) to give a clean MAP estimate.
    The GGN diagonal posterior is fitted over the Linear(64 → 10) head on
    the CIFAR-10 training set (with known-corruption augmentation).

    Parameters
    ----------
    prior_precision :
        Precision of the zero-mean isotropic Gaussian prior over head weights.
    n_samples :
        Weight samples drawn per batch at inference time.
    """

    # Dataset selector — the CIFAR-100 / WRN subclasses override these.
    _DATASET = "cifar10"
    _N_CLASSES = 10
    _ARCH = "ResNet-20"
    _BUILD_FN = staticmethod(_build_resnet20)

    def __init__(
        self,
        n_samples: int = 20,
        prior_precision: float = 1.0,
        data_root: str = "./artifacts/cifar10",
        ckpt_path: str = "./artifacts/models/resnet20_cifar_kn_laplace.pt",
        train_corruptions: Optional[List[str]] = None,
        corruption_severity: int = 3,
        clean_fraction: float = 0.0,
        mixup_alpha: float = 0.0,
        pretrain_epochs: int = 30,
        pretrain_batch_size: int = 128,
        pretrain_lr: float = 0.1,
        pretrain_seed: int = 0,
        label_smoothing: float = 0.0,
        lr_schedule: str = "cosine",
        warmup_epochs: int = 0,
        nesterov: bool = False,
        grad_clip: float = 0.0,
        calibrate_prior: bool = True,
        seed: int = 0,
    ) -> None:
        self.n_samples = int(n_samples)
        self.prior_precision = float(prior_precision)
        self.label_smoothing = float(label_smoothing)
        self.calibrate_prior = bool(calibrate_prior)
        self.data_root = str(data_root)
        self.ckpt_path = str(ckpt_path)
        self.train_corruptions = list(train_corruptions) if train_corruptions else None
        self.corruption_severity = int(corruption_severity)
        self.clean_fraction = float(clean_fraction)
        self.mixup_alpha = float(mixup_alpha)
        self.pretrain_epochs = int(pretrain_epochs)
        self.pretrain_batch_size = int(pretrain_batch_size)
        self.pretrain_lr = float(pretrain_lr)
        self.pretrain_seed = int(pretrain_seed)
        self.lr_schedule = str(lr_schedule)
        self.warmup_epochs = int(warmup_epochs)
        self.nesterov = bool(nesterov)
        self.grad_clip = float(grad_clip)
        self.seed = int(seed)
        self._model = None
        self._device = None
        self._posterior: dict | None = None

    def setup(self, spec: StreamSpec) -> None:
        import torch
        require_image_spec(
            spec, "resnet_laplace_cifar",
            channels=(3,), spatial=((32, 32),), n_classes=self._N_CLASSES,
        )
        ckpt = Path(self.ckpt_path)
        if not ckpt.exists():
            print(f"[{type(self).__name__}] Pretraining {self._ARCH} (no dropout) on "
                  f"{self._DATASET} ({self._N_CLASSES}-way, {self.pretrain_epochs} epochs)…", flush=True)
            _pretrain_resnet20(
                data_root=Path(self.data_root),
                ckpt_path=ckpt,
                epochs=self.pretrain_epochs,
                batch_size=self.pretrain_batch_size,
                lr=self.pretrain_lr,
                seed=self.pretrain_seed,
                p_drop=0.0,
                train_corruptions=self.train_corruptions,
                corruption_severity=self.corruption_severity,
                clean_fraction=self.clean_fraction,
                mixup_alpha=self.mixup_alpha,
                n_classes=self._N_CLASSES,
                dataset=self._DATASET,
                build_fn=self._BUILD_FN,
                label_smoothing=self.label_smoothing,
                lr_schedule=self.lr_schedule,
                warmup_epochs=self.warmup_epochs,
                nesterov=self.nesterov,
                grad_clip=self.grad_clip,
            )
        model = self._BUILD_FN(n_classes=self._N_CLASSES, p_drop=0.0)
        model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
        model.eval()
        self._device = _select_device()
        self._model = model.to(self._device)

        print(f"[{type(self).__name__}] Computing diagonal GGN posterior over head…", flush=True)
        self._posterior = _compute_laplace_posterior(
            model=self._model,
            data_root=Path(self.data_root),
            train_corruptions=self.train_corruptions,
            corruption_severity=self.corruption_severity,
            batch_size=self.pretrain_batch_size,
            prior_precision=self.prior_precision,
            seed=self.pretrain_seed,
            dataset=self._DATASET,
            calibrate=self.calibrate_prior,
        )
        print(f"[{type(self).__name__}] Laplace posterior ready.", flush=True)

    def predict(self, batch: StreamBatch) -> Prediction:
        import torch
        assert self._model is not None and self._posterior is not None
        rng = np.random.default_rng(self.seed + batch.index)

        x = _normalize_batch(batch.x, self._DATASET).to(self._device)
        with torch.no_grad():
            feats_np = self._model.embed(x).cpu().numpy().astype(np.float64)  # (B, D)

        W_map = self._posterior["W_map"]          # (K, 64)
        b_map = self._posterior["b_map"]          # (K,)
        W_std = np.sqrt(self._posterior["W_var"]) # (K, 64)
        b_std = np.sqrt(self._posterior["b_var"]) # (K,)

        samples: list = []
        for _ in range(self.n_samples):
            W_s = W_map + rng.standard_normal(W_map.shape) * W_std
            b_s = b_map + rng.standard_normal(b_map.shape) * b_std
            logits = feats_np @ W_s.T + b_s[None, :]  # (B, K)
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            samples.append((e / e.sum(axis=1, keepdims=True)).astype(np.float64))

        mc_probs = np.stack(samples, axis=0)   # (S, B, K)
        return Prediction(
            probs=mc_probs.mean(axis=0),
            features=None,
            extras={"mc_probs": mc_probs},
        )

    def observe(self, batch: StreamBatch, prediction: Prediction) -> None:
        del batch, prediction


# ---------------------------------------------------------------------------
# ResNet-20 on CIFAR-100 — capacity-ablation arm
# ---------------------------------------------------------------------------
#
# Same 270K-param architecture as the CIFAR-10 arms above, just pointed at the
# 100-way task via the _DATASET/_N_CLASSES/_BUILD_FN switch (identical pattern
# to wide_resnet.py's WRN-28-10 subclasses). Exists to test whether epistemic
# uncertainty separates more clearly from total entropy on a capacity-
# constrained backbone: across the vision arms already in AUROC_FPR95.json,
# the epistemic-vs-total AUROC gap is near zero (MNIST/CIFAR-10 ResNet-20 and
# ViT alike, gaps in [-0.06, +0.01]) or forced to exactly zero by a ceiling
# effect (CIFAR-100 ViT-B/16: both signals hit AUROC=1.0000) — unlike audio,
# where the gap is +0.01 to +0.40. Neither existing CIFAR-100 arm (WRN-28-10,
# 36M params; ViT-B/16, 86M) tests a genuinely capacity-constrained backbone
# on the harder 100-way task specifically, which is the untested regime.
#
# Deliberately kept on the SAME severity/known/novel corruption split as the
# WRN-28-10 and ViT-B/16 CIFAR-100 arms (not adjusted to dodge the ViT ceiling
# effect) so this is a clean, single-variable (backbone capacity) comparison
# against those two, not a confound of task difficulty AND capacity at once.

@register("model", "resnet20_cifar100_mc_dropout")
class ResNet20CIFAR100MCDropout(ResNetMCDropoutCIFAR):
    """ResNet-20 (270K params) + MC-Dropout for CIFAR-100 known-vs-novel."""

    _DATASET = "cifar100"
    _N_CLASSES = 100
    _ARCH = "ResNet-20"
    _BUILD_FN = staticmethod(_build_resnet20)


@register("model", "resnet20_cifar100_laplace")
class ResNet20CIFAR100Laplace(ResNetLaplaceCIFAR):
    """ResNet-20 (270K params) + diagonal last-layer Laplace for CIFAR-100 known-vs-novel."""

    _DATASET = "cifar100"
    _N_CLASSES = 100
    _ARCH = "ResNet-20"
    _BUILD_FN = staticmethod(_build_resnet20)
