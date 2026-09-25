"""Pretrained ViT-B/16 (ImageNet-21k, via timm) with MC-Dropout + Laplace.

The pretrained-backbone analogue of :mod:`vit` (which trains a compact ViT from
scratch).  Weights come from **timm**, not torchvision, because torchvision only
ships ImageNet-**1k** ViT weights while the standard transfer protocol — and the
ViT paper's own CIFAR-100 result — starts from **ImageNet-21k**:

* ``vit_base_patch16_224.augreg_in21k``        — IN-21k pretrained (CIFAR-100 arm)
* ``vit_base_patch16_224.augreg_in21k_ft_in1k`` — IN-21k pretrained, IN-1k
  fine-tuned; used for the ImageNet arm, where a *trained* 1000-way head is
  required to classify the stream.

Adaptation to a new label space (CIFAR-100) is a **full fine-tune** by default
(``freeze_backbone: false``) with the same known-corruption augmentation the
other arms use — that augmentation is the experiment, not a regulariser: it is
what keeps the model calibrated on known corruptions so epistemic uncertainty
stays low there and rises on novel ones.  ``freeze_backbone: true`` falls back to
a linear probe (much cheaper, materially weaker).

Uncertainty:
* **MC-Dropout** — timm's ``drop_rate`` puts a dropout before the head; it is
  kept active at inference for ``n_samples`` stochastic passes.
* **Last-layer Laplace** — diagonal Gauss-Newton posterior over the final
  ``Linear(768 -> n_classes)`` head, fit post-hoc.

Normalisation and input size are read from timm's own data config for the chosen
checkpoint (IN-21k AugReg uses mean=std=0.5, *not* the ImageNet stats), so the
preprocessing always matches the weights.

Registered components
---------------------
* ``vit_b16_mc_dropout`` — pretrained ViT-B/16 + MC-Dropout.
* ``vit_b16_laplace``    — pretrained ViT-B/16 + diagonal last-layer Laplace.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional

import numpy as np

from uncertainty_driven_drift.data.base import StreamBatch, StreamSpec
from uncertainty_driven_drift.models.base import Classifier, Prediction
from uncertainty_driven_drift.registry import register

# IN-21k backbone for transfer; IN-21k->IN-1k for the ImageNet arm's 1000-way head.
DEFAULT_MODEL_21K = "vit_base_patch16_224.augreg_in21k"
DEFAULT_MODEL_IN1K = "vit_base_patch16_224.augreg_in21k_ft_in1k"


def _select_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_timm_vit(model_name: str, n_classes: int, p_drop: float,
                    pretrained: bool = True):
    """timm ViT wrapped with ``embed()`` + a ``head`` Linear, plus its data config.

    Returns ``(module, data_cfg)`` where ``data_cfg`` carries the checkpoint's own
    ``mean``/``std``/``input_size`` — using ImageNet stats with a 21k AugReg
    checkpoint (mean=std=0.5) would silently degrade every prediction.
    """
    import timm
    import torch.nn as nn

    base = timm.create_model(model_name, pretrained=pretrained,
                             num_classes=n_classes, drop_rate=p_drop)
    cfg = timm.data.resolve_model_data_config(base)

    class ViTWrap(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = base

        @property
        def head(self):                      # the Linear the Laplace posterior is over
            return self.base.head

        def embed(self, x):
            """(B, D) pooled pre-logits feature (timm applies head_drop here)."""
            return self.base.forward_head(self.base.forward_features(x), pre_logits=True)

        def embed_deterministic(self, x):
            """(B, D) pooled feature BEFORE head_drop.

            Lets MC-Dropout run the (deterministic) backbone once per batch and
            resample only dropout+head — identical in distribution to N full
            passes, ~N times cheaper.
            """
            b = self.base
            t = b.forward_features(x)
            if getattr(b, "attn_pool", None) is not None:
                t = b.attn_pool(t)
            elif b.global_pool == "avg":
                t = t[:, b.num_prefix_tokens:].mean(dim=1)
            elif b.global_pool:
                t = t[:, 0]
            return b.fc_norm(t)

        def forward(self, x):
            return self.base(x)

    return ViTWrap(), cfg


def _enable_mc_dropout(model) -> None:
    import torch.nn as nn
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def _prep(x_np: np.ndarray, image_size: int, mean, std, device):
    """[0,1] (B,3,H,W) -> resized + checkpoint-normalised tensor on ``device``."""
    import torch
    import torch.nn.functional as F
    x = torch.from_numpy(np.ascontiguousarray(x_np)).float()
    if x.shape[-1] != image_size or x.shape[-2] != image_size:
        x = F.interpolate(x, size=(image_size, image_size), mode="bilinear",
                          align_corners=False)
    m = torch.tensor(mean).view(1, 3, 1, 1)
    s = torch.tensor(std).view(1, 3, 1, 1)
    return ((x - m) / s).to(device)


# ---------------------------------------------------------------------------
# Fine-tuning (full network by default) + Laplace fit pool
# ---------------------------------------------------------------------------

def _cifar100_train_arrays(data_root: str, n_images: int = 0, seed: int = 0):
    from torchvision import datasets
    ds = datasets.CIFAR100(str(data_root), train=True, download=False)
    x = (ds.data.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)   # (N,3,32,32)
    y = np.array(ds.targets, dtype=np.int64)
    if n_images and n_images < x.shape[0]:
        idx = np.random.default_rng(seed).permutation(x.shape[0])[:n_images]
        x, y = x[idx], y[idx]
    return x, y


def _imagenet_train_loader(data_root: str, image_size: int, batch_size: int,
                           seed: int, n_images: int = 0):
    """Streaming loader over a locally-downloaded SLICE of the ImageNet train split.

    NOT the 1.28M-image full split -- see scripts/hf_imagenet_extract.py.
    Kept in a directory separate from the val split streamed for known/novel
    evaluation, so fine-tuning here never touches the images the drift stream
    later scores on.

    Streams from disk rather than materialising the slice as one array, because
    the array form does not survive the jump from CIFAR to ImageNet resolution:
    50k x 3 x 224 x 224 float32 is 30 GB and was OOM-killed against the
    cluster's 16 GB cap (the CIFAR-100 equivalent is 0.6 GB, which is why the
    array path is fine there and was kept for it).

    Labels are remapped from ImageFolder's local class indices to the canonical
    0-999 ImageNet indices, without which a partial slice trains the 1000-way
    head against the wrong classes entirely.
    """
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import transforms

    from uncertainty_driven_drift.components.imagenet_c import (
        global_label_remap, wnid_image_folder,
    )

    tfm = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    try:
        ds = wnid_image_folder(Path(data_root), tfm)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"no ImageNet train slice at {data_root} — run "
            f"scripts/hf_imagenet_extract.py --split train first.") from exc

    remap = global_label_remap(Path(data_root), ds.classes)
    if n_images and n_images < len(ds):
        idx = np.random.default_rng(seed).permutation(len(ds))[:n_images]
        ds = Subset(ds, idx.tolist())

    g = torch.Generator(); g.manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g,
                        num_workers=4, pin_memory=True, drop_last=True)
    return loader, remap


def _finetune(model, x, y, image_size, mean, std, epochs, batch_size,
              lr, weight_decay, freeze_backbone, train_corruptions,
              corruption_severity, clean_fraction, seed, device,
              loader=None, label_remap=None, steps_per_epoch=None):
    """Fine-tune with known-corruption augmentation, from arrays or a loader.

    Recipe follows the ViT transfer protocol: SGD + momentum, cosine schedule and
    gradient clipping at 1.0 (ViT fine-tuning is unstable without it).

    Two input paths, same optimiser and same augmentation:
      * ``x``/``y`` arrays -- CIFAR-100, where the whole set is 0.6 GB;
      * ``loader`` -- ImageNet, where materialising 50k images at 224px would be
        30 GB. Pass ``label_remap`` alongside it to convert ImageFolder's local
        class indices to canonical 0-999 ones.
    """
    import torch
    from torch import optim

    torch.manual_seed(seed)
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    n = 0 if x is None else x.shape[0]

    if freeze_backbone:
        for p in model.base.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True
        params = list(model.head.parameters())
        print("  [vit fine-tune] LINEAR PROBE (backbone frozen)")
    else:
        for p in model.parameters():
            p.requires_grad = True
        params = list(model.parameters())
        print("  [vit fine-tune] FULL fine-tune")

    model.to(device).train()
    opt = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if steps_per_epoch is None:
        steps_per_epoch = len(loader) if loader is not None else max(1, n // batch_size)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * steps_per_epoch))
    loss_fn = torch.nn.CrossEntropyLoss()

    if train_corruptions:
        from uncertainty_driven_drift.components.cifar_c_corruptions import apply_corruption

    def _epoch_batches():
        """(xb, yb) numpy batches for one epoch, from whichever source was given."""
        if loader is not None:
            for xb_t, yb_t in loader:
                yb = yb_t.numpy()
                if label_remap is not None:
                    yb = np.asarray(label_remap, dtype=np.int64)[yb]
                yield xb_t.numpy(), yb
        else:
            order = np_rng.permutation(n)
            for i in range(0, n - batch_size + 1, batch_size):
                bi = order[i:i + batch_size]
                yield x[bi], y[bi]

    for ep in range(int(epochs)):
        running, nb = 0.0, 0
        for xb, yb in _epoch_batches():
            if train_corruptions and py_rng.random() >= clean_fraction:
                corruption = py_rng.choice(train_corruptions)
                xb = apply_corruption(corruption, xb, corruption_severity, np_rng)
            xt = _prep(xb, image_size, mean, std, device)
            yt = torch.from_numpy(np.ascontiguousarray(yb)).long().to(device)
            opt.zero_grad()
            loss = loss_fn(model(xt), yt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            running += loss.item(); nb += 1
        print(f"  [vit fine-tune] epoch {ep+1}/{epochs}  loss={running/max(nb,1):.4f}  "
              f"lr={opt.param_groups[0]['lr']:.2e}", flush=True)
    model.eval()


def _fit_pool(dataset, data_root, n_classes, image_size, n_images, seed, synthetic,
              train_corruptions, corruption_severity, train_data_root=None):
    """Images the Laplace posterior is accumulated on (same distribution the model
    was calibrated on: known corruptions for CIFAR-100, and — once a train slice
    plus train_corruptions is configured — known corruptions for ImageNet too;
    otherwise ImageNet falls back to clean val, matching pre-fine-tune behaviour)."""
    if not synthetic and dataset == "cifar100":
        x, y = _cifar100_train_arrays(data_root)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(x.shape[0])[:n_images]
        x, y = x[idx], y[idx]
        if train_corruptions:
            from uncertainty_driven_drift.components.cifar_c_corruptions import apply_corruption
            py_rng = random.Random(seed); np_rng = np.random.default_rng(seed)
            out = []
            for i in range(0, x.shape[0], 128):
                c = py_rng.choice(train_corruptions)
                out.append(apply_corruption(c, x[i:i + 128], corruption_severity, np_rng))
            x = np.concatenate(out, axis=0)
        return x, y
    if not synthetic and dataset == "imagenet":
        if train_corruptions and train_data_root:
            # Bounded by fit_images (1024 by default, ~0.6 GB at 224px), so the
            # array form is fine here -- unlike the fine-tune above, which needs
            # the streaming loader. Raising fit_images past a few thousand will
            # reintroduce the 30 GB blow-up, so keep it small.
            from uncertainty_driven_drift.components.imagenet_c import _load_imagenet_val_pool
            x, y = _load_imagenet_val_pool(Path(train_data_root), n_images, image_size, seed)
            if x is None:
                raise FileNotFoundError(
                    f"no ImageNet train slice at {train_data_root} — run "
                    f"scripts/hf_imagenet_extract.py --split train first.")
            from uncertainty_driven_drift.components.cifar_c_corruptions import apply_corruption
            py_rng = random.Random(seed); np_rng = np.random.default_rng(seed)
            out = []
            for i in range(0, x.shape[0], 128):
                c = py_rng.choice(train_corruptions)
                out.append(apply_corruption(c, x[i:i + 128], corruption_severity, np_rng))
            return np.concatenate(out, axis=0), y
        from uncertainty_driven_drift.components.imagenet_c import _load_imagenet_val_pool
        x, y = _load_imagenet_val_pool(Path(data_root), n_images, image_size, seed)
        if x is not None:
            return x, y
    rng = np.random.default_rng(seed)
    return (rng.random((n_images, 3, image_size, image_size), dtype=np.float32),
            rng.integers(0, n_classes, size=n_images).astype(np.int64))


def _fit_laplace_head(model, x, y, image_size, mean, std, batch_size,
                      prior_precision, device, calibrate=True, seed=0):
    import torch
    import torch.nn.functional as F
    from uncertainty_driven_drift.components.laplace_util import calibrate_prior_precision

    model.to(device).eval()
    K = model.head.out_features
    D = model.head.in_features
    H_W = np.zeros((K, D), dtype=np.float64)
    H_b = np.zeros(K, dtype=np.float64)
    feat_cache = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = _prep(x[i:i + batch_size], image_size, mean, std, device)
            feats = model.embed(xb)
            probs = F.softmax(model.head(feats), dim=-1).cpu().numpy()
            fn = feats.cpu().numpy()
            pk = probs * (1.0 - probs)
            H_W += pk.T @ (fn ** 2)
            H_b += pk.sum(axis=0)
            feat_cache.append(fn)
    W_map = model.head.weight.detach().cpu().numpy().astype(np.float64)
    b_map = model.head.bias.detach().cpu().numpy().astype(np.float64)

    # A fixed prior_precision makes the posterior width depend on backbone, class
    # count and fit-set size; at 1.0 the CIFAR-100 ViT head drowned in its own
    # sampling noise (chance accuracy, 2.7 nats of fake "epistemic"). Pick the
    # widest posterior that still preserves the MAP predictions.
    if calibrate:
        feats_np = np.concatenate(feat_cache, axis=0).astype(np.float64)
        prior_precision = calibrate_prior_precision(
            feats_np, np.asarray(y)[:feats_np.shape[0]], W_map, b_map, H_W, H_b,
            seed=seed, label="vit_b16_laplace")
    return {
        "W_map": W_map,
        "b_map": b_map,
        "W_var": 1.0 / (H_W + prior_precision),
        "b_var": 1.0 / (H_b + prior_precision),
        "prior_precision": float(prior_precision),
    }


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------

class _ViTB16Base(Classifier):
    def __init__(
        self,
        n_classes: int = 1000,
        dataset: str = "imagenet",
        data_root: str = "./artifacts/imagenet/val",
        train_data_root: str = "",
        model_name: str = "",
        image_size: int = 0,
        n_samples: int = 20,
        p_drop: float = 0.1,
        ckpt_path: str = "",
        finetune_epochs: int = 10,
        finetune_lr: float = 0.01,
        finetune_batch_size: int = 64,
        finetune_weight_decay: float = 0.0,
        finetune_images: int = 0,
        freeze_backbone: bool = False,
        train_corruptions: Optional[List[str]] = None,
        corruption_severity: int = 3,
        clean_fraction: float = 0.2,
        fit_images: int = 2048,
        synthetic: bool = False,
        seed: int = 0,
    ) -> None:
        self.n_classes = int(n_classes)
        self.dataset = str(dataset)
        self.data_root = str(data_root)
        # ImageNet: fine-tuning and the Laplace fit pool read from a SEPARATE train
        # slice (scripts/hf_imagenet_extract.py), never from data_root
        # (the val split the known/novel stream evaluates on) — kept apart so
        # fine-tuning can never leak into the images later scored for drift.
        self.train_data_root = str(train_data_root) or None
        # Default checkpoint: IN-21k for transfer, IN-21k->IN-1k when we need a
        # trained 1000-way head (the ImageNet stream).
        self.model_name = str(model_name) or (
            DEFAULT_MODEL_IN1K if self.n_classes == 1000 else DEFAULT_MODEL_21K)
        self.image_size = int(image_size) if image_size else 0
        self.n_samples = int(n_samples)
        self.p_drop = float(p_drop)
        self.ckpt_path = str(ckpt_path)
        self.finetune_epochs = int(finetune_epochs)
        self.finetune_lr = float(finetune_lr)
        self.finetune_batch_size = int(finetune_batch_size)
        self.finetune_weight_decay = float(finetune_weight_decay)
        self.finetune_images = int(finetune_images)   # 0 = use every image loaded (CIFAR-100 default)
        self.freeze_backbone = bool(freeze_backbone)
        self.train_corruptions = list(train_corruptions) if train_corruptions else None
        self.corruption_severity = int(corruption_severity)
        self.clean_fraction = float(clean_fraction)
        self.fit_images = int(fit_images)
        self.synthetic = bool(synthetic)
        self.seed = int(seed)
        self._device = None
        self._model = None
        self._mean = self._std = None

    def _prepare(self, p_drop: float):
        """Build the backbone, restore/produce a fine-tuned checkpoint, set preprocessing."""
        import torch
        self._device = _select_device()
        model, cfg = _build_timm_vit(self.model_name, self.n_classes, p_drop,
                                     pretrained=True)
        self._mean, self._std = cfg["mean"], cfg["std"]
        if not self.image_size:
            self.image_size = int(cfg["input_size"][-1])
        print(f"[vit_b16] {self.model_name} | n_classes={self.n_classes} | "
              f"{self.image_size}px | mean={tuple(round(m,3) for m in self._mean)}")

        ckpt = Path(self.ckpt_path) if self.ckpt_path else None
        if ckpt and ckpt.exists():
            model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
            print(f"[vit_b16] loaded fine-tuned checkpoint {ckpt}")
            model.to(self._device)
        elif self.dataset == "cifar100" and not self.synthetic:
            print(f"[vit_b16] fine-tuning on CIFAR-100 ({self.finetune_epochs} epochs)…")
            x, y = _cifar100_train_arrays(self.data_root, self.finetune_images, self.seed)
            _finetune(
                model, x, y, self.image_size, self._mean, self._std,
                self.finetune_epochs, self.finetune_batch_size, self.finetune_lr,
                self.finetune_weight_decay, self.freeze_backbone,
                self.train_corruptions, self.corruption_severity,
                self.clean_fraction, self.seed, self._device)
            if ckpt:
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.to("cpu").state_dict(), str(ckpt))
                print(f"[vit_b16] saved → {ckpt}")
                model.to(self._device)
        elif self.dataset == "imagenet" and not self.synthetic and self.train_data_root:
            n_ft = self.finetune_images or 0
            loader, label_remap = _imagenet_train_loader(
                self.train_data_root, self.image_size, self.finetune_batch_size,
                self.seed, n_images=n_ft)
            print(f"[vit_b16] fine-tuning on an ImageNet train slice "
                  f"({len(loader.dataset)} images, {len(loader)} steps/epoch, "
                  f"{self.finetune_epochs} epochs, streamed)…")
            _finetune(
                model, None, None, self.image_size, self._mean, self._std,
                self.finetune_epochs, self.finetune_batch_size, self.finetune_lr,
                self.finetune_weight_decay, self.freeze_backbone,
                self.train_corruptions, self.corruption_severity,
                self.clean_fraction, self.seed, self._device,
                loader=loader, label_remap=label_remap)
            if ckpt:
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.to("cpu").state_dict(), str(ckpt))
                print(f"[vit_b16] saved → {ckpt}")
                model.to(self._device)
        else:
            model.to(self._device)
        return model

    def observe(self, batch: StreamBatch, prediction: Prediction) -> None:
        del batch, prediction


@register("model", "vit_b16_mc_dropout")
class ViTB16MCDropout(_ViTB16Base):
    def setup(self, spec: StreamSpec) -> None:
        model = self._prepare(self.p_drop)
        _enable_mc_dropout(model)
        self._model = model

    def predict(self, batch: StreamBatch) -> Prediction:
        import torch
        import torch.nn.functional as F
        assert self._model is not None
        x = _prep(batch.x, self.image_size, self._mean, self._std, self._device)
        samples = []
        with torch.no_grad():
            feats = self._model.embed_deterministic(x)        # backbone runs ONCE
            head_drop = self._model.base.head_drop
            for _ in range(self.n_samples):
                logits = self._model.head(head_drop(feats))
                samples.append(F.softmax(logits, dim=-1).cpu().numpy().astype(np.float64))
        mc_probs = np.stack(samples, axis=0)
        return Prediction(probs=mc_probs.mean(axis=0), features=None,
                          extras={"mc_probs": mc_probs})


@register("model", "vit_b16_laplace")
class ViTB16Laplace(_ViTB16Base):
    def __init__(self, prior_precision: float = 1.0,
                 calibrate_prior: bool = True, **kw) -> None:
        kw.setdefault("p_drop", 0.0)
        super().__init__(**kw)
        self.prior_precision = float(prior_precision)
        # calibrate_prior=False pins the value above (reproducing an old run).
        self.calibrate_prior = bool(calibrate_prior)
        self._posterior = None

    def setup(self, spec: StreamSpec) -> None:
        model = self._prepare(0.0)
        model.eval()
        self._model = model
        print("[vit_b16_laplace] fitting diagonal GGN posterior over head…")
        x, y = _fit_pool(self.dataset, self.data_root, self.n_classes, self.image_size,
                         self.fit_images, self.seed, self.synthetic,
                         self.train_corruptions, self.corruption_severity,
                         train_data_root=self.train_data_root)
        self._posterior = _fit_laplace_head(
            model, x, y, self.image_size, self._mean, self._std, 64,
            self.prior_precision, self._device,
            calibrate=self.calibrate_prior, seed=self.seed)
        print("[vit_b16_laplace] posterior ready.")

    def predict(self, batch: StreamBatch) -> Prediction:
        import torch
        assert self._model is not None and self._posterior is not None
        rng = np.random.default_rng(self.seed + batch.index)
        x = _prep(batch.x, self.image_size, self._mean, self._std, self._device)
        with torch.no_grad():
            feats_np = self._model.embed(x).cpu().numpy().astype(np.float64)
        W_map, b_map = self._posterior["W_map"], self._posterior["b_map"]
        W_std = np.sqrt(self._posterior["W_var"]); b_std = np.sqrt(self._posterior["b_var"])
        samples = []
        for _ in range(self.n_samples):
            W_s = W_map + rng.standard_normal(W_map.shape) * W_std
            b_s = b_map + rng.standard_normal(b_map.shape) * b_std
            logits = feats_np @ W_s.T + b_s[None, :]
            logits -= logits.max(axis=1, keepdims=True)
            e = np.exp(logits)
            samples.append((e / e.sum(axis=1, keepdims=True)).astype(np.float64))
        mc_probs = np.stack(samples, axis=0)
        return Prediction(probs=mc_probs.mean(axis=0), features=None,
                          extras={"mc_probs": mc_probs})
