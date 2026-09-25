"""Camelyon17-WILDS hospital-level domain-shift stream + RGB LeNet MC-Dropout model.

Camelyon17 (Bandi et al. 2019, WILDS benchmark) contains 96×96-pixel
histopathology patches from 5 hospitals.  Hospital identity acts as a
natural domain label: a model pretrained on hospitals 0-2 (in-distribution)
faces genuine covariate shift when hospitals 3-4 (out-of-distribution)
appear in the stream.

Registered components
---------------------
* ``camelyon17_known_novel`` — DatasetStream yielding known-hospital batches
  followed by novel-hospital batches.  ``spec.extras["novel_start_batch"]``
  marks the boundary so plotting code annotates it identically to the MNIST
  known-vs-novel experiment.
* ``lenet_mc_dropout_rgb``   — Adapted LeNet-5 for 3-channel 96×96 inputs
  with MC-Dropout inference.  Pretrained offline on the known-hospital split
  and frozen during streaming (same philosophy as the MNIST backbone).
* ``resnet_mc_dropout``      — ResNet-18 with MC-Dropout (dropout injected
  before the classification head).  Use this when the LeNet baseline
  under-fits Camelyon17.

Requirements
------------
* ``pip install wilds``  (tested with wilds≥2.0)
* The first run downloads ~15 GB of patch images to ``data_root``.

Hospital split (WILDS default)
------------------------------
* ``train`` split  → hospitals 0, 1, 2  (known)
* ``test``  split  → hospitals 3, 4     (novel)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np

from uncertainty_driven_drift.data.base import DatasetStream, StreamBatch, StreamSpec
from uncertainty_driven_drift.models.base import Classifier, Prediction
from uncertainty_driven_drift.registry import register


# ---------------------------------------------------------------------------
# Shared image transform
# ---------------------------------------------------------------------------

def _camelyon_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ---------------------------------------------------------------------------
# Hospital-level index builder
# ---------------------------------------------------------------------------

def _build_hospital_subsets(
    data_root: str,
    known_hospitals: List[int],
    novel_hospitals: List[int],
) -> Dict[int, Any]:
    """Return a dict mapping hospital_id → torch Subset.

    WILDS train split has known hospitals; OOD test split has novel hospitals.
    Falls back to filtering the combined dataset if the expected split does
    not contain the requested hospital.
    """
    try:
        from wilds import get_dataset
    except ImportError as exc:
        raise ImportError(
            "camelyon17_known_novel requires the 'wilds' package.\n"
            "Install: pip install wilds"
        ) from exc

    import torch
    from torch.utils.data import Subset

    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)

    dataset = get_dataset("camelyon17", root_dir=str(root), download=True)
    transform = _camelyon_transform()

    all_hospitals = known_hospitals + novel_hospitals

    # Try each split; hospital-0 and 1 are in 'train', hospital 3 and 4 in 'test'.
    subsets: Dict[int, Any] = {}
    for split_name in ("train", "test", "val"):
        try:
            split = dataset.get_subset(split_name, transform=transform)
        except Exception:
            continue
        metadata = split.metadata_array          # (N, n_fields)
        hosp_col = metadata[:, 0].numpy()
        for h in all_hospitals:
            if h in subsets:
                continue
            mask = hosp_col == h
            if mask.any():
                subsets[h] = Subset(split, np.where(mask)[0].tolist())

    missing = [h for h in all_hospitals if h not in subsets]
    if missing:
        raise ValueError(
            f"Could not locate hospitals {missing} in any Camelyon17 WILDS split. "
            f"Found hospitals: {sorted(subsets.keys())}"
        )
    return subsets


# ---------------------------------------------------------------------------
# Dataset stream
# ---------------------------------------------------------------------------

@register("dataset", "camelyon17_known_novel")
class Camelyon17KnownNovelStream(DatasetStream):
    """Camelyon17-WILDS known-hospital / novel-hospital stream.

    The stream presents ``n_batches_per_hospital`` batches from each
    known hospital in order, then ``n_batches_per_hospital`` batches from
    each novel hospital.  All transitions are abrupt.

    ``spec.extras["novel_start_batch"]`` is set to the first novel-phase
    batch index so plotting code can draw the known|novel boundary
    identically to the MNIST known-vs-novel experiment.

    Parameters
    ----------
    known_hospitals :
        Hospital IDs the model was pretrained on.  Defaults to [0, 1, 2]
        (WILDS in-distribution train split).
    novel_hospitals :
        Hospital IDs the model has never seen.  Defaults to [3, 4]
        (WILDS OOD test split).
    n_batches_per_hospital :
        Number of batches streamed per hospital phase.
    batch_size :
        Samples per batch.
    num_workers :
        DataLoader worker processes for image loading.
    seed :
        Controls shuffling within each hospital's DataLoader.
    data_root :
        Directory where WILDS stores / downloads Camelyon17 data (~15 GB).
    """

    def __init__(
        self,
        known_hospitals: Sequence[int] = (0, 1, 2),
        novel_hospitals: Sequence[int] = (3, 4),
        n_batches_per_hospital: int = 30,
        batch_size: int = 64,
        num_workers: int = 4,
        seed: int = 0,
        data_root: str = "./artifacts/camelyon17",
    ) -> None:
        self.known_hospitals = list(known_hospitals)
        self.novel_hospitals = list(novel_hospitals)
        self.n_batches_per_hospital = int(n_batches_per_hospital)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.seed = int(seed)
        self.data_root = str(data_root)

        all_phases = self.known_hospitals + self.novel_hospitals
        novel_start = self.n_batches_per_hospital * len(self.known_hospitals)
        total_batches = self.n_batches_per_hospital * len(all_phases)
        drift_indices = tuple(
            self.n_batches_per_hospital * i for i in range(1, len(all_phases))
        )

        self.spec = StreamSpec(
            name="camelyon17_known_novel",
            input_shape=(3, 96, 96),
            n_classes=2,
            n_batches=total_batches,
            batch_size=batch_size,
            drift_indices=drift_indices,
            has_true_posterior=False,
            extras={
                "known_hospitals": self.known_hospitals,
                "novel_hospitals": self.novel_hospitals,
                "n_batches_per_hospital": self.n_batches_per_hospital,
                "novel_start_batch": novel_start,
            },
        )

        self._hospital_subsets = _build_hospital_subsets(
            data_root, self.known_hospitals, self.novel_hospitals
        )

    def __iter__(self) -> Iterator[StreamBatch]:
        from torch.utils.data import DataLoader

        drift_set = set(self.spec.drift_indices)
        novel_set = set(self.novel_hospitals)
        t = 0

        for phase_idx, hospital in enumerate(self.known_hospitals + self.novel_hospitals):
            subset = self._hospital_subsets[hospital]
            loader = DataLoader(
                subset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                worker_init_fn=lambda wid: np.random.seed(self.seed + wid),
                drop_last=True,
            )
            loader_iter = iter(loader)
            is_novel = hospital in novel_set

            for _ in range(self.n_batches_per_hospital):
                try:
                    x_t, y_t, _meta = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(loader)
                    x_t, y_t, _meta = next(loader_iter)

                # WILDS returns (PIL → tensor, int label, metadata_tensor)
                x = x_t.numpy().astype(np.float32)   # (B, 3, 96, 96)
                y = y_t.numpy().astype(np.int64)

                yield StreamBatch(
                    index=t,
                    x=x,
                    y=y,
                    concept_id=phase_idx,
                    is_drift=(t in drift_set),
                    true_posterior=None,
                    extras={
                        "hospital": hospital,
                        "is_novel": is_novel,
                    },
                )
                t += 1


# ---------------------------------------------------------------------------
# LeNet-5 adapted for 3-channel 96×96 inputs (MC-Dropout)
# ---------------------------------------------------------------------------

def _build_lenet_rgb_dropout(n_classes: int = 2, p_drop: float = 0.3):
    """LeNet-style CNN for 3-channel 96×96 inputs with dropout on FC layers."""
    import torch.nn as nn

    class LeNet5RGBDropout(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, padding=2),   # → 32×96×96
                nn.ReLU(inplace=True),
                nn.MaxPool2d(4),                               # → 32×24×24
                nn.Conv2d(32, 64, kernel_size=5, padding=2),  # → 64×24×24
                nn.ReLU(inplace=True),
                nn.MaxPool2d(4),                               # → 64×6×6
            )
            self.fc = nn.Sequential(
                nn.Flatten(),                      # 64*6*6 = 2304
                nn.Linear(2304, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p_drop),
                nn.Linear(256, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p_drop),
            )
            self.head = nn.Linear(64, n_classes)

        def forward(self, x):
            return self.head(self.fc(self.features(x)))

    return LeNet5RGBDropout()


def _pretrain_camelyon17(
    data_root: str,
    ckpt_path: Path,
    known_hospitals: List[int],
    n_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    p_drop: float,
    num_workers: int = 4,
) -> None:
    """Train the adapted LeNet on known-hospital patches; save weights."""
    from wilds import get_dataset
    import torch
    from torch import optim
    from torch.utils.data import DataLoader, Subset

    torch.manual_seed(seed)

    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    dataset = get_dataset("camelyon17", root_dir=str(root), download=True)
    transform = _camelyon_transform()
    train_split = dataset.get_subset("train", transform=transform)

    # Filter to known hospitals only.
    hosp_col = train_split.metadata_array[:, 0].numpy()
    mask = np.isin(hosp_col, known_hospitals)
    known_indices = np.where(mask)[0].tolist()
    known_subset = Subset(train_split, known_indices)

    loader = DataLoader(
        known_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    model = _build_lenet_rgb_dropout(n_classes=n_classes, p_drop=p_drop)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for x, y, _meta in loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        print(f"  [camelyon17 pretrain] epoch {epoch + 1}/{epochs} done")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(ckpt_path))


def _enable_mc_dropout_rgb(model) -> None:
    import torch.nn as nn
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


@register("model", "lenet_mc_dropout_rgb")
class LeNetRGBMCDropout(Classifier):
    """Adapted LeNet-5 for 3-channel 96×96 inputs with MC-Dropout.

    Same inference protocol as :class:`LeNetMCDropout`: ``n_samples``
    stochastic forward passes with dropout active; mean written to
    ``Prediction.probs`` and the full ``(S, B, K)`` stack to
    ``extras["mc_probs"]`` for the BALD decomposition.

    The model is pretrained offline on the known-hospital split of
    Camelyon17 and frozen during streaming.

    Parameters
    ----------
    n_samples :
        MC-Dropout forward passes per batch.
    p_drop :
        Dropout probability for both training and MC inference.
    data_root :
        WILDS data root (used during pretraining; must match the stream).
    ckpt_path :
        Checkpoint location.  If absent the model is pretrained and saved.
    known_hospitals :
        Hospital IDs used for pretraining (must match the stream config).
    pretrain_epochs, pretrain_batch_size, pretrain_lr, pretrain_seed :
        Pretraining hyperparameters.
    num_workers :
        DataLoader workers during pretraining.
    seed :
        Torch seed for MC-Dropout inference reproducibility.
    """

    def __init__(
        self,
        n_samples: int = 30,
        p_drop: float = 0.3,
        data_root: str = "./artifacts/camelyon17",
        ckpt_path: str = "./artifacts/models/lenet_rgb_camelyon17.pt",
        known_hospitals: Sequence[int] = (0, 1, 2),
        pretrain_epochs: int = 5,
        pretrain_batch_size: int = 64,
        pretrain_lr: float = 1e-3,
        pretrain_seed: int = 0,
        num_workers: int = 4,
        seed: int = 0,
    ) -> None:
        self.n_samples = int(n_samples)
        self.p_drop = float(p_drop)
        self.data_root = str(data_root)
        self.ckpt_path = str(ckpt_path)
        self.known_hospitals = list(known_hospitals)
        self.pretrain_epochs = int(pretrain_epochs)
        self.pretrain_batch_size = int(pretrain_batch_size)
        self.pretrain_lr = float(pretrain_lr)
        self.pretrain_seed = int(pretrain_seed)
        self.num_workers = int(num_workers)
        self.seed = int(seed)
        self._cnn = None

    def setup(self, spec: StreamSpec) -> None:
        if tuple(spec.input_shape) != (3, 96, 96):
            raise ValueError(
                f"lenet_mc_dropout_rgb expects (3,96,96); got {spec.input_shape}"
            )
        import torch

        ckpt = Path(self.ckpt_path)
        if not ckpt.exists():
            print(
                f"Checkpoint not found at {ckpt}. Pretraining on "
                f"hospitals {self.known_hospitals} (this may take a while)..."
            )
            _pretrain_camelyon17(
                data_root=self.data_root,
                ckpt_path=ckpt,
                known_hospitals=self.known_hospitals,
                n_classes=spec.n_classes,
                epochs=self.pretrain_epochs,
                batch_size=self.pretrain_batch_size,
                lr=self.pretrain_lr,
                seed=self.pretrain_seed,
                p_drop=self.p_drop,
                num_workers=self.num_workers,
            )

        torch.manual_seed(self.seed)
        model = _build_lenet_rgb_dropout(n_classes=spec.n_classes, p_drop=self.p_drop)
        model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
        _enable_mc_dropout_rgb(model)
        self._cnn = model

    def predict(self, batch: StreamBatch) -> Prediction:
        import torch
        import torch.nn.functional as F

        assert self._cnn is not None
        x = torch.from_numpy(np.ascontiguousarray(batch.x)).float()

        samples: list = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                logits = self._cnn(x)
                probs = F.softmax(logits, dim=-1)
                samples.append(probs.detach().numpy().astype(np.float64))

        mc_probs = np.stack(samples, axis=0)   # (S, B, K)
        mean_probs = mc_probs.mean(axis=0)     # (B, K)
        return Prediction(
            probs=mean_probs.astype(np.float64),
            features=None,
            extras={"mc_probs": mc_probs},
        )

    def observe(self, batch: StreamBatch, prediction: Prediction) -> None:
        del batch, prediction


# ---------------------------------------------------------------------------
# ResNet-18 with MC-Dropout (fallback for Camelyon17)
# ---------------------------------------------------------------------------

def _build_resnet_dropout(n_classes: int = 2, p_drop: float = 0.3, pretrained: bool = True):
    """ResNet-18 with a dropout layer injected before the classification head."""
    import torch.nn as nn
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    except Exception:
        from torchvision.models import resnet18
        model = resnet18(pretrained=pretrained)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=p_drop),
        nn.Linear(in_features, n_classes),
    )
    return model


def _pretrain_resnet_camelyon17(
    data_root: str,
    ckpt_path: Path,
    known_hospitals: List[int],
    n_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    p_drop: float,
    pretrained: bool,
    num_workers: int = 4,
) -> None:
    from wilds import get_dataset
    import torch
    from torch import optim
    from torch.utils.data import DataLoader, Subset

    torch.manual_seed(seed)

    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    dataset = get_dataset("camelyon17", root_dir=str(root), download=True)
    transform = _camelyon_transform()
    train_split = dataset.get_subset("train", transform=transform)

    hosp_col = train_split.metadata_array[:, 0].numpy()
    mask = np.isin(hosp_col, known_hospitals)
    known_subset = Subset(train_split, np.where(mask)[0].tolist())

    loader = DataLoader(
        known_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    model = _build_resnet_dropout(n_classes=n_classes, p_drop=p_drop, pretrained=pretrained)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for x, y, _meta in loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        print(f"  [camelyon17 resnet pretrain] epoch {epoch + 1}/{epochs} done")

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(ckpt_path))


@register("model", "resnet_mc_dropout")
class ResNetMCDropout(Classifier):
    """ResNet-18 with MC-Dropout for Camelyon17 (fallback for lenet_mc_dropout_rgb).

    Pretraining uses ImageNet-pretrained weights fine-tuned on the
    known-hospital split.  MC-Dropout keeps the dropout layer in the
    classification head active at inference time.

    Parameters mirror :class:`LeNetRGBMCDropout`.
    """

    def __init__(
        self,
        n_samples: int = 30,
        p_drop: float = 0.3,
        data_root: str = "./artifacts/camelyon17",
        ckpt_path: str = "./artifacts/models/resnet_camelyon17.pt",
        known_hospitals: Sequence[int] = (0, 1, 2),
        pretrain_epochs: int = 5,
        pretrain_batch_size: int = 64,
        pretrain_lr: float = 1e-4,
        pretrain_seed: int = 0,
        pretrained_backbone: bool = True,
        num_workers: int = 4,
        seed: int = 0,
    ) -> None:
        self.n_samples = int(n_samples)
        self.p_drop = float(p_drop)
        self.data_root = str(data_root)
        self.ckpt_path = str(ckpt_path)
        self.known_hospitals = list(known_hospitals)
        self.pretrain_epochs = int(pretrain_epochs)
        self.pretrain_batch_size = int(pretrain_batch_size)
        self.pretrain_lr = float(pretrain_lr)
        self.pretrain_seed = int(pretrain_seed)
        self.pretrained_backbone = bool(pretrained_backbone)
        self.num_workers = int(num_workers)
        self.seed = int(seed)
        self._model = None

    def setup(self, spec: StreamSpec) -> None:
        if tuple(spec.input_shape) != (3, 96, 96):
            raise ValueError(
                f"resnet_mc_dropout expects (3,96,96); got {spec.input_shape}"
            )
        import torch

        ckpt = Path(self.ckpt_path)
        if not ckpt.exists():
            print(
                f"Checkpoint not found at {ckpt}. Fine-tuning ResNet-18 on "
                f"hospitals {self.known_hospitals} (this will take several minutes)..."
            )
            _pretrain_resnet_camelyon17(
                data_root=self.data_root,
                ckpt_path=ckpt,
                known_hospitals=self.known_hospitals,
                n_classes=spec.n_classes,
                epochs=self.pretrain_epochs,
                batch_size=self.pretrain_batch_size,
                lr=self.pretrain_lr,
                seed=self.pretrain_seed,
                p_drop=self.p_drop,
                pretrained=self.pretrained_backbone,
                num_workers=self.num_workers,
            )

        import torch.nn as nn
        torch.manual_seed(self.seed)
        model = _build_resnet_dropout(
            n_classes=spec.n_classes, p_drop=self.p_drop, pretrained=False
        )
        model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
        model.eval()
        # Keep dropout in training mode for MC inference.
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
        self._model = model

    def predict(self, batch: StreamBatch) -> Prediction:
        import torch
        import torch.nn.functional as F

        assert self._model is not None
        x = torch.from_numpy(np.ascontiguousarray(batch.x)).float()

        samples: list = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                logits = self._model(x)
                probs = F.softmax(logits, dim=-1)
                samples.append(probs.detach().numpy().astype(np.float64))

        mc_probs = np.stack(samples, axis=0)
        mean_probs = mc_probs.mean(axis=0)
        return Prediction(
            probs=mean_probs.astype(np.float64),
            features=None,
            extras={"mc_probs": mc_probs},
        )

    def observe(self, batch: StreamBatch, prediction: Prediction) -> None:
        del batch, prediction
