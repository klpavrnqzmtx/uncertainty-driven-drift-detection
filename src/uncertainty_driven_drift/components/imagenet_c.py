"""ImageNet known-vs-novel corruption stream (ImageNet-C style).

The 1000-class, 224x224 analogue of :mod:`cifar_c`.  Streams ImageNet images
with on-the-fly corruptions from the same corruption family; a *known* phase of
corruptions the model is calibrated on, then a *novel* phase.

Data
----
Real runs read the ImageNet **validation** set as a torchvision ``ImageFolder``
under ``data_root`` (``data_root/<wnid>/*.JPEG``), resized/centre-cropped to
224x224.  ImageNet is not auto-downloadable, so for local **smoke tests** the
stream falls back to a small synthetic pool of random images (set
``synthetic: true`` or just point ``data_root`` at a missing path) — enough to
exercise the full pipeline (model forward, uncertainty, detectors, output) on
CPU without the dataset.

Registered components
---------------------
* ``imagenet_known_novel`` — DatasetStream, input_shape=(3, 224, 224), n_classes=1000
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import numpy as np

from uncertainty_driven_drift.components.cifar_c import _SHORT_NAME
from uncertainty_driven_drift.components.cifar_c_corruptions import (
    apply_corruption,
    available_corruptions,
)
from uncertainty_driven_drift.data.base import DatasetStream, StreamBatch, StreamSpec
from uncertainty_driven_drift.registry import register

_IMAGENET_CLASSES = 1000
_WNID_RE = re.compile(r"n\d{8}")
_INPUT_SHAPE = (3, 224, 224)


try:                                   # torchvision stays optional at import time
    from torchvision.datasets import ImageFolder as _ImageFolder
    _HAVE_TORCHVISION = True
except ImportError:                     # pragma: no cover
    _ImageFolder, _HAVE_TORCHVISION = object, False


class WnidImageFolder(_ImageFolder):
    """ImageFolder restricted to wnid-shaped (n########) class directories.

    Plain ImageFolder treats EVERY subdirectory as a class, which breaks on a
    download tree that also carries bookkeeping: the resumable extractor
    (scripts/hf_imagenet_extract.py) keeps per-shard completion markers in a
    `.shard_done/` directory beside the classes, and stock ImageFolder either
    raises on it (no images inside) or -- worse, had it been tolerated -- would
    have ranked "." before "n" and shifted every wnid's class index by one,
    silently mislabelling the entire split.

    Restricting to n######## also pins the local class order to wnid order,
    which is what global_label_remap() relies on.

    Defined at MODULE scope deliberately: a DataLoader with num_workers > 0
    pickles the dataset, and a class defined inside a function is not
    picklable ("Can't pickle local object").
    """

    def find_classes(self, directory):
        names = sorted(e.name for e in os.scandir(directory)
                       if e.is_dir() and _WNID_RE.fullmatch(e.name))
        if not names:
            raise FileNotFoundError(
                f"no wnid class directories (n########) under {directory}")
        return names, {n: i for i, n in enumerate(names)}


def wnid_image_folder(data_root, transform):
    """WnidImageFolder over ``data_root``; see that class for why it exists."""
    if not _HAVE_TORCHVISION:           # pragma: no cover
        raise ImportError("imagenet_known_novel requires torchvision")
    return WnidImageFolder(str(data_root), transform=transform)


def global_label_remap(data_root: Path, classes: Sequence[str]):
    """local ImageFolder class index -> canonical 0-999 ImageNet index.

    ImageFolder numbers whichever classes are PRESENT, so a partial directory
    (a fine-tuning slice) diverges from the model's real 1000-way head.
    scripts/hf_imagenet_extract.py writes wnid_index.json (the canonical order)
    into every directory it populates; returns None when that file is absent,
    in which case the caller must assume the directory holds all 1000 classes.
    """
    idx_file = Path(data_root) / "wnid_index.json"
    if not idx_file.exists():
        return None
    canonical = json.loads(idx_file.read_text())
    global_of = {wnid: i for i, wnid in enumerate(canonical)}
    return [global_of[w] for w in classes]


def _load_imagenet_val_pool(data_root: Path, max_images: int, image_size: int,
                            seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Load up to ``max_images`` ImageNet-val images as (N, 3, H, W) float32 [0,1].

    Uses a torchvision ``ImageFolder`` (val layout).  Returns ``(None, None)``
    if the directory does not look like an ImageNet folder, so the caller can
    fall back to the synthetic pool.
    """
    if not data_root.exists() or not any(data_root.iterdir()):
        return None, None  # type: ignore[return-value]
    try:
        import torch
        from torchvision import datasets, transforms
    except ImportError as exc:  # pragma: no cover
        raise ImportError("imagenet_known_novel requires torchvision") from exc

    tfm = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),                       # (3, H, W) float [0,1]
    ])

    try:
        ds = wnid_image_folder(data_root, tfm)
    except FileNotFoundError:
        return None, None  # type: ignore[return-value]
    if len(ds) == 0:
        return None, None  # type: ignore[return-value]

    # ImageFolder assigns LOCAL class indices 0..(k-1) by alphabetically
    # sorting whichever wnid subfolders are actually PRESENT — identical to the
    # true global 0-999 ImageNet class index only when all 1000 are present
    # (the full val set). A partial directory (e.g. a fine-tune-only train
    # SLICE — see scripts/hf_imagenet_extract.py) has far fewer than
    # 1000, so local alphabetical rank silently diverges from the model's real
    # 1000-way class index: caught by testing the fine-tune path end-to-end
    # before this ever reached the cluster. scripts/hf_imagenet_extract.py
    # writes wnid_index.json (the canonical global order) into every directory
    # it populates; remap through it here whenever it's present, no network or
    # HF_TOKEN needed at eval/fine-tune time.
    remap = global_label_remap(data_root, ds.classes)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))[:max_images]
    xs, ys = [], []
    for i in idx:
        img, label = ds[int(i)]
        xs.append(img.numpy())
        ys.append(remap[label] if remap is not None else int(label))
    x = np.stack(xs).astype(np.float32)
    y = np.array(ys, dtype=np.int64)
    return x, y


def _synthetic_pool(pool_size: int, image_size: int, n_classes: int,
                    seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Random-image pool for smoke tests (no ImageNet download needed)."""
    rng = np.random.default_rng(seed)
    x = rng.random((pool_size, 3, image_size, image_size), dtype=np.float32)
    y = rng.integers(0, n_classes, size=pool_size).astype(np.int64)
    return x, y


@register("dataset", "imagenet_known_novel")
class ImageNetKnownNovelStream(DatasetStream):
    """ImageNet stream with known-then-novel corruption phases."""

    def __init__(
        self,
        known_corruptions: Sequence[str],
        novel_corruptions: Sequence[str],
        n_batches_per_phase: int = 20,
        severity: int = 3,
        batch_size: int = 32,
        seed: int = 0,
        data_root: str = "./artifacts/imagenet/val",
        image_size: int = 224,
        max_pool: int = 4000,
        synthetic: bool = False,
        synthetic_pool: int = 256,
    ) -> None:
        if not known_corruptions or not novel_corruptions:
            raise ValueError("known and novel corruptions must both be non-empty")
        avail = set(available_corruptions())
        bad = [c for c in list(known_corruptions) + list(novel_corruptions) if c not in avail]
        if bad:
            raise KeyError(f"Unknown corruptions {bad!r}; have {sorted(avail)}")

        self.known_corruptions = list(known_corruptions)
        self.novel_corruptions = list(novel_corruptions)
        self.n_batches_per_phase = int(n_batches_per_phase)
        self.severity = int(severity)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.image_size = int(image_size)

        pool_x, pool_y = (None, None)
        if not synthetic:
            pool_x, pool_y = _load_imagenet_val_pool(
                Path(data_root), int(max_pool), self.image_size, self.seed)
        if pool_x is None:
            print(f"[imagenet_known_novel] no ImageNet at {data_root} (or synthetic=True) "
                  f"— using a synthetic pool of {synthetic_pool} random images (SMOKE MODE).")
            pool_x, pool_y = _synthetic_pool(
                int(synthetic_pool), self.image_size, _IMAGENET_CLASSES, self.seed)
            self._synthetic = True
        else:
            self._synthetic = False
        self._pool_x, self._pool_y = pool_x, pool_y

        n_known, n_novel = len(self.known_corruptions), len(self.novel_corruptions)
        total_batches = (n_known + n_novel) * self.n_batches_per_phase
        novel_start = n_known * self.n_batches_per_phase
        drift_indices = [i * self.n_batches_per_phase for i in range(1, n_known + n_novel)]
        channel_names = [_SHORT_NAME.get(c, c)
                         for c in self.known_corruptions + self.novel_corruptions]

        self.spec = StreamSpec(
            name="imagenet_known_novel",
            input_shape=_INPUT_SHAPE,
            n_classes=_IMAGENET_CLASSES,
            n_batches=total_batches,
            batch_size=self.batch_size,
            drift_indices=tuple(drift_indices),
            has_true_posterior=False,
            extras={
                "severity": self.severity,
                "known_corruptions": self.known_corruptions,
                "novel_corruptions": self.novel_corruptions,
                "novel_start_batch": novel_start,
                "channel_names": channel_names,
                "pool_size": int(self._pool_x.shape[0]),
                "synthetic": self._synthetic,
                "available_corruptions": available_corruptions(),
            },
        )

    def __iter__(self) -> Iterator[StreamBatch]:
        sample_rng = np.random.default_rng(self.seed)
        corrupt_rng = np.random.default_rng(self.seed + 1)
        phase_corruptions = self.known_corruptions + self.novel_corruptions
        novel_start = self.spec.extras["novel_start_batch"]
        drift_set = set(self.spec.drift_indices)
        pool_n = self._pool_x.shape[0]

        t = 0
        for phase_i, corruption in enumerate(phase_corruptions):
            for _ in range(self.n_batches_per_phase):
                idx = sample_rng.integers(0, pool_n, size=self.batch_size)
                x_clean = self._pool_x[idx]                 # (B, 3, H, W)
                y = self._pool_y[idx]
                x = apply_corruption(corruption, x_clean, self.severity, corrupt_rng)
                yield StreamBatch(
                    index=t, x=x, y=y, concept_id=phase_i,
                    is_drift=(t in drift_set), true_posterior=None,
                    extras={"corruption": corruption, "is_novel": t >= novel_start},
                )
                t += 1
