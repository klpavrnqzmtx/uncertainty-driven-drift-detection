"""LeNet-5 with MC-Dropout for 10-way MNIST(-C) inference.

The Figure-2 pivot (MNIST-C) needs a 10-way classifier with a principled
unsupervised uncertainty channel. We train a small LeNet-5 variant with
dropout layers on clean MNIST, cache the weights, and at inference time
draw ``n_samples`` stochastic forward passes by keeping the dropout
layers in train mode while everything else stays in eval.

The mean of those passes is written to ``Prediction.probs``; the full
``(S, B, 10)`` stack is stashed in ``Prediction.extras['mc_probs']`` so
that :class:`MCDropoutUncertainty` can pick it up without re-doing the
forward passes.

We deliberately keep the CNN frozen during the stream: the point of the
MNIST-C benchmark is to measure how uncertainty tracks an *unchanging*
model against a drifting input distribution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from uncertainty_driven_drift.data.base import StreamBatch, StreamSpec
from uncertainty_driven_drift.models.base import Classifier, Prediction
from uncertainty_driven_drift.registry import register


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def _build_lenet_dropout(n_classes: int = 10, p_drop: float = 0.25):
    """LeNet-5 with dropout on the FC block."""
    import torch.nn as nn

    class LeNet5Dropout(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 6, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(2),
                nn.Conv2d(6, 16, kernel_size=5),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(2),
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(16 * 5 * 5, 120),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p_drop),
                nn.Linear(120, 84),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p_drop),
            )
            self.head = nn.Linear(84, n_classes)

        def forward(self, x):
            return self.head(self.fc(self.features(x)))

    return LeNet5Dropout()


_DATASET_CLS = {
    "mnist": "MNIST",
    "fashion_mnist": "FashionMNIST",
    "kmnist": "KMNIST",
}


def _pretrain(
    data_root: Path,
    ckpt_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    p_drop: float,
    train_corruptions: list | None = None,
    corruption_severity: int = 3,
    dataset_name: str = "mnist",
) -> None:
    """Train LeNet+Dropout 10-way on the chosen grayscale dataset; save weights.

    When ``train_corruptions`` is provided, each mini-batch is randomly
    corrupted by one of the named corruption types before the forward
    pass. This makes the model robust to those specific corruptions so
    that epistemic uncertainty stays low when it sees them at test time,
    while spiking on novel (unseen) corruptions.
    """
    import random

    import torch
    from torch import optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    torch.manual_seed(seed)
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    ds_cls_name = _DATASET_CLS.get(dataset_name, "MNIST")
    ds_cls = getattr(datasets, ds_cls_name)

    tfm = transforms.ToTensor()
    train_ds = ds_cls(str(data_root), train=True, download=True, transform=tfm)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = _build_lenet_dropout(n_classes=10, p_drop=p_drop)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    if train_corruptions:
        from uncertainty_driven_drift.components.mnist_c_corruptions import apply_corruption

    model.train()
    for _ in range(int(epochs)):
        for xb, yb in loader:
            if train_corruptions:
                corruption = py_rng.choice(train_corruptions)
                xb_np = xb.numpy()          # (B, 1, 28, 28) float32
                xb_np = apply_corruption(corruption, xb_np, corruption_severity, np_rng)
                xb = torch.from_numpy(xb_np)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(ckpt_path))


def _enable_mc_dropout(model) -> None:
    """Set all Dropout layers to train mode while everything else stays eval.

    LeNet has no BatchNorm, so we could equivalently call ``model.train()``;
    this is more surgical and survives future additions of BN / running-stats
    modules without accidentally corrupting their statistics.
    """
    import torch.nn as nn

    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


# ---------------------------------------------------------------------------
# Classifier wrapper
# ---------------------------------------------------------------------------

@register("model", "lenet_mc_dropout")
class LeNetMCDropout(Classifier):
    """Frozen MC-Dropout LeNet exposing ``(S, B, n_classes)`` MC samples.

    Parameters
    ----------
    n_samples :
        Number of stochastic forward passes per batch at inference time.
    p_drop :
        Dropout probability, fixed across train and MC inference.
    data_root :
        Cache location for the MNIST train set (used for pre-training).
    ckpt_path :
        Where to cache pretrained weights. If the file exists it is
        loaded and ``pretrain_*`` parameters are ignored.
    pretrain_epochs, pretrain_batch_size, pretrain_lr, pretrain_seed :
        Pre-training hyperparameters. Only used on first run.
    seed :
        Seeds the torch RNG used by dropout so the MC sampling is
        reproducible across calls.
    """

    def __init__(
        self,
        n_samples: int = 20,
        p_drop: float = 0.25,
        data_root: str = "./artifacts/mnist",
        ckpt_path: str = "./artifacts/models/lenet_mc_mnist.pt",
        pretrain_epochs: int = 2,
        pretrain_batch_size: int = 128,
        pretrain_lr: float = 1e-3,
        pretrain_seed: int = 0,
        seed: int = 0,
        train_corruptions: list | None = None,
        corruption_severity: int = 3,
        dataset_name: str = "mnist",
    ) -> None:
        self.n_samples = int(n_samples)
        self.p_drop = float(p_drop)
        self.data_root = str(data_root)
        self.ckpt_path = str(ckpt_path)
        self.pretrain_epochs = int(pretrain_epochs)
        self.pretrain_batch_size = int(pretrain_batch_size)
        self.pretrain_lr = float(pretrain_lr)
        self.pretrain_seed = int(pretrain_seed)
        self.seed = int(seed)
        self.train_corruptions = list(train_corruptions) if train_corruptions else None
        self.corruption_severity = int(corruption_severity)
        self.dataset_name = str(dataset_name)

        self._cnn = None  # populated in setup()

    def setup(self, spec: StreamSpec) -> None:
        if spec.n_classes != 10:
            raise ValueError(
                f"lenet_mc_dropout is 10-way; got n_classes={spec.n_classes}"
            )
        if tuple(spec.input_shape) != (1, 28, 28):
            raise ValueError(
                f"lenet_mc_dropout expects (1,28,28); got {spec.input_shape}"
            )
        import torch

        ckpt = Path(self.ckpt_path)
        if not ckpt.exists():
            _pretrain(
                data_root=Path(self.data_root),
                ckpt_path=ckpt,
                epochs=self.pretrain_epochs,
                batch_size=self.pretrain_batch_size,
                lr=self.pretrain_lr,
                seed=self.pretrain_seed,
                p_drop=self.p_drop,
                train_corruptions=self.train_corruptions,
                corruption_severity=self.corruption_severity,
                dataset_name=self.dataset_name,
            )

        torch.manual_seed(self.seed)
        model = _build_lenet_dropout(n_classes=10, p_drop=self.p_drop)
        model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
        _enable_mc_dropout(model)
        self._cnn = model

    def predict(self, batch: StreamBatch) -> Prediction:
        import torch
        import torch.nn.functional as F

        assert self._cnn is not None
        x = torch.from_numpy(np.ascontiguousarray(batch.x)).float()

        samples: list[np.ndarray] = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                logits = self._cnn(x)
                probs = F.softmax(logits, dim=-1)
                samples.append(probs.detach().numpy().astype(np.float64))

        mc_probs = np.stack(samples, axis=0)          # (S, B, 10)
        mean_probs = mc_probs.mean(axis=0)            # (B, 10)
        return Prediction(
            probs=mean_probs.astype(np.float64),
            features=None,
            extras={"mc_probs": mc_probs},
        )

    def observe(self, batch: StreamBatch, prediction: Prediction) -> None:
        """Frozen backbone — observing labels is a no-op by design."""
        del batch, prediction
