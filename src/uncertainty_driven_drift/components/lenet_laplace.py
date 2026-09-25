"""LeNet feature extractor + frozen last-layer Laplace binary head.

The Figure-2 benchmark asks for a small CNN pre-trained on MNIST, with a
last-layer Laplace approximation providing per-sample uncertainty. We
decompose that into:

1. A LeNet-5-style CNN trained 10-way on MNIST to produce a frozen
   84-D feature representation. Weights are cached under
   ``artifacts/models/lenet_mnist.pt`` so the pre-training runs once.
2. A binary :class:`BayesianLogisticRegression` head on top of those
   frozen features, fitted on the *training split of the first task
   only*. Because that component already implements Newton/IRLS + a
   Laplace Gaussian posterior with MC sampling, reusing it gives us
   last-layer Laplace with the same uncertainty decomposition used in
   Figure 1 — no new math, no extra dependency on ``laplace-torch``.

During the stream the default is ``refit_every = 0`` (frozen head):
``observe`` is a no-op, so posterior-MI changes only because the *data
distribution* changes, not because the model adapts. Set
``refit_every`` to a positive integer to enable online refits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from uncertainty_driven_drift.components.bayes_logreg import BayesianLogisticRegression
from uncertainty_driven_drift.components.mnist_tasks import task_label_vector
from uncertainty_driven_drift.data.base import StreamBatch, StreamSpec
from uncertainty_driven_drift.models.base import Classifier, Prediction
from uncertainty_driven_drift.registry import register


# ---------------------------------------------------------------------------
# LeNet-5 feature extractor (torch)
# ---------------------------------------------------------------------------

_FEATURE_DIM = 84


def _build_lenet():
    """Construct the LeNet-5 used for pre-training. Import torch lazily."""
    import torch.nn as nn

    class LeNet5(nn.Module):
        def __init__(self, n_classes: int = 10, feature_dim: int = _FEATURE_DIM) -> None:
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
                nn.Linear(120, feature_dim),
                nn.ReLU(inplace=True),
            )
            self.head = nn.Linear(feature_dim, n_classes)

        def embed(self, x):
            return self.fc(self.features(x))

        def forward(self, x):
            return self.head(self.embed(x))

    return LeNet5()


def _pretrain_lenet(
    data_root: Path,
    ckpt_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> None:
    """Train LeNet 10-way on MNIST for a few epochs and save weights."""
    import torch
    from torch import optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    torch.manual_seed(seed)
    tfm = transforms.ToTensor()
    train_ds = datasets.MNIST(str(data_root), train=True, download=True, transform=tfm)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = _build_lenet()
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for _ in range(int(epochs)):
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(ckpt_path))


def _embed_numpy(model, x: np.ndarray) -> np.ndarray:
    """Feed a numpy batch through ``model.embed`` and return 84-D features."""
    import torch

    with torch.no_grad():
        xt = torch.from_numpy(np.ascontiguousarray(x)).float()
        feats = model.embed(xt)
    return feats.detach().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

@register("model", "lenet_laplace")
class LeNetLaplace(Classifier):
    """Frozen-CNN + binary last-layer Laplace head for MNIST task streams.

    Parameters
    ----------
    initial_task :
        Name of the first task in the stream. The binary head's initial
        Laplace fit uses the MNIST *training* split labelled under this
        task.
    prior_precision, max_newton_iter :
        Passed to the underlying :class:`BayesianLogisticRegression`.
    n_init_fit :
        Number of training-split images used for the initial fit.
    refit_every :
        If ``0`` (default), the head is frozen: ``observe`` is a no-op
        so posterior and MAP stay at the initial fit. Otherwise the
        head refits on a sliding window of the last ``window_size``
        *stream* features every ``refit_every`` batches.
    window_size :
        Sliding-window size used only when ``refit_every > 0``.
    data_root, ckpt_path :
        Cache locations for MNIST data and LeNet weights.
    pretrain_epochs, pretrain_batch_size, pretrain_lr, pretrain_seed :
        LeNet pre-training hyperparameters. Only used if ``ckpt_path``
        does not already exist.
    """

    def __init__(
        self,
        initial_task: str,
        prior_precision: float = 1.0,
        max_newton_iter: int = 50,
        n_init_fit: int = 10000,
        refit_every: int = 0,
        window_size: int = 5000,
        data_root: str = "./artifacts/mnist",
        ckpt_path: str = "./artifacts/models/lenet_mnist.pt",
        pretrain_epochs: int = 2,
        pretrain_batch_size: int = 128,
        pretrain_lr: float = 1e-3,
        pretrain_seed: int = 0,
    ) -> None:
        self.initial_task = str(initial_task)
        self.prior_precision = float(prior_precision)
        self.max_newton_iter = int(max_newton_iter)
        self.n_init_fit = int(n_init_fit)
        self.refit_every = int(refit_every)
        self.window_size = int(window_size)
        self.data_root = str(data_root)
        self.ckpt_path = str(ckpt_path)
        self.pretrain_epochs = int(pretrain_epochs)
        self.pretrain_batch_size = int(pretrain_batch_size)
        self.pretrain_lr = float(pretrain_lr)
        self.pretrain_seed = int(pretrain_seed)

        self._cnn = None  # populated in setup()
        self._head: Optional[BayesianLogisticRegression] = None

    # ---- API ---------------------------------------------------------------

    def setup(self, spec: StreamSpec) -> None:
        if spec.n_classes != 2:
            raise ValueError(
                f"lenet_laplace is binary; got n_classes={spec.n_classes}"
            )
        if tuple(spec.input_shape) != (1, 28, 28):
            raise ValueError(
                f"lenet_laplace expects input_shape=(1,28,28); got {spec.input_shape}"
            )

        self._cnn = self._load_or_pretrain_cnn()
        self._head = self._fit_initial_head()

    def predict(self, batch: StreamBatch) -> Prediction:
        assert self._cnn is not None and self._head is not None
        feats = _embed_numpy(self._cnn, batch.x)
        feat_batch = _FeatureBatch(x=feats, y=batch.y)
        pred = self._head.predict(feat_batch)
        return Prediction(probs=pred.probs, features=pred.features)

    def observe(self, batch: StreamBatch, prediction: Prediction) -> None:
        assert self._head is not None
        if self.refit_every <= 0:
            return
        feats = _embed_numpy(self._cnn, batch.x)
        feat_batch = _FeatureBatch(x=feats, y=batch.y)
        self._head.observe(feat_batch, prediction)

    # ---- Introspection used by laplace_mc ---------------------------------
    #
    # laplace_mc requires a BayesianLogisticRegression; expose the head's
    # attributes transparently so ``isinstance(self._head, BLR)`` holds and
    # the estimator can reach it via ``model._head`` or via the duck-typing
    # properties below.

    @property
    def d(self) -> int:
        assert self._head is not None
        return self._head.d

    @property
    def is_fitted(self) -> bool:
        return self._head is not None and self._head.is_fitted

    @property
    def map_weights(self) -> np.ndarray:
        assert self._head is not None
        return self._head.map_weights

    def posterior_samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        assert self._head is not None
        return self._head.posterior_samples(n, rng)

    # ---- Internals --------------------------------------------------------

    def _load_or_pretrain_cnn(self):
        import torch

        ckpt = Path(self.ckpt_path)
        if not ckpt.exists():
            _pretrain_lenet(
                data_root=Path(self.data_root),
                ckpt_path=ckpt,
                epochs=self.pretrain_epochs,
                batch_size=self.pretrain_batch_size,
                lr=self.pretrain_lr,
                seed=self.pretrain_seed,
            )
        model = _build_lenet()
        model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
        model.eval()
        return model

    def _fit_initial_head(self) -> BayesianLogisticRegression:
        """Compute features on the MNIST train split and fit the binary head."""
        x, digits = _load_mnist_train_pool(Path(self.data_root))
        rng = np.random.default_rng(self.pretrain_seed)
        n = min(self.n_init_fit, x.shape[0])
        idx = rng.choice(x.shape[0], size=n, replace=False)
        x_sub = x[idx]
        d_sub = digits[idx]

        task_vec = task_label_vector(self.initial_task)
        y_sub = task_vec[d_sub].astype(np.int64)

        feats = _embed_in_chunks(self._cnn, x_sub, chunk=1024)

        # Initialize the BLR with refit_every=1 so the priming observe()
        # always triggers a first Laplace fit, independent of the user's
        # adaptive cadence. We apply ``self.refit_every`` to the head only
        # *after* the initial fit has completed.
        head = BayesianLogisticRegression(
            prior_precision=self.prior_precision,
            # Window must fit the whole init set so the first fit sees it all.
            window_size=max(n, self.window_size),
            warmup=0,
            refit_every=1,
            max_newton_iter=self.max_newton_iter,
        )
        head.setup(StreamSpec(
            name="lenet_features",
            input_shape=(feats.shape[1],),
            n_classes=2,
            n_batches=0,
            batch_size=0,
        ))

        feat_batch = _FeatureBatch(x=feats, y=y_sub)
        pred = head.predict(feat_batch)
        head.observe(feat_batch, pred)  # appends + fits

        if not head.is_fitted:
            raise RuntimeError(
                "Initial Laplace fit did not converge — consider increasing "
                "n_init_fit or max_newton_iter."
            )

        # Hand over to the user's adaptive cadence and sliding-window size.
        if self.refit_every <= 0:
            # Frozen: clear the buffer so any accidental observe() is a no-op.
            head._buffer_X.clear()
            head._buffer_y.clear()
            head.refit_every = 1  # value is moot while observe() is guarded
        else:
            head.refit_every = int(self.refit_every)
            # Reset the per-BLR observation counter so the next refit happens
            # exactly ``refit_every`` stream batches from now, rather than
            # drifting off the single priming call.
            head._n_observed = 0
        head.window_size = self.window_size
        return head


def _load_mnist_train_pool(cache_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load MNIST *train* images + digit labels for the initial head fit."""
    try:
        from torchvision import datasets  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "lenet_laplace requires torchvision; install '.[torch]' extras"
        ) from exc
    cache_root.mkdir(parents=True, exist_ok=True)
    ds = datasets.MNIST(str(cache_root), train=True, download=True, transform=None)
    x = ds.data.numpy().astype(np.float32) / 255.0
    x = x[:, None, :, :]
    y = ds.targets.numpy().astype(np.int64)
    return x, y


# Small helper so we can call the BLR's predict/observe on raw feature
# batches without inventing a new StreamBatch variant. Only .x and .y are
# accessed by BayesianLogisticRegression.
class _FeatureBatch:
    __slots__ = ("x", "y")

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.y = y


def _embed_in_chunks(cnn, x: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """Embed ``x`` in fixed-size chunks to cap peak memory."""
    parts = []
    for start in range(0, x.shape[0], chunk):
        parts.append(_embed_numpy(cnn, x[start : start + chunk]))
    return np.concatenate(parts, axis=0)
