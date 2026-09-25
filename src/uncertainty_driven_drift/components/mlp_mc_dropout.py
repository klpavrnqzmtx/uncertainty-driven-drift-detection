"""Tabular MLP with MC-Dropout — online prequential learner.

Designed for real-world streaming tabular datasets (Elec2, Insects, etc.).

Architecture
------------
A stack of ``n_layers`` blocks (Linear → ReLU → Dropout) followed by a
linear classification head.  Dropout is left **active during inference**
so that ``n_samples`` stochastic forward passes yield the standard BALD
uncertainty decomposition::

    total_i     = H[ E_θ p_θ(y|x_i) ]   (predictive entropy)
    aleatoric_i = E_θ H[ p_θ(y|x_i) ]   (expected per-sample entropy)
    epistemic_i = total_i - aleatoric_i  (mutual information)

This is identical to the MC-Dropout treatment used for MNIST-C, so the
same ``mc_dropout`` uncertainty estimator works without modification.

Online learning
---------------
After each prediction the model takes one Adam gradient step on the
current batch (cross-entropy loss).  A ``warmup`` period of ``warmup``
batches suppresses uncertainty reads until the model has seen enough data
to produce meaningful estimates (during warmup ``predict`` still works —
it just returns near-uniform predictions that the uncertainty estimator
will correctly score as highly uncertain).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from uncertainty_driven_drift.data.base import StreamBatch, StreamSpec
from uncertainty_driven_drift.models.base import Classifier, Prediction
from uncertainty_driven_drift.registry import register


# ---------------------------------------------------------------------------
# Torch MLP definition
# ---------------------------------------------------------------------------

def _build_mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    n_classes: int,
    p_drop: float,
):
    import torch.nn as nn

    layers: List = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(p=p_drop)]
        prev = h
    layers.append(nn.Linear(prev, n_classes))

    return nn.Sequential(*layers)


def _enable_dropout(model) -> None:
    """Set all Dropout layers to training mode (activates stochastic masking)."""
    import torch.nn as nn
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


# ---------------------------------------------------------------------------
# Registered classifier
# ---------------------------------------------------------------------------

@register("model", "mlp_mc_dropout")
class TabularMLPMCDropout(Classifier):
    """Tabular MLP with MC-Dropout uncertainty for streaming tabular data.

    Parameters
    ----------
    hidden_dims :
        Hidden layer widths, e.g. ``[64, 32]``.
    p_drop :
        Dropout probability applied after every hidden ReLU.
    n_samples :
        Number of stochastic forward passes at inference time.
    lr :
        Adam learning rate for online updates.
    weight_decay :
        L2 regularisation for Adam.
    warmup :
        Number of batches to train before uncertainty estimates are
        considered reliable.  Predictions are still made during warmup.
    seed :
        RNG seed for torch weight initialisation.
    """

    def __init__(
        self,
        hidden_dims: Sequence[int] = (64, 32),
        p_drop: float = 0.3,
        n_samples: int = 30,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup: int = 5,
        seed: int = 0,
    ) -> None:
        self.hidden_dims = list(hidden_dims)
        self.p_drop = float(p_drop)
        self.n_samples = int(n_samples)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.warmup = int(warmup)
        self.seed = int(seed)

        self._model = None
        self._optimizer = None
        self._n_classes: int = 2
        self._in_dim: int = 0
        self._n_batches: int = 0

    def setup(self, spec: StreamSpec) -> None:
        import torch

        torch.manual_seed(self.seed)
        self._n_classes = int(spec.n_classes)
        self._in_dim = int(np.prod(spec.input_shape))
        self._model = _build_mlp(
            self._in_dim, self.hidden_dims, self._n_classes, self.p_drop
        )
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self._n_batches = 0

    def predict(self, batch: StreamBatch) -> Prediction:
        import torch
        import torch.nn.functional as F

        assert self._model is not None
        x = torch.tensor(
            np.asarray(batch.x, dtype=np.float32).reshape(-1, self._in_dim),
            dtype=torch.float32,
        )

        # MC-Dropout inference: dropout active, no_grad for memory efficiency.
        self._model.eval()
        _enable_dropout(self._model)

        mc_logits: List[np.ndarray] = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                logits = self._model(x)
                mc_logits.append(F.softmax(logits, dim=-1).cpu().numpy())

        # (S, B, K)
        mc_probs = np.stack(mc_logits, axis=0)
        mean_probs = mc_probs.mean(axis=0)          # (B, K)

        return Prediction(
            probs=mean_probs,
            extras={"mc_probs": mc_probs},
        )

    def observe(self, batch: StreamBatch, prediction: Prediction) -> None:
        import torch
        import torch.nn.functional as F

        assert self._model is not None and self._optimizer is not None

        x = torch.tensor(
            np.asarray(batch.x, dtype=np.float32).reshape(-1, self._in_dim),
            dtype=torch.float32,
        )
        y = torch.tensor(
            np.asarray(batch.y, dtype=np.int64),
            dtype=torch.long,
        )

        self._model.train()
        self._optimizer.zero_grad()
        logits = self._model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        self._optimizer.step()

        self._n_batches += 1
