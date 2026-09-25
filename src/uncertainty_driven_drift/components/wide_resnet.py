"""Wide ResNet WRN-28-10 (Zagoruyko & Komodakis, 2016) for CIFAR-100.

The strong from-scratch CIFAR-100 baseline (~36M params).  Reuses the CIFAR
training loop, known-corruption augmentation, MC-Dropout enabling and diagonal
last-layer Laplace machinery from :mod:`resnet_cifar` — the only new piece is
the backbone builder, injected through the ``_BUILD_FN`` class attribute.  So
the two arms are, again, small subclasses.

Registered components
---------------------
* ``wrn2810_mc_dropout`` — WRN-28-10 + MC-Dropout, CIFAR-100 (100-way).
* ``wrn2810_laplace``    — WRN-28-10 + diagonal last-layer Laplace, CIFAR-100.
"""

from __future__ import annotations

from uncertainty_driven_drift.components.resnet_cifar import (
    ResNetLaplaceCIFAR,
    ResNetMCDropoutCIFAR,
)
from uncertainty_driven_drift.registry import register


def _build_wrn2810(n_classes: int = 100, p_drop: float = 0.0, in_channels: int = 3,
                   depth: int = 28, widen: int = 10, block_drop: float = 0.1):
    """WRN-``depth``-``widen`` with a ``head`` Linear and an ``embed()`` returning
    the pooled penultimate feature.

    Two dropouts, deliberately separate:

    * ``block_drop`` — between the two 3x3 convs of each block, the paper's
      regulariser. Used during training only. Paper default is 0.3, tuned for
      flip+crop-only augmentation; lowered to 0.1 here because this backbone is
      additionally trained under heavy per-batch corruption augmentation
      (severity-3 noise/blur/compression on up to 80% of batches) — stacking
      the paper's full block dropout on top of that already-aggressive input
      noise was believed to under-fit. MEASURED 12 Sep: it does not -- 0.1 and
      the paper's 0.3 land within noise of each other (v4 clean 0.753 vs v3
      0.760), so this is a preference, not a correction.
    * ``p_drop`` (``head_drop``) — on the pooled penultimate feature, immediately
      before the head. This is the one MC-Dropout samples, matching ResNet-20 and
      the ViT arm.

    The distinction matters: perturbing conv feature maps deep in the network adds
    noise that is largely input-INDEPENDENT, which swamps the input-dependent
    disagreement MC-Dropout needs. A WRN with only block dropout produced a flat
    epistemic signal (ratio 0.99 across the drift boundary) while the pre-head
    arms responded 3-4x.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    assert (depth - 4) % 6 == 0, "WRN depth must be 6n+4"
    n = (depth - 4) // 6
    widths = [16, 16 * widen, 32 * widen, 64 * widen]

    class BasicBlock(nn.Module):
        def __init__(self, in_p: int, out_p: int, stride: int) -> None:
            super().__init__()
            self.bn1 = nn.BatchNorm2d(in_p)
            self.conv1 = nn.Conv2d(in_p, out_p, 3, stride, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_p)
            self.conv2 = nn.Conv2d(out_p, out_p, 3, 1, 1, bias=False)
            self.drop = nn.Dropout(block_drop)
            self.equal = (in_p == out_p and stride == 1)
            self.shortcut = None if self.equal else nn.Conv2d(in_p, out_p, 1, stride, 0, bias=False)

        def forward(self, x):
            o = F.relu(self.bn1(x))
            s = x if self.equal else self.shortcut(o)
            o = self.conv1(o)
            o = self.conv2(self.drop(F.relu(self.bn2(o))))
            return o + s

    def _group(in_p, out_p, stride):
        return nn.Sequential(*[
            BasicBlock(in_p if i == 0 else out_p, out_p, stride if i == 0 else 1)
            for i in range(n)
        ])

    class WRN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, widths[0], 3, 1, 1, bias=False)
            self.g1 = _group(widths[0], widths[1], 1)
            self.g2 = _group(widths[1], widths[2], 2)
            self.g3 = _group(widths[2], widths[3], 2)
            self.bn = nn.BatchNorm2d(widths[3])
            # Sampled by MC-Dropout; `embed()` deliberately excludes it so the
            # Laplace arm gets deterministic features.
            self.head_drop = nn.Dropout(p_drop)
            self.head = nn.Linear(widths[3], n_classes)

        def embed(self, x):
            o = self.conv1(x)
            o = self.g3(self.g2(self.g1(o)))
            o = F.relu(self.bn(o))
            return F.adaptive_avg_pool2d(o, 1).flatten(1)   # (B, widths[3])

        def forward(self, x):
            return self.head(self.head_drop(self.embed(x)))

    model = WRN()

    # Explicit He/MSR init (Zagoruyko & Komodakis' own reference implementation
    # inits conv weights this way rather than relying on PyTorch's default
    # kaiming_uniform_(a=sqrt(5))). Matters more here than for ResNet-20: a
    # 36M-param, 28-layer net trained from scratch under noisy per-batch
    # corruption augmentation is more sensitive to init-time activation scale.
    def _init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.zeros_(m.bias)

    model.apply(_init)
    return model


@register("model", "wrn2810_mc_dropout")
class WRN2810MCDropout(ResNetMCDropoutCIFAR):
    """WRN-28-10 + MC-Dropout for CIFAR-100 known-vs-novel detection."""

    _DATASET = "cifar100"
    _N_CLASSES = 100
    _ARCH = "WRN-28-10"
    _BUILD_FN = staticmethod(_build_wrn2810)


@register("model", "wrn2810_laplace")
class WRN2810Laplace(ResNetLaplaceCIFAR):
    """WRN-28-10 + diagonal last-layer Laplace for CIFAR-100 known-vs-novel."""

    _DATASET = "cifar100"
    _N_CLASSES = 100
    _ARCH = "WRN-28-10"
    _BUILD_FN = staticmethod(_build_wrn2810)
