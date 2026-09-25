"""Input-shape guards and channel statistics shared by the image backbones.

Every vision component in this repo is written against a *specific* tensor
layout — LeNet assumes ``(1, 28, 28)``, ResNet-20 assumes ``(3, 32, 32)``,
the Camelyon17 ResNet-18 assumes ``(3, 96, 96)``.  Handing one of them a
tabular stream (Elec2 ``(8,)``), a synthetic GMM stream ``(2,)`` or the
wireless QPSK stream ``(2,)`` produces either a confusing torch shape error
deep inside the forward pass or — worse — a silently wrong run.

:func:`require_image_spec` turns that into an explicit, named failure at
``setup()`` time, which is where a config typo should surface.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from uncertainty_driven_drift.data.base import StreamSpec

# Per-channel statistics used to normalise inputs before a backbone.
# Grayscale = MNIST train set; RGB = CIFAR-10 train set.
_MNIST_MEAN = (0.1307,)
_MNIST_STD = (0.3081,)
_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2470, 0.2435, 0.2616)


def require_image_spec(
    spec: StreamSpec,
    component: str,
    *,
    channels: Optional[Sequence[int]] = None,
    spatial: Optional[Sequence[Tuple[int, int]]] = None,
    n_classes: Optional[int] = None,
) -> Tuple[int, int, int]:
    """Assert ``spec`` describes an image stream this component can consume.

    Parameters
    ----------
    component :
        Registry name, used verbatim in the error message.
    channels :
        Allowed channel counts, e.g. ``(1, 3)``. ``None`` allows any.
    spatial :
        Allowed ``(H, W)`` pairs. ``None`` allows any.
    n_classes :
        Required class count. ``None`` allows any.

    Returns
    -------
    The validated ``(C, H, W)``.
    """
    shape = tuple(int(d) for d in spec.input_shape)
    if len(shape) != 3:
        raise ValueError(
            f"{component} is an image model and needs a (C, H, W) stream; "
            f"stream {spec.name!r} has input_shape={shape}. "
            f"Image streams in this repo: mnist_c, mnist_known_novel, mnist_tasks, "
            f"fashion_known_novel, kmnist_known_novel, cifar_known_novel, "
            f"camelyon17_known_novel."
        )

    c, h, w = shape
    if channels is not None and c not in tuple(channels):
        raise ValueError(
            f"{component} supports {tuple(channels)} input channel(s); "
            f"stream {spec.name!r} has {c}."
        )
    if spatial is not None and (h, w) not in {tuple(s) for s in spatial}:
        raise ValueError(
            f"{component} supports spatial sizes {[tuple(s) for s in spatial]}; "
            f"stream {spec.name!r} has ({h}, {w})."
        )
    if n_classes is not None and spec.n_classes != n_classes:
        raise ValueError(
            f"{component} is {n_classes}-way; stream {spec.name!r} has "
            f"n_classes={spec.n_classes}."
        )
    return c, h, w


def default_norm_stats(channels: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel ``(mean, std)`` for ``channels``-channel image streams."""
    if channels == 1:
        mean, std = _MNIST_MEAN, _MNIST_STD
    elif channels == 3:
        mean, std = _CIFAR_MEAN, _CIFAR_STD
    else:
        mean, std = (0.5,) * channels, (0.5,) * channels
    return np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)
