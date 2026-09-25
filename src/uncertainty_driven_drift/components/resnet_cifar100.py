"""CIFAR-100 ResNet-20 arms (MC-Dropout + diagonal last-layer Laplace).

The 100-class analogue of :mod:`resnet_cifar`.  These reuse the *entire*
ResNet-20 architecture, known-corruption augmentation, MC-Dropout and Laplace
machinery from :mod:`resnet_cifar` — the only differences are the dataset
(CIFAR-100), the class count (100) and the normalisation statistics, all of
which the base classes read from the ``_DATASET`` / ``_N_CLASSES`` class
attributes.  So the CIFAR-100 arms are one-line subclasses.

Registered components
---------------------
* ``resnet_mc_dropout_cifar100`` — ResNet-20 + MC-Dropout, 100-way.
* ``resnet_laplace_cifar100``    — ResNet-20 + diagonal last-layer Laplace, 100-way.
"""

from __future__ import annotations

from uncertainty_driven_drift.components.resnet_cifar import (
    ResNetLaplaceCIFAR,
    ResNetMCDropoutCIFAR,
)
from uncertainty_driven_drift.registry import register


@register("model", "resnet_mc_dropout_cifar100")
class ResNetMCDropoutCIFAR100(ResNetMCDropoutCIFAR):
    """ResNet-20 + MC-Dropout for CIFAR-100 known-vs-novel detection."""

    _DATASET = "cifar100"
    _N_CLASSES = 100
    _ARCH = "ResNet-20"


@register("model", "resnet_laplace_cifar100")
class ResNetLaplaceCIFAR100(ResNetLaplaceCIFAR):
    """ResNet-20 + diagonal last-layer Laplace for CIFAR-100 known-vs-novel detection."""

    _DATASET = "cifar100"
    _N_CLASSES = 100
    _ARCH = "ResNet-20"
