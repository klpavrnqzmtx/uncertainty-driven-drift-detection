"""Phase 3 component bundle.

Importing this module registers everything Figure 2 needs:

* the MNIST changing-task stream + frozen LeNet + last-layer Laplace head
  (``mnist_tasks`` + ``lenet_laplace``), kept for the ablation/archive;
* the MNIST-C corrupted-input stream + MC-Dropout LeNet + MC-Dropout
  uncertainty decomposition, which is the current headline benchmark;
* the river-backed drift detectors shared by both.

Phase 2 components are pulled in transitively so a single import of
``phase3`` covers any Figure-2-family run.
"""

from uncertainty_driven_drift.components import phase2                    # noqa: F401
from uncertainty_driven_drift.components import lenet_laplace             # noqa: F401
from uncertainty_driven_drift.components import mc_dropout_lenet          # noqa: F401
from uncertainty_driven_drift.components import mc_dropout_uncertainty    # noqa: F401
from uncertainty_driven_drift.components import mlp_mc_dropout            # noqa: F401  (tabular MLP for Elec2/Insects)
from uncertainty_driven_drift.components import mnist_c                   # noqa: F401  (also registers mnist_known_novel)
from uncertainty_driven_drift.components import mnist_tasks               # noqa: F401
from uncertainty_driven_drift.components import river_datasets            # noqa: F401  (elec2, insects)
from uncertainty_driven_drift.components import river_detectors           # noqa: F401
from uncertainty_driven_drift.components import fashion_kmnist            # noqa: F401  (fashion_known_novel, kmnist_known_novel)
from uncertainty_driven_drift.components import camelyon17                # noqa: F401  (camelyon17_known_novel, lenet_mc_dropout_rgb, resnet_mc_dropout)
from uncertainty_driven_drift.components import quadriga                   # noqa: F401  (quadriga_known_novel, rnn_mc_dropout_qpsk)
from uncertainty_driven_drift.components import quadriga_drift             # noqa: F401  (quadriga_channel_drift, rnn_equalizer_mc_dropout/laplace)
from uncertainty_driven_drift.components import quadriga_snr_drift         # noqa: F401  (quadriga_snr_drift, rnn_equalizer_snr_mc_dropout/laplace)
from uncertainty_driven_drift.components import cifar_c                   # noqa: F401  (cifar_known_novel)
from uncertainty_driven_drift.components import cifar100_c               # noqa: F401  (cifar100_known_novel)
from uncertainty_driven_drift.components import cifar_severity           # noqa: F401  (cifar_severity_shift)
from uncertainty_driven_drift.components import resnet_cifar              # noqa: F401  (resnet_mc_dropout_cifar, resnet_laplace_cifar)
from uncertainty_driven_drift.components import resnet_cifar100           # noqa: F401  (resnet_mc_dropout_cifar100, resnet_laplace_cifar100)
from uncertainty_driven_drift.components import wide_resnet               # noqa: F401  (wrn2810_mc_dropout, wrn2810_laplace)
from uncertainty_driven_drift.components import imagenet_c                # noqa: F401  (imagenet_known_novel)
from uncertainty_driven_drift.components import pretrained_vit            # noqa: F401  (vit_b16_mc_dropout, vit_b16_laplace)
from uncertainty_driven_drift.components import resnet_mnist             # noqa: F401  (resnet_mc_dropout_mnist, resnet_laplace_mnist)
from uncertainty_driven_drift.components import vit                      # noqa: F401  (vit_mc_dropout, vit_laplace)
from uncertainty_driven_drift.components import cifar_ambiguity         # noqa: F401  (cifar_known_ambiguous — scenario S2)
from uncertainty_driven_drift.components import emnist_ambiguity        # noqa: F401  (emnist_label_prior — S2-benign)
from uncertainty_driven_drift.components import audio_c                  # noqa: F401  (audio_known_novel — S1, audio modality)
from uncertainty_driven_drift.components import audio_cnn                # noqa: F401  (audio_cnn_mc_dropout, audio_cnn_laplace)
