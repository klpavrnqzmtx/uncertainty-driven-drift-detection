"""CIFAR-10 known-then-AMBIGUOUS stream — scenario S2.

What S2 is for
--------------
S1 (``cifar_known_novel``) shifts to corruption families the model has never seen, and
both total entropy and epistemic uncertainty rise: on the committed runs their AUROCs are
0.799 and 0.802, i.e. the decomposition buys nothing there.

S2 shifts to inputs that are **ambiguous but familiar**. Two test images of different
classes are blended, so the label is genuinely uncertain — but a model pretrained with
mixup has seen exactly this kind of input, so nothing about it is novel. The prediction
should therefore become *spread* without becoming *unfamiliar*:

    total entropy   rises sharply   -> a total-entropy detector (UDD) alarms   [WRONG]
    epistemic       stays flat      -> an epistemic detector stays quiet       [RIGHT]

Measured on a 10-epoch mixup fine-tune of the committed ViT, at blend 0.5:
total +12.98 sigma, epistemic -0.11 sigma. Without mixup pretraining the same shift gives
+7.96 vs +7.69 sigma (no separation at all), because unseen blends are simply
out-of-distribution — so ``mixup_alpha`` in the model config is load-bearing, not a
regularisation detail.

Stream layout (mirrors S1 so every existing figure and detector works unchanged)
-------------------------------------------------------------------------------
Phase A: the known corruptions, unblended (``blend_lambda = 1.0``).
Phase B: the same known corruptions, now blended at a ramp of decreasing lambda, so
         ambiguity increases monotonically.

Only ONE thing changes at the boundary — whether images are blended. The corruption
families are identical on both sides, which is what makes any alarm attributable to
ambiguity rather than to a new input family.

``novel_start_batch`` is reused verbatim as the boundary key so ``plot_mnist_c_panel``,
``paper_panel.py`` and the AUROC/FAR tables need no changes.

Interpreting accuracy in phase B
--------------------------------
A blend at lambda carries a dominant class (the first image), used as the label. Error
rises in phase B and that is **correct behaviour, not degradation**: the added error is
irreducible, since no amount of retraining recovers a label that the input genuinely does
not determine. That is precisely why an alarm here is a false alarm in the actionable
sense — retraining is not the remedy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Sequence

import numpy as np

from uncertainty_driven_drift.components.cifar_c import _load_cifar10_test_pool, _SHORT_NAME
from uncertainty_driven_drift.components.cifar_c_corruptions import (
    apply_corruption,
    available_corruptions,
)
from uncertainty_driven_drift.data.base import DatasetStream, StreamBatch, StreamSpec
from uncertainty_driven_drift.registry import register

_CIFAR_CLASSES = 10
_INPUT_SHAPE = (3, 32, 32)


@register("dataset", "cifar_known_ambiguous")
class CIFARKnownAmbiguousStream(DatasetStream):
    """CIFAR-10 stream: known corruptions, then the same corruptions on blended images.

    Parameters
    ----------
    known_corruptions :
        Corruption families the model was trained on. Used in **both** phases — they are
        deliberately not the variable under test.
    blend_lambdas :
        One phase per value, in order. Each is the weight of the dominant image, so 0.5 is
        maximally ambiguous and 1.0 is no blending. Defaults to a 0.7 / 0.6 / 0.5 ramp,
        mirroring S1's three novel corruptions.
    n_batches_per_phase :
        Batches per phase. Phase A runs ``len(known_corruptions)`` phases, phase B runs
        ``len(blend_lambdas)``, so the default 7 + 3 at 20 gives the same 200 batches and
        the same boundary at 140 as S1.
    severity, batch_size, seed, data_root :
        As in :class:`~uncertainty_driven_drift.components.cifar_c.CIFARKnownNovelStream`.
    """

    def __init__(
        self,
        known_corruptions: Sequence[str],
        blend_lambdas: Sequence[float] = (0.7, 0.6, 0.5),
        n_batches_per_phase: int = 20,
        severity: int = 1,
        batch_size: int = 64,
        seed: int = 0,
        data_root: str = "./artifacts/cifar10",
    ) -> None:
        if not known_corruptions:
            raise ValueError("known_corruptions must not be empty")
        if not blend_lambdas:
            raise ValueError("blend_lambdas must not be empty")
        bad_lam = [lam for lam in blend_lambdas if not 0.5 <= float(lam) < 1.0]
        if bad_lam:
            raise ValueError(
                f"blend_lambdas must lie in [0.5, 1.0); got {bad_lam}. Below 0.5 the "
                f"'dominant' image is no longer dominant and the label would be wrong."
            )
        avail = set(available_corruptions())
        unknown = [c for c in known_corruptions if c not in avail]
        if unknown:
            raise KeyError(f"Unknown corruptions {unknown!r}; have {sorted(avail)}")

        self.known_corruptions = list(known_corruptions)
        self.blend_lambdas = [float(lam) for lam in blend_lambdas]
        self.n_batches_per_phase = int(n_batches_per_phase)
        self.severity = int(severity)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.data_root = str(data_root)

        n_clean = len(self.known_corruptions)
        n_blend = len(self.blend_lambdas)
        total_batches = (n_clean + n_blend) * self.n_batches_per_phase
        ambiguous_start = n_clean * self.n_batches_per_phase

        drift_indices: List[int] = [
            i * self.n_batches_per_phase for i in range(1, n_clean + n_blend)
        ]

        self._pool_x, self._pool_y = _load_cifar10_test_pool(Path(self.data_root))

        channel_names = [_SHORT_NAME.get(c, c) for c in self.known_corruptions]
        channel_names += [f"blend {lam:g}" for lam in self.blend_lambdas]

        self.spec = StreamSpec(
            name="cifar_known_ambiguous",
            input_shape=_INPUT_SHAPE,
            n_classes=_CIFAR_CLASSES,
            n_batches=total_batches,
            batch_size=self.batch_size,
            drift_indices=tuple(drift_indices),
            has_true_posterior=False,
            extras={
                "severity": self.severity,
                "known_corruptions": self.known_corruptions,
                "blend_lambdas": self.blend_lambdas,
                # Reused verbatim so the existing panels/tables mark the boundary. Here it
                # is the onset of AMBIGUITY, not of a novel corruption family.
                "novel_start_batch": ambiguous_start,
                "shift_kind": "ambiguity",
                "channel_names": channel_names,
                "pool_size": int(self._pool_x.shape[0]),
            },
        )

    def __iter__(self) -> Iterator[StreamBatch]:
        sample_rng = np.random.default_rng(self.seed)
        corrupt_rng = np.random.default_rng(self.seed + 1)
        blend_rng = np.random.default_rng(self.seed + 2)

        ambiguous_start = self.spec.extras["novel_start_batch"]
        drift_set = set(self.spec.drift_indices)
        pool_n = self._pool_x.shape[0]
        n_clean = len(self.known_corruptions)

        phases = [(c, 1.0) for c in self.known_corruptions]
        # Phase B cycles the SAME known corruptions so the corruption family is not a
        # confound; only the blend weight changes.
        phases += [
            (self.known_corruptions[i % n_clean], lam)
            for i, lam in enumerate(self.blend_lambdas)
        ]

        t = 0
        for phase_i, (corruption, lam) in enumerate(phases):
            for _ in range(self.n_batches_per_phase):
                idx = sample_rng.integers(0, pool_n, size=self.batch_size)
                x = self._pool_x[idx]
                y = self._pool_y[idx]

                if lam < 1.0:
                    # Partner images must come from a DIFFERENT class, or the blend is not
                    # ambiguous at all and the scenario silently degrades to a no-op.
                    partner = sample_rng.integers(0, pool_n, size=self.batch_size)
                    same = self._pool_y[partner] == y
                    for _ in range(10):
                        if not same.any():
                            break
                        partner[same] = sample_rng.integers(0, pool_n, size=int(same.sum()))
                        same = self._pool_y[partner] == y
                    x = lam * x + (1.0 - lam) * self._pool_x[partner]
                    x = x.astype(np.float32)

                x = apply_corruption(corruption, x, self.severity, corrupt_rng)

                yield StreamBatch(
                    index=t,
                    x=x,
                    y=y,
                    concept_id=phase_i,
                    is_drift=(t in drift_set),
                    true_posterior=None,
                    extras={
                        "corruption": corruption,
                        "blend_lambda": lam,
                        "is_novel": t >= ambiguous_start,
                    },
                )
                t += 1
        del blend_rng
