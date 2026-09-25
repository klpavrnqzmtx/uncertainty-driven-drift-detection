"""MNIST-C stream with per-edge abrupt/gradual transitions.

A single schedule concatenates corruption phases. Unlike ``mnist_tasks``
where ``transition`` is global, each phase (except the first) declares
how the stream *arrives* at that phase:

* ``transition: abrupt`` — the previous phase ends and this one starts
  on a batch boundary.
* ``transition: gradual`` (with ``gradual_span: int``) — a ramp window
  of ``gradual_span`` batches precedes this phase, during which each
  sample is corrupted either by the previous or the incoming rule with
  probability ``(1 - α, α)`` for ``α`` linearly growing from 0 to 1.

This mixes abrupt and gradual transitions inside one run, which is the
point of the MNIST-C benchmark here: one stream, one plot, one outputs
directory.

The label ``y`` is the *true* MNIST digit — corruption does not change
it — so downstream 10-way classifiers evaluate accuracy against the
clean-MNIST ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np

from uncertainty_driven_drift.components.mnist_c_corruptions import (
    apply_corruption,
    available_corruptions,
)
from uncertainty_driven_drift.data.base import DatasetStream, StreamBatch, StreamSpec
from uncertainty_driven_drift.registry import register


@dataclass
class _Segment:
    """One materialized segment of the unrolled schedule."""

    kind: str                 # "pure" | "transition"
    corruption_a: str
    corruption_b: str         # equal to corruption_a for pure segments
    start: int                # inclusive
    end: int                  # exclusive
    gradual_span: int         # width of the transition (used for alpha calc)


def _validate_phase(i: int, phase: Dict[str, object]) -> Tuple[str, int, str, int]:
    if "corruption" not in phase or "n_batches" not in phase:
        raise ValueError(
            f"Schedule phase {i} must have 'corruption' and 'n_batches': {phase!r}"
        )
    name = str(phase["corruption"])
    if name not in available_corruptions():
        raise KeyError(
            f"Unknown corruption {name!r}; have {available_corruptions()}"
        )
    n = int(phase["n_batches"])
    if n <= 0:
        raise ValueError(f"n_batches must be positive in phase {i}; got {n}")

    if i == 0:
        if "transition" in phase:
            raise ValueError(
                "First phase must not declare a 'transition' — it has no predecessor"
            )
        return name, n, "abrupt", 0

    transition = str(phase.get("transition", "abrupt"))
    if transition not in ("abrupt", "gradual"):
        raise ValueError(
            f"phase {i}: transition must be 'abrupt' or 'gradual', got {transition!r}"
        )
    span = int(phase.get("gradual_span", 0))
    if transition == "gradual":
        if span <= 0:
            raise ValueError(
                f"phase {i}: transition='gradual' requires positive 'gradual_span'"
            )
    else:
        if span not in (0,):
            # Not an error to declare it for abrupt, but it must be zero.
            raise ValueError(
                f"phase {i}: 'gradual_span' is only valid with transition='gradual'"
            )
    return name, n, transition, span


def _materialize_schedule(
    schedule: Sequence[Dict[str, object]],
) -> Tuple[List[_Segment], List[int], List[Dict[str, object]]]:
    """Expand ``schedule`` into flat segments and drift indices.

    Returns
    -------
    segments :
        Flat list of ``_Segment`` covering ``[0, n_batches)``.
    drift_indices :
        The batch index where each non-first phase *starts* (i.e. the
        first batch of the pure segment of that phase). For gradual
        transitions, this is *after* the ramp.
    normalized :
        The schedule with defaults filled in, for persistence.
    """
    if not schedule:
        raise ValueError("schedule must contain at least one phase")

    segments: List[_Segment] = []
    drift_indices: List[int] = []
    normalized: List[Dict[str, object]] = []
    t = 0
    prev_name: str | None = None
    for i, phase in enumerate(schedule):
        name, n, transition, span = _validate_phase(i, phase)
        norm: Dict[str, object] = {"corruption": name, "n_batches": n}
        if i > 0:
            norm["transition"] = transition
            if transition == "gradual":
                norm["gradual_span"] = span
                segments.append(_Segment(
                    kind="transition", corruption_a=prev_name or name,
                    corruption_b=name, start=t, end=t + span,
                    gradual_span=span,
                ))
                t += span
        segments.append(_Segment(
            kind="pure", corruption_a=name, corruption_b=name,
            start=t, end=t + n, gradual_span=0,
        ))
        if i > 0:
            drift_indices.append(t)
        t += n
        prev_name = name
        normalized.append(norm)

    return segments, drift_indices, normalized


def _find_segment(segments: Sequence[_Segment], t: int) -> _Segment:
    for seg in segments:
        if seg.start <= t < seg.end:
            return seg
    raise IndexError(f"No segment contains t={t}")


def _load_mnist_test_pool(cache_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load the plain MNIST test set as ``(N, 1, 28, 28)`` float + labels."""
    cache_root.mkdir(parents=True, exist_ok=True)
    try:
        from torchvision import datasets  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "mnist_c requires torchvision. Install the optional extras: "
            "pip install '.[torch]'"
        ) from exc
    ds = datasets.MNIST(
        root=str(cache_root), train=False, download=True, transform=None,
    )
    x = ds.data.numpy().astype(np.float32) / 255.0
    x = x[:, None, :, :]
    y = ds.targets.numpy().astype(np.int64)
    return x, y


@register("dataset", "mnist_c")
class MNISTCStream(DatasetStream):
    """Concatenated MNIST-C stream with per-edge abrupt/gradual transitions.

    Parameters
    ----------
    schedule :
        Ordered list of phases. The first phase must not declare a
        transition. Subsequent phases accept ``transition`` (``abrupt``
        by default) and ``gradual_span`` (required if transition is
        gradual). Each phase declares the ``corruption`` to apply and
        ``n_batches`` for its pure section.
    severity :
        MNIST-C severity on the 1..5 scale, shared by every corruption
        in the stream.
    batch_size :
        Samples per batch, drawn uniformly from the MNIST test pool.
    seed :
        Controls image sampling, per-sample mixing in gradual segments,
        and per-corruption stochasticity (noise, angles, etc.).
    data_root :
        Cache location for the downloaded MNIST test set.
    """

    def __init__(
        self,
        schedule: Sequence[Dict[str, object]],
        severity: int = 3,
        batch_size: int = 128,
        seed: int = 0,
        data_root: str = "./artifacts/mnist",
    ) -> None:
        self.schedule = list(schedule)
        self.severity = int(severity)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.data_root = str(data_root)

        segments, drift_indices, normalized = _materialize_schedule(self.schedule)
        self._segments = segments
        self._normalized_schedule = normalized

        n_batches = self._segments[-1].end
        self._pool_x, self._pool_y = self._load_pool()

        self.spec = StreamSpec(
            name="mnist_c",
            input_shape=(1, 28, 28),
            n_classes=10,
            n_batches=n_batches,
            batch_size=self.batch_size,
            drift_indices=tuple(drift_indices),
            has_true_posterior=False,
            extras={
                "severity": self.severity,
                "schedule": normalized,
                "pool_size": int(self._pool_x.shape[0]),
                "available_corruptions": available_corruptions(),
                "segments": [
                    {
                        "kind": s.kind,
                        "corruption_a": s.corruption_a,
                        "corruption_b": s.corruption_b,
                        "start": s.start,
                        "end": s.end,
                        "gradual_span": s.gradual_span,
                    }
                    for s in self._segments
                ],
            },
        )

    def _load_pool(self) -> Tuple[np.ndarray, np.ndarray]:
        return _load_mnist_test_pool(Path(self.data_root))

    def __iter__(self) -> Iterator[StreamBatch]:
        # Two independent RNG streams: one for image sampling / mixing
        # decisions (keeps indices deterministic across corruption
        # implementations), one for corruption parameters (so tweaking
        # corruption code does not change which images appear).
        sample_rng = np.random.default_rng(self.seed)
        corrupt_rng = np.random.default_rng(self.seed + 1)
        drift_set = set(self.spec.drift_indices)
        pool_n = self._pool_x.shape[0]

        for t in range(self.spec.n_batches):
            idx = sample_rng.integers(0, pool_n, size=self.batch_size)
            x_clean = self._pool_x[idx]
            y = self._pool_y[idx]

            seg = _find_segment(self._segments, t)
            if seg.kind == "pure":
                x = apply_corruption(seg.corruption_a, x_clean, self.severity, corrupt_rng)
                concept_id = self._segments.index(seg)
                seg_extras = {"segment_kind": "pure", "corruption": seg.corruption_a}
            else:
                alpha = (t - seg.start) / float(seg.gradual_span)
                alpha = float(np.clip(alpha, 0.0, 1.0))
                use_b = sample_rng.random(size=self.batch_size) < alpha
                x = np.empty_like(x_clean, dtype=np.float32)
                if (~use_b).any():
                    x[~use_b] = apply_corruption(
                        seg.corruption_a, x_clean[~use_b], self.severity, corrupt_rng,
                    )
                if use_b.any():
                    x[use_b] = apply_corruption(
                        seg.corruption_b, x_clean[use_b], self.severity, corrupt_rng,
                    )
                concept_id = self._segments.index(seg)
                seg_extras = {
                    "segment_kind": "transition",
                    "corruption_a": seg.corruption_a,
                    "corruption_b": seg.corruption_b,
                    "alpha": alpha,
                }

            yield StreamBatch(
                index=t,
                x=x,
                y=y.astype(np.int64),
                concept_id=concept_id,
                is_drift=(t in drift_set),
                true_posterior=None,
                extras=seg_extras,
            )


@register("dataset", "mnist_known_novel")
class MNISTKnownNovelStream(MNISTCStream):
    """MNIST-C stream designed to expose epistemic uncertainty on novel corruptions.

    The model is pretrained on clean MNIST augmented with ``known_corruptions``.
    This stream first presents batches drawn from the known corruptions (in
    round-robin order), then switches to ``novel_corruptions`` the model has
    never seen.  All transitions inside each region are abrupt.

    Expected behaviour
    ------------------
    * Uncertainty-based detectors: quiet during the known phase (model is
      calibrated), alarm at/shortly after the known → novel boundary.
    * Input-based detectors: fire at *every* corruption boundary because the
      raw pixel distribution changes regardless of whether the model knows the
      corruption.
    * Supervised detectors: alarm when accuracy drops, which should coincide
      with the known → novel boundary.

    The index of the first novel batch is stored in
    ``spec.extras["novel_start_batch"]`` so the plotting code can annotate it
    with a special marker.

    Parameters
    ----------
    known_corruptions :
        Corruption types the model was trained on.  Streamed first in the order
        given, ``n_batches_per_phase`` batches each.
    novel_corruptions :
        Corruption types the model has *not* seen during training.  Streamed
        after the known phase, same cadence.
    n_batches_per_phase :
        Pure batches per corruption.
    severity, batch_size, seed, data_root :
        Forwarded to :class:`MNISTCStream`.
    """

    def __init__(
        self,
        known_corruptions: Sequence[str],
        novel_corruptions: Sequence[str],
        n_batches_per_phase: int = 20,
        severity: int = 3,
        batch_size: int = 128,
        seed: int = 0,
        data_root: str = "./artifacts/mnist",
    ) -> None:
        if not known_corruptions:
            raise ValueError("known_corruptions must not be empty")
        if not novel_corruptions:
            raise ValueError("novel_corruptions must not be empty")

        schedule: List[Dict[str, object]] = []
        for i, c in enumerate(known_corruptions):
            phase: Dict[str, object] = {"corruption": c, "n_batches": n_batches_per_phase}
            if i > 0:
                phase["transition"] = "abrupt"
            schedule.append(phase)
        for c in novel_corruptions:
            schedule.append({"corruption": c, "n_batches": n_batches_per_phase, "transition": "abrupt"})

        super().__init__(
            schedule=schedule,
            severity=severity,
            batch_size=batch_size,
            seed=seed,
            data_root=data_root,
        )

        novel_start = n_batches_per_phase * len(known_corruptions)
        self.spec.extras["novel_start_batch"] = novel_start
        self.spec.extras["known_corruptions"] = list(known_corruptions)
        self.spec.extras["novel_corruptions"] = list(novel_corruptions)
        self.spec.name = "mnist_known_novel"
