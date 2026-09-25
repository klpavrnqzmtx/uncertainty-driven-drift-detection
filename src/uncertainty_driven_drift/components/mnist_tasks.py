"""MNIST stream with changing binary task definitions.

The image distribution ``P(X)`` stays fixed (uniform samples from the
MNIST test pool) but the labelling rule ``y = task(digit)`` changes
across phases, so ``P(Y|X)`` undergoes genuine concept drift.

A schedule is a list of ``(task, n_batches)`` segments. Between adjacent
segments there is either:

* **abrupt** drift — labels switch on a single batch boundary, or
* **gradual** drift — a ``gradual_span``-wide window of per-sample
  mixing, with ``α`` ramping linearly from 0 to 1. Each sample in the
  transition is labelled by ``task_a`` with probability ``1-α`` and by
  ``task_b`` with probability ``α``.

Four built-in tasks match the Figure-2 benchmark:

* ``odd_even``   — ``y = d % 2``
* ``gt4``        — ``y = 1`` iff ``d > 4``
* ``prime``      — ``y = 1`` iff ``d ∈ {2, 3, 5, 7}``
* ``in_2_5``     — ``y = 1`` iff ``d ∈ {2, 3, 4, 5}``

Additional tasks can be registered via :func:`register_task`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from uncertainty_driven_drift.data.base import DatasetStream, StreamBatch, StreamSpec
from uncertainty_driven_drift.registry import register


TaskFn = Callable[[int], int]


_TASKS: Dict[str, TaskFn] = {
    "odd_even": lambda d: int(d % 2),
    "gt4": lambda d: int(d > 4),
    "prime": lambda d: int(d in {2, 3, 5, 7}),
    "in_2_5": lambda d: int(d in {2, 3, 4, 5}),
}


def register_task(name: str, fn: TaskFn) -> None:
    """Register an extra digit→{0,1} task by name."""
    if name in _TASKS:
        raise ValueError(f"Task {name!r} already registered")
    _TASKS[name] = fn


def available_tasks() -> List[str]:
    return sorted(_TASKS)


def task_label_vector(task_name: str) -> np.ndarray:
    """Return the length-10 lookup ``[task(0), task(1), ..., task(9)]``."""
    if task_name not in _TASKS:
        raise KeyError(f"Unknown task {task_name!r}; have {available_tasks()}")
    fn = _TASKS[task_name]
    return np.asarray([fn(d) for d in range(10)], dtype=np.int64)


@dataclass
class _Segment:
    """One materialized segment of the unrolled schedule."""

    kind: str            # "pure" | "transition"
    task_a: str
    task_b: str          # equal to task_a for pure segments
    start: int           # inclusive
    end: int             # exclusive
    gradual_span: int    # width of the transition (used for alpha calc)


def _materialize_schedule(
    schedule: Sequence[Dict[str, object]],
    transition: str,
    gradual_span: int,
) -> Tuple[List[_Segment], List[int]]:
    """Expand ``schedule`` into a flat list of batch-indexed segments.

    Returns the segment list and the list of drift indices (where a
    transition starts, for ``gradual``, or where the task switches, for
    ``abrupt``).
    """
    if transition not in ("abrupt", "gradual"):
        raise ValueError(f"transition must be 'abrupt' or 'gradual', got {transition!r}")

    if not schedule:
        raise ValueError("schedule must contain at least one phase")

    segments: List[_Segment] = []
    drift_indices: List[int] = []
    t = 0
    for i, phase in enumerate(schedule):
        if "task" not in phase or "n_batches" not in phase:
            raise ValueError(
                f"Schedule phase {i} must have 'task' and 'n_batches': {phase!r}"
            )
        task = str(phase["task"])
        if task not in _TASKS:
            raise KeyError(f"Unknown task {task!r}; have {available_tasks()}")
        n = int(phase["n_batches"])
        if n <= 0:
            raise ValueError(f"n_batches must be positive; got {n} in phase {i}")

        segments.append(_Segment(
            kind="pure", task_a=task, task_b=task,
            start=t, end=t + n, gradual_span=0,
        ))
        t += n

        is_last = i == len(schedule) - 1
        if is_last:
            continue
        next_task = str(schedule[i + 1]["task"])
        if next_task not in _TASKS:
            raise KeyError(
                f"Unknown task {next_task!r}; have {available_tasks()}"
            )
        drift_indices.append(t)
        if transition == "gradual":
            if gradual_span <= 0:
                raise ValueError("gradual_span must be positive for transition='gradual'")
            segments.append(_Segment(
                kind="transition", task_a=task, task_b=next_task,
                start=t, end=t + gradual_span, gradual_span=gradual_span,
            ))
            t += gradual_span
        # For abrupt: drift index is exactly `t` (start of next pure segment).

    return segments, drift_indices


def _find_segment(segments: Sequence[_Segment], t: int) -> _Segment:
    """Linear scan; schedules are small (≤ tens of segments)."""
    for seg in segments:
        if seg.start <= t < seg.end:
            return seg
    raise IndexError(f"No segment contains t={t}")


# ---------------------------------------------------------------------------
# MNIST data loading
# ---------------------------------------------------------------------------

def _load_mnist_test_pool(cache_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load MNIST test images + digit labels as numpy arrays.

    Downloads via ``torchvision`` on first call; subsequent calls read
    the cached raw files directly.
    """
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    try:
        from torchvision import datasets  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "mnist_tasks requires torchvision. Install the optional "
            "extras: pip install '.[torch]'"
        ) from exc

    ds = datasets.MNIST(
        root=str(cache_root), train=False, download=True, transform=None,
    )
    # ds.data: uint8 tensor (N, 28, 28); ds.targets: int64 tensor (N,).
    x = ds.data.numpy().astype(np.float32) / 255.0   # scale to [0, 1]
    x = x[:, None, :, :]                             # (N, 1, 28, 28)
    y = ds.targets.numpy().astype(np.int64)
    return x, y


# ---------------------------------------------------------------------------
# Stream
# ---------------------------------------------------------------------------

@register("dataset", "mnist_tasks")
class MNISTTaskStream(DatasetStream):
    """Uniform-pool MNIST stream with changing binary label rules.

    Parameters
    ----------
    schedule :
        List of ``{"task": str, "n_batches": int}`` dicts, in order.
    transition :
        ``"abrupt"`` (default) or ``"gradual"``. Applied between every
        consecutive pair of phases.
    gradual_span :
        Number of batches in each gradual transition window.
    batch_size :
        Samples drawn per batch from the MNIST pool.
    seed :
        Controls image sampling and per-sample task mixing.
    data_root :
        Root directory for the MNIST cache (downloaded once).
    """

    def __init__(
        self,
        schedule: Sequence[Dict[str, object]],
        transition: str = "abrupt",
        gradual_span: int = 10,
        batch_size: int = 128,
        seed: int = 0,
        data_root: str = "./artifacts/mnist",
    ) -> None:
        self.schedule = list(schedule)
        self.transition = str(transition)
        self.gradual_span = int(gradual_span)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.data_root = str(data_root)

        self._segments, drift_indices = _materialize_schedule(
            self.schedule, self.transition, self.gradual_span,
        )
        n_batches = self._segments[-1].end

        self._pool_x, self._pool_y = _load_mnist_test_pool(Path(self.data_root))
        self._task_vectors = {
            name: task_label_vector(name) for name in _TASKS
        }

        self.spec = StreamSpec(
            name="mnist_tasks",
            input_shape=(1, 28, 28),
            n_classes=2,
            n_batches=n_batches,
            batch_size=self.batch_size,
            drift_indices=tuple(drift_indices),
            has_true_posterior=False,
            extras={
                "transition": self.transition,
                "gradual_span": self.gradual_span,
                "schedule": [
                    {"task": str(p["task"]), "n_batches": int(p["n_batches"])}
                    for p in self.schedule
                ],
                "pool_size": int(self._pool_x.shape[0]),
                "available_tasks": available_tasks(),
                "segments": [
                    {
                        "kind": s.kind,
                        "task_a": s.task_a,
                        "task_b": s.task_b,
                        "start": s.start,
                        "end": s.end,
                    }
                    for s in self._segments
                ],
            },
        )

    def __iter__(self) -> Iterator[StreamBatch]:
        rng = np.random.default_rng(self.seed)
        drift_set = set(self.spec.drift_indices)
        pool_n = self._pool_x.shape[0]

        # concept_id = segment index (pure and transition segments both count).
        for t in range(self.spec.n_batches):
            # Sample a batch of images uniformly with replacement.
            idx = rng.integers(0, pool_n, size=self.batch_size)
            x = self._pool_x[idx]
            digits = self._pool_y[idx]

            seg = _find_segment(self._segments, t)
            if seg.kind == "pure":
                labels = self._task_vectors[seg.task_a][digits]
            else:
                alpha = (t - seg.start) / float(seg.gradual_span)
                alpha = float(np.clip(alpha, 0.0, 1.0))
                la = self._task_vectors[seg.task_a][digits]
                lb = self._task_vectors[seg.task_b][digits]
                use_b = rng.random(size=self.batch_size) < alpha
                labels = np.where(use_b, lb, la)

            concept_id = self._segments.index(seg)
            yield StreamBatch(
                index=t,
                x=x,
                y=labels.astype(np.int64),
                concept_id=concept_id,
                is_drift=(t in drift_set),
                true_posterior=None,
                extras={
                    "segment_kind": seg.kind,
                    "task_a": seg.task_a,
                    "task_b": seg.task_b,
                    "digits": digits.astype(np.int64),
                },
            )
