"""Streaming result serialization.

Per-step records are appended to ``steps.jsonl`` as they are produced so a
crash never loses more than the current batch. Aggregate run metadata and
summary metrics land in ``run.json`` / ``metrics.json`` at the end.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np

from uncertainty_driven_drift.utils.io import ensure_dir, write_json


@dataclass
class StepRecord:
    """One prequential step, written as a single JSONL line."""

    step: int
    concept_id: int
    is_drift: bool
    accuracy: float
    mean_total: float
    mean_epistemic: float
    mean_aleatoric: float | None
    mean_tv: float | None = None
    mean_max_softmax: float | None = None
    bayes_accuracy: float | None = None
    ber: float | None = None
    detectors: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class ResultWriter:
    """Stream step records to disk and collect summary metrics.

    The writer is also in charge of persisting the resolved config so a
    run can always be re-created from its output directory alone.
    """

    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = ensure_dir(out_dir)
        self._steps_path = self.out_dir / "steps.jsonl"
        self._steps_fh = self._steps_path.open("w")
        self._n_steps = 0

    def write_config(self, resolved_config: Dict[str, Any]) -> None:
        write_json(self.out_dir / "config.resolved.json", resolved_config)

    def log_step(self, record: StepRecord) -> None:
        payload = asdict(record)
        self._steps_fh.write(json.dumps(payload, default=_json_default) + "\n")
        self._steps_fh.flush()
        self._n_steps += 1

    def write_metrics(self, metrics: Dict[str, Any]) -> None:
        write_json(self.out_dir / "metrics.json", metrics)

    def close(self) -> None:
        if not self._steps_fh.closed:
            self._steps_fh.close()

    def __enter__(self) -> "ResultWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Cannot serialize {type(obj).__name__}")
