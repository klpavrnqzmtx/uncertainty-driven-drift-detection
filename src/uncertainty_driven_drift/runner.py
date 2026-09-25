"""Experiment runner.

Orchestrates the prequential loop:

1. Build components from the config via the registry.
2. Iterate the stream; for each batch:
   a. predict with the model
   b. score uncertainty
   c. update every detector
   d. log a step record
   e. optionally let the model observe the labels
3. Collect summary metrics and persist them.

The runner intentionally knows nothing about concrete datasets, models,
detectors, or uncertainty families. It only talks to the APIs declared
in :mod:`uncertainty_driven_drift.data`, ``.models``, ``.uncertainty``,
``.detectors``.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from uncertainty_driven_drift.config import ComponentSpec, ExperimentConfig
from uncertainty_driven_drift.data.base import DatasetStream
from uncertainty_driven_drift.detectors.base import DriftDetector
from uncertainty_driven_drift.models.base import Classifier
from uncertainty_driven_drift.registry import build
from uncertainty_driven_drift.results.writer import ResultWriter, StepRecord
from uncertainty_driven_drift.utils.environment import capture as capture_environment
from uncertainty_driven_drift.uncertainty.base import UncertaintyEstimator
from uncertainty_driven_drift.utils.io import ensure_dir
from uncertainty_driven_drift.utils.seed import set_global_seed


def _warn_on_oversized_warm_start(cfg: ExperimentConfig, spec) -> None:
    """Warn when a per-sample detector's warm-up covers much of the known phase.

    DDM/EDDM count warm_start in SAMPLES, but the stream is scored in BATCHES, so
    a value that looks modest hides a long blind spot: warm_start=8000 at
    batch_size=64 is 125 batches, which silently disabled the supervised baseline
    for most of a 140-batch known phase and made its false-alarm rate look
    perfect. At batch_size=32 it exceeded the entire stream, so the detector could
    never fire at all. Cheap to check, expensive to discover in a figure.
    """
    bs = int(getattr(spec, "batch_size", 0) or 0)
    if bs <= 0:
        return
    extras = getattr(spec, "extras", None) or {}
    ref = int(extras.get("novel_start_batch") or getattr(spec, "n_batches", 0) or 0)
    if ref <= 0:
        return
    for d in cfg.detectors:
        ws = (d.params or {}).get("warm_start")
        if not ws:
            continue
        blind = float(ws) / bs
        if blind > 0.25 * ref:
            name = (d.params or {}).get("name", d.name)
            print(
                f"[warn] detector {name!r}: warm_start={ws} samples / batch_size={bs} "
                f"= {blind:.0f} batches blind, out of {ref} before the drift boundary. "
                f"It cannot react for {100*blind/ref:.0f}% of that phase, so its "
                f"false-alarm rate is understated."
            )


def run_experiment(cfg: ExperimentConfig) -> Path:
    """Run one experiment and return its output directory."""
    set_global_seed(cfg.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ensure_dir(Path(cfg.output_dir) / cfg.experiment / timestamp)

    dataset: DatasetStream = build("dataset", cfg.dataset.name, **cfg.dataset.params)
    spec = dataset.spec

    model: Classifier = build("model", cfg.model.name, **cfg.model.params)
    model.setup(spec)

    uncertainty: UncertaintyEstimator = build(
        "uncertainty", cfg.uncertainty.name, **cfg.uncertainty.params
    )
    uncertainty.setup(model)

    detectors: List[DriftDetector] = [
        build("detector", d.name, **d.params) for d in cfg.detectors
    ]
    _warn_on_oversized_warm_start(cfg, spec)

    with ResultWriter(out_dir) as writer:
        writer.write_config(_resolved_config_dict(cfg))
        accs: List[float] = []

        for batch in dataset:
            prediction = model.predict(batch)
            scores = uncertainty.score(batch, prediction)

            preds = prediction.probs.argmax(axis=1)
            acc = float((preds == batch.y).mean())
            accs.append(acc)
            mean_max_softmax = float(prediction.probs.max(axis=1).mean())

            # BER — only populated when the stream provides bit-level ground
            # truth (extras['bits']) and the model provides decoded bits
            # (prediction.extras['pred_bits']).  Generic for any modulation.
            ber: float | None = None
            true_bits = batch.extras.get("bits")
            pred_bits = (prediction.extras or {}).get("pred_bits")
            if true_bits is not None and pred_bits is not None:
                ber = float(
                    (np.asarray(pred_bits) != np.asarray(true_bits)).mean()
                )

            mean_tv: float | None = None
            bayes_acc: float | None = None
            if batch.true_posterior is not None:
                p_star = np.asarray(batch.true_posterior, dtype=np.float64)
                p_hat = np.asarray(prediction.probs, dtype=np.float64)
                mean_tv = float(0.5 * np.abs(p_star - p_hat).sum(axis=1).mean())
                bayes_acc = float((p_star.argmax(axis=1) == batch.y).mean())

            detector_events: Dict[str, Dict[str, Any]] = {}
            for det in detectors:
                event = det.update(batch, prediction, scores)
                detector_events[det.name] = {
                    "alarm": bool(event.alarm),
                    "statistic": float(event.statistic),
                    **event.diagnostics,
                }

            writer.log_step(
                StepRecord(
                    step=batch.index,
                    concept_id=batch.concept_id,
                    is_drift=batch.is_drift,
                    accuracy=acc,
                    mean_total=float(np.mean(scores.total)),
                    mean_epistemic=float(np.mean(scores.epistemic)),
                    mean_aleatoric=(
                        float(np.mean(scores.aleatoric))
                        if scores.aleatoric is not None
                        else None
                    ),
                    mean_tv=mean_tv,
                    mean_max_softmax=mean_max_softmax,
                    bayes_accuracy=bayes_acc,
                    ber=ber,
                    detectors=detector_events,
                )
            )

            model.observe(batch, prediction)

        metrics = {
            "experiment": cfg.experiment,
            "seed": cfg.seed,
            "n_steps": len(accs),
            "mean_accuracy": float(np.mean(accs)) if accs else float("nan"),
            "stream": asdict(spec),
            "detectors": [
                _detector_summary(spec_d, inst)
                for spec_d, inst in zip(cfg.detectors, detectors)
            ],
            # The config alone does not pin the numbers: river owns the detector
            # implementations, so a minor-version bump can move alarm counts with no
            # change here. Record what actually decided the output.
            "environment": capture_environment(),
        }
        writer.write_metrics(metrics)

    return out_dir


def _detector_summary(
    spec: ComponentSpec, instance: DriftDetector
) -> Dict[str, Any]:
    """Collect the user-facing hyperparameters for a detector instance.

    We start from the config params (what the user wrote in YAML) and
    enrich with a handful of *resolved* attributes that live on the
    adapter or on the wrapped river object. This way the metrics file
    reflects the actual running configuration, including defaults we
    picked up from the library.
    """
    params: Dict[str, Any] = dict(spec.params)
    # Always surface the signal channel and resolved display name.
    for attr in ("signal", "name"):
        val = getattr(instance, attr, None)
        if val is not None:
            params.setdefault(attr, val)
    # For river-backed adapters, expose the resolved river knobs even
    # if the user relied on library defaults.
    wrapped = getattr(instance, "_detector", None)
    if wrapped is not None:
        for attr in (
            "min_instances", "delta", "threshold", "alpha",
            "mode", "window_size", "stat_size",
        ):
            val = getattr(wrapped, attr, None)
            if val is not None:
                params.setdefault(attr, val)
    # Our home-grown χ² detector stores params directly on the instance.
    for attr in ("stat_size", "n_bins", "alpha", "min_expected"):
        val = getattr(instance, attr, None)
        if val is not None:
            params.setdefault(attr, val)
    return {
        "type": spec.name,
        "name": params.get("name", instance.name),
        "params": {k: _jsonable(v) for k, v in params.items()},
    }


def _jsonable(v: Any) -> Any:
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    try:
        import numpy as _np

        if isinstance(v, _np.generic):
            return v.item()
    except Exception:
        pass
    return str(v)


def _resolved_config_dict(cfg: ExperimentConfig) -> Dict[str, Any]:
    def _spec(s: ComponentSpec) -> Dict[str, Any]:
        return {"name": s.name, "params": s.params}

    return {
        "experiment": cfg.experiment,
        "seed": cfg.seed,
        "output_dir": cfg.output_dir,
        "dataset": _spec(cfg.dataset),
        "model": _spec(cfg.model),
        "uncertainty": _spec(cfg.uncertainty),
        "detectors": [_spec(d) for d in cfg.detectors],
        "extras": cfg.extras,
    }
