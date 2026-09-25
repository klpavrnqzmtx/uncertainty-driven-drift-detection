"""Typed experiment configuration loaded from YAML.

A config is a tree of ``ComponentSpec`` entries (``name`` + free-form
``params``). The runner resolves them via :mod:`uncertainty_driven_drift.registry`.
Schema is intentionally small; components validate their own params.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class ComponentSpec:
    """Reference to a registered component and its constructor kwargs."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any] | None) -> "ComponentSpec":
        if d is None:
            raise ValueError("Component spec is missing")
        if "name" not in d:
            raise ValueError(f"Component spec missing 'name': {d!r}")
        return cls(name=str(d["name"]), params=dict(d.get("params") or {}))


@dataclass
class ExperimentConfig:
    """Top-level experiment config.

    Attributes
    ----------
    experiment :
        Short identifier used in the output path, e.g. ``synthetic_tv``.
    seed :
        Global seed applied to numpy / torch / python random.
    output_dir :
        Base directory under which each run gets a timestamped subfolder.
    dataset, model, uncertainty :
        Single-component specs.
    detectors :
        Ordered list of detector specs; each runs in parallel on the same stream.
    extras :
        Free-form block for plotting / analysis flags consumed downstream.
    """

    experiment: str
    seed: int = 42
    output_dir: str = "./results/raw"
    dataset: ComponentSpec = field(default_factory=lambda: ComponentSpec("noop"))
    model: ComponentSpec = field(default_factory=lambda: ComponentSpec("noop"))
    uncertainty: ComponentSpec = field(default_factory=lambda: ComponentSpec("noop"))
    detectors: List[ComponentSpec] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        required = {"experiment", "dataset", "model", "uncertainty"}
        missing = required - d.keys()
        if missing:
            raise ValueError(f"Config missing required keys: {sorted(missing)}")

        detectors_raw = d.get("detectors") or []
        if not isinstance(detectors_raw, list):
            raise ValueError("'detectors' must be a list of component specs")

        return cls(
            experiment=str(d["experiment"]),
            seed=int(d.get("seed", 42)),
            output_dir=str(d.get("output_dir", "./results/raw")),
            dataset=ComponentSpec.from_dict(d["dataset"]),
            model=ComponentSpec.from_dict(d["model"]),
            uncertainty=ComponentSpec.from_dict(d["uncertainty"]),
            detectors=[ComponentSpec.from_dict(x) for x in detectors_raw],
            extras=dict(d.get("extras") or {}),
        )


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}")
    return ExperimentConfig.from_dict(raw)
