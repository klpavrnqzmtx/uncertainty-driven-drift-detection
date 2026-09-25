"""Smoke tests for the experimentation infrastructure.

These tests deliberately avoid torch / river / laplace-torch so the core
runner can be validated on its own.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import uncertainty_driven_drift.components.builtin  # noqa: F401 — register stubs
from uncertainty_driven_drift.config import ExperimentConfig, load_config
from uncertainty_driven_drift.registry import available
from uncertainty_driven_drift.runner import run_experiment


def test_registry_has_builtin_components() -> None:
    assert "bernoulli_flip" in available("dataset")
    assert "prior_classifier" in available("model")
    assert "predictive_entropy" in available("uncertainty")
    assert "running_mean_error" in available("detector")


def test_config_loader_parses_smoke_yaml() -> None:
    cfg = load_config("configs/experiments/smoke.yaml")
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.dataset.name == "bernoulli_flip"
    assert cfg.detectors[0].name == "running_mean_error"


def test_runner_writes_steps_and_metrics(tmp_path: Path) -> None:
    cfg = load_config("configs/experiments/smoke.yaml")
    cfg.output_dir = str(tmp_path)
    out_dir = run_experiment(cfg)

    steps_path = out_dir / "steps.jsonl"
    metrics_path = out_dir / "metrics.json"
    config_path = out_dir / "config.resolved.json"

    assert steps_path.exists()
    assert metrics_path.exists()
    assert config_path.exists()

    lines = steps_path.read_text().splitlines()
    assert len(lines) == cfg.dataset.params["n_batches"]
    first = json.loads(lines[0])
    assert {"step", "concept_id", "is_drift", "accuracy", "mean_total",
            "mean_epistemic", "detectors"}.issubset(first)

    metrics = json.loads(metrics_path.read_text())
    assert metrics["experiment"] == "smoke"
    assert metrics["n_steps"] == cfg.dataset.params["n_batches"]
    assert 0.0 <= metrics["mean_accuracy"] <= 1.0


def test_runner_rejects_unknown_component(tmp_path: Path) -> None:
    cfg = load_config("configs/experiments/smoke.yaml")
    cfg.output_dir = str(tmp_path)
    cfg.dataset.name = "does_not_exist"
    with pytest.raises(KeyError):
        run_experiment(cfg)
