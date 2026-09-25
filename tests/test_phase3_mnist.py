"""Phase 3 tests: MNIST task-drift schedule, runner integration, plot.

Pre-trained LeNet weights are cached across tests (and runs) under
``artifacts/models/lenet_mnist.pt`` so the expensive CNN fit only
happens once.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import uncertainty_driven_drift.components.phase3  # noqa: F401 — register components
from uncertainty_driven_drift.components.mnist_tasks import (
    _materialize_schedule,
    available_tasks,
    task_label_vector,
)
from uncertainty_driven_drift.config import load_config
from uncertainty_driven_drift.runner import run_experiment


# ---------------------------------------------------------------------------
# Task functions
# ---------------------------------------------------------------------------

def test_task_label_vector_shapes_and_values() -> None:
    for name in ("odd_even", "gt4", "prime", "in_2_5"):
        v = task_label_vector(name)
        assert v.shape == (10,)
        assert set(np.unique(v).tolist()).issubset({0, 1})

    np.testing.assert_array_equal(
        task_label_vector("odd_even"), [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    )
    np.testing.assert_array_equal(
        task_label_vector("gt4"), [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    )
    np.testing.assert_array_equal(
        task_label_vector("prime"), [0, 0, 1, 1, 0, 1, 0, 1, 0, 0]
    )
    np.testing.assert_array_equal(
        task_label_vector("in_2_5"), [0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    )


def test_available_tasks_contains_defaults() -> None:
    names = set(available_tasks())
    assert {"odd_even", "gt4", "prime", "in_2_5"}.issubset(names)


# ---------------------------------------------------------------------------
# Schedule materialization
# ---------------------------------------------------------------------------

def test_schedule_abrupt_and_gradual_layouts() -> None:
    sched = [
        {"task": "odd_even", "n_batches": 5},
        {"task": "gt4",      "n_batches": 5},
        {"task": "prime",    "n_batches": 5},
    ]

    segs, drifts = _materialize_schedule(sched, "abrupt", gradual_span=0)
    assert [s.kind for s in segs] == ["pure", "pure", "pure"]
    assert [(s.start, s.end) for s in segs] == [(0, 5), (5, 10), (10, 15)]
    assert drifts == [5, 10]

    segs, drifts = _materialize_schedule(sched, "gradual", gradual_span=3)
    kinds = [s.kind for s in segs]
    assert kinds == ["pure", "transition", "pure", "transition", "pure"]
    assert [(s.start, s.end) for s in segs] == [
        (0, 5), (5, 8), (8, 13), (13, 16), (16, 21),
    ]
    assert drifts == [5, 13]


def test_schedule_rejects_unknown_task_and_bad_transition() -> None:
    with pytest.raises(KeyError):
        _materialize_schedule(
            [{"task": "bogus", "n_batches": 1}], "abrupt", 0,
        )
    with pytest.raises(ValueError):
        _materialize_schedule(
            [{"task": "odd_even", "n_batches": 1}], "weird", 0,
        )
    with pytest.raises(ValueError):
        _materialize_schedule(
            [{"task": "odd_even", "n_batches": 1},
             {"task": "gt4", "n_batches": 1}],
            "gradual", gradual_span=0,
        )


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("config", ["abrupt.yaml", "gradual.yaml"])
def test_mnist_tasks_run_end_to_end(tmp_path: Path, config: str) -> None:
    """Tiny run of each YAML config — asserts the full pipeline is wired."""
    cfg = load_config(f"configs/experiments/mnist_tasks/{config}")
    cfg.output_dir = str(tmp_path)
    # Shrink everything so the test stays fast.
    cfg.dataset.params["schedule"] = [
        {"task": "odd_even", "n_batches": 4},
        {"task": "gt4",      "n_batches": 4},
    ]
    cfg.dataset.params["batch_size"] = 32
    if "gradual_span" in cfg.dataset.params:
        cfg.dataset.params["gradual_span"] = 2
    cfg.model.params["n_init_fit"] = 1000
    cfg.model.params["pretrain_epochs"] = 1
    cfg.uncertainty.params["n_samples"] = 10

    out_dir = run_experiment(cfg)

    steps_path = out_dir / "steps.jsonl"
    metrics_path = out_dir / "metrics.json"
    assert steps_path.exists() and metrics_path.exists()

    steps = [json.loads(line) for line in steps_path.read_text().splitlines()]
    expected_n = sum(p["n_batches"] for p in cfg.dataset.params["schedule"])
    if cfg.dataset.params.get("transition") == "gradual":
        expected_n += (len(cfg.dataset.params["schedule"]) - 1) \
            * int(cfg.dataset.params["gradual_span"])
    assert len(steps) == expected_n

    for r in steps:
        assert 0.0 <= r["accuracy"] <= 1.0
        assert r["mean_epistemic"] >= 0.0
        assert 0.0 <= r["mean_max_softmax"] <= 1.0 + 1e-12
        assert set(r["detectors"]) == {
            "page_hinkley", "adwin", "kswin", "ddm", "eddm",
        }

    metrics = json.loads(metrics_path.read_text())
    assert metrics["stream"]["n_classes"] == 2
    assert metrics["stream"]["extras"]["transition"] in {"abrupt", "gradual"}
    assert len(metrics["stream"]["drift_indices"]) == 1  # 2-phase schedule


def test_mnist_tasks_plot_emits_png(tmp_path: Path) -> None:
    cfg = load_config("configs/experiments/mnist_tasks/abrupt.yaml")
    cfg.output_dir = str(tmp_path)
    cfg.dataset.params["schedule"] = [
        {"task": "odd_even", "n_batches": 3},
        {"task": "gt4",      "n_batches": 3},
    ]
    cfg.dataset.params["batch_size"] = 32
    cfg.model.params["n_init_fit"] = 500
    cfg.model.params["pretrain_epochs"] = 1
    cfg.uncertainty.params["n_samples"] = 5

    out_dir = run_experiment(cfg)

    from uncertainty_driven_drift.analysis.plots import plot_mnist_tasks_panel

    fig_path = plot_mnist_tasks_panel(out_dir, timestamp=False)
    assert fig_path.exists()
    assert fig_path.stat().st_size > 1000
