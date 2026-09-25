"""Tests for the MNIST-C pivot: corruptions, stream, model, plot.

Pretrained MC-Dropout LeNet weights are cached under
``artifacts/models/lenet_mc_mnist.pt`` so the expensive CNN fit only
happens once across tests and runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import uncertainty_driven_drift.components.phase3  # noqa: F401 — register components
from uncertainty_driven_drift.components.mnist_c import _materialize_schedule
from uncertainty_driven_drift.components.mnist_c_corruptions import (
    apply_corruption,
    available_corruptions,
)
from uncertainty_driven_drift.config import load_config
from uncertainty_driven_drift.runner import run_experiment


# ---------------------------------------------------------------------------
# Corruption toolkit
# ---------------------------------------------------------------------------

def _make_batch(b: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(size=(b, 1, 28, 28)).astype(np.float32)


@pytest.mark.parametrize("name", [
    "identity", "gaussian_noise", "impulse_noise", "motion_blur",
    "rotate", "translate", "brightness", "fog",
])
def test_corruption_preserves_shape_and_range(name: str) -> None:
    x = _make_batch(b=3)
    rng = np.random.default_rng(1)
    out = apply_corruption(name, x, severity=3, rng=rng)

    assert out.shape == x.shape
    assert out.dtype == np.float32
    assert float(out.min()) >= 0.0 - 1e-6
    assert float(out.max()) <= 1.0 + 1e-6


def test_identity_is_numerically_a_noop() -> None:
    x = _make_batch()
    out = apply_corruption("identity", x, severity=3, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(out, x.astype(np.float32, copy=False))


def test_severity_controls_magnitude_for_noise() -> None:
    """Higher severity should produce more deviation from clean input."""
    x = _make_batch(b=16)
    diffs = []
    for sev in (1, 3, 5):
        out = apply_corruption(
            "gaussian_noise", x, severity=sev, rng=np.random.default_rng(0),
        )
        diffs.append(float(np.abs(out - x).mean()))
    assert diffs[0] < diffs[1] < diffs[2]


def test_unknown_corruption_raises() -> None:
    with pytest.raises(KeyError):
        apply_corruption("bogus", _make_batch(), 3, np.random.default_rng(0))


def test_available_corruptions_is_nonempty() -> None:
    names = available_corruptions()
    assert "identity" in names
    assert "gaussian_noise" in names


# ---------------------------------------------------------------------------
# Schedule materialisation
# ---------------------------------------------------------------------------

def test_schedule_abrupt_only_layout() -> None:
    sched = [
        {"corruption": "identity",       "n_batches": 4},
        {"corruption": "gaussian_noise", "n_batches": 4, "transition": "abrupt"},
        {"corruption": "rotate",         "n_batches": 4, "transition": "abrupt"},
    ]
    segs, drifts, norm = _materialize_schedule(sched)
    assert [s.kind for s in segs] == ["pure", "pure", "pure"]
    assert [(s.start, s.end) for s in segs] == [(0, 4), (4, 8), (8, 12)]
    assert drifts == [4, 8]
    assert norm[0] == {"corruption": "identity", "n_batches": 4}
    assert norm[1]["transition"] == "abrupt"


def test_schedule_mixed_abrupt_gradual_layout() -> None:
    sched = [
        {"corruption": "identity",       "n_batches": 5},
        {"corruption": "gaussian_noise", "n_batches": 5, "transition": "abrupt"},
        {"corruption": "motion_blur",    "n_batches": 5,
         "transition": "gradual", "gradual_span": 3},
        {"corruption": "rotate",         "n_batches": 5, "transition": "abrupt"},
    ]
    segs, drifts, _ = _materialize_schedule(sched)
    # pure (0..5), pure (5..10), transition (10..13), pure (13..18),
    # pure (18..23).
    assert [s.kind for s in segs] == [
        "pure", "pure", "transition", "pure", "pure",
    ]
    assert [(s.start, s.end) for s in segs] == [
        (0, 5), (5, 10), (10, 13), (13, 18), (18, 23),
    ]
    # Drift indices mark where the NEW pure segment begins.
    assert drifts == [5, 13, 18]


def test_schedule_first_phase_must_not_declare_transition() -> None:
    with pytest.raises(ValueError):
        _materialize_schedule([
            {"corruption": "identity", "n_batches": 1, "transition": "abrupt"},
        ])


def test_schedule_gradual_requires_positive_span() -> None:
    with pytest.raises(ValueError):
        _materialize_schedule([
            {"corruption": "identity", "n_batches": 1},
            {"corruption": "rotate", "n_batches": 1, "transition": "gradual"},
        ])


def test_schedule_unknown_corruption_raises() -> None:
    with pytest.raises(KeyError):
        _materialize_schedule([{"corruption": "bogus", "n_batches": 1}])


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

def test_mnist_c_run_end_to_end(tmp_path: Path) -> None:
    """Tiny MNIST-C run exercises stream + MC-dropout + detectors + schema."""
    cfg = load_config("configs/experiments/mnist_c/default.yaml")
    cfg.output_dir = str(tmp_path)
    cfg.dataset.params["schedule"] = [
        {"corruption": "identity",       "n_batches": 3},
        {"corruption": "gaussian_noise", "n_batches": 3, "transition": "abrupt"},
        {"corruption": "rotate",         "n_batches": 3,
         "transition": "gradual", "gradual_span": 2},
    ]
    cfg.dataset.params["batch_size"] = 16
    cfg.model.params["pretrain_epochs"] = 1
    cfg.model.params["n_samples"] = 4

    out_dir = run_experiment(cfg)

    steps_path = out_dir / "steps.jsonl"
    metrics_path = out_dir / "metrics.json"
    assert steps_path.exists() and metrics_path.exists()

    steps = [json.loads(line) for line in steps_path.read_text().splitlines()]
    # 3 + 3 + 2 (gradual) + 3 = 11 batches.
    assert len(steps) == 11

    for r in steps:
        assert 0.0 <= r["accuracy"] <= 1.0
        assert r["mean_epistemic"] >= 0.0
        assert 0.0 <= r["mean_max_softmax"] <= 1.0 + 1e-12
        # Paired unsupervised detectors (epistemic vs input) for PH / KSWIN /
        # ADWIN, plus supervised DDM/EDDM. No softmax channel.
        names = set(r["detectors"])
        expected = {
            "ph_epistemic", "ph_input",
            "kswin_epistemic", "kswin_input",
            "adwin_epistemic", "adwin_input",
            "ddm", "eddm",
        }
        assert expected.issubset(names), f"missing: {expected - names}"
        assert not any("softmax" in n for n in names)
        # mean_input is now L2(per-sample mean, std); values are ≥ 0.
        ph_input = r["detectors"]["ph_input"]
        assert float(ph_input["statistic"]) >= 0.0

    metrics = json.loads(metrics_path.read_text())
    assert metrics["stream"]["n_classes"] == 10
    assert metrics["stream"]["extras"]["severity"] == cfg.dataset.params["severity"]
    # One gradual edge + two abrupt edges → 2 drift_indices.
    assert len(metrics["stream"]["drift_indices"]) == 2


def test_mnist_c_plot_emits_png(tmp_path: Path) -> None:
    cfg = load_config("configs/experiments/mnist_c/default.yaml")
    cfg.output_dir = str(tmp_path)
    cfg.dataset.params["schedule"] = [
        {"corruption": "identity",       "n_batches": 3},
        {"corruption": "gaussian_noise", "n_batches": 3, "transition": "abrupt"},
        {"corruption": "rotate",         "n_batches": 3,
         "transition": "gradual", "gradual_span": 2},
    ]
    cfg.dataset.params["batch_size"] = 16
    cfg.model.params["pretrain_epochs"] = 1
    cfg.model.params["n_samples"] = 4

    out_dir = run_experiment(cfg)

    from uncertainty_driven_drift.analysis.plots import plot_mnist_c_panel

    fig_path = plot_mnist_c_panel(out_dir, timestamp=False)
    assert fig_path.exists()
    assert fig_path.stat().st_size > 1000
