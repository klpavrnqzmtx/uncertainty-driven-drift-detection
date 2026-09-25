"""Figure generation for completed runs.

Each function reads from a run directory (``steps.jsonl`` + ``metrics.json``)
and writes a PNG under ``<run_dir>/figures/``. Plotting is kept separate
from the runner so the same data can be re-plotted without re-running.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


_SCENARIO_ORDER = ("virtual", "real", "gradual")
_SCENARIO_LABELS = {
    "virtual": "virtual drift",
    "real": "concept drift — abrupt",
    "gradual": "concept drift — gradual",
    "abrupt": "abrupt drift",
    "mixed": "mixed drift",
}


_SMOOTHING_CHOICES = ("none", "rolling", "cumulative")


def plot_scenarios_grid(
    run_dirs: Sequence[str | Path],
    out_path: str | Path,
    start: int = 20,
    end: int | None = 100,
    scenarios: Sequence[str] = _SCENARIO_ORDER,
    smoothing: str = "rolling",
    window: int = 10,
    show_raw: bool = False,  # noqa: ARG001 — kept for API compat, raw overlay removed
    timestamp: bool = True,
    title: str | None = None,
) -> Path:
    """Combined 1×N figure: square panels, zoomed on the drift window.

    Each panel shares the same layout:

    * Left y-axis — TV (solid blue), mean total (dashed grey), mean
      aleatoric (dotted purple). Shared across all panels; tick labels
      shown only on the leftmost panel.
    * Right y-axis — mean epistemic MI (solid green). Shared across
      panels; tick labels shown only on the rightmost panel.
    * Vertical red dashed lines at scheduled drift events.

    The x-axis is clipped to ``[start, end]`` so the drift dynamics fill
    the plot. Runs are matched to panels by
    ``metrics.stream.extras.drift_mode``.

    Parameters
    ----------
    smoothing :
        Temporal statistic applied to every mean series before plotting:
        ``"none"`` (raw per-batch mean), ``"rolling"`` (trailing
        rolling mean of length ``window``), or ``"cumulative"``
        (expanding mean from the first clipped batch).
    window :
        Window length for ``smoothing="rolling"``. Ignored otherwise.
    show_raw :
        Ignored — raw overlay is permanently disabled in this figure for
        visual clarity. Kept for API compat.
    timestamp :
        If ``True`` (default), append ``_YYYYmmdd_HHMMSS`` to the output
        filename so successive trials accumulate instead of overwriting.
    """
    import matplotlib.pyplot as plt

    runs_by_mode: Dict[str, Path] = {}
    for d in run_dirs:
        d = Path(d)
        metrics = _load_metrics(d)
        mode = (metrics.get("stream", {}).get("extras") or {}).get("drift_mode")
        if mode in scenarios:
            runs_by_mode[mode] = d

    missing = [s for s in scenarios if s not in runs_by_mode]
    if missing:
        raise ValueError(f"No run found for drift_mode(s): {missing}")

    _check_smoothing(smoothing)

    # Preload + window every series so we can compute shared y-lims first.
    data: Dict[str, Dict[str, np.ndarray]] = {}
    drift_by_mode: Dict[str, List[int]] = {}
    for mode in scenarios:
        run = runs_by_mode[mode]
        steps = _load_steps(run)
        metrics = _load_metrics(run)

        t_all = np.array([r["step"] for r in steps])
        mask = t_all >= start
        if end is not None:
            mask &= t_all <= end
        raw = {
            "tv": _series(steps, "mean_tv")[mask],
            "epi": _series(steps, "mean_epistemic")[mask],
            "total": _series(steps, "mean_total")[mask],
            "aleatoric": _series(steps, "mean_aleatoric")[mask],
        }
        data[mode] = {
            "t": t_all[mask],
            "raw": raw,
            "tv": _smooth(raw["tv"], smoothing, window),
            "epi": _smooth(raw["epi"], smoothing, window),
            "total": _smooth(raw["total"], smoothing, window),
            "aleatoric": _smooth(raw["aleatoric"], smoothing, window),
        }
        stream = metrics.get("stream", {}) or {}
        drift_by_mode[mode] = [
            d for d in (stream.get("drift_indices") or [])
            if d >= start and (end is None or d <= end)
        ]

    left_vals = np.concatenate([
        np.concatenate([
            data[m]["tv"][~np.isnan(data[m]["tv"])],
            data[m]["total"][~np.isnan(data[m]["total"])],
            data[m]["aleatoric"][~np.isnan(data[m]["aleatoric"])],
        ])
        for m in scenarios
    ])
    right_vals = np.concatenate([
        data[m]["epi"][~np.isnan(data[m]["epi"])] for m in scenarios
    ])
    left_lim = _padded_limits(left_vals)
    right_lim = _padded_limits(right_vals)

    tv_label = _series_label(r"Mean TV$(p^*, \hat p)$", smoothing, window)
    total_label = _series_label(r"Mean total $H[\hat p]$", smoothing, window)
    aleatoric_label = _series_label(
        r"Mean aleatoric $E_w H[p_w]$", smoothing, window
    )
    epi_label = _series_label("Mean epistemic (MI)", smoothing, window)

    n = len(scenarios)
    panel = 3.6  # inches per square panel
    fig, axes = plt.subplots(
        1, n, figsize=(panel * n + 1.0, panel + 0.8), dpi=150, sharey=True,
    )
    if n == 1:
        axes = [axes]
    twin_axes: List[Any] = []
    handles: Dict[str, Any] = {}

    for i, (ax, mode) in enumerate(zip(axes, scenarios)):
        d = data[mode]
        ax_r = ax.twinx()
        if twin_axes:
            ax_r.sharey(twin_axes[0])
        twin_axes.append(ax_r)

        lines: List[Any] = []

        if not np.all(np.isnan(d["tv"])):
            (ln,) = ax.plot(d["t"], d["tv"], color="#1f77b4", lw=1.5,
                            label=tv_label)
            lines.append(ln)
        if not np.all(np.isnan(d["total"])):
            (ln,) = ax.plot(d["t"], d["total"], color="#555555", lw=0.9,
                            ls="--", alpha=0.75,
                            label=total_label)
            lines.append(ln)
        if not np.all(np.isnan(d["aleatoric"])):
            (ln,) = ax.plot(d["t"], d["aleatoric"], color="#b07aa1", lw=0.9,
                            ls=":", alpha=0.85,
                            label=aleatoric_label)
            lines.append(ln)

        (ln_epi,) = ax_r.plot(d["t"], d["epi"], color="#2ca02c", lw=1.5,
                              label=epi_label)
        lines.append(ln_epi)

        for dr in drift_by_mode[mode]:
            ax.axvline(dr, color="#d62728", ls="--", lw=1.2, alpha=0.7)
        if drift_by_mode[mode]:
            drift_handle = ax.plot(
                [], [], color="#d62728", ls="--", lw=1.2, alpha=0.7,
                label="Drift event",
            )[0]
            lines.append(drift_handle)

        ax.set_xlim(start, end if end is not None else d["t"].max())
        ax.set_ylim(*left_lim)
        ax_r.set_ylim(*right_lim)
        ax.set_box_aspect(1.0)  # square panel

        ax.set_title(_SCENARIO_LABELS.get(mode, mode),
                     fontsize=11, fontweight="bold")
        ax.set_xlabel(r"Batch index $t$")

        if i == 0:
            ax.set_ylabel("TV / nats (left)")
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
        if i == n - 1:
            ax_r.set_ylabel("Epistemic MI (right)", color="#2ca02c")
            ax_r.tick_params(axis="y", colors="#2ca02c")
        else:
            ax_r.tick_params(axis="y", right=False, labelright=False)

        ax.grid(axis="y", alpha=0.2)

        for ln in lines:
            handles.setdefault(ln.get_label(), ln)

    if title is None:
        # Auto-detect uncertainty method from first available run's config.
        unc_label = "uncertainty"
        for d in run_dirs:
            cfg_path = Path(d) / "config.resolved.json"
            if cfg_path.exists():
                import json as _json
                _cfg = _json.loads(cfg_path.read_text())
                _unc = (_cfg.get("uncertainty") or {}).get("name", "")
                if "laplace" in _unc:
                    unc_label = "Laplace"
                elif "mc_dropout" in _unc or "mc-dropout" in _unc:
                    unc_label = "MC-Dropout"
                break
        title = f"Synthetic GMM — TV vs {unc_label} uncertainty decomposition (batches {start}–{end})"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.04)
    fig.legend(
        handles.values(),
        handles.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=min(5, len(handles)),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    out_path = _finalize_out_path(out_path, timestamp)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _padded_limits(v: np.ndarray, pad: float = 0.08) -> tuple[float, float]:
    if v.size == 0:
        return (0.0, 1.0)
    lo, hi = float(np.min(v)), float(np.max(v))
    span = hi - lo if hi > lo else max(abs(hi), 1e-6)
    return (lo - pad * span, hi + pad * span)


def plot_tv_vs_uncertainty(
    run_dir: str | Path,
    out_path: str | Path | None = None,
    smoothing: str = "rolling",
    window: int = 10,
    show_raw: bool = True,
    timestamp: bool = True,
) -> Path:
    """Headline synthetic-GMM figure.

    Single axis with batch index on x and:

    * **Mean TV** (solid) — ``(1/B) Σ_i ½ Σ_k |p*_{i,k} - ĥ_{i,k}|``.
    * **Mean epistemic** (solid) — MI from Laplace MC.
    * **Mean total** and **mean aleatoric** (dashed / dotted, faint) for
      reference, so the decomposition is visible in the same frame.

    Scheduled drift batches (from the stream spec) are marked with
    vertical red dashed lines, and the title shows the drift mode.

    Parameters
    ----------
    smoothing :
        Temporal statistic applied before plotting: ``"none"``,
        ``"rolling"`` (trailing rolling mean of length ``window``), or
        ``"cumulative"`` (expanding mean from batch 0).
    window :
        Window length for ``smoothing="rolling"``. Ignored otherwise.
    show_raw :
        If ``True`` and ``smoothing != "none"``, overlay the raw TV and
        epistemic series as faint background lines.
    timestamp :
        If ``True`` (default), append ``_YYYYmmdd_HHMMSS`` to the output
        filename so successive trials accumulate instead of overwriting.
    """

    import matplotlib.pyplot as plt

    _check_smoothing(smoothing)

    run_dir = Path(run_dir)
    steps = _load_steps(run_dir)
    metrics = _load_metrics(run_dir)

    t = np.array([r["step"] for r in steps])
    tv_raw = _series(steps, "mean_tv")
    epi_raw = _series(steps, "mean_epistemic")
    total_raw = _series(steps, "mean_total")
    aleatoric_raw = _series(steps, "mean_aleatoric")

    tv = _smooth(tv_raw, smoothing, window)
    epi = _smooth(epi_raw, smoothing, window)
    total = _smooth(total_raw, smoothing, window)
    aleatoric = _smooth(aleatoric_raw, smoothing, window)

    stream = metrics.get("stream", {}) or {}
    drift_indices = list(stream.get("drift_indices") or [])
    drift_mode = (stream.get("extras") or {}).get("drift_mode", "?")

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=150)

    draw_raw = show_raw and smoothing != "none"
    if draw_raw:
        if not np.all(np.isnan(tv_raw)):
            ax.plot(t, tv_raw, color="#1f77b4", lw=0.8, alpha=0.25)
        if not np.all(np.isnan(epi_raw)):
            ax.plot(t, epi_raw, color="#2ca02c", lw=0.8, alpha=0.25)

    tv_label = _series_label(r"Mean TV$(p^*, \hat p)$", smoothing, window)
    epi_label = _series_label("Mean epistemic (MI)", smoothing, window)
    total_label = _series_label(r"Mean total $H[\hat p]$", smoothing, window)
    aleatoric_label = _series_label(
        r"Mean aleatoric $E_w H[p_w]$", smoothing, window
    )

    if not np.all(np.isnan(tv)):
        ax.plot(t, tv, color="#1f77b4", lw=1.5, label=tv_label)
    ax.plot(t, epi, color="#2ca02c", lw=1.5, label=epi_label)
    if not np.all(np.isnan(total)):
        ax.plot(t, total, color="#555555", lw=0.9, ls="--", alpha=0.75,
                label=total_label)
    if not np.all(np.isnan(aleatoric)):
        ax.plot(t, aleatoric, color="#b07aa1", lw=0.9, ls=":", alpha=0.85,
                label=aleatoric_label)

    for d in drift_indices:
        ax.axvline(d, color="#d62728", ls="--", lw=1.2, alpha=0.7)
    if drift_indices:
        ax.plot([], [], color="#d62728", ls="--", lw=1.2,
                alpha=0.7, label="Drift event")

    ax.set_xlabel(r"Batch index $t$")
    ax.set_ylabel("nats (uncertainty) / TV")
    ax.set_title(
        f"Synthetic GMM — drift_mode = {drift_mode}",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()

    if out_path is None:
        out_path = run_dir / "figures" / "tv_vs_uncertainty.png"
    out_path = _finalize_out_path(out_path, timestamp)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_mnist_tasks_panel(
    run_dir: str | Path,
    out_path: str | Path | None = None,
    smoothing: str = "rolling",
    window: int = 10,
    show_raw: bool = True,
    timestamp: bool = True,
) -> Path:
    """Figure 2: MNIST changing-task benchmark.

    Four stacked panels sharing the x-axis (batch index ``t``):

    1. Block **error rate** ``1 - accuracy``.
    2. Mean **MI** (epistemic) and mean **predictive entropy** (total).
    3. Mean **max-softmax** confidence.
    4. Detector **alarms** — one row per configured detector, with a
       tick at each batch that raised an alarm.

    Vertical red dashed lines mark scheduled task-change boundaries.
    The task name for each phase is annotated along the top.

    ``smoothing`` / ``window`` / ``show_raw`` apply to the error / MI /
    entropy / max-softmax channels only.
    """
    import matplotlib.pyplot as plt

    _check_smoothing(smoothing)

    run_dir = Path(run_dir)
    steps = _load_steps(run_dir)
    metrics = _load_metrics(run_dir)

    t = np.array([r["step"] for r in steps])
    err_raw = 1.0 - _series(steps, "accuracy")
    mi_raw = _series(steps, "mean_epistemic")
    ent_raw = _series(steps, "mean_total")
    ms_raw = _series(steps, "mean_max_softmax")

    err = _smooth(err_raw, smoothing, window)
    mi = _smooth(mi_raw, smoothing, window)
    ent = _smooth(ent_raw, smoothing, window)
    ms = _smooth(ms_raw, smoothing, window)

    stream = metrics.get("stream", {}) or {}
    extras = stream.get("extras") or {}
    drift_indices = list(stream.get("drift_indices") or [])
    schedule = extras.get("schedule") or []
    transition = extras.get("transition", "abrupt")

    detector_names = sorted({
        name
        for r in steps
        for name in (r.get("detectors") or {}).keys()
    })
    detector_alarms = {
        name: np.array([
            bool((r.get("detectors") or {}).get(name, {}).get("alarm", False))
            for r in steps
        ])
        for name in detector_names
    }

    n_detector_rows = 1 if detector_names else 0
    fig, axes = plt.subplots(
        3 + n_detector_rows, 1, figsize=(11, 1.8 * (3 + n_detector_rows) + 1.0),
        dpi=150, sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.5, 1.2] + [0.9] * n_detector_rows},
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    ax_err, ax_unc, ax_ms = axes[0], axes[1], axes[2]
    ax_det = axes[3] if n_detector_rows else None

    def _maybe_raw(ax, y_raw, color):
        if show_raw and smoothing != "none" and not np.all(np.isnan(y_raw)):
            ax.plot(t, y_raw, color=color, lw=0.8, alpha=0.25)

    err_label = _series_label("Block error rate", smoothing, window)
    _maybe_raw(ax_err, err_raw, "#d62728")
    ax_err.plot(t, err, color="#d62728", lw=1.5, label=err_label)
    ax_err.set_ylabel("error")
    ax_err.set_ylim(-0.02, 1.02)
    ax_err.grid(axis="y", alpha=0.2)

    mi_label = _series_label("Mean epistemic (MI)", smoothing, window)
    ent_label = _series_label("Mean total entropy", smoothing, window)
    _maybe_raw(ax_unc, mi_raw, "#2ca02c")
    _maybe_raw(ax_unc, ent_raw, "#555555")
    ax_unc.plot(t, mi, color="#2ca02c", lw=1.5, label=mi_label)
    ax_unc.plot(t, ent, color="#555555", lw=0.9, ls="--", alpha=0.85, label=ent_label)
    ax_unc.set_ylabel("nats")
    ax_unc.grid(axis="y", alpha=0.2)
    ax_unc.legend(fontsize=8, frameon=False, loc="upper right")

    ms_label = _series_label("Mean max-softmax", smoothing, window)
    _maybe_raw(ax_ms, ms_raw, "#1f77b4")
    ax_ms.plot(t, ms, color="#1f77b4", lw=1.5, label=ms_label)
    ax_ms.set_ylabel("max p(y|x)")
    ax_ms.set_ylim(0.0, 1.02)
    ax_ms.grid(axis="y", alpha=0.2)

    for ax in (ax_err, ax_unc, ax_ms):
        for d in drift_indices:
            ax.axvline(d, color="#d62728", ls="--", lw=1.0, alpha=0.6)

    # Task-name annotations along the top of the error axis.
    if schedule:
        phase_starts = _phase_starts(schedule, transition, extras.get("gradual_span", 0))
        phase_ends = phase_starts[1:] + [int(stream.get("n_batches", t.max() + 1))]
        y_top = ax_err.get_ylim()[1]
        for phase, s, e in zip(schedule, phase_starts, phase_ends):
            mid = 0.5 * (s + e)
            ax_err.text(
                mid, y_top - 0.08, str(phase.get("task", "?")),
                ha="center", va="top", fontsize=9, fontweight="bold",
                color="#333333",
                bbox=dict(boxstyle="round,pad=0.2",
                          fc="white", ec="#cccccc", lw=0.6, alpha=0.85),
            )

    if ax_det is not None:
        for i, name in enumerate(detector_names):
            alarms = detector_alarms[name]
            tt = t[alarms]
            ax_det.scatter(
                tt, np.full(tt.shape[0], i), marker="|", s=60, lw=1.5,
                color=_detector_color(i),
            )
        ax_det.set_yticks(range(len(detector_names)))
        ax_det.set_yticklabels(detector_names, fontsize=8)
        ax_det.set_ylim(-0.5, len(detector_names) - 0.5)
        ax_det.set_ylabel("alarms")
        ax_det.grid(axis="x", alpha=0.15)
        for d in drift_indices:
            ax_det.axvline(d, color="#d62728", ls="--", lw=1.0, alpha=0.6)

    axes[-1].set_xlabel(r"Batch index $t$")
    fig.suptitle(
        f"MNIST changing tasks — transition={transition}",
        fontsize=12, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    if out_path is None:
        out_path = run_dir / "figures" / "mnist_tasks_panel.png"
    out_path = _finalize_out_path(out_path, timestamp)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_mnist_c_panel(
    run_dir: str | Path,
    out_path: str | Path | None = None,
    smoothing: str = "rolling",
    window: int = 5,
    show_raw: bool = True,
    timestamp: bool = True,
    title: str | None = None,
) -> Path:
    """Figure 2 — MNIST-C benchmark, B&W-safe concatenated panel.

    Four stacked axes (shared x):

    1. **Block error** — ``1 - accuracy`` (solid black).
    2. **Uncertainty signal** — mean predictive entropy (solid) and mean
       epistemic MI (dashed) in grayscale.
    3. **Detector alarms** — rows grouped into *Supervised* / *Uncertainty*
       / *Input-based* with distinct marker shapes per group so the figure
       reads unambiguously in black-and-white print.

    Gradual transition segments are marked with ``////`` hatching rather
    than colour fills.  Vertical black dashed lines mark phase boundaries.
    """
    import matplotlib.pyplot as plt

    _check_smoothing(smoothing)

    run_dir = Path(run_dir)
    steps = _load_steps(run_dir)
    metrics = _load_metrics(run_dir)

    t = np.array([r["step"] for r in steps])

    # Use BER when logged (wireless experiments), fall back to 1-accuracy (SER)
    _ber_vals = [r.get("ber") for r in steps]
    _has_ber = any(v is not None for v in _ber_vals)
    if _has_ber:
        err_raw = np.array([v if v is not None else float("nan") for v in _ber_vals])
        err_ylabel = "BER"
        err_series_label = "Bit error rate"
    else:
        err_raw = 1.0 - _series(steps, "accuracy")
        err_ylabel = "error rate"
        err_series_label = "Block error rate"

    mi_raw = _series(steps, "mean_epistemic")
    ent_raw = _series(steps, "mean_total")

    err = _smooth(err_raw, smoothing, window)
    mi = _smooth(mi_raw, smoothing, window)
    ent = _smooth(ent_raw, smoothing, window)

    stream = metrics.get("stream", {}) or {}
    extras = stream.get("extras") or {}
    drift_indices = list(stream.get("drift_indices") or [])
    segments = extras.get("segments") or []
    severity = extras.get("severity", "?")
    novel_start = extras.get("novel_start_batch")  # set by mnist_known_novel

    all_detector_names = sorted({
        name for r in steps for name in (r.get("detectors") or {}).keys()
    })
    detector_alarms = {
        name: np.array([
            bool((r.get("detectors") or {}).get(name, {}).get("alarm", False))
            for r in steps
        ])
        for name in all_detector_names
    }
    groups = _group_detectors(all_detector_names)
    ordered_rows = _ordered_detector_rows(groups)  # [(group_label, det_name), …]

    n_rows = len(ordered_rows)
    n_det = 1 if n_rows > 0 else 0
    det_height = max(0.35 * n_rows, 1.5)

    fig, axes = plt.subplots(
        2 + n_det, 1,
        figsize=(12, 1.8 * 2 + det_height + 1.2),
        dpi=150, sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.5] + [det_height] * n_det},
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    ax_err, ax_unc = axes[0], axes[1]
    ax_det = axes[2] if n_det else None

    # --- Error axis ---
    err_label = _series_label(err_series_label, smoothing, window)
    if _has_ber:
        # Log scale: replace zeros/negatives with NaN so they don't break the axis
        err_raw = np.where(err_raw > 0, err_raw, np.nan)
        err     = np.where(err     > 0, err,     np.nan)
    ax_err.plot(t, err, color="black", lw=1.5, ls="-", label=err_label)
    ax_err.set_ylabel(err_ylabel)
    ax_err.legend(fontsize=8, frameon=False, loc="best")
    if _has_ber:
        ax_err.set_yscale("log")
        ax_err.set_ylim(1e-3, 1.0)
        ax_err.yaxis.set_major_formatter(
            plt.matplotlib.ticker.LogFormatterSciNotation(labelOnlyBase=False)
        )
    else:
        # Fit the axis to the data: low-error runs (e.g. MNIST) get a 0–0.5 view
        # so the curve is legible instead of a flat line; runs that genuinely
        # exceed 0.45 (e.g. CIFAR novel corruptions) keep the full 0–1 range.
        _emax = np.nanmax(err) if np.isfinite(np.nanmax(err)) else 1.0
        _top = 0.5 if _emax <= 0.45 else 1.02
        ax_err.set_ylim(-0.01 * _top, _top)
    ax_err.grid(axis="y", alpha=0.2)

    # --- Uncertainty panel: total on left axis, epistemic on right twin axis ---
    _TOTAL_COLOR = "#1f77b4"   # blue — left axis
    _EPI_COLOR   = "#d62728"   # red  — right axis

    ent_label = _series_label("Mean total entropy", smoothing, window)
    mi_label  = _series_label("Mean epistemic (MI)", smoothing, window)

    ax_epi = ax_unc.twinx()   # shares x and the same panel; independent y

    ax_unc.plot(t, ent, color=_TOTAL_COLOR, lw=1.5, ls="-",  label=ent_label)
    ax_epi.plot(t, mi,  color=_EPI_COLOR,   lw=1.2, ls="--", label=mi_label)

    ax_unc.set_ylabel("total entropy (nats)", color=_TOTAL_COLOR, fontsize=9)
    ax_epi.set_ylabel("epistemic MI (nats)",  color=_EPI_COLOR,   fontsize=9)
    ax_unc.tick_params(axis="y", labelcolor=_TOTAL_COLOR)
    ax_epi.tick_params(axis="y", labelcolor=_EPI_COLOR)

    # Grid from the left axis only (avoids doubled grid lines)
    ax_unc.grid(axis="y", alpha=0.2)
    ax_epi.grid(False)

    # Combined legend
    handles = [
        ax_unc.get_lines()[-1],
        ax_epi.get_lines()[-1],
    ]
    ax_unc.legend(handles=handles, fontsize=8, frameon=False, loc="upper left")

    # --- Shading + drift lines ---
    shade_ranges = [
        (int(s["start"]), int(s["end"]))
        for s in segments if s.get("kind") == "transition"
    ]
    drawing_axes = [ax_err, ax_unc]
    if ax_det is not None:
        drawing_axes.append(ax_det)
    for ax in drawing_axes:
        for lo, hi in shade_ranges:
            ax.axvspan(lo, hi, facecolor="none", edgecolor="black",
                       hatch="////", alpha=0.25, lw=0)
        for d in drift_indices:
            ax.axvline(d, color="black", ls="--", lw=0.9, alpha=0.55)
        if novel_start is not None:
            ax.axvline(novel_start, color="black", ls="-", lw=1.6, alpha=0.9)

    # --- Phase labels on error panel ---
    if segments:
        y_top = ax_err.get_ylim()[1]
        for seg in segments:
            if seg.get("kind") != "pure":
                continue
            mid = 0.5 * (int(seg["start"]) + int(seg["end"]))
            ax_err.text(
                mid, y_top - 0.06, str(seg.get("corruption_a", "?")),
                ha="center", va="top", fontsize=8, fontweight="bold",
                color="black",
                bbox=dict(boxstyle="round,pad=0.2",
                          fc="white", ec="black", lw=0.5, alpha=0.85),
            )
    # Channel name labels (Quadriga/wireless) or fallback known-vs-novel annotation
    channel_names = extras.get("channel_names")
    if channel_names and drift_indices:
        bounds = [0] + sorted(drift_indices) + [int(t[-1]) + 1]
        # Blended transform: x in data coords, y in axes coords (0=bottom, 1=top).
        # This pins labels to the top edge regardless of linear vs log scale.
        from matplotlib.transforms import blended_transform_factory
        trans = blended_transform_factory(ax_err.transData, ax_err.transAxes)
        for i, cname in enumerate(channel_names[: len(bounds) - 1]):
            mid = 0.5 * (bounds[i] + bounds[i + 1])
            is_novel = (novel_start is not None and bounds[i] >= novel_start)
            fc = "#fff8e1" if is_novel else "white"
            ax_err.text(
                mid, 0.97, cname,
                ha="center", va="top", fontsize=8, fontweight="bold",
                color="black", transform=trans,
                bbox=dict(boxstyle="round,pad=0.2", fc=fc, ec="black",
                          lw=0.5, alpha=0.9),
            )

    # --- Detector alarm panel (B&W grouped) ---
    # Drift lines are drawn at a low zorder so alarm markers that land exactly
    # on the boundary (delay 0) stay visible on top of the bold line.
    if ax_det is not None:
        for lo, hi in shade_ranges:
            ax_det.axvspan(lo, hi, facecolor="none", edgecolor="black",
                           hatch="////", alpha=0.25, lw=0, zorder=0.5)
        for d in drift_indices:
            ax_det.axvline(d, color="black", ls="--", lw=0.9, alpha=0.55, zorder=0.5)
        if novel_start is not None:
            ax_det.axvline(novel_start, color="black", ls="-", lw=1.4, alpha=0.9,
                           zorder=0.5)
        _draw_alarm_panel(ax_det, t, ordered_rows, detector_alarms, groups)

    axes[-1].set_xlabel(r"Batch index $t$")
    if title is None:
        # Derive the dataset label from the stream name so CIFAR/wireless/etc.
        # don't inherit the MNIST-C default.
        _ds_titles = {
            "cifar": "CIFAR-10-C",
            "quadriga": "Quadriga QPSK",
            "fashion": "Fashion-MNIST-C",
            "kmnist": "KMNIST-C",
            "mnist": "MNIST-C",
        }
        ds_name = str(stream.get("name", ""))
        ds_label = next(
            (v for k, v in _ds_titles.items() if ds_name.startswith(k)), "MNIST-C"
        )
        # Wireless (Quadriga) streams shift channel, not corruption.
        noun = "channel" if ds_name.startswith("quadriga") else "corruption"
        title = (
            f"{ds_label} — abrupt+gradual {noun} stream (severity={severity})"
            if novel_start is None
            else f"{ds_label} — known vs novel {noun} (known phase | novel phase)"
        )
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.998)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    if out_path is None:
        out_path = run_dir / "figures" / "mnist_c_panel.png"
    out_path = _finalize_out_path(out_path, timestamp)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_summary_table(
    runs: Dict[str, str | Path],
    out_path: str | Path,
    tolerance: int = 10,
    timestamp: bool = True,
) -> Path:
    """Figure 3 — multi-dataset detector alarm summary table.

    Generates a publication-ready matplotlib table where:
    * **Columns** are datasets (keys of ``runs``).
    * **Rows** are detectors, grouped into *Supervised* / *Uncertainty* /
      *Input-based* with bold group-header rows.
    * **Cell values** are ``"N_total (N_true)"`` where ``N_true`` counts
      alarms that fired within ``tolerance`` batches after a known drift
      event.  For datasets with no labelled drift indices (e.g. Elec2) the
      true-alarm field shows ``—``.

    Parameters
    ----------
    runs :
        Mapping from a short dataset label (used as column header) to the
        run output directory produced by :func:`run_experiment`.
    out_path :
        Where to write the PNG.
    tolerance :
        Number of batches after a drift event within which an alarm counts
        as a true positive.
    timestamp :
        Append ``_YYYYmmdd_HHMMSS`` to the filename if ``True``.
    """
    import matplotlib.pyplot as plt

    dataset_labels = list(runs.keys())
    all_steps: Dict[str, List[Dict[str, Any]]] = {}
    all_metrics: Dict[str, Dict[str, Any]] = {}
    for lbl, d in runs.items():
        all_steps[lbl] = _load_steps(Path(d))
        all_metrics[lbl] = _load_metrics(Path(d))

    # Collect detector names from all runs (union, preserving insertion order).
    seen: dict = {}
    for lbl in dataset_labels:
        for r in all_steps[lbl]:
            for name in (r.get("detectors") or {}):
                seen[name] = True
    all_det_names = list(seen)

    groups = _group_detectors(all_det_names)
    ordered_rows = _ordered_detector_rows(groups)

    # Build table data: rows = detectors (+group headers), cols = datasets.
    # We need to interleave group-header rows with detector rows.
    row_labels: List[str] = []
    is_header: List[bool] = []
    cell_data: List[List[str]] = []

    prev_group: str | None = None
    for group_label, det_name in ordered_rows:
        if group_label != prev_group:
            row_labels.append(group_label)
            is_header.append(True)
            cell_data.append([""] * len(dataset_labels))
            prev_group = group_label

        row_labels.append(det_name)
        is_header.append(False)
        row_cells: List[str] = []
        for lbl in dataset_labels:
            steps = all_steps[lbl]
            stream_meta = all_metrics[lbl].get("stream", {}) or {}
            drift_idx = list(stream_meta.get("drift_indices") or [])

            alarms = [
                r["step"]
                for r in steps
                if (r.get("detectors") or {}).get(det_name, {}).get("alarm", False)
            ]
            n_total = len(alarms)

            if not drift_idx:
                row_cells.append(f"{n_total} (—)")
            else:
                true_windows = set()
                for di in drift_idx:
                    true_windows.update(range(di, di + tolerance + 1))
                n_true = sum(1 for a in alarms if a in true_windows)
                pct = 100.0 * n_true / n_total if n_total > 0 else 0.0
                row_cells.append(f"{n_total} ({n_true}, {pct:.0f}%)")

        cell_data.append(row_cells)

    n_rows = len(row_labels)
    n_cols = len(dataset_labels)
    col_w = 1.6
    row_h = 0.38
    fig_w = 1.8 + col_w * n_cols
    fig_h = 0.6 + row_h * n_rows

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    ax.axis("off")

    table = ax.table(
        cellText=cell_data,
        rowLabels=row_labels,
        colLabels=dataset_labels,
        cellLoc="center",
        rowLoc="right",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)

    # Style header rows (group names) and column headers.
    for key, cell in table.get_celld().items():
        row = key[0]
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#dddddd")
        elif is_header[row - 1]:
            cell.set_text_props(fontweight="bold", fontstyle="italic")
            cell.set_facecolor("#f0f0f0")
        else:
            cell.set_facecolor("white")
        cell.set_edgecolor("#999999")

    ax.set_title(
        f"Detector alarm summary — N_total (N_true, % precision within {tolerance} batches)",
        fontsize=10, fontweight="bold", pad=8,
    )

    out_path = _finalize_out_path(out_path, timestamp)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_hyperparams_table(
    runs: Dict[str, str | Path],
    out_path: str | Path,
    timestamp: bool = True,
) -> Path:
    """Hyperparameter reference table across datasets.

    Rows: detectors grouped into Supervised / Uncertainty / Input-based.
    Columns: dataset labels (keys of ``runs``).
    Cells: compact key=value strings for the discriminating parameters
    (threshold, alpha, delta, stat_size, window). The ``name`` and
    ``signal`` keys are implicit and omitted to save space.

    Parameters are read from ``metrics.json`` (the ``detectors`` list),
    which is written by the runner at the end of every experiment.
    """
    import matplotlib.pyplot as plt

    dataset_labels = list(runs.keys())
    all_metrics: Dict[str, Dict[str, Any]] = {}
    for lbl, d in runs.items():
        all_metrics[lbl] = _load_metrics(Path(d))

    det_params_by_dataset: Dict[str, Dict[str, dict]] = {}
    det_order: List[str] = []
    seen_det: dict = {}
    for lbl in dataset_labels:
        det_list = all_metrics[lbl].get("detectors") or []
        det_params_by_dataset[lbl] = {}
        for entry in det_list:
            name = str(entry.get("name", "?"))
            params = dict(entry.get("params") or {})
            det_params_by_dataset[lbl][name] = params
            if name not in seen_det:
                seen_det[name] = True
                det_order.append(name)

    groups = _group_detectors(det_order)
    ordered_rows = _ordered_detector_rows(groups)

    row_labels: List[str] = []
    is_header: List[bool] = []
    cell_data: List[List[str]] = []
    prev_group: str | None = None
    for group_label, det_name in ordered_rows:
        if group_label != prev_group:
            row_labels.append(group_label)
            is_header.append(True)
            cell_data.append([""] * len(dataset_labels))
            prev_group = group_label
        row_labels.append(det_name)
        is_header.append(False)
        row: List[str] = []
        for lbl in dataset_labels:
            params = det_params_by_dataset.get(lbl, {}).get(det_name, {})
            row.append(_format_params_compact(params) if params else "—")
        cell_data.append(row)

    if not cell_data:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 1), dpi=150)
        ax.axis("off")
        ax.text(0.5, 0.5, "No detectors found", ha="center", va="center")
        out_path = _finalize_out_path(out_path, timestamp)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return Path(out_path)

    n_rows = len(row_labels)
    n_cols = len(dataset_labels)
    col_w = 2.2
    row_h = 0.38
    fig_w = 2.0 + col_w * n_cols
    fig_h = 0.6 + row_h * n_rows

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    ax.axis("off")

    table = ax.table(
        cellText=cell_data,
        rowLabels=row_labels,
        colLabels=dataset_labels,
        cellLoc="center",
        rowLoc="right",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)

    for key, cell in table.get_celld().items():
        row = key[0]
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#dddddd")
        elif is_header[row - 1]:
            cell.set_text_props(fontweight="bold", fontstyle="italic")
            cell.set_facecolor("#f0f0f0")
        else:
            cell.set_facecolor("white")
        cell.set_edgecolor("#999999")

    ax.set_title("Detector hyperparameters", fontsize=10, fontweight="bold", pad=8)

    out_path = _finalize_out_path(out_path, timestamp)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_drift_panel(
    run_dir: str | Path,
    title: str | None = None,
    out_path: str | Path | None = None,
    smoothing: str = "rolling",
    window: int = 10,
    timestamp: bool = True,
) -> Path:
    """Generic drift detection panel for tabular streaming datasets (Elec2, Insects).

    Three stacked axes sharing the x-axis:

    1. **Error rate** — ``1 - accuracy`` (solid black).
    2. **Uncertainty** — mean epistemic MI (dashed) and mean total entropy
       (solid) in grayscale.
    3. **Detector alarms** — B&W grouped panel (same layout as
       :func:`plot_mnist_c_panel`).

    For datasets without labelled drift indices (e.g. Elec2), no vertical
    drift lines are drawn; supervised DDM/EDDM alarms serve as the
    de-facto reference.
    """
    import matplotlib.pyplot as plt

    _check_smoothing(smoothing)

    run_dir = Path(run_dir)
    steps = _load_steps(run_dir)
    metrics = _load_metrics(run_dir)

    t = np.array([r["step"] for r in steps])
    err_raw = 1.0 - _series(steps, "accuracy")
    mi_raw = _series(steps, "mean_epistemic")
    ent_raw = _series(steps, "mean_total")

    err = _smooth(err_raw, smoothing, window)
    mi = _smooth(mi_raw, smoothing, window)
    ent = _smooth(ent_raw, smoothing, window)

    stream = metrics.get("stream", {}) or {}
    extras = stream.get("extras") or {}
    drift_indices = list(stream.get("drift_indices") or [])
    dataset_name = stream.get("name", str(run_dir.parent.name))

    all_detector_names = sorted({
        name for r in steps for name in (r.get("detectors") or {}).keys()
    })
    detector_alarms = {
        name: np.array([
            bool((r.get("detectors") or {}).get(name, {}).get("alarm", False))
            for r in steps
        ])
        for name in all_detector_names
    }
    groups = _group_detectors(all_detector_names)
    ordered_rows = _ordered_detector_rows(groups)

    n_rows = len(ordered_rows)
    n_det = 1 if n_rows > 0 else 0
    det_height = max(0.35 * n_rows, 1.5)

    fig, axes = plt.subplots(
        2 + n_det, 1,
        figsize=(12, 1.8 * 2 + det_height + 1.2),
        dpi=150, sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.5] + [det_height] * n_det},
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    ax_err, ax_unc = axes[0], axes[1]
    ax_det = axes[2] if n_det else None

    err_label = _series_label("Block error rate", smoothing, window)
    ax_err.plot(t, err, color="black", lw=1.5, ls="-", label=err_label)
    ax_err.set_ylabel("error rate")
    ax_err.set_ylim(-0.02, 1.02)
    ax_err.grid(axis="y", alpha=0.2)
    ax_err.legend(fontsize=8, frameon=False, loc="upper right")

    ent_label = _series_label("Mean total entropy", smoothing, window)
    mi_label = _series_label("Mean epistemic (MI)", smoothing, window)
    ax_unc.plot(t, ent, color="black", lw=1.5, ls="-", label=ent_label)
    ax_unc.plot(t, mi, color="black", lw=1.0, ls="--", alpha=0.7, label=mi_label)
    ax_unc.set_ylabel("nats")
    ax_unc.grid(axis="y", alpha=0.2)
    ax_unc.legend(fontsize=8, frameon=False, loc="upper right")

    for ax in [ax_err, ax_unc]:
        for d in drift_indices:
            ax.axvline(d, color="black", ls="--", lw=0.9, alpha=0.55)

    if ax_det is not None:
        _draw_alarm_panel(ax_det, t, ordered_rows, detector_alarms, groups)
        for d in drift_indices:
            ax_det.axvline(d, color="black", ls="--", lw=0.9, alpha=0.55)
        if drift_indices:
            ax_det.plot([], [], color="black", ls="--", lw=0.9,
                        label="drift event")

    extra_info = []
    for k in ("variant", "n_features"):
        if k in extras:
            extra_info.append(f"{k}={extras[k]}")
    subtitle = f" ({', '.join(extra_info)})" if extra_info else ""
    axes[-1].set_xlabel(r"Batch index $t$")
    plot_title = title or f"{dataset_name}{subtitle} — drift detection panel"
    fig.suptitle(plot_title, fontsize=12, fontweight="bold", y=0.998)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    if out_path is None:
        out_path = run_dir / "figures" / "drift_panel.png"
    out_path = _finalize_out_path(out_path, timestamp)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _compute_auroc_per_detector(
    steps: List[Dict[str, Any]],
    novel_start: int,
) -> Dict[str, float]:
    """AUROC of each detector's raw signal statistic vs known/novel phase label.

    Label 0 = known phase (step < novel_start), 1 = novel phase.
    ``statistic`` is the raw signal value logged by the runner — higher
    values indicate more drift for all detector types (epistemic/total
    entropy, mean_input L2, or error rate for DDM/EDDM).

    Returns ``NaN`` for a detector if the known or novel phase has fewer
    than 2 distinct values (degenerate ROC).
    """
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "plot_auroc_table requires scikit-learn: pip install scikit-learn"
        ) from exc

    t_arr = np.array([r["step"] for r in steps])
    labels = (t_arr >= novel_start).astype(int)

    if labels.sum() == 0 or (1 - labels).sum() == 0:
        return {}

    all_det_names = sorted({
        name for r in steps for name in (r.get("detectors") or {})
    })
    aurocs: Dict[str, float] = {}
    for det in all_det_names:
        scores = np.array([
            float((r.get("detectors") or {}).get(det, {}).get("statistic", np.nan))
            for r in steps
        ])
        if np.isnan(scores).all():
            aurocs[det] = np.nan
            continue
        finite = ~np.isnan(scores)
        if finite.sum() < 4 or len(np.unique(labels[finite])) < 2:
            aurocs[det] = np.nan
            continue
        try:
            aurocs[det] = float(roc_auc_score(labels[finite], scores[finite]))
        except Exception:
            aurocs[det] = np.nan
    return aurocs


def _compute_far_delay(
    steps: List[Dict[str, Any]],
    novel_start: int,
    target_far: float = 0.0,
) -> Dict[str, int | None]:
    """Detection delay (batches) at a controlled false alarm rate.

    Threshold = (1 - target_far)-th percentile of known-phase statistics.
    Detection delay = (first novel batch with statistic > threshold) - novel_start.
    Returns ``None`` when no novel-phase batch exceeds the threshold.

    target_far=0.0 sets threshold = max(known_stats) so zero false alarms.
    target_far=0.05 sets threshold at the 95th percentile of known stats.
    """
    t_arr = np.array([r["step"] for r in steps])
    known_mask = t_arr < novel_start
    novel_mask = t_arr >= novel_start

    if not known_mask.any() or not novel_mask.any():
        return {}

    all_det_names = sorted({
        name for r in steps for name in (r.get("detectors") or {})
    })
    delays: Dict[str, int | None] = {}
    for det in all_det_names:
        scores = np.array([
            float((r.get("detectors") or {}).get(det, {}).get("statistic", np.nan))
            for r in steps
        ])
        known_scores = scores[known_mask]
        novel_scores = scores[novel_mask]
        novel_times = t_arr[novel_mask]

        valid_known = known_scores[~np.isnan(known_scores)]
        if valid_known.size == 0:
            delays[det] = None
            continue

        if target_far <= 0.0:
            threshold = float(np.max(valid_known))
        else:
            threshold = float(np.quantile(valid_known, 1.0 - target_far))

        fired = np.where((novel_scores > threshold) & (~np.isnan(novel_scores)))[0]
        if fired.size == 0:
            delays[det] = None
        else:
            delays[det] = int(novel_times[fired[0]]) - novel_start
    return delays


def plot_auroc_table(
    runs: Dict[str, str | Path],
    out_path: str | Path,
    timestamp: bool = True,
) -> Path:
    """Figure 3 (main) — AUROC table across datasets and detectors.

    Uses the raw ``statistic`` value logged per detector per step as the
    continuous score for ROC computation.  This is threshold-free: it
    measures each signal's *intrinsic* ability to separate known-phase
    from novel-phase batches, independent of the alarm threshold chosen
    in the config.

    Parameters
    ----------
    runs :
        Mapping from a short dataset label to a run directory.  Each run
        must have ``metrics.json`` with ``stream.extras.novel_start_batch``
        set (i.e. the run used a known-vs-novel stream).
    out_path :
        Output PNG path.
    timestamp :
        Append ``_YYYYmmdd_HHMMSS`` to the filename.
    """
    import matplotlib.pyplot as plt

    dataset_labels = list(runs.keys())
    all_steps: Dict[str, List] = {}
    all_metrics: Dict[str, Dict] = {}
    for lbl, d in runs.items():
        all_steps[lbl] = _load_steps(Path(d))
        all_metrics[lbl] = _load_metrics(Path(d))

    novel_starts: Dict[str, int] = {}
    for lbl in dataset_labels:
        extras = (all_metrics[lbl].get("stream") or {}).get("extras") or {}
        ns = extras.get("novel_start_batch")
        if ns is None:
            raise ValueError(
                f"Run '{lbl}' has no novel_start_batch in stream extras. "
                "Only known-vs-novel runs support the AUROC table."
            )
        novel_starts[lbl] = int(ns)

    seen: dict = {}
    for lbl in dataset_labels:
        for r in all_steps[lbl]:
            for name in (r.get("detectors") or {}):
                seen[name] = True
    all_det_names = list(seen)

    groups = _group_detectors(all_det_names)
    ordered_rows = _ordered_detector_rows(groups)

    aurocs_by_label: Dict[str, Dict[str, float]] = {
        lbl: _compute_auroc_per_detector(all_steps[lbl], novel_starts[lbl])
        for lbl in dataset_labels
    }

    row_labels: List[str] = []
    is_header: List[bool] = []
    cell_data: List[List[str]] = []
    prev_group: str | None = None

    for group_label, det_name in ordered_rows:
        if group_label != prev_group:
            row_labels.append(group_label)
            is_header.append(True)
            cell_data.append([""] * len(dataset_labels))
            prev_group = group_label
        row_labels.append(det_name)
        is_header.append(False)
        row: List[str] = []
        for lbl in dataset_labels:
            v = aurocs_by_label[lbl].get(det_name, np.nan)
            row.append("—" if np.isnan(v) else f"{v:.3f}")
        cell_data.append(row)

    n_rows = len(row_labels)
    n_cols = len(dataset_labels)
    col_w = 1.4
    row_h = 0.38
    fig_w = 1.8 + col_w * n_cols
    fig_h = 0.6 + row_h * n_rows

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    ax.axis("off")

    table = ax.table(
        cellText=cell_data,
        rowLabels=row_labels,
        colLabels=dataset_labels,
        cellLoc="center",
        rowLoc="right",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)

    for key, cell in table.get_celld().items():
        row = key[0]
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#dddddd")
        elif is_header[row - 1]:
            cell.set_text_props(fontweight="bold", fontstyle="italic")
            cell.set_facecolor("#f0f0f0")
        else:
            txt = cell.get_text().get_text()
            if txt not in ("", "—"):
                try:
                    v = float(txt)
                    if v >= 0.90:
                        cell.set_facecolor("#d4edda")
                    elif v >= 0.75:
                        cell.set_facecolor("#fff3cd")
                    else:
                        cell.set_facecolor("#f8d7da")
                except ValueError:
                    cell.set_facecolor("white")
            else:
                cell.set_facecolor("white")
        cell.set_edgecolor("#999999")

    ax.set_title(
        "AUROC — known vs novel phase discrimination (higher is better)",
        fontsize=10, fontweight="bold", pad=8,
    )

    out_path = _finalize_out_path(out_path, timestamp)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


def plot_far_table(
    runs: Dict[str, str | Path],
    out_path: str | Path,
    target_far: float = 0.05,
    timestamp: bool = True,
) -> Path:
    """Appendix table — detection delay at a controlled false alarm rate.

    For each detector and dataset, reports the number of batches after
    ``novel_start_batch`` until the first alarm fires when the decision
    threshold is set to achieve ``target_far`` false alarms in the known
    phase.  ``"∞"`` means the detector never fires at that FAR level.

    Two FAR levels are always shown side by side: FAR=0 (max known-phase
    threshold) and the requested ``target_far``.

    Parameters
    ----------
    runs :
        Same format as :func:`plot_auroc_table`.
    out_path :
        Output PNG path.
    target_far :
        Controlled false alarm rate (e.g. 0.05 for 5%).
    timestamp :
        Append timestamp to filename.
    """
    import matplotlib.pyplot as plt

    dataset_labels = list(runs.keys())
    all_steps: Dict[str, List] = {}
    all_metrics: Dict[str, Dict] = {}
    for lbl, d in runs.items():
        all_steps[lbl] = _load_steps(Path(d))
        all_metrics[lbl] = _load_metrics(Path(d))

    novel_starts: Dict[str, int] = {}
    for lbl in dataset_labels:
        extras = (all_metrics[lbl].get("stream") or {}).get("extras") or {}
        ns = extras.get("novel_start_batch")
        if ns is None:
            raise ValueError(
                f"Run '{lbl}' has no novel_start_batch in stream extras."
            )
        novel_starts[lbl] = int(ns)

    seen: dict = {}
    for lbl in dataset_labels:
        for r in all_steps[lbl]:
            for name in (r.get("detectors") or {}):
                seen[name] = True
    all_det_names = list(seen)
    groups = _group_detectors(all_det_names)
    ordered_rows = _ordered_detector_rows(groups)

    delays_far0: Dict[str, Dict[str, int | None]] = {
        lbl: _compute_far_delay(all_steps[lbl], novel_starts[lbl], target_far=0.0)
        for lbl in dataset_labels
    }
    delays_fark: Dict[str, Dict[str, int | None]] = {
        lbl: _compute_far_delay(all_steps[lbl], novel_starts[lbl], target_far=target_far)
        for lbl in dataset_labels
    }

    # Build column headers: one pair per dataset
    col_labels = []
    for lbl in dataset_labels:
        col_labels.append(f"{lbl}\nFAR=0")
        col_labels.append(f"{lbl}\nFAR={target_far:.0%}")

    row_labels: List[str] = []
    is_header: List[bool] = []
    cell_data: List[List[str]] = []
    prev_group: str | None = None

    for group_label, det_name in ordered_rows:
        if group_label != prev_group:
            row_labels.append(group_label)
            is_header.append(True)
            cell_data.append([""] * len(col_labels))
            prev_group = group_label
        row_labels.append(det_name)
        is_header.append(False)
        row: List[str] = []
        for lbl in dataset_labels:
            for delays in (delays_far0[lbl], delays_fark[lbl]):
                d = delays.get(det_name)
                row.append("∞" if d is None else str(d))
        cell_data.append(row)

    n_rows = len(row_labels)
    n_cols = len(col_labels)
    col_w = 1.1
    row_h = 0.42
    fig_w = 2.0 + col_w * n_cols
    fig_h = 0.8 + row_h * n_rows

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    ax.axis("off")

    table = ax.table(
        cellText=cell_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="right",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)

    for key, cell in table.get_celld().items():
        row = key[0]
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#dddddd")
        elif is_header[row - 1]:
            cell.set_text_props(fontweight="bold", fontstyle="italic")
            cell.set_facecolor("#f0f0f0")
        else:
            txt = cell.get_text().get_text()
            if txt == "∞":
                cell.set_facecolor("#f8d7da")
            elif txt == "":
                cell.set_facecolor("white")
            else:
                try:
                    v = int(txt)
                    if v <= 5:
                        cell.set_facecolor("#d4edda")
                    elif v <= 15:
                        cell.set_facecolor("#fff3cd")
                    else:
                        cell.set_facecolor("#fde8d8")
                except ValueError:
                    cell.set_facecolor("white")
        cell.set_edgecolor("#999999")

    ax.set_title(
        f"Detection delay (batches after novel start) at FAR=0 and FAR={target_far:.0%}\n"
        "Green ≤ 5 batches, yellow ≤ 15, orange > 15, red = never fired",
        fontsize=9, fontweight="bold", pad=8,
    )

    out_path = _finalize_out_path(out_path, timestamp)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


def _format_params_compact(params: dict) -> str:
    """Compact key=value string for the hyperparams table cells."""
    abbrev = {
        "threshold": "thr", "alpha": "α", "delta": "δ",
        "window_size": "w", "stat_size": "st", "n_bins": "bins",
        "min_instances": "min_n", "min_expected": "min_e",
    }
    skip = {"name", "signal", "mode"}
    parts = []
    for k in sorted(params):
        if k in skip:
            continue
        v = params[k]
        label = abbrev.get(k, k)
        if isinstance(v, float):
            parts.append(f"{label}={v:.3g}")
        else:
            parts.append(f"{label}={v}")
    return ", ".join(parts) if parts else "default"


# ---------------------------------------------------------------------------
# Detector grouping helpers (shared by panel and table)
# ---------------------------------------------------------------------------

_GROUP_SUPERVISED = "Supervised"
_GROUP_UNCERTAINTY = "Uncertainty-based"
_GROUP_INPUT = "Input-based"
_GROUP_ORDER = [_GROUP_SUPERVISED, _GROUP_UNCERTAINTY, _GROUP_INPUT]

# Marker shapes per group — distinct in B&W print.
_GROUP_MARKER = {
    _GROUP_SUPERVISED: "o",   # filled circle
    _GROUP_UNCERTAINTY: "|",  # vertical bar
    _GROUP_INPUT: "x",        # cross
}
_GROUP_MS = {
    _GROUP_SUPERVISED: 40,
    _GROUP_UNCERTAINTY: 70,
    _GROUP_INPUT: 40,
}


_EXCLUDED_DETECTORS = {"chi2_total", "chi2_input"}


def _group_detectors(names: List[str]) -> Dict[str, List[str]]:
    """Classify detector names into supervised / uncertainty / input groups.

    Detectors in ``_EXCLUDED_DETECTORS`` are silently dropped so they never
    appear in any panel or table.
    """
    groups: Dict[str, List[str]] = {g: [] for g in _GROUP_ORDER}
    for name in names:
        if name in _EXCLUDED_DETECTORS:
            continue
        n = name.lower()
        if any(k in n for k in ("ddm", "eddm")):
            groups[_GROUP_SUPERVISED].append(name)
        elif any(k in n for k in ("input",)):
            groups[_GROUP_INPUT].append(name)
        else:
            groups[_GROUP_UNCERTAINTY].append(name)
    return groups


def _ordered_detector_rows(
    groups: Dict[str, List[str]],
) -> List[tuple]:
    """Return ``[(group_label, det_name), …]`` in canonical group order.

    Within the Uncertainty-based group, detectors are sub-sorted so that all
    epistemic-MI-based detectors appear before all total-entropy-based ones
    (ours first, then the baseline), and same-signal detectors cluster together.
    """
    rows = []
    for g in _GROUP_ORDER:
        dets = list(groups.get(g, []))
        if g == _GROUP_UNCERTAINTY:
            def _signal_sort_key(name: str) -> tuple:
                n = name.lower()
                if "epistemic" in n or "_mi" in n:
                    return (0, name)
                if "total" in n or "entropy" in n:
                    return (1, name)
                return (2, name)
            dets = sorted(dets, key=_signal_sort_key)
        else:
            dets = sorted(dets)
        for det in dets:
            rows.append((g, det))
    return rows


def _draw_alarm_panel(
    ax,
    t: np.ndarray,
    ordered_rows: List[tuple],
    detector_alarms: Dict[str, np.ndarray],
    groups: Dict[str, List[str]],
) -> None:
    """Populate the alarm axis with grouped, B&W-safe scatter rows."""
    row_idx = 0
    ytick_positions: List[float] = []
    ytick_labels: List[str] = []
    separator_positions: List[float] = []

    prev_group: str | None = None
    for group_label, det_name in ordered_rows:
        if group_label != prev_group and prev_group is not None:
            separator_positions.append(row_idx - 0.5)
        prev_group = group_label

        alarms = detector_alarms.get(det_name, np.zeros(len(t), dtype=bool))
        tt = t[alarms]
        marker = _GROUP_MARKER[group_label]
        ms = _GROUP_MS[group_label]
        yy = np.full(len(tt), row_idx)
        # White halo behind each marker so alarms landing on the bold drift
        # line (delay 0) remain visible; markers sit above the drift lines.
        ax.scatter(tt, yy, marker=marker, s=ms * 1.7, color="white",
                   linewidths=3.2, zorder=3)
        ax.scatter(tt, yy, marker=marker, s=ms,
                   color=_signal_color(det_name), linewidths=2.0, zorder=4)

        ytick_positions.append(row_idx)
        ytick_labels.append(det_name)
        row_idx += 1

    for sep in separator_positions:
        ax.axhline(sep, color="black", lw=0.6, ls=":", alpha=0.6)

    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels, fontsize=7)
    ax.set_ylim(-0.5, row_idx - 0.5)
    ax.set_ylabel("detector alarms")
    ax.grid(axis="x", alpha=0.12)

    # Group labels in left margin.
    if row_idx > 0:
        group_starts: Dict[str, int] = {}
        for i, (g, _) in enumerate(ordered_rows):
            if g not in group_starts:
                group_starts[g] = i
        for g, start in group_starts.items():
            end = start + len(groups[g]) - 1
            mid = 0.5 * (start + end)
            # Push the rotated group label into the left margin, clear of the
            # per-detector y-tick labels (e.g. "kswin_input").
            ax.annotate(
                g,
                xy=(0.0, (mid + 0.5) / row_idx), xycoords=ax.transAxes,
                xytext=(-92, 0), textcoords="offset points",
                ha="center", va="center",
                fontsize=7, fontstyle="italic", fontweight="bold", color="black",
                rotation=90, annotation_clip=False,
            )


def _phase_starts(schedule: Sequence[Dict[str, Any]], transition: str, gradual_span: int) -> List[int]:
    starts: List[int] = []
    t = 0
    for i, phase in enumerate(schedule):
        starts.append(t)
        t += int(phase.get("n_batches", 0))
        if i < len(schedule) - 1 and transition == "gradual":
            t += int(gradual_span)
    return starts


def _detector_color(i: int) -> str:
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    return palette[i % len(palette)]


def _signal_color(det_name: str) -> str:
    """Alarm marker color keyed on the uncertainty signal the detector monitors."""
    n = det_name.lower()
    if "total" in n or "entropy" in n:
        return "#ff7f0e"   # orange — total predictive entropy
    if "epistemic" in n or "_mi" in n:
        return "#1f77b4"   # blue — epistemic MI
    if any(k in n for k in ("ddm", "eddm")):
        return "#7f7f7f"   # gray — supervised
    if "input" in n:
        return "#2ca02c"   # green — input-based
    return "#d62728"       # red — fallback


def _load_steps(run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "steps.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _load_metrics(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _series(steps: List[Dict[str, Any]], key: str) -> np.ndarray:
    vals = []
    for r in steps:
        v = r.get(key)
        vals.append(np.nan if v is None else float(v))
    return np.array(vals, dtype=np.float64)


def _check_smoothing(smoothing: str) -> None:
    if smoothing not in _SMOOTHING_CHOICES:
        raise ValueError(
            f"smoothing must be one of {_SMOOTHING_CHOICES}, got {smoothing!r}"
        )


def _smooth(y: np.ndarray, smoothing: str, window: int) -> np.ndarray:
    """Apply a temporal statistic to ``y``.

    ``"none"`` returns the input unchanged. ``"rolling"`` returns a
    trailing rolling mean of length ``window`` with partial windows at
    the start (``min_periods=1``). ``"cumulative"`` returns the
    expanding mean from index 0. Both aggregations ignore NaN entries:
    positions where the window contains only NaN stay NaN.
    """
    if smoothing == "none":
        return y
    if y.size == 0:
        return y

    if smoothing == "rolling":
        if window <= 1:
            return y
        return _rolling_nanmean(y, int(window))
    if smoothing == "cumulative":
        return _expanding_nanmean(y)

    raise ValueError(f"Unsupported smoothing={smoothing!r}")


def _rolling_nanmean(y: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling mean of length ``window``, NaN-aware.

    Each output index ``i`` is the mean of ``y[max(0, i-window+1) : i+1]``
    over non-NaN entries. If the window is entirely NaN the output is
    NaN.
    """
    n = y.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    mask = ~np.isnan(y)
    vals = np.where(mask, y, 0.0)

    c_vals = np.concatenate([[0.0], np.cumsum(vals)])
    c_mask = np.concatenate([[0], np.cumsum(mask.astype(np.int64))])

    idx = np.arange(n)
    lo = np.maximum(0, idx - window + 1)
    hi = idx + 1
    s = c_vals[hi] - c_vals[lo]
    k = c_mask[hi] - c_mask[lo]
    nonzero = k > 0
    out[nonzero] = s[nonzero] / k[nonzero]
    return out


def _expanding_nanmean(y: np.ndarray) -> np.ndarray:
    """Expanding (cumulative) mean from index 0, ignoring NaN."""
    mask = ~np.isnan(y)
    vals = np.where(mask, y, 0.0)
    c_vals = np.cumsum(vals)
    c_mask = np.cumsum(mask.astype(np.int64))
    out = np.full(y.shape[0], np.nan, dtype=np.float64)
    nonzero = c_mask > 0
    out[nonzero] = c_vals[nonzero] / c_mask[nonzero]
    return out


def _finalize_out_path(out_path: str | Path, timestamp: bool) -> Path:
    """Optionally inject ``_YYYYmmdd_HHMMSS`` before the file extension."""
    path = Path(out_path)
    if not timestamp:
        return path
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = path.suffix or ".png"
    stem = path.stem if path.suffix else path.name
    return path.with_name(f"{stem}_{ts}{suffix}")


def _series_label(base: str, smoothing: str, window: int) -> str:
    if smoothing == "rolling":
        return f"{base} (rolling w={window})"
    if smoothing == "cumulative":
        return f"{base} (cumulative)"
    return base
