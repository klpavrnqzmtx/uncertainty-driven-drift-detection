#!/usr/bin/env python3
"""Unified threshold sweep for drift detectors via a delay-vs-FAR operating curve.

Motivation
----------
Detector knobs are *not* comparable across algorithms: a DDM ``drift_threshold``,
an ADWIN ``delta`` and a PageHinkley ``threshold`` live on unrelated scales.  What
*is* comparable is the **operating point** each knob setting produces:

    * FAR   — false-alarm rate: fraction of *known-phase* batches that alarm
              (for ``multi_drift`` mode: alarms outside any post-drift window).
    * delay — detection delay: batches from the drift boundary to the first
              alarm after it (``inf`` = missed).

We therefore tune *every* detector by the same rule: trace its (FAR, delay)
operating curve as its primary knob varies, then pick the knob at a shared FAR
budget.  This makes DDM tuning directly comparable to PageHinkley-on-epistemic
and PageHinkley-on-input tuning.

Portability
-----------
* ADWIN ``delta`` and KSWIN ``alpha`` are scale-free confidence parameters →
  fixed log grids that port across datasets.
* PageHinkley ``threshold`` is signal-scale dependent → its grid is derived from
  each signal channel's *known-phase* standard deviation (measured on a probe
  run), so the same sweep works on CIFAR entropy (~0.18), MNIST epistemic
  (~0.06) or wireless input (~1.0) without hand-tuning.
* Detectors are auto-discovered from the config, so ``pilot_ddm``/``pilot_eddm``
  (wireless) are handled exactly like ``ddm``/``eddm``.

Efficiency
----------
The full grid is expanded into *parallel* detectors in a single config, so one
model forward pass evaluates every (detector, threshold) variant at once.  The
run is repeated over a few dataset seeds and the operating points are averaged.

Usage
-----
    python scripts/sweep_detectors.py \
        --config configs/experiments/cifar_kn_comparison/known_vs_novel.yaml \
        --mode known_novel --seeds 7,8,9 --target-far 0.05 \
        --out results/sweeps/cifar_mc

Modes
-----
``known_novel``  single known->novel boundary (needs ``novel_start_batch``).
``multi_drift``  several labelled drift points (uses ``drift_indices``); FAR is
                 measured outside post-drift tolerance windows and delay is the
                 mean over drift points.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Local imports (registration side effects mirror the CLI's --with default).
import sys

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import importlib

for _m in (
    "uncertainty_driven_drift.components.builtin",
    "uncertainty_driven_drift.components.phase2",
    "uncertainty_driven_drift.components.phase3",
):
    importlib.import_module(_m)

from uncertainty_driven_drift.config import ComponentSpec, ExperimentConfig, load_config
from uncertainty_driven_drift.runner import run_experiment


# ---------------------------------------------------------------------------
# Knob specification per detector type
# ---------------------------------------------------------------------------

# (param_name, grid_kind) — grid_kind "fixed" uses FIXED_GRIDS, "signal" derives
# the grid from the detector signal's known-phase std.
KNOB: Dict[str, Tuple[str, str]] = {
    "ddm":          ("drift_threshold", "fixed"),
    "pilot_ddm":    ("drift_threshold", "fixed"),
    "eddm":         ("beta",            "fixed"),
    "pilot_eddm":   ("beta",            "fixed"),
    "adwin":        ("delta",           "fixed"),
    "kswin":        ("alpha",           "fixed"),
    "page_hinkley": ("threshold",       "signal"),
}

FIXED_GRIDS: Dict[str, List[float]] = {
    "drift_threshold": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
    # EDDM requires alpha >= beta; we keep beta <= 0.95 and pin alpha per variant.
    "beta":            [0.70, 0.80, 0.85, 0.90, 0.95],
    "delta":           [1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 3e-1],
    "alpha":           [1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2e-1],
}

# Multipliers of the signal's known-phase std for PageHinkley's threshold grid.
PH_SIGMA_MULTS = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]


def _family(name: str) -> str:
    n = name.lower()
    if "ddm" in n or "eddm" in n:
        return "supervised"
    if "input" in n:
        return "input"
    return "uncertainty"


# ---------------------------------------------------------------------------
# Probe: measure per-signal known-phase statistics + drift structure
# ---------------------------------------------------------------------------

def _load_steps(run_dir: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in (run_dir / "steps.jsonl").read_text().splitlines() if l]


def _load_metrics(run_dir: Path) -> Dict[str, Any]:
    return json.loads((run_dir / "metrics.json").read_text())


def _signal_series(steps: List[Dict[str, Any]], channel: str,
                   base_specs: List[ComponentSpec]) -> np.ndarray:
    """Recover a signal channel's per-batch series from a probe run.

    ``mean_total`` / ``mean_epistemic`` / ``mean_aleatoric`` are logged directly
    on each step record.  Other channels (e.g. ``mean_input``) are recovered from
    the ``statistic`` field of whichever base detector monitors that channel.
    """
    if channel in ("mean_total", "mean_epistemic", "mean_aleatoric"):
        return np.array([s.get(channel, np.nan) for s in steps], dtype=float)
    for spec in base_specs:
        if spec.params.get("signal") == channel:
            nm = spec.params.get("name", spec.name)
            return np.array(
                [((s.get("detectors") or {}).get(nm) or {}).get("statistic", np.nan)
                 for s in steps],
                dtype=float,
            )
    return np.full(len(steps), np.nan)


@dataclass
class DriftSpec:
    mode: str
    novel_start: Optional[int]
    drift_points: List[int]
    n_batches: int


def _drift_spec(metrics: Dict[str, Any], steps: List[Dict[str, Any]], mode: str,
                err_thresh: float = 0.10, err_window: int = 5) -> DriftSpec:
    stream = metrics.get("stream") or {}
    extras = stream.get("extras") or {}
    n = len(steps)
    if mode == "known_novel":
        ns = extras.get("novel_start_batch")
        if ns is None:
            raise SystemExit(
                "known_novel mode requires 'novel_start_batch' in stream extras; "
                "this run has none (use --mode multi_drift if it has drift_indices)."
            )
        return DriftSpec("known_novel", int(ns), [int(ns)], n)
    if mode == "error_rate":
        # Reference event = first batch where the error rate *sustainably* crosses
        # the threshold, i.e. where the model starts failing for real. Detectors
        # are then tuned to fire at that crossing; alarms before it are false
        # alarms. Prefer BER when the run logs it (wireless), else use SER
        # (1-accuracy). A *forward* window makes the crossing lock onto the onset
        # of a sustained rise rather than a transient one-batch spike.
        ber = [s.get("ber") for s in steps]
        if any(v is not None for v in ber):
            err = np.array([v if v is not None else np.nan for v in ber], dtype=float)
        else:
            err = 1.0 - np.array([s.get("accuracy", np.nan) for s in steps], dtype=float)
        roll = _forward_mean(err, err_window)
        cross = np.where(roll > err_thresh)[0]
        if cross.size == 0:
            raise SystemExit(
                f"error_rate mode: forward-rolling error never exceeds {err_thresh:g}; "
                "nothing to tune to (try a lower --error-threshold)."
            )
        t = int(cross[0])
        return DriftSpec("error_rate", t, [t], n)
    # multi_drift
    di = sorted(int(d) for d in (stream.get("drift_indices") or []))
    if not di:
        raise SystemExit(
            "multi_drift mode requires labelled 'drift_indices'; this run has none "
            "(real-world streams without ground-truth drift can't be swept this way)."
        )
    return DriftSpec("multi_drift", None, di, n)


def _rolling_mean(y: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return y
    out = np.full(len(y), np.nan)
    for i in range(len(y)):
        seg = y[max(0, i - w + 1): i + 1]
        seg = seg[~np.isnan(seg)]
        if seg.size:
            out[i] = seg.mean()
    return out


def _forward_mean(y: np.ndarray, w: int) -> np.ndarray:
    """Forward rolling mean: out[i] = mean(y[i : i+w]). Used to place the
    error-rate crossing at the *onset* of a sustained rise (ignoring transient
    one-batch spikes that a trailing window would trip on)."""
    if w <= 1:
        return y
    out = np.full(len(y), np.nan)
    for i in range(len(y)):
        seg = y[i: i + w]
        seg = seg[~np.isnan(seg)]
        if seg.size:
            out[i] = seg.mean()
    return out


# ---------------------------------------------------------------------------
# Grid expansion
# ---------------------------------------------------------------------------

@dataclass
class Variant:
    name: str          # unique detector name in the grid config
    base: str          # base detector display name (e.g. ph_epistemic)
    family: str
    param: str
    value: float
    params: Dict[str, Any]   # full detector params for this variant (for write-back)


def _expand(base_specs: List[ComponentSpec],
            sigma_by_channel: Dict[str, float],
            grid_points: int) -> Tuple[List[ComponentSpec], List[Variant]]:
    specs: List[ComponentSpec] = []
    variants: List[Variant] = []
    for spec in base_specs:
        kn = KNOB.get(spec.name)
        base_name = spec.params.get("name", spec.name)
        if kn is None:
            specs.append(spec)  # keep un-sweepable detectors verbatim
            continue
        param, kind = kn
        if kind == "fixed":
            grid = list(FIXED_GRIDS[param])
        else:  # signal-relative (PageHinkley threshold)
            ch = spec.params.get("signal", "mean_total")
            sigma = sigma_by_channel.get(ch, np.nan)
            if not np.isfinite(sigma) or sigma <= 0:
                sigma = 1.0
            grid = [round(sigma * c, 6) for c in PH_SIGMA_MULTS]
        if grid_points and grid_points < len(grid):
            idx = np.linspace(0, len(grid) - 1, grid_points).round().astype(int)
            grid = [grid[i] for i in sorted(set(idx))]
        for i, val in enumerate(grid):
            p = dict(spec.params)
            p[param] = val
            p["name"] = f"{base_name}__k{i:02d}"
            if param == "beta":
                # EDDM requires alpha >= beta (warning fires before drift).
                p["alpha"] = round(max(float(p.get("alpha", 0.95)), val), 4)
            if kind == "signal":
                ch = spec.params.get("signal", "mean_total")
                sigma = sigma_by_channel.get(ch, 1.0)
                if not np.isfinite(sigma) or sigma <= 0:
                    sigma = 1.0
                # Always re-derive PageHinkley's delta from the signal scale,
                # overriding any stale value in the base config — otherwise a
                # delta copied from a different backbone (much larger than this
                # signal's threshold) silently suppresses the detector.
                p["delta"] = round(0.5 * sigma, 6)
                p.setdefault("min_instances", 5)
            specs.append(ComponentSpec(spec.name, p))
            variants.append(Variant(p["name"], base_name, _family(base_name),
                                    param, float(val), dict(p)))
    return specs, variants


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _alarms(steps: List[Dict[str, Any]], det_name: str) -> np.ndarray:
    return np.array(
        [bool(((s.get("detectors") or {}).get(det_name) or {}).get("alarm", False))
         for s in steps],
        dtype=bool,
    )


def _far_delay(alarms: np.ndarray, ds: DriftSpec, tol: int) -> Tuple[float, Optional[float]]:
    idx = np.arange(len(alarms))
    if ds.novel_start is not None:   # single-boundary: known_novel or error_rate
        ns = ds.novel_start
        known = idx < ns
        far = float(alarms[known].mean()) if known.any() else np.nan
        fired = idx[(idx >= ns) & alarms]
        delay = float(fired.min() - ns) if len(fired) else None
        return far, delay
    # multi_drift: delay = mean over drift points; FAR outside post-drift windows
    dps = ds.drift_points + [ds.n_batches]
    true_win = np.zeros(len(alarms), dtype=bool)
    delays: List[float] = []
    for j, d in enumerate(ds.drift_points):
        nxt = dps[j + 1]
        true_win[d:min(d + tol + 1, len(alarms))] = True
        fired = idx[(idx >= d) & (idx < nxt) & alarms]
        if len(fired):
            delays.append(float(fired.min() - d))
    outside = ~true_win
    far = float(alarms[outside].mean()) if outside.any() else np.nan
    delay = float(np.mean(delays)) if delays else None
    return far, delay


# ---------------------------------------------------------------------------
# Run the grid over seeds
# ---------------------------------------------------------------------------

def _run_once(base_cfg: ExperimentConfig, detectors: List[ComponentSpec],
              seed: int, tag: str) -> Path:
    cfg = copy.deepcopy(base_cfg)
    cfg.detectors = detectors
    cfg.seed = seed
    if "seed" in cfg.dataset.params:
        cfg.dataset.params["seed"] = seed
    cfg.experiment = f"{base_cfg.experiment}__sweep_{tag}"
    return run_experiment(cfg)


# ---------------------------------------------------------------------------
# Selection + plot
# ---------------------------------------------------------------------------

@dataclass
class OpPoint:
    value: float
    far: float
    delay: float          # inf encodes "miss / not detected across seeds"
    det_rate: float


def _aggregate(variants: List[Variant], run_dirs: List[Path], ds_list: List[DriftSpec],
               tol: int) -> Dict[str, Dict[float, OpPoint]]:
    """Return {base_detector: {knob_value: OpPoint}} averaged over seeds."""
    out: Dict[str, Dict[float, OpPoint]] = {}
    steps_by_run = [_load_steps(rd) for rd in run_dirs]
    for v in variants:
        fars, delays, hits = [], [], []
        for steps, ds in zip(steps_by_run, ds_list):
            far, delay = _far_delay(_alarms(steps, v.name), ds, tol)
            if np.isfinite(far):
                fars.append(far)
            if delay is not None:
                delays.append(delay)
                hits.append(1.0)
            else:
                hits.append(0.0)
        det_rate = float(np.mean(hits)) if hits else 0.0
        mean_far = float(np.mean(fars)) if fars else np.nan
        mean_delay = float(np.mean(delays)) if delays else float("inf")
        out.setdefault(v.base, {})[v.value] = OpPoint(v.value, mean_far, mean_delay, det_rate)
    return out


def _select(points: Dict[float, OpPoint], target_far: float,
            rule: str = "min_delay", far_weight: float = 100.0) -> OpPoint:
    """Pick an operating point.

    ``min_delay`` — earliest detection subject to FAR<=target (else lowest FAR).
    ``closest``   — alarms clustered as tightly on the drift boundary as
                    possible: minimise ``delay + far_weight*FAR`` so that both
                    late detection *and* premature (pre-boundary) firing are
                    penalised. ``far_weight`` is the known-phase length, so
                    ``far_weight*FAR`` ≈ the number of pre-boundary false alarms.
    """
    if rule == "closest":
        def score(p: OpPoint) -> float:
            d = p.delay if np.isfinite(p.delay) else 1e6
            f = p.far if np.isfinite(p.far) else 1.0
            return d + far_weight * f
        return min(points.values(), key=score)
    ok = [p for p in points.values() if np.isfinite(p.far) and p.far <= target_far
          and np.isfinite(p.delay)]
    if ok:
        return min(ok, key=lambda p: (p.delay, p.far))
    finite = [p for p in points.values() if np.isfinite(p.far)]
    if not finite:
        return list(points.values())[0]
    return min(finite, key=lambda p: (p.far, p.delay))


# Which knob(s) to persist back into the config per detector type. PageHinkley
# needs delta/min_instances too, since the sweep evaluated it with signal-scaled
# delta — writing only threshold would not reproduce the selected operating point.
PERSIST_KEYS = {
    "ddm": ["drift_threshold"], "pilot_ddm": ["drift_threshold"],
    "eddm": ["alpha", "beta"], "pilot_eddm": ["alpha", "beta"],
    "adwin": ["delta"], "kswin": ["alpha"],
    "page_hinkley": ["threshold", "delta", "min_instances"],
}


def _fmt(v: Any) -> str:
    """Format a number so PyYAML re-parses it as a float (not a string).

    PyYAML's implicit float resolver rejects bare scientific notation like
    ``1e-05`` (no dot, unsigned exponent) — it must be ``1.0e-05``.
    """
    if not isinstance(v, (int, float)):
        return str(v)
    s = repr(float(v))
    if "e" in s or "E" in s:
        mant, exp = re.split("[eE]", s)
        if "." not in mant:
            mant += ".0"
        if not (exp.startswith("+") or exp.startswith("-")):
            exp = "+" + exp
        s = f"{mant}e{exp}"
    return s


def _set_inline(line: str, key: str, val: Any) -> str:
    """Set ``key: val`` inside a one-line ``params: {...}`` dict (replace or insert)."""
    pat = re.compile(rf"(\b{re.escape(key)}:\s*)([^,}}]+)")
    if pat.search(line):
        return pat.sub(rf"\g<1>{_fmt(val)}", line, count=1)
    return re.sub(r"(\{)", rf"\g<1>{key}: {_fmt(val)}, ", line, count=1)


def _apply_to_config(path: Path, selected: Dict[str, "OpPoint"],
                     var_params: Dict[Tuple[str, float], Dict[str, Any]]) -> List[str]:
    """Write selected knobs back into the source YAML, preserving comments/layout."""
    text = path.read_text()
    lines = text.splitlines()
    cur_type: Optional[str] = None
    changed: List[str] = []
    for i, line in enumerate(lines):
        m = re.match(r"\s*-\s*name:\s*(\S+)\s*$", line)
        if m:
            cur_type = m.group(1)
            continue
        if cur_type and re.search(r"^\s*params:\s*\{", line):
            dm = re.search(r"name:\s*([A-Za-z0-9_]+)", line)
            disp = dm.group(1) if dm else cur_type
            if disp in selected:
                params = var_params.get((disp, round(selected[disp].value, 8)), {})
                for key in PERSIST_KEYS.get(cur_type, []):
                    if key in params:
                        line = _set_inline(line, key, params[key])
                lines[i] = line
                changed.append(disp)
            cur_type = None
    path.write_text("\n".join(lines) + ("\n" if text.endswith("\n") else ""))
    return changed


_FAMILY_COLOR = {"supervised": "#7f7f7f", "uncertainty": "#1f77b4", "input": "#2ca02c"}
_FAMILY_MARK = {"supervised": "o", "uncertainty": "s", "input": "^"}


def _plot(op: Dict[str, Dict[float, OpPoint]], variants: List[Variant],
          selected: Dict[str, OpPoint], target_far: float, ds_list: List[DriftSpec],
          title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fam_of = {v.base: v.family for v in variants}
    n_novel = ds_list[0].n_batches - (ds_list[0].novel_start or 0)
    miss_y = max(2.0, 1.15 * n_novel)  # where to draw "missed" points

    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    for base, pts in op.items():
        fam = fam_of.get(base, "uncertainty")
        ser = sorted(pts.values(), key=lambda p: p.far)
        xs = [p.far for p in ser]
        ys = [p.delay if np.isfinite(p.delay) else miss_y for p in ser]
        ax.plot(xs, ys, "-", color=_FAMILY_COLOR[fam], alpha=0.35, lw=1.0, zorder=1)
        ax.scatter(xs, ys, s=28, color=_FAMILY_COLOR[fam],
                   marker=_FAMILY_MARK[fam], alpha=0.8, zorder=2,
                   label=f"{base} ({fam})")
        sel = selected[base]
        sy = sel.delay if np.isfinite(sel.delay) else miss_y
        ax.scatter([sel.far], [sy], s=180, facecolors="none",
                   edgecolors=_FAMILY_COLOR[fam], linewidths=2.0, zorder=3)

    ax.axvline(target_far, color="black", ls="--", lw=1.0, alpha=0.6)
    ax.text(target_far, 0.995, f"FAR budget = {target_far:.0%}",
            transform=ax.get_xaxis_transform(), va="top", ha="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.axhline(miss_y, color="red", ls=":", lw=0.8, alpha=0.5)
    ax.text(0.01, miss_y, "missed / never detected", va="bottom", ha="left",
            fontsize=8, color="red", alpha=0.7)
    ax.set_xlabel("False-alarm rate (before drift event)")
    ax.set_ylabel("Detection delay (batches after drift event)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(alpha=0.2)
    circle = plt.Line2D([], [], marker="o", markersize=11, markerfacecolor="none",
                        markeredgecolor="black", linestyle="none",
                        label="selected (min delay @ FAR budget)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [circle], labels + [circle.get_label()],
              fontsize=7, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["known_novel", "error_rate", "multi_drift"],
                    default="known_novel",
                    help="Reference drift event: known->novel boundary, the batch "
                         "where error rate first exceeds --error-threshold, or "
                         "labelled drift points.")
    ap.add_argument("--seeds", default="7,8,9",
                    help="Comma-separated dataset seeds to average over.")
    ap.add_argument("--target-far", type=float, default=0.05)
    ap.add_argument("--select", choices=["min_delay", "closest"], default="min_delay",
                    help="min_delay: earliest alarm at FAR<=target-far. "
                         "closest: alarms clustered as tightly on the drift "
                         "boundary as possible (penalises both delay and "
                         "pre-boundary false alarms).")
    ap.add_argument("--error-threshold", type=float, default=0.10,
                    help="error_rate mode: error-rate crossing that defines the drift event.")
    ap.add_argument("--error-window", type=int, default=5,
                    help="error_rate mode: rolling window (batches) for the error rate.")
    ap.add_argument("--tolerance", type=int, default=5,
                    help="multi_drift: batches after a drift counted as a true detection.")
    ap.add_argument("--grid-points", type=int, default=0,
                    help="Sub-sample each knob grid to at most this many points (0=all).")
    ap.add_argument("--apply-config", default=None,
                    help="Write the selected thresholds back into this YAML config "
                         "(comment-preserving). Default: only report, don't modify.")
    ap.add_argument("--out", required=True, help="Output directory for plot/table/json.")
    args = ap.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = load_config(args.config)
    base_specs = base_cfg.detectors

    # --- Probe run: measure signal scales + drift structure --------------------
    print(f"[sweep] probe run (seed {seeds[0]})…")
    probe_dir = _run_once(base_cfg, base_specs, seeds[0], "probe")
    probe_steps = _load_steps(probe_dir)
    probe_metrics = _load_metrics(probe_dir)
    ds0 = _drift_spec(probe_metrics, probe_steps, args.mode,
                      args.error_threshold, args.error_window)
    if args.mode == "error_rate":
        print(f"[sweep] error>{args.error_threshold:.0%} first crossed at batch "
              f"{ds0.novel_start} (probe)")

    channels = {s.params.get("signal") for s in base_specs if s.params.get("signal")}
    sigma_by_channel: Dict[str, float] = {}
    known_end = ds0.novel_start if ds0.novel_start is not None else ds0.drift_points[0]
    for ch in channels:
        ser = _signal_series(probe_steps, ch, base_specs)[:known_end]
        sigma_by_channel[ch] = float(np.nanstd(ser)) if ser.size else float("nan")
    print("[sweep] known-phase signal std:",
          {k: round(v, 5) for k, v in sigma_by_channel.items()})

    # --- Expand grid + run over seeds -----------------------------------------
    grid_specs, variants = _expand(base_specs, sigma_by_channel, args.grid_points)
    print(f"[sweep] {len(variants)} threshold variants across "
          f"{len({v.base for v in variants})} detectors; running {len(seeds)} seeds…")
    run_dirs: List[Path] = []
    ds_list: List[DriftSpec] = []
    for sd in seeds:
        print(f"[sweep]   seed {sd}…")
        rd = _run_once(base_cfg, grid_specs, sd, f"s{sd}")
        run_dirs.append(rd)
        ds_list.append(_drift_spec(_load_metrics(rd), _load_steps(rd), args.mode,
                                   args.error_threshold, args.error_window))

    # --- Aggregate + select ----------------------------------------------------
    op = _aggregate(variants, run_dirs, ds_list, args.tolerance)
    far_weight = float(known_end)   # ~ number of pre-boundary batches
    selected = {base: _select(pts, args.target_far, args.select, far_weight)
                for base, pts in op.items()}
    var_params = {(v.base, round(v.value, 8)): v.params for v in variants}

    # --- Outputs ---------------------------------------------------------------
    fam_of = {v.base: v.family for v in variants}
    param_of = {v.base: v.param for v in variants}
    rows = []
    order = {"supervised": 0, "uncertainty": 1, "input": 2}
    for base in sorted(op, key=lambda b: (order.get(fam_of[b], 3), b)):
        sel = selected[base]
        if not np.isfinite(sel.delay) or sel.det_rate <= 0.0:
            status = "UNDETECTABLE"          # never fires in the novel phase
        elif not np.isfinite(sel.far) or sel.far > args.target_far + 1e-9:
            status = "FAR>budget"            # cannot meet the FAR budget
        else:
            status = "ok"
        rows.append({
            "detector": base, "family": fam_of[base], "knob": param_of[base],
            "selected_value": sel.value, "FAR": round(sel.far, 4),
            "delay": (None if not np.isfinite(sel.delay) else round(sel.delay, 2)),
            "detection_rate": round(sel.det_rate, 3), "status": status,
        })

    (out_dir / "selection.json").write_text(json.dumps({
        "config": str(args.config), "mode": args.mode, "seeds": seeds,
        "target_far": args.target_far, "signal_std": sigma_by_channel,
        "selection": rows,
        "curves": {b: [vars(p) for p in pts.values()] for b, pts in op.items()},
    }, indent=2))

    title = f"{base_cfg.experiment} — detector tuning ({args.mode}, {len(seeds)} seeds)"
    _plot(op, variants, selected, args.target_far, ds_list, title,
          out_dir / "operating_curve.png")

    # Console table
    w = max(len(r["detector"]) for r in rows)
    print("\n=== selected thresholds (min delay @ FAR <= "
          f"{args.target_far:.0%}) ===")
    print(f"{'detector':<{w}}  {'family':<11} {'knob':<15} {'value':>9} "
          f"{'FAR':>6} {'delay':>7} {'det_rate':>8}  status")
    for r in rows:
        d = "inf" if r["delay"] is None else f"{r['delay']:.1f}"
        print(f"{r['detector']:<{w}}  {r['family']:<11} {r['knob']:<15} "
              f"{r['selected_value']:>9.4g} {r['FAR']:>6.2f} {d:>7} "
              f"{r['detection_rate']:>8.2f}  {r['status']}")
    print(f"\n[sweep] wrote {out_dir/'operating_curve.png'} and "
          f"{out_dir/'selection.json'}")

    if args.apply_config:
        changed = _apply_to_config(Path(args.apply_config), selected, var_params)
        print(f"[sweep] applied selected thresholds to {args.apply_config} "
              f"({len(changed)} detectors: {', '.join(changed)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
