#!/usr/bin/env python
"""Score an S1 known-vs-novel run: does epistemic separate what total entropy cannot?

    python scripts/screen_audio.py results/raw/audio_kn_speech_commands/<timestamp>
    python scripts/screen_audio.py results/raw/audio_kn_*/*/ --csv out.csv

Reports, per run, the quantities the S1/S2 claims are actually made of:

* **known-phase accuracy** — is the backbone worth interpreting at all?
* **shift in sigma** — ``(novel_mean - known_mean) / known_sd`` for each of the
  three signals.  Sigma is the *known-phase batch-to-batch* standard deviation,
  so a shift is measured against the noise a detector actually sees.
* **R = shift_total / shift_epistemic** — the S2 figure of merit, quoted here to
  confirm the *opposite* of S2: on a genuine novelty shift R should sit near 1,
  i.e. epistemic fires as readily as total entropy and the decomposition costs
  nothing.  R >> 1 here would mean the "novel" corruptions are not novel to the
  model, which is a scenario bug, not a result.
* **AUROC** — per-batch separation of known from novel for each signal.
* **corr(total, epistemic)** across the stream, the number that falls from 0.97
  (S1, CIFAR) to 0.17 (S2, CIFAR) when the two signals come apart.
* **alarm counts** per detector as ``(known-phase false alarms, post-boundary)``.

Nothing here is audio-specific — it reads ``steps.jsonl`` — so it scores the
image arms identically and the modalities stay comparable.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

SIGNALS = ("mean_epistemic", "mean_total", "mean_input")


def _load(run_dir: Path) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    steps = [json.loads(line) for line in (run_dir / "steps.jsonl").read_text().splitlines() if line]
    metrics = json.loads((run_dir / "metrics.json").read_text())
    return steps, metrics


def _novel_start(steps: List[Dict[str, Any]], metrics: Dict[str, Any]) -> Optional[int]:
    extras = (metrics.get("stream") or {}).get("extras") or {}
    if "novel_start_batch" in extras:
        return int(extras["novel_start_batch"])
    drifts = (metrics.get("stream") or {}).get("drift_indices") or []
    return int(drifts[-1]) if drifts else None


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney U), tie-aware; no sklearn dependency needed."""
    finite = np.isfinite(scores)
    labels, scores = labels[finite], scores[finite]
    n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks within ties so a constant signal scores 0.5, not 0 or 1
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def score_run(run_dir: Path) -> Dict[str, Any]:
    steps, metrics = _load(run_dir)
    novel_start = _novel_start(steps, metrics)
    if novel_start is None:
        raise ValueError(f"{run_dir}: no novel_start_batch / drift_indices in metrics.json")

    t = np.array([s["step"] for s in steps])
    known = t < novel_start
    novel = ~known
    acc = np.array([s["accuracy"] for s in steps], dtype=float)

    out: Dict[str, Any] = {
        "run": str(run_dir),
        "experiment": metrics.get("experiment", "?"),
        "dataset": ((metrics.get("stream") or {}).get("extras") or {}).get("dataset", ""),
        "n_batches": len(steps),
        "novel_start": novel_start,
        "acc_known": float(acc[known].mean()),
        "acc_novel": float(acc[novel].mean()),
    }

    series: Dict[str, np.ndarray] = {}
    for sig in SIGNALS:
        if sig == "mean_input":
            # The runner does not log mean_input as a column; recover it from the
            # detector that consumes it (its `statistic` IS the channel value).
            vals = [
                (s.get("detectors") or {}).get("adwin_input", {}).get("statistic", np.nan)
                for s in steps
            ]
            v = np.asarray(vals, dtype=float)
        else:
            v = np.array([s.get(sig, np.nan) for s in steps], dtype=float)
        if not np.isfinite(v).any():
            continue
        series[sig] = v
        sd = float(v[known].std())
        shift = float(v[novel].mean() - v[known].mean())
        out[f"{sig}_known"] = float(v[known].mean())
        out[f"{sig}_novel"] = float(v[novel].mean())
        out[f"{sig}_sd_known"] = sd
        out[f"{sig}_shift_sigma"] = shift / sd if sd > 0 else float("nan")
        out[f"{sig}_auroc"] = _auroc(novel.astype(int), v)

    if "mean_total" in series and "mean_epistemic" in series:
        se = out.get("mean_epistemic_shift_sigma", float("nan"))
        out["R"] = (out["mean_total_shift_sigma"] / se) if se not in (0.0,) else float("inf")
        out["corr_total_epistemic"] = float(
            np.corrcoef(series["mean_total"], series["mean_epistemic"])[0, 1]
        )

    alarms: Dict[str, tuple] = {}
    for name in sorted({n for s in steps for n in (s.get("detectors") or {})}):
        fired = np.array(
            [bool((s.get("detectors") or {}).get(name, {}).get("alarm", False)) for s in steps]
        )
        alarms[name] = (int(fired[known].sum()), int(fired[novel].sum()))
    out["alarms"] = alarms
    return out


def _print(rows: List[Dict[str, Any]]) -> None:
    for r in rows:
        print(f"\n=== {r['experiment']}  ({r['dataset'] or 'n/a'})")
        print(f"    {r['run']}")
        print(f"    {r['n_batches']} batches, novel starts at {r['novel_start']}")
        print(f"    accuracy   known {r['acc_known']:.4f} -> novel {r['acc_novel']:.4f}")
        print(f"    {'signal':16s} {'known':>9s} {'novel':>9s} {'sd_known':>9s} "
              f"{'shift':>8s} {'AUROC':>7s}")
        for sig in SIGNALS:
            if f"{sig}_known" not in r:
                continue
            print(f"    {sig:16s} {r[f'{sig}_known']:9.4f} {r[f'{sig}_novel']:9.4f} "
                  f"{r[f'{sig}_sd_known']:9.4f} {r[f'{sig}_shift_sigma']:7.2f}σ "
                  f"{r[f'{sig}_auroc']:7.3f}")
        inverted = [
            sig for sig in SIGNALS
            if f"{sig}_auroc" in r and r[f"{sig}_auroc"] < 0.5
        ]
        if inverted:
            # AUROC < 0.5 is not "uninformative": the signal separates the phases
            # but moves DOWN on novelty. A one-sided rise detector misses it
            # entirely; ADWIN/KSWIN are two-sided and still react. Say so, because
            # 0.36 read as "worse than chance" is the wrong conclusion.
            print(f"    note: {', '.join(inverted)} decrease(s) on the novel phase "
                  f"(AUROC < 0.5) — informative only to a two-sided detector")
        if "R" in r:
            print(f"    R = shift_total / shift_epistemic = {r['R']:.2f}"
                  f"   corr(total, epistemic) = {r['corr_total_epistemic']:.3f}")
            verdict = (
                "epistemic tracks the novelty (S1 as intended)"
                if r["mean_epistemic_shift_sigma"] >= 1.0
                else "epistemic did NOT move — check that the novel corruptions are "
                     "actually novel to this backbone"
            )
            print(f"    -> {verdict}")
        print("    alarms (known-phase, post-boundary):")
        for name, (fa, post) in r["alarms"].items():
            print(f"      {name:18s} ({fa}, {post})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories (steps.jsonl + metrics.json)")
    ap.add_argument("--csv", help="also write one row per run here")
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    for pattern in args.runs:
        path = Path(pattern)
        candidates = [path] if (path / "steps.jsonl").exists() else sorted(path.glob("**/steps.jsonl"))
        for c in candidates:
            run_dir = c if c.is_dir() else c.parent
            try:
                rows.append(score_run(run_dir))
            except Exception as exc:                       # keep going over a batch of runs
                print(f"SKIP {run_dir}: {type(exc).__name__}: {exc}", file=sys.stderr)
    if not rows:
        print("no scorable runs found", file=sys.stderr)
        return 1

    _print(rows)
    if args.csv:
        flat = [{k: v for k, v in r.items() if k != "alarms"} for r in rows]
        keys = sorted({k for r in flat for k in r})
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(flat)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
