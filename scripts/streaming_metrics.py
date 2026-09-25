#!/usr/bin/env python3
"""Streaming detection metrics (FAR / delay / detection rate) at a matched FAR budget.

Answers the question Table 1 cannot: AUROC and FPR95 are threshold-FREE
separability measures, so they say nothing about how the sequential detectors
actually behave once deployed. This reports, per arm and per monitored signal,
the operating point selected by scripts/sweep_detectors.py under the shared
false-alarm budget described in Appendix F.8:

    FAR    fraction of KNOWN-phase batches that raise an alarm
    delay  batches from the true drift boundary to the first alarm after it
    det    detection rate (fraction of seeds in which the novel phase alarmed)

FRESHNESS GUARD (why this script is picky)
------------------------------------------
results/sweeps/<tag>/selection.json is written by the sweep, but the config it
swept can be re-swept later without the JSON being regenerated -- that already
happened once here: every audio arm's selection.json still records
target_far=0.0 from a superseded zero-FAR-budget sweep, while the configs carry
values from a later re-sweep. Reporting those numbers under a "5% budget"
heading would be wrong. So each arm is checked on two axes before it is
reported, and anything that fails is listed as STALE rather than quietly
averaged in:

  1. the recorded target_far must equal the budget being reported;
  2. every selected knob value must still match the live config.

    python scripts/streaming_metrics.py --target-far 0.05 \
        --out results/tables/STREAMING_METRICS
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

# Signal grouping: which monitored signal each detector reads.
SIGNALS = ["supervised", "epistemic", "total", "input"]
SIGNAL_LABEL = {
    "supervised": "Error-based (labels)",
    "epistemic": "Epistemic",
    "total": "Total entropy",
    "input": "Input stat.",
}
# Sequential detector algorithms, in the order the paper lists them.
ALGOS = ["adwin", "kswin", "ph"]
ALGO_LABEL = {"adwin": "ADWIN", "kswin": "KSWIN", "ph": "Page-Hinkley"}


def signal_of(detector: str) -> str:
    n = detector.lower()
    if "epistemic" in n:
        return "epistemic"
    if "total" in n:
        return "total"
    if "input" in n:
        return "input"
    return "supervised"          # ddm / eddm / pilot_*


def algo_of(detector: str) -> str:
    n = detector.lower()
    for a in ALGOS:
        if n.startswith(a):
            return a
    return n.split("_")[0]


def live_config_params(cfg_path: str) -> dict:
    """detector-id -> params, straight from the YAML on disk."""
    import yaml
    raw = yaml.safe_load(open(cfg_path))
    out = {}
    for d in raw.get("detectors", []) or []:
        params = d.get("params", {}) or {}
        out[params.get("name", d["name"])] = params
    return out


def check_freshness(sel: dict, cfg_path: str, target_far: float) -> list[str]:
    """Empty list = safe to report. Otherwise the reasons it is not."""
    problems = []
    rec = sel.get("target_far")
    if rec is None or abs(float(rec) - target_far) > 1e-12:
        problems.append(f"swept at target_far={rec}, not {target_far}")
    if not os.path.exists(cfg_path):
        problems.append("config missing")
        return problems
    live = live_config_params(cfg_path)
    drift = []
    for e in sel.get("selection", []):
        det, knob, val = e["detector"], e["knob"], e["selected_value"]
        cur = (live.get(det) or {}).get(knob)
        if cur is None:
            drift.append(f"{det}.{knob} absent from config")
        elif abs(float(cur) - float(val)) > 1e-6 * max(1.0, abs(float(val))):
            drift.append(f"{det}.{knob} config={cur} != swept={val}")
    if drift:
        problems.append(f"{len(drift)} knob(s) diverged from config (e.g. {drift[0]})")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default="results/tables/AUROC_FPR95.json",
                    help="Defines which arms (and their display names) to report.")
    ap.add_argument("--target-far", type=float, default=0.05)
    ap.add_argument("--out", required=True, help="Output stem (.json/.tex written).")
    a = ap.parse_args(argv)

    arms = json.load(open(a.table))["rows"]

    # config -> newest selection.json
    newest: dict[str, tuple[float, str, dict]] = {}
    for p in glob("results/sweeps/*/selection.json"):
        try:
            s = json.load(open(p))
        except Exception:
            continue
        c = s.get("config")
        if not c:
            continue
        m = os.path.getmtime(p)
        if c not in newest or m > newest[c][0]:
            newest[c] = (m, p, s)

    out_rows, stale, missing = [], [], []
    for r in arms:
        cfg = r["config"]
        hit = newest.get(cfg)
        if not hit:
            missing.append((r["dataset"], r["arm"], "no selection.json"))
            continue
        _, path, sel = hit
        problems = check_freshness(sel, cfg, a.target_far)
        if problems:
            stale.append((r["dataset"], r["arm"], "; ".join(problems)))
            continue
        per_signal: dict[str, dict] = {}
        for e in sel["selection"]:
            sig, algo = signal_of(e["detector"]), algo_of(e["detector"])
            per_signal.setdefault(sig, {})[algo] = {
                "detector": e["detector"],
                "knob": e["knob"],
                "value": e["selected_value"],
                "far": e["FAR"],
                "delay": e["delay"],
                "det_rate": e.get("detection_rate"),
                "status": e.get("status"),
            }
        out_rows.append({
            "dataset": r["dataset"], "arm": r["arm"], "config": cfg,
            "source": path, "seeds": sel.get("seeds"),
            "target_far": sel.get("target_far"),
            "signals": per_signal,
        })

    payload = {
        "target_far": a.target_far,
        "rows": out_rows,
        "excluded_stale": [{"dataset": d, "arm": m, "reason": w} for d, m, w in stale],
        "excluded_missing": [{"dataset": d, "arm": m, "reason": w} for d, m, w in missing],
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2))

    print(f"reportable arms : {len(out_rows)}")
    for d, m, w in stale:
        print(f"  STALE   {d:<16} {m:<24} {w}")
    for d, m, w in missing:
        print(f"  MISSING {d:<16} {m:<24} {w}")
    print(f"wrote {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
