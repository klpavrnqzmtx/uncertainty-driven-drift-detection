#!/usr/bin/env python3
"""Standard-detection-theory table: delay at MATCHED false-alarm-rate budgets.

    python scripts/far_targets_table.py results/sweeps/*/selection.json \
        --targets 0,0.01,0.05,0.10 --out results/tables/FAR_TARGETS

Problem this fixes
-------------------
Comparing "detector A fires after 0.4 batches" against "detector B fires after
2.1 batches" is meaningless unless both are operating at the SAME false-alarm
budget. ``sweep_detectors.py`` already traces each detector's full (FAR, delay)
operating curve across its threshold grid and saves it in
``results/sweeps/<tag>/selection.json["curves"]`` — but only the single point
matching one ``--target-far`` was ever read back out. This script reads the
already-computed curves and reports, for EVERY requested FAR budget, the best
delay any grid point achieves without exceeding it — i.e. the standard
detection-theory comparison, at no extra compute.

For a detector whose grid never reaches a given budget (this happened to DDM
under the un-fixed warm_start, and can happen to any detector on a hard grid),
the cell reads ``n/a`` rather than silently reporting its best available FAR.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

ORACLE = {"ddm", "eddm", "pilot_ddm", "pilot_eddm"}


def _family(name: str) -> str:
    if name in ORACLE:
        return "supervised"
    if name.endswith("_input"):
        return "input"
    return "uncertainty"


def delay_at_far(points: List[dict], target: float):
    """Min delay among grid points with far <= target; None if unreachable."""
    ok = [p for p in points
          if p.get("far") is not None and np.isfinite(p["far"]) and p["far"] <= target
          and p.get("delay") is not None and np.isfinite(p["delay"])]
    if not ok:
        return None, None
    best = min(ok, key=lambda p: p["delay"])
    return best["delay"], best["far"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("selections", nargs="+", help="results/sweeps/*/selection.json files")
    ap.add_argument("--targets", default="0.0,0.01,0.05,0.10",
                    help="Comma-separated FAR budgets.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    targets = [float(x) for x in a.targets.split(",")]

    rows = []
    for f in sorted(a.selections):
        d = json.load(open(f))
        tag = Path(f).parent.name
        curves = d.get("curves") or {}
        for det in sorted(curves, key=lambda n: (_family(n), n)):
            cell = {"experiment": tag, "detector": det, "family": _family(det)}
            for tgt in targets:
                delay, achieved_far = delay_at_far(curves[det], tgt)
                cell[f"delay@{tgt}"] = delay
                cell[f"far@{tgt}"] = achieved_far
            rows.append(cell)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"targets": targets, "rows": rows}, open(out.with_suffix(".json"), "w"), indent=1)

    def fmt_delay(v):
        return "n/a" if v is None else f"{v:.1f}"

    hdr = ["experiment", "detector", "family"] + [f"delay@FAR≤{t:g}" for t in targets]
    w_exp = max(len(r["experiment"]) for r in rows)
    w_det = max(len(r["detector"]) for r in rows)
    with open(out.with_suffix(".md"), "w") as fh:
        fh.write("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n")
        for r in rows:
            cells = [r["experiment"], r["detector"], r["family"]] + \
                    [fmt_delay(r[f"delay@{t}"]) for t in targets]
            fh.write("| " + " | ".join(cells) + " |\n")

    print(f"{'experiment':{w_exp}}  {'detector':{w_det}}  {'family':12}" +
          "".join(f"{'FAR<='+str(t):>12}" for t in targets))
    cur = None
    for r in rows:
        if r["experiment"] != cur:
            cur = r["experiment"]
            print(f"-- {cur} " + "-" * 40)
        print(f"{'':{w_exp}}  {r['detector']:{w_det}}  {r['family']:12}" +
              "".join(f"{fmt_delay(r[f'delay@{t}']):>12}" for t in targets))
    print(f"\nwrote {out.with_suffix('.json')} and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
