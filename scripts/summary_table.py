#!/usr/bin/env python3
"""Consolidate per-experiment eval tables into one cross-experiment paper table.

    python scripts/summary_table.py results/tables/*.json --out results/tables/SUMMARY

Two tables, because they answer different questions:

**Signal separability (AUROC).** AUROC is a property of the monitored SIGNAL, not
the algorithm — ph/kswin/adwin on the same signal share it — so this table has one
column per signal: the supervised error stream (the oracle), epistemic MI, total
predictive entropy, and the model-free input statistic. It says which quantity
carries information about the drift, independent of any threshold.

**Operating points (FAR / delay).** For the tuned detectors: how quickly each fires
after the boundary and how often it cries wolf during the known phase. This is
threshold-dependent, so it reflects the sweep as much as the signal.

The error/DDM column is the ORACLE: it monitors true prediction errors, i.e. the
labels the unsupervised detectors never see. Epistemic matching or beating it is the
headline claim; epistemic near 0.5 means the signal is uninformative, and below 0.5
means it is inverted (higher on known data than novel).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

SIGNALS = [("ddm", "error†"), ("ph_epistemic", "epistemic"),
           ("ph_total", "total"), ("ph_input", "input")]


def _pretty(exp: str) -> str:
    e = exp.replace("known_vs_novel", "kn").replace("__", " / ")
    return e.replace("cifar_kn_comparison", "CIFAR-10").replace("cifar100_c", "CIFAR-100") \
            .replace("mnist_c", "MNIST").replace("imagenet_comparison", "ImageNet")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tables", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    data = []
    for f in sorted(a.tables):
        d = json.load(open(f))
        data.append((_pretty(Path(f).stem), {r["detector"]: r for r in d["rows"]}, len(d["seeds"])))

    lines_auroc, lines_op = [], []
    w = max(len(n) for n, _, _ in data)

    hdr = f"{'experiment':{w}}  {'seeds':>5} " + " ".join(f"{lab:>11}" for _, lab in SIGNALS)
    lines_auroc.append(hdr)
    lines_auroc.append("-" * len(hdr))
    for name, rows, ns in data:
        cells = []
        for det, _ in SIGNALS:
            r = rows.get(det)
            cells.append(f"{r['auroc_mean']:.3f}" if r and np.isfinite(r["auroc_mean"]) else "  -  ")
        lines_auroc.append(f"{name:{w}}  {ns:>5} " + " ".join(f"{c:>11}" for c in cells))

    hdr2 = (f"{'experiment':{w}}  " + f"{'ddm† delay':>11} {'ddm† FAR':>9}  "
            f"{'epi delay':>10} {'epi FAR':>8}  {'input delay':>12} {'input FAR':>10}")
    lines_op.append(hdr2)
    lines_op.append("-" * len(hdr2))
    for name, rows, _ in data:
        def cell(det, key):
            r = rows.get(det)
            if not r:
                return "  -  "
            if key == "delay":
                return "miss" if r["delay_mean"] is None else f"{r['delay_mean']:.1f}"
            return f"{r['far_mean']:.3f}"
        lines_op.append(
            f"{name:{w}}  {cell('ddm','delay'):>11} {cell('ddm','far'):>9}  "
            f"{cell('ph_epistemic','delay'):>10} {cell('ph_epistemic','far'):>8}  "
            f"{cell('ph_input','delay'):>12} {cell('ph_input','far'):>10}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    txt = ("AUROC by monitored signal (mean over seeds; higher = better separation)\n"
           + "\n".join(lines_auroc)
           + "\n\nOperating points from the swept thresholds (PageHinkley per signal)\n"
           + "\n".join(lines_op)
           + "\n\n† error/DDM is the SUPERVISED ORACLE: it monitors true prediction errors,\n"
             "  i.e. the labels the unsupervised detectors never observe.\n"
             "  AUROC 0.5 = uninformative; <0.5 = inverted (higher on known than novel).\n")
    out.with_suffix(".txt").write_text(txt)
    print(txt)

    with open(out.with_suffix(".md"), "w") as f:
        f.write("| experiment | seeds | " + " | ".join(l for _, l in SIGNALS) + " |\n")
        f.write("|" + "---|" * (len(SIGNALS) + 2) + "\n")
        for name, rows, ns in data:
            cs = [(f"{rows[d]['auroc_mean']:.3f}" if d in rows and np.isfinite(rows[d]["auroc_mean"]) else "–")
                  for d, _ in SIGNALS]
            f.write(f"| {name} | {ns} | " + " | ".join(cs) + " |\n")
    print(f"wrote {out.with_suffix('.txt')} and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
