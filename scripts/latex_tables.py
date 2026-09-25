#!/usr/bin/env python3
"""Emit the paper's result tables as real LaTeX (booktabs), not rendered images.

Three tables:

  auroc    Table 1 -- AUROC / FPR95 per monitored signal, now WITH the
           across-seed standard deviation that was already being computed and
           silently dropped at render time (results/tables/AUROC_FPR95.json
           carries auroc_std / fpr95_std over 5 seeds per arm).

  fpr      FPR at the 95% / 99% / 100% known-phase operating points, mean +/- std
           over seeds (tab:fpr_ablation), from the same AUROC_FPR95.json.

  stream   Streaming detection metrics -- FAR in the known phase and detection
           delay at the novel boundary, at the matched false-alarm budget of
           Appendix F.8. Built from results/tables/STREAMING_METRICS.json
           (see scripts/streaming_metrics.py, which refuses to emit arms whose
           sweep artifacts are stale).

    python scripts/latex_tables.py auroc  --out results/tables/table1.tex
    python scripts/latex_tables.py fpr    --out results/tables/table_fpr.tex
    python scripts/latex_tables.py stream --out results/tables/table_streaming.tex
"""

from __future__ import annotations

import argparse
import json
import math

DATASET_ORDER = ["MNIST", "CIFAR-10", "CIFAR-100", "ESC-50", "Speech Commands",
                 "UrbanSound8K", "Wireless"]
SIGNALS = ["ddm", "epistemic", "total", "input"]
SIGNAL_TEX = {"ddm": "Error", "epistemic": "EU",
              "total": "Entropy", "input": "Input"}
STREAM_SIGNALS = ["supervised", "epistemic", "total", "input"]
STREAM_TEX = {"supervised": "Error (labels)", "epistemic": "Epistemic",
              "total": "Total ent.", "input": "Input stat."}


# Display names where one JSON arm label means different models across datasets:
# the CIFAR-100 arm is the pretrained ViT-B/16, whereas "ViT" on MNIST and
# CIFAR-10 is a compact ViT trained from scratch.
ARM_DISPLAY = {("CIFAR-100", "ViT Laplace"): "ViT-B/16 Laplace"}


def _grouped(rows):
    out = []
    for ds in DATASET_ORDER:
        rs = [dict(r, arm=ARM_DISPLAY.get((ds, r["arm"]), r["arm"]))
              for r in rows if r["dataset"] == ds]
        if rs:
            out.append((ds, rs))
    return out


def _rank(vals, higher_is_better):
    """Indices tied for best / second, ranked on the DISPLAYED 2dp value."""
    rounded = {i: round(v, 2) for i, v in enumerate(vals) if v is not None}
    if not rounded:
        return set(), set()
    uniq = sorted(set(rounded.values()), reverse=higher_is_better)
    best = {i for i, v in rounded.items() if v == uniq[0]}
    # With only two distinct values the runner-up is also the worst in the row;
    # underlining it would mark e.g. a 1.00 FPR as "second best".
    second = set() if len(uniq) < 3 else {i for i, v in rounded.items() if v == uniq[1]}
    return best, second


def table_auroc(a) -> str:
    d = json.load(open(a.table))
    rows = d["rows"]
    n_seeds = sorted({s.get("n_seeds") for r in rows for s in
                      (r.get(g) or {} for g in SIGNALS) if s.get("n_seeds")})
    seed_note = f"{n_seeds[0]}" if len(n_seeds) == 1 else f"{min(n_seeds)}--{max(n_seeds)}"

    L = []
    L.append(r"\begin{table}[t]")
    L.append(r"\centering")
    L.append(r"\caption{Drift detection performance across the evaluated settings. "
             r"AUROC ($\uparrow$) and FPR95 ($\downarrow$), reported as mean $\pm$ standard "
             rf"deviation over {seed_note} stream seeds. "
             r"Best per row in \textbf{bold}, second best \underline{underlined}. "
             r"Error uses deployment labels and serves only as a supervised reference.}")
    L.append(r"\label{tab:main}")
    L.append(r"\small")
    L.append(r"\setlength{\tabcolsep}{4pt}")
    L.append(r"\begin{tabular}{l" + "c" * (2 * len(SIGNALS)) + "}")
    L.append(r"\toprule")
    L.append(r" & \multicolumn{4}{c}{AUROC $\uparrow$} & \multicolumn{4}{c}{FPR95 $\downarrow$} \\")
    L.append(r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}")
    L.append(" & " + " & ".join(SIGNAL_TEX[s] for s in SIGNALS)
             + " & " + " & ".join(SIGNAL_TEX[s] for s in SIGNALS) + r" \\")
    L.append(r"\midrule")

    for ds, rs in _grouped(rows):
        L.append(rf"\multicolumn{{{1 + 2 * len(SIGNALS)}}}{{l}}{{\textbf{{{ds}}}}} \\")
        for r in rs:
            cells = []
            for metric, hib in (("auroc", True), ("fpr95", False)):
                vals = [(r.get(s) or {}).get(metric) for s in SIGNALS]
                stds = [(r.get(s) or {}).get(f"{metric}_std") for s in SIGNALS]
                best, second = _rank(vals, hib)
                for i, (v, sd) in enumerate(zip(vals, stds)):
                    if v is None:
                        cells.append("--")
                        continue
                    txt = f"{v:.2f}"
                    if sd is not None:
                        txt += rf"{{\tiny$\pm${sd:.2f}}}"
                    if i in best:
                        txt = rf"\textbf{{{txt}}}"
                    elif i in second:
                        txt = rf"\underline{{{txt}}}"
                    cells.append(txt)
            L.append(f"\\quad {_tex_escape(r['arm'])} & " + " & ".join(cells) + r" \\")
        L.append(r"\addlinespace[2pt]")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


def table_fpr(a) -> str:
    """FPR at 95/99/100% known-phase retention, with the across-seed std.

    Same AUROC_FPR95.json and the same 5 seeds as Table 1 -- score_arm() already
    computed all three levels and both stds per level, so this needs no re-run.
    FPR100 is the strictest (threshold = max known-phase statistic, i.e. zero
    known-phase false alarms) and upper-bounds the other two by construction.
    """
    LEVELS = [("fpr95", "FPR95"), ("fpr99", "FPR99"), ("fpr100", "FPR100")]
    d = json.load(open(a.table))
    rows = d["rows"]
    n_seeds = sorted({s.get("n_seeds") for r in rows for s in
                      (r.get(g) or {} for g in SIGNALS) if s.get("n_seeds")})
    seed_note = f"{n_seeds[0]}" if len(n_seeds) == 1 else f"{min(n_seeds)}--{max(n_seeds)}"

    n_cols = len(LEVELS) * len(SIGNALS)
    L = []
    L.append(r"\begin{table}[t]")
    L.append(r"\centering")
    L.append(r"\caption{FPR at three known-phase operating points (95\%, 99\%, and 100\%), "
             rf"reported as mean $\pm$ standard deviation over {seed_note} stream seeds. "
             r"Lower is better; best per row and operating point in \textbf{bold}, second best "
             r"\underline{underlined}. Error uses deployment labels and serves only as a "
             r"supervised reference.}")
    L.append(r"\label{tab:fpr_ablation}")
    L.append(r"\small")
    L.append(r"\setlength{\tabcolsep}{3pt}")
    L.append(r"\resizebox{\textwidth}{!}{%")
    L.append(r"\begin{tabular}{l" + "c" * n_cols + "}")
    L.append(r"\toprule")
    L.append(" & " + " & ".join(rf"\multicolumn{{{len(SIGNALS)}}}{{c}}{{{lbl} $\downarrow$}}"
                               for _, lbl in LEVELS) + r" \\")
    L.append("".join(rf"\cmidrule(lr){{{2 + i * len(SIGNALS)}-{1 + (i + 1) * len(SIGNALS)}}}"
                     for i in range(len(LEVELS))))
    L.append(" & " + " & ".join(" & ".join(SIGNAL_TEX[s] for s in SIGNALS)
                                for _ in LEVELS) + r" \\")
    L.append(r"\midrule")

    for ds, rs in _grouped(rows):
        L.append(rf"\multicolumn{{{1 + n_cols}}}{{l}}{{\textbf{{{ds}}}}} \\")
        for r in rs:
            cells = []
            for key, _ in LEVELS:
                vals = [(r.get(s) or {}).get(key) for s in SIGNALS]
                stds = [(r.get(s) or {}).get(f"{key}_std") for s in SIGNALS]
                best, second = _rank(vals, False)     # lower is better
                for i, (v, sd) in enumerate(zip(vals, stds)):
                    if v is None:
                        cells.append("--")
                        continue
                    txt = f"{v:.2f}"
                    if sd is not None:
                        txt += rf"{{\tiny$\pm${sd:.2f}}}"
                    if i in best:
                        txt = rf"\textbf{{{txt}}}"
                    elif i in second:
                        txt = rf"\underline{{{txt}}}"
                    cells.append(txt)
            L.append(f"\\quad {_tex_escape(r['arm'])} & " + " & ".join(cells) + r" \\")
        L.append(r"\addlinespace[2pt]")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}}")
    L.append(r"\end{table}")
    return "\n".join(L)


def _tex_escape(s: str) -> str:
    return (s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
             .replace("->", r"$\rightarrow$")
             .replace("(severity)", "(sev.)"))


def _agg(algos, key, how="mean"):
    """(aggregate, any-violation) over the sequential algorithms for one signal.

    FAR is summarised by the WORST algorithm, not the mean: the claim being made
    is that a signal can be held to the budget, and a mean hides a single
    detector blowing through it (wireless total-entropy is 0.31/0.00/0.02 --
    a mean of 0.11 reads as compliant when ADWIN plainly is not).
    """
    vals = [v[key] for v in algos.values()
            if v[key] is not None and not (isinstance(v[key], float) and math.isinf(v[key]))]
    bad = any(v.get("status") not in (None, "ok") for v in algos.values())
    if not vals:
        return None, bad
    return (max(vals) if how == "max" else sum(vals) / len(vals)), bad


def table_stream(a) -> str:
    d = json.load(open(a.stream))
    budget = d["target_far"]
    rows = d["rows"]

    L = []
    L.append(r"\begin{table}[t]")
    L.append(r"\centering")
    L.append(rf"\caption{{Sequential detection behaviour at a matched false-alarm budget "
             rf"of {budget:.0%} (Appendix~\ref{{app:calibration}}). FAR is the fraction of "
             r"\emph{known}-phase batches raising an alarm; delay is the number of batches "
             r"from the novel-phase boundary to the first alarm. Each entry averages the "
             r"three sequential detectors (ADWIN, KSWIN, Page--Hinkley); FAR is the worst "
             r"of the three and delay their mean, each detector independently "
             r"calibrated to minimum delay subject to the budget. $\dagger$ marks a signal "
             r"for which at least one detector could not meet the budget at any grid point.}")
    L.append(r"\label{tab:streaming}")
    L.append(r"\small")
    L.append(r"\setlength{\tabcolsep}{4pt}")
    L.append(r"\begin{tabular}{l" + "c" * (2 * len(STREAM_SIGNALS)) + "}")
    L.append(r"\toprule")
    L.append(r" & \multicolumn{4}{c}{FAR, known phase $\downarrow$} & \multicolumn{4}{c}{Detection delay $\downarrow$} \\")
    L.append(r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}")
    L.append(" & " + " & ".join(STREAM_TEX[s] for s in STREAM_SIGNALS)
             + " & " + " & ".join(STREAM_TEX[s] for s in STREAM_SIGNALS) + r" \\")
    L.append(r"\midrule")

    for ds, rs in _grouped(rows):
        L.append(rf"\multicolumn{{{1 + 2 * len(STREAM_SIGNALS)}}}{{l}}{{\textbf{{{ds}}}}} \\")
        for r in rs:
            far_cells, del_cells = [], []
            for s in STREAM_SIGNALS:
                algos = r["signals"].get(s)
                if not algos:
                    far_cells.append("--"); del_cells.append("--"); continue
                far, bad = _agg(algos, "far", how="max")
                dly, _ = _agg(algos, "delay")
                far_cells.append(("--" if far is None else f"{far:.2f}") + (r"$\dagger$" if bad else ""))
                del_cells.append("--" if dly is None else f"{dly:.1f}")
            L.append(f"\\quad {_tex_escape(r['arm'])} & " + " & ".join(far_cells + del_cells) + r" \\")
        L.append(r"\addlinespace[2pt]")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", choices=["auroc", "fpr", "stream"])
    ap.add_argument("--table", default="results/tables/AUROC_FPR95.json")
    ap.add_argument("--stream", default="results/tables/STREAMING_METRICS.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    tex = {"auroc": table_auroc, "fpr": table_fpr, "stream": table_stream}[a.which](a)
    open(a.out, "w").write(tex + "\n")
    print(f"wrote {a.out}  ({len(tex.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
