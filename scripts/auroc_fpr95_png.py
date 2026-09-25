#!/usr/bin/env python3
"""Render results/tables/AUROC_FPR95.json in the visual style of Table 1 of
Ballas et al., "Hydra Ensembles" (arXiv:2510.18358): plain black-on-white,
booktabs-style horizontal rules only, best-in-column **bold**, second-best
_underlined_ — no colour coding.

Grouped by METRIC, not by signal: one block per metric (AUROC, FPR95, FPR99,
FPR100), each with one column per signal, side by side — mirroring the
paper's side-by-side dataset blocks — rather than interleaving AUROC/FPR
within each signal, which made cross-signal comparison on a single metric
awkward.

    python scripts/auroc_fpr95_png.py results/tables/AUROC_FPR95.json \
        --out results/figures/paper/auroc_fpr95_table
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SIGNALS = ["ddm", "epistemic", "total", "input"]
# The supervised column is named for what it monitors, not the algorithm. Wrapped
# onto three lines: at 6.9pt the one-line form is ~0.95in against a 0.62in column.
SIGNAL_LABEL = {"ddm": "Error-based\nsignal\n(labels)", "epistemic": "Epistemic",
                "total": "Total ent.", "input": "Input stat."}
# key -> (label, higher_is_better). One side-by-side block per selected metric.
ALL_METRICS = {
    "auroc": ("AUROC", True),
    "fpr95": ("FPR95", False),
    "fpr99": ("FPR99", False),
    "fpr100": ("FPR100", False),
}
DATASET_ORDER = ["MNIST", "CIFAR-10", "CIFAR-100", "ESC-50", "Speech Commands", "UrbanSound8K", "Wireless"]


def rank_cols(vals, higher_is_better):
    """Indices tied for best and for second-best, ranked on the DISPLAYED
    (2-decimal-rounded) value — so cells that print identically are always
    marked identically, instead of an arbitrary index-order tiebreak."""
    rounded = {i: round(v, 2) for i, v in enumerate(vals) if v is not None}
    if not rounded:
        return set(), set()
    uniq = sorted(set(rounded.values()), reverse=higher_is_better)
    best_val = uniq[0]
    second_val = uniq[1] if len(uniq) >= 2 else None
    best = {i for i, v in rounded.items() if v == best_val}
    second = {i for i, v in rounded.items() if second_val is not None and v == second_val}
    return best, second


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table", help="results/tables/AUROC_FPR95.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics", default="auroc,fpr95,fpr99,fpr100",
                    help="Comma-separated subset of auroc,fpr95,fpr99,fpr100 to render "
                         "as side-by-side blocks, in order.")
    a = ap.parse_args(argv)

    METRICS = [(key, *ALL_METRICS[key]) for key in a.metrics.split(",")]

    import matplotlib.pyplot as plt

    d = json.load(open(a.table))
    rows = d["rows"]
    groups: list[tuple[str, list[dict]]] = []
    for ds in DATASET_ORDER:
        rs = [r for r in rows if r["dataset"] == ds]
        if rs:
            groups.append((ds, rs))

    # DIM was a #444 grey for the sub-headers; hard to read, so it is plain black now.
    INK, DIM, LINE = "#000000", "#000000", "#000000"

    n_sig = len(SIGNALS)
    n_metrics = len(METRICS)
    arm_w = 2.25
    val_w = 0.62               # one signal's cell width within a metric block
    block_w = n_sig * val_w    # one metric block (all 4 signals)
    block_gap = 0.34           # visual gap between adjacent metric blocks
    total_w = arm_w + n_metrics * block_w + (n_metrics - 1) * block_gap
    row_h = 0.27
    grp_gap = 0.22
    top_h = 0.62      # metric-block header row (AUROC / FPR95 spanning the 4 signals)
    sub_h = 0.46      # per-signal sub-header row (tall enough for the 3-line label)

    n_data_rows = len(rows)
    n_grp = len(groups)
    fig_w = total_w + 0.3
    # The Axes spans the ENTIRE figure canvas (add_axes([0,0,1,1]) below), so
    # bbox_inches="tight" has no outer margin to trim — it only crops space
    # around a smaller axes, not unused space *inside* one that already fills
    # the canvas. fig_h must therefore match the drawn content almost exactly,
    # not just be "big enough": 0.05 top margin (before the top rule) + header
    # + one row_h per data row + grp_gap*1.5 per group (label lead-in + pre-row
    # gap + post-row gap before the separator — matches the drawing loop below
    # exactly) + a small fixed bottom margin below the final rule.
    fig_h = 0.05 + top_h + sub_h + n_data_rows * row_h + n_grp * grp_gap * 1.5 + 0.06

    rc = {
        "font.family": "DejaVu Serif", "font.size": 8.2,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=220)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.set_xlim(0, total_w)
        ax.set_ylim(0, fig_h)
        ax.axis("off")

        block_x0 = {}
        x = arm_w
        for key, _, _ in METRICS:
            block_x0[key] = x
            x += block_w + block_gap

        y = fig_h - 0.05

        # top booktabs rule
        ax.plot([0, total_w], [y, y], color=INK, lw=1.3)
        y -= top_h

        # metric-block spanning headers ("AUROC" / "FPR95"), each over its 4
        # signal columns — the side-by-side-blocks layout, one metric per block.
        for key, label, _ in METRICS:
            x0 = block_x0[key]
            cx = x0 + block_w / 2
            ax.text(cx, y + top_h - 0.20, label, fontsize=9.0,
                    fontweight="bold", ha="center", va="top")
            ax.plot([x0 + 0.04, x0 + block_w - 0.04], [y + top_h - 0.34] * 2,
                    color=INK, lw=0.6)
        y_sub_top = y + top_h - 0.34 - 0.02

        # per-signal sub-header, repeated under each metric block
        for key, _, _ in METRICS:
            x = block_x0[key]
            for sig in SIGNALS:
                ax.text(x + val_w / 2, y_sub_top - sub_h / 2, SIGNAL_LABEL[sig],
                        fontsize=6.9, color=DIM, ha="center", va="center")
                x += val_w
        y = y_sub_top - sub_h
        ax.plot([0, total_w], [y, y], color=INK, lw=0.9)
        div_y_top = y

        def fmt(v):
            return "—" if v is None else f"{v:.2f}"

        for ds_name, ds_rows in groups:
            y -= grp_gap * 0.5
            ax.text(0.02, y - 0.02, ds_name, fontsize=7.6, fontweight="bold",
                    color=INK, va="top", fontfamily="DejaVu Sans")
            y -= grp_gap * 0.5
            for r in ds_rows:
                ry0 = y - row_h
                ax.text(0.02, y - row_h / 2, r["arm"], fontsize=7.9, color=INK,
                        va="center", ha="left")

                for key, _, higher_is_better in METRICS:
                    vals = [(r.get(s) or {}).get(key) for s in SIGNALS]
                    best_set, second_set = rank_cols(vals, higher_is_better)
                    x = block_x0[key]
                    for i, val in enumerate(vals):
                        cx = x + val_w / 2
                        txt = fmt(val)
                        fw = "bold" if i in best_set else "normal"
                        ax.text(cx, y - row_h / 2, txt, fontsize=7.9, color=INK,
                                ha="center", va="center", fontfamily="monospace",
                                fontweight=fw)
                        if i in second_set:
                            # manual underline under the text (matplotlib text
                            # underline via bbox is unreliable across backends)
                            tw = 0.10 * max(len(txt), 1)
                            ax.plot([cx - tw / 2, cx + tw / 2],
                                    [y - row_h / 2 - 0.085, y - row_h / 2 - 0.085],
                                    color=INK, lw=0.7)
                        x += val_w
                y = ry0
            y -= grp_gap * 0.5
            ax.plot([0, total_w], [y, y], color=LINE, lw=0.5, alpha=0.5)

        # bottom booktabs rule (replace the last faint group rule with a solid one)
        ax.plot([0, total_w], [y, y], color=INK, lw=1.1)

        # faint vertical dividers between adjacent metric blocks, bounded to
        # the actual table content — NOT down to y=0, or bbox_inches="tight"
        # pads the image with the fixed bottom margin as empty space.
        for i in range(1, n_metrics):
            div_x = arm_w + i * block_w + (i - 0.5) * block_gap
            ax.plot([div_x, div_x], [y, div_y_top], color=LINE, lw=0.5, alpha=0.35)

        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300, facecolor="white")
        plt.close(fig)
    print(f"wrote {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
