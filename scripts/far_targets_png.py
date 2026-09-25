#!/usr/bin/env python3
"""Render the matched-FAR delay comparison (results/tables/FAR_TARGETS.json) as a
publication-style PNG table, grouped by dataset, with the oracle/epistemic/total/
input columns and green/amber cells showing who beats the supervised baseline.

    python scripts/far_targets_png.py results/tables/FAR_TARGETS.json \
        --out results/figures/paper/far_targets_summary --far 0.05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PRETTY_ARM = {
    "known_vs_novel_resnet": "ResNet-20 MC",
    "known_vs_novel_resnet_laplace": "ResNet-20 Laplace",
    "known_vs_novel_vit": "ViT MC",
    "known_vs_novel_vit_laplace": "ViT Laplace",
    "known_vs_novel_laplace": "ResNet-20 Laplace",
    "severity_shift_resnet": "ResNet-20 MC (severity)",
    "severity_shift_resnet_laplace": "ResNet-20 Laplace (severity)",
    "known_vs_novel_wrn_mc_dropout": "WRN-28-10 MC",
    "known_vs_novel_wrn_laplace": "WRN-28-10 Laplace",
    "known_vs_novel_vit_mc_dropout": "ViT-B/16 MC",
}
DATASET_PREFIX = [("mnist_c__", "MNIST"), ("cifar_kn_comparison__", "CIFAR-10"),
                  ("cifar100_c__", "CIFAR-100"), ("imagenet_comparison__", "ImageNet")]


def load_rows(path: Path, far: float):
    d = json.load(open(path))
    if far not in d["targets"]:
        raise SystemExit(f"--far {far} not in the table's targets {d['targets']}")
    key = f"delay@{far}"
    rows = {(r["experiment"], r["detector"]): r.get(key) for r in d["rows"]}

    vision = [e for e, _ in rows if any(e.startswith(p) for p, _ in DATASET_PREFIX)]
    vision = sorted(set(vision))

    def ds_of(exp):
        return next(nm for pre, nm in DATASET_PREFIX if exp.startswith(pre))

    order = {nm: i for i, (_, nm) in enumerate(DATASET_PREFIX)}
    vision.sort(key=lambda e: (order[ds_of(e)], e))

    out = []
    for exp in vision:
        ds = ds_of(exp)
        arm_key = exp.split("__", 1)[1]
        if arm_key == "known_vs_novel":
            arm = "LeNet MC" if ds == "MNIST" else "ResNet-20 MC"
        else:
            arm = PRETTY_ARM.get(arm_key, arm_key)
        out.append(dict(
            dataset=ds, arm=arm,
            ddm=rows.get((exp, "ddm")), epi=rows.get((exp, "ph_epistemic")),
            tot=rows.get((exp, "ph_total")), inp=rows.get((exp, "ph_input")),
        ))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table", help="results/tables/FAR_TARGETS.json")
    ap.add_argument("--out", required=True, help="Output path stem (no extension)")
    ap.add_argument("--far", type=float, default=0.05, help="Which FAR-budget column to render.")
    a = ap.parse_args(argv)

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    rows = load_rows(Path(a.table), a.far)

    # DIM was a slate grey for secondary text; hard to read, so it now matches INK.
    INK, DIM, LINE = "#171b26", "#171b26", "#dfe3ee"
    GOOD, BAD, ACCENT = "#0d8f6f", "#b45309", "#4f46bd"
    GOOD_BG, BAD_BG = "#e3f4ee", "#fdf1e0"
    GRP_BG = "#eef0f8"

    # group rows by dataset, inserting a header band before each
    groups: list[tuple[str, list[dict]]] = []
    for r in rows:
        if not groups or groups[-1][0] != r["dataset"]:
            groups.append((r["dataset"], []))
        groups[-1][1].append(r)

    row_h = 0.30
    grp_h = 0.26
    header_h = 0.34
    n_data_rows = len(rows)
    n_grp_rows = len(groups)
    body_h = n_grp_rows * grp_h + n_data_rows * row_h
    fig_w = 7.4
    total_w = 2.7 + 1.5 * 4  # matches table 1's column widths (widest table)

    block1_h = 0.72 + header_h + body_h + 0.50 + 0.34
    block2_h = 0.62 + header_h + body_h + 0.34 + 0.22
    fig_h = 0.5 + block1_h + 0.35 + block2_h + 0.35

    def fmt(v):
        return "—" if v is None else f"{v:.1f}"

    def draw_table(ax, y, headers, col_w, keys, color_fn):
        x = 0.0
        header_y0 = y - header_h
        for c, w in zip(headers, col_w):
            ha = "left" if c == headers[0] else "right"
            tx = x + 0.06 if ha == "left" else x + w - 0.06
            ax.text(tx, y - header_h / 2, c, fontsize=7.3, fontweight="bold",
                    color=DIM, ha=ha, va="center", fontfamily="monospace")
            x += w
        ax.plot([0, total_w], [header_y0, header_y0], color=INK, lw=1.1,
                solid_capstyle="butt")
        y = header_y0

        for ds_name, ds_rows in groups:
            gy0 = y - grp_h
            ax.add_patch(mpatches.Rectangle((0, gy0), total_w, grp_h,
                                            facecolor=GRP_BG, edgecolor="none"))
            ax.text(0.06, y - grp_h / 2, ds_name, fontsize=7.6, fontweight="bold",
                    color=INK, va="center", fontfamily="monospace")
            y = gy0
            for r in ds_rows:
                ry0 = y - row_h
                x = 0.0
                ax.text(x + 0.06, y - row_h / 2, r["arm"], fontsize=7.9, color=INK,
                        va="center", ha="left", fontweight="medium")
                x += col_w[0]
                for key, w in zip(keys, col_w[1:]):
                    v = r[key]
                    cell_color, bg = color_fn(r, key)
                    if bg:
                        ax.add_patch(mpatches.Rectangle((x + 0.05, ry0 + 0.02),
                                                        w - 0.10, row_h - 0.04,
                                                        facecolor=bg, edgecolor="none"))
                    fw = "bold" if bg else "normal"
                    ax.text(x + w - 0.08, y - row_h / 2, fmt(v), fontsize=8.0,
                            color=cell_color, va="center", ha="right",
                            fontfamily="monospace", fontweight=fw)
                    x += w
                ax.plot([0, total_w], [ry0, ry0], color=LINE, lw=0.6)
                y = ry0
        return y

    def color_vs_ddm(r, key):
        v, ddm = r[key], r["ddm"]
        if key != "ddm" and v is not None and ddm is not None:
            if v < ddm:
                return GOOD, GOOD_BG
            if v > ddm:
                return BAD, BAD_BG
        return INK, None

    def color_best_of_three(r, key):
        vals = [r[k] for k in ("epi", "tot", "inp") if r[k] is not None]
        v = r[key]
        if v is not None and vals and v == min(vals):
            return GOOD, GOOD_BG
        return INK, None

    rc = {
        "font.family": "DejaVu Sans", "font.size": 8.3,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=220)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.set_xlim(0, total_w)
        ax.set_ylim(0, fig_h)
        ax.axis("off")

        y = fig_h - 0.45

        # ---- Table 1: all four signals, colored against the DDM oracle ----
        ax.text(0, y, "Detection delay at a matched false-alarm budget",
                fontsize=12.5, fontweight="bold", color=INK, va="top")
        y -= 0.30
        ax.text(0, y, f"Every detector swept to the same FAR ≤ {a.far:g} on the known "
                       "phase before delay (batches) is compared.  Lower is better.",
                fontsize=7.6, color=DIM, va="top")
        y -= 0.42

        y = draw_table(
            ax, y,
            # Two lines: one-line form is ~1.8in in a 1.5in column; fits header_h.
            headers=["arm", "Error-based signal\n(labels) †", "epistemic", "total", "input"],
            col_w=[2.7, 1.5, 1.5, 1.5, 1.5],
            keys=["ddm", "epi", "tot", "inp"],
            color_fn=color_vs_ddm,
        )

        y -= 0.20
        ax.add_patch(mpatches.Rectangle((0.0, y - 0.14), 0.20, 0.14,
                                        facecolor=GOOD_BG, edgecolor=GOOD, lw=0.8))
        ax.text(0.28, y - 0.07, "faster than the oracle", fontsize=7.2, color=DIM, va="center")
        ax.add_patch(mpatches.Rectangle((2.6, y - 0.14), 0.20, 0.14,
                                        facecolor=BAD_BG, edgecolor=BAD, lw=0.8))
        ax.text(2.88, y - 0.07, "slower than the oracle", fontsize=7.2, color=DIM, va="center")
        y -= 0.30
        ax.text(0, y, "† The error-based signal (labels) is the supervised oracle: it monitors true prediction "
                       "errors, the labels the other signals never observe.",
                fontsize=6.8, color=DIM, va="top", style="italic")

        # ---- Table 2: label-free signals only, best of the three highlighted ----
        y -= 0.55
        ax.text(0, y, "Unsupervised signals only", fontsize=11, fontweight="bold",
                color=INK, va="top")
        y -= 0.26
        ax.text(0, y, "Same budget and data, with the error-based signal removed — epistemic, total "
                      "entropy and input distance compared head-to-head.",
                fontsize=7.4, color=DIM, va="top")
        y -= 0.36

        value_w = (total_w - 2.7) / 3
        y = draw_table(
            ax, y,
            headers=["arm", "epistemic", "total", "input"],
            col_w=[2.7, value_w, value_w, value_w],
            keys=["epi", "tot", "inp"],
            color_fn=color_best_of_three,
        )

        y -= 0.20
        ax.add_patch(mpatches.Rectangle((0.0, y - 0.14), 0.20, 0.14,
                                        facecolor=GOOD_BG, edgecolor=GOOD, lw=0.8))
        ax.text(0.28, y - 0.07, "fastest of the three label-free signals",
                fontsize=7.2, color=DIM, va="center")

        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300,
                   facecolor="white")
        plt.close(fig)
    print(f"wrote {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
