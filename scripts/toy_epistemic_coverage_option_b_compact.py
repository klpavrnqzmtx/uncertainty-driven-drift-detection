#!/usr/bin/env python3
"""A more compact version of Option B (small multiples) from
toy_epistemic_coverage_options.py — same layout (3 mini scatters + EU strip),
shrunk for tighter placement in a paper column. Reuses the existing compute
code and eu_panel() via import rather than duplicating it; only defines a new,
smaller-footprint option_b_compact().

    python scripts/toy_epistemic_coverage_option_b_compact.py \
        --out results/figures/paper/fig1_toy_options
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))
from uncertainty_driven_drift.analysis import figure_style as FS  # noqa: E402
from toy_epistemic_coverage_options import (  # noqa: E402
    PHASES, class_legend, compute_all, eu_panel, sample_batch, scatter_batch,
    scatter_training,
)


def option_b_compact(d, out):
    """Small multiples, shrunk: smaller canvas, tighter spacing, smaller type."""
    rc = FS.rc(base=5.5)
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(3.6, 2.05), dpi=300)
        gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.62, wspace=0.10)
        rng2 = np.random.default_rng(1)
        for i, (name, c0, c1) in enumerate(PHASES):
            ax = fig.add_subplot(gs[0, i])
            scatter_training(ax, d, s=1.2)
            Xp = sample_batch(c0, c1, 80, rng2)
            scatter_batch(ax, name, Xp, s=3.2)
            ax.set_title(name, fontsize=5.0, color=FS.INK, fontweight="bold", pad=2)
            if i == 0:
                class_legend(ax, fontsize=4.0, ms=2.0)
            ax.set_xlim(-7, 7.5)
            ax.set_ylim(-1.1, 4.9)
            ax.set_xticks([-5, 5])
            ax.tick_params(labelsize=4.3, pad=1, length=2)
            if i == 0:
                ax.set_yticks([0, 4])
            else:
                ax.set_yticks([0, 4])
                ax.set_yticklabels([])
            ax.set_xlabel(r"$x_1$", labelpad=0.5, fontsize=5.0)

        axB = fig.add_subplot(gs[1, :])
        eu_panel(axB, d, rotate_labels=False, short_labels=True)
        axB.tick_params(labelsize=4.3, pad=1, length=2)
        axB.xaxis.label.set_fontsize(5.0)
        # y label moved inside the axes to save horizontal space; the top-left
        # is empty because EU stays near zero until the novel phase.
        axB.set_ylabel("")
        axB.text(0.012, 0.96, r"mean EU  $\widehat{U}$", transform=axB.transAxes,
                 ha="left", va="top", fontsize=5.0, color=FS.INK,
                 bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
        # Panel letters hug the tick labels rather than the figure edge: with the
        # y labels gone, a letter at x=0.005 would hold open an empty strip under
        # bbox_inches="tight".
        x_letter = axB.get_position().x0 - 0.075
        fig.text(x_letter, 0.975, "A", fontsize=7, fontweight="bold")
        fig.text(x_letter, 0.42, "B", fontsize=7, fontweight="bold")

        fig.savefig(out / "option_B_small_multiples_compact.png", bbox_inches="tight",
                    dpi=300, facecolor="white")
        plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    global plt
    import matplotlib.pyplot as plt

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = compute_all(a.seed)

    option_b_compact(d, out)
    print(f"wrote option_B_small_multiples_compact.png -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
