#!/usr/bin/env python3
"""Three genuinely different layout options for the Figure-1 toy experiment,
aiming for a cleaner, more restrained academic look (conference main-text
style) than the single crowded two-panel version. Same underlying model and
data as scripts/toy_epistemic_coverage.py (duplicated here, not imported,
since this is a throwaway comparison script — delete once one option wins).

  A: minimal two-panel, muted grey/blue palette, no on-plot legend or
     annotations (colored words in the subtitle do the labeling instead)
  B: small multiples — one mini scatter per phase, side by side, classic
     "before / during / after" toy-example layout, + one EU strip below
  C: single hero plot — EU trace only, geometry dropped to the caption/text

    python scripts/toy_epistemic_coverage_options.py --out results/figures/paper/fig1_toy_options
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
from uncertainty_driven_drift.analysis import figure_style as FS  # noqa: E402

SIGMA = np.diag([1.0, 0.05])
LAMBDA0 = 1.0
N_PER_CENTER = 150
S_SAMPLES = 100
BATCH_SIZE = 100
N_BATCHES_PER_PHASE = 10

PHASES = [
    ("Original", (-2.0, 0.0), (2.0, 0.0)),
    ("Training-covered shift", (-4.0, 0.0), (4.0, 0.0)),
    ("Novel shift", (-2.0, 4.0), (2.0, 4.0)),
]
TRAIN_CENTERS = [((-2.0, 0.0), 0), ((2.0, 0.0), 1), ((-4.0, 0.0), 0), ((4.0, 0.0), 1)]

# Muted, meaning-coded palette: grey = "nothing interesting happening" for the
# two low-EU phases, the paper's own epistemic-blue reserved for the one phase
# that actually matters. Not three arbitrary hues.
PHASE_COLOR = {"Original": "#b0b0b0", "Training-covered shift": "#6e6e6e",
              "Novel shift": FS.PALETTE["epistemic"]}

# Option B (and its compact variant) no longer use the greys above: measured at
# 1.4-2.2:1 contrast on white they are unreadable. Points wear the paper palette
# instead -- the two classes of the original data in red/blue, the novel shift in
# the epistemic green, so the novel batch in panel A and the EU trace in panel B
# share one colour. Red vs green collapses under deuteranopia (OKLab dE 3.9), so
# the novel batch also gets its own marker; its position (x2 = 4) separates it too.
CLASS_COLOR = {0: FS.PALETTE["input"], 1: FS.PALETTE["supervised"]}   # red, blue
NOVEL_COLOR = FS.PALETTE["epistemic"]                                  # green
NOVEL_MARKER = "D"


def scatter_training(ax, d, s, alpha=0.3):
    """Training data as a faint backdrop, coloured by class."""
    for k, col in CLASS_COLOR.items():
        m = d["y_train"] == k
        ax.scatter(d["X_train"][m, 0], d["X_train"][m, 1], s=s, color=col,
                   alpha=alpha, linewidths=0, zorder=1)


def scatter_batch(ax, name, Xp, s):
    """One phase's batch, solid. sample_batch() stacks class 0 then class 1."""
    if name == "Novel shift":
        ax.scatter(Xp[:, 0], Xp[:, 1], s=s * 0.8, color=NOVEL_COLOR, marker=NOVEL_MARKER,
                   alpha=0.9, linewidths=0, zorder=2)
        return
    h = len(Xp) // 2
    for k, part in ((0, Xp[:h]), (1, Xp[h:])):
        ax.scatter(part[:, 0], part[:, 1], s=s, color=CLASS_COLOR[k], alpha=0.9,
                   linewidths=0, zorder=2)


def class_legend(ax, fontsize, ms):
    """Key for the red/blue/green encoding -- colour alone would not say it."""
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker="o", ms=ms, color=CLASS_COLOR[0], label="class 0"),
         Line2D([], [], ls="", marker="o", ms=ms, color=CLASS_COLOR[1], label="class 1"),
         Line2D([], [], ls="", marker=NOVEL_MARKER, ms=ms * 0.85, color=NOVEL_COLOR,
                label="novel shift")]
    ax.legend(handles=h, loc="upper left", fontsize=fontsize, frameon=False,
              handlelength=0.8, handletextpad=0.2, labelspacing=0.2, borderaxespad=0.15)


def augment(X):
    return np.hstack([X, np.ones((len(X), 1))])


def fit_map(Xb, y, lam, n_iter=50):
    w = np.zeros(Xb.shape[1])
    for _ in range(n_iter):
        p = 1.0 / (1.0 + np.exp(-(Xb @ w)))
        grad = Xb.T @ (p - y) + lam * w
        Hess = (Xb * (p * (1 - p))[:, None]).T @ Xb + lam * np.eye(Xb.shape[1])
        step = np.linalg.solve(Hess, grad)
        w = w - step
        if np.linalg.norm(step) < 1e-10:
            break
    return w


def laplace_covariance(Xb, w_map, lam):
    p = 1.0 / (1.0 + np.exp(-(Xb @ w_map)))
    Hess = (Xb * (p * (1 - p))[:, None]).T @ Xb + lam * np.eye(Xb.shape[1])
    return np.linalg.inv(Hess)


def epistemic_uncertainty(X, w_map, Sigma_post, rng, n_samples=S_SAMPLES):
    Xb = augment(X)
    ws = rng.multivariate_normal(w_map, Sigma_post, size=n_samples)
    logits = Xb @ ws.T
    p = np.clip(1.0 / (1.0 + np.exp(-logits)), 1e-7, 1 - 1e-7)
    mean_p = p.mean(axis=1)
    H_mean = -(mean_p * np.log(mean_p) + (1 - mean_p) * np.log(1 - mean_p))
    H_each = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return H_mean - H_each.mean(axis=1)


def sample_batch(c0, c1, n, rng):
    n0 = n // 2
    X0 = rng.multivariate_normal(c0, SIGMA, size=n0)
    X1 = rng.multivariate_normal(c1, SIGMA, size=n - n0)
    return np.vstack([X0, X1])


def compute_all(seed):
    rng = np.random.default_rng(seed)
    X_train, y_train = [], []
    for center, label in TRAIN_CENTERS:
        X_train.append(rng.multivariate_normal(center, SIGMA, size=N_PER_CENTER))
        y_train.append(np.full(N_PER_CENTER, label))
    X_train, y_train = np.vstack(X_train), np.concatenate(y_train)
    Xb_train = augment(X_train)
    w_map = fit_map(Xb_train, y_train, LAMBDA0)
    Sigma_post = laplace_covariance(Xb_train, w_map, LAMBDA0)

    batch_eu, phase_of_batch, batch_t = [], [], []
    t = 0
    for name, c0, c1 in PHASES:
        for _ in range(N_BATCHES_PER_PHASE):
            Xt = sample_batch(c0, c1, BATCH_SIZE, rng)
            eu = epistemic_uncertainty(Xt, w_map, Sigma_post, rng)
            batch_eu.append(eu.mean())
            phase_of_batch.append(name)
            batch_t.append(t)
            t += 1
    return dict(X_train=X_train, y_train=y_train, w_map=w_map, Sigma_post=Sigma_post,
               batch_eu=np.array(batch_eu), batch_t=np.array(batch_t),
               phase_of_batch=phase_of_batch, rng=rng)


def eu_panel(ax, d, rotate_labels=False, short_labels=False):
    n = N_BATCHES_PER_PHASE
    label_map = {"Original": "Original", "Training-covered shift": "Covered shift",
                "Novel shift": "Novel shift"} if short_labels else {n: n for n, *_ in PHASES}
    for i, (name, *_ ) in enumerate(PHASES):
        lo, hi = i * n, (i + 1) * n
        if i % 2 == 1:
            ax.axvspan(lo - 0.5, hi - 0.5, color=FS.BAND, lw=0, zorder=0)
        mid = 0.5 * (lo + hi - 1)
        kw = dict(rotation=18, ha="left") if rotate_labels else dict(ha="center")
        ax.text(mid, 1.04, label_map[name], transform=ax.get_xaxis_transform(),
               va="bottom", fontsize=6.0, color=FS.INK, fontweight="medium", **kw)
    for i in (1, 2):
        ax.axvline(i * n - 0.5, color=FS.INK, lw=0.8, zorder=2, alpha=0.7)
    ax.plot(d["batch_t"], d["batch_eu"], color=FS.PALETTE["epistemic"], lw=1.1, zorder=3)
    ax.scatter(d["batch_t"], d["batch_eu"], s=9, color=FS.PALETTE["epistemic"],
              edgecolors="white", linewidths=0.5, zorder=4)
    ax.set_xlabel("deployment batch index", labelpad=2)
    ax.set_ylabel(r"mean EU  $\widehat{U}$", labelpad=2)
    ax.set_xlim(-0.5, 3 * n - 0.5)
    ax.set_ylim(0, max(d["batch_eu"]) * 1.18)
    ax.grid(axis="y")


def option_a(d, out):
    """Minimal two-panel: muted grey/blue, no legend, colored-word subtitle."""
    rc = FS.rc(base=7.5)
    with plt.rc_context(rc):
        fig, (axA, axB) = plt.subplots(
            2, 1, figsize=(2.7, 4.1), dpi=300,
            gridspec_kw={"height_ratios": [1.2, 1.0], "hspace": 0.45},
        )
        rng2 = np.random.default_rng(1)
        axA.scatter(d["X_train"][:, 0], d["X_train"][:, 1], s=3, color="#d8d8d8",
                   alpha=0.7, linewidths=0, zorder=1)
        for name, c0, c1 in PHASES:
            Xp = sample_batch(c0, c1, 80, rng2)
            axA.scatter(Xp[:, 0], Xp[:, 1], s=6, color=PHASE_COLOR[name],
                       alpha=0.9, linewidths=0, zorder=2)
        x2_grid = np.linspace(-1.1, 4.9, 50)
        w = d["w_map"]
        x1_boundary = -(w[1] * x2_grid + w[2]) / w[0]
        axA.plot(x1_boundary, x2_grid, ls=(0, (3, 2)), lw=0.8, color=FS.INK, alpha=0.35, zorder=0)
        axA.set_xlabel(r"$x_1$", labelpad=1)
        axA.set_ylabel(r"$x_2$", labelpad=1)
        axA.set_xlim(-7, 7.5)
        axA.set_ylim(-1.1, 4.9)

        # colored-word "legend" as a title line, doing double duty as caption
        y0 = 1.06
        parts = [("training  ", "#c8c8c8"), ("Original  ", PHASE_COLOR["Original"]),
                (" → ", FS.INK_MUTED), ("Covered  ", PHASE_COLOR["Training-covered shift"]),
                (" → ", FS.INK_MUTED), ("Novel", PHASE_COLOR["Novel shift"])]
        xt = 0.0
        renderer = fig.canvas.get_renderer()
        for text, color in parts:
            txt_obj = axA.text(xt, y0, text, transform=axA.transAxes, fontsize=6.6,
                              color=color, fontweight="bold", va="bottom", ha="left")
            bbox = txt_obj.get_window_extent(renderer=renderer)
            bbox_axes = bbox.transformed(axA.transAxes.inverted())
            xt = bbox_axes.x1

        eu_panel(axB, d, rotate_labels=False, short_labels=True)
        axA.text(-0.22, 1.06, "A", transform=axA.transAxes, fontsize=9, fontweight="bold")
        axB.text(-0.22, 1.20, "B", transform=axB.transAxes, fontsize=9, fontweight="bold")

        fig.savefig(out / "option_A_minimal.png", bbox_inches="tight", dpi=300, facecolor="white")
        plt.close(fig)


def option_b(d, out):
    """Small multiples: one mini scatter per phase, + EU strip below."""
    rc = FS.rc(base=7.0)
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(5.4, 3.0), dpi=300)
        gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.55, wspace=0.12)
        rng2 = np.random.default_rng(1)
        for i, (name, c0, c1) in enumerate(PHASES):
            ax = fig.add_subplot(gs[0, i])
            scatter_training(ax, d, s=2.2)
            Xp = sample_batch(c0, c1, 80, rng2)
            scatter_batch(ax, name, Xp, s=5.5)
            ax.set_title(name, fontsize=6.6, color=FS.INK, fontweight="bold", pad=3)
            if i == 0:
                class_legend(ax, fontsize=5.2, ms=2.6)
            ax.set_xlim(-7, 7.5)
            ax.set_ylim(-1.1, 4.9)
            ax.set_xticks([-5, 0, 5])
            if i == 0:
                ax.set_ylabel(r"$x_2$", labelpad=1)
                ax.set_yticks([0, 2, 4])
            else:
                ax.set_yticks([0, 2, 4])
                ax.set_yticklabels([])
            ax.set_xlabel(r"$x_1$", labelpad=1, fontsize=6.0)

        axB = fig.add_subplot(gs[1, :])
        eu_panel(axB, d, rotate_labels=False, short_labels=True)
        fig.text(0.005, 0.97, "A", fontsize=9, fontweight="bold")
        fig.text(0.005, 0.44, "B", fontsize=9, fontweight="bold")

        fig.savefig(out / "option_B_small_multiples.png", bbox_inches="tight", dpi=300, facecolor="white")
        plt.close(fig)


def option_c(d, out):
    """Single hero plot: EU trace only, large and clean, geometry dropped."""
    rc = FS.rc(base=8.5)
    with plt.rc_context(rc):
        fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.1), dpi=300)
        eu_panel(ax, d, rotate_labels=False, short_labels=True)
        fig.savefig(out / "option_C_hero_plot.png", bbox_inches="tight", dpi=300, facecolor="white")
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

    print("=== mean EU per phase ===")
    for name, *_ in PHASES:
        mask = [p == name for p in d["phase_of_batch"]]
        print(f"  {name:24s} EU: {d['batch_eu'][mask].mean():.4f}")

    option_a(d, out)
    option_b(d, out)
    option_c(d, out)
    print(f"\nwrote option_A_minimal.png, option_B_small_multiples.png, "
          f"option_C_hero_plot.png -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
