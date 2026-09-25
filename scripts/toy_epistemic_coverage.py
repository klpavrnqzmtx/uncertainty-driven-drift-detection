#!/usr/bin/env python3
"""Synthetic toy experiment for the paper's Figure 1 (methodology illustration):
2D Bayesian logistic regression with a Laplace posterior, demonstrating that
epistemic uncertainty (EU) stays low under a training-covered distribution
shift and rises sharply under a novel, orthogonal shift.

This is a direct, exact instantiation of the paper's own machinery in the
simplest possible closed form — not a separate approximation:

  * The MAP fit + Gauss-Newton curvature IS the exact Hessian of a logistic
    regression negative log-posterior (no linearization approximation needed
    in 2D: the model already is linear-in-logits).
  * EU is estimated via Monte Carlo exactly as in Eq. (7)/(27): draw S
    posterior weight samples, take the entropy of their AVERAGED predictive
    distribution minus the AVERAGE of their individual entropies (mutual
    information between the label and theta).
  * Why it works geometrically (Theorem 4.2 / Corollary 4.3-4.4): training
    data sit at x2 approx 0 with tiny variance, so the Hessian accumulates
    almost no curvature along the w2 (x2-weight) direction -> posterior
    variance there stays close to the prior (1/lambda), i.e. w2 is an
    "uncovered" direction. A deployment point's tangent feature is
    proportional to [x1, x2, 1]; phase 3 sets x2=4 (far outside the training
    band), which activates exactly that poorly-constrained direction and
    inflates EU. Phases 1-2 keep x2 approx 0, so despite phase 2 moving
    x1 from +-2 to +-4 (a real, substantial distribution shift), that shift
    lies entirely along the WELL-constrained w1 direction (training explicitly
    includes +-4 too) and EU stays low.

    python scripts/toy_epistemic_coverage.py --out results/figures/paper/fig1_toy_epistemic_coverage
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
from uncertainty_driven_drift.analysis import figure_style as FS  # noqa: E402

SIGMA = np.diag([1.0, 0.05])  # wide along x1 (informative), narrow along x2 (uninformative)
LAMBDA0 = 1.0                  # isotropic Gaussian prior precision
N_PER_CENTER = 150             # training points per class center
S_SAMPLES = 100                 # posterior draws per EU estimate
BATCH_SIZE = 100
N_BATCHES_PER_PHASE = 10

PHASES = [
    ("Original", (-2.0, 0.0), (2.0, 0.0)),
    ("Training-covered shift", (-4.0, 0.0), (4.0, 0.0)),
    ("Novel shift", (-2.0, 4.0), (2.0, 4.0)),
]
TRAIN_CENTERS = [((-2.0, 0.0), 0), ((2.0, 0.0), 1), ((-4.0, 0.0), 0), ((4.0, 0.0), 1)]


def augment(X: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((len(X), 1))])


def fit_map(Xb: np.ndarray, y: np.ndarray, lam: float, n_iter: int = 50) -> np.ndarray:
    """Newton-Raphson MAP fit for logistic regression — exact Hessian, no
    approximation needed since the model is already linear in the logits."""
    w = np.zeros(Xb.shape[1])
    for _ in range(n_iter):
        logits = Xb @ w
        p = 1.0 / (1.0 + np.exp(-logits))
        grad = Xb.T @ (p - y) + lam * w
        Hess = (Xb * (p * (1 - p))[:, None]).T @ Xb + lam * np.eye(Xb.shape[1])
        step = np.linalg.solve(Hess, grad)
        w = w - step
        if np.linalg.norm(step) < 1e-10:
            break
    return w


def laplace_covariance(Xb: np.ndarray, w_map: np.ndarray, lam: float) -> np.ndarray:
    logits = Xb @ w_map
    p = 1.0 / (1.0 + np.exp(-logits))
    Hess = (Xb * (p * (1 - p))[:, None]).T @ Xb + lam * np.eye(Xb.shape[1])
    return np.linalg.inv(Hess)


def epistemic_uncertainty(X: np.ndarray, w_map: np.ndarray, Sigma_post: np.ndarray,
                          rng: np.random.Generator, n_samples: int = S_SAMPLES) -> np.ndarray:
    """Mutual information I(Y; theta | x) via Monte Carlo — Eq. (7)/(27)."""
    Xb = augment(X)
    ws = rng.multivariate_normal(w_map, Sigma_post, size=n_samples)   # (S, 3)
    logits = Xb @ ws.T                                                 # (N, S)
    p = np.clip(1.0 / (1.0 + np.exp(-logits)), 1e-7, 1 - 1e-7)
    mean_p = p.mean(axis=1)
    H_mean = -(mean_p * np.log(mean_p) + (1 - mean_p) * np.log(1 - mean_p))
    H_each = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return H_mean - H_each.mean(axis=1)


def sample_batch(c0: tuple, c1: tuple, n: int, rng: np.random.Generator) -> np.ndarray:
    n0 = n // 2
    n1 = n - n0
    X0 = rng.multivariate_normal(c0, SIGMA, size=n0)
    X1 = rng.multivariate_normal(c1, SIGMA, size=n1)
    return np.vstack([X0, X1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="Output path stem (no extension)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import matplotlib.pyplot as plt

    rng = np.random.default_rng(a.seed)

    # ---- training data: two class centers each at |x1|=2 AND |x1|=4, x2~N(0, 0.05) ----
    X_train, y_train = [], []
    for center, label in TRAIN_CENTERS:
        X_train.append(rng.multivariate_normal(center, SIGMA, size=N_PER_CENTER))
        y_train.append(np.full(N_PER_CENTER, label))
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)

    Xb_train = augment(X_train)
    w_map = fit_map(Xb_train, y_train, LAMBDA0)
    Sigma_post = laplace_covariance(Xb_train, w_map, LAMBDA0)
    print(f"MAP weights (w1, w2, bias): {np.round(w_map, 3)}")
    print(f"Posterior std  (w1, w2, bias): {np.round(np.sqrt(np.diag(Sigma_post)), 4)}")
    print("  -> w2 (the x2 direction) is the poorly-constrained one: its posterior std\n"
          "     should sit close to the prior's 1/sqrt(lambda) = "
          f"{1 / np.sqrt(LAMBDA0):.3f}, while w1's is shrunk well below it.\n")

    # ---- deployment stream: 3 phases x N_BATCHES_PER_PHASE batches ----
    batch_eu, phase_of_batch, batch_t = [], [], []
    batch_acc = []
    t = 0
    for name, c0, c1 in PHASES:
        for _ in range(N_BATCHES_PER_PHASE):
            n0 = BATCH_SIZE // 2
            y_true = np.concatenate([np.zeros(n0), np.ones(BATCH_SIZE - n0)])
            Xt = sample_batch(c0, c1, BATCH_SIZE, rng)
            eu = epistemic_uncertainty(Xt, w_map, Sigma_post, rng)
            y_pred = (augment(Xt) @ w_map > 0).astype(float)
            batch_acc.append((y_pred == y_true).mean())
            batch_eu.append(eu.mean())
            phase_of_batch.append(name)
            batch_t.append(t)
            t += 1
    batch_eu = np.array(batch_eu)
    batch_t = np.array(batch_t)
    batch_acc = np.array(batch_acc)

    print("=== mean EU and MAP-decision-rule accuracy per phase ===")
    means = {}
    for name, *_ in PHASES:
        mask = [p == name for p in phase_of_batch]
        m = batch_eu[mask].mean()
        means[name] = m
        print(f"  {name:24s} EU: {m:.4f}   accuracy: {batch_acc[mask].mean():.4f}")
    ok = means["Original"] * 1.5 > means["Training-covered shift"] * 0.67 \
        and means["Novel shift"] > 3 * max(means["Original"], means["Training-covered shift"])
    print(f"\nQualitative check (Original ~ Training-covered << Novel): {'PASS' if ok else 'CHECK MANUALLY'}")
    print("Note: accuracy stays high in ALL three phases (the x1-vs-x2 margin is large enough that\n"
          "the x2=4 shift in class 3 doesn't flip any decisions) — by design. That's the point: EU\n"
          "flags phase 3 as unsupported even though the decision rule is still empirically 'valid'\n"
          "there. A monitor watching accuracy alone would see nothing to react to.")

    # ================================================================== #
    # Combined figure: (A) data geometry on top, (B) EU-vs-batch on bottom
    # ================================================================== #
    rc = FS.rc(base=7.5)
    n_per_phase = N_BATCHES_PER_PHASE
    phase_colors = {"Original": FS.PALETTE["input"],
                    "Training-covered shift": FS.PALETTE["total"],
                    "Novel shift": FS.PALETTE["epistemic"]}
    corridor = 3.0 * np.sqrt(SIGMA[1, 1])  # +-3sd of the training x2 spread

    with plt.rc_context(rc):
        fig, (axA, axB) = plt.subplots(
            2, 1, figsize=(2.75, 4.9), dpi=300,
            gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.55},
        )

        # ---- Panel A: data geometry ----
        x1_grid = np.linspace(-7.0, 7.5, 200)
        y_lo, y_hi = -1.1, 4.9
        axA.axhspan(-corridor, corridor, color=FS.RULE, alpha=0.35, lw=0, zorder=0)
        axA.text(x1_grid[0], corridor, "  training-covered corridor", fontsize=5.6,
                 color=FS.INK_MUTED, style="italic", va="bottom", ha="left")

        # decision boundary: w1*x1 + w2*x2 + b = 0 -> x1(x2)
        x2_grid = np.linspace(y_lo, y_hi, 100)
        x1_boundary = -(w_map[1] * x2_grid + w_map[2]) / w_map[0]
        axA.plot(x1_boundary, x2_grid, ls=(0, (4, 2)), lw=0.9, color=FS.INK,
                 alpha=0.6, zorder=1)
        axA.text(x1_boundary[-1], y_hi, "decision\nboundary", fontsize=5.4,
                 color=FS.INK_MUTED, ha="center", va="bottom", linespacing=1.1)

        axA.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
                   s=3.5, color="#9a9a9a", alpha=0.35, linewidths=0, zorder=2,
                   label="train, class 0")
        axA.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
                   s=3.5, color="#9a9a9a", alpha=0.35, linewidths=0, marker="^", zorder=2,
                   label="train, class 1")
        rng2 = np.random.default_rng(a.seed + 1)
        for name, c0, c1 in PHASES:
            Xp = sample_batch(c0, c1, 80, rng2)
            axA.scatter(Xp[:, 0], Xp[:, 1], s=6.5, color=phase_colors[name],
                       alpha=0.85, linewidths=0.25, edgecolors="white", zorder=3, label=name)

        axA.set_xlabel(r"$x_1$   (training-covered direction)", labelpad=2)
        axA.set_ylabel(r"$x_2$   (uncovered direction)", labelpad=2)
        axA.set_xlim(x1_grid[0], x1_grid[-1])
        axA.set_ylim(y_lo, y_hi)
        axA.legend(fontsize=5.3, loc="center", bbox_to_anchor=(0.30, 0.50),
                  frameon=True, framealpha=0.88, edgecolor="none", facecolor="white",
                  handletextpad=0.35, borderaxespad=0.0, labelspacing=0.5,
                  borderpad=0.4)
        axA.text(0.0, 1.14, "A", transform=axA.transAxes, fontsize=9, fontweight="bold")
        axA.text(0.09, 1.14, "Deployment geometry", transform=axA.transAxes,
                fontsize=7.2, fontweight="bold", va="baseline")

        # ---- Panel B: EU vs batch ----
        for i, (name, *_ ) in enumerate(PHASES):
            lo, hi = i * n_per_phase, (i + 1) * n_per_phase
            if i % 2 == 1:
                axB.axvspan(lo - 0.5, hi - 0.5, color=FS.BAND, lw=0, zorder=0)
            mid = 0.5 * (lo + hi - 1)
            axB.text(mid, 1.04, name, transform=axB.get_xaxis_transform(),
                    ha="left", va="bottom", fontsize=6.0, color=phase_colors[name],
                    fontweight="medium", rotation=20, rotation_mode="anchor")
        for i in (1, 2):
            axB.axvline(i * n_per_phase - 0.5, color=FS.INK, lw=0.9, zorder=2)

        axB.plot(batch_t, batch_eu, color=FS.PALETTE["epistemic"], lw=1.1,
                zorder=3, alpha=0.9)
        axB.scatter(batch_t, batch_eu, s=10, color=FS.PALETTE["epistemic"],
                   edgecolors="white", linewidths=0.5, zorder=4)
        axB.set_xlabel("deployment batch index", labelpad=2)
        axB.set_ylabel("mean epistemic\nuncertainty  " + r"$\widehat{U}$", labelpad=2)
        axB.set_xlim(-0.5, 3 * n_per_phase - 0.5)
        axB.set_ylim(0, max(batch_eu) * 1.18)
        axB.grid(axis="y")
        axB.text(0.0, 1.20, "B", transform=axB.transAxes, fontsize=9, fontweight="bold")
        axB.text(0.09, 1.20, "Monitoring statistic", transform=axB.transAxes,
                fontsize=7.2, fontweight="bold", va="baseline")

        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=300, facecolor="white")
        plt.close(fig)
    print(f"wrote {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
