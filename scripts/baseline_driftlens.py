#!/usr/bin/env python3
"""DRIFT-LENS-style embedding-distribution baseline, scored like Table 1.

The only model-free baseline in Table 1 is a two-moment summary of raw pixel
intensity (``mean_input`` = L2 of per-sample mean and std, batch-averaged),
which is a weak opponent. This adds the closest real competitor: an unsupervised,
model-aware detector that monitors the *embedding distribution* rather than
uncertainty, via a Frechet distance between the current batch and a reference
window -- the mechanism behind DRIFT LENS (Greco et al., 2024).

    d^2(N_ref, N_t) = ||mu_ref - mu_t||^2 + tr(S_ref + S_t - 2 (S_ref S_t)^{1/2})

Two deliberate deviations from the published method, both forced by the batch
sizes used here and both stated so the comparison is not overclaimed:

  * embeddings are projected onto the top ``--n-components`` principal
    directions of the REFERENCE window before the distance is computed. With
    B=64 samples in 768 dimensions the per-batch covariance is otherwise rank
    deficient and the distance is dominated by estimation noise rather than
    drift;
  * covariances are shrunk toward a scaled identity (Ledoit-Wolf style, fixed
    intensity ``--shrinkage``) for the same reason.

The reference window is the first ``--ref-batches`` batches of the KNOWN phase,
i.e. the detector is given a clean in-distribution baseline, which is the
favourable setting for it.

Output is the per-batch statistic plus AUROC / FPR95 / FPR99 / FPR100 computed
exactly as scripts/auroc_fpr95_table.py computes them, so the numbers drop
straight into Table 1 as an extra column.

    python scripts/baseline_driftlens.py \
        --config configs/experiments/cifar_kn_comparison/known_vs_novel_laplace.yaml \
        --seeds 0,1,2,3,4 --out results/tables/baselines/cifar_kn_laplace.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from uncertainty_driven_drift.components import phase3  # noqa: F401  (registers everything)
from uncertainty_driven_drift.config import load_config  # noqa: E402
from uncertainty_driven_drift.registry import build  # noqa: E402

FPR_LEVELS = [0.95, 0.99, 1.0]
FPR_KEYS = {0.95: "fpr95", 0.99: "fpr99", 1.0: "fpr100"}


def _shrunk_cov(X: np.ndarray, alpha: float) -> np.ndarray:
    S = np.cov(X, rowvar=False)
    S = np.atleast_2d(S)
    mu = np.trace(S) / S.shape[0]
    return (1.0 - alpha) * S + alpha * mu * np.eye(S.shape[0])


def _frechet(mu1, S1, mu2, S2) -> float:
    """Frechet (2-Wasserstein) distance between two Gaussians.

    tr((S1 S2)^{1/2}) is evaluated through the eigenvalues of S1 S2 rather than
    a matrix square root: after the PCA projection the matrices are small, and
    the eigenvalue route avoids scipy's sqrtm returning complex residue on
    near-singular products.
    """
    diff = mu1 - mu2
    ev = np.linalg.eigvals(S1 @ S2)
    tr_sqrt = float(np.sqrt(np.clip(ev.real, 0.0, None)).sum())
    return float(diff @ diff + np.trace(S1) + np.trace(S2) - 2.0 * tr_sqrt)


def _embeddings_for(model, dataset):
    """Per-batch penultimate embeddings, obtained by running the arm's own predict().

    Earlier this reimplemented each family's preprocessing by importing
    ``_normalize_batch`` from the model's module. That worked only for the arm
    it was tested against (CIFAR-10): MNIST's ``_normalize_batch`` takes a
    different signature, the pretrained ViT normalises inside ``_prep`` and
    resizes to 224 instead, and the scratch-ViT and audio arms expose neither.
    Feeding those families a raw batch produced embeddings the model never sees
    -- when it did not simply crash.

    Calling predict() instead makes preprocessing the model's problem, which is
    where it already correctly lives; see scripts/arm_features.py.
    """
    from arm_features import capture_features

    embs, idxs = [], []
    max_spread = [0.0]
    with capture_features(model) as cap:
        for batch in dataset:
            before = len(cap.feats)
            model.predict(batch)
            if len(cap.feats) == before:
                raise SystemExit(
                    f"{type(model).__name__} produced no embed() call during predict(); "
                    "cannot run the embedding-distribution baseline on this arm.")
            # A sampling predict() may call embed() once per posterior sample.
            # Usually those calls are identical -- the sampling happens in the
            # head, AFTER embed -- so averaging is a no-op. But the scratch-ViT
            # family puts Dropout *inside* its blocks, making embed stochastic
            # there, and picking one arbitrary draw would feed the Frechet
            # distance a sample of dropout noise. Average instead, and report
            # the spread so a stochastic backbone is visible rather than silent.
            got = cap.feats[before:]
            stacked = np.stack([g.astype(np.float64) for g in got], axis=0)
            spread = float(np.abs(stacked - stacked[0]).max()) if len(got) > 1 else 0.0
            max_spread[0] = max(max_spread[0], spread)
            embs.append(stacked.mean(axis=0))
            idxs.append(batch.index)
            cap.feats.clear()
    if max_spread[0] > 0.0:
        print(f"  [driftlens] embed() is stochastic across posterior samples "
              f"(max deviation {max_spread[0]:.3g}); averaged over calls.", flush=True)
    return embs, idxs


def score_seed(config_path: str, seed: int, ref_batches: int,
               n_components: int, shrinkage: float) -> dict:
    base = load_config(config_path)
    cfg = copy.deepcopy(base)
    cfg.seed = seed
    if "seed" in cfg.dataset.params:
        cfg.dataset.params["seed"] = seed
    if "seed" in cfg.model.params:
        cfg.model.params["seed"] = seed

    dataset = build("dataset", cfg.dataset.name, **cfg.dataset.params)
    spec = dataset.spec
    model = build("model", cfg.model.name, **cfg.model.params)
    model.setup(spec)

    novel_start = int((spec.extras or {}).get("novel_start_batch") or 0)
    embs, idxs = _embeddings_for(model, dataset)

    if novel_start and ref_batches >= novel_start:
        raise SystemExit(f"--ref-batches={ref_batches} reaches into the novel phase "
                         f"(starts at {novel_start}); the reference must be known-phase only.")

    ref = np.concatenate(embs[:ref_batches], axis=0)
    mu_ref_full = ref.mean(axis=0)
    Xc = ref - mu_ref_full
    # PCA basis from the reference window only -- the detector never sees novel data.
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(n_components, Vt.shape[0])
    P = Vt[:k].T

    R = (ref - mu_ref_full) @ P
    mu_ref, S_ref = R.mean(axis=0), _shrunk_cov(R, shrinkage)

    stat = []
    for e in embs:
        Z = (e - mu_ref_full) @ P
        stat.append(_frechet(mu_ref, S_ref, Z.mean(axis=0), _shrunk_cov(Z, shrinkage)))
    stat = np.asarray(stat, dtype=np.float64)

    labels = (np.asarray(idxs) >= novel_start).astype(int) if novel_start else np.zeros(len(stat), int)
    out = {"statistic": stat.tolist(), "novel_start_batch": novel_start}
    if labels.sum() >= 2 and (labels == 0).sum() >= 2:
        from sklearn.metrics import roc_auc_score
        out["auroc"] = float(roc_auc_score(labels, stat))
        known, novel = stat[labels == 0], stat[labels == 1]
        for lv in FPR_LEVELS:
            thr = float(np.quantile(known, lv))
            out[FPR_KEYS[lv]] = float((novel <= thr).mean())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--ref-batches", type=int, default=20)
    ap.add_argument("--n-components", type=int, default=32)
    ap.add_argument("--shrinkage", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    per_seed = []
    for s in seeds:
        print(f"[driftlens] seed {s}", flush=True)
        r = score_seed(a.config, s, a.ref_batches, a.n_components, a.shrinkage)
        per_seed.append(r)
        if "auroc" in r:
            print(f"  AUROC {r['auroc']:.3f}  FPR95 {r['fpr95']:.2f}", flush=True)

    agg = {}
    for key in ["auroc"] + [FPR_KEYS[l] for l in FPR_LEVELS]:
        vals = [r[key] for r in per_seed if key in r]
        if vals:
            agg[key] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))

    payload = {
        "config": a.config, "seeds": seeds, "signal": "embedding_frechet",
        "ref_batches": a.ref_batches, "n_components": a.n_components,
        "shrinkage": a.shrinkage, **agg,
        "per_seed": [{k: v for k, v in r.items() if k != "statistic"} for r in per_seed],
    }
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))
    print(f"\nembedding-Frechet: AUROC {agg.get('auroc', float('nan')):.3f} "
          f"+/- {agg.get('auroc_std', 0):.3f}   FPR95 {agg.get('fpr95', float('nan')):.2f}")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
