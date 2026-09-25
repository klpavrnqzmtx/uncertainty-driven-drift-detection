#!/usr/bin/env python3
"""Posterior diagnostics for a last-layer Laplace arm: lambda, rho_x, runtime.

Answers three separate questions from one pass over a stream, without
touching the prediction hot path:

1. CALIBRATED PRIOR PRECISION. The configs all read ``prior_precision: 1.0``,
   which is misleading: calibrate_prior_precision() overrides it at setup time
   and the chosen value is only ever printed, never persisted. This reports the
   value actually used, per arm, so it can go in the paper.

2. LOCALITY OF THE EXPANSION (rho_x). Proposition 4.1 is a small-variance
   expansion whose remainder is O(rho_x^3), with rho_x^2 = tr(V_x),
   V_x = G_x Sigma G_x^T. For a diagonal last-layer posterior this is cheap in
   closed form -- no sampling needed:

       rho_x^2 = sum_c [ sum_j sigma^2_W[c,j] h_j(x)^2 + sigma^2_b[c] ]
               = (h(x)^2) . colsum(W_var) + sum(b_var)

   Reported separately over known- and novel-phase batches, which is exactly the
   check that matters: does the approximation stay local precisely where
   the method claims a signal?

3. RUNTIME. Wall-clock per batch for the posterior-sampling predict, against a
   plain deterministic forward pass, to quantify monitoring overhead.

    python scripts/laplace_diagnostics.py \
        --config configs/experiments/cifar_kn_comparison/known_vs_novel_laplace.yaml \
        --out results/tables/diagnostics/cifar_kn_laplace.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from uncertainty_driven_drift.components import phase3  # noqa: F401  (registers everything)
from uncertainty_driven_drift.config import load_config
from uncertainty_driven_drift.registry import build


from arm_features import capture_features  # noqa: E402


def _summary(v: np.ndarray) -> dict:
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return {}
    return {
        "mean": float(v.mean()), "std": float(v.std()),
        "p05": float(np.percentile(v, 5)), "median": float(np.median(v)),
        "p95": float(np.percentile(v, 95)), "max": float(v.max()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-batches", type=int, default=0,
                    help="Stop early (0 = whole stream); for smoke tests.")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    dataset = build("dataset", cfg.dataset.name, **cfg.dataset.params)
    spec = dataset.spec
    model = build("model", cfg.model.name, **cfg.model.params)
    model.setup(spec)
    unc = build("uncertainty", cfg.uncertainty.name, **cfg.uncertainty.params)
    unc.setup(model)

    post = getattr(model, "_posterior", None)
    if post is None:
        raise SystemExit(f"{cfg.model.name} exposes no last-layer Laplace posterior "
                         "(is this an MC-dropout arm?); rho_x is not defined here.")

    W_var, b_var = np.asarray(post["W_var"]), np.asarray(post["b_var"])
    # Only resnet_cifar and pretrained_vit run calibrate_prior_precision(); the
    # MNIST, scratch-ViT, audio and wireless families use the config value
    # verbatim. Absence of the key therefore means UNCALIBRATED, not unknown --
    # a distinction the paper has to make when it answers "how is lambda set?".
    lam = post.get("prior_precision")
    lam_calibrated = lam is not None
    # tr(V_x) needs only the per-feature column sums of the weight variances.
    w_colsum, b_sum = W_var.sum(axis=0), float(b_var.sum())

    novel_start = int((spec.extras or {}).get("novel_start_batch") or 0)
    rec = {"known": {"rho2": [], "eu": []}, "novel": {"rho2": [], "eu": []}}
    t_predict, t_embed, n_batches, n_samples_seen = 0.0, 0.0, 0, 0
    embed_err = None

    # The features and the deterministic-forward timing both come from the
    # arm's OWN predict(), intercepted at embed(). Re-deriving the input
    # pipeline here instead is what broke the MNIST arms (different
    # _normalize_batch signature) and the ViT arm (normalises inside _prep and
    # resizes to 224, so a raw 32x32 batch reached the backbone). It also stops
    # the deterministic baseline running the backbone a second time.
    with capture_features(model) as cap:
        for batch in dataset:
            if a.max_batches and n_batches >= a.max_batches:
                break
            n_calls_before, sec_before = len(cap.feats), cap.seconds
            t0 = time.perf_counter()
            pred = model.predict(batch)
            t_predict += time.perf_counter() - t0

            scores = unc.score(batch, pred)

            feats = None
            got = cap.feats[n_calls_before:]
            if got:
                # Cost of ONE deterministic forward -- the work a non-monitored
                # deployment already does. A sampling predict() may call embed()
                # once per posterior sample, so divide by the number of calls.
                t_embed += (cap.seconds - sec_before) / len(got)
                feats = np.stack([g.astype(np.float64) for g in got], axis=0).mean(axis=0)
                del cap.feats[n_calls_before:]
            elif embed_err is None:
                embed_err = "predict() made no embed() call"

            if feats is not None and feats.shape[1] == w_colsum.shape[0]:
                rho2 = (feats ** 2) @ w_colsum + b_sum
            else:
                rho2 = np.full(len(batch.x), np.nan)

            phase = "novel" if (novel_start and batch.index >= novel_start) else "known"
            rec[phase]["rho2"].append(np.asarray(rho2, dtype=np.float64))
            rec[phase]["eu"].append(np.asarray(scores.epistemic, dtype=np.float64))
            n_batches += 1
            n_samples_seen += len(batch.x)

    out = {
        "config": a.config,
        "experiment": cfg.experiment,
        "model": cfg.model.name,
        "prior_precision_used": lam if lam_calibrated else cfg.model.params.get("prior_precision"),
        "prior_precision_calibrated": lam_calibrated,
        "config_prior_precision": cfg.model.params.get("prior_precision"),
        "embed_error": embed_err,
        "n_posterior_samples_S": cfg.model.params.get("n_samples"),
        "batch_size_B": cfg.dataset.params.get("batch_size"),
        "novel_start_batch": novel_start,
        "n_batches": n_batches,
        "runtime": {
            "predict_ms_per_batch": 1e3 * t_predict / max(n_batches, 1),
            "deterministic_embed_ms_per_batch": 1e3 * t_embed / max(n_batches, 1),
            "overhead_ratio": (t_predict / t_embed) if t_embed > 0 else None,
        },
    }
    for phase in ("known", "novel"):
        if not rec[phase]["rho2"]:
            continue
        rho2 = np.concatenate(rec[phase]["rho2"])
        eu = np.concatenate(rec[phase]["eu"])
        finite = np.isfinite(rho2)
        out[phase] = {
            "rho2": _summary(rho2[finite]),
            "rho": _summary(np.sqrt(rho2[finite])),
            "epistemic": _summary(eu),
        }

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))

    print(f"\n=== {cfg.experiment} ===")
    tag = "calibrated" if lam_calibrated else "FIXED, not calibrated"
    print(f"  prior precision        : {out['prior_precision_used']}  ({tag}; "
          f"config says {out['config_prior_precision']})")
    print(f"  B={out['batch_size_B']}  S={out['n_posterior_samples_S']}  "
          f"novel_start={novel_start}  batches={n_batches}")
    for phase in ("known", "novel"):
        if phase not in out:
            continue
        r, e = out[phase]["rho"], out[phase]["epistemic"]
        eu = f"EU mean {e['mean']:.4f}" if e else "EU n/a"
        if r:
            print(f"  {phase:<6} rho: mean {r['mean']:.4f} p95 {r['p95']:.4f} "
                  f"max {r['max']:.4f}   |  {eu}")
        else:
            print(f"  {phase:<6} rho: UNAVAILABLE ({embed_err or 'no embed()'})   |  {eu}")
    rt = out["runtime"]
    print(f"  predict {rt['predict_ms_per_batch']:.1f} ms/batch vs deterministic "
          f"{rt['deterministic_embed_ms_per_batch']:.1f} ms/batch")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
