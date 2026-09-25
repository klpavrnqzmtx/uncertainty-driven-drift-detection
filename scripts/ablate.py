#!/usr/bin/env python3
"""Sweep one config parameter and report AUROC / FPR95 at each value.

One driver for the three sensitivity analyses, since they differ only in which
key is varied:

    prior precision   --param model.params.prior_precision   --grid 1,10,100,1000,10000
    posterior samples --param model.params.n_samples         --grid 5,10,20,50
    batch size        --param dataset.params.batch_size      --grid 16,32,64,128

For the prior-precision sweep the automatic calibration must be disabled or it
will simply override every grid point -- pass --fix-prior, which sets
model.params.calibrate_prior=False so the grid value is the one actually used.

Each grid point is scored over multiple stream seeds with the same machinery
that builds Table 1 (scripts/auroc_fpr95_table.py), so the numbers are directly
comparable to it: the backbone checkpoint is reused throughout and only the
stream draw and posterior sampling are reseeded.

    python scripts/ablate.py \
        --config configs/experiments/cifar_kn_comparison/known_vs_novel_laplace.yaml \
        --param model.params.n_samples --grid 5,10,20,50 --seeds 0,1,2 \
        --out results/tables/ablation/cifar_S.json
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

from auroc_fpr95_table import FPR_KEYS, FPR_LEVELS, score_steps  # noqa: E402

from uncertainty_driven_drift.config import load_config  # noqa: E402
from uncertainty_driven_drift.runner import run_experiment  # noqa: E402

SIGNALS = ["ddm", "epistemic", "total", "input"]


def _set_path(cfg, dotted: str, value):
    """Set e.g. 'model.params.n_samples' on a loaded config object."""
    head, _, tail = dotted.partition(".")
    obj = getattr(cfg, head)
    keys = tail.split(".")
    if keys[0] == "params":
        params = obj.params
        for k in keys[1:-1]:
            params = params[k]
        params[keys[-1]] = value
        return
    for k in keys[:-1]:
        obj = getattr(obj, k)
    setattr(obj, keys[-1], value)


def _coerce(text: str):
    try:
        v = float(text)
        return int(v) if v.is_integer() and "." not in text and "e" not in text.lower() else v
    except ValueError:
        return text


def run_point(config_path: str, dotted: str, value, seed: int, fix_prior: bool) -> dict:
    base = load_config(config_path)
    cfg = copy.deepcopy(base)
    _set_path(cfg, dotted, value)
    if fix_prior:
        cfg.model.params["calibrate_prior"] = False
    cfg.seed = seed
    if "seed" in cfg.dataset.params:
        cfg.dataset.params["seed"] = seed
    if "seed" in cfg.model.params:
        cfg.model.params["seed"] = seed
    tag = str(value).replace(".", "p").replace("-", "m")
    cfg.experiment = f"{base.experiment}__abl_{dotted.split('.')[-1]}{tag}_s{seed}"
    run_dir = run_experiment(cfg)
    steps = [json.loads(l) for l in open(run_dir / "steps.jsonl")]
    metrics = json.load(open(run_dir / "metrics.json"))
    scored = score_steps(steps, metrics)
    scored["_mean_accuracy"] = metrics.get("mean_accuracy")
    return scored


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--param", required=True, help="Dotted path, e.g. model.params.n_samples")
    ap.add_argument("--grid", required=True, help="Comma-separated values")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--fix-prior", action="store_true",
                    help="Disable automatic prior-precision calibration (required "
                         "when sweeping model.params.prior_precision).")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    grid = [_coerce(g.strip()) for g in a.grid.split(",") if g.strip()]
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    if a.param.endswith("prior_precision") and not a.fix_prior:
        print("WARNING: sweeping prior_precision without --fix-prior; calibration "
              "will override every grid point and all rows will be identical.",
              file=sys.stderr)

    out_rows = []
    for value in grid:
        per_seed = []
        for seed in seeds:
            print(f"[ablate] {a.param}={value} seed={seed}", flush=True)
            per_seed.append(run_point(a.config, a.param, value, seed, a.fix_prior))
        row = {"value": value, "n_seeds": len(per_seed),
               "mean_accuracy": float(np.mean([p["_mean_accuracy"] for p in per_seed
                                               if p["_mean_accuracy"] is not None]))}
        for sig in SIGNALS:
            agg = {}
            for metric in ["auroc"] + [FPR_KEYS[l] for l in FPR_LEVELS]:
                vals = [p[sig][metric] for p in per_seed if p[sig].get(metric) is not None]
                if vals:
                    agg[metric] = float(np.mean(vals))
                    agg[f"{metric}_std"] = float(np.std(vals))
            row[sig] = agg
        out_rows.append(row)
        print(f"  -> AUROC epistemic {row['epistemic'].get('auroc', float('nan')):.3f} "
              f"total {row['total'].get('auroc', float('nan')):.3f} "
              f"acc {row['mean_accuracy']:.3f}", flush=True)

    payload = {"config": a.config, "param": a.param, "grid": grid,
               "seeds": seeds, "fix_prior": a.fix_prior, "rows": out_rows}
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
