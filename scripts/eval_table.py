#!/usr/bin/env python3
"""Multi-seed detector comparison table: FAR, delay, detection rate, AUROC.

    python scripts/eval_table.py --config <cfg> --seeds 10 --out results/tables/<tag>

Runs the experiment ``--seeds`` times (fresh dataset + MC/posterior sampling seed
each), scores every detector on each run, and reports mean +/- std. Writes
``<out>.json``, ``<out>.md`` and ``<out>.tex``.

Metrics
-------
FAR      false-alarm rate: fraction of KNOWN-phase batches that alarm.
delay    batches from the drift boundary to the first novel-phase alarm
         (``inf`` if a seed never fires; the mean is over seeds that did).
det      detection rate: fraction of seeds that ever fire after the boundary.
AUROC    threshold-FREE separability of the detector's monitored signal between
         known and novel batches. FAR/delay describe one operating point; AUROC
         describes the signal itself, so a detector can have a poor FAR merely
         from a badly chosen threshold yet still carry a strong signal.

The DDM row is the **oracle**: it monitors the true prediction-error stream, i.e.
it consumes the labels the unsupervised detectors never see. It is the reference
these methods are trying to approach without supervision, not a competitor.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from uncertainty_driven_drift.components import phase3  # noqa: F401  (registers everything)
from uncertainty_driven_drift.config import ExperimentConfig, load_config
from uncertainty_driven_drift.runner import run_experiment

ORACLE = {"ddm", "eddm", "pilot_ddm", "pilot_eddm"}
FAMILY_ORDER = {"supervised": 0, "uncertainty": 1, "input": 2}


def _family(name: str) -> str:
    if name in ORACLE:
        return "supervised"
    if name.endswith("_input"):
        return "input"
    return "uncertainty"


def _signal(name: str) -> str:
    if name in ORACLE:
        return "error (labels)"
    if name.endswith("_epistemic"):
        return "epistemic MI"
    if name.endswith("_total"):
        return "total entropy"
    if name.endswith("_input"):
        return "input L2"
    return "?"


def _run_seed(base: ExperimentConfig, seed: int) -> Path:
    cfg = copy.deepcopy(base)
    cfg.seed = seed
    if "seed" in cfg.dataset.params:
        cfg.dataset.params["seed"] = seed
    if "seed" in cfg.model.params:
        cfg.model.params["seed"] = seed
    cfg.experiment = f"{base.experiment}__eval_s{seed}"
    return run_experiment(cfg)


def _score(run_dir: Path) -> Dict[str, Dict[str, float]]:
    steps = [json.loads(l) for l in open(run_dir / "steps.jsonl")]
    metrics = json.load(open(run_dir / "metrics.json"))
    extras = (metrics.get("stream") or {}).get("extras") or {}
    ns = int(extras.get("novel_start_batch") or len(steps) // 2)
    t = np.arange(len(steps))
    labels = (t >= ns).astype(int)

    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        roc_auc_score = None

    out: Dict[str, Dict[str, float]] = {}
    for name in sorted({n for s in steps for n in (s.get("detectors") or {})}):
        alarms = np.array([bool((s.get("detectors") or {}).get(name, {}).get("alarm", False))
                           for s in steps])
        stat = np.array([(s.get("detectors") or {}).get(name, {}).get("statistic", np.nan)
                         for s in steps], dtype=float)
        far = float(alarms[:ns].mean()) if ns > 0 else np.nan
        fired = t[ns:][alarms[ns:]]
        delay = float(fired.min() - ns) if fired.size else np.inf
        auroc = np.nan
        if roc_auc_score is not None and np.isfinite(stat).all() and len(set(labels)) == 2:
            try:
                auroc = float(roc_auc_score(labels, stat))
            except Exception:
                pass
        out[name] = {"far": far, "delay": delay, "auroc": auroc}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, default=10, help="Number of seeds (>=10 recommended).")
    ap.add_argument("--seed0", type=int, default=1000, help="First seed; seeds are seed0..seed0+N-1.")
    ap.add_argument("--out", required=True, help="Output stem (no extension).")
    a = ap.parse_args(argv)

    base = load_config(a.config)
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    print(f"[eval] {base.experiment}: {len(seeds)} seeds")

    per_seed: List[Dict[str, Dict[str, float]]] = []
    for i, s in enumerate(seeds, 1):
        print(f"[eval]   seed {s} ({i}/{len(seeds)})…", flush=True)
        per_seed.append(_score(_run_seed(base, s)))

    names = sorted(per_seed[0], key=lambda n: (FAMILY_ORDER[_family(n)], n))
    rows = []
    for n in names:
        far = np.array([p[n]["far"] for p in per_seed], float)
        dly = np.array([p[n]["delay"] for p in per_seed], float)
        auc = np.array([p[n]["auroc"] for p in per_seed], float)
        finite = np.isfinite(dly)
        rows.append({
            "detector": n, "family": _family(n), "signal": _signal(n),
            "oracle": n in ORACLE,
            "far_mean": float(np.nanmean(far)), "far_std": float(np.nanstd(far)),
            "delay_mean": float(dly[finite].mean()) if finite.any() else None,
            "delay_std": float(dly[finite].std()) if finite.any() else None,
            "det_rate": float(finite.mean()),
            "auroc_mean": float(np.nanmean(auc)), "auroc_std": float(np.nanstd(auc)),
        })

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"config": a.config, "experiment": base.experiment, "seeds": seeds,
               "rows": rows}, open(out.with_suffix(".json"), "w"), indent=1)

    def fmt(r, tex=False):
        d = "--" if r["delay_mean"] is None else f"{r['delay_mean']:.1f} ± {r['delay_std']:.1f}"
        star = r"$^\dagger$" if (tex and r["oracle"]) else ("†" if r["oracle"] else "")
        return (f"{r['detector']}{star}", r["signal"],
                f"{r['far_mean']:.3f} ± {r['far_std']:.3f}", d,
                f"{r['det_rate']:.2f}", f"{r['auroc_mean']:.3f} ± {r['auroc_std']:.3f}")

    hdr = ("detector", "signal", "FAR", "delay", "det", "AUROC")
    with open(out.with_suffix(".md"), "w") as f:
        f.write(f"# {base.experiment} — {len(seeds)} seeds\n\n")
        f.write("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n")
        for r in rows:
            f.write("| " + " | ".join(fmt(r)) + " |\n")
        f.write("\n† supervised oracle: monitors the true error stream (uses labels).\n")
    with open(out.with_suffix(".tex"), "w") as f:
        f.write("\\begin{tabular}{llrrrr}\n\\toprule\n")
        f.write(" & ".join(hdr) + " \\\\\n\\midrule\n")
        for r in rows:
            f.write(" & ".join(fmt(r, tex=True)).replace("±", "$\\pm$") + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    w = max(len(r["detector"]) for r in rows) + 2
    print(f"\n=== {base.experiment} ({len(seeds)} seeds) ===")
    print(f"{'detector':{w}}{'signal':16}{'FAR':>14}{'delay':>14}{'det':>6}{'AUROC':>16}")
    for r in rows:
        c = fmt(r)
        print(f"{c[0]:{w}}{c[1]:16}{c[2]:>14}{c[3]:>14}{c[4]:>6}{c[5]:>16}")
    print(f"\nwrote {out.with_suffix('.json')}, .md, .tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
