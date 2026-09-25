#!/usr/bin/env python3
"""AUROC / FPR95 table, in the style of Table 1 of Ballas et al., "Hydra
Ensembles" (arXiv:2510.18358) — one row per architecture/UQ arm, bold best /
underline second-best per signal, grouped by dataset.

Their Table 1 scores OOD detection with a single confidence score (MSP) on
static, curated ID/OOD image pools. We have no separate OOD dataset — "OOD" is
the novel-phase batches of the same corrupted stream — and three competing
label-free scores instead of one (epistemic, total entropy, input L2), plus
DDM as the labelled reference. So here every SIGNAL plays the role their
Table 1 gives to a METHOD: one column-pair (AUROC / FPR95) per signal, rows
grouped by dataset, computed from each arm's raw per-batch statistic and the
known/novel phase label — same definitions used throughout this repo:

    AUROC        threshold-free separability of the signal between known
                 (label 0) and novel (label 1) batches (Mann-Whitney U, via
                 sklearn).
    FPR95/99/100 fraction of NOVEL batches that fall on the known side of the
                 threshold set to keep 95%/99%/100% of KNOWN batches
                 correctly classified — i.e. the threshold is the
                 95th/99th/100th percentile of known-phase statistic (100% =
                 the arm's max known-phase value = zero known-phase false
                 alarms), and FPR@level is the novel-phase MISS rate there.
                 Same "FPR at X% TPR-of-the-reference-class" definition as the
                 OOD literature (Hendrycks & Gimpel, 2017 popularized FPR95
                 specifically; the underlying concept is general detection
                 theory, not OOD-specific — "reference class" is "known" here
                 rather than an image being in-distribution). FPR100 is the
                 strictest and upper-bounds the other two by construction.

Both are averaged over multiple seeds per arm — a fresh stream draw + MC/
posterior-sampling seed each time, same trained backbone reused throughout
(matches --seed-scope stream on the CLI: this checks that the SEPARABILITY
holds up across draws, not that a single lucky run looked good; it does not
re-verify architecture/training robustness, which would need --seed-scope
full at N times the cost). Each arm's config is re-run fresh, so this needs
either a GPU (vision) or is cheap enough for CPU (audio/wireless) — see
--seeds and --only to scope a run.

    python scripts/auroc_fpr95_table.py --seeds 0,1,2,3,4 \
        --out results/figures/paper/auroc_fpr95_table
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

from uncertainty_driven_drift.components import phase3  # noqa: F401  (registers everything)

# arm -> (dataset group, display label, config path)
ARMS = [
    ("MNIST", "ResNet-20 Laplace", "configs/experiments/mnist_c/known_vs_novel_resnet_laplace.yaml"),
    ("MNIST", "ViT Laplace", "configs/experiments/mnist_c/known_vs_novel_vit_laplace.yaml"),
    ("CIFAR-10", "ResNet-20 Laplace", "configs/experiments/cifar_kn_comparison/known_vs_novel_laplace.yaml"),
    ("CIFAR-10", "ViT Laplace", "configs/experiments/cifar_kn_comparison/known_vs_novel_vit_laplace.yaml"),
    ("CIFAR-10", "ResNet-20 Laplace (severity)", "configs/experiments/cifar_kn_comparison/severity_shift_resnet_laplace.yaml"),
    ("CIFAR-100", "ViT Laplace", "configs/experiments/cifar100_c/known_vs_novel_vit_laplace.yaml"),
    # The ResNet-20 CIFAR-100 capacity arm is deliberately NOT here. It ran, but
    # reaches only 0.54 known-phase accuracy on the 100-way task, so its
    # epistemic signal reflects a weak representation rather than capacity per
    # se. Result (for the record, configs and runs retained): the
    # epistemic-minus-total AUROC gap goes MORE negative as capacity shrinks --
    # ResNet-20 -0.13 (MC) / -0.32 (Laplace), WRN-28-10 -0.00 / -0.21, ViT-B/16
    # +0.00 / 0.00 -- i.e. the opposite of the capacity hypothesis, which is
    # useful as an elimination but not as a Table 1 row.
    ("ESC-50", "CNN MC", "configs/experiments/audio_kn_comparison/esc50_known_vs_novel.yaml"),
    ("ESC-50", "ResNet-18 MC", "configs/experiments/audio_kn_comparison/esc50_known_vs_novel_resnet18.yaml"),
    ("ESC-50", "CNN-wide MC", "configs/experiments/audio_kn_comparison/esc50_known_vs_novel_wide.yaml"),
    ("Speech Commands", "CNN MC", "configs/experiments/audio_kn_comparison/speech_commands_known_vs_novel.yaml"),
    ("Speech Commands", "ResNet-18 MC", "configs/experiments/audio_kn_comparison/speech_commands_known_vs_novel_resnet18.yaml"),
    ("Speech Commands", "CNN-wide MC", "configs/experiments/audio_kn_comparison/speech_commands_known_vs_novel_wide.yaml"),
    ("UrbanSound8K", "CNN MC", "configs/experiments/audio_kn_comparison/urbansound8k_known_vs_novel.yaml"),
    ("UrbanSound8K", "ResNet-18 MC", "configs/experiments/audio_kn_comparison/urbansound8k_known_vs_novel_resnet18.yaml"),
    ("UrbanSound8K", "CNN-wide MC", "configs/experiments/audio_kn_comparison/urbansound8k_known_vs_novel_wide.yaml"),
    ("Wireless", "LOS-drift Laplace", "configs/experiments/wireless_setup/multitap_channel_drift/los_drift_highway_urban_indoor_laplace.yaml"),
    ("Wireless", "LOS->NLOS Laplace", "configs/experiments/wireless_setup/multitap_channel_drift/train_los_test_nlos_laplace.yaml"),
]

SIGNALS = ["ddm", "epistemic", "total", "input"]
SIGNAL_LABEL = {"ddm": "Error / DDM †", "epistemic": "Epistemic",
                "total": "Total entropy", "input": "Input stat."}

# FPR at TPR-of-the-known-class = 95% / 99% / 100%. 100% is the strictest
# (zero known-phase false alarms — threshold = max known-phase statistic),
# so it upper-bounds the other two: FPR100 >= FPR99 >= FPR95 always.
FPR_LEVELS = [0.95, 0.99, 1.0]
FPR_KEYS = {0.95: "fpr95", 0.99: "fpr99", 1.0: "fpr100"}


def _signal_of(detector: str) -> str:
    if detector in ("ddm", "eddm", "pilot_ddm", "pilot_eddm"):
        return "ddm"
    if detector.endswith("_epistemic"):
        return "epistemic"
    if detector.endswith("_total"):
        return "total"
    if detector.endswith("_input"):
        return "input"
    return "?"


def score_steps(steps: list, metrics: dict) -> dict:
    from sklearn.metrics import roc_auc_score

    extras = (metrics.get("stream") or {}).get("extras") or {}
    ns = int(extras.get("novel_start_batch") or len(steps) // 2)
    t = np.array([r["step"] for r in steps])
    labels = (t >= ns).astype(int)

    # one representative detector per signal (they share the same statistic)
    rep = {}
    for r in steps:
        for name in (r.get("detectors") or {}):
            sig = _signal_of(name)
            if sig in SIGNALS and sig not in rep:
                rep[sig] = name

    empty = {"auroc": None, **{FPR_KEYS[lv]: None for lv in FPR_LEVELS}}
    out = {}
    for sig in SIGNALS:
        det = rep.get(sig)
        if det is None:
            out[sig] = dict(empty)
            continue
        stat = np.array([float((r.get("detectors") or {}).get(det, {}).get("statistic", np.nan))
                         for r in steps])
        finite = np.isfinite(stat)
        known = finite & (labels == 0)
        novel = finite & (labels == 1)
        if known.sum() < 2 or novel.sum() < 2 or len(set(labels[finite])) < 2:
            out[sig] = dict(empty)
            continue
        auroc = float(roc_auc_score(labels[finite], stat[finite]))
        row = {"auroc": auroc}
        for lv in FPR_LEVELS:
            thresh = float(np.quantile(stat[known], lv))       # keeps lv of known below it
            row[FPR_KEYS[lv]] = float((stat[novel] <= thresh).mean())  # novel batches missed there
        out[sig] = row
    return out


def run_seed(config_path: str, seed: int) -> Path:
    """Fresh stream draw + MC/posterior-sampling seed; SAME trained backbone
    reused across seeds (checkpoint path is untouched) — matches
    --seed-scope stream on the CLI. Cheap: no retraining."""
    from uncertainty_driven_drift.config import load_config
    from uncertainty_driven_drift.runner import run_experiment

    base = load_config(config_path)
    cfg = copy.deepcopy(base)
    cfg.seed = seed
    if "seed" in cfg.dataset.params:
        cfg.dataset.params["seed"] = seed
    if "seed" in cfg.model.params:
        cfg.model.params["seed"] = seed
    cfg.experiment = f"{base.experiment}__auroc_s{seed}"
    return run_experiment(cfg)


def score_arm(config_path: str, seeds: list[int]) -> dict:
    per_seed = []
    for seed in seeds:
        run_dir = run_seed(config_path, seed)
        steps = [json.loads(l) for l in open(run_dir / "steps.jsonl")]
        metrics = json.load(open(run_dir / "metrics.json"))
        per_seed.append(score_steps(steps, metrics))

    agg = {}
    for sig in SIGNALS:
        aurocs = [p[sig]["auroc"] for p in per_seed if p[sig]["auroc"] is not None]
        row = {
            "auroc": float(np.mean(aurocs)) if aurocs else None,
            "auroc_std": float(np.std(aurocs)) if aurocs else None,
            "n_seeds": len(aurocs),
        }
        for lv in FPR_LEVELS:
            key = FPR_KEYS[lv]
            fprs = [p[sig][key] for p in per_seed if p[sig][key] is not None]
            row[key] = float(np.mean(fprs)) if fprs else None
            row[f"{key}_std"] = float(np.std(fprs)) if fprs else None
        agg[sig] = row
    return agg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4",
                    help="Comma-separated seeds; each re-runs the arm's stream fresh "
                         "(same backbone, new stream/MC-sampling draw).")
    ap.add_argument("--only", default=None,
                    help="Comma-separated substrings to filter arms by dataset or name "
                         "(e.g. 'Wireless' or 'ESC-50,Speech'), for a partial/cheap run.")
    a = ap.parse_args(argv)
    seeds = [int(s) for s in a.seeds.split(",")]

    arms = ARMS
    if a.only:
        needles = [n.strip() for n in a.only.split(",")]
        arms = [r for r in ARMS if any(n in r[0] or n in r[1] for n in needles)]

    rows = []
    for ds, arm, cfg_path in arms:
        if not Path(cfg_path).exists():
            print(f"  [skip] {ds} / {arm}: no config at {cfg_path}")
            continue
        print(f"[{ds} / {arm}] running {len(seeds)} seed(s)…", flush=True)
        scored = score_arm(cfg_path, seeds)
        rows.append({"dataset": ds, "arm": arm, "config": cfg_path, "seeds": seeds, **scored})
        print(f"  {ds:10s} {arm:32s} " +
              "  ".join(
                  f"{s}: auroc={scored[s]['auroc']:.3f}±{scored[s]['auroc_std']:.3f} " +
                  " ".join(f"{FPR_KEYS[lv]}={scored[s][FPR_KEYS[lv]]:.3f}±{scored[s][FPR_KEYS[lv]+'_std']:.3f}"
                           for lv in FPR_LEVELS)
                  if scored[s]["auroc"] is not None else f"{s}:n/a"
                  for s in SIGNALS))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Merge with any existing rows not covered by this invocation (so --only
    # partial runs don't clobber the rest of the table).
    prior = {}
    if out.with_suffix(".json").exists():
        old = json.load(open(out.with_suffix(".json")))
        prior = {(r["dataset"], r["arm"]): r for r in old.get("rows", [])}
    for r in rows:
        prior[(r["dataset"], r["arm"])] = r
    all_rows = list(prior.values())

    json.dump({"signals": SIGNALS, "rows": all_rows}, open(out.with_suffix(".json"), "w"), indent=1)
    print(f"\nwrote {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
