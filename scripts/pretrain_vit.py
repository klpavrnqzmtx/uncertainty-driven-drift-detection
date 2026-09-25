#!/usr/bin/env python3
"""Pretrain the ViT backbone(s) named by one or more experiment configs.

Why this exists separately from ``uncertainty-driven-drift run``
--------------------------------------------------------
Both ``vit_mc_dropout`` and ``vit_laplace`` pretrain lazily inside ``setup()``
when their checkpoint is missing, so a plain ``run`` would train *and* stream in
one process.  On a Slurm cluster those two phases want different treatment: the
pretrain is a long GPU job whose only output is a checkpoint on shared storage,
while the prequential run is short and can be repeated cheaply across seeds and
detector sweeps once the checkpoint exists.  Splitting them means a failed or
requeued stream never costs a retrain, and ``--dependency=afterok`` can chain
the two.

This script only builds the model and calls ``setup()``; all training logic
lives in :mod:`uncertainty_driven_drift.components.vit`, so there is no second copy of
the recipe to drift out of sync with the registered component.

Usage
-----
    python scripts/pretrain_vit.py --config configs/experiments/mnist_c/known_vs_novel_vit.yaml
    python scripts/pretrain_vit.py --config a.yaml --config b.yaml   # sequentially
    python scripts/pretrain_vit.py --config a.yaml --epochs 2 --force  # quick smoke
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from uncertainty_driven_drift import registry                        # noqa: E402
from uncertainty_driven_drift.components import phase3               # noqa: E402,F401  (registers components)
from uncertainty_driven_drift.config import load_config              # noqa: E402


def _build_spec(cfg):
    """Instantiate the config's dataset just far enough to read its ``spec``."""
    stream = registry.build("dataset", cfg.dataset.name, **cfg.dataset.params)
    return stream.spec


def pretrain_one(config_path: str, *, epochs: int | None, force: bool) -> dict:
    cfg = load_config(config_path)
    params = dict(cfg.model.params)

    if not cfg.model.name.startswith("vit"):
        raise SystemExit(
            f"{config_path}: model is {cfg.model.name!r}, not a ViT. This script only "
            f"pretrains vit_mc_dropout / vit_laplace."
        )
    if epochs is not None:
        params["pretrain_epochs"] = int(epochs)

    ckpt = Path(params.get("ckpt_path", ""))
    if force and ckpt.exists():
        print(f"[pretrain_vit] --force: removing existing {ckpt}")
        ckpt.unlink()

    if ckpt.exists():
        # The Laplace variant still refits its GGN posterior on every setup(), so
        # skipping here saves the expensive part (backbone training) only.
        print(f"[pretrain_vit] {cfg.experiment}: checkpoint present, nothing to train → {ckpt}")
        return {"experiment": cfg.experiment, "ckpt": str(ckpt), "trained": False, "seconds": 0.0}

    print(f"[pretrain_vit] {cfg.experiment}: {cfg.model.name} → {ckpt}")
    print(f"[pretrain_vit]   epochs={params.get('pretrain_epochs')} "
          f"dataset_name={params.get('dataset_name')} "
          f"corruptions={params.get('train_corruptions')}")

    spec = _build_spec(cfg)
    model = registry.build("model", cfg.model.name, **params)

    t0 = time.time()
    model.setup(spec)      # trains + saves the checkpoint (and fits Laplace, if applicable)
    elapsed = time.time() - t0

    if not ckpt.exists():
        raise SystemExit(f"[pretrain_vit] setup() finished but {ckpt} was not written.")
    size_mb = ckpt.stat().st_size / 1e6
    print(f"[pretrain_vit] {cfg.experiment}: done in {elapsed/60:.1f} min → {ckpt} ({size_mb:.1f} MB)")
    return {"experiment": cfg.experiment, "ckpt": str(ckpt), "trained": True, "seconds": elapsed}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", action="append", required=True,
                    help="Experiment YAML naming a vit_* model. Repeatable.")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override pretrain_epochs (for smoke runs).")
    ap.add_argument("--force", action="store_true",
                    help="Delete an existing checkpoint and retrain from scratch.")
    args = ap.parse_args()

    results = [pretrain_one(c, epochs=args.epochs, force=args.force) for c in args.config]

    print("\n" + "=" * 70)
    for r in results:
        status = f"trained in {r['seconds']/60:.1f} min" if r["trained"] else "already present"
        print(f"  {r['experiment']:36s} {status:24s} {r['ckpt']}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
