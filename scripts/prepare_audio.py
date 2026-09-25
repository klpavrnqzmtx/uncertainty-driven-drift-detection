#!/usr/bin/env python
"""Decode the audio tarballs into the int16 waveform caches the streams read.

    python scripts/prepare_audio.py --dataset all
    python scripts/prepare_audio.py --dataset esc50 --force

Run this ON A MACHINE WITH THE TARBALLS PRESENT (the login node, or your
workstation followed by an rsync of artifacts/audio).  A compute node never
needs to: after this step every split is a plain ``.npy`` memmap.

Downloading is a separate step on purpose — see scripts/download_audio.sh.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from uncertainty_driven_drift.components.audio_pools import (  # noqa: E402
    DATASETS,
    available_datasets,
    build_cache,
    class_names,
    load_pool,
)


def _report(dataset: str, root: Path) -> None:
    spec = DATASETS[dataset]
    print(f"\n== {dataset}  ({spec.citation})")
    for split in ("train", "val", "test"):
        x, y = load_pool(dataset, split, root, build_if_missing=False)
        counts = np.bincount(y, minlength=spec.n_classes)
        print(
            f"   {split:5s} n={x.shape[0]:6d}  samples/clip={x.shape[1]:6d}  "
            f"classes={int((counts > 0).sum()):3d}/{spec.n_classes}  "
            f"per-class min/max={counts.min()}/{counts.max()}  "
            f"{x.nbytes / 1e6:.0f} MB"
        )
    print(f"   classes: {', '.join(class_names(dataset, root)[:8])}, …")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="all",
                    choices=["all", *available_datasets()])
    ap.add_argument("--audio-root", default="./artifacts/audio",
                    help="parent dir holding one sub-dir per dataset")
    ap.add_argument("--force", action="store_true", help="rebuild existing caches")
    args = ap.parse_args()

    datasets = available_datasets() if args.dataset == "all" else [args.dataset]
    rc = 0
    for dataset in datasets:
        root = Path(args.audio_root) / dataset
        try:
            build_cache(dataset, root, force=args.force)
            _report(dataset, root)
        except FileNotFoundError as exc:
            print(f"SKIP {dataset}: {exc}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
