#!/usr/bin/env python
"""Measure per-corruption uncertainty level, to choose a level-matched known set.

    python scripts/probe_corruptions.py --dataset esc50 \
        --ckpt artifacts/models/audio_cnn_esc50_kn_mc.pt

Why this exists
---------------
In a blocked known-vs-novel stream the "known" phase is 7 corruption blocks in a
row.  If those blocks sit at different epistemic levels, every change detector
fires at the block edges — correctly, because the level really did change — and
the run reports them as known-phase false alarms.  Measured on ESC-50 the spread
of block means (0.0245) exceeded the within-block noise (0.0173), and the alarms
duly clustered at the band_stop edges.

So this probe reports, per corruption, the mean epistemic and total uncertainty
and the accuracy, letting the known set be chosen **level-matched**: a known
phase that is approximately stationary, which is what "in-distribution
operation" is supposed to mean.

Non-circularity
---------------
Probed on the **train** split, which the stream never draws from (the streams use
`test` or `val+test`).  The selection therefore never sees the data it will later
be evaluated on.  It does use the trained model, which is unavoidable — the
quantity of interest is "which corruptions does *this* model find equally
familiar" — and that is why the split separation matters.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uncertainty_driven_drift.components.audio_corruptions import (  # noqa: E402
    CORRUPTION_GROUPS,
    apply_corruption,
    available_corruptions,
)
from uncertainty_driven_drift.components.audio_pools import get_spec, load_pool, to_float  # noqa: E402
from uncertainty_driven_drift.data.base import StreamBatch  # noqa: E402
from uncertainty_driven_drift.registry import build  # noqa: E402
import uncertainty_driven_drift.components.phase3  # noqa: F401,E402
from uncertainty_driven_drift.components.mc_dropout_uncertainty import MCDropoutUncertainty  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch", default="cnn")
    ap.add_argument("--split", default="train", help="probe split; must NOT be the stream's")
    ap.add_argument("--severity", type=int, default=3)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    root = f"./artifacts/audio/{args.dataset}"
    spec = get_spec(args.dataset)
    pool_x, pool_y = load_pool(args.dataset, args.split, root, build_if_missing=False)

    model = build("model", "audio_cnn_mc_dropout", dataset=args.dataset, data_root=root,
                  ckpt_path=args.ckpt, arch=args.arch, n_samples=20, p_drop=0.3)
    from uncertainty_driven_drift.data.base import StreamSpec
    model.setup(StreamSpec(name="audio_known_novel", input_shape=(1, spec.n_samples),
                           n_classes=spec.n_classes, n_batches=1, batch_size=args.batch_size,
                           extras={"dataset": args.dataset}))
    unc = MCDropoutUncertainty()
    unc.setup(model)

    rng = np.random.default_rng(args.seed)
    rows = []
    group_of = {c: g for g, names in CORRUPTION_GROUPS.items() for c in names}
    for corruption in available_corruptions():
        srng = np.random.default_rng(args.seed)
        crng = np.random.default_rng(args.seed + 1)
        epi, tot, acc = [], [], []
        for _ in range(args.batches):
            idx = np.sort(srng.integers(0, pool_x.shape[0], size=args.batch_size))
            x = to_float(pool_x[idx])[:, None, :]
            y = np.asarray(pool_y[idx], dtype=np.int64)
            x = apply_corruption(corruption, x, args.severity, crng)
            batch = StreamBatch(index=0, x=x, y=y)
            pred = model.predict(batch)
            sc = unc.score(batch, pred)
            epi.append(float(np.mean(sc.epistemic)))
            tot.append(float(np.mean(sc.total)))
            acc.append(float((pred.probs.argmax(1) == y).mean()))
        rows.append((corruption, group_of.get(corruption, "-"),
                     float(np.mean(epi)), float(np.std(epi)),
                     float(np.mean(tot)), float(np.mean(acc))))

    rows.sort(key=lambda r: r[2])
    clean = next(r for r in rows if r[0] == "identity")
    print(f"\n{args.dataset} / {args.arch} / severity {args.severity} / split={args.split} "
          f"({args.batches} batches x {args.batch_size})")
    print(f"{'corruption':16s} {'family':12s} {'epistemic':>10s} {'(sd)':>8s} "
          f"{'total':>8s} {'acc':>6s}  {'epi vs clean':>12s}")
    for c, g, e, esd, t, a in rows:
        print(f"{c:16s} {g:12s} {e:10.4f} {esd:8.4f} {t:8.4f} {a:6.3f}  {e - clean[2]:+12.4f}")
    print(f"\nclean (identity) epistemic = {clean[2]:.4f}, accuracy = {clean[5]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
