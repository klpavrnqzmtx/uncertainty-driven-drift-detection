#!/usr/bin/env python3
"""Solve the benign EMNIST class prior on a HELD-OUT VALIDATION SPLIT.

The S2-benign scenario needs a class mix q, replacing the uniform prior p, such that a
batch drawn under q has

    HIGHER mean total entropy   -> a total-entropy detector (UDD) fires   [false alarm]
    UNCHANGED mean epistemic    -> the epistemic detector stays quiet     [correct]
    UNCHANGED mean accuracy     -> nothing has actually gone wrong        [correct]

Every batch mean is linear in the class proportions:

    mean_total(q) = sum_c q_c h_c      mean_epi(q) = sum_c q_c e_c      acc(q) = sum_c q_c a_c

so this is a small linear program over the 47-dim simplex.

WHY A VALIDATION SPLIT. Solving on the test stream would fit the scenario to the very
predictions the detectors are about to see — the shift would be tuned to the model.
Here h_c/e_c/a_c are measured on data held out of training AND disjoint from the stream
(which draws from the test split). The structure is
independently justified anyway: 1/I/L and 0/O collide for typographic reasons that have
nothing to do with this model.

    python scripts/derive_benign_prior.py \
        --config configs/experiments/emnist_label_prior/benign_shift.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from uncertainty_driven_drift import registry                       # noqa: E402
from uncertainty_driven_drift.components import phase3              # noqa: E402,F401
from uncertainty_driven_drift.components.emnist_ambiguity import (   # noqa: E402
    LABELS, _emnist, train_val_split,
)
from uncertainty_driven_drift.config import load_config             # noqa: E402
from uncertainty_driven_drift.data.base import StreamSpec           # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-class-weight", type=float, default=0.15,
                    help="Cap per class so the shift stays a realistic mix, not a spike.")
    ap.add_argument("--epi-tolerance-sd", type=float, default=0.25,
                    help="Allowed epistemic movement, in units of batch noise.")
    ap.add_argument("--acc-tolerance", type=float, default=0.0,
                    help="Allowed accuracy movement (0.0 = pinned exactly).")
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F

    cfg = load_config(a.config)
    model = registry.build("model", cfg.model.name, **cfg.model.params)
    spec = StreamSpec(name="emnist_label_prior", input_shape=(1, 28, 28), n_classes=47,
                      n_batches=1, batch_size=64)
    model.setup(spec)                      # pretrains if the checkpoint is missing
    dev = model._device
    net = model._model

    x, y = _emnist(Path(model.data_root), train=True)
    _, val_idx = train_val_split(len(x), model.val_fraction, model.split_seed)
    x, y = x[val_idx], y[val_idx]
    print(f"validation holdout: {len(x)} images (held out of training, disjoint from the stream)")

    T, E, C = [], [], []
    for i in range(0, len(x), 500):
        xb = model.to_backbone_input(x[i:i + 500]).to(dev)
        with torch.no_grad():
            ps = torch.stack([F.softmax(net(xb), -1) for _ in range(model.n_samples)])
        mp = ps.mean(0)
        h = -(mp * torch.log(mp + 1e-12)).sum(-1)
        mh = -(ps * torch.log(ps + 1e-12)).sum(-1).mean(0)
        T.append(h.cpu().numpy())
        E.append((h - mh).cpu().numpy())
        C.append((mp.argmax(-1).cpu().numpy() == y[i:i + 500]).astype(float))
    T = np.concatenate(T)
    E = np.concatenate(E)
    C = np.concatenate(C)

    K = 47
    h_c = np.array([T[y == c].mean() for c in range(K)])
    e_c = np.array([E[y == c].mean() for c in range(K)])
    a_c = np.array([C[y == c].mean() for c in range(K)])

    rng = np.random.default_rng(0)
    bs = 64
    s_e = float(np.std([E[rng.choice(len(E), bs, replace=False)].mean() for _ in range(400)]))
    s_t = float(np.std([T[rng.choice(len(T), bs, replace=False)].mean() for _ in range(400)]))

    p = np.full(K, 1.0 / K)
    from scipy.optimize import linprog
    eps_e = a.epi_tolerance_sd * s_e
    A_ub = np.vstack([e_c, -e_c, -a_c, a_c])
    b_ub = np.array([p @ e_c + eps_e, -(p @ e_c - eps_e),
                     -(p @ a_c - a.acc_tolerance), p @ a_c + a.acc_tolerance])
    r = linprog(-h_c, A_ub=A_ub, b_ub=b_ub, A_eq=np.ones((1, K)), b_eq=[1.0],
                bounds=[(0, a.max_class_weight)] * K, method="highs")
    if not r.success:
        raise SystemExit(f"LP infeasible: {r.message}\n"
                         f"Relax --acc-tolerance or --epi-tolerance-sd.")
    q = r.x
    d_t, d_e, d_a = q @ h_c - p @ h_c, q @ e_c - p @ e_c, q @ a_c - p @ a_c

    print(f"\nvalidation-split baseline: total {p@h_c:.4f}  epistemic {p@e_c:.4f}  acc {p@a_c:.4f}")
    print(f"batch noise (n={bs}): sigma_total {s_t:.4f}  sigma_epistemic {s_e:.4f}")
    print("\nsolved prior (top classes):")
    for c in np.argsort(-q)[:10]:
        if q[c] > 0.01:
            print(f"   {LABELS[c]:>3s}  w={q[c]:.3f}   entropy {h_c[c]:.3f}  "
                  f"epistemic {e_c[c]:.4f}  accuracy {a_c[c]:.3f}")
    print(f"\npredicted shift:  total {d_t:+.4f} = {d_t/s_t:+.2f} sd")
    print(f"                  epist {d_e:+.4f} = {d_e/s_e:+.2f} sd")
    print(f"                  accur {d_a:+.4f}   <- pinned")
    print(f"                  R = {abs(d_t/s_t)/max(abs(d_e/s_e),1e-9):.1f}")

    out = Path(cfg.dataset.params.get("prior_path",
                                      "./artifacts/models/emnist_benign_prior.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "prior": q.tolist(),
        "derived_on": "validation_holdout",
        "n_val": int(len(x)),
        "pred_total_sd": round(float(d_t / s_t), 3),
        "pred_epistemic_sd": round(float(d_e / s_e), 3),
        "pred_delta_accuracy": round(float(d_a), 5),
        "max_class_weight": a.max_class_weight,
    }, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
