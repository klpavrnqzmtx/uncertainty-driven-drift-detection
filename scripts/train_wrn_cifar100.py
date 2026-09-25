#!/usr/bin/env python3
"""Standalone WRN-28-10 / CIFAR-100 trainer + accuracy diagnostic.

The full known-vs-novel pipeline (``uncertainty_driven_drift.cli run``) trains this
backbone lazily on first use and only reports accuracy on the drift STREAM
(mixed known-corruption batches) — useful for the end result, useless for
quickly checking whether a recipe change actually fixed training. This script
trains the backbone directly and immediately reports:

  * clean CIFAR-100 test accuracy
  * per-corruption test accuracy (severity 3) for each of the 5 known
    corruptions the known_vs_novel_wrn_*.yaml configs stream

...so a recipe change can be validated in one run before committing to the
full pipeline (which also runs the sweep + multi-seed eval on top).

v3 checkpoint (plain cosine schedule, block_drop=0.3, default init) plateaued
at ~67% known-phase accuracy regardless of epoch budget (100 vs. 200 gave the
same result) — see wide_resnet.py / resnet_cifar.py for the recipe fix
(Zagoruyko & Komodakis 2016 step-schedule + warmup + Nesterov + grad clip,
lower block dropout, explicit He/MSR init). This script trains under that
fixed recipe by default.

    python scripts/train_wrn_cifar100.py --variant mc_dropout
    python scripts/train_wrn_cifar100.py --variant laplace --epochs 160

Both variants share the recipe; only head dropout (p_drop) differs, matching
what known_vs_novel_wrn_mc_dropout.yaml / known_vs_novel_wrn_laplace.yaml do.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

KNOWN_CORRUPTIONS = ["shot_noise", "zoom_blur", "brightness", "contrast", "jpeg_compression"]
TRAIN_CORRUPTIONS = ["gaussian_noise", "shot_noise", "motion_blur", "zoom_blur",
                     "brightness", "contrast", "jpeg_compression"]


def evaluate(model, data_root: Path, device, corruption: str | None, severity: int = 3):
    """Top-1 accuracy on the CIFAR-100 test set, optionally corrupted."""
    import numpy as np
    import torch
    from torchvision import datasets, transforms

    from uncertainty_driven_drift.components.resnet_cifar import _normalize_batch
    if corruption:
        from uncertainty_driven_drift.components.cifar_c_corruptions import apply_corruption

    # ToTensor() already yields (3, 32, 32) float32 in [0, 1] -- CHW, and
    # scaled. An earlier version transposed as though these were HWC uint8 and
    # divided by 255 a second time, producing (N, 32, 3, 32) arrays in
    # [0, 0.004]; training completed and the checkpoint was written, then this
    # function died on the normalisation broadcast. Take the tensors as they are.
    ds = datasets.CIFAR100(str(data_root), train=False, download=True,
                           transform=transforms.ToTensor())
    xs = np.stack([np.asarray(img) for img, _ in ds]).astype(np.float32)
    ys = np.array([label for _, label in ds])
    assert xs.ndim == 4 and xs.shape[1] == 3, f"expected (N,3,H,W), got {xs.shape}"

    if corruption:
        rng = np.random.default_rng(0)
        xs = apply_corruption(corruption, xs, severity, rng)

    model.eval()
    correct, n = 0, 0
    bs = 500
    with torch.no_grad():
        for i in range(0, len(xs), bs):
            xb = _normalize_batch(xs[i:i + bs], "cifar100").to(device)
            logits = model.head(model.head_drop(model.embed(xb))) \
                if hasattr(model, "head_drop") else model(xb)
            pred = logits.argmax(dim=-1).cpu().numpy()
            correct += (pred == ys[i:i + bs]).sum()
            n += len(pred)
    return correct / n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=["mc_dropout", "laplace"], required=True)
    ap.add_argument("--data-root", default="./artifacts/cifar100")
    ap.add_argument("--ckpt-path", default=None,
                    help="Default: artifacts/models/wrn2810_cifar100_v4_<variant>.pt "
                         "(matches the known_vs_novel_wrn_*.yaml configs).")
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--lr-schedule", choices=["cosine", "step"], default="step")
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--nesterov", action="store_true", default=True)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--clean-fraction", type=float, default=0.2)
    ap.add_argument("--corruption-severity", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force-retrain", action="store_true",
                    help="Retrain even if the checkpoint already exists.")
    a = ap.parse_args(argv)

    import torch
    from uncertainty_driven_drift.components.resnet_cifar import _pretrain_resnet20, _select_device
    from uncertainty_driven_drift.components.wide_resnet import _build_wrn2810

    p_drop = 0.3 if a.variant == "mc_dropout" else 0.0
    ckpt_path = Path(a.ckpt_path) if a.ckpt_path else \
        Path(f"./artifacts/models/wrn2810_cifar100_v4_{a.variant}.pt")
    data_root = Path(a.data_root)

    if a.force_retrain and ckpt_path.exists():
        ckpt_path.unlink()

    if not ckpt_path.exists():
        print(f"=== training WRN-28-10 ({a.variant}, p_drop={p_drop}) -> {ckpt_path} ===")
        print(f"    epochs={a.epochs} lr={a.lr} schedule={a.lr_schedule} "
              f"warmup={a.warmup_epochs} nesterov={a.nesterov} grad_clip={a.grad_clip} "
              f"label_smoothing={a.label_smoothing}")
        _pretrain_resnet20(
            data_root=data_root,
            ckpt_path=ckpt_path,
            epochs=a.epochs,
            batch_size=a.batch_size,
            lr=a.lr,
            seed=a.seed,
            p_drop=p_drop,
            train_corruptions=TRAIN_CORRUPTIONS,
            corruption_severity=a.corruption_severity,
            clean_fraction=a.clean_fraction,
            mixup_alpha=0.0,
            n_classes=100,
            dataset="cifar100",
            build_fn=_build_wrn2810,
            label_smoothing=a.label_smoothing,
            lr_schedule=a.lr_schedule,
            warmup_epochs=a.warmup_epochs,
            nesterov=a.nesterov,
            grad_clip=a.grad_clip,
        )
    else:
        print(f"=== checkpoint already exists at {ckpt_path} (use --force-retrain to redo) ===")

    print("\n=== evaluating ===")
    device = _select_device()
    model = _build_wrn2810(n_classes=100, p_drop=0.0)   # eval deterministically regardless of variant
    model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
    model = model.to(device)
    model.eval()

    clean_acc = evaluate(model, data_root, device, corruption=None)
    print(f"  clean test accuracy:          {clean_acc:.4f}")
    corr_accs = []
    for c in KNOWN_CORRUPTIONS:
        acc = evaluate(model, data_root, device, corruption=c, severity=a.corruption_severity)
        corr_accs.append(acc)
        print(f"  {c:<18s} (sev {a.corruption_severity}):  {acc:.4f}")
    print(f"  mean known-corruption accuracy: {sum(corr_accs) / len(corr_accs):.4f}")
    print("\n(reference: WRN-28-10 on CLEAN CIFAR-100 is ~0.80 in the literature;"
          "\n these runs train with 80% of batches corrupted, which costs clean"
          "\n accuracy and buys robustness. Do NOT compare against the ~0.90 of the"
          "\n ResNet-20 arm (CIFAR-10, 10 classes) or the ViT arm (ImageNet-21k"
          "\n pretrained) -- neither is the same task.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
