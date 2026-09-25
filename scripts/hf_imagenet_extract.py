#!/usr/bin/env python3
"""Shared HF ``ILSVRC/imagenet-1k`` Parquet -> ImageFolder extractor.

Extracts the full val split (50k images) and a shard subset of train, used
only to fine-tune the ImageNet ViT arm on known corruptions before evaluating
known-vs-novel drift on the separately extracted val split; see
pretrained_vit.py's ``_imagenet_train_arrays``.

Handles what a naive tar-based extractor gets wrong against the current
hosting shape: val/train ship as Parquet (image bytes + integer ClassLabel per
row, NOT tar), and the integer label's parquet-embedded name is a human
description ("tench, Tinca tinca"), not a wnid — resolved instead from this
HF repo's own classes.py (IMAGENET2012_CLASSES: OrderedDict[wnid, description],
in canonical index order), cross-validated against the parquet's own
description list before trusting the wnid order: a silent mismatch would
mislabel every image, which is worse than the loud failure this replaces.

Resumable at shard granularity via a .done marker per shard, so a login-node
kill mid-run only costs the shard in flight, not the whole download.

Shards are class-sorted with a narrow per-shard class span (empirically: val's
first 6/14 shards, ~43% of images, already covered ~43% of the 1000 classes) —
so --n-shards below picks a SUBSET evenly spaced across the full shard range,
not the first N, so the classes that ARE covered span the whole alphabet
instead of one contiguous chunk.

    python scripts/hf_imagenet_extract.py --out artifacts/imagenet/val --split val
    python scripts/hf_imagenet_extract.py --out artifacts/imagenet/train_slice \
        --split train --n-shards 20
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path


def _evenly_spaced_indices(n_total: int, n_want: int) -> list[int]:
    """n_want indices in [0, n_total), spanning the full range (not clustered)."""
    if n_want <= 0 or n_want >= n_total:
        return list(range(n_total))
    if n_want == 1:
        return [0]
    return sorted({round(i * (n_total - 1) / (n_want - 1)) for i in range(n_want)})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=["val", "train"], required=True)
    ap.add_argument("--n-shards", type=int, default=0,
                    help="Use only this many shards, evenly spaced across the full "
                         "range (0 = all shards; use 0 for val, a small number for "
                         "a train slice).")
    ap.add_argument("--repo-id", default="ILSVRC/imagenet-1k")
    ap.add_argument("--filename-prefix", default=None,
                    help="Output filename stem; default 'ILSVRC2012_val' for val, "
                         "'train' for train.")
    a = ap.parse_args(argv)

    try:
        import hf_transfer  # noqa: F401
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    except ImportError:
        pass

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    token = os.environ["HF_TOKEN"]
    repo_id = a.repo_id
    fname_prefix = a.filename_prefix or ("ILSVRC2012_val" if a.split == "val" else "train")

    api = HfApi(token=token)
    all_shards = sorted(f for f in api.list_repo_files(repo_id, repo_type="dataset")
                        if f.startswith(f"data/{a.split}") and f.endswith(".parquet"))
    if not all_shards:
        raise SystemExit(f"  ERROR: no data/{a.split}*.parquet files found "
                         f"— has the repo layout changed?")

    idx = _evenly_spaced_indices(len(all_shards), a.n_shards)
    shards = [all_shards[i] for i in idx]
    print(f"  {len(all_shards)} {a.split} shard(s) on the hub; using {len(shards)}", flush=True)

    classes_py = hf_hub_download(repo_id=repo_id, repo_type="dataset",
                                 filename="classes.py", token=token)
    spec = importlib.util.spec_from_file_location("_imagenet1k_classes", classes_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    wnids = list(mod.IMAGENET2012_CLASSES.keys())
    wnid_descs = list(mod.IMAGENET2012_CLASSES.values())
    print(f"  classes.py: {len(wnids)} wnids, e.g. {wnids[0]} = {wnid_descs[0]!r}")

    # Resume at shard granularity: a marker is written only after a shard's
    # images are fully placed on disk, so a login-node kill mid-run costs at
    # most the one shard in flight.
    #
    # NOTE this directory lives INSIDE the ImageFolder tree, so any reader must
    # ignore non-wnid directories. torchvision's stock ImageFolder does not --
    # it treats every subdirectory as a class and refuses outright ("Found no
    # valid file for the classes .shard_done"), and if it had merely tolerated
    # the empty class, "." sorting before "n" would have shifted every wnid's
    # index by one and mislabelled the whole split. imagenet_c.py's loader
    # restricts to n######## directories for exactly this reason; keep the two
    # in step if this path ever changes.
    markers = out / ".shard_done"; markers.mkdir(exist_ok=True)

    label_names = None   # wnids, once cross-validated against the first shard below
    global_idx = 0        # stable numbering across resumes
    total_placed = 0
    for shard in shards:
        marker = markers / (Path(shard).name + ".done")
        if marker.exists():
            global_idx += int(marker.read_text().strip())
            print(f"  skip (already done): {shard}")
            continue

        print(f"  downloading {shard} ...", flush=True)
        local_path = hf_hub_download(repo_id=repo_id, repo_type="dataset",
                                     filename=shard, token=token)
        pf = pq.ParquetFile(local_path)
        if label_names is None:
            meta = pf.schema_arrow.metadata or {}
            if b"huggingface" not in meta:
                raise SystemExit("  ERROR: no 'huggingface' schema metadata on this shard; "
                                 "cannot cross-validate classes.py's wnid order.")
            hf_meta = json.loads(meta[b"huggingface"].decode("utf-8"))
            parquet_descs = hf_meta["info"]["features"]["label"]["names"]
            if parquet_descs != wnid_descs:
                mismatch = next(i for i, (x, y) in enumerate(zip(parquet_descs, wnid_descs)) if x != y)
                raise SystemExit(
                    "  ERROR: classes.py's description order does not match the parquet "
                    f"label order — refusing to guess the wnid mapping. First mismatch at index {mismatch}.")
            label_names = wnids
            print(f"  cross-validated: classes.py order matches parquet label order ({len(wnids)} classes)")

        shard_placed = 0
        for batch in pf.iter_batches(batch_size=256, columns=["image", "label"]):
            for img, label in zip(batch.column("image").to_pylist(),
                                  batch.column("label").to_pylist()):
                global_idx += 1
                d = out / label_names[label]; d.mkdir(exist_ok=True)
                with open(d / f"{fname_prefix}_{global_idx:08d}.JPEG", "wb") as f:
                    f.write(img["bytes"])
                shard_placed += 1
        marker.write_text(str(shard_placed))
        total_placed += shard_placed
        print(f"    {shard}: {shard_placed} images (running total {total_placed})", flush=True)

    n_classes = len(list(out.glob("n*")))
    print(f"  arranged {total_placed} new image(s) this run; {n_classes} wnid folders total")
    if n_classes == 0:
        raise SystemExit("  ERROR: nothing extracted.")

    # torchvision.ImageFolder assigns LOCAL class indices 0..(k-1) by
    # alphabetically sorting whichever wnid folders are actually present. That
    # only matches the true global 0-999 ImageNet class index when all 1000
    # are present (val, eventually) — for a train SLICE (a subset), it silently
    # renumbers classes, so fine-tuning would train against the wrong labels
    # for every class. Writing the canonical order here means imagenet_c.py's
    # loader can remap local -> global indices without needing classes.py (and
    # therefore network + HF_TOKEN) again at eval/fine-tune time.
    (out / "wnid_index.json").write_text(json.dumps(wnids))
    print(f"  wrote wnid_index.json (canonical global class order, {len(wnids)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
