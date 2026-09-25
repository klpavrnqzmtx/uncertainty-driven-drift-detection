"""Waveform pools for the three audio datasets, cached as int16 memmaps.

Why a cache and not a torch ``Dataset``
---------------------------------------
Speech Commands is 105 829 one-second WAVs.  Decoding those per epoch is
slow anywhere and *pathological* on a network home directory (the cluster)
or an NTFS mount (this workstation).  So each dataset is decoded exactly
once, straight out of its tarball, into

    artifacts/audio/<dataset>/cache/<split>_16k_<dur>s.{npy,labels.npy}

as ``int16`` at 16 kHz, fixed length.  int16 rather than float32 is a
factor-two saving that costs nothing: these files *are* 16-bit PCM at
source (Speech Commands) or are quantised well below 16 bits of real
resolution.  Loading is then an ``mmap_mode='r'`` open — no decode, no
per-file syscalls, and a compute node needs no network.

Reading directly from the ``.tar.gz`` is deliberate: extracting 105 k small
files takes longer than decoding them, and leaves 105 k inodes on a shared
filesystem for nothing.

Splits are the datasets' own official ones — Speech Commands'
``validation_list.txt`` / ``testing_list.txt`` hashing, ESC-50's five folds,
UrbanSound8K's ten folds.  Never a random split: both fold-based sets are
grouped by source recording, and a random split leaks the same recording
across train and test, which inflates accuracy and — the reason it matters
here — makes an "epistemically familiar" claim untestable.
"""

from __future__ import annotations

import csv
import io
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from uncertainty_driven_drift.components.audio_features import SAMPLE_RATE, fix_length, resample

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class AudioDatasetSpec:
    """Static description of one audio dataset."""

    name: str
    archive: str                 # tarball name inside the dataset dir
    n_classes: int
    clip_seconds: float
    class_names: Tuple[str, ...]
    citation: str

    @property
    def n_samples(self) -> int:
        return int(round(self.clip_seconds * SAMPLE_RATE))


_SC_WORDS = (
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow",
    "forward", "four", "go", "happy", "house", "learn", "left", "marvin", "nine",
    "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three",
    "tree", "two", "up", "visual", "wow", "yes", "zero",
)

_US8K_CLASSES = (
    "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
    "engine_idling", "gun_shot", "jackhammer", "siren", "street_music",
)

DATASETS: Dict[str, AudioDatasetSpec] = {
    # The MNIST/CIFAR-10 of audio: large, balanced, single-word utterances.
    "speech_commands": AudioDatasetSpec(
        name="speech_commands",
        archive="speech_commands_v0.02.tar.gz",
        n_classes=35,
        clip_seconds=1.0,
        class_names=_SC_WORDS,
        citation="Warden 2018, Speech Commands v0.02 (105,829 clips, 35 words)",
    ),
    # The CIFAR-100 of audio: 50 semantically distinct classes, 40 clips each,
    # five human-curated superclasses. Small-data regime by construction.
    "esc50": AudioDatasetSpec(
        name="esc50",
        archive="ESC-50-master.tar.gz",
        n_classes=50,
        clip_seconds=5.0,
        class_names=tuple(f"class_{i:02d}" for i in range(50)),  # replaced from meta CSV
        citation="Piczak 2015, ESC-50 (2,000 clips, 50 classes, 5 folds)",
    ),
    # Field recordings with real recording-condition variation across folds —
    # the closest audio analogue to a natural-image benchmark with nuisance shift.
    "urbansound8k": AudioDatasetSpec(
        name="urbansound8k",
        archive="UrbanSound8K.tar.gz",
        n_classes=10,
        clip_seconds=4.0,
        class_names=_US8K_CLASSES,
        citation="Salamon et al. 2014, UrbanSound8K (8,732 excerpts, 10 classes, 10 folds)",
    ),
}


def available_datasets() -> List[str]:
    return sorted(DATASETS)


def get_spec(dataset: str) -> AudioDatasetSpec:
    try:
        return DATASETS[dataset]
    except KeyError as exc:
        raise KeyError(
            f"Unknown audio dataset {dataset!r}; have {available_datasets()}"
        ) from exc


# ---------------------------------------------------------------------------
# decode helpers
# ---------------------------------------------------------------------------

def _decode_member(raw: bytes, n_samples: int) -> np.ndarray | None:
    """Decode WAV bytes to fixed-length mono float32 at 16 kHz, or None if unreadable."""
    from scipy.io import wavfile

    try:
        sr, data = wavfile.read(io.BytesIO(raw))
    except Exception:
        return None
    x = np.asarray(data)
    if x.size == 0:
        return None
    if x.dtype == np.uint8:
        x = (x.astype(np.float32) - 128.0) / 128.0
    elif np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / float(-np.iinfo(x.dtype).min)
    else:
        x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SAMPLE_RATE:
        x = resample(x, sr, SAMPLE_RATE)
    return fix_length(x, n_samples)


def _to_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(x * 32767.0, -32768.0, 32767.0).astype(np.int16)


# ---------------------------------------------------------------------------
# per-dataset manifests: (member_path, label, split)
# ---------------------------------------------------------------------------

def _manifest_speech_commands(tar: tarfile.TarFile) -> List[Tuple[str, int, str]]:
    """Official Speech Commands split from the two list files in the archive."""
    label_of = {w: i for i, w in enumerate(_SC_WORDS)}

    def _read_list(name: str) -> set:
        # The official tarball prefixes every member with "./"; a hand-repacked
        # copy may not. Accept either rather than failing on a cosmetic difference.
        for candidate in (name, f"./{name}"):
            try:
                member = tar.getmember(candidate)
            except KeyError:
                continue
            fh = tar.extractfile(member)
            assert fh is not None
            return {line.decode().strip() for line in fh if line.strip()}
        raise KeyError(f"{name} not found in the Speech Commands archive")

    val_set = _read_list("validation_list.txt")
    test_set = _read_list("testing_list.txt")

    manifest: List[Tuple[str, int, str]] = []
    for member in tar.getmembers():
        path = member.name.lstrip("./")
        if not member.isfile() or not path.endswith(".wav"):
            continue
        if path.startswith("_background_noise_"):
            continue                      # long noise recordings, not labelled words
        word = path.split("/")[0]
        if word not in label_of:
            continue
        split = "test" if path in test_set else ("val" if path in val_set else "train")
        manifest.append((member.name, label_of[word], split))
    return sorted(manifest)


def _manifest_esc50(tar: tarfile.TarFile) -> Tuple[List[Tuple[str, int, str]], Tuple[str, ...]]:
    """ESC-50: folds 1-3 train, fold 4 val, fold 5 test (the canonical held-out fold)."""
    meta_name = next(m for m in tar.getnames() if m.endswith("meta/esc50.csv"))
    fh = tar.extractfile(tar.getmember(meta_name))
    assert fh is not None
    rows = list(csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8")))

    names: Dict[int, str] = {}
    audio_prefix = meta_name.split("meta/")[0] + "audio"
    manifest: List[Tuple[str, int, str]] = []
    for row in rows:
        fold = int(row["fold"])
        target = int(row["target"])
        names[target] = row["category"]
        split = "test" if fold == 5 else ("val" if fold == 4 else "train")
        manifest.append((f"{audio_prefix}/{row['filename']}", target, split))
    class_names = tuple(names[i] for i in range(len(names)))
    return sorted(manifest), class_names


def _manifest_urbansound8k(tar: tarfile.TarFile) -> List[Tuple[str, int, str]]:
    """UrbanSound8K: folds 1-8 train, fold 9 val, fold 10 test (the standard protocol)."""
    meta_name = next(m for m in tar.getnames() if m.endswith("metadata/UrbanSound8K.csv"))
    fh = tar.extractfile(tar.getmember(meta_name))
    assert fh is not None
    rows = list(csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8")))

    audio_root = meta_name.split("metadata/")[0] + "audio"
    manifest: List[Tuple[str, int, str]] = []
    for row in rows:
        fold = int(row["fold"])
        split = "test" if fold == 10 else ("val" if fold == 9 else "train")
        manifest.append(
            (f"{audio_root}/fold{fold}/{row['slice_file_name']}", int(row["classID"]), split)
        )
    return sorted(manifest)


# ---------------------------------------------------------------------------
# cache build / load
# ---------------------------------------------------------------------------

def cache_paths(dataset: str, split: str, data_root: str | Path) -> Tuple[Path, Path]:
    spec = get_spec(dataset)
    root = Path(data_root) / "cache"
    stem = f"{split}_{SAMPLE_RATE // 1000}k_{spec.clip_seconds:g}s"
    return root / f"{stem}.npy", root / f"{stem}.labels.npy"


def build_cache(
    dataset: str,
    data_root: str | Path,
    splits: Sequence[str] = SPLITS,
    force: bool = False,
    verbose: bool = True,
) -> Dict[str, Tuple[Path, Path]]:
    """Decode ``dataset`` from its tarball into per-split int16 memmaps.

    Returns ``{split: (waveform_npy, label_npy)}``.  Already-complete splits
    are skipped unless ``force``.
    """
    spec = get_spec(dataset)
    root = Path(data_root)
    archive = root / spec.archive
    if not archive.exists():
        raise FileNotFoundError(
            f"{spec.name}: missing {archive}. Run scripts/download_audio.sh first."
        )

    wanted = [s for s in splits if force or not all(p.exists() for p in cache_paths(dataset, s, root))]
    out = {s: cache_paths(dataset, s, root) for s in splits}
    if not wanted:
        if verbose:
            print(f"[{spec.name}] cache already complete for {list(splits)}")
        return out

    (root / "cache").mkdir(parents=True, exist_ok=True)
    n_samples = spec.n_samples
    class_names = spec.class_names

    with tarfile.open(archive, "r:gz") as tar:
        if dataset == "speech_commands":
            manifest = _manifest_speech_commands(tar)
        elif dataset == "esc50":
            manifest, class_names = _manifest_esc50(tar)
        elif dataset == "urbansound8k":
            manifest = _manifest_urbansound8k(tar)
        else:                                       # pragma: no cover - guarded by get_spec
            raise KeyError(dataset)

        by_split: Dict[str, List[Tuple[str, int]]] = {s: [] for s in SPLITS}
        for path, label, split in manifest:
            by_split[split].append((path, label))

        # Streaming a .tar.gz is sequential-only, so one pass fills every split
        # at once; seeking per split would re-inflate the whole archive N times.
        writers = {}
        for split in wanted:
            n = len(by_split[split])
            wav_path, lab_path = cache_paths(dataset, split, root)
            arr = np.lib.format.open_memmap(
                wav_path, mode="w+", dtype=np.int16, shape=(n, n_samples)
            )
            writers[split] = {
                "arr": arr,
                "labels": np.zeros(n, dtype=np.int64),
                "valid": np.zeros(n, dtype=bool),
                "index": {p: (i, lab) for i, (p, lab) in enumerate(by_split[split])},
                "written": 0,
                "paths": (wav_path, lab_path),
            }
        if verbose:
            for split in wanted:
                print(f"[{spec.name}] {split}: {len(by_split[split])} clips -> "
                      f"{writers[split]['paths'][0].name}")

        skipped = 0
        for member in tar:
            if not member.isfile():
                continue
            for state in writers.values():
                hit = state["index"].get(member.name)
                if hit is None:
                    continue
                i, label = hit
                fh = tar.extractfile(member)
                raw = fh.read() if fh is not None else b""
                wav = _decode_member(raw, n_samples)
                if wav is None:
                    skipped += 1
                    continue
                state["arr"][i] = _to_int16(wav)
                state["labels"][i] = label
                state["valid"][i] = True
                state["written"] += 1
                break

    for split, state in writers.items():
        wav_path, lab_path = state["paths"]
        n_total = state["arr"].shape[0]
        valid = state["valid"]
        state["arr"].flush()

        if not valid.all():
            # UrbanSound8K ships a handful of WAVs scipy cannot read. Leaving their
            # rows in place would keep them as SILENT clips carrying label 0 — nine
            # fabricated air_conditioner examples, which is worse than nine missing
            # ones. Compact the arrays so only decoded clips survive.
            n_bad = int((~valid).sum())
            print(f"[{spec.name}] {split}: dropping {n_bad} undecodable clip(s) "
                  f"({n_total} -> {int(valid.sum())})")
            kept = np.array(state["arr"][valid])          # materialise the survivors
            del state["arr"]
            np.save(wav_path, kept)
            np.save(lab_path, state["labels"][valid])
        else:
            np.save(lab_path, state["labels"])
        if verbose:
            print(f"[{spec.name}] {split}: wrote {int(valid.sum())}/{n_total}")
    if skipped and verbose:
        print(f"[{spec.name}] {skipped} clip(s) were undecodable and are not in the cache")

    np.save(root / "cache" / "class_names.npy", np.array(class_names, dtype=object),
            allow_pickle=True)
    return out


def load_pool(
    dataset: str,
    split: str,
    data_root: str | Path,
    build_if_missing: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(waveforms_int16, labels)`` for one split, or a union of splits.

    ``split`` may name several splits joined by ``+`` (e.g. ``"val+test"``).
    That exists for the small datasets: ESC-50's test fold is 400 clips, so a
    200-batch stream of 64 redraws it 32 times and the batch-to-batch variance
    stops reflecting anything but resampling noise — which matters because the
    known-phase standard deviation is the denominator of every shift we report.
    The audio backbones use no validation split (no early stopping, no
    hyperparameter search on it), so folding ``val`` in adds genuinely unseen
    examples without weakening the held-out claim.

    A single split stays an ``int16`` memmap — callers convert only the slices
    they touch, via :func:`to_float`, which keeps a 2.7 GB pool out of RAM. A
    union has to materialise, so use it only where the pool is small.
    """
    if "+" in split:
        parts = [p for p in (s.strip() for s in split.split("+")) if p]
        loaded = [load_pool(dataset, p, data_root, build_if_missing) for p in parts]
        x = np.concatenate([np.asarray(a) for a, _ in loaded], axis=0)
        y = np.concatenate([b for _, b in loaded], axis=0)
        return x, y

    wav_path, lab_path = cache_paths(dataset, split, data_root)
    if not (wav_path.exists() and lab_path.exists()):
        if not build_if_missing:
            raise FileNotFoundError(
                f"{dataset}/{split} cache missing at {wav_path}. "
                f"Run: python scripts/prepare_audio.py --dataset {dataset}"
            )
        build_cache(dataset, data_root, splits=(split,))
    x = np.load(wav_path, mmap_mode="r")
    y = np.load(lab_path)
    return x, y


def to_float(x: np.ndarray) -> np.ndarray:
    """int16 pool slice -> float32 waveform in [-1, 1]."""
    return np.ascontiguousarray(np.asarray(x, dtype=np.float32) / 32768.0)


def class_names(dataset: str, data_root: str | Path) -> Tuple[str, ...]:
    """Class names, preferring the ones recovered from the dataset's own metadata."""
    path = Path(data_root) / "cache" / "class_names.npy"
    if path.exists():
        return tuple(np.load(path, allow_pickle=True).tolist())
    return get_spec(dataset).class_names
