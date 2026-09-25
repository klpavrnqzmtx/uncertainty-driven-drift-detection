#!/usr/bin/env python
"""Per-dataset audio result and hyperparameter tables, sized to be read.

    python scripts/audio_tables.py

Writes, per dataset, into results/figures/audio/:

    results_<dataset>.png    signals x backbones: AUROC, false alarms, delay
    knobs_<dataset>.png      the ONE swept knob per detector, per backbone

Why not the shared plot_auroc_table / plot_hyperparams_table
-----------------------------------------------------------
Those put every arm in one figure, which for audio means 7 columns and, in the
hyperparameter table, whole parameter dumps per cell — the text overflowed into
neighbouring columns and was unreadable.  Two specific fixes here:

* **One row per signal, not per detector.**  AUROC is computed from the signal
  each detector reads, so ``adwin_total``, ``kswin_total`` and ``ph_total`` are
  *the same number* (0.592 on ESC50-CNN, all three).  Nine near-duplicate rows
  carried four distinct values.  False alarms and delay *do* differ per
  algorithm, so those stay per algorithm.
* **Only the swept knob.**  ADWIN's delta, KSWIN's alpha and PageHinkley's
  threshold are what the sweep sets; ``delta=0.00327, min_instances=10,
  alpha=1`` are identical in every arm and belong in a caption, not in 21 cells.

Splitting by dataset also means each figure compares backbones on one task,
which is the comparison the text actually makes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

OUT_DIR = Path("results/figures/audio")

# dataset -> [(column label, experiment name)], in the order the columns appear
DATASETS: Dict[str, List[Tuple[str, str]]] = {
    "speech_commands": [("CNN 247K", "audio_kn_speech_commands"),
                        ("CNN-wide 1.58M", "audio_kn_speech_commands_wide"),
                        ("ResNet-18 11.2M", "audio_kn_speech_commands_resnet18")],
    "esc50": [("CNN 247K", "audio_kn_esc50"),
              ("CNN-wide 1.58M", "audio_kn_esc50_wide"),
              ("ResNet-18 11.2M", "audio_kn_esc50_resnet18")],
    "urbansound8k": [("CNN 247K", "audio_kn_urbansound8k"),
                     ("CNN-wide 1.58M", "audio_kn_urbansound8k_wide"),
                     ("ResNet-18 11.2M", "audio_kn_urbansound8k_resnet18")],
}
PRETTY = {"speech_commands": "Speech Commands v2", "esc50": "ESC-50",
          "urbansound8k": "UrbanSound8K"}
SIGNALS = [("epistemic (ours)", "epistemic"), ("total entropy", "total"),
           ("input (model-free)", "input")]
ALGOS = ["adwin", "kswin", "ph"]
KNOB = {"adwin": ("delta", "δ"), "kswin": ("alpha", "α"), "ph": ("threshold", "thr")}


def _auroc(v: np.ndarray, lab: np.ndarray) -> float:
    finite = np.isfinite(v)
    v, lab = v[finite], lab[finite]
    if lab.sum() == 0 or (1 - lab).sum() == 0:
        return float("nan")
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v))
    ranks[order] = np.arange(1, len(v) + 1)
    n1, n0 = lab.sum(), (1 - lab).sum()
    return float((ranks[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _complete_runs(exp: str) -> list[Path]:
    """Runs with a metrics.json — i.e. finished. A directory alone is not enough:
    the runner creates it up front, so an in-flight arm looks present but has no
    steps.jsonl yet."""
    return [r for r in sorted(Path("results/raw").glob(f"{exp}/*/"))
            if (r / "metrics.json").exists() and (r / "steps.jsonl").exists()]


def _load(exp: str) -> tuple[list, int, dict]:
    runs = _complete_runs(exp)
    if not runs:
        raise FileNotFoundError(exp)
    run = runs[-1]
    steps = [json.loads(x) for x in (run / "steps.jsonl").read_text().splitlines() if x]
    metrics = json.loads((run / "metrics.json").read_text())
    novel_start = int((metrics["stream"]["extras"])["novel_start_batch"])
    return steps, novel_start, metrics


def _signal_series(steps: list, signal: str) -> np.ndarray:
    if signal == "input":
        return np.array([(s.get("detectors") or {}).get("adwin_input", {}).get("statistic", np.nan)
                         for s in steps], float)
    return np.array([s.get(f"mean_{signal}", np.nan) for s in steps], float)


def _alarms(steps: list, det: str, novel_start: int) -> tuple[int, float]:
    fa = sum(1 for s in steps if s["step"] < novel_start
             and (s.get("detectors") or {}).get(det, {}).get("alarm"))
    post = [s["step"] for s in steps if s["step"] >= novel_start
            and (s.get("detectors") or {}).get(det, {}).get("alarm")]
    return fa, (post[0] - novel_start if post else float("inf"))


def _render(rows: List[List[str]], row_labels: List[str], col_labels: List[str],
            title: str, caption: str, out: Path, header_rows: set[int],
            colours: List[List[str]] | None = None, col_w: float = 0.115) -> Path:
    import matplotlib.pyplot as plt

    n_rows, n_cols = len(rows), len(col_labels)
    fig_w = 2.6 + col_w * 26 * n_cols / 3
    # Height tracks the row count directly; the table is the whole axes, so there
    # is no dead space under it (bbox_inches="tight" then crops the margins).
    fig, ax = plt.subplots(figsize=(max(fig_w, 6.5), 0.30 * (n_rows + 1) + 1.0))
    ax.axis("off")
    # Reserve strips for the title and the caption; the table then fills the rest
    # exactly, via explicit cell heights below. Scaling instead leaves the table
    # smaller than the axes and the figure ends up mostly blank.
    ax.set_position([0.0, 0.10, 1.0, 0.80])
    tbl = ax.table(cellText=rows, rowLabels=row_labels, colLabels=col_labels,
                   cellLoc="center", rowLoc="right", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    row_h = 1.0 / (n_rows + 1)
    for cell in tbl.get_celld().values():
        cell.set_height(row_h)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.5)
        if r == 0 or c == -1:
            cell.set_facecolor("#ececec")
            cell.set_text_props(fontweight="bold" if r == 0 else "normal")
        if r - 1 in header_rows and r > 0:
            cell.set_facecolor("#f4f4f4")
            cell.set_text_props(fontstyle="italic", fontweight="bold")
        if colours and r > 0 and c >= 0:
            col = colours[r - 1][c]
            if col:
                cell.set_facecolor(col)
    ax.set_title(title, fontweight="bold", fontsize=11, pad=14)
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=7.5, color="#444",
             wrap=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


def _auroc_colour(v: float) -> str:
    if not np.isfinite(v):
        return "#f2f2f2"
    return "#d8ecd8" if v >= 0.9 else "#fdf3d7" if v >= 0.7 else "#f8dcdc"


def results_table(dataset: str) -> Path:
    cols = DATASETS[dataset]
    rows: List[List[str]] = []
    row_labels: List[str] = []
    colours: List[List[str]] = []
    headers: set[int] = set()

    cols = [(lb, e) for lb, e in cols if _complete_runs(e)]
    per_col = {}
    for label, exp in cols:
        steps, ns, _ = _load(exp)
        per_col[label] = (steps, ns, (np.arange(len(steps)) >= ns).astype(int))

    for sig_label, sig in SIGNALS:
        headers.add(len(rows))
        rows.append([""] * len(cols))
        row_labels.append(sig_label)
        colours.append([""] * len(cols))

        auroc_row, auroc_col = [], []
        for label, _ in cols:
            steps, ns, lab = per_col[label]
            a = _auroc(_signal_series(steps, sig), lab)
            auroc_row.append(f"{a:.3f}")
            auroc_col.append(_auroc_colour(a))
        rows.append(auroc_row)
        row_labels.append("AUROC")
        colours.append(auroc_col)

        for algo in ALGOS:
            cells, cols_c = [], []
            for label, _ in cols:
                steps, ns, _ = per_col[label]
                fa, delay = _alarms(steps, f"{algo}_{sig}", ns)
                d = "never" if not np.isfinite(delay) else f"{int(delay)}"
                cells.append(f"{fa}  /  {d}")
                # Green needs BOTH: quiet before the boundary and prompt after it.
                # Zero false alarms with a 23-batch delay is not a good operating
                # point, and colouring it green would say the opposite.
                if not np.isfinite(delay):
                    cols_c.append("#f8dcdc")
                elif fa == 0 and delay <= 5:
                    cols_c.append("#d8ecd8")
                elif fa <= 2 and delay <= 10:
                    cols_c.append("#fdf3d7")
                else:
                    cols_c.append("#f8dcdc")
            rows.append(cells)
            row_labels.append(f"{algo}: FA / delay")
            colours.append(cols_c)

    headers.add(len(rows))
    rows.append([""] * len(cols))
    row_labels.append("supervised (labels)")
    colours.append([""] * len(cols))
    cells, cols_c = [], []
    for label, _ in cols:
        steps, ns, lab = per_col[label]
        err = 1.0 - np.array([s["accuracy"] for s in steps], float)
        a = _auroc(err, lab)
        fa, delay = _alarms(steps, "ddm", ns)
        d = "never" if not np.isfinite(delay) else f"{int(delay)}"
        cells.append(f"{a:.3f}   {fa} / {d}")
        cols_c.append(_auroc_colour(a))
    rows.append(cells)
    row_labels.append("ddm: AUROC  FA/delay")
    colours.append(cols_c)

    steps, ns, _ = per_col[cols[0][0]]
    return _render(
        rows, row_labels, [c for c, _ in cols],
        f"{PRETTY[dataset]} — known vs novel corruptions",
        f"AUROC of the signal (threshold-free, one value per signal — all three algorithms read it).  "
        f"FA = alarms in the {ns} known-phase batches;  delay = batches from the boundary to the first alarm.  "
        f"Green: AUROC>=0.9, or FA=0 with delay<=5.  Red: AUROC<0.7, never detected, "
        f"or FA>2 / delay>10.",
        OUT_DIR / f"results_{dataset}.png", headers, colours)


def knobs_table(dataset: str) -> Path:
    cols = DATASETS[dataset]
    rows: List[List[str]] = []
    row_labels: List[str] = []
    headers: set[int] = set()
    cols = [(lb, e) for lb, e in cols if _complete_runs(e)]
    params = {}
    for label, exp in cols:
        _, _, metrics = _load(exp)
        params[label] = {d["name"]: d["params"] for d in metrics["detectors"]}

    for sig_label, sig in SIGNALS:
        headers.add(len(rows))
        rows.append([""] * len(cols))
        row_labels.append(sig_label)
        for algo in ALGOS:
            key, sym = KNOB[algo]
            cells = []
            for label, _ in cols:
                v = params[label].get(f"{algo}_{sig}", {}).get(key)
                cells.append("—" if v is None else f"{sym} = {float(v):.4g}")
            rows.append(cells)
            row_labels.append(algo)

    headers.add(len(rows))
    rows.append([""] * len(cols))
    row_labels.append("supervised (labels)")
    for det, key, sym in (("ddm", "drift_threshold", "thr"), ("eddm", "beta", "β")):
        cells = []
        for label, _ in cols:
            v = params[label].get(det, {}).get(key)
            cells.append("—" if v is None else f"{sym} = {float(v):.4g}")
        rows.append(cells)
        row_labels.append(det)

    return _render(
        rows, row_labels, [c for c, _ in cols],
        f"{PRETTY[dataset]} — swept detector knobs",
        "Only the knob the sweep sets is shown. Held fixed in every arm: ADWIN nothing else; "
        "KSWIN window=20, stat_size=8; PageHinkley delta=0.00327 (0.0192 on input), "
        "min_instances=10 (5 on input), alpha=0.9999; DDM warm_start=8000; EDDM alpha=0.95, warm_start=300.  "
        "Swept per arm at a zero-false-alarm budget with --select closest.",
        OUT_DIR / f"knobs_{dataset}.png", headers)


def main() -> int:
    written = []
    for dataset in DATASETS:
        written.append(results_table(dataset))
        written.append(knobs_table(dataset))
    for p in written:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
