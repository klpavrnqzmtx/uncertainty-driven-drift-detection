#!/usr/bin/env python3
"""Compact, conference-ready version of the known-vs-novel detector panel.

Separate from the main ``plot_mnist_c_panel`` figure — this one is tuned for a
conference full-text-width slot (~5.5in): small fonts, no on-figure title
(put the description in the caption), trimmed legends/labels, and vector (PDF)
output. It reuses the data-loading / grouping helpers from
``uncertainty_driven_drift.analysis.plots`` so it stays in sync with the main figure's
content while owning its own layout.

Usage
-----
    python scripts/paper_panel.py --run results/raw/cifar_kn_comparison/<ts> \
        --out results/figures/paper/cifar_kn_mc

Two-panel layout: top error/BER, bottom detector-alarm raster (DDM supervised
baseline + epistemic and total unsupervised detectors + input baseline).
Writes ``<out>.png`` only (no PDF).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from uncertainty_driven_drift.analysis import figure_style as FS
from uncertainty_driven_drift.analysis import plots as P


def make_paper_panel(run_dir: Path, out_stem: Path, width: float = 5.5,
                     smoothing: str = "rolling", window: int = 10,
                     tag_rotation: int | None = None,
                     detectors: list[str] | None = None) -> Path:
    import matplotlib.pyplot as plt

    steps = P._load_steps(run_dir)
    metrics = P._load_metrics(run_dir)
    t = np.array([r["step"] for r in steps])

    ber = [r.get("ber") for r in steps]
    has_ber = any(v is not None for v in ber)
    if has_ber:
        err_raw = np.array([v if v is not None else np.nan for v in ber])
        err_ylabel = "BER"
    else:
        err_raw = 1.0 - P._series(steps, "accuracy")
        err_ylabel = "error rate"
    err = P._smooth(err_raw, smoothing, window)

    stream = metrics.get("stream", {}) or {}
    extras = stream.get("extras") or {}
    drift_indices = list(stream.get("drift_indices") or [])
    novel_start = extras.get("novel_start_batch")
    channel_names = extras.get("channel_names")

    names = sorted({n for r in steps for n in (r.get("detectors") or {})})
    alarms = {n: np.array([bool((r.get("detectors") or {}).get(n, {}).get("alarm", False))
                           for r in steps]) for n in names}
    groups = P._group_detectors(names)
    ordered = P._ordered_detector_rows(groups)
    # Paper panel: supervised baseline is DDM only (drop EDDM to declutter).
    ordered = [(g, det) for (g, det) in ordered if det != "eddm"]
    if detectors:
        # Substring match, so `--detectors epistemic,total,ddm` keeps the three
        # algorithms on each of those signals without naming all nine.
        wanted = [w.strip() for w in detectors if w.strip()]
        ordered = [(g, det) for (g, det) in ordered
                   if any(w in det for w in wanted)]
        if not ordered:
            raise SystemExit(
                f"--detectors {detectors} matched nothing; available: "
                f"{sorted(names)}"
            )
    n_rows = len(ordered)

    # -- house style (palette, typography, naming) in analysis/figure_style ---
    rc = FS.rc(base=7.0)
    det_h = max(0.19 * n_rows, 0.85)
    height = 0.95 + det_h + 0.45

    with plt.rc_context(rc):
        fig, axes = plt.subplots(
            2, 1, figsize=(width, height), dpi=200, sharex=True,
            gridspec_kw={"height_ratios": [0.85, det_h], "hspace": 0.12},
        )
        ax_err, ax_det = axes

        # 1) error / BER
        if has_ber:
            err = np.where(err > 0, err, np.nan)
        ax_err.plot(t, err, color=FS.INK, lw=1.2, zorder=3)
        ax_err.set_ylabel(err_ylabel, labelpad=3)
        if has_ber:
            ax_err.set_yscale("log"); ax_err.set_ylim(1e-3, 1.0)
        else:
            # Round ticks only. The previous version derived the top from the data
            # (1.02) and halved it, which printed "0.51" on the axis.
            _emax = np.nanmax(err) if np.isfinite(np.nanmax(err)) else 1.0
            _top = 0.5 if _emax <= 0.45 else 1.0
            ax_err.set_ylim(-0.02 * _top, _top * 1.04)
            ax_err.set_yticks([0.0, _top / 2, _top])
            ax_err.set_yticklabels([f"{v:g}" for v in (0.0, _top / 2, _top)])
        ax_err.grid(axis="y")

        # phase / corruption tags along the very top of the error panel.
        # CIFAR/wireless expose channel_names + drift_indices; the MNIST-family
        # streams expose pure `segments` with per-block corruption names.
        tags = []  # (mid, label, is_novel)
        if channel_names and drift_indices:
            bounds = [0] + sorted(drift_indices) + [int(t[-1]) + 1]
            for i, c in enumerate(channel_names[: len(bounds) - 1]):
                mid = 0.5 * (bounds[i] + bounds[i + 1])
                tags.append((mid, FS.channel_label(c),
                             novel_start is not None and bounds[i] >= novel_start))
        else:
            for seg in (extras.get("segments") or []):
                if seg.get("kind") != "pure":
                    continue
                s, e = int(seg["start"]), int(seg["end"])
                tags.append((0.5 * (s + e), FS.channel_label(seg.get("corruption_a", "?")),
                             novel_start is not None and s >= novel_start))
        if tags:
            from matplotlib.transforms import blended_transform_factory
            tr = blended_transform_factory(ax_err.transData, ax_err.transAxes)
            # Longer names (MNIST 'gaussian_noise') are angled to avoid overlap.
            # A stream with few, wide phases has room to lay them flat, so allow the
            # caller to override the length heuristic.
            rot = (tag_rotation if tag_rotation is not None
                   else (0 if max(len(lbl) for _, lbl, _ in tags) <= 9 else 25))
            for mid, label, novel in tags:
                ax_err.text(mid, 1.03, label, transform=tr, ha="center", va="bottom",
                            fontsize=5.5, rotation=rot,
                            # Known phases in black, not muted grey (hard to read);
                            # novel phases keep their own dark accent.
                            color=FS.NOVEL_INK if novel else FS.INK)

        # 2) detector alarms (drift lines BEHIND markers; halo so on-line alarms show)
        # Alternating faint bands instead of dashed rules: a dashed line reads as
        # a threshold, and these are only phase boundaries. The one line that IS
        # meaningful — the known->novel boundary — stays solid, and is labelled.
        bounds_all = [0] + sorted(drift_indices) + [int(t[-1]) + 1]
        for ax in (ax_err, ax_det):
            for i in range(len(bounds_all) - 1):
                if i % 2 == 1:
                    ax.axvspan(bounds_all[i], bounds_all[i + 1], color=FS.BAND,
                               lw=0, zorder=0)
            if novel_start is not None:
                ax.axvline(novel_start, color=FS.INK, lw=1.0, alpha=0.9, zorder=2)
        if novel_start is not None:
            # Sits INSIDE the error panel, clear of the tag row above it: anchored
            # at 1.0 it read as floating between the tags and the plot.
            ax_err.annotate(f"novel onset  $t$ = {int(novel_start)}",
                            xy=(novel_start, 0.86), xycoords=("data", "axes fraction"),
                            xytext=(5, 0), textcoords="offset points",
                            ha="left", va="center", fontsize=5.5,
                            color=FS.NOVEL_INK)
        row = 0
        yticks, ylabels, seps = [], [], []
        prev = None
        for gi, (g, det) in enumerate(ordered):
            if prev is not None and g != prev:
                seps.append(row - 0.5)
            prev = g
            tt = t[alarms.get(det, np.zeros(len(t), bool))]
            yy = np.full(len(tt), row)
            # Filled per-signal shapes rather than the family's line glyphs: a
            # line marker ('|') has no fill, so it needs linewidths to exist at
            # all, and it cannot carry a surface ring. The ring matters because
            # an alarm can land exactly on the boundary rule.
            m = FS.signal_marker(det)
            ax_det.scatter(tt, yy, marker=m, s=30, color="white",
                           linewidths=0.0, zorder=3)
            ax_det.scatter(tt, yy, marker=m, s=13, color=FS.signal_color(det),
                           linewidths=0.0, zorder=4)
            yticks.append(row)
            ylabels.append(FS.detector_label(det))
            row += 1
        for sep in seps:
            ax_det.axhline(sep, color=FS.RULE, lw=0.5, zorder=1)
        ax_det.set_yticks(yticks)
        ax_det.set_yticklabels(ylabels, fontsize=6)
        ax_det.set_ylim(-0.7, row - 0.3)
        ax_det.grid(axis="x")
        # Row labels on the RIGHT, "alarms" on the left. The detector names are
        # long, and on the left they pushed the plot area rightwards and left the
        # error panel's own label crowded against them.
        ax_det.yaxis.tick_right()
        ax_det.yaxis.set_label_position("left")
        ax_det.set_ylabel("alarms", labelpad=3)
        # The label already names the signal in words; the colour repeats it so
        # the row and its marks read as one object.
        for lbl, (_g, det) in zip(ax_det.get_yticklabels(), ordered):
            lbl.set_color(FS.signal_color(det))

        ax_det.set_xlabel(r"batch index $t$")
        ax_det.invert_yaxis()   # first row at the top, reading order
        # No y-label alignment: let the error label sit close to its own panel
        # (the detector rows keep their own tick labels).
        fig.subplots_adjust(left=0.105, right=0.735, top=0.88, bottom=0.12)

        out_stem.parent.mkdir(parents=True, exist_ok=True)
        png = out_stem.with_suffix(".png")
        # PNG only, by request: no PDF copy.
        fig.savefig(png, bbox_inches="tight", dpi=300)
        plt.close(fig)
    return png


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="Run directory (steps.jsonl + metrics.json).")
    ap.add_argument("--out", required=True, help="Output path stem (no extension).")
    ap.add_argument("--width", type=float, default=5.5, help="Figure width in inches.")
    ap.add_argument("--smoothing", default="rolling", choices=["none", "rolling", "cumulative"])
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--detectors", default=None,
                    help="Comma-separated substrings selecting which detector rows to "
                         "draw, e.g. 'epistemic,total' for just the two uncertainty "
                         "signals, or 'epistemic,total,ddm' to keep the supervised "
                         "reference. Default: every detector in the run.")
    ap.add_argument("--tag-rotation", type=int, default=None,
                    help="Rotation for the phase tags above the error panel. Default "
                         "auto: flat for short names, 30 degrees for long ones. Pass 0 "
                         "when a stream has few wide phases with room to lay them flat.")
    a = ap.parse_args(argv)
    png = make_paper_panel(Path(a.run), Path(a.out), a.width, a.smoothing, a.window,
                           a.tag_rotation,
                           detectors=(a.detectors.split(",") if a.detectors else None))
    # No on-figure title (the filename identifies the arm), but the arm's
    # human-readable name is still worth having for the caption.
    caption = FS.arm_title(P._load_metrics(Path(a.run)))
    print(f"wrote {png}"
          + (f"   [{caption}]" if caption else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
