"""Command-line entry point."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from uncertainty_driven_drift.config import load_config
from uncertainty_driven_drift.runner import run_experiment


def _print_detector_hyperparams(run_dir: Path) -> None:
    """Pretty-print the resolved detector hyperparameters from metrics.json."""
    path = Path(run_dir) / "metrics.json"
    try:
        metrics = json.loads(path.read_text())
    except FileNotFoundError:
        return
    dets = metrics.get("detectors") or []
    if not dets:
        return
    print("\nDetector hyperparameters:")
    name_w = max(len(str(d.get("name", ""))) for d in dets)
    type_w = max(len(str(d.get("type", ""))) for d in dets)
    for d in dets:
        params = d.get("params") or {}
        shown = {k: v for k, v in params.items() if k != "name"}
        items = ", ".join(f"{k}={v}" for k, v in sorted(shown.items()))
        print(
            f"  {str(d.get('name','')).ljust(name_w)}  "
            f"[{str(d.get('type','')).ljust(type_w)}]  {items}"
        )


def _load_entry_modules(paths: Iterable[str]) -> None:
    """Import user-supplied modules so their ``@register`` decorators fire."""
    for dotted in paths:
        dotted = dotted.strip()
        if dotted:
            importlib.import_module(dotted)


_DEFAULT_WITH = (
    "uncertainty_driven_drift.components.builtin,"
    "uncertainty_driven_drift.components.phase2,"
    "uncertainty_driven_drift.components.phase3"
)

_ALL_FIGURES = [
    "tv_vs_uncertainty",
    "scenarios_grid",
    "mnist_tasks_panel",
    "mnist_c_panel",
    "drift_panel",
    "summary_table",
    "hyperparams_table",
    "auroc_table",
    "far_table",
]


def _parse_runs_map(runs_str: str) -> dict:
    """Parse 'Label1:path1,Label2:path2,...' into an ordered dict."""
    runs_map = {}
    for entry in runs_str.split(","):
        entry = entry.strip()
        if ":" in entry:
            label, path = entry.split(":", 1)
            runs_map[label.strip()] = path.strip()
        else:
            runs_map[Path(entry).name] = entry
    return runs_map


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="uncertainty-driven-drift")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run one experiment from a YAML config")
    p_run.add_argument("--config", required=True, type=str)
    # Error bars are the one thing every result in this repo is missing (audited:
    # no experiment has ever been run at more than one seed), and this is the only
    # way to get them without copying a config per seed. Runs get a _seed<N>
    # experiment suffix so they land in distinct directories.
    #
    # Two scopes, because they cost very different amounts and answer different
    # questions. The distinction is NOT cosmetic: model training is seeded by
    # model.params.pretrain_seed, not by the global cfg.seed, and the checkpoint
    # is cached — so overriding the global seed alone leaves the backbone
    # bit-identical and gives you replicates of the stream, not of the result.
    p_run.add_argument(
        "--seed", type=int, default=None,
        help="Override seeds for a replicate run; appends _seed<N> to the "
             "experiment name. See --seed-scope for what actually varies.",
    )
    p_run.add_argument(
        "--seed-scope", choices=["stream", "full"], default="stream",
        help="stream (default): vary the global seed, the stream's sampling seed "
             "and MC sampling; the BACKBONE IS REUSED, so error bars cover stream "
             "and inference noise only. full: also vary pretrain_seed and give the "
             "checkpoint a per-seed path, so each replicate trains its own "
             "backbone — the honest error bar, at N times the training cost.",
    )
    p_run.add_argument(
        "--with",
        dest="with_modules",
        default=_DEFAULT_WITH,
        help="Comma-separated dotted paths of modules to import before running; "
             "use these to register custom components.",
    )

    p_plot = sub.add_parser("plot", help="Generate figures for a completed run")
    p_plot.add_argument("--run", type=str,
                        help="Path to a single run directory.")
    p_plot.add_argument("--runs", type=str,
                        help="Comma-separated run directories; for multi-run "
                             "figures use 'Label:path,...' format.")
    p_plot.add_argument("--out", type=str,
                        help="Output directory. Defaults to "
                             "<run_dir>/figures/<timestamp>/ for single-run "
                             "figures.")
    p_plot.add_argument("--start", type=int, default=20,
                        help="Clip: only plot step >= start (scenarios_grid).")
    p_plot.add_argument("--end", type=int, default=100,
                        help="Clip: only plot step <= end (scenarios_grid).")
    p_plot.add_argument(
        "--figure", default="tv_vs_uncertainty",
        choices=_ALL_FIGURES,
        help="Which figure to emit (legacy; prefer --figures).",
    )
    p_plot.add_argument(
        "--figures",
        default=None,
        help="Comma-separated list of figures to generate, or 'all'. "
             "Overrides --figure when set. Choices: " + ", ".join(_ALL_FIGURES),
    )
    p_plot.add_argument("--title", type=str, default=None,
                        help="Custom title (used by drift_panel).")
    p_plot.add_argument("--name", type=str, default=None,
                        help="Override the output filename stem (e.g. 'wireless_kn_comparison').")
    p_plot.add_argument("--smoothing", default="rolling",
                        choices=["none", "rolling", "cumulative"])
    p_plot.add_argument("--window", type=int, default=10,
                        help="Window length for --smoothing=rolling.")
    p_plot.add_argument("--no-raw-overlay", action="store_true",
                        help="Suppress the faded raw backdrop (where applicable).")
    p_plot.add_argument("--no-timestamp", action="store_true",
                        help="Do not create a timestamped output subdirectory.")
    p_plot.add_argument("--far", type=float, default=0.05,
                        help="Target false-alarm rate for far_table (default 0.05).")

    args = parser.parse_args(argv)

    if args.command == "run":
        _load_entry_modules(args.with_modules.split(","))
        cfg = load_config(args.config)
        if args.seed is not None:
            seed = int(args.seed)
            cfg.seed = seed
            if "seed" in cfg.dataset.params:
                cfg.dataset.params["seed"] = seed
            if "seed" in cfg.model.params:
                cfg.model.params["seed"] = seed          # MC sampling / weight draws
            scope = args.seed_scope
            if scope == "full":
                cfg.model.params["pretrain_seed"] = seed
                ckpt = cfg.model.params.get("ckpt_path")
                if ckpt:
                    stem, dot, ext = str(ckpt).rpartition(".")
                    cfg.model.params["ckpt_path"] = (
                        f"{stem}_seed{seed}{dot}{ext}" if dot else f"{ckpt}_seed{seed}"
                    )
            cfg.experiment = f"{cfg.experiment}_seed{seed}"
            print(f"[cli] seed override ({scope}): {cfg.experiment}")
            if scope == "stream":
                print("[cli]   backbone is REUSED — these error bars cover stream and "
                      "inference noise, not training. Use --seed-scope full for that.")
            else:
                print(f"[cli]   own backbone: {cfg.model.params.get('ckpt_path')}")
        out_dir = run_experiment(cfg)
        print(f"Wrote run to {out_dir}")
        _print_detector_hyperparams(out_dir)
        return 0

    if args.command == "plot":
        from uncertainty_driven_drift.analysis.plots import (
            plot_auroc_table,
            plot_drift_panel,
            plot_far_table,
            plot_hyperparams_table,
            plot_mnist_c_panel,
            plot_mnist_tasks_panel,
            plot_scenarios_grid,
            plot_summary_table,
            plot_tv_vs_uncertainty,
        )

        show_raw = not args.no_raw_overlay
        use_timestamp = not args.no_timestamp

        # Determine which figures to generate.
        if args.figures:
            if args.figures.strip().lower() == "all":
                requested = list(_ALL_FIGURES)
            else:
                requested = [f.strip() for f in args.figures.split(",") if f.strip()]
        else:
            requested = [args.figure]

        # Determine output directory.
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.out:
            out_dir = Path(args.out)
        elif args.run:
            subdir = ts if use_timestamp else "latest"
            out_dir = Path(args.run) / "figures" / subdir
        else:
            subdir = ts if use_timestamp else "latest"
            out_dir = Path(".") / "figures" / subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build runs_map for multi-run figures.
        runs_map: dict = {}
        if args.runs:
            runs_map = _parse_runs_map(args.runs)

        written: List[Path] = []

        for fig in requested:

            if fig == "tv_vs_uncertainty":
                if not args.run:
                    print(f"[skip] {fig}: --run required", file=sys.stderr)
                    continue
                out = plot_tv_vs_uncertainty(
                    args.run,
                    out_path=out_dir / "tv_vs_uncertainty.png",
                    smoothing=args.smoothing,
                    window=args.window,
                    show_raw=show_raw,
                    timestamp=False,
                )
                written.append(out)

            elif fig == "scenarios_grid":
                if not runs_map:
                    print(f"[skip] {fig}: --runs required", file=sys.stderr)
                    continue
                dirs = list(runs_map.values())
                out = plot_scenarios_grid(
                    dirs,
                    out_path=out_dir / "scenarios_grid.png",
                    start=args.start, end=args.end,
                    smoothing=args.smoothing,
                    window=args.window,
                    title=args.title if args.title else None,
                    timestamp=False,
                )
                written.append(out)

            elif fig == "mnist_tasks_panel":
                if not args.run:
                    print(f"[skip] {fig}: --run required", file=sys.stderr)
                    continue
                out = plot_mnist_tasks_panel(
                    args.run,
                    out_path=out_dir / "mnist_tasks_panel.png",
                    smoothing=args.smoothing,
                    window=args.window,
                    show_raw=show_raw,
                    timestamp=False,
                )
                written.append(out)

            elif fig == "mnist_c_panel":
                if not args.run:
                    print(f"[skip] {fig}: --run required", file=sys.stderr)
                    continue
                fig_stem = args.name if args.name else "mnist_c_panel"
                out = plot_mnist_c_panel(
                    args.run,
                    out_path=out_dir / f"{fig_stem}.png",
                    smoothing=args.smoothing,
                    window=args.window,
                    show_raw=show_raw,
                    timestamp=False,
                    title=args.title if args.title else None,
                )
                written.append(out)

            elif fig == "drift_panel":
                if not args.run:
                    print(f"[skip] {fig}: --run required", file=sys.stderr)
                    continue
                out = plot_drift_panel(
                    args.run,
                    title=args.title,
                    out_path=out_dir / "drift_panel.png",
                    smoothing=args.smoothing,
                    window=args.window,
                    timestamp=False,
                )
                written.append(out)

            elif fig == "summary_table":
                if not runs_map:
                    print(f"[skip] {fig}: --runs required", file=sys.stderr)
                    continue
                out = plot_summary_table(
                    runs_map,
                    out_path=out_dir / "summary_table.png",
                    timestamp=False,
                )
                written.append(out)

            elif fig == "hyperparams_table":
                target = runs_map if runs_map else (
                    {Path(args.run).parent.name: args.run} if args.run else {}
                )
                if not target:
                    print(f"[skip] {fig}: --run or --runs required", file=sys.stderr)
                    continue
                out = plot_hyperparams_table(
                    target,
                    out_path=out_dir / "hyperparams_table.png",
                    timestamp=False,
                )
                written.append(out)

            elif fig == "auroc_table":
                if not runs_map:
                    print(f"[skip] {fig}: --runs required", file=sys.stderr)
                    continue
                out = plot_auroc_table(
                    runs_map,
                    out_path=out_dir / "auroc_table.png",
                    timestamp=False,
                )
                written.append(out)

            elif fig == "far_table":
                if not runs_map:
                    print(f"[skip] {fig}: --runs required", file=sys.stderr)
                    continue
                out = plot_far_table(
                    runs_map,
                    out_path=out_dir / "far_table.png",
                    target_far=args.far,
                    timestamp=False,
                )
                written.append(out)

        # Always generate hyperparams table alongside any other figure
        # (unless it was already explicitly requested or there's no run to read).
        if "hyperparams_table" not in requested:
            target = runs_map if runs_map else (
                {Path(args.run).parent.name: args.run} if args.run else {}
            )
            if target:
                hp_out = plot_hyperparams_table(
                    target,
                    out_path=out_dir / "hyperparams_table.png",
                    timestamp=False,
                )
                written.append(hp_out)

        for p in written:
            print(f"Wrote {p}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
