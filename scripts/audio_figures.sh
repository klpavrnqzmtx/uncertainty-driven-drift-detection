#!/usr/bin/env bash
# Regenerate every audio figure and table from the committed runs.
#
#     bash scripts/audio_figures.sh
#
# Picks the LATEST run directory per experiment, so it stays correct as arms are
# re-run. Two kinds of output:
#
#   results/figures/paper/audio_*.png   one two-panel arm figure each
#                                             (error rate + alarm raster)
#   results/figures/audio/{auroc,far,summary,hyperparams}_table.png
#                                             the cross-arm tables, same
#                                             generators as the image arms
#
# The tables are the ones worth reading first: auroc_table is threshold-free,
# and far_table reports detection delay at matched false-alarm budgets (FAR=0
# and FAR=5%), which is the only apples-to-apples way to compare detectors whose
# knobs live on unrelated scales.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
PY="${PY:-.venv/bin/python}"
PAPER_DIR="${PAPER_DIR:-results/figures/paper}"
TABLE_DIR="${TABLE_DIR:-results/figures/audio}"
mkdir -p "$PAPER_DIR" "$TABLE_DIR"

# label -> experiment name. Label order is the column order in the tables:
# all three datasets on the 247 K CNN, then the same on ResNet-18, so a column
# pair reads as "same data, different backbone".
ARMS=(
  "SC-CNN:audio_kn_speech_commands"
  "ESC50-CNN:audio_kn_esc50"
  "US8K-CNN:audio_kn_urbansound8k"
  "SC-R18:audio_kn_speech_commands_resnet18"
  "ESC50-R18:audio_kn_esc50_resnet18"
  "US8K-R18:audio_kn_urbansound8k_resnet18"
  "SC-wide:audio_kn_speech_commands_wide"
  "ESC50-wide:audio_kn_esc50_wide"
  "US8K-wide:audio_kn_urbansound8k_wide"
)

latest_run() {   # latest_run <experiment> -> path or empty
  local exp="$1" d
  d=$(ls -d "results/raw/$exp"/*/ 2>/dev/null | sort | tail -1)
  [ -n "$d" ] && [ -f "${d}metrics.json" ] && echo "${d%/}"
}

RUNS_ARG=""
echo "== arms found"
for entry in "${ARMS[@]}"; do
  label="${entry%%:*}"; exp="${entry##*:}"
  run="$(latest_run "$exp")"
  if [ -z "$run" ]; then
    echo "   MISSING  $label  ($exp) — skipped"
    continue
  fi
  echo "   ok       $label  $run"
  RUNS_ARG+="${RUNS_ARG:+,}${label}:${run}"

  # ONE figure per arm by default: the panel the paper uses — all three detector
  # algorithms on our epistemic signal and on the total-entropy baseline, plus DDM
  # as the supervised reference. The input-based rows are omitted here because they
  # never fire at any usable operating point; they are in the auroc/far tables, and
  # AUDIO_FULL_PANELS=1 re-emits the full eleven-row raster as audio_<arm>_full
  # for debugging.
  stem="$PAPER_DIR/audio_$(echo "$label" | tr 'A-Z-' 'a-z_')"
  $PY scripts/paper_panel.py --run "$run" --out "$stem" \
      --detectors epistemic,total,ddm >/dev/null || \
    echo "   WARN: panel failed for $label"
  if [ "${AUDIO_FULL_PANELS:-0}" = "1" ]; then
    $PY scripts/paper_panel.py --run "$run" --out "${stem}_full" >/dev/null || \
      echo "   WARN: full panel failed for $label"
  fi
done

if [ -z "$RUNS_ARG" ]; then
  echo "No runs found. Run the arms first, e.g." >&2
  echo "  $PY -m uncertainty_driven_drift.cli run --config configs/experiments/audio_kn_comparison/esc50_known_vs_novel.yaml" >&2
  exit 1
fi

echo
echo "== per-dataset tables (readable: one column per backbone)"
$PY scripts/audio_tables.py || exit 1

echo
echo "== per-dataset detection delay at matched false-alarm budgets"
# The shared far_table generator, called once per dataset instead of once for all
# seven arms: 14 columns was unreadable, 2-3 is fine. It sweeps each detector's
# threshold internally to FAR=0 and FAR=5%, which the per-arm tables above cannot
# show (they report the single operating point the config carries).
for ds in speech_commands esc50 urbansound8k; do
  case $ds in
    speech_commands) sub="CNN:audio_kn_speech_commands,CNN-wide:audio_kn_speech_commands_wide,R18:audio_kn_speech_commands_resnet18" ;;
    esc50)           sub="CNN:audio_kn_esc50,CNN-wide:audio_kn_esc50_wide,R18:audio_kn_esc50_resnet18" ;;
    urbansound8k)    sub="CNN:audio_kn_urbansound8k,CNN-wide:audio_kn_urbansound8k_wide,R18:audio_kn_urbansound8k_resnet18" ;;
  esac
  args=""
  for entry in $(echo "$sub" | tr ',' ' '); do
    lbl="${entry%%:*}"; exp="${entry##*:}"
    run=$(ls -d "results/raw/$exp"/*/ 2>/dev/null | sort | tail -1)
    [ -n "$run" ] && args+="${args:+,}${lbl}:${run%/}"
  done
  [ -z "$args" ] && continue
  $PY -m uncertainty_driven_drift.cli plot --figures far_table --runs "$args" \
      --out "$TABLE_DIR/_far_$ds" >/dev/null \
    && mv "$TABLE_DIR/_far_$ds/far_table.png" "$TABLE_DIR/far_$ds.png" \
    && rm -rf "$TABLE_DIR/_far_$ds" \
    && echo "   wrote $TABLE_DIR/far_$ds.png"
done

echo
echo "== cross-arm overview (AUROC only; one row per signal is enough)"
$PY -m uncertainty_driven_drift.cli plot --figures auroc_table \
    --runs "$RUNS_ARG" --out "$TABLE_DIR" || exit 1
# The all-arms hyperparameter and far tables are deliberately NOT generated: at
# seven columns the parameter dumps overflowed their cells. knobs_<dataset>.png
# and far_<dataset>.png above carry the same information, legibly.
rm -f "$TABLE_DIR/hyperparams_table.png" "$TABLE_DIR/far_table.png" "$TABLE_DIR/summary_table.png"

echo
echo "== signal-level summary"
# shellcheck disable=SC2086
$PY scripts/screen_audio.py $(echo "$RUNS_ARG" | tr ',' '\n' | cut -d: -f2- | tr '\n' ' ') \
    --csv "$TABLE_DIR/screen_audio.csv" | tail -40
echo
echo "Panels: $PAPER_DIR/audio_*.png"
echo "Tables: $TABLE_DIR/"
