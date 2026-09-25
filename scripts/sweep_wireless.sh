#!/usr/bin/env bash
# Calibrate detector thresholds for the wireless (QuaDRiGa) arms. Local — these
# are small RNN/MLP equalizers with existing checkpoints, so unlike the vision
# scaling arms this doesn't need a cluster GPU allocation.
#
#     bash scripts/sweep_wireless.sh              # report only
#     APPLY=1 bash scripts/sweep_wireless.sh       # also write thresholds into the configs
#     ONLY=snr_drift_urban_los_only APPLY=1 bash scripts/sweep_wireless.sh   # just one arm (substring match)
#
# Same rationale as the audio sweep: PageHinkley's threshold lives on
# the signal's own scale, and wireless BER/input-L2 statistics are nowhere near
# CIFAR's. Every threshold is chosen at the same false-alarm budget (5% of
# known-phase batches) for every signal and every arm, which is the only way a
# delay comparison between epistemic, total entropy and the input baseline
# means anything. This also re-sweeps DDM's drift_threshold under the fixed
# warm_start gate (samples, not batches) — every wireless selection.json on
# disk predates that fix and is not trustworthy.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
PY="${PY:-.venv/bin/python}"
command -v "$PY" >/dev/null || PY=python

SEEDS="${SEEDS:-7,8,9}"
TARGET_FAR="${TARGET_FAR:-0.05}"

# tag -> config, as two parallel arrays (macOS ships bash 3.2 — no `declare -A`).
TAGS=(
  wireless_los_drift_mc
  wireless_los_drift_laplace
  wireless_los_vs_nlos_mc
  wireless_los_vs_nlos_laplace
  wireless_snr_drift_mc
  wireless_snr_drift_urban_los_only_mc
  wireless_equalizer_mlp
  wireless_equalizer_mlp_diff
)
CFGS=(
  configs/experiments/wireless_setup/multitap_channel_drift/los_drift_highway_urban_indoor_mc_dropout.yaml
  configs/experiments/wireless_setup/multitap_channel_drift/los_drift_highway_urban_indoor_laplace.yaml
  configs/experiments/wireless_setup/multitap_channel_drift/train_los_test_nlos_mc_dropout.yaml
  configs/experiments/wireless_setup/multitap_channel_drift/train_los_test_nlos_laplace.yaml
  configs/experiments/wireless_setup/multitap_channel_drift/snr_drift_urban_mc_dropout.yaml
  configs/experiments/wireless_setup/multitap_channel_drift/snr_drift_urban_los_only_mc_dropout.yaml
  configs/experiments/wireless_setup/multitap_channel_drift/equalizer_mlp_raw_window.yaml
  configs/experiments/wireless_setup/multitap_channel_drift/equalizer_mlp_differential.yaml
)

rc=0
for i in "${!TAGS[@]}"; do
  tag="${TAGS[$i]}"
  cfg="${CFGS[$i]}"
  if [ -n "${ONLY:-}" ]; then
    case "$tag" in *"$ONLY"*) ;; *) continue ;; esac
  fi
  out="results/sweeps/$tag"
  echo "== sweep $tag  ($cfg) -> $out"

  if ! "$PY" - "$cfg" <<'PY'
import os, sys
sys.path.insert(0, "src")
from uncertainty_driven_drift.config import load_config
ck = load_config(sys.argv[1]).model.params.get("ckpt_path")
if not ck or not os.path.exists(ck):
    raise SystemExit(f"ERROR: {ck} does not exist. Train the arm first.")
print(f"  using checkpoint {ck}")
PY
  then
    rc=1; continue
  fi

  APPLY_ARG=()
  [ "${APPLY:-0}" = "1" ] && APPLY_ARG=(--apply-config "$cfg")
  "$PY" -u scripts/sweep_detectors.py \
      --config "$cfg" --mode known_novel \
      --seeds "$SEEDS" --target-far "$TARGET_FAR" \
      --out "$out" "${APPLY_ARG[@]}" || rc=1
done

echo
if [ $rc -eq 0 ]; then
  echo "Sweeps done. Operating curves + selection.json under results/sweeps/wireless_*."
  [ "${APPLY:-0}" = "1" ] && echo "Configs updated in place — commit the diff so the numbers stay reproducible."
else
  echo "One or more sweeps failed." >&2
fi
exit $rc
