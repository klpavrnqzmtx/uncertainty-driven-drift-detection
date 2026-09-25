#!/usr/bin/env bash
# Fetch the three audio datasets. RUN ON A MACHINE WITH INTERNET (login node or
# workstation); compute nodes have none.
#
#     bash scripts/download_audio.sh                  # all three
#     DATASETS="esc50" bash scripts/download_audio.sh # just one
#
# All three transfers are resumable (curl -C -) and verified by size, then by
# `gzip -t`. That second check is the one that matters: an interrupted transfer
# that is later resumed by a *second* concurrent client produces a file of the
# right-ish size with trailing garbage, which only shows up as a decode error
# hours later. Ask me how I know.
#
# After this: python scripts/prepare_audio.py --dataset all
set -uo pipefail

AUDIO_ROOT="${AUDIO_ROOT:-./artifacts/audio}"
DATASETS="${DATASETS:-speech_commands esc50 urbansound8k}"

url_for() {
  case "$1" in
    speech_commands) echo "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz" ;;
    esc50)           echo "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.tar.gz" ;;
    urbansound8k)    echo "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz" ;;
    *) return 1 ;;
  esac
}

file_for() {
  case "$1" in
    speech_commands) echo "speech_commands_v0.02.tar.gz" ;;
    esc50)           echo "ESC-50-master.tar.gz" ;;
    urbansound8k)    echo "UrbanSound8K.tar.gz" ;;
    *) return 1 ;;
  esac
}

# Expected sizes in bytes, as served at the time of writing. Used only as a hint
# in the log — `gzip -t` is the actual gate.
size_for() {
  case "$1" in
    speech_commands) echo 2428923189 ;;
    esc50)           echo 647224334 ;;
    urbansound8k)    echo 6023741708 ;;
    *) echo 0 ;;
  esac
}

rc=0
for ds in $DATASETS; do
  url="$(url_for "$ds")" || { echo "unknown dataset $ds" >&2; rc=1; continue; }
  dest_dir="$AUDIO_ROOT/$ds"
  dest="$dest_dir/$(file_for "$ds")"
  mkdir -p "$dest_dir"

  echo "== $ds -> $dest"
  if [ -f "$dest" ] && gzip -t "$dest" 2>/dev/null; then
    echo "   present and valid, skipping"
    continue
  fi
  if [ -f "$dest" ]; then
    echo "   partial ($(du -h "$dest" | cut -f1) of ~$(( $(size_for "$ds") / 1000000 )) MB) — resuming"
  fi
  curl -L -C - --retry 8 --retry-delay 5 --retry-connrefused --connect-timeout 30 \
       --progress-bar -o "$dest" "$url" || { echo "   transfer failed" >&2; rc=1; continue; }

  if ! gzip -t "$dest" 2>/dev/null; then
    echo "   ERROR: $dest is not a valid gzip archive." >&2
    echo "          Delete it and re-run — a resumed-over-a-live-download file cannot be repaired." >&2
    rc=1
  else
    echo "   ok ($(du -h "$dest" | cut -f1))"
  fi
done

echo
if [ $rc -eq 0 ]; then
  du -sh "$AUDIO_ROOT"/* 2>/dev/null
  echo "Next: python scripts/prepare_audio.py --dataset all"
else
  echo "One or more datasets failed." >&2
fi
exit $rc
