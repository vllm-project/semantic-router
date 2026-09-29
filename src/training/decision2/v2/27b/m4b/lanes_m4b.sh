#!/usr/bin/env bash
# M4b readout lane (node B host side): development readouts one after another on one GPU of GPU0-2, each through
# run_readout.sh. Items are NAME=CKPT or NAME=CKPT@TRAIN_RUN (arm-seeds). A failed readout is recorded in
# LANES.jsonl and the lane moves on.
# Usage: lanes_m4b.sh MIRROR_SHA GPU ITEM [ITEM ...]
set -uo pipefail
echo "$(date -u +%FT%TZ) m4b lane $* start"
SHA=$1 GPU=$2
shift 2
CODE=/data/dev2/src/$SHA/src/training/decision2
[ -d "$CODE" ] || CODE=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2
[ -f "$CODE/v2/27b/m4b/run_readout.sh" ] || { echo "no m4b code in mirror $SHA" >&2; exit 2; }
LOG=/data/dev2/runs/27b/m4b/LANES.jsonl
for item in "$@"; do
  name=${item%%=*} rest=${item#*=}
  ckpt=${rest%@*} train=""
  [ "$rest" != "$ckpt" ] && train=${rest#*@}
  if [ -f "/data/dev2/runs/27b/m4b/readouts/$name/READOUT-M4B.json" ]; then
    echo "$(date -u +%FT%TZ) $name already read; skipped"
    continue
  fi
  echo "$(date -u +%FT%TZ) lane GPU$GPU: $name"
  if TRAIN_RUN=$train bash "$CODE/v2/27b/m4b/run_readout.sh" "$name" "$ckpt" "$GPU" "$SHA"; then
    status=complete
  else
    status=failed
  fi
  printf '{"utc": "%s", "gpu": %s, "item": "%s", "status": "%s"}\n' "$(date -u +%FT%TZ)" "$GPU" "$name" "$status" >> "$LOG"
done
echo "$(date -u +%FT%TZ) m4b lane GPU$GPU done"
