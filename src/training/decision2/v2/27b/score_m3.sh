#!/usr/bin/env bash
# Milestone 3 development contrasts of the ~27B kernel-path readouts (node B host, CPU; never a
# release score).
# Usage: score_m3.sh SRC OUTPUT [LIMIT]
#   Arm-seeds: /data/dev2/runs/27b/M3-{A,S}-s{1,2}/readout-kernel-LIMIT with their trainer runs
#   (<arm-seed>/full/$(cat RUN_DIR)). Soups (optional): SOUP_A / SOUP_S readout directories; with
#   both seeds of an arm and its soup, the 17:15 soup rule picks that arm's candidate artifact.
#   OUTPUT is written once (v2/27b/m3_contrast.py).
set -euo pipefail

SRC=$1 OUTPUT=$2 LIMIT=${3:-32768}
S=/data/dev2/src/$SRC/src/training/decision2
R=/data/dev2/runs/27b
MX=/data/dev2/private/27b/m3-data/mixtures-m3-1
CAL_FILE=/data/dev2/hf-cache/datasets--llm-semantic-router--decision-2.0-training-data/snapshots/ed87a03ab80ca5b9560780bba51a83a77ff47d14/m2/cal/CAL698/cal.jsonl
CAL_SHA256=19cc1a8c4ebe6fd13031079f6b7d131046ce503f435f894e3ea672f91c2ed41f
SOUP_A=${SOUP_A:-} SOUP_S=${SOUP_S:-}
cd "$S"
export PYTHONPATH=$S TMPDIR=/data/dev2/tmp

args=()
for arm in M3-A-s1 M3-A-s2 M3-S-s1 M3-S-s2; do
  args+=(--candidate "$arm=$R/$arm/readout-kernel-$LIMIT" --train-run "$arm=$R/$arm/full/$(cat "$R/$arm/full/RUN_DIR")")
done
[ -z "$SOUP_A" ] || args+=(--candidate "M3-A-soup=$SOUP_A" --soup "M3-A=M3-A-soup:M3-A-s1+M3-A-s2")
[ -z "$SOUP_S" ] || args+=(--candidate "M3-S-soup=$SOUP_S" --soup "M3-S=M3-S-soup:M3-S-s1+M3-S-s2")
python3 -m v2.27b.m3_contrast --cal-rows "$CAL_FILE" --cal-sha256 "$CAL_SHA256" "${args[@]}" \
  --aho "A6g=$MX/aho-A6g.jsonl" --aho "A6h=$MX/aho-A6h.jsonl" --aho "A7=$MX/aho-A7.jsonl" \
  --contrast M3-A-s1:M3-S-s1 --contrast M3-A-s2:M3-S-s2 --contrast M3-A-s1+M3-A-s2:M3-S-s1+M3-S-s2 \
  --output "$OUTPUT"
