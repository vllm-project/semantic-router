#!/usr/bin/env bash
# JevBench public-231 guard checks (v2.eval.gates public231) on stored, sealed runs.
# Runs on node A from the verified mirror of the commit that added the guard; CPU only.
#   ssh <node-a> 'bash -s' < run-guard-checks.sh
set -euo pipefail
SRC=5551fb38edb9f6805ae5d13c6efb3a3bc830b5ad-src_training_decision2
S=/data/dev2/src/$SRC/src/training/decision2
OUT=/data/dev2/runs/eval/jevbench-value/guard
R=/data/dev2/runs
mkdir -p "$OUT"
cd /tmp
check() {
  local name=$1 left=$2 right=$3 left_name=$4 right_name=$5
  echo "== $name"
  PYTHONPATH=$S python3 -m v2.eval.gates public231 --left "$left" --right "$right" \
    --left-name "$left_name" --right-name "$right_name" --output "$OUT/$name.json"
}
check dev2-8b-vs-lux1-same-renderer "$R/9b/formal-m4/K-a13-16k" "$R/9b/formal-m3/lux1-16k-shared" "DEV2.0-8B (K-a13)" "Lux 1.0 same-renderer"
check dev2-8b-vs-lux1-adopted "$R/9b/formal-m4/K-a13-16k" "$R/eval/m1/d1-lux1-autotune-cache" "DEV2.0-8B (K-a13)" "Lux 1.0 adopted"
check dev2-4b-vs-nox1 "$R/dec/formal/m4/m4-N4XF-soup-nodeA" "$R/eval/m1-adopt/nox1" "DEV2.0-4B (N4XF)" "Nox 1.0"
check dev2-4b-vs-decider4b "$R/dec/formal/m4/m4-N4XF-soup-nodeA" "$R/eval/m1-adopt/decider4b" "DEV2.0-4B (N4XF)" "Decider 4B"
check dec-m5-n5b-vs-dev2-4b "$R/dec/formal/m5/m5-N5B-soup" "$R/dec/formal/m4/m4-N4XF-soup-nodeA" "m5-N5B soup" "DEV2.0-4B (N4XF)"
check 9b-u-a13-vs-dev2-8b "$R/9b/formal-m4/U-a13-16k" "$R/9b/formal-m4/K-a13-16k" "U-a13" "DEV2.0-8B (K-a13)"
check 06b-m7-mx-vs-dev2-06b "$R/06b/m7/formal/m7-mx-soup" "$R/06b/m4/formal/m4-t-a7-soup" "m7-mx soup" "DEV2.0-0.6B (T soup)"
sha256sum "$OUT"/*.json
