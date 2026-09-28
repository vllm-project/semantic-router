#!/usr/bin/env bash
# Node A: M3a AutoJev-27B runtime qualification (preregistration sections 3-4).
#
# Usage: qualify_nodeA.sh MIRROR_DIR WORK_DIR
#
# Expects WORK_DIR/qual/{warm,qual}.prompts.jsonl and sets.json from `v2.data.m3.qualify sets`.
# Warm-up on GPU2 with the empty shared cache, round 1 (P1 GPU2, P2 GPU3, P3 GPU4 concurrently),
# round 2 (P4 GPU2), then `v2.data.m3.qualify compare` -> WORK_DIR/qualification.json.
set -uo pipefail
src="$1"; work="$2"
job="$src/src/training/decision2/v2/data/m3/autojev_job.sh"
cache="$work/triton-cache-autojev-nodeA"
q="$work/qual"; out="$q/out"
ref=/data/dev2/runs/eval/m1-adopt/autojev27/output
mkdir -p "$cache" "$out"
event() { echo "{\"event\":\"$1\",\"rc\":${2:-0},\"utc\":\"$(date -u +%FT%TZ)\"}" >> "$work/qualify.events.jsonl"; }
run() { bash "$job" --gpu "$1" --src "$src" --input "$2" --output "$out/$3.jsonl" --cache "$cache" --label "m3a-$3"; }

[[ -z "$(find "$cache" -type f -name '*.autotune.json' | head -n 1)" ]] || { event cache_not_empty 2; exit 2; }
event warm_start
run 2 "$q/warm.prompts.jsonl" warm; rc=$?; event warm_done "$rc"
[[ $rc -eq 0 ]] || exit 1
event round1_start
run 2 "$q/qual.prompts.jsonl" p1 & a=$!
run 3 "$q/qual.prompts.jsonl" p2 & b=$!
run 4 "$q/qual.prompts.jsonl" p3 & c=$!
wait $a; ra=$?; wait $b; rb=$?; wait $c; rc=$?
event round1_done $((ra + rb + rc))
run 2 "$q/qual.prompts.jsonl" p4; rd=$?
event round2_done "$rd"
cd "$src/src/training/decision2" || exit 1
python3 -m v2.data.m3.qualify compare --sets "$q/sets.json" \
  --warm-manifest "$out/warm.jsonl.manifest.json" \
  --run "P1=$out/p1.jsonl=$out/p1.jsonl.manifest.json" \
  --run "P2=$out/p2.jsonl=$out/p2.jsonl.manifest.json" \
  --run "P3=$out/p3.jsonl=$out/p3.jsonl.manifest.json" \
  --run "P4=$out/p4.jsonl=$out/p4.jsonl.manifest.json" \
  --pair P1:P2 --pair P1:P3 --pair P2:P3 --pair P1:P4 \
  --reference "eval=$ref/typed-final.predictions.jsonl,$ref/public231.predictions.jsonl,$ref/css15.predictions.jsonl" \
  --spot-run P1 --out "$work/qualification.json" > "$work/qualification.stdout" 2>&1
event compare_done $?
