#!/usr/bin/env bash
# Milestone 1, step 1: starting-point verification on one leased GPU (no training).
# Usage: m1_verify.sh <gpu-index> <commit-sha> <kai|lex|encoders>
set -euo pipefail
gpu="$1"
sha="$2"
what="$3"
run="/data/dev2/src/$sha/src/training/decision2/v2/06b/run_container.sh"
out=/runs/m1/verify
mkdir -p /data/dev2/runs/06b/m1/verify
declare -A REV=([kai]=7185f514f54b8f93c55998b1e8f9c5cc67f0d029 [lex]=ee8e74d912fca8328a353c11d174b44da3f91781)
declare -A DIR=([kai]=Decision-1.0-Kai-0.6B [lex]=Decision-1.0-Lex-0.6B)
declare -A REF=([kai]=kai06b [lex]=lex06b)

if [ "$what" = kai ] || [ "$what" = lex ]; then
  m=/work/models/${DIR[$what]}
  bash "$run" "$gpu" "$sha" "m1-$what-verify-only" -- -m inference.kai_lex --backend "$what" \
    --model-path "$m" --model-revision "${REV[$what]}" --verify-only
  for panel in dev css-pilot; do
    input=/work/runs/dev.prompts.jsonl
    reference="/runs/reference/${REF[$what]}-dev.predictions.jsonl"
    if [ "$panel" = css-pilot ]; then
      input=/work/runs/css-transfer-v1/css-pilot.prompts.jsonl
      reference="/runs/reference/css-pilot.${REF[$what]}.predictions.jsonl"
    fi
    bash "$run" "$gpu" "$sha" "m1-$what-$panel-pinned1k" -- -m inference.kai_lex --backend "$what" \
      --model-path "$m" --model-revision "${REV[$what]}" --input "$input" --output "$out/$what.$panel.pinned1k.jsonl"
    bash "$run" "$gpu" "$sha" "m1-$what-$panel-native8k" -- -m v2.06b.predict benchmark --family kai-native \
      --bundle "$m" --backend "$what" --input "$input" --output "$out/$what.$panel.native8k.jsonl" \
      --model-id "llm-semantic-router/${DIR[$what]}" --model-revision "${REV[$what]}" --backend-label "$what-native-8k"
    bash "$run" "$gpu" "$sha" "m1-$what-$panel-identity-archived" -- -m v2.06b.identity \
      --reference "$reference" \
      --candidate "$out/$what.$panel.pinned1k.jsonl" --output "$out/$what.$panel.identity-archived.json" || true
    bash "$run" "$gpu" "$sha" "m1-$what-$panel-identity-8k" -- -m v2.06b.identity \
      --reference "$out/$what.$panel.pinned1k.jsonl" --candidate "$out/$what.$panel.native8k.jsonl" \
      --output "$out/$what.$panel.identity-8k.json" || true
  done
elif [ "$what" = encoders ]; then
  bash "$run" "$gpu" "$sha" "m1-cpu-tests" -- -m unittest v2.06b.tests.test_common v2.06b.tests.test_encoder_cpu -v
  for source in eurobert-610m mmbert-base; do
    path=/runs/models/EuroBERT-610m
    [ "$source" = mmbert-base ] && path=/runs/models/mmBERT-base
    bash "$run" "$gpu" "$sha" "m1-$source-zero-step" -- -m v2.06b.preflight_encoder --source "$source" \
      --source-path "$path" --bundle /work/models/Decision-1.0-Kai-0.6B --output "$out/$source.zero-step.json" \
      || echo "$source zero-step exited nonzero"
  done
fi
