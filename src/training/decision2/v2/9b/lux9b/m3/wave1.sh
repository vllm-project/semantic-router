#!/usr/bin/env bash
# usage: wave1.sh SHA
# Milestone 3 wave 1 on node A GPU6-7. GPU6: arm A (Lux + mx-v2-full-M + own-Lux KL + A7
# natural24k retention replay), primary seed, with preflights. GPU7: Lux 1.0 zero-step typed
# DEV / CSS pilot readout under this mirror, then arm A seed 1 once the preflight has passed.
set -uo pipefail
sha=$1
S=/data/dev2/src/$sha-src_training_decision2/src/training/decision2
L=$S/v2/9b/lux9b/m3
M3=/data/dev2/runs/9b/m3
LUX_REV=bd45a30aee8c84032791c245c70f86dee5389cc8
(
  "$L/arm.sh" "$sha" 6 A-s1 20260926 a-full-M --preflight
) > "$M3/logs/A-s1.log" 2>&1 &
(
  for panel in dev css-pilot; do
    "$L/job.sh" "$sha" 7 "lux0-$panel" "M3 Lux 1.0 zero-step readout $panel" 15 -- -m v2.dec.infer_1p0 \
      --package /model --model-id llm-semantic-router/Decision-1.0-Lux-9B --model-revision "$LUX_REV" \
      --input "/panels/$panel.prompts.jsonl" --output "/out/$panel.predictions.jsonl" --package-temperatures || exit 1
  done
  status=""
  for _ in $(seq 1 360); do
    for step in zero one check; do
      code=$(cat "$M3/pf-A-s1-$step/exit-code.txt" 2>/dev/null || true)
      [ -n "$code" ] && [ "$code" != 0 ] && { echo "preflight $step exited $code; seed 1 not started"; exit 1; }
    done
    if [ -f "$M3/pf-A-s1-check/exit-code.txt" ]; then
      status=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$M3/pf-A-s1-check/preflight.json")
      break
    fi
    sleep 10
  done
  [ "$status" = PASS ] || { echo "preflight status '$status'; seed 1 not started"; exit 1; }
  "$L/arm.sh" "$sha" 7 A-s2 1 a-full-M
) > "$M3/logs/A-s2.log" 2>&1 &
wait
echo WAVE1-DONE
