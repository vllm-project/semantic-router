#!/usr/bin/env bash
# Milestone 6 queue on one leased GPU, one training process at a time. For each run in order:
# the family preflight on the family's s1 (once per family, marker PREFLIGHT_PASS/FAIL), or a
# wait (30 s polls, at most 20 min) for that marker when another GPU runs it; then the full run
# and, when COMPLETE, the typed DEV + CSS pilot readout. A failed preflight skips every run of
# its family; a collapsed or failed run is recorded and not rerun. The lease owner file shows
# status, purpose, current run and expected end.
# Usage (node A): m6_queue.sh <gpu> <mirror-dir-name> <run> [<run> ...]   (specs records/arms/<run>.json)
set -uo pipefail
gpu="$1"
sha="$2"
shift 2
D=/data/dev2/src/$sha/src/training/decision2/v2/06b
A=/data/dev2/runs/06b/m1/arms
M=/data/dev2/runs/06b/m6/queue
logs=/data/dev2/logs/06b
owner=/data/dev2/leases/gpu$gpu.lock/owner
mkdir -p "$M" "$logs"
exec >> "$logs/queue-m6-gpu$gpu.log" 2>&1
# Every qwen-causal step runs on the image's own Python (Transformers 5.17).
export DEV2_PYTHON=python3

say() { echo "$(date -u +%FT%TZ) gpu$gpu $*"; }
status() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['status'])" "$1" 2> /dev/null || echo MISSING; }
lease() {
  grep -qs '^track=06b-encoder$' "$owner" || {
    say "lease owner file lost track=06b-encoder: queue stopped"
    exit 2
  }
  printf 'track=06b-encoder\nstatus=%s\npurpose=%s\ncurrent_run=%s\nexpected_end_utc=%s\nupdated_utc=%s\n' \
    "$1" "$2" "$3" "$4" "$(date -u +%FT%TZ)" > "$owner"
}
until_utc() { date -u -d "+$1 min" +%FT%TZ; }
record() { echo "$2" > "$M/$1.status"; say "$1: $2"; }

for run in "$@"; do
  family="${run%-s[0-9]}"
  [ -f "$D/records/arms/$run.json" ] || {
    record "$run" "SKIPPED (no arm spec in the mirror)"
    continue
  }
  if [ -e "$M/$family.PREFLIGHT_FAIL" ]; then
    record "$run" "SKIPPED ($family preflight failed)"
    continue
  fi
  if [ ! -e "$M/$family.PREFLIGHT_PASS" ]; then
    if [ "$run" = "$family-s1" ] && mkdir "$M/$family.preflight.claim" 2> /dev/null; then
      lease active "M6 $family preflight" "$run" "$(until_utc 15)"
      say "$family preflight on $run"
      if bash "$D/m1_arm.sh" "$gpu" "$sha" "$run" preflight; then
        date -u +%FT%TZ > "$M/$family.PREFLIGHT_PASS"
      else
        date -u +%FT%TZ > "$M/$family.PREFLIGHT_FAIL"
        record "$run" "SKIPPED ($family preflight failed on this GPU)"
        continue
      fi
    else
      lease active "M6 waiting for the $family preflight" "$run" "$(until_utc 20)"
      waited=0
      while [ ! -e "$M/$family.PREFLIGHT_PASS" ] && [ ! -e "$M/$family.PREFLIGHT_FAIL" ] && [ "$waited" -lt 1200 ]; do
        sleep 30
        waited=$((waited + 30))
      done
      if [ ! -e "$M/$family.PREFLIGHT_PASS" ]; then
        record "$run" "SKIPPED ($family preflight failed or not passed within 20 min)"
        continue
      fi
    fi
  fi
  lease active "M6 training (one process on this GPU)" "$run" "$(until_utc 75)"
  say "$run full"
  bash "$D/m1_arm.sh" "$gpu" "$sha" "$run" full > "$logs/$run.driver.log" 2>&1
  result=$(status "$A/$run/full/COMPLETE.json")
  if [ "$result" != COMPLETE ]; then
    if [ -f "$A/$run/full/STOPPED.json" ]; then
      record "$run" "COLLAPSED (preregistered collapse stop; not rerun)"
    else
      record "$run" "FAILED (full run status $result; not rerun)"
    fi
    continue
  fi
  lease active "M6 readout (typed DEV + CSS pilot)" "$run" "$(until_utc 5)"
  if bash "$D/m1_arm.sh" "$gpu" "$sha" "$run" readout >> "$logs/$run.driver.log" 2>&1; then
    record "$run" "COMPLETE (readout written)"
  else
    record "$run" "COMPLETE (readout failed)"
  fi
done
printf 'track=06b-encoder\nstatus=idle (0.6B M6 queue finished; no job running)\npurpose=allocation retained by the 0.6B track\nlast_job_end_utc=%s\n' \
  "$(date -u +%FT%TZ)" > "$owner"
say "queue done: $*"
