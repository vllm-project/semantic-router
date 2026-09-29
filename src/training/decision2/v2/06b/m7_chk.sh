#!/usr/bin/env bash
# Uncorrected CHK probabilities (m7_probs, trainer path) of one frozen causal 0.6B run or soup on
# one GPU, then its B-D1 devcheck (m7_scorebias devcheck, host Python). M7 prereg sections 2.3/2.4.
# Usage (node A): m7_chk.sh <gpu> <mirror-dir-name> <arm> <out-dir> [cotenant|owner]
#   <arm>      /data/dev2/runs/06b/m1/arms/<arm>/full (BEST.json run or SOUP.json soup)
#   <out-dir>  under /data/dev2/runs/06b; receives PROBS.json, chk.probs.jsonl, devcheck.json
#   cotenant   (default) the GPU's queue may still train: the job is recorded in
#              owner.m7b-post (status running -> ended); `owner` is not touched
#   owner      the queue has ended: `owner` is written in the queue's format (active -> idle)
# One job per GPU at a time (flock on /data/dev2/runs/06b/m7/locks/gpu<N>.lock). Soups get
# --cal-parity against their cal.probs.jsonl and the committed recipe whose sha256 is SOUP.json's
# recipe_spec_sha256. Write-once: an existing PROBS.json / devcheck.json is kept (step skipped);
# a probs.started marker without PROBS.json is a recorded failure and is not retried.
# Container timing goes to /data/dev2/runs/06b/m7/timing/.
set -euo pipefail
gpu="$1"
sha="$2"
arm="$3"
out="$4"
mode="${5:-cotenant}"
R=/data/dev2/runs/06b
S=/data/dev2/src/$sha/src/training/decision2
D=$S/v2/06b
run_dir=$R/m1/arms/$arm/full
aho=$R/m7/a/aho
timing=$R/m7/timing
logs=/data/dev2/logs/06b
lease_dir=/data/dev2/leases/gpu$gpu.lock
name="m7b-chk-$arm"
case "$out" in
  "$R"/*) ;;
  *) echo "out-dir must be under $R" >&2; exit 2 ;;
esac
case "$mode" in cotenant | owner) ;; *) echo "mode must be cotenant or owner" >&2; exit 2 ;; esac
[ -d "$D" ] || { echo "missing mirror $S" >&2; exit 2; }
mkdir -p "$out" "$timing" "$R/m7/locks" "$logs"
utc() { date -u +%FT%TZ; }
say() { echo "$(utc) gpu$gpu $name: $*"; }
host_py() { PYTHONPATH="$S" PYTHONDONTWRITEBYTECODE=1 python3 "$@"; }

lease_start() {
  grep -qs '^track=06b-encoder$' "$lease_dir/owner" || { say "gpu$gpu owner is not track=06b-encoder"; exit 2; }
  local end
  end=$(date -u -d "+10 min" +%FT%TZ)
  if [ "$mode" = cotenant ]; then
    printf 'track=06b-encoder\nstatus=running\npurpose=0.6B M7(b) CHK probabilities of %s (recorded co-tenant with this track'"'"'s M7(b) training)\ncurrent_run=%s\nstart_utc=%s\nexpected_end_utc=%s\nupdated_utc=%s\n' \
      "$arm" "$name" "$1" "$end" "$1" > "$lease_dir/owner.m7b-post"
  else
    printf 'track=06b-encoder\nstatus=active\npurpose=0.6B M7(b) CHK probabilities of %s\ncurrent_run=%s\nexpected_end_utc=%s\nupdated_utc=%s\n' \
      "$arm" "$name" "$end" "$1" > "$lease_dir/owner"
  fi
}
lease_end() {
  if [ "$mode" = cotenant ]; then
    printf 'track=06b-encoder\nstatus=ended\npurpose=0.6B M7(b) CHK probabilities of %s (recorded co-tenant with this track'"'"'s M7(b) training)\ncurrent_run=%s\nstart_utc=%s\nlast_job_end_utc=%s\nlast_job_exit=%s\nupdated_utc=%s\n' \
      "$arm" "$name" "$1" "$2" "$3" "$2" > "$lease_dir/owner.m7b-post"
  else
    printf 'track=06b-encoder\nstatus=idle (0.6B M7(b) post-training: %s CHK done; no job running)\npurpose=allocation retained by the 0.6B track\nlast_job_end_utc=%s\nlast_job_exit=%s\nupdated_utc=%s\n' \
      "$arm" "$2" "$3" "$2" > "$lease_dir/owner"
  fi
}

if [ ! -f "$out/PROBS.json" ]; then
  if [ -e "$out/probs.started" ]; then
    say "an earlier attempt started ($(cat "$out/probs.started")) without PROBS.json: recorded failure, not retried"
    exit 1
  fi
  args=(--run-dir "/runs/m1/arms/$arm/full" --pair /runs/m7/a/aho/chk.jsonl "/runs/${out#"$R"/}/chk.probs.jsonl"
    --manifest "/runs/${out#"$R"/}/PROBS.json")
  if [ -f "$run_dir/SOUP.json" ]; then
    want=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['recipe_spec_sha256'])" "$run_dir/SOUP.json")
    recipe=$(cd "$D/records/arms" && sha256sum -- *.json | awk -v h="$want" '$1 == h {print $2; exit}')
    [ -n "$recipe" ] || { say "no committed recipe with sha256 $want"; exit 2; }
    args+=(--recipe "/src/src/training/decision2/v2/06b/records/arms/$recipe" --cal-parity)
  elif [ ! -f "$run_dir/BEST.json" ]; then
    say "$run_dir has neither SOUP.json nor BEST.json"
    exit 2
  fi
  exec 9> "$R/m7/locks/gpu$gpu.lock"
  say "waiting for the gpu$gpu job lock"
  flock 9
  start=$(utc)
  echo "$start $mode" > "$out/probs.started"
  lease_start "$start"
  say "start ($mode)"
  set +e
  DEV2_PYTHON=python3 bash "$D/run_container.sh" "$gpu" "$sha" "$name" -- -m v2.06b.m7_probs "${args[@]}"
  status=$?
  set -e
  end=$(utc)
  lease_end "$start" "$end" "$status"
  flock -u 9
  cp "$logs/$name.timing.json" "$timing/$name.timing.json"
  printf '{"step":"m7b CHK probabilities (m7_probs, trainer path)","arm":"%s","gpu":%s,"mode":"%s","lease":"%s","start_utc":"%s","end_utc":"%s","exit":%s,"container_timing":"%s"}\n' \
    "$arm" "$gpu" "$mode" "$([ "$mode" = cotenant ] && echo owner.m7b-post || echo owner)" "$start" "$end" "$status" \
    "$timing/$name.timing.json" > "$timing/$name.wrapper.timing.json"
  say "end exit $status"
  [ "$status" = 0 ] || exit "$status"
fi
if [ ! -f "$out/devcheck.json" ]; then
  host_py -m v2.06b.m7_scorebias devcheck --aho-dir "$aho" --probs "$out/PROBS.json" --label "$arm" \
    --output "$out/devcheck.json"
fi
say "done"
