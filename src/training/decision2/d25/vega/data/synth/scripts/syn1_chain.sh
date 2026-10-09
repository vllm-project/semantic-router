#!/usr/bin/env bash
# Unattended SYN1 chain for a CPU pod on the server node (restartPolicy OnFailure; every step is idempotent and
# finished steps are skipped through markers in $D/markers):
#   plan -> per stage: generate until every shard of the stage is sealed -> blind audit -> release
#   (assemble + dedupe + decontaminate -> SYN1T soft targets -> publish SYN1[-a] -> M3T[-a] = M2T-v5 + SYN1T -> publish)
#   The last stage touches $D/GEN_DONE after its audit, which makes serve_loop.sh stop the vLLM server (GPUs freed).
# Env: TAG (code tag under /data/d25/vega/src), SEEDS (plan size), STAGES ("name:first:last ..." shard ranges,
#      100 seeds per shard; the final stage must be named "final"), LLM_URL.
set -euo pipefail
: "${TAG:?}"
SRC=/data/d25/vega/src/$TAG/src/training/decision2
export PYTHONPATH=$SRC
cd "$SRC"
W=/data/d25/vega/synth
D=$W/syn1
R=/data/d25/vega/data
S3=/data/d25/shared/index-suite-0.3
IDX=$R/decontam/index-s03p-v5
MODEL=/data/d25/shared/models/qwen3.5-397b-a17b-fp8
SERVED=qwen3.5-397b-a17b-fp8
URL=${LLM_URL:-http://d25-vega-synth-llm:8000}
SEEDS=${SEEDS:-24000}
STAGES=${STAGES:-"a:0:100 final:100:240"}
PLAN=$W/plan/syn1-plan-$SEEDS.jsonl.gz
mkdir -p "$D/markers" "$D/release" "$W/logs"

log() { echo "$(date -u +%FT%TZ) $*"; }
done_mark() { [[ -f $D/markers/$1 ]]; }
mark() { date -u +%FT%TZ > "$D/markers/$1"; }
unsealed() { local n=0; for ((i = $1; i < $2; i++)); do [[ -f $(printf "%s/shards/%05d/DONE" "$D" "$i") ]] || n=$((n + 1)); done; echo $n; }
wait_server() {
  local waited=0
  until [[ "$(curl -s -o /dev/null -w '%{http_code}' "$URL/health")" == 200 ]]; do
    (( waited % 600 == 0 )) && log "waiting for $URL"
    sleep 30; waited=$((waited + 30))
  done
}

[[ -f $PLAN ]] || python -m d25.vega.data.synth.plan --seeds "$SEEDS" --out "$PLAN"
# Shard 0 of the pilot N2 run used the same plan seeds, prompts and flags: replay its request cache.
if [[ ! -f $D/shards/00000/cache.jsonl && -f $W/pilotn2/shards/00000/cache.jsonl ]]; then
  mkdir -p "$D/shards/00000" && cp "$W/pilotn2/shards/00000/cache.jsonl" "$D/shards/00000/cache.jsonl"
fi

ensure_index() {
  [[ -f $IDX/rule.json && -f $IDX/item.npz ]] && return 0
  python -m d25.vega.data.fetch --what proxy
  local suite="$S3/selected-rows.jsonl.gz $S3/added-rows.jsonl.gz $S3/gsm8k-rows.jsonl.gz $R/raw/gsm8k/test-pseudo-suite.jsonl.gz $R/raw/proxy/pv1/protected-items.jsonl.gz"
  # shellcheck disable=SC2086
  python -m d25.vega.data.decontam build-index --workers 16 --suite $suite --out "$IDX"
}

ensure_base() {
  local base=/data/d25/shared/data/v1/M2T-v5
  if [[ -f $base/VERIFIED ]] && python -m d25.vega.data.publish verify --dir "$base" >/dev/null; then return 0; fi
  python -m d25.vega.data.publish fetch --dest "$base" --path-in-repo v1/M2T-v5
}

release() {
  local name=$1 suffix syn m3
  suffix=$([[ $name == final ]] && echo "" || echo "-$name")
  syn=SYN1$suffix; m3=M3T$suffix
  ensure_index
  python -m d25.vega.data.synth.assemble --work "$D" --index "$IDX" --out "$D/release/assembled-$name" --workers 16 --code-tag "$TAG"
  python -m d25.vega.data.synth.release syn1t --assembled "$D/release/assembled-$name" --audit "$D/audit-$name" \
    --out "$D/release/$syn" --tokenizer /models/Qwen3.8-27B --workers 16
  python -m d25.vega.data.publish push --src "$D/release/$syn" --dest "/data/d25/shared/data/synth/$syn" --path-in-repo "synth/$syn" --name "$syn"
  ensure_base
  python -m d25.vega.data.synth.release m3t --base /data/d25/shared/data/v1/M2T-v5 --syn1t "$D/release/$syn" \
    --out "$D/release/$m3" --name "$m3"
  python -m d25.vega.data.publish push --src "$D/release/$m3" --dest "/data/d25/shared/data/v1/$m3" --path-in-repo "v1/$m3" --name "$m3"
}

audit() {
  local name=$1
  done_mark "audit-$name" && return 0
  wait_server
  python -m d25.vega.data.synth.spotcheck --rows "$D/shards/*/rows.jsonl.gz" --per-archetype 40 --url "$URL" \
    --model "$SERVED" --repo Qwen/Qwen3.5-397B-A17B-FP8 --revision ea5b4f81096f3901c91dea97f81324302495781d \
    --budget 4096 --concurrency 192 --out "$D/audit-$name" || log "audit $name failed (release continues without it)"
  mark "audit-$name"
}

stage_release() {
  local name=$1
  audit "$name"
  done_mark "release-$name" && return 0
  release "$name"
  mark "release-$name"
}

for stage in $STAGES; do
  IFS=: read -r name first last <<< "$stage"
  if ! done_mark "gen-$name"; then
    while [[ $(unsealed "$first" "$last") -gt 0 ]]; do
      wait_server
      log "stage $name: $(unsealed "$first" "$last") unsealed shards in $first:$last"
      python -m d25.vega.data.synth.generate --plan "$PLAN" --work "$D" --shards "$first:$last" --gen-url "$URL" \
        --gen-model "$SERVED" --tokenizer "$MODEL" --parallel-shards 8 --inflight-seeds 160 --concurrency 480 \
        2>&1 | tee -a "$W/logs/generate-syn1.log" || log "generate exited with an error; retrying"
      sleep 30
    done
    mark "gen-$name"
  fi
  if [[ $name == final ]]; then
    audit final
    touch "$D/GEN_DONE"
    wait  # earlier stage releases
    stage_release final
  else
    # intermediate releases run beside the next stage's generation (the audit shares the server)
    ( stage_release "$name" ) > "$W/logs/release-$name.log" 2>&1 &
  fi
done
wait
log "SYN1 chain complete"
