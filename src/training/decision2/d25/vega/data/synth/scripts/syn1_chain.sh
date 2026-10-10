#!/usr/bin/env bash
# Unattended SYN1 chain for a CPU pod on the server node (restartPolicy OnFailure; every step is idempotent and
# finished steps are skipped through markers in $D/markers):
#   plan -> per stage: generate until every shard of the stage is sealed -> blind audit -> release over the fixed shard
#   prefix 0:<stage end> (assemble + dedupe + decontaminate -> SYN1T soft targets -> publish SYN1[-x] -> M3T[-x] =
#   M2T-v5 + SYN1T -> publish). Intermediate releases run beside the next stage's generation. After the final audit:
#   optional Qwen3.5-397B soft labels for M2's knowledge rows, then $D/GEN_DONE (serve_loop.sh stops the server), then
#   the final release and a last attempt at any Hub uploads left pending.
# Publishing never depends on the Hub: the shared node copy is verified first, uploads retry with backoff (token re-read
# from the mounted secret file) and otherwise leave HF_UPLOAD_PENDING; training nodes can fetch from the file server.
# Env: TAG (code tag under /data/d25/vega/src), SEEDS (plan size), STAGES ("name:first:last ..." shard ranges of 100
#      seeds; the last stage must be named "final"), LLM_URL, KNOWLEDGE_SOFTLABELS (1 = on).
set -euo pipefail
: "${TAG:?}"
SCRIPT=$(readlink -f "$0")
SRC=/data/d25/vega/src/$TAG/src/training/decision2
export PYTHONPATH=$SRC
cd "$SRC"
W=/data/d25/vega/synth
D=$W/syn1
R=/data/d25/vega/data
S3=/data/d25/shared/index-suite-0.3
IDX=$R/decontam/index-s03p-v5
BASE=/data/d25/shared/data/v1/M2T-v5
MODEL=/data/d25/shared/models/qwen3.5-397b-a17b-fp8
SERVED=qwen3.5-397b-a17b-fp8
URL=${LLM_URL:-http://d25-vega-synth-llm:8000}
SEEDS=${SEEDS:-24000}
STAGES=${STAGES:-"a:0:40 b:40:140 final:140:200"}
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
  if [[ -f $BASE/VERIFIED ]] && python -m d25.vega.data.publish verify --dir "$BASE" >/dev/null; then return 0; fi
  python -m d25.vega.data.publish fetch --dest "$BASE" --path-in-repo v1/M2T-v5 --wait-seconds 21600 \
    --fallback-url "${FILES_URL:-http://d25-vega-data-files-06:8080}"
}

release() {
  local name=$1 last=$2 suffix syn m3
  suffix=$([[ $name == final ]] && echo "" || echo "-$name")
  syn=SYN1$suffix; m3=M3T$suffix
  ensure_index
  python -m d25.vega.data.synth.assemble --work "$D" --shards "0:$last" --index "$IDX" --out "$D/release/assembled-$name" \
    --workers 16 --code-tag "$TAG"
  python -m d25.vega.data.synth.release syn1t --assembled "$D/release/assembled-$name" --audit "$D/audit-$name" \
    --out "$D/release/$syn" --tokenizer /models/Qwen3.8-27B --workers 16
  python -m d25.vega.data.publish push --src "$D/release/$syn" --dest "/data/d25/shared/data/synth/$syn" --path-in-repo "synth/$syn" \
    --name "$syn" --upload-hours 3
  ensure_base
  python -m d25.vega.data.synth.release m3t --base "$BASE" --syn1t "$D/release/$syn" --out "$D/release/$m3" --name "$m3"
  python -m d25.vega.data.publish push --src "$D/release/$m3" --dest "/data/d25/shared/data/v1/$m3" --path-in-repo "v1/$m3" \
    --name "$m3" --upload-hours 6
}

audit() {
  local name=$1 per=$2
  done_mark "audit-$name" && return 0
  wait_server
  python -m d25.vega.data.synth.spotcheck --rows "$D/shards/*/rows.jsonl.gz" --per-archetype "$per" --url "$URL" \
    --model "$SERVED" --repo Qwen/Qwen3.5-397B-A17B-FP8 --revision ea5b4f81096f3901c91dea97f81324302495781d \
    --budget 4096 --concurrency 192 --out "$D/audit-$name" || log "audit $name failed (release continues without it)"
  mark "audit-$name"
}

stage_release() {
  local name=$1 last=$2 per=$3
  audit "$name" "$per"
  done_mark "release-$name" && return 0
  release "$name" "$last"
  mark "release-$name"
}

knowledge_softlabels() {
  done_mark softlabel-knowledge && return 0
  local out=$W/softlabel/knowledge-M2-v5 input=$W/softlabel/knowledge-M2-v5.input.jsonl.gz
  mkdir -p "$out"
  ensure_base
  python - "$BASE" "$input" <<'EOF'
import json, sys
from pathlib import Path
from d25.vega.data.util import read_jsonl, write_jsonl
base, out = Path(sys.argv[1]), sys.argv[2]
files = json.loads((base / "manifest.json").read_text())["files"]
rows = (r for f in files for r in read_jsonl(base / f) if r["meta"].get("part") == "knowledge")
print("knowledge rows:", write_jsonl(out, rows))
EOF
  wait_server
  python -m d25.vega.data.synth.softlabel --rows "$input" --out "$out" --name qwen35-397b --url "$URL" --model "$SERVED" \
    --tokenizer "$MODEL" --concurrency 384
  python - "$out" <<'EOF'
import json, sys
from pathlib import Path
from d25.vega.data.util import sha256_file, write_json
out = Path(sys.argv[1])
files = {p.name: {"rows": None, "sha256": sha256_file(p)} for p in sorted(out.glob("part-*.jsonl.gz"))}
report = json.loads((out / "report.json").read_text()) if (out / "report.json").exists() else {}
write_json(out / "manifest.json", {"name": "qwen35-397b-knowledge-M2-v5", "files": files, "report": report,
            "note": "M2-v5/M2T-v5 knowledge rows (meta.part == knowledge) with meta.teachers.qwen35-397b: option-code "
                    "probabilities of Qwen/Qwen3.5-397B-A17B-FP8@ea5b4f81 with the d25-vega prompt (thinking off)"})
EOF
  python -m d25.vega.data.publish push --src "$out" --dest /data/d25/shared/data/teachers/qwen35-397b-knowledge-M2-v5 \
    --path-in-repo teachers/qwen35-397b-knowledge-M2-v5 --name qwen35-397b-knowledge-M2-v5 --upload-hours 2
  mark softlabel-knowledge
}

# Sub-commands run as separate bash processes so that `set -e` applies inside them (a function called on the left of
# `||`/`&&` runs with errexit disabled, which could publish stale outputs after a failed step).
case "${1:-}" in
  release-stage) shift; stage_release "$@"; exit 0 ;;
  knowledge) knowledge_softlabels; exit 0 ;;
esac

run_release() {
  local name=$1 last=$2 per=$3 attempt
  for attempt in 1 2 3 4 5 6; do
    if bash "$SCRIPT" release-stage "$name" "$last" "$per"; then return 0; fi
    log "release $name attempt $attempt failed; retrying in 10 min"
    sleep 600
  done
  return 1
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
    audit final 40
    if [[ ${KNOWLEDGE_SOFTLABELS:-1} == 1 ]]; then bash "$SCRIPT" knowledge || log "knowledge soft labels failed (optional)"; fi
    touch "$D/GEN_DONE"
    wait  # earlier stage releases
    run_release final "$last" 40 || exit 1  # the pod restarts (OnFailure) and resumes from the markers
  else
    # intermediate releases run beside the next stage's generation (the audit shares the server)
    run_release "$name" "$last" 25 >> "$W/logs/release-$name.log" 2>&1 &
  fi
done
wait
python -m d25.vega.data.publish retry-uploads --upload-hours 2 || log "some Hub uploads are still pending (HF_UPLOAD_PENDING markers)"
log "SYN1 chain complete"
