#!/usr/bin/env bash
# Score5-typed-DEV v1 validation collections on node A from an exact mirror:
#
#   bash collect.sh <sha>-src_training_decision2 <gpu>
#
# One full-panel score5t-dev collection per model with its formal run's settings, in the order
# m6-mxcx-soup, m6-mxcxa-soup, m7-mx-soup, m7-mxcx-soup, m4-t-a7-soup, m4-t-a7-soup-cache (the
# released soup with the frozen autotune cache, sensitivity), kai1-smoke (20 items) and kai1
# (only after a smoke with 20 valid answers). Per job: GPU check (VRAM <= 50% and no KFD process
# on the GPU; waits up to 10 min, then skips), revision check, a fresh copy of the frozen Triton
# autotune cache where the formal run had one (v2.06b.m6_cache seed, then record and unchanged
# check), run_same_panel.sh under the shared lease owner.eval-score5t within timeout 900, and the
# gold-free seal. A failed model is recorded and the others continue. No job starts with less
# than 300 s of the 0.3 GPU-h budget left, and none can run past it. Then score5t validate into
# validation-v1/; on exit the release lines go to the lease entry. Never overwrites; the whole
# output is logged to /data/dev2/runs/eval/score5t-dev/collect.log.
# Rules: v2/eval/records/score5t-dev-prereg-2026-09-29.md §8-§9.
set -euo pipefail

[[ $# -eq 2 ]] || { echo "usage: collect.sh <sha>-src_training_decision2 <gpu>" >&2; exit 2; }
SRC=$1
GPU=$2
MIRROR=/data/dev2/src/$SRC
S=$MIRROR/src/training/decision2
[[ $GPU =~ ^[01]$ ]] || { echo "gpu must be 0 or 1 (node A allocation)" >&2; exit 2; }
[[ -f $MIRROR/.dev2-mirror.json && -d $S ]] || { echo "no mirror at $MIRROR" >&2; exit 1; }
[[ $(realpath "$0") == "$MIRROR"/* ]] || { echo "run the copy inside $MIRROR" >&2; exit 1; }

ROOT=/data/dev2/runs/eval/score5t-dev
COLLECT=$ROOT/collect
VALIDATION=$ROOT/validation-v1
LOG=$ROOT/collect.log
PANELS=/data/dev2/private/panels
PROMPTS=$PANELS/goldfree/score5t-dev.prompts.jsonl
IMAGE=decision20-train-fast:host2
IMAGE_ID=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
SNAPSHOT=/data/dev2/runs/06b/m6/triton-cache-frozen
SNAPSHOT_MANIFEST=$SNAPSHOT.MANIFEST.json
TREE=e215f8bd5145181404bae7c502c084033d85c2e767ee4651942e870dcc72a94f
TREE_FILES=303
ARMS=/data/dev2/runs/06b/m1/arms
CAUSAL=$S/v2/06b/records/adapters/dev2-06b-causal-8k.json
KAI_SPEC=$S/v2/06b/records/adapters/dev2-06b-kai-native-8k.json
KAI_MODEL=/data/decision20-20260926/models/Decision-1.0-Kai-0.6B
KAI_REV=7185f514f54b8f93c55998b1e8f9c5cc67f0d029
KAI_ENV=/data/dev2/tools/envs/kai-lex
# Tree digests (v2.eval.sealed.event3 digest) as pinned for kai1-8k in v2/eval/sealed/event3-models.json.
KAI_TREES=("$KAI_MODEL=731925de7347d706db61b96c29dae98125a71d5e341419b2d02f4d48942606f2"
  "$KAI_ENV=d3ae64ba0266d492345f69bd6109ac6412e31eeec53d7fe00240bf5106194b9b")
LEASE_NAME=eval-score5t
LEASE=/data/dev2/leases/gpu$GPU.lock/owner.$LEASE_NAME
BUDGET_S=1080
MIN_LEFT_S=300
JOB_S=900
MODELS=(m6-mxcx-soup m6-mxcxa-soup m7-mx-soup m7-mxcx-soup m4-t-a7-soup m4-t-a7-soup-cache kai1-smoke kai1)
declare -A REV=(
  [m6-mxcx-soup]=3ae009ad838487c0df839a55898a98f5665276136d2a8cab3c1260dd0b29dfa2
  [m6-mxcxa-soup]=02892a86f28d0dd97b2b5b64cc82d51fa064a49eddf9f4a09f97603eddd3fdcd
  [m7-mx-soup]=87dcd8fb2e24a86baf2a2d5382881e1f41e882d0fddd691c9befdb2c358fa08a
  [m7-mxcx-soup]=93aafa0437f62cb652cbc8182bbe63c42b4942de1b8f938ff44d62276f00d5a8
  [m4-t-a7-soup]=0f96aa3932ea501589f794eff9949b52fd1c64835b92f0392da860c22b0442bf
)
declare -A CACHED=([m6-mxcx-soup]=1 [m6-mxcxa-soup]=1 [m7-mx-soup]=1 [m7-mxcx-soup]=1 [m4-t-a7-soup-cache]=1)
VALIDATE=(m7-mx-soup m6-mxcx-soup m7-mxcx-soup m6-mxcxa-soup m4-t-a7-soup kai1 m4-t-a7-soup-cache)
declare -A STATUS=()
SPENT=0
JOBS=0
SMOKE_OK=0
RELEASED=0

mkdir -p "$ROOT"
[[ -e $LOG ]] || install -m 600 /dev/null "$LOG"
exec > >(tee -a "$LOG") 2>&1
export PYTHONPATH=$S
cd /tmp
step() { echo "== $(date -u +%FT%TZ) $*"; }

release() {
  (( RELEASED == 0 && JOBS > 0 )) || return 0
  RELEASED=1
  local now
  now=$(date -u +%FT%TZ)
  printf 'status=idle-released (%s)\nupdated_utc=%s\n' "$now" "$now" >>"$LEASE"
  echo "lease entry $LEASE:"
  cat "$LEASE"
}

snapshot_check() {
  python3 -m v2.06b.m6_cache tree "$SNAPSHOT" | python3 -c '
import json, sys
tree, frozen = json.load(sys.stdin), json.load(open(sys.argv[1]))
assert tree["tree_sha256"] == frozen["tree_sha256"] == sys.argv[2], "snapshot tree changed"
assert tree["files"] == frozen["files"] == int(sys.argv[3]), "snapshot file count changed"
print("frozen snapshot", tree["tree_sha256"], tree["files"], "files")' "$SNAPSHOT_MANIFEST" "$TREE" "$TREE_FILES"
}

soup_revision() {
  local arm=$1 home=$ARMS/$1/full rev
  rev=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["best_export_manifest_sha256"])' \
    "$home/COMPLETE.json") || return 1
  [[ $rev == "${REV[$arm]}" ]] || { echo "$arm: COMPLETE.json names $rev, expected ${REV[$arm]}" >&2; return 1; }
  [[ $(sha256sum <"$home/best-export.MANIFEST.json" | cut -d' ' -f1) == "$rev" ]] ||
    { echo "$arm: best-export.MANIFEST.json does not hash to $rev" >&2; return 1; }
  [[ ! -e $home/best-export/score_bias.json ]] || { echo "$arm: package holds score_bias.json" >&2; return 1; }
  echo "$rev"
}

kai_identity() {
  local revs pair digest
  revs=$(find "$KAI_MODEL/.cache/huggingface/download" -name '*.metadata' -exec head -n 1 {} \; | sort -u) || return 1
  [[ $revs == "$KAI_REV" ]] || { echo "kai1: download records name ${revs:-no revision}" >&2; return 1; }
  echo "kai1 download records: revision $KAI_REV"
  for pair in "${KAI_TREES[@]}"; do
    digest=$(python3 -m v2.eval.sealed.event3 digest "${pair%%=*}") || return 1
    echo "$digest"
    [[ ${digest%% *} == "${pair#*=}" ]] || { echo "kai1: tree digest differs from ${pair#*=}" >&2; return 1; }
  done
}

gpu_idle() {
  local vram pids
  vram=$(rocm-smi -d "$GPU" --showmemuse 2>/dev/null | awk -F': ' '/VRAM%/ {v = $NF} END {print v}') || true
  pids=$(rocm-smi --showpidgpus 2>/dev/null | awk -v g="$GPU" '
    /^PID [0-9]+ is using/ {pid = $2; listed = ($5 > 0); next}
    listed {for (i = 1; i <= NF; i++) if ($i == g) printf " %s", pid; listed = 0}') || true
  echo "$(date -u +%FT%TZ) gpu$GPU VRAM%=${vram:-unknown}, KFD processes on it:${pids:- none}"
  [[ $vram =~ ^[0-9]+$ ]] && (( vram <= 50 )) && [[ -z $pids ]]
}

wait_idle() {
  local try
  for try in $(seq 0 20); do
    if (( try > 0 )); then sleep 30; fi
    if gpu_idle; then return 0; fi
  done
  return 1
}

job_limit() {
  local left=$((BUDGET_S - SPENT))
  (( left >= MIN_LEFT_S )) || return 1
  echo $((left < JOB_S ? left : JOB_S))
}

receipt_ok() {
  python3 - "$@" <<'EOF'
import json, sys
run, name, rows = sys.argv[1], sys.argv[2], int(sys.argv[3])
(entry,) = json.load(open(f"{run}/{name}"))["panels"]
keys = ("panel", "exit_code", "output_rows", "expected_rows", "wall_seconds")
print(name, json.dumps({key: entry[key] for key in keys}))
sys.exit(0 if (entry["panel"], entry["exit_code"], entry["output_rows"]) == ("score5t-dev", 0, rows) else 1)
EOF
}

valid_answers() {
  python3 - "$PROMPTS" "$1" <<'EOF'
import json, sys
from benchmark.score import evaluate_answer
with open(sys.argv[1]) as stream:
    questions = {row["id"]: row["questions"]["decision"] for row in map(json.loads, stream)}
with open(sys.argv[2]) as stream:
    rows = [json.loads(line) for line in stream]
valid = 0
for row in rows:
    answer = (row.get("answers") or {}).get("decision")
    if isinstance(answer, dict):
        # The point never depends on gold; the placeholder only feeds the correctness fields.
        result = evaluate_answer(questions[row["id"]], {"value": 0}, answer)
        valid += result["status"] == "ok" and result["point"] is not None
print("answers (gold-free)", json.dumps({"rows": len(rows), "valid": valid, "invalid": len(rows) - valid}))
sys.exit(0 if rows and valid == len(rows) else 1)
EOF
}

cache_unchanged() {
  python3 - "$1" "$TREE" "$TREE_FILES" <<'EOF'
import json, sys
record = json.load(open(sys.argv[1]))
keys = ("unchanged", "before_tree_sha256", "after_tree_sha256", "after_files", "added", "removed", "changed")
print("cache", json.dumps({key: record[key] for key in keys}))
same = (record["before_tree_sha256"], record["after_tree_sha256"]) == (sys.argv[2], sys.argv[2])
sys.exit(0 if record["unchanged"] and same and record["after_files"] == int(sys.argv[3]) else 1)
EOF
}

job() {
  local name=$1 run=$COLLECT/$1 cache=$COLLECT/$1.triton-cache arm rev model limit t0 code note=""
  local runner=() collector=()
  step "$name"
  if [[ $name == kai1 && $SMOKE_OK != 1 ]]; then STATUS[$name]="not run: kai1-smoke did not pass"; return; fi
  if ! limit=$(job_limit); then
    STATUS[$name]="not run: under $MIN_LEFT_S s of the $BUDGET_S s budget left ($SPENT s spent)"; return
  fi
  if [[ $name == kai1* ]]; then
    if ! kai_identity; then STATUS[$name]="not run: Kai1 identity check failed"; return; fi
    runner=(--model-dir "$KAI_MODEL" --mount "$KAI_ENV")
    collector=(--adapter-spec "$KAI_SPEC" --model-path "$KAI_MODEL" --revision "$KAI_REV"
      --extra backend=kai --extra model_id=dev2-06b/kai1-native-8k --panels score5t-dev)
    if [[ $name == kai1-smoke ]]; then collector+=(--max-items 20); fi
  else
    arm=${name%-cache}
    if ! rev=$(soup_revision "$arm"); then STATUS[$name]="not run: revision check failed"; return; fi
    echo "$arm revision $rev"
    model=$ARMS/$arm/full/best-export
    runner=(--model-dir "$model")
    collector=(--adapter-spec "$CAUSAL" --model-path "$model" --revision "$rev"
      --extra "model_id=dev2-06b/$arm" --panels score5t-dev)
  fi
  if ! wait_idle; then STATUS[$name]="skipped: gpu$GPU busy for 10 min"; return; fi
  if [[ -n ${CACHED[$name]:-} ]]; then
    if ! python3 -m v2.06b.m6_cache seed --snapshot "$SNAPSHOT" --manifest "$SNAPSHOT_MANIFEST" --dest "$cache"; then
      STATUS[$name]="not run: cache seed failed"; return
    fi
    runner+=(--env TRITON_CACHE_AUTOTUNING=1 --env "TRITON_CACHE_DIR=$cache" --mount-rw "$cache")
  fi
  JOBS=$((JOBS + 1))
  t0=$(date +%s)
  set +e
  timeout "$limit" bash "$S/v2/eval/run_same_panel.sh" --gpu "$GPU" --track eval --src "$SRC" \
    --run-dir "$run" --shared-lease "$LEASE_NAME" --purpose "Score5-typed-DEV v1 validation: $name" \
    --expected-end "$(date -u -d '+15 minutes' +%FT%TZ)" "${runner[@]}" -- "${collector[@]}"
  code=$?
  set -e
  SPENT=$((SPENT + $(date +%s) - t0 + 1))
  echo "$(date -u +%FT%TZ) $name: runner exit $code (timeout $limit s; $SPENT of $BUDGET_S s spent)"
  if [[ -n ${CACHED[$name]:-} ]]; then
    if python3 -m v2.06b.m6_cache record --manifest "$SNAPSHOT_MANIFEST" --cache "$cache" \
        --output "$run/M6-CACHE.json" && cache_unchanged "$run/M6-CACHE.json"; then
      note="; cache unchanged"
    else
      note="; CACHE CHANGED OR NOT RECORDED"
    fi
  fi
  if [[ $name == kai1* ]]; then cat "$run"/*/score5t-dev.predictions.jsonl.manifest.json || true; echo; fi
  if [[ $name == kai1-smoke ]]; then
    if (( code == 0 )) && receipt_ok "$run" SMOKE.json 20 && valid_answers "$run/smoke/score5t-dev.predictions.jsonl"; then
      SMOKE_OK=1
      STATUS[$name]="passed: 20 of 20 answers valid"
    else
      STATUS[$name]="failed (runner exit $code): Kai1 dropped"
    fi
    return
  fi
  if (( code != 0 )) || ! receipt_ok "$run" COLLECT.json 800; then
    STATUS[$name]="failed: runner exit $code$note"; return
  fi
  valid_answers "$run/output/score5t-dev.predictions.jsonl" || true
  if ! python3 -m v2.eval.htdev.score seal --prompts "$PROMPTS" \
      --predictions "$run/output/score5t-dev.predictions.jsonl" --output "$run/SEAL-SCORE5T.json"; then
    STATUS[$name]="failed: seal$note"; return
  fi
  STATUS[$name]="collected and sealed$note"
}

step "start: $SRC on gpu$GPU"
cat "$MIRROR/.dev2-mirror.json"
echo
commit=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["commit"])' "$MIRROR/.dev2-mirror.json")
[[ $SRC == "$commit"-* ]] || { echo "mirror commit $commit does not match $SRC" >&2; exit 1; }
[[ ! -e $COLLECT && ! -e $VALIDATION ]] || { echo "$COLLECT or $VALIDATION exists; one collection per model" >&2; exit 1; }
image_id=$(docker image inspect --format '{{.Id}}' "$IMAGE")
[[ $image_id == "$IMAGE_ID" ]] || { echo "$IMAGE is $image_id, expected $IMAGE_ID" >&2; exit 1; }
echo "image $IMAGE $image_id"
registered=$(python3 -c 'from v2.eval import panels; print(panels.ALL["score5t-dev"]["prompts_sha256"])')
[[ $(sha256sum <"$PROMPTS" | cut -d' ' -f1) == "$registered" ]] || { echo "$PROMPTS is not the registered panel" >&2; exit 1; }
echo "prompts $registered"
snapshot_check
for arm in m6-mxcx-soup m6-mxcxa-soup m7-mx-soup m7-mxcx-soup m4-t-a7-soup; do
  rev=$(soup_revision "$arm") || { echo "stopping before any job: $arm revision check failed" >&2; exit 1; }
  echo "$arm revision $rev"
done
echo "gpu$GPU owner entry:"
cat "/data/dev2/leases/gpu$GPU.lock/owner" || true

mkdir -m 700 "$COLLECT"
trap release EXIT
for name in "${MODELS[@]}"; do
  job "$name"
done
release

step "validate"
runs=()
for name in "${VALIDATE[@]}"; do
  if [[ -f $COLLECT/$name/SEAL-SCORE5T.json ]]; then
    runs+=(--run "$name=$COLLECT/$name")
  else
    echo "not validated: $name (${STATUS[$name]:-not run})"
  fi
done
if (( ${#runs[@]} )); then
  timeout 1h python3 -m v2.eval.score5t validate --panels-root "$PANELS" "${runs[@]}" \
    --output "$VALIDATION/VALIDATION.json" || echo "validate failed: exit $?"
fi

step "summary"
for name in "${MODELS[@]}"; do
  echo "$name: ${STATUS[$name]:-not reached}"
done
python3 - "$COLLECT" <<'EOF'
import json, sys
from pathlib import Path
total = 0.0
for path in sorted(Path(sys.argv[1]).glob("*/GPU-TIME.json")):
    record = json.loads(path.read_text())
    total += record["wall_seconds"]
    print(f"{path.parent.name}: {record['wall_seconds']:.1f} s, exit {record['exit_code']}, gpu{record['gpu']}, "
          f"VRAM% at start {record.get('vram_pct_at_start')}, {record['image_id']}")
print(f"total {total:.1f} GPU-seconds = {total / 3600:.4f} GPU-h (runner GPU-TIME.json)")
EOF
echo "driver-measured job time $SPENT s"
snapshot_check
step "done"
