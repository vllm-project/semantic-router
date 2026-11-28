#!/bin/bash
# recipe-lab live smoke (Sub-issue 3 end-to-end): router + capability-aware
# mock + sr-bench baseline/candidate runs -> compare -> three-state verdict
# -> ledger. Everything runs inside one GPU job so the sr-bench service
# reaches the stack on localhost.
#
#   setsid nohup srun --gres=gpu:1 --cpus-per-task=8 --time=00:55:00 \
#     bash tools/recipe-lab/examples/smoke/live-smoke.sh \
#     > live-smoke-launch.log 2>&1 < /dev/null &
#
# Environment (all optional; defaults derive from this script's location):
#   REPO            semantic-router checkout        (default: derived)
#   RECIPE_LAB_BIN  dir with recipe-lab CLI+python  (default: $REPO/.venv/bin)
#   SR_BUILD        build prefix                    (default: $REPO/build)
#   ROUTER_BIN      router binary                   (default: $SR_BUILD/bin/semantic-router)
#   WORK_ROOT       scratch root for run dirs       (default: $REPO/.tmp-smoke-runs)
#   DATASET_DIR     local MMLU-Pro snapshot         (default: $WORK_ROOT/dataset/mmlu-pro)
set -u
ASSETS="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$ASSETS/../../.." && pwd)}"
RL="${RECIPE_LAB_BIN:-$REPO/.venv/bin}"
SR_BUILD="${SR_BUILD:-$REPO/build}"
ROUTER_BIN="${ROUTER_BIN:-$SR_BUILD/bin/semantic-router}"
WORK_ROOT="${WORK_ROOT:-$REPO/.tmp-smoke-runs}"
DATASET_DIR="${DATASET_DIR:-$WORK_ROOT/dataset/mmlu-pro}"
PYBIN="${PYTHON_BIN:-$RL/python}"
WORK="$WORK_ROOT/live-smoke-$(date +%m%d%H%M%S)"
mkdir -p "$WORK"
echo "WORK=$WORK"

cleanup() {
  for pid in ${ROUTER_PID:-} ${MOCK_PID:-}; do kill "$pid" 2>/dev/null; done
}
trap cleanup EXIT

# 1. manifests from the local MMLU snapshot (config hash patched after the
#    router is up; provisional 64-hex placeholder keeps the builder happy)
"$RL/python" "$ASSETS/build_live_manifest.py" \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$WORK" \
  --config-hash 0000000000000000000000000000000000000000000000000000000000000000 || exit 1

# 1b. render the experiment spec: the workload identity is the labeled cases
#     file hashed at runtime (A4) - the template ships with placeholders
CASES_HASH=$("$RL/recipe-lab" hash --file "$WORK/cases-mcq.jsonl" | "$PYBIN" -c "import json,sys; print(json.load(sys.stdin)['content_hash'])")
sed -e "s|\${CASES_FILE}|$WORK/cases-mcq.jsonl|g" -e "s|\${CASES_HASH}|$CASES_HASH|g" \
  "$ASSETS/smoke-spec.yaml.tmpl" > "$WORK/spec.yaml"

# 2. mock backend with the capability table, then the router
"$PYBIN" "$ASSETS/mock_backend.py" --port 8000 \
  --cases "$WORK/cases-mcq.jsonl" > "$WORK/mock.log" 2>&1 &
MOCK_PID=$!
sleep 2
"$ROUTER_BIN" -config "$ASSETS/smoke-config.yaml" -gateway standalone \
  > "$WORK/router.log" 2>&1 &
ROUTER_PID=$!
READY=0
for i in $(seq 1 60); do
  sleep 5
  kill -0 $ROUTER_PID 2>/dev/null || { echo LIVE-FAILED router-died; tail -30 "$WORK/router.log"; exit 1; }
  if curl -sf http://127.0.0.1:8080/api/v1/inventory/model-runtime 2>/dev/null | grep -q '"ready":true'; then
    READY=1; echo "ROUTER-READY after ~$((i*5))s"; break
  fi
done
[ "$READY" = "1" ] || { echo LIVE-FAILED not-ready; tail -30 "$WORK/router.log"; exit 1; }

# 3. grab the router's frozen config hash from the management API
#    (/api/v1/config/hash, pkg/apiserver api_paths.go) and patch the
#    candidate manifest - a placeholder hash fails the mom target's
#    expected-config-hash check with HTTP 400
HASH=$(curl -sf http://127.0.0.1:8080/api/v1/config/hash 2>/dev/null | grep -o "[0-9a-f]\{64\}" | head -1)
echo "CONFIG-HASH=$HASH"
if [ -n "$HASH" ]; then
  sed -i "s/0000000000000000000000000000000000000000000000000000000000000000/$HASH/" "$WORK/candidate-manifest.json"
fi

# 4. sr-bench: baseline run, then candidate run (durable store under the
#    workspace; --detach returns immediately with the durable run ID)
export SR_BENCH_STORE="$WORK/srbench-store"
B=$("$RL/vllm-sr" benchmark run --manifest "$WORK/baseline-manifest.json" --detach 2>&1)
echo "$B" > "$WORK/baseline-submit.json"; tail -5 "$WORK/baseline-submit.json"
BASE_ID=$("$RL/python" -c "import json; d=json.load(open('$WORK/baseline-submit.json')); print(d.get('id') or d.get('run_id',''))" 2>/dev/null)
echo "BASELINE-RUN=$BASE_ID"
[ -n "$BASE_ID" ] || { echo LIVE-FAILED baseline; exit 1; }

C=$("$RL/vllm-sr" benchmark run --manifest "$WORK/candidate-manifest.json" --detach 2>&1)
echo "$C" > "$WORK/candidate-submit.json"; tail -5 "$WORK/candidate-submit.json"
CAND_ID=$("$RL/python" -c "import json; d=json.load(open('$WORK/candidate-submit.json')); print(d.get('id') or d.get('run_id',''))" 2>/dev/null)
echo "CANDIDATE-RUN=$CAND_ID"
[ -n "$CAND_ID" ] || { echo LIVE-FAILED candidate; exit 1; }

# 5. wait for both runs to go terminal
for i in $(seq 1 40); do
  sleep 10
  S=$("$RL/vllm-sr" benchmark show "$BASE_ID" 2>/dev/null | "$RL/python" -c "import json,sys; print(json.load(sys.stdin).get('status',''))" 2>/dev/null)
  T=$("$RL/vllm-sr" benchmark show "$CAND_ID" 2>/dev/null | "$RL/python" -c "import json,sys; print(json.load(sys.stdin).get('status',''))" 2>/dev/null)
  echo "status: baseline=$S candidate=$T"
  case "$S$T" in *running*|*pending*|*queued*) ;; "") break;; esac
done

# 6. compare -> three-state verdict -> ledger
cat > "$WORK/candidate.json" <<'EOF'
{"kind":"config","ref":"smoke-config@main","change_summary":"live smoke candidate","provenance":{"proposer":"threshold-search","insight_refs":[]}}
EOF
"$RL/recipe-lab" live-verdict \
  --spec "$WORK/spec.yaml" \
  --baseline-run-id "$BASE_ID" \
  --candidate-run-id "$CAND_ID" \
  --ledger "$WORK/ledger.jsonl" \
  --candidate "$WORK/candidate.json" \
  --srbench-bin "$RL/vllm-sr" 2>&1 | tee "$WORK/live-verdict.log"
echo ====LEDGER====
"$RL/recipe-lab" ledger show --file "$WORK/ledger.jsonl" --tier live --json 2>/dev/null | head -30
echo "LIVE-DONE $WORK"
