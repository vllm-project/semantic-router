#!/bin/bash
# recipe-lab replay smoke: router (d3-lite + router_replay) + mock backend
# + labeled requests + replay dump + join + attribution.
#
# Needs a GPU node (the router loads the decision model). Example:
#   srun --gres=gpu:1 --cpus-per-task=8 --time=00:45:00 \
#     bash tools/recipe-lab/examples/smoke/replay-smoke.sh
# Poll stdout for SMOKE-DONE / SMOKE-FAILED.
#
# Environment (all optional; defaults derive from this script's location):
#   REPO            semantic-router checkout        (default: derived)
#   RECIPE_LAB_BIN  dir with recipe-lab CLI+python  (default: $REPO/.venv/bin)
#   PYTHON_BIN      python for the mock backend     (default: $RECIPE_LAB_BIN/python)
#   SR_BUILD        build prefix                    (default: $REPO/build)
#   ROUTER_BIN      router binary                   (default: $SR_BUILD/bin/semantic-router)
#   WORK_ROOT       scratch root for run dirs       (default: $REPO/.tmp-smoke-runs)
set -u
ASSETS="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$ASSETS/../../.." && pwd)}"
RL="${RECIPE_LAB_BIN:-$REPO/.venv/bin}"
PYBIN="${PYTHON_BIN:-$RL/python}"
SR_BUILD="${SR_BUILD:-$REPO/build}"
ROUTER_BIN="${ROUTER_BIN:-$SR_BUILD/bin/semantic-router}"
WORK_ROOT="${WORK_ROOT:-$REPO/.tmp-smoke-runs}"
SMOKE_DIR="$WORK_ROOT/replay-smoke-$(date +%m%d%H%M%S)"
mkdir -p "$SMOKE_DIR"
echo "SMOKE_DIR=$SMOKE_DIR"

# 0. render the spec template: the frozen dataset identity is the real cases
#    file hash (A4); attribute/run verify it before touching the data
CASES_HASH=$("$RL/recipe-lab" hash --file "$ASSETS/cases.jsonl" | "$PYBIN" -c "import json,sys; print(json.load(sys.stdin)['content_hash'])")
sed -e "s|\${CASES_FILE}|$ASSETS/cases.jsonl|g" -e "s|\${CASES_HASH}|$CASES_HASH|g" \
  "$ASSETS/smoke-spec.yaml.tmpl" > "$SMOKE_DIR/spec.yaml"

cleanup() {
  for pid in ${ROUTER_PID:-} ${MOCK_PID:-}; do kill "$pid" 2>/dev/null; done
}
trap cleanup EXIT

# 1. mock OpenAI backend on :8000
"$PYBIN" "$ASSETS/mock_backend.py" --port 8000 > "$SMOKE_DIR/mock.log" 2>&1 &
MOCK_PID=$!
sleep 2
if ! curl -sf http://127.0.0.1:8000/v1/models > /dev/null; then echo SMOKE-FAILED mock; cat "$SMOKE_DIR/mock.log"; exit 1; fi
echo MOCK-OK

# 2. router with d3-lite (model_runtime) + router_replay on every decision
# -gateway standalone: the Router serves the OpenAI listeners itself (no Envoy)
"$ROUTER_BIN" -config "$ASSETS/smoke-config.yaml" -gateway standalone > "$SMOKE_DIR/router.log" 2>&1 &
ROUTER_PID=$!

# 3. wait for the decision model to report ready via the management API
READY=0
for i in $(seq 1 60); do
  sleep 5
  if ! kill -0 $ROUTER_PID 2>/dev/null; then
    echo SMOKE-FAILED router-died; tail -40 "$SMOKE_DIR/router.log"; exit 1
  fi
  if curl -sf http://127.0.0.1:8080/api/v1/inventory/model-runtime 2>/dev/null | grep -q '"ready":true'; then
    READY=1; echo "ROUTER-READY after ~$((i*5))s"; break
  fi
done
if [ "$READY" != "1" ]; then echo SMOKE-FAILED not-ready; tail -40 "$SMOKE_DIR/router.log"; exit 1; fi

# 4. send the labeled cases through the router
"$RL/python" "$ASSETS/send_requests.py" --cases "$ASSETS/cases.jsonl" --router http://127.0.0.1:8899 2>&1 | tee "$SMOKE_DIR/requests.log"

# 5. dump replay records from the management API
curl -sf 'http://127.0.0.1:8080/api/v1/observability/replays?showDetails=true' > "$SMOKE_DIR/replays.json"
echo "REPLAYS-BYTES=$(wc -c < "$SMOKE_DIR/replays.json")"

# 6. join with labeled cases -> attribution records
"$RL/recipe-lab" collect-replays --url http://127.0.0.1:8080 \
  --cases "$ASSETS/cases.jsonl" \
  --output "$SMOKE_DIR/records.jsonl" --family d3 2>&1 | tee "$SMOKE_DIR/collect.log"

# 7. answer bank: query each case against each routing model directly, then
#    join correctness by (case_id, routed model) -> no_evidence becomes real outcomes
"$RL/recipe-lab" bank build --cases "$ASSETS/cases.jsonl" \
  --models local/private-qwen,cloud/frontier-reasoning \
  --endpoint http://127.0.0.1:8000 \
  --output "$SMOKE_DIR/answer-bank.jsonl" 2>&1 | tee "$SMOKE_DIR/bank-build.log"
"$RL/recipe-lab" bank apply --bank "$SMOKE_DIR/answer-bank.jsonl" \
  --records "$SMOKE_DIR/records.jsonl" --output "$SMOKE_DIR/records-graded.jsonl" 2>&1 | tee "$SMOKE_DIR/bank-apply.log"

# 8. attribution over graded records (--cases: A4 hash check against the spec)
"$RL/recipe-lab" attribute --spec "$SMOKE_DIR/spec.yaml" \
  --records "$SMOKE_DIR/records-graded.jsonl" --cases "$ASSETS/cases.jsonl" \
  --output-dir "$SMOKE_DIR" 2>&1 | tee "$SMOKE_DIR/attribute.log"

# 9. threshold proposer: candidate configs for the depth signal
# (--deltas= form: argparse rejects values that start with a dash otherwise)
"$RL/recipe-lab" propose-thresholds --base-config "$ASSETS/smoke-config.yaml" \
  --signal depth "--deltas=-0.1,-0.05,0.05,0.1" \
  --output-dir "$SMOKE_DIR/candidates" 2>&1 | tee "$SMOKE_DIR/propose.log"

echo "SMOKE-DONE $SMOKE_DIR"
