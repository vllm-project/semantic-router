#!/bin/bash
# recipe-lab loop smoke (Sub-issue 2 closed loop):
# answer bank -> threshold proposer -> replay-tier run over candidates (each
# candidate gets its own router lifecycle) -> RANKED records -> ranking.
#
# Needs a GPU node (the router loads the decision model). Example:
#   srun --gres=gpu:1 --cpus-per-task=8 --time=00:50:00 \
#     bash tools/recipe-lab/examples/smoke/loop-smoke.sh
#
# Environment (all optional; defaults derive from this script's location):
#   REPO            semantic-router checkout        (default: derived)
#   RECIPE_LAB_BIN  dir with recipe-lab CLI+python  (default: $REPO/.venv/bin)
#   PYTHON_BIN      python for the mock backend     (default: $RECIPE_LAB_BIN/python)
#   SR_BUILD        build prefix                    (default: $REPO/build)
#   ROUTER_BIN      router binary                   (default: $SR_BUILD/bin/semantic-router)
#   WORK_ROOT       scratch root for run dirs       (default: $REPO/.tmp-smoke-runs)
#   HF_HOME         HuggingFace cache               (default: $WORK_ROOT/hf-home)
set -u
ASSETS="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$ASSETS/../../.." && pwd)}"
RL="${RECIPE_LAB_BIN:-$REPO/.venv/bin}"
PYBIN="${PYTHON_BIN:-$RL/python}"
SR_BUILD="${SR_BUILD:-$REPO/build}"
ROUTER_BIN="${ROUTER_BIN:-$SR_BUILD/bin/semantic-router}"
WORK_ROOT="${WORK_ROOT:-$REPO/.tmp-smoke-runs}"
HF_HOME="${HF_HOME:-$WORK_ROOT/hf-home}"
WORK="$WORK_ROOT/loop-smoke-$(date +%m%d%H%M%S)"
mkdir -p "$WORK"
echo "WORK=$WORK"

# 0. render the spec template: the frozen dataset identity is the real cases
#    file hash (A4); run verifies it before touching the data
CASES_HASH=$("$RL/recipe-lab" hash --file "$ASSETS/cases.jsonl" | "$PYBIN" -c "import json,sys; print(json.load(sys.stdin)['content_hash'])")
sed -e "s|\${CASES_FILE}|$ASSETS/cases.jsonl|g" -e "s|\${CASES_HASH}|$CASES_HASH|g" \
  "$ASSETS/smoke-spec.yaml.tmpl" > "$WORK/spec.yaml"

# 1. answer bank: mock backend up temporarily (direct queries, no router)
"$PYBIN" "$ASSETS/mock_backend.py" --port 8000 > "$WORK/mock.log" 2>&1 &
MOCK_PID=$!
sleep 2
"$RL/recipe-lab" bank build --cases "$ASSETS/cases.jsonl" \
  --models local/private-qwen,cloud/frontier-reasoning \
  --endpoint http://127.0.0.1:8000 --output "$WORK/answer-bank.jsonl" || exit 1
kill $MOCK_PID 2>/dev/null
wait $MOCK_PID 2>/dev/null

# 2. threshold proposer: candidate configs for the depth signal
"$RL/recipe-lab" propose-thresholds --base-config "$ASSETS/smoke-config.yaml" \
  --signal depth "--deltas=-0.1,-0.05,0.05,0.1" --output-dir "$WORK/candidates" || exit 1

# 3. replay-tier run: baseline anchors the deltas, then each candidate gets its
#    own router lifecycle; RANKED records land in the ledger
"$RL/recipe-lab" run --tier replay \
  --spec "$WORK/spec.yaml" \
  --cases "$ASSETS/cases.jsonl" \
  --bank "$WORK/answer-bank.jsonl" \
  --candidates "$WORK/candidates" \
  --baseline-config "$ASSETS/smoke-config.yaml" \
  --proposer threshold-search \
  --ledger "$WORK/ledger.jsonl" \
  --output-dir "$WORK/evals" \
  --router-bin "$ROUTER_BIN" \
  --mock-backend "$ASSETS/mock_backend.py" \
  --python-bin "$PYBIN" \
  --hf-home "$HF_HOME" \
  --round 1 || exit 1

# 4. ranking
echo ====LEDGER-TOP====
"$RL/recipe-lab" ledger top --file "$WORK/ledger.jsonl" --metric-key capability.score
echo "LOOP-DONE $WORK"
