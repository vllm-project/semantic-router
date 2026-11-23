#!/bin/bash
# recipe-lab FULL end-to-end (one fresh run directory, no reused state):
#
#   Stage 0  frozen MMLU-Pro case list + spec (content hash patched)
#   Stage 1  collect router_replay records -> answer bank -> attribution (Sub-1)
#   Stage 2  confidence-informed threshold proposal -> replay tier -> RANKED
#            ledger with --attribution-ref (Sub-2)
#   Stage 3  sr-bench live: baseline (no routing, local lane only) vs the best
#            replay candidate behind the router -> preview -> three-state
#            verdict -> live ledger record (Sub-3)
#   Stage 4  report / next / provenance / record-draft (Sub-4)
#
# The capability-aware mock makes the local lane fail deep questions, so both
# tiers can show real routing benefit (the live baseline deliberately runs
# without routing).
#
# Run on a GPU node (the router loads the decision model):
#   setsid nohup srun --gres=gpu:1 --cpus-per-task=8 --time=01:00:00 \
#     bash "$REPO/tools/recipe-lab/examples/smoke/e2e-full.sh" \
#     > e2e-full-launch.log 2>&1 < /dev/null &
# Environment (all optional; defaults derive from this script's location):
#   REPO            semantic-router checkout        (default: derived)
#   RECIPE_LAB_BIN  dir with recipe-lab CLI+python  (default: $REPO/.venv/bin)
#   PYTHON_BIN      python for the mock backend     (default: $RECIPE_LAB_BIN/python)
#   SR_BUILD        build prefix                    (default: $REPO/build)
#   ROUTER_BIN      router binary                   (default: $SR_BUILD/bin/semantic-router)
#   WORK_ROOT       scratch root for run dirs       (default: $REPO/.tmp-smoke-runs)
#   DATASET_DIR     local MMLU-Pro snapshot         (default: $WORK_ROOT/dataset/mmlu-pro)
#   HF_HOME         HuggingFace cache               (default: $WORK_ROOT/hf-home)
set -u
ASSETS="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$ASSETS/../../.." && pwd)}"
RL="${RECIPE_LAB_BIN:-$REPO/.venv/bin}"
PYBIN="${PYTHON_BIN:-$RL/python}"
SR_BUILD="${SR_BUILD:-$REPO/build}"
ROUTER_BIN="${ROUTER_BIN:-$SR_BUILD/bin/semantic-router}"
WORK_ROOT="${WORK_ROOT:-$REPO/.tmp-smoke-runs}"
DATASET_DIR="${DATASET_DIR:-$WORK_ROOT/dataset/mmlu-pro}"
HF_HOME="${HF_HOME:-$WORK_ROOT/hf-home}"
WORK="$WORK_ROOT/e2e-$(date +%m%d%H%M%S)"
mkdir -p "$WORK"
echo "E2E-WORK=$WORK"

ROUTER_PID=""
MOCK_PID=""
start_mock() { # $1 = cases file for the capability table
  "$PYBIN" "$ASSETS/mock_backend.py" --port 8000 --cases "$1" > "$WORK/mock-$2.log" 2>&1 &
  MOCK_PID=$!
  sleep 2
  curl -sf http://127.0.0.1:8000/v1/models > /dev/null || { echo E2E-FAILED mock; cat "$WORK/mock-$2.log"; exit 1; }
}
start_router() { # $1 = config file
  "$SR_BUILD/bin/semantic-router" -config "$1" -gateway standalone > "$WORK/router-$2.log" 2>&1 &
  ROUTER_PID=$!
  for i in $(seq 1 60); do
    sleep 5
    kill -0 $ROUTER_PID 2>/dev/null || { echo E2E-FAILED router-died; tail -30 "$WORK/router-$2.log"; exit 1; }
    if curl -sf http://127.0.0.1:8080/api/v1/inventory/model-runtime 2>/dev/null | grep -q '"ready":true'; then
      echo "ROUTER-READY ($2) after ~$((i*5))s"; return 0
    fi
  done
  echo E2E-FAILED not-ready; tail -30 "$WORK/router-$2.log"; exit 1
}
stop_stack() {
  for pid in ${ROUTER_PID:-} ${MOCK_PID:-}; do kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; done
  ROUTER_PID=""; MOCK_PID=""
}
cleanup() { stop_stack; }
trap cleanup EXIT

# ---------------- Stage 0: frozen dataset + spec ----------------
BUILD=$("$RL/python" "$ASSETS/build_mmlu_cases.py" \
  --dataset-dir "$DATASET_DIR" \
  --output "$WORK/cases-mmlu.jsonl" --per-deep 12 --per-standard 12 --seed 42) || exit 1
echo "$BUILD" | tee "$WORK/build.log"
# render the experiment spec: the workload identity is the cases file hashed
# at runtime (A4) - the template ships with placeholders
CASES_HASH=$("$RL/recipe-lab" hash --file "$WORK/cases-mmlu.jsonl" | "$RL/python" -c "import json,sys; print(json.load(sys.stdin)['content_hash'])")
sed -e "s|\${CASES_FILE}|$WORK/cases-mmlu.jsonl|g" -e "s|\${CASES_HASH}|$CASES_HASH|g" \
  "$ASSETS/smoke-spec.yaml.tmpl" > "$WORK/spec.yaml"
echo "SPEC-HASH=$CASES_HASH"

# ---------------- Stage 1: collect + attribute (Sub-1) ----------------
start_mock "$WORK/cases-mmlu.jsonl" "replay"
start_router "$ASSETS/smoke-config.yaml" "baseline"

"$RL/python" "$ASSETS/send_requests.py" --cases "$WORK/cases-mmlu.jsonl" \
  --router http://127.0.0.1:8899 > "$WORK/requests.log" 2>&1 || exit 1
curl -sf 'http://127.0.0.1:8080/api/v1/observability/replays?showDetails=true' > "$WORK/replays.json"
echo "REPLAYS-BYTES=$(wc -c < "$WORK/replays.json")"

"$RL/recipe-lab" collect-replays --url http://127.0.0.1:8080 \
  --cases "$WORK/cases-mmlu.jsonl" --output "$WORK/records.jsonl" --family d3 \
  > "$WORK/collect.json" 2>"$WORK/collect.err" || exit 1
"$RL/recipe-lab" bank build --cases "$WORK/cases-mmlu.jsonl" \
  --models local/private-qwen,cloud/frontier-reasoning \
  --endpoint http://127.0.0.1:8000 --output "$WORK/answer-bank.jsonl" \
  > "$WORK/bank.json" 2>"$WORK/bank.err" || exit 1
"$RL/recipe-lab" bank apply --bank "$WORK/answer-bank.jsonl" \
  --records "$WORK/records.jsonl" --output "$WORK/records-graded.jsonl" \
  > "$WORK/bank-apply.json" 2>"$WORK/bank-apply.err" || exit 1
"$RL/recipe-lab" attribute --spec "$WORK/spec.yaml" \
  --records "$WORK/records-graded.jsonl" --output-dir "$WORK/attrib" \
  > "$WORK/attribute.json" 2>"$WORK/attribute.err" || exit 1
echo ====ATTRIBUTION-SUMMARY====
cat "$WORK/attribute.json"
ATT_HASH="sha256:$(sha256sum "$WORK/attrib/attribution.jsonl" | cut -d' ' -f1)"
ATT_REF="$WORK/attrib/attribution.jsonl@$ATT_HASH"
echo "ATT_REF=$ATT_REF"

# confidence distribution of the frozen baseline pass (evidence for the
# proposal deltas; taken from the raw replay records)
"$PYBIN" - "$WORK/replays.json" > "$WORK/confidence.txt" <<'PYEOF'
import json, sys
records = json.load(open(sys.argv[1]))
rows = records.get("replays") or records.get("records") or records
if isinstance(rows, dict):
    rows = list(rows.values())
scores = []
for r in rows:
    det = r if isinstance(r, dict) else {}
    det = det.get("details", det)
    for key in ("confidence_score", "confidence", "score"):
        v = det.get(key)
        if isinstance(v, (int, float)):
            scores.append(float(v)); break
scores.sort()
n = len(scores)
def pct(p):
    return scores[min(n - 1, int(p * n))] if n else None
print(json.dumps({"count": n, "min": pct(0.0), "p10": pct(0.1), "p25": pct(0.25),
                  "p50": pct(0.5), "p75": pct(0.75), "p90": pct(0.9), "max": pct(1.0)}))
PYEOF
echo ====CONFIDENCE-DISTRIBUTION====
cat "$WORK/confidence.txt"

stop_stack

# ---------------- Stage 2: propose + replay tier (Sub-2) ----------------
# deltas probe around the threshold: tight when the distribution sits near
# 0.6, wide when it does not (the proposer reads the evidence, not a guess)
DELTA_SET="--deltas=-0.3,-0.15,0.15,0.3"
P25=$("$PYBIN" -c "import json; print(json.load(open('$WORK/confidence.txt'))['p25'])" 2>/dev/null)
P75=$("$PYBIN" -c "import json; print(json.load(open('$WORK/confidence.txt'))['p75'])" 2>/dev/null)
if [ -n "$P25" ] && [ -n "$P75" ]; then
  NEAR=$("$PYBIN" -c "print(1 if (abs($P25-0.6)<0.15 or abs($P75-0.6)<0.15) else 0)")
  [ "$NEAR" = "1" ] && DELTA_SET="--deltas=-0.1,-0.05,0.05,0.1"
fi
echo "DELTA_SET=$DELTA_SET"
"$RL/recipe-lab" propose-thresholds --base-config "$ASSETS/smoke-config.yaml" \
  --signal depth "$DELTA_SET" --output-dir "$WORK/candidates" \
  > "$WORK/propose.json" 2>"$WORK/propose.err" || exit 1

"$RL/recipe-lab" run --tier replay \
  --spec "$WORK/spec.yaml" \
  --cases "$WORK/cases-mmlu.jsonl" \
  --bank "$WORK/answer-bank.jsonl" \
  --candidates "$WORK/candidates" \
  --baseline-config "$ASSETS/smoke-config.yaml" \
  --proposer threshold-search \
  --attribution-ref "$ATT_REF" \
  --ledger "$WORK/ledger.jsonl" \
  --output-dir "$WORK/evals" \
  --router-bin "$ROUTER_BIN" \
  --mock-backend "$ASSETS/mock_backend.py" \
  --mock-extra-args "--cases $WORK/cases-mmlu.jsonl" \
  --python-bin "$PYBIN" \
  --hf-home "$HF_HOME" \
  --round 1 > "$WORK/run-replay.json" 2>"$WORK/run-replay.err" || exit 1
echo ====REPLAY-RESULT====
cat "$WORK/run-replay.json"
TOP_REF=$("$PYBIN" -c "
import json
d = json.load(open('$WORK/run-replay.json'))
best = d.get('best') or {}
print((best.get('candidate') or {}).get('ref', ''))
")
echo "TOP_REF=$TOP_REF"
[ -n "$TOP_REF" ] || TOP_REF="$ASSETS/smoke-config.yaml"

# ---------------- Stage 3: live tier via sr-bench (Sub-3) ----------------
"$RL/python" "$ASSETS/build_live_manifest.py" \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$WORK" \
  --per-deep 12 --per-standard 12 --seed 42 \
  --config-hash 0000000000000000000000000000000000000000000000000000000000000000 > "$WORK/manifest-build.json" || exit 1
cat "$WORK/manifest-build.json"

start_mock "$WORK/cases-mcq.jsonl" "live"
start_router "$TOP_REF" "candidate"

# explicit sr-bench service (a store with prior evidence refuses autostart;
# serve takes its store from --store via the module entrypoint)
cd "$REPO/src/vllm-sr"
setsid nohup "$RL/python" -m cli.sr_bench.service \
  --store "$WORK/srbench-store" --port 18099 > "$WORK/srbench-serve.log" 2>&1 < /dev/null &
sleep 4
export SRBENCH_URL="http://127.0.0.1:18099"
B=$("$RL/vllm-sr" benchmark --url "$SRBENCH_URL" run --manifest "$WORK/baseline-manifest.json" --detach 2>&1)
echo "$B" > "$WORK/baseline-submit.json"; tail -3 "$WORK/baseline-submit.json"
BASE_ID=$("$PYBIN" -c "import json; d=json.load(open('$WORK/baseline-submit.json')); print(d.get('id') or d.get('run_id',''))" 2>/dev/null)
echo "BASELINE-RUN=$BASE_ID"
[ -n "$BASE_ID" ] || { echo E2E-FAILED baseline; exit 1; }

C=$("$RL/vllm-sr" benchmark --url "$SRBENCH_URL" run --manifest "$WORK/candidate-manifest.json" --detach 2>&1)
echo "$C" > "$WORK/candidate-submit.json"; tail -3 "$WORK/candidate-submit.json"
CAND_ID=$("$PYBIN" -c "import json; d=json.load(open('$WORK/candidate-submit.json')); print(d.get('id') or d.get('run_id',''))" 2>/dev/null)
echo "CANDIDATE-RUN=$CAND_ID"
[ -n "$CAND_ID" ] || { echo E2E-FAILED candidate; exit 1; }

for i in $(seq 1 40); do
  sleep 10
  S=$("$RL/vllm-sr" benchmark --url "$SRBENCH_URL" show "$BASE_ID" 2>/dev/null | "$PYBIN" -c "import json,sys; print(json.load(sys.stdin).get('status',''))" 2>/dev/null)
  T=$("$RL/vllm-sr" benchmark --url "$SRBENCH_URL" show "$CAND_ID" 2>/dev/null | "$PYBIN" -c "import json,sys; print(json.load(sys.stdin).get('status',''))" 2>/dev/null)
  echo "status: baseline=$S candidate=$T"
  case "$S" in *running*|*pending*|*queued*) continue ;; esac
  case "$T" in *running*|*pending*|*queued*) continue ;; esac
  break
done

echo ====PREVIEW====
"$RL/vllm-sr" benchmark preview --manifest "$WORK/preview-manifest.json" \
  --url "$SRBENCH_URL" > "$WORK/preview.json" 2>"$WORK/preview.err" || echo "preview failed (see preview.err)"
head -40 "$WORK/preview.json"

cat > "$WORK/candidate.json" <<EOF
{"kind":"config","ref":"$TOP_REF","change_summary":"best replay-tier candidate behind the live router","provenance":{"proposer":"threshold-search","insight_refs":["$ATT_REF"]}}
EOF
set +e
"$RL/recipe-lab" live-verdict \
  --spec "$WORK/spec.yaml" \
  --baseline-run-id "$BASE_ID" \
  --candidate-run-id "$CAND_ID" \
  --ledger "$WORK/ledger.jsonl" \
  --candidate "$WORK/candidate.json" \
  --attribution-ref "$ATT_REF" \
  --srbench-bin "$RL/vllm-sr" --srbench-url "$SRBENCH_URL" > "$WORK/live-verdict.json" 2>"$WORK/live-verdict.err"
VERDICT_RC=$?
set -u
echo "LIVE-VERDICT-EXIT=$VERDICT_RC (0=PASS 1=FAIL 2=INCONCLUSIVE)"
cat "$WORK/live-verdict.json"

# ---------------- Stage 4: report / next / provenance (Sub-4) ----------------
"$RL/recipe-lab" report --ledger "$WORK/ledger.jsonl" --output "$WORK/report.md" > "$WORK/report.json" 2>&1
"$RL/recipe-lab" next --ledger "$WORK/ledger.jsonl" --output "$WORK/next-context.md" > "$WORK/next.json" 2>&1
"$RL/recipe-lab" provenance --ledger "$WORK/ledger.jsonl" --output "$WORK/manifest.json" > "$WORK/prov.json" 2>&1
"$RL/recipe-lab" record-draft --ledger "$WORK/ledger.jsonl" --output "$WORK/promotion-draft.md" > "$WORK/draft.json" 2>"$WORK/draft.err"
echo ====REPORT====
cat "$WORK/report.md"
echo ====PROVENANCE====
head -30 "$WORK/manifest.json"
echo "E2E-DONE $WORK"
