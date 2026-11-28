#!/bin/bash
# recipe-lab calibration smoke (Sub-issue 7 / J2): the offline BUG-3 loop.
#
# diagnose (coverage / distributions / reachability) -> propose-calibrated
# (the observed p75 of difficulty) over a synthetic pilot-shaped row set and
# the agent-recipe fixture. No GPU, no router. The same flow is asserted as a
# pytest case in tests/test_calibration_smoke.py.
#
#   bash tools/recipe-lab/examples/smoke/calibration-smoke.sh
#
# Environment (all optional; defaults derive from this script's location):
#   REPO            semantic-router checkout        (default: derived)
#   RECIPE_LAB_BIN  dir with recipe-lab CLI+python  (default: $REPO/.venv/bin)
#   PYTHON_BIN      python for the row generator    (default: $RECIPE_LAB_BIN/python)
#   WORK_ROOT       scratch root for run dirs       (default: $REPO/.tmp-smoke-runs)
set -u
ASSETS="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$ASSETS/../../../.." && pwd)}"
RL="${RECIPE_LAB_BIN:-$REPO/.venv/bin}"
PYBIN="${PYTHON_BIN:-$RL/python}"
WORK_ROOT="${WORK_ROOT:-$REPO/.tmp-smoke-runs}"
WORK="$WORK_ROOT/calibration-smoke-$(date +%m%d%H%M%S)"
mkdir -p "$WORK"
echo "WORK=$WORK"
export PYTHONPATH="$REPO/tools/recipe-lab${PYTHONPATH:+:$PYTHONPATH}"

# the console script when installed, the module entry otherwise (pure checkout)
run_cli() {
  if [ -x "$RL/recipe-lab" ]; then
    "$RL/recipe-lab" "$@"
  else
    "$PYBIN" -m recipe_lab.cli "$@"
  fi
}

# 1. synthetic pilot-shaped rows: 344 difficulty scores in [0.08, 1.87],
#    nothing at or above the stale 2.5 thresholds (BUG-3 appendix A)
FIXTURE="$REPO/tools/recipe-lab/tests/fixtures/agent_recipe_min.yaml"
ROWS="$WORK/rows.jsonl"
"$PYBIN" - "$ROWS" <<'PY' || { echo "FAIL: row generation"; exit 1; }
import sys

from tests.fixtures import pilot_rows_jsonl

with open(sys.argv[1], "w", encoding="utf-8") as fh:
    fh.write(pilot_rows_jsonl())
PY

# 2. diagnose: dead lanes, compressed difficulty scale, unreachable gte 2.5
run_cli diagnose \
  --records "$ROWS" \
  --config "$FIXTURE" \
  --output-dir "$WORK/diag" \
  --source "calibration smoke: synthetic BUG-3 pilot" \
  > "$WORK/diagnose.json" || { echo "FAIL: diagnose"; cat "$WORK/diagnose.json"; exit 1; }
"$PYBIN" - "$WORK/diagnose.json" <<'PY' || { echo "FAIL: diagnose assertions"; exit 1; }
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["dead_decisions"] == ["domain_code_complex", "complex_general"], payload["dead_decisions"]
assert len(payload["unreachable_sites"]) == 4, payload["unreachable_sites"]
assert any(a["type"] == "compressed_range" and a["signal"] == "difficulty" for a in payload["alerts"])
print("diagnose: 2 dead lanes, 4 unreachable sites, compressed difficulty scale")
PY

# 3. propose-calibrated: move all four gte 2.5 sites to the observed p75
run_cli propose-calibrated \
  --diagnosis "$WORK/diag/diagnosis.json" \
  --base-config "$FIXTURE" \
  --signal difficulty \
  --from 2.5 \
  --quantiles 0.75 \
  --output-dir "$WORK/cands" \
  > "$WORK/propose.json" || { echo "FAIL: propose-calibrated"; cat "$WORK/propose.json"; exit 1; }
"$PYBIN" - "$WORK/propose.json" "$FIXTURE" <<'PY' || { echo "FAIL: calibration assertions"; exit 1; }
import json
import sys
from pathlib import Path

import yaml

from recipe_lab.proposers import rule_threshold

payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["candidates"] == ["difficulty gte 2.5 -> 1.75 (p75 of n=344)"], payload["candidates"]
candidate = Path(payload["files"][0])
sidecar = json.loads(candidate.with_suffix(".candidate.json").read_text(encoding="utf-8"))
assert sidecar["provenance"]["proposer"] == "distribution-calibration"
assert len(sidecar["provenance"]["insight_refs"]) == 4, sidecar["provenance"]
fixture = yaml.safe_load(Path(sys.argv[2]).read_text(encoding="utf-8"))
config = yaml.safe_load(candidate.read_text(encoding="utf-8"))
golden = rule_threshold.propose_rule_thresholds(fixture, "difficulty", values=[1.75], from_value=2.5)
assert len(golden) == 1 and config == golden[0].config, "candidate must match the rule-level golden"
print("propose-calibrated: difficulty gte 2.5 -> 1.75 on all four sites (golden match)")
PY

echo "CALIBRATION-SMOKE-OK"
