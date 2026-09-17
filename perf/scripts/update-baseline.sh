#!/usr/bin/env bash
# Preserve the same parsed model identities used by the regression gate.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PERF_DIR="$(dirname "$SCRIPT_DIR")"
BASELINE_DIR="$PERF_DIR/testdata/baselines"
PROJECT_ROOT="$(dirname "$PERF_DIR")"
BENCH_RESULTS="$PROJECT_ROOT/reports/bench-results.txt"
PARSED_RESULTS="$(mktemp)"
trap 'rm -f "$PARSED_RESULTS"' EXIT
if [[ ! -f "$BENCH_RESULTS" ]]; then
  echo "Benchmark results not found at $BENCH_RESULTS; run make perf-baseline-update" >&2
  exit 1
fi
(
  cd "$PERF_DIR"
  go run ./cmd/perftest --parse-bench="$BENCH_RESULTS" --output="$PARSED_RESULTS"
)
python3 - "$PARSED_RESULTS" "$BASELINE_DIR" <<'PY'
import json
import pathlib
import re
import sys
source = json.loads(pathlib.Path(sys.argv[1]).read_text())
destination = pathlib.Path(sys.argv[2])
destination.mkdir(parents=True, exist_ok=True)
patterns = {
    "classification": r"^Benchmark(Classify|CGO)",
    "decision": r"^Benchmark(Evaluate|Rule|Priority)",
    "cache": r"^BenchmarkCache",
    "looper": r"^Benchmark(ReMoM|Fusion|Flow|Base)",
}
for family, pattern in patterns.items():
    metrics = {name: metric for name, metric in source["benchmarks"].items()
               if re.match(pattern, name)}
    if family in {"classification", "cache"}:
        for name, metric in metrics.items():
            if not metric.get("model_identity"):
                raise SystemExit(f"{name} has no model identity; rerun the current benchmark harness")
    payload = {**source, "benchmarks": metrics}
    (destination / f"{family}.json").write_text(json.dumps(payload, indent=2) + "\n")
print(f"Updated baselines from source {source['git_commit']}")
PY
