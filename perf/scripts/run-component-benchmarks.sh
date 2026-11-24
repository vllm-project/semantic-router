#!/usr/bin/env bash
set -euo pipefail

scope="${1:?usage: run-component-benchmarks.sh models|all BENCHTIME TIMEOUT}"
benchtime="${2:?missing benchmark duration}"
timeout="${3:?missing Go test timeout}"

case "$scope" in
  models|all) ;;
  *) echo "Unknown benchmark scope: $scope" >&2; exit 2 ;;
esac

cd "$(dirname "${BASH_SOURCE[0]}")/.."

run_group() {
  local group="$1"
  shift
  printf 'Running %s benchmarks\n' "$group"
  go test -run '^$' "$@" -benchmem -benchtime="$benchtime" -timeout="$timeout" ./benchmarks/...
}

# TestMain releases the model-runtime lease when each Go test process exits.
run_group "cache" -bench '^BenchmarkCache'
run_group "classification" -bench '^BenchmarkClassify' -skip '^BenchmarkClassifyInputLength$'
run_group "long-input classification" -bench '^BenchmarkClassifyInputLength$'

if [[ "$scope" == all ]]; then
  run_group "non-model component" -bench . -skip '^(BenchmarkCache|BenchmarkClassify)'
fi
