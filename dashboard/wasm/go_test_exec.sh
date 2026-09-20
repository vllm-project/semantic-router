#!/usr/bin/env bash
set -euo pipefail

# Go's WASM startup reserves only 8 KiB for argv and environment together.
# Keep the Go-provided Node runner, but do not copy the CI job environment
# into the test module. Compilation still uses the caller's full environment.
runner="$(go env GOROOT)/lib/wasm/go_js_wasm_exec"
exec env -i PATH="$PATH" TMPDIR="${TMPDIR:-${TMP:-${TEMP:-/tmp}}}" \
    "$runner" "$@"
