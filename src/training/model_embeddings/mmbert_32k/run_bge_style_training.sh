#!/usr/bin/env bash

set -euo pipefail

MMBERT32K_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MMBERT32K_REPO_ROOT="$(cd -- "${MMBERT32K_SCRIPT_DIR}/../../../.." && pwd)"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is required. Install it: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

export PYTHONPATH="${MMBERT32K_REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

uv_args=(run --locked --project "${MMBERT32K_SCRIPT_DIR}")
if [[ -n "${MMBERT32K_PYTHON_BIN:-}" ]]; then
  uv_args+=(--python "${MMBERT32K_PYTHON_BIN}")
fi

exec uv "${uv_args[@]}" \
  python -m training.model_embeddings.mmbert_32k \
  --config "${MMBERT32K_SCRIPT_DIR}/configs/embedder.json" \
  "$@"
