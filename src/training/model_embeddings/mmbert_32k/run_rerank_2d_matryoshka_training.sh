#!/usr/bin/env bash

set -euo pipefail

MMBERT32K_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MMBERT32K_REPO_ROOT="$(cd -- "${MMBERT32K_SCRIPT_DIR}/../../../.." && pwd)"
MMBERT32K_PYTHON_BIN="${MMBERT32K_PYTHON_BIN:-python3}"

export PYTHONPATH="${MMBERT32K_REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${MMBERT32K_PYTHON_BIN}" \
  -m training.model_embeddings.mmbert_32k \
  --config "${MMBERT32K_SCRIPT_DIR}/configs/reranker.json" \
  "$@"
