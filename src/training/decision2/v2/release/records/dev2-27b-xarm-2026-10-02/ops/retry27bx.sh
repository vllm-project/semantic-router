#!/usr/bin/env bash
# Decision-2.0-Vega-27B (27B worker, continuation #6): retry release27bx.sh ARM --release once the organization's HF
# private storage has room. The first --release of M6-IBxIB2-m50 passed every pre-upload check and was refused at the
# commit ("Private repository storage limit reached"); its package is 14.98 GB. Every POLL seconds (default 300) this
# checks hf_headroom.sh --min-free-gb NEED (default 20), that Vega's main is still the superseded revision of
# release27bx.sh and that no Vega-27B release.sh runs; then it starts release27bx.sh once (the script re-checks both)
# and exits with its status. Without room by DEADLINE (UTC, default 24 h after the start) it exits 3 without releasing.
# Usage (node A, detached): TF518_DIGEST=... bash <mirror>/v2/release/records/dev2-27b-xarm-2026-10-02/ops/retry27bx.sh \
#   ARM --gpu N [--need GB] [--poll S] [--deadline YYYY-MM-DDTHH:MM:SSZ]
set -euo pipefail
ARM="${1:-}"
shift || true
gpu="" need=20 poll=300 deadline=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu=$2; shift 2 ;;
    --need) need=$2; shift 2 ;;
    --poll) poll=$2; shift 2 ;;
    --deadline) deadline=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ "$gpu" =~ ^[0-7]$ && "$need" =~ ^[0-9]+$ && "$poll" =~ ^[0-9]+$ ]] || { echo "bad --gpu / --need / --poll" >&2; exit 2; }
[[ -n "${TF518_DIGEST:-}" ]] || { echo "TF518_DIGEST is required" >&2; exit 2; }
OPS=$(cd "$(dirname "$0")" && pwd)
S=$(cd "$OPS/../../../../.." && pwd)
superseded=$(sed -n 's/^superseded=\([0-9a-f]\{40\}\)$/\1/p' "$OPS/release27bx.sh")
[[ "$superseded" =~ ^[0-9a-f]{40}$ ]] || { echo "no superseded revision in release27bx.sh" >&2; exit 2; }
end=$(date -u -d "${deadline:-24 hours}" +%s)
HFPY=/data/dev2/tools/hf-cli/bin/python
echo "$(date -u +%FT%TZ) retry27bx $ARM gpu $gpu: waiting for >= $need GB private headroom (superseded $superseded)"
while :; do
  if bash "$S/v2/common/hf_headroom.sh" --min-free-gb "$need" > /dev/null 2>&1; then
    main=$("$HFPY" -c 'import sys; from huggingface_hub import HfApi; print(HfApi().model_info(sys.argv[1]).sha)' \
      vllm-sr/Decision-2.0-Vega-27B)
    if [[ "$main" != "$superseded" ]]; then
      echo "$(date -u +%FT%TZ) Vega main moved to $main; not releasing" >&2
      exit 3
    fi
    if ! pgrep -af "v2/release/release[.]sh" | grep -q -- "/specs/dev2-27b-"; then
      echo "$(date -u +%FT%TZ) headroom >= $need GB; starting release27bx.sh $ARM --release --gpu $gpu"
      bash "$OPS/release27bx.sh" "$ARM" --release --gpu "$gpu"
      exit $?
    fi
  fi
  if (( $(date -u +%s) > end )); then
    echo "$(date -u +%FT%TZ) no room by the deadline; not releasing" >&2
    exit 3
  fi
  sleep "$poll"
done
