#!/usr/bin/env bash
# bf16z end-to-end check on a small released model (coordinator note 2026-09-30 07:20: finish the codec and tests).
# Node A, from the exact mirror holding this file.
#   e2e-0p8b.sh compress          bf16z of DEV2.0-0.8B's BF16 checkpoint (bede7938) + full decompress-and-hash verify (CPU)
#   e2e-0p8b.sh run --gpu <0|1>   release.sh WITHOUT upload on the bf16z spec: build, examples in two processes, card
#                                 example and subset parity (150 / 200 / 100 / 150) through the package runtime's bf16z
#                                 restore; nothing is written to the Hub
set -euo pipefail
mode="${1:-}"
shift || true
gpu=""
[[ "${1:-}" == --gpu ]] && gpu=$2
TS=$(date -u +%Y%m%dT%H%M%SZ)
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
SRC=$(basename "$(cd "$S/../../.." && pwd)")
IMG=decision20-train-fast:host2
SOURCE=/data/dev2/runs/release/inputs/dev2-0p8b-bf16
IN=/data/dev2/runs/release/inputs/dev2-0p8b-bf16z
G=/data/dev2/private/panels/goldfree
P=/data/dev2/runs/release/inputs/dev2-0p8b-t1/derived
D=/data/dev2/runs/release/decisions
CACHE=/data/dev2/runs/release/bf16z-cache
FROZEN=/data/dev2/runs/dec/formal/m2/m2-E8F-soup-nodeA-triton
FROZEN_DIGEST=5e37a14373b70584bcc2e0056ad7ef5ebaf3d01b0a1820d216717a23232f09e2
cpu=(docker run --rm --network none -e HIP_VISIBLE_DEVICES= -e PYTHONPATH="$S" -v "$S:$S:ro" -w "$S" --entrypoint python3)
case "$mode" in
  compress)
    mkdir "$IN"
    "${cpu[@]}" -v "$SOURCE:$SOURCE:ro" -v "$IN:$IN" "$IMG" -B -m v2.release.runtime.bf16z compress \
      --source "$SOURCE/checkpoint" --output "$IN/checkpoint" --receipt "$IN/bf16z.json"
    "${cpu[@]}" -v "$IN:$IN" "$IMG" -B -m v2.release.runtime.bf16z verify --dir "$IN/checkpoint" \
      --receipt "$IN/bf16z.json" --output "$IN/bf16z-verify.json"
    sha256sum "$IN/bf16z.json" "$IN/bf16z-verify.json" ;;
  run)
    [[ "$gpu" == 0 || "$gpu" == 1 ]] || { echo "node-A GPU0 or GPU1 only" >&2; exit 2; }
    [[ -e "$D/DEV2.0-0.8B.decision.bf16.json" ]] || { echo "the BF16 revision's decision is missing" >&2; exit 1; }
    digest() { (cd "$1" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64); }
    [[ "$(digest "$FROZEN")" == "$FROZEN_DIGEST" ]] || { echo "frozen cache changed" >&2; exit 1; }
    TC=/data/dev2/runs/release/triton/bf16z-e2e-0p8b-$TS
    cp -a "$FROZEN" "$TC"
    mkdir -p "$CACHE"
    W=/data/dev2/runs/release/bf16z-e2e-0p8b-$TS
    trap 'rm -f "/data/dev2/leases/gpu$gpu.lock/owner.release-bf16z"' EXIT
    "$S/v2/release/release.sh" --spec "$S/v2/release/specs/dev2-0p8b-bf16z-e2e.json" --src "$SRC" --work "$W" \
      --gpu "$gpu" --track release-bf16z --shared-lease release-bf16z --threads 4 \
      --site /opt/decision-fla --require-kernels --env TRITON_CACHE_AUTOTUNING=1 --env HIP_FORCE_DEV_KERNARG=1 \
      --env "TRITON_CACHE_DIR=$TC" --mount-rw "$TC" --env "DECISION2_CACHE=$CACHE" --mount-rw "$CACHE" \
      --mount "$G" --mount "$P" --mount "$IN" --mount "$SOURCE" \
      --parity "typed-final:$G/typed-final.prompts.jsonl:$P/typed-final.predictions.jsonl:150" \
      --parity "css15:$G/css15.prompts.jsonl:$P/css15.predictions.jsonl:200" \
      --parity "public231:$G/public231.prompts.jsonl:$P/public231.predictions.jsonl:100" \
      --parity "mlx-diag:$G/mlx-diag.prompts.jsonl:$P/mlx-diag.predictions.jsonl:150"
    echo "work=$W restored=$(ls "$CACHE/bf16z")" ;;
  *) echo "mode: compress | run --gpu N" >&2; exit 2 ;;
esac
