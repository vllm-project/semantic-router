#!/usr/bin/env bash
# DEV2.0-8B storage test, step 2: no-upload staging run of the BF16-storage package through release.sh (build,
# System One examples, repeatability, card, full-panel parity) on node A GPU6 with a fresh copy of the persisted
# autotune cache that the scored run and its mlx-diag run shared. Parity is category-only (--parity-tolerance 1) so
# the receipts count every answer change and report the largest probability drift; BF16 is adopted only with 0
# answer changes on all four panels. Nothing is uploaded.
set -euo pipefail
TS=$(date -u +%Y%m%dT%H%M%SZ)
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
SRC=$(basename "$(cd "$S/../../.." && pwd)")
G=/data/dev2/private/panels/goldfree
P=/data/dev2/runs/release/inputs/dev2-8b-t1/derived
FROZEN=/data/dev2/runs/9b/formal-m4/triton-cache
TC=/data/dev2/runs/release/triton/dev2-8b-K-a13-copy-$TS
W=/data/dev2/runs/release/dev2-8b-bf16-parity-$TS
trap 'rm -f /data/dev2/leases/gpu6.lock/owner.release' EXIT
cp -a "$FROZEN" "$TC"
digest() { (cd "$1" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64); }
echo "mirror $SRC; cache copy $TC files=$(find "$TC" -type f | wc -l) digest=$(digest "$TC") frozen_digest=$(digest "$FROZEN")"
set -x
"$S/v2/release/release.sh" --spec "$S/v2/release/specs/dev2-8b-bf16-parity-staging.json" --src "$SRC" --work "$W" \
  --gpu 6 --track release-9b --shared-lease release --threads 4 \
  --site /opt/decision-fla --require-kernels \
  --env HIP_FORCE_DEV_KERNARG=1 --env TRITON_CACHE_AUTOTUNING=1 --env TRITON_CACHE_DIR="$TC" --mount-rw "$TC" \
  --mount "$G" --mount "$P" --parity-tolerance 1 \
  --parity "typed-final:$G/typed-final.prompts.jsonl:$P/typed-final.predictions.jsonl:1600" \
  --parity "css15:$G/css15.prompts.jsonl:$P/css15.predictions.jsonl:6547" \
  --parity "public231:$G/public231.prompts.jsonl:$P/public231.predictions.jsonl:231" \
  --parity "mlx-diag:$G/mlx-diag.prompts.jsonl:$P/mlx-diag.predictions.jsonl:2275"
set +x
echo "cache after run files=$(find "$TC" -type f | wc -l) digest=$(digest "$TC") frozen_digest=$(digest "$FROZEN")"
echo "work=$W"
