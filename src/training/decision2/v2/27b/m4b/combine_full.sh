#!/usr/bin/env bash
# M4b soups and the interpolation line on node B (host side; a CPU-only container of the pinned image, no GPU,
# no network): v2.27b.m4b.interp_full build (with its own bitwise verification) then a separate verify pass.
# Usage: combine_full.sh MIRROR_SHA NAME soup CKPT_1 CKPT_2 [CKPT...]
#        combine_full.sh MIRROR_SHA NAME interp ALPHA S_CKPT B_CKPT      (theta = ALPHA*S + (1 - ALPHA)*B)
#   -> /data/dev2/runs/27b/m4b/NAME/checkpoint (+ combination_manifest.json inside), NAME/verify.json
set -euo pipefail
echo "m4b combine_full $* start $(date -u +%FT%TZ)"

MIRROR=$1 NAME=$2 MODE=$3
shift 3
S=/data/dev2/src/$MIRROR/src/training/decision2
[ -d "$S" ] || S=/data/dev2/src/$MIRROR-src_training_decision2/src/training/decision2
[ -f "$S/v2/27b/m4b/interp_full.py" ] || { echo "no m4b code in mirror $MIRROR" >&2; exit 2; }
S=$(cd "$S" && pwd -P)
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "NAME must be one directory name" >&2; exit 2; }
IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
# COMBINE_ROOT / COMBINE_PREFIX: Milestone 5 output root and container prefix (default M4b's).
OUT=${COMBINE_ROOT:-/data/dev2/runs/27b/m4b}/$NAME
CPUS=${COMBINE_CPUS:-16}
case "$MODE" in
  soup) [ $# -ge 2 ] || { echo "a soup needs two or more checkpoints" >&2; exit 2; }
    members=("$@") args=()
    for m in "${members[@]}"; do args+=(--member "$m"); done ;;
  interp) [ $# -eq 3 ] || { echo "interp takes ALPHA S_CKPT B_CKPT" >&2; exit 2; }
    [[ "$1" =~ ^[0-9]+(/[0-9]+)?$ ]] || { echo "ALPHA must be p/q" >&2; exit 2; }
    members=("$2" "$3") args=(--s "$2" --b "$3" --alpha "$1") ;;
  *) echo "mode is soup or interp" >&2; exit 2 ;;
esac
[ ! -e "$OUT/checkpoint" ] || { echo "$OUT/checkpoint exists" >&2; exit 66; }
mounts=(--mount "type=bind,src=$S,dst=$S,readonly" --mount "type=bind,src=$OUT,dst=$OUT")
for m in "${members[@]}"; do
  [ -f "$m/decision_config.json" ] || { echo "missing checkpoint $m" >&2; exit 2; }
  mounts+=(--mount "type=bind,src=$m,dst=$m,readonly")
done
mkdir -p "$OUT"
export TMPDIR=/data/dev2/tmp
cpu() {
  local name=$1
  shift
  docker run --rm --name "${COMBINE_PREFIX:-d2-27b-m4b}-$NAME-$name" --network none --cpus "$CPUS" -e "OMP_NUM_THREADS=$CPUS" \
    -e HIP_VISIBLE_DEVICES= -e ROCR_VISIBLE_DEVICES= -e "PYTHONPATH=$S" -e PYTHONDONTWRITEBYTECODE=1 \
    "${mounts[@]}" -w "$S" --entrypoint python3 "$IMAGE" -m v2.27b.m4b.interp_full "$@"
}
cpu build "$MODE" "${args[@]}" --output "$OUT/checkpoint" 2>&1 | tee "$OUT/combine.log"
cpu verify verify --output "$OUT/checkpoint" --report "$OUT/verify.json" 2>&1 | tee "$OUT/verify.log"
echo "m4b combine_full $NAME complete"
