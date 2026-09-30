#!/usr/bin/env bash
# usage: line.sh SHA GPU LINE MEMBER_RUN...
# One Milestone 7 line: the uniform FP32 soup LINE-soup of the member continuations (CPU), the
# interpolations toward Lux 1.0 at alpha 1/3, 1/2, 2/3 (LINE-a13 / -a12 / -a23; built concurrently
# on CPU), then for each point in that order CAL698 and the 16K readouts typed DEV, CSS pilot,
# HT-DEV v2, PN1 dev and MLX-DEV-9B (readout.sh) and its development screens (screens.sh vs the
# incumbent's re-read ref-ka13). The soup itself (alpha 1) is not read. Stops at the first failure.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
sha=$1; gpu=$2; line=$3; shift 3
L=$(code_dir "$sha")/v2/9b/lux9b/m7
[ $# -ge 2 ] || { echo "a line needs at least two member runs" >&2; exit 2; }
if [ ! -f "$M7/$line-soup-build/end-utc.txt" ]; then
  bash "$L/soup.sh" "$sha" "$gpu" "$line-soup" --no-readout "$@" || exit 1
fi
pids=()
mkdir -p "$M7/logs"
for p in 1/3=a13 1/2=a12 2/3=a23; do
  f=${p%=*}; tag=${p#*=}
  [ -f "$M7/$line-$tag-build/end-utc.txt" ] && continue
  bash "$L/interp.sh" "$sha" "$gpu" "$line-$tag" "/m7/$line-soup-build/soup" "${f%/*}" "${f#*/}" --no-readout \
    > "$M7/logs/build-$line-$tag.log" 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid" || { echo "an interpolation build of $line failed" >&2; exit 1; }; done
for tag in a13 a12 a23; do
  bash "$L/readout.sh" "$sha" "$gpu" "$line-$tag" "/m7/$line-$tag-build/soup" --panel dev --panel css-pilot \
    --panel ht-dev2 --panel pn1-dev --mlxdev || exit 1
  bash "$L/screens.sh" "$sha" "$line-$tag" ref-ka13 || exit 1
done
echo "line $line done"
