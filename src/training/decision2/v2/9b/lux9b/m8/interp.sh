#!/usr/bin/env bash
# usage: interp.sh SHA GPU NAME SOUP_PATH NUM DEN [--base PATH] [--runtime RSHA] [--no-readout]
# Weight interpolation alpha = NUM/DEN from the base toward SOUP_PATH (a full checkpoint under
# /m8/, /m7/, /m6/, /m4/ or /m3/): soup.sh with NUM copies of SOUP_PATH and DEN-NUM copies of the base
# (default the checkpoint-form Lux 1.0 /m3/pf-D-s1-zero/run/checkpoint-0000000), then CAL698 and
# the 16K typed DEV + CSS pilot readout unless --no-readout (CPU build only).
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
sha=$1; gpu=$2; name=$3; soup=$4; num=$5; den=$6; shift 6
base=/m3/pf-D-s1-zero/run/checkpoint-0000000; rt=$RUNTIME_DEFAULT; opts=()
while [ $# -gt 0 ]; do
  case "$1" in
    --base) base=${2:?--base needs a path}; shift 2 ;;
    --runtime) rt=${2:?--runtime needs a SHA}; shift 2 ;;
    --no-readout) opts+=(--no-readout); shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ "$num" =~ ^[0-9]+$ && "$den" =~ ^[0-9]+$ ]] && [ "$num" -ge 1 ] && [ "$num" -lt "$den" ] \
  || { echo "need integers 1 <= NUM < DEN" >&2; exit 2; }
members=()
for _ in $(seq 1 "$num"); do members+=("$soup"); done
for _ in $(seq 1 $((den - num))); do members+=("$base"); done
exec "$(code_dir "$sha")/v2/9b/lux9b/m8/soup.sh" "$sha" "$gpu" "$name" --runtime "$rt" "${opts[@]}" "${members[@]}"
