#!/usr/bin/env bash
# usage: interp.sh SHA GPU NAME SOUP_PATH NUM DEN
# Weight interpolation alpha = NUM/DEN from Lux 1.0 toward SOUP_PATH (a full checkpoint under
# /m4/ or /m3/): soup.sh with NUM copies of SOUP_PATH and DEN-NUM copies of the Lux full
# checkpoint /m3/pf-D-s1-zero/run/checkpoint-0000000, then CAL698 and 16K dev readouts.
set -uo pipefail
sha=$1; gpu=$2; name=$3; soup=$4; num=$5; den=$6
LUX=/m3/pf-D-s1-zero/run/checkpoint-0000000
[[ "$num" =~ ^[0-9]+$ && "$den" =~ ^[0-9]+$ ]] && [ "$num" -ge 1 ] && [ "$num" -lt "$den" ] \
  || { echo "need integers 1 <= NUM < DEN" >&2; exit 2; }
members=()
for _ in $(seq 1 "$num"); do members+=("$soup"); done
for _ in $(seq 1 $((den - num))); do members+=("$LUX"); done
exec "/data/dev2/src/$sha-src_training_decision2/src/training/decision2/v2/9b/lux9b/m4/soup.sh" \
  "$sha" "$gpu" "$name" "${members[@]}"
