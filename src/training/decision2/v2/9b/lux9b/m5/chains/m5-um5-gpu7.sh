#!/usr/bin/env bash
# usage: m5-um5-gpu7.sh SHA   9B M5 UM5 line (S = 1/2 KD soup + 1/2 KG soup; both soups won the seed rule), node A GPU7.
set -uo pipefail
SHA=${1:?mirror sha}
L=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2/v2/9b/lux9b/m5
[ -x "$L/chain-step.sh" ] || { echo "mirror $L missing" >&2; exit 2; }
G=7
C=m5-um5-gpu7
LUX=/m3/pf-D-s1-zero/run/checkpoint-0000000
step() { "$L/chain-step.sh" "$SHA" "$G" "$C" "$@"; }
step UM5-a12 30 -- "$L/soup.sh" "$SHA" "$G" UM5-a12 /m5/KD-soup-build/soup /m5/KG-soup-build/soup $LUX $LUX || exit 1
step UM5-a23 30 -- "$L/soup.sh" "$SHA" "$G" UM5-a23 /m5/KD-soup-build/soup /m5/KG-soup-build/soup $LUX || exit 1
echo "chain $C done"
