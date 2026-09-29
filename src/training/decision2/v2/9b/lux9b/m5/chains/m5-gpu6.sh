#!/usr/bin/env bash
# usage: m5-gpu6.sh SHA   (started on node A by m5/launch.sh after upload_chain.sh verified it)
# 9B Milestone 5 chain, node A GPU6 (prereg records/lux9b-m5-prereg-2026-09-29.md): the incumbent's
# reference readout, arm KD (two seeds, preflights with the first), its soup and its interpolation
# line toward Lux 1.0 (alpha 1/3, 1/2, 2/3). Stops at the first failed step.
set -uo pipefail
SHA=${1:?mirror sha}
L=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2/v2/9b/lux9b/m5
[ -x "$L/chain-step.sh" ] || { echo "mirror $L missing" >&2; exit 2; }
G=6
C=m5-gpu6
step() { "$L/chain-step.sh" "$SHA" "$G" "$C" "$@"; }
step ref-ka13 20 -- "$L/readout.sh" "$SHA" "$G" ref-ka13 /m4/K-a13-build/soup || exit 1
step KD-s1 240 -- "$L/arm.sh" "$SHA" "$G" KD-s1 20260926 m5-kd-x60-a7fe24 --preflight --kl 1.0 || exit 1
step KD-s2 215 -- "$L/arm.sh" "$SHA" "$G" KD-s2 1 m5-kd-x60-a7fe24 --kl 1.0 || exit 1
step KD-soup 30 -- "$L/soup.sh" "$SHA" "$G" KD-soup KD-s1 KD-s2 || exit 1
step KD-a13 30 -- "$L/interp.sh" "$SHA" "$G" KD-a13 /m5/KD-soup-build/soup 1 3 || exit 1
step KD-a12 30 -- "$L/interp.sh" "$SHA" "$G" KD-a12 /m5/KD-soup-build/soup 1 2 || exit 1
step KD-a23 30 -- "$L/interp.sh" "$SHA" "$G" KD-a23 /m5/KD-soup-build/soup 2 3 || exit 1
echo "chain $C done"
