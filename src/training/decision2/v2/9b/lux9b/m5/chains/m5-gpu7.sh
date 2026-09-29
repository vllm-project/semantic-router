#!/usr/bin/env bash
# usage: m5-gpu7.sh SHA   (started on node A by m5/launch.sh after upload_chain.sh verified it)
# 9B Milestone 5 chain, node A GPU7 (prereg records/lux9b-m5-prereg-2026-09-29.md): the Lux 1.0
# reference readout, arm KG (dose rows gold-only via --teacher-partial; two seeds, preflights with
# the first), its soup and its interpolation line toward Lux 1.0 (alpha 1/3, 1/2, 2/3). Every step
# first waits while the eval track's C1 lease entry on GPU7 is active. Stops at the first failed step.
set -uo pipefail
SHA=${1:?mirror sha}
L=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2/v2/9b/lux9b/m5
[ -x "$L/chain-step.sh" ] || { echo "mirror $L missing" >&2; exit 2; }
G=7
C=m5-gpu7
step() { "$L/chain-step.sh" "$SHA" "$G" "$C" "$@"; }
step ref-lux 20 -- "$L/readout.sh" "$SHA" "$G" ref-lux /m3/pf-D-s1-zero/run/checkpoint-0000000 || exit 1
step KG-s1 240 -- "$L/arm.sh" "$SHA" "$G" KG-s1 20260926 m5-kg-x60-a7fe24g --preflight --kl 1.0 --teacher-partial || exit 1
step KG-s2 215 -- "$L/arm.sh" "$SHA" "$G" KG-s2 1 m5-kg-x60-a7fe24g --kl 1.0 --teacher-partial || exit 1
step KG-soup 30 -- "$L/soup.sh" "$SHA" "$G" KG-soup KG-s1 KG-s2 || exit 1
step KG-a13 30 -- "$L/interp.sh" "$SHA" "$G" KG-a13 /m5/KG-soup-build/soup 1 3 || exit 1
step KG-a12 30 -- "$L/interp.sh" "$SHA" "$G" KG-a12 /m5/KG-soup-build/soup 1 2 || exit 1
step KG-a23 30 -- "$L/interp.sh" "$SHA" "$G" KG-a23 /m5/KG-soup-build/soup 2 3 || exit 1
echo "chain $C done"
