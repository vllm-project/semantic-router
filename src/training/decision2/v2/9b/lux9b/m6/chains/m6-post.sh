#!/usr/bin/env bash
# usage: m6-post.sh SHA GPU NAME   (started on node A by m6/launch.sh after its lock record is pushed)
# 9B Milestone 6 post-lock stage for one finalist NAME: the committed formal-stage chain m6-formal.sh (formal
# post-key run, hs1-dev diagnostics, shipped-calibration rule, T = 1 derivation when it applies), then the
# HT-DEV v2 diagnostic m6-htdev2.sh. Stops at the first failed chain.
set -uo pipefail
SHA=${1:?mirror sha}; G=${2:?gpu}; NAME=${3:?finalist}
L=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2/v2/9b/lux9b/m6
[ -f "$L/chains/m6-formal.sh" ] || { echo "mirror $L missing" >&2; exit 2; }
bash "$L/chains/m6-formal.sh" "$SHA" "$G" "$NAME" || exit 1
bash "$L/chains/m6-htdev2.sh" "$SHA" "$G" "$NAME" || exit 1
echo "chain m6-post-$NAME done"
