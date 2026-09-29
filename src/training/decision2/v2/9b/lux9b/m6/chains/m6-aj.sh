#!/usr/bin/env bash
# usage: m6-aj.sh SHA   (started on node A by m6/launch.sh after upload_chain.sh verified it)
# 9B Milestone 6, node A GPU6 + GPU7 (prereg records/lux9b-m6-prereg-2026-09-30.md): the
# AutoJev-27B wave on the human-rated rows without production targets, then the KA teacher build
# (CPU). Stops at the first failed step. Training is launched separately after the data freeze.
set -uo pipefail
SHA=${1:?mirror sha}
L=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2/v2/9b/lux9b/m6
[ -f "$L/aj_wave.sh" ] || { echo "mirror $L missing" >&2; exit 2; }
. "$L/lib.sh"
budget_ok 1.5 || exit 1
bash "$L/aj_wave.sh" "$SHA" m6-split || exit 1
bash "$L/data.sh" "$SHA" ka m6-ka-x60-ajS.json m6-ka \
  --wave-targets /m6/aj-wave/aj-9b-m6.targets.jsonl --wave-report /m6/aj-wave/report.json || exit 1
echo "chain m6-aj done"
