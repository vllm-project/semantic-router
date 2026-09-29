#!/usr/bin/env bash
# usage: m6-gpu7.sh SHA   (started on node A by m6/launch.sh after upload_chain.sh verified it)
# 9B Milestone 6 chain, node A GPU7 (prereg records/lux9b-m6-prereg-2026-09-30.md): the incumbent's
# reference readout, the matched control K (x60 = M4's K recipe byte for byte; seeds 3 and 4,
# preflights with the first) with its alpha 1/3 early-stop point, the five-seed K soup (M4 K-s1..s3 +
# K-s4, K-s5) and its line (1/3, 1/2, 2/3), the two-seed control soup and its line (1/3, 1/2;
# report-only contrasts), then KH's second seed, soup and line only if KH's early rule (written on
# GPU6) continues. Every training step first checks the 24 GPU-h budget. Stops at the first
# failed step.
set -uo pipefail
SHA=${1:?mirror sha}
L=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2/v2/9b/lux9b/m6
[ -f "$L/chain-step.sh" ] || { echo "mirror $L missing" >&2; exit 2; }
. "$L/lib.sh"
G=7
C=m6-gpu7
step() { bash "$L/chain-step.sh" "$SHA" "$G" "$C" "$@"; }
line() {  # line NAME SOUP_NAME NUM/DEN...
  local name=$1 soup=$2 f; shift 2
  for f in "$@"; do
    step "$name-a${f%/*}${f#*/}" 20 -- bash "$L/interp.sh" "$SHA" "$G" "$name-a${f%/*}${f#*/}" \
      "/m6/$soup-build/soup" "${f%/*}" "${f#*/}" || return 1
  done
}
step ref-ka13 20 -- bash "$L/readout.sh" "$SHA" "$G" ref-ka13 /m4/K-a13-build/soup || exit 1
budget_ok 7.1 || exit 1
step K-s4 200 -- bash "$L/arm.sh" "$SHA" "$G" K-s4 3 m4:m4-k-xl-r2-60m --preflight --kl 1.0 || exit 1
step K-s4-e13 20 -- bash "$L/interp.sh" "$SHA" "$G" K-s4-e13 "/m6/K-s4/run/$(best_of "$M6/K-s4/run/BEST.json")" 1 3 || exit 1
budget_ok 6.6 || exit 1
step K-s5 180 -- bash "$L/arm.sh" "$SHA" "$G" K-s5 4 m4:m4-k-xl-r2-60m --kl 1.0 || exit 1
step K5-soup 40 -- bash "$L/soup.sh" "$SHA" "$G" K5-soup /m4/K-s1/run/checkpoint-0001624 \
  /m4/K-s2/run/checkpoint-0001420 /m4/K-s3/run/checkpoint-0001424 K-s4 K-s5 || exit 1
line K5 K5-soup 1/3 1/2 2/3 || exit 1
step K2-soup 30 -- bash "$L/soup.sh" "$SHA" "$G" K2-soup K-s4 K-s5 || exit 1
line K2 K2-soup 1/3 1/2 || exit 1
waited=0
while [ ! -f "$M6/rules/early-KH-s4.json" ]; do
  [ "$waited" -lt 480 ] || { echo "no early rule for KH-s4 after 8 h; KH second seed not started"; exit 0; }
  sleep 60; waited=$((waited + 1))
done
if early_continue KH-s4; then
  budget_ok 6.6 || exit 1
  step KH-s5 180 -- bash "$L/arm.sh" "$SHA" "$G" KH-s5 4 m6-kh --kl 1.0 --teacher-partial || exit 1
  step KH-soup 30 -- bash "$L/soup.sh" "$SHA" "$G" KH-soup KH-s4 KH-s5 || exit 1
  line KH KH-soup 1/3 1/2 2/3 || exit 1
else
  echo "KH stopped by its early rule; no second seed or line"
fi
echo "chain $C done"
