#!/usr/bin/env bash
# ~27B M6 launch (workstation side), after m6-build.sh finished on node B and the data lock is committed and mirrored.
# Node addresses come from the private nodes.env and are never printed. For each ARM spec:
#   1. a mixture whose seed runs on node A is pushed over the M6 link, moved into
#      /data/dev2/private/27b/m6-data/mixtures-m6-1 there and checked against BUILD.json;
#   2. reference slices on node B GPU0 / GPU1 (A20r-<slice> and M5-L128-<slice>; skipped when present);
#   3. the arm's two seeds (m6-arm.sh; caps from BUILD.json), containers d2-27b-M6-* confirmed;
#   4. node A: m6-relay.sh for a node A seed and m6-mlx-watch.sh for the arm (detached);
#   5. node B: one m6-chain.sh per arm with its own IB DEV slice and aux GPU (detached; logs
#      /data/dev2/runs/27b/m6/logs/chain-<ARM>.log); PIDs and first log lines confirmed.
# Usage: m6-launch.sh MIRROR_SHA ARM:MIXTURE:SLICE:S1,S2:AUX ...
#   S1 / S2 = b0 | b1 | b5 | a2 (node and GPU of each seed); SLICE = ib (IB1 DEV) | ib12 (IB1 + IB2 DEV); AUX = the node B
#   GPU of that arm's readout, slices and formal run (one of its own node B training GPUs)
#   e.g. m6-launch.sh <sha> M6-IB:a20ib1:ib:b5,b1:1 M6-IB2:a20ib12:ib12:b0,a2:0
set -euo pipefail
SHA=${1:?MIRROR_SHA}
shift
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
[ $# -ge 1 ] || { echo "at least one ARM spec" >&2; exit 2; }
NODES=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
A=$(grep '^node-a=' "$NODES" | cut -d= -f2-) B=$(grep '^node-b=' "$NODES" | cut -d= -f2-)
M=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2/v2/27b/m6
O=/data/dev2/private/27b/m6-data MX=$O/mixtures-m6-1 R=/data/dev2/runs/27b/m6
PN1=/data/dev2/hf-cache/datasets--llm-semantic-router--decision-2.0-training-data/snapshots/27b1d2f130292268b43a618584bebab5d4e4a6b5/m4/pn1/dev/pn1.dev.jsonl
PN1_SHA=c3b68ac113fc1f2afb5554530112f472a9c62f7997285bcaf89a6b655bb7f4cf
onb() { ssh -o BatchMode=yes "$B" "$@"; }
ona() { ssh -o BatchMode=yes "$A" "$@"; }
for n in onb ona; do $n "test -f $M/m6-chain.sh" || { echo "mirror $SHA missing ($n)" >&2; exit 2; }; done
build=$(onb "cat $O/BUILD.json")
val() { python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(eval(sys.argv[2], {'d': d}))" "$build" "$1"; }
slice_rows() { case "$1" in ib) val "d['ib1_dev_path']" ;; ib12) val "d['ib12_dev_path']" ;; *) return 2 ;; esac; }
slice_sha() { case "$1" in ib) val "d['ib1_dev_sha256']" ;; ib12) val "d['ib12_dev_sha256']" ;; *) return 2 ;; esac; }
declare -A ARM_MIX ARM_SLICE ARM_S1 ARM_S2 ARM_AUX
ARMS=()
for spec in "$@"; do
  IFS=: read -r arm mix slice seeds aux <<< "$spec"
  [[ "$arm" =~ ^M6-(IB|IBX|IB2)$ ]] && [[ "$mix" =~ ^a20ib(1|1x|12)$ ]] && [[ "$slice" =~ ^(ib|ib12)$ ]] ||
    { echo "bad spec $spec" >&2; exit 2; }
  ARMS+=("$arm") ARM_MIX[$arm]=$mix ARM_SLICE[$arm]=$slice ARM_S1[$arm]=${seeds%,*} ARM_S2[$arm]=${seeds#*,} ARM_AUX[$arm]=$aux
  for loc in "${seeds%,*}" "${seeds#*,}"; do
    case "$loc" in b0 | b1 | b5 | a2) ;; *) echo "bad seed location $loc" >&2; exit 2 ;; esac
  done
  case "$aux" in 0 | 1 | 5) ;; *) echo "bad aux GPU $aux" >&2; exit 2 ;; esac
done
echo "$(date -u +%FT%TZ) launch from mirror $SHA: ${ARMS[*]}"
# 1. node A mixtures
for arm in "${ARMS[@]}"; do
  for loc in "${ARM_S1[$arm]}" "${ARM_S2[$arm]}"; do
    [ "${loc:0:1}" = a ] || continue
    mix=${ARM_MIX[$arm]} want=$(val "d['files_sha256']['${ARM_MIX[$arm]}.train.jsonl']")
    if ! ona "test -f $MX/$mix.train.jsonl"; then
      onb "set -e; K=/data/dev2/tmp/27b-m6-xfer; X=\"ssh -i \$K/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=\$K/known_hosts -o StrictHostKeyChecking=yes -o BatchMode=yes\"; rsync -a --mkpath -e \"\$X\" $MX/$mix.train.jsonl root@\$(cat \$K/peer):relay/mix/$mix.train.jsonl"
      ona "set -e; umask 077; mkdir -p $MX; chmod 700 $O; mv /data/dev2/xfer/27b-m6/relay/mix/$mix.train.jsonl $MX/; rmdir /data/dev2/xfer/27b-m6/relay/mix"
    fi
    ona "test \"\$(sha256sum < $MX/$mix.train.jsonl | cut -c1-64)\" = $want" || { echo "node A $mix is not $want" >&2; exit 3; }
    echo "node A $mix hash equal"
  done
done
# 2. reference slices, two at a time on node B GPU0 / GPU1
for slice in $(printf '%s\n' "${ARM_SLICE[@]}" | sort -u); do
  rows=$(slice_rows "$slice") sha=$(slice_sha "$slice")
  onb "set -e; L=$R/logs; mkdir -p \$L; \
    [ -f $R/slices/A20r-$slice/probs/slices.json ] || bash $M/m6-tail.sh slices $SHA A20r-$slice /data/dev2/runs/27b/M4-A20r-soup/soup/checkpoint 0 $slice=$rows=$sha > \$L/slices-A20r-$slice.log 2>&1 & \
    [ -f $R/slices/M5-L128-$slice/probs/slices.json ] || bash $M/m6-tail.sh slices $SHA M5-L128-$slice /data/dev2/runs/27b/m5/M5-L128/checkpoint 1 $slice=$rows=$sha > \$L/slices-M5-L128-$slice.log 2>&1 & \
    wait; test -f $R/slices/A20r-$slice/probs/slices.json && test -f $R/slices/M5-L128-$slice/probs/slices.json"
  echo "reference slices $slice done"
done
# 3. arm-seeds
declare -A PID
pid() { sed -n 's/.*(pid \([0-9]*\)).*/\1/p' <<< "$1"; }
for arm in "${ARMS[@]}"; do
  mix=${ARM_MIX[$arm]} msha=$(val "d['files_sha256']['${ARM_MIX[$arm]}.train.jsonl']")
  save=$(val "d['checks']['$mix']['save_every']") cap=$(val "d['checks']['$mix']['cap_gpu_h']")
  for s in s1 s2; do
    loc=$([ $s = s1 ] && echo "${ARM_S1[$arm]}" || echo "${ARM_S2[$arm]}")
    run=$([ "${loc:0:1}" = a ] && echo ona || echo onb)
    out=$($run "bash $M/m6-arm.sh ${loc:0:1} ${loc:1} $arm $s $mix $msha $save $cap")
    echo "$out"
    PID[$arm-$s]=$(pid "$out")
    [[ "${PID[$arm-$s]}" =~ ^[0-9]+$ ]] || { echo "$arm-$s did not launch" >&2; exit 3; }
  done
done
# 4. node A watchers
for arm in "${ARMS[@]}"; do
  for s in s1 s2; do
    loc=$([ $s = s1 ] && echo "${ARM_S1[$arm]}" || echo "${ARM_S2[$arm]}")
    [ "${loc:0:1}" = a ] || continue
    ona "mkdir -p $R/logs; setsid nohup bash $M/m6-relay.sh $arm-$s ${PID[$arm-$s]} > $R/logs/relay-$arm-$s.log 2>&1 < /dev/null & echo relay $arm-$s \$!"
  done
  ona "mkdir -p $R/logs; setsid nohup bash $M/m6-mlx-watch.sh $SHA $arm > $R/logs/mlx-watch-$arm.log 2>&1 < /dev/null & echo mlx-watch $arm \$!"
done
# 5. node B chains, one per arm
for arm in "${ARMS[@]}"; do
  slice=${ARM_SLICE[$arm]} rows=$(slice_rows "${ARM_SLICE[$arm]}") sha=$(slice_sha "${ARM_SLICE[$arm]}")
  seeds=()
  for s in s1 s2; do
    loc=$([ $s = s1 ] && echo "${ARM_S1[$arm]}" || echo "${ARM_S2[$arm]}")
    if [ "${loc:0:1}" = a ]; then seeds+=("$arm-$s=a"); else seeds+=("$arm-$s=b:${PID[$arm-$s]}"); fi
  done
  aux=AUX_${arm//-/_}
  onb "setsid nohup env PN1_ROWS=$PN1 PN1_SHA=$PN1_SHA IB_ROWS=$rows IB_SHA=$sha IB_SLICE=$slice REF_IB=A20r-$slice \
    IN_DIST='w2c isarc hover gsm2' $aux=${ARM_AUX[$arm]} bash $M/m6-chain.sh $SHA ${seeds[*]} \
    > $R/logs/chain-$arm.log 2>&1 < /dev/null & echo chain $arm \$!"
done
sleep 90
onb "docker ps --format '{{.Names}} {{.Status}}' | grep d2-27b-M6 || true; for f in $R/logs/chain-*.log; do head -n 1 \$f; done"
ona "docker ps --format '{{.Names}} {{.Status}}' | grep d2-27b-M6 || true; for f in $R/logs/relay-*.log $R/logs/mlx-watch-*.log; do head -n 1 \$f; done"
for k in "${!PID[@]}"; do echo "driver $k pid ${PID[$k]}"; done
echo "launch complete: $(date -u +%FT%TZ)"
