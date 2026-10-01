#!/usr/bin/env bash
# ~27B M6 stage-1 launch (workstation side; the state file's runbook steps 3-7), after m6-build.sh finished on node B
# and the data lock is committed. Node addresses come from the private nodes.env and are never printed.
#   3. a20ib1x.train.jsonl -> node A over the M6 link, moved into /data/dev2/private/27b/m6-data/mixtures-m6-1 there and
#      checked against BUILD.json
#   4. IB DEV reference slices on node B: A20r-ib1 (GPU0) and M5-L128-ib1 (GPU1), in parallel, foreground
#   5. the four arm-seeds (m6-arm.sh): M6-IB-s1 node B GPU5, M6-IB-s2 node B GPU1, M6-IBX-s1 node B GPU0,
#      M6-IBX-s2 node A GPU2; containers d2-27b-M6-* confirmed
#   6. node A: m6-relay.sh M6-IBX-s2 <pid>, m6-mlx-watch.sh for M6-IB and M6-IBX (detached)
#   7. node B: m6-chain.sh (detached; log /data/dev2/runs/27b/m6/logs/chain-stage1.log); PID and first log line confirmed
# Usage: m6-stage1.sh MIRROR_SHA   (the mirror must exist on both nodes)
set -euo pipefail
SHA=${1:?MIRROR_SHA}
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
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
IB_SHA=$(val "d['ib1_dev_sha256']") IB_ROWS=$(val "d['ib1_dev_path']")
SHA_IB=$(val "d['files_sha256']['a20ib1.train.jsonl']") SHA_IBX=$(val "d['files_sha256']['a20ib1x.train.jsonl']")
SAVE_IB=$(val "d['checks']['a20ib1']['save_every']") SAVE_IBX=$(val "d['checks']['a20ib1x']['save_every']")
CAP=16.0
echo "$(date -u +%FT%TZ) stage 1: a20ib1 $SHA_IB (save $SAVE_IB), a20ib1x $SHA_IBX (save $SAVE_IBX), IB DEV $IB_SHA"
# 3. the IBX mixture to node A
if ! ona "test -f $MX/a20ib1x.train.jsonl"; then
  onb "set -e; K=/data/dev2/tmp/27b-m6-xfer; X=\"ssh -i \$K/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=\$K/known_hosts -o StrictHostKeyChecking=yes -o BatchMode=yes\"; rsync -a --mkpath -e \"\$X\" $MX/a20ib1x.train.jsonl root@\$(cat \$K/peer):relay/mix/a20ib1x.train.jsonl"
  ona "set -e; umask 077; mkdir -p $MX; chmod 700 $O; mv /data/dev2/xfer/27b-m6/relay/mix/a20ib1x.train.jsonl $MX/; rmdir /data/dev2/xfer/27b-m6/relay/mix"
fi
ona "test \"\$(sha256sum < $MX/a20ib1x.train.jsonl | cut -c1-64)\" = $SHA_IBX" || { echo "node A a20ib1x is not $SHA_IBX" >&2; exit 3; }
echo "node A a20ib1x hash equal"
# 4. IB DEV reference slices (foreground, in parallel)
onb "set -e; L=$R/logs; mkdir -p \$L; \
  [ -f $R/slices/A20r-ib1/probs/slices.json ] || bash $M/m6-tail.sh slices $SHA A20r-ib1 /data/dev2/runs/27b/M4-A20r-soup/soup/checkpoint 0 ib=$IB_ROWS=$IB_SHA > \$L/slices-A20r-ib1.log 2>&1 & \
  [ -f $R/slices/M5-L128-ib1/probs/slices.json ] || bash $M/m6-tail.sh slices $SHA M5-L128-ib1 /data/dev2/runs/27b/m5/M5-L128/checkpoint 1 ib=$IB_ROWS=$IB_SHA > \$L/slices-M5-L128-ib1.log 2>&1 & \
  wait; test -f $R/slices/A20r-ib1/probs/slices.json && test -f $R/slices/M5-L128-ib1/probs/slices.json"
echo "IB DEV reference slices done"
# 5. arm-seeds
pid() { sed -n 's/.*(pid \([0-9]*\)).*/\1/p' <<< "$1"; }
out=$(onb "bash $M/m6-arm.sh b 5 M6-IB s1 a20ib1 $SHA_IB $SAVE_IB $CAP"); echo "$out"; P_IB1=$(pid "$out")
out=$(onb "bash $M/m6-arm.sh b 1 M6-IB s2 a20ib1 $SHA_IB $SAVE_IB $CAP"); echo "$out"; P_IB2=$(pid "$out")
out=$(onb "bash $M/m6-arm.sh b 0 M6-IBX s1 a20ib1x $SHA_IBX $SAVE_IBX $CAP"); echo "$out"; P_IBX1=$(pid "$out")
out=$(ona "bash $M/m6-arm.sh a 2 M6-IBX s2 a20ib1x $SHA_IBX $SAVE_IBX $CAP"); echo "$out"; P_IBX2=$(pid "$out")
for p in "$P_IB1" "$P_IB2" "$P_IBX1" "$P_IBX2"; do [[ "$p" =~ ^[0-9]+$ ]] || { echo "an arm-seed did not launch" >&2; exit 3; }; done
# 6. node A watchers
ona "set -e; L=/data/dev2/runs/27b/m6/logs; mkdir -p \$L; \
  setsid nohup bash $M/m6-relay.sh M6-IBX-s2 $P_IBX2 > \$L/relay-M6-IBX-s2.log 2>&1 < /dev/null & echo relay \$!; \
  setsid nohup bash $M/m6-mlx-watch.sh $SHA M6-IB > \$L/mlx-watch-M6-IB.log 2>&1 < /dev/null & echo mlx-IB \$!; \
  setsid nohup bash $M/m6-mlx-watch.sh $SHA M6-IBX > \$L/mlx-watch-M6-IBX.log 2>&1 < /dev/null & echo mlx-IBX \$!"
# 7. node B chain
onb "set -e; setsid nohup env PN1_ROWS=$PN1 PN1_SHA=$PN1_SHA IB_ROWS=$IB_ROWS IB_SHA=$IB_SHA IN_DIST='w2c isarc' \
  AUX_M6_IB=1 AUX_M6_IBX=0 bash $M/m6-chain.sh $SHA M6-IB-s1=b:$P_IB1 M6-IB-s2=b:$P_IB2 M6-IBX-s1=b:$P_IBX1 M6-IBX-s2=a \
  > $R/logs/chain-stage1.log 2>&1 < /dev/null & echo chain \$!"
sleep 90
onb "docker ps --format '{{.Names}} {{.Status}}' | grep d2-27b-M6 || true; head -n 1 $R/logs/chain-stage1.log; for s in M6-IB-s1 M6-IB-s2 M6-IBX-s1; do tail -n 1 /data/dev2/runs/27b/\$s/driver.log; done"
ona "docker ps --format '{{.Names}} {{.Status}}' | grep d2-27b-M6 || true; tail -n 1 /data/dev2/runs/27b/M6-IBX-s2/driver.log; head -n 1 /data/dev2/runs/27b/m6/logs/relay-M6-IBX-s2.log"
echo "stage 1 launched: drivers b5 $P_IB1, b1 $P_IB2, b0 $P_IBX1, a2 $P_IBX2"
