#!/usr/bin/env bash
# ~27B M6 node D staging (workstation side), once IX1 has released node D. Copies from node B over node B's transfer
# key (/root/.ssh/d2_temp_cd, authorised on nodes C-F) the inputs run_lora_arm.sh mounts, at the same paths: the pinned
# base Qwen3.8-27B (52 GB), the rights-clean SELECT / CAL rows, the dev and CSS-pilot prompt files, the frozen T0
# training cache and the named M6 mixtures; then checks on node D: the base tree c457c994..., T0 tree 1933eb36...,
# every data file's SHA-256 equal to node B's, every mixture equal to BUILD.json. Writes node D's address (never printed)
# to node B's link directory as peer-d for the chain's pull-d. The code mirror is mirror_to_node.sh's job (node-d).
# Usage: m6-stage-d.sh MIRROR_SHA MIXTURE...   (e.g. a20ib1x; the mirror must already be on node D)
# Environment: M6_BUILD (build record in m6-data, default BUILD.json; BUILD-pn.json stages amendment 3's a20ib12pn from
#   its mixtures_dir, mixtures-m6pn-1); M6_DATA=m7-data stages M7 mixtures (m7-data/mixtures-m7-1, M7's BUILD.json);
#   M6_STAGE_NODE=e stages node E the same way (27B M8; peer file peer-e) instead of node D.
set -euo pipefail
SHA=${1:?MIRROR_SHA}
shift
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
STAGE_NODE=${M6_STAGE_NODE:-d}
[[ "$STAGE_NODE" =~ ^[de]$ ]] || { echo "M6_STAGE_NODE is d or e" >&2; exit 2; }
NODES=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
B=$(grep '^node-b=' "$NODES" | cut -d= -f2-) D=$(grep "^node-$STAGE_NODE=" "$NODES" | cut -d= -f2-)
[ -n "$B" ] && [ -n "$D" ] || { echo "node-b / node-$STAGE_NODE missing in $NODES" >&2; exit 2; }
onb() { ssh -o BatchMode=yes "$B" "$@"; }
ond() { ssh -o BatchMode=yes "$D" "$@"; }
SRC=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2
BASE=/data/decision20-20260926/models/Qwen3.8-27B BASE_TREE=c457c9941d0f28581289cd67a3ef2af10bb18cfb400a4b00db9ccffa64854ab6
T0=/data/dev2/runs/27b/m4-train-cache-T0 T0_TREE=1933eb36d746a3bf3b716967ce97f977620eab0f7a390f751e79bfd4f8b2e21f
FILES="data/rights_clean_goemotions_v2/select.jsonl data/rights_clean_goemotions_v2/cal.jsonl runs/dev.prompts.jsonl runs/css-transfer-v1/css-pilot.prompts.jsonl"
BUILD_FILE=${M6_BUILD:-BUILD.json}
[[ "$BUILD_FILE" =~ ^BUILD(-[a-z0-9]+)?\.json$ ]] || { echo "bad M6_BUILD $BUILD_FILE" >&2; exit 2; }
DATA=${M6_DATA:-m6-data}
[[ "$DATA" =~ ^m[67]-data$ ]] || { echo "bad M6_DATA $DATA" >&2; exit 2; }
build=$(onb "cat /data/dev2/private/27b/$DATA/$BUILD_FILE")
MX=/data/dev2/private/27b/$DATA/$(python3 -c "import json,sys; print(json.loads(sys.argv[1]).get('mixtures_dir', sys.argv[2]))" "$build" "mixtures-${DATA%-data}-1")
ond "test -f $SRC/v2/27b/triton_cache.py" || { echo "mirror $SHA is not on node ${STAGE_NODE^^}" >&2; exit 2; }
for m in "$@"; do [[ "$m" =~ ^a20ib[0-9a-z]+$ ]] || { echo "bad mixture $m" >&2; exit 2; }; done
onb "umask 077; test -d /data/dev2/tmp/27b-m6-xfer && printf '%s\n' '${D#*@}' > /data/dev2/tmp/27b-m6-xfer/peer-$STAGE_NODE"
ond "set -e; mkdir -p /data/decision20-20260926/models /data/decision20-20260926/data/rights_clean_goemotions_v2 \
  /data/decision20-20260926/runs/css-transfer-v1 /data/dev2/runs/27b /data/dev2/tmp /data/dev2/xfer/27b-m6/relay \
  /data/dev2/runs/27b/m6/logs; umask 077; mkdir -p $MX; chmod 700 /data/dev2/private/27b/$DATA"
echo "$(date -u +%FT%TZ) copying base, data files, T0 and mixtures ($*) from node B to node ${STAGE_NODE^^}"
onb "set -e; E='ssh -i /root/.ssh/d2_temp_cd -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes'; \
  P=root@\$(cat /data/dev2/tmp/27b-m6-xfer/peer-$STAGE_NODE); \
  rsync -a -e \"\$E\" $BASE/ \$P:$BASE/; \
  for f in $FILES; do rsync -a -e \"\$E\" /data/decision20-20260926/\$f \$P:/data/decision20-20260926/\$f; done; \
  rsync -a -e \"\$E\" $T0/ \$P:$T0/; \
  for m in $*; do rsync -a -e \"\$E\" $MX/\$m.train.jsonl \$P:$MX/; done"
echo "$(date -u +%FT%TZ) copy done; verifying on node ${STAGE_NODE^^}"
tree=$(ond "cd $SRC && PYTHONPATH=$SRC python3 -m v2.27b.triton_cache digest $BASE")
[ "$tree" = "$BASE_TREE" ] || { echo "node ${STAGE_NODE^^} base tree $tree is not $BASE_TREE" >&2; exit 3; }
tree=$(ond "cd $SRC && PYTHONPATH=$SRC python3 -m v2.27b.triton_cache digest $T0")
[ "$tree" = "$T0_TREE" ] || { echo "node ${STAGE_NODE^^} T0 tree $tree is not $T0_TREE" >&2; exit 3; }
for f in $FILES; do
  b=$(onb "sha256sum < /data/decision20-20260926/$f | cut -c1-64") d=$(ond "sha256sum < /data/decision20-20260926/$f | cut -c1-64")
  [ "$b" = "$d" ] || { echo "node ${STAGE_NODE^^} $f differs from node B" >&2; exit 3; }
done
for m in "$@"; do
  want=$(python3 -c "
import json, sys
b, m = json.loads(sys.argv[1]), sys.argv[2]
print(b['files_sha256'][m + '.train.jsonl'] if 'files_sha256' in b else b['mixtures'][m]['sha256'])" "$build" "$m")
  got=$(ond "sha256sum < $MX/$m.train.jsonl | cut -c1-64")
  [ "$got" = "$want" ] || { echo "node ${STAGE_NODE^^} $m is not $want" >&2; exit 3; }
done
ond "docker image inspect sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1 --format '{{.Id}}'" > /dev/null
echo "node ${STAGE_NODE^^} staged: base tree, T0 tree, data files and mixtures ($*) equal to node B; training image present"
