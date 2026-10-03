#!/usr/bin/env bash
# ~27B M6 private Index run of one frozen formal finalist (workstation side; prereg amendments 4-5, the Index path of
# COORDINATION 2026-10-02 02:05 / 03:40: Index runs for Index-path candidates and the chosen successor, item 1' needs a
# significantly positive paired Index delta vs A20r; runs use the eval allowance on node C GPU1-7 and node D GPU4-7).
# IX1's harness (v2/eval/ix1: image host2, kit 87d4650b, panel-8, the 86-request parity gate, dual scoring) set up as
# IX1 ran the M5-L128 diagnostic: the frozen soup checkpoint restaged into DEV2.0-27B 4e89288d with the forward-budget
# runtime (fix2 package, e876fbe; T = 1), A20r's frozen autotune cache, the same panel. The mirror carries the ARM's
# DIAGNOSTIC entry in v2/eval/ix1/launch.sh and is on node D (and node C when it takes shards; mirror_to_node.sh ...
# node-d / node-c). Index values stay in the nodes' /data/dev2/private/eval/index021/ix1/ and the local private
# folder; this script prints none.
# Usage: m6-index.sh MIRROR_SHA ARM STAGE
#   ARM      an M6 arm, a cross-arm soup M6-IBxIB2-mNN (seeds of M6-IB and M6-IB2; NN = M6-IB2's weight in percent) or
#            an M7 arm soup (M7-IB124ML, M7-IB14ML), an M8 arm soup (M8-IB14, M8-IB124) or a preregistered M7 / M8 / M9
#            cross-arm soup (X7-IBxIB2xIB14ML, X7-4ARM, X8-IBxIB2-8, X8-ML, X9-LRH2, X9-LRH2xM50, X9-LRH, X9-LRHxM50,
#            X9-ML0, X9-IBxIB2-10); its soup is node B m6/ARM/checkpoint
#   plan     prints which shards run on which node and GPU for M6_INDEX_GPUS (no node is touched)
#   stage    node B m6/ARM/checkpoint -> node D /data/dev2/models/ix1/m6/ARM-ckpt over node B's transfer key (SHA-256
#            lists equal), then v2.eval.ix1.restage -> /data/dev2/models/ix1/m6/ARM-re876fbe with the model SHA-256 of
#            m6/ARM/package/PACKAGE.json; checks the loaded count (base 25,629,863,936 with the head + 7,295,488 per
#            unit of the soup's LoRA rank: 27,497,508,864 at rank 256) and the identity
#   stage-c  after stage: the same on node C (the fix2 template relayed node D -> node C through node B once, its
#            SHA-256 list equal to node D's), and node C's restaged package must equal node D's file for file
#   control  once for M6 (any ARM): A20r's own package through the same runtime (DEV2.0-27B-budget) read by its entry
#            point over the 86 compatibility requests on node D GPU4, vs IX1's A20r kit results -> must pass
#   parity   launch.sh parity on node D GPU M6_PARITY_GPU (default 4; detached; waits for it, ~0.1 GPU-h);
#            parity/ARM/parity.json must pass
#   stage-e  after stage: node D's restaged package and A20r's frozen cache (with its .sha256) relayed node D -> node E
#            through node B (node E holds the same base snapshot, kit and panel), SHA-256 lists equal to node D's
#   stage-b  the same pulled by node B from node D, plus IX1's image (ID checked), kit and panel-8; node B's base
#            snapshot must equal node D's
#   run      m6-index-run.sh detached on each node with shards (log ix1/logs/m6-index-ARM-NODE.log): the shards of
#            M6_INDEX_SHARDS (default all 8, e.g. "2 3 4 5 6 7" to re-split shards not yet run) on the entries of
#            M6_INDEX_GPUS (default "d4 d5 d6 d7"; dN = node D GPU N in 0-7 (GPU0-3 are M6's own leases, used once its
#            seeds ended, under an eval-ix1 owner naming the arm), bN = node B GPU0 / 1 / 5 (27B's idle training leases,
#            set aside as owner.m6-set-aside-<UTC>), cN = node C GPU N in 1-7, eN = node E GPU N in 0-3 or 6-7 (never
#            GPU4-5), a bare N = node D), the i-th listed shard on entry i mod n; node B / C / E need stage-b / stage-c /
#            stage-e, each gets node D's parity record (SHA-256 equal); M6_INDEX_STAGGER (s) passes through, and so
#            does M6_INDEX_AFTER (a shard of an earlier launch every listed GPU first waits for); node B also takes
#            released owners of other tracks (set aside the same way)
#   status   per node and shard: records written, ended, exit code; GPU-h so far
#   collect  after node B's / C's / E's shards ended 0: their result files (no Triton cache, no home) and launcher
#            records -> node D's run directory, SHA-256 lists equal; never overwrites a node D shard
#   score    after all 8 shards ended 0 on node D: score.sh (merge, port + kit scoring, compare incl. the frontier peer),
#            family_delta vs A20r's IX1 run (merged-budget) and vs M5-L128's, and the paired bootstrap vs A20r
#            (paired_boot: 2,000 replicates, seed 20261002; the Index gate = its 95% lower bound > 0) over
#            merged-budget-r: the same records as merged-budget, re-merged by v2.eval.ix1.merge from the same shard and
#            rerun files, with the ix1 run receipt the release gate's IF1 binds; with M6_INDEX_BASE=<arm> (the current
#            release's arm, e.g. M6-IB) also family_delta and the same paired bootstrap vs that arm's merged run
#            (paired-boot-vs-<arm>.json; the successor gate); private outputs copied to
#            ~/code/decision2-program/private/m6/ARM/ (mode 700)
#   audit-arm  the ARM's own training files (node B mixtures, SHA-256 checked; the seeds' files identical; both arms'
#            files for a cross-arm soup) audited the same way on node D -> ix1/runs/m6-audit-ARM (the release gate's
#            IF3 evidence; prints the planted control only)
#   audit    once for M6 (any ARM; CPU): v2.eval.ix1.contamination of a20ib12pn (it contains every arm's rows) and
#            a20ib1x against the panel -> ix1/runs/m6-audit (backs the card's "audited at row level" footnote)
#   hold     M6_INDEX_GPUS (dN / bN / eN) that are idle and free (no owner, released, or an eval-ix1 run whose 8 shards
#            ended) get a 27B reserved-idle owner naming ARM, so other tracks' launchers refuse them until run / parity
#            set it aside; unhold releases the holds of ARM
#   release  owner files that launch.sh wrote for ARM on node D GPU0-7, node B GPU0 / 1 / 5, node C GPU1-7 and node E
#            GPU0-3 / 6-7 -> status released, or the set-aside 27B owner back (no ARM container running)
set -euo pipefail
SHA=${1:?MIRROR_SHA} ARM=${2:?ARM} STAGE=${3:?STAGE}
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
ARM_RE='^(M6-(IB|IBX|IB2|IB2PN|IBxIB2-m[0-9]{2})|M7-(IB124ML|IB14ML)|M8-(IB14|IB124)|X7-(IBxIB2xIB14ML|4ARM)|X8-(IBxIB2-8|ML)|X9-(LRH2|LRH2xM50|LRH|LRHxM50|ML0|IBxIB2-10))$'
[[ "$ARM" =~ $ARM_RE ]] || { echo "bad ARM $ARM" >&2; exit 2; }
placement() {  # one line per node with shards: "NODE SHARDS GPU..." (i-th shard of M6_INDEX_SHARDS on entry i mod n)
  local entries=() norm=() todo=() e i k node g
  local -A seen=() shards=() gpus=() once=()
  read -r -a entries <<< "${M6_INDEX_GPUS:-d4 d5 d6 d7}"
  read -r -a todo <<< "${M6_INDEX_SHARDS:-0 1 2 3 4 5 6 7}"
  (( ${#todo[@]} >= 1 )) || { echo "M6_INDEX_SHARDS: at least one shard" >&2; return 2; }
  for k in "${todo[@]}"; do
    [[ "$k" =~ ^[0-7]$ ]] || { echo "M6_INDEX_SHARDS: shard indices 0-7, not '$k'" >&2; return 2; }
    [ -z "${once[$k]:-}" ] || { echo "M6_INDEX_SHARDS lists $k twice" >&2; return 2; }
    once[$k]=1
  done
  (( ${#entries[@]} >= 1 && ${#entries[@]} <= 8 )) || { echo "M6_INDEX_GPUS: 1-8 entries" >&2; return 2; }
  for e in "${entries[@]}"; do
    [[ "$e" =~ ^[4-7]$ ]] && e=d$e
    [[ "$e" =~ ^(d[0-7]|b[015]|c[1-7]|e[0-367])$ ]] ||
      { echo "M6_INDEX_GPUS: node D GPU0-7 (d0-d7), node B GPU0 / 1 / 5, node C GPU1-7 or node E GPU0-3 / 6-7, not '$e'" >&2
        return 2; }
    [ -z "${seen[$e]:-}" ] || { echo "M6_INDEX_GPUS lists $e twice" >&2; return 2; }
    seen[$e]=1
    norm+=("$e")
  done
  for i in "${!todo[@]}"; do
    k=${todo[$i]} e=${norm[$((i % ${#norm[@]}))]} node=${e:0:1} g=${e:1}
    shards[$node]+="${shards[$node]:+,}$k"
    [[ " ${gpus[$node]:-} " == *" $g "* ]] || gpus[$node]+="${gpus[$node]:+ }$g"
  done
  for node in d b c e; do
    [ -z "${shards[$node]:-}" ] || echo "$node ${shards[$node]} ${gpus[$node]}"
  done
}
if [ "$STAGE" = plan ]; then
  plan=$(placement) || exit 2
  while read -r node shards gpus; do
    echo "node $node: shards $shards on GPU $gpus"
    read -r -a gl <<< "$gpus"
    M6_INDEX_DRY=1 bash "$(dirname "$0")/m6-index-run.sh" "$SHA" "$ARM" "$node" "$shards" "${gl[@]}" | sed 's/^/  /'
  done <<< "$plan"
  exit 0
fi
NODES=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
B=$(grep '^node-b=' "$NODES" | cut -d= -f2-) C=$(grep '^node-c=' "$NODES" | cut -d= -f2-)
D=$(grep '^node-d=' "$NODES" | cut -d= -f2-) E=$(grep '^node-e=' "$NODES" | cut -d= -f2- || true)
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=4)
onb() { "${SSH[@]}" "$B" "$@"; }
onc() { "${SSH[@]}" "$C" "$@"; }
ond() { "${SSH[@]}" "$D" "$@"; }
one() { "${SSH[@]}" "$E" "$@"; }
XFER="ssh -i /root/.ssh/d2_temp_cd -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes"
M=/data/dev2/src/$SHA-src_training_decision2
S=$M/src/training/decision2
R=/data/dev2/private/eval/index021/ix1
R6=/data/dev2/runs/27b/m6
MD=/data/dev2/models/ix1/m6
FIX2=/data/dev2/models/ix1/fix2/DEV2.0-27B-4e89288d-re876fbe
PKG=$MD/$ARM-re876fbe CK=$MD/$ARM-ckpt
CACHE=$R/parity/DEV2.0-27B/cache-frozen
KIT=/data/dev2/private/eval/index021/kit-87d4650b IMAGE=decision20-train-fast:host2 IMAGE_ID=sha256:f83b1d10
BASE=/data/dev2/hf-cache/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
LOADED_BASE=25629863936 LOADED_PER_RANK=7295488
LOCAL=${M6_INDEX_LOCAL:-$HOME/code/decision2-program/private/m6}/$ARM
TAG=ix1-$(tr 'A-Z.' 'a-z_' <<< "$ARM")-
has_mirror() {  # NODE: the mirror with ARM's DIAGNOSTIC entry is on that node
  "on$1" "test -f $S/v2/eval/ix1/launch.sh" || { echo "mirror $SHA is not on node ${1^^}" >&2; return 2; }
  "on$1" "grep -q '^  \[$ARM\]=\"DEV2.0-27B [0-9a-f]* $PKG\"' $S/v2/eval/ix1/launch.sh" ||
    { echo "mirror $SHA has no DIAGNOSTIC entry $ARM -> $PKG" >&2; return 2; }
}
sums() { echo "cd $1 && find . -type f | sort | xargs -P 8 -n 4 sha256sum | sort -k2"; }
model_sha() {  # the frozen package's identity; M6_INDEX_FROM_SOUP=1: the soup's, for a candidate without a formal run
  local model soup
  onb "test -f $R6/$ARM/checkpoint/soup_manifest.json" || { echo "no frozen soup for $ARM on node B" >&2; return 3; }
  soup=$(onb "python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[\"output\"][\"model_sha256\"])' $R6/$ARM/checkpoint/soup_manifest.json")
  [[ "$soup" =~ ^[0-9a-f]{64}$ ]] || { echo "bad model SHA-256 in soup_manifest.json" >&2; return 3; }
  if onb "test -f $R6/$ARM/package/PACKAGE.json"; then
    model=$(onb "python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[\"model_sha256\"])' $R6/$ARM/package/PACKAGE.json")
    [ "$model" = "$soup" ] || { echo "PACKAGE.json and the soup name different identities" >&2; return 3; }
  elif [ "${M6_INDEX_FROM_SOUP:-0}" != 1 ]; then
    echo "no frozen package for $ARM on node B (M6_INDEX_FROM_SOUP=1 stages its soup at T = 1; amendment 7)" >&2
    return 3
  fi
  echo "$soup"
}
loaded() {  # the restaged package's loaded count from the soup's LoRA rank
  local rank
  rank=$(onb "python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[\"lora\"][\"rank\"])' $R6/$ARM/checkpoint/soup_manifest.json")
  [[ "$rank" =~ ^[1-9][0-9]{1,3}$ ]] || { echo "bad LoRA rank in soup_manifest.json" >&2; return 3; }
  echo $((LOADED_BASE + LOADED_PER_RANK * rank))
}
copy_checkpoint() {  # NODE: node B soup checkpoint -> that node's $CK over node B's transfer key, SHA-256 lists equal
  local addr b t
  [ "$1" = c ] && addr=${C#*@} || addr=${D#*@}
  "on$1" "umask 077; mkdir -p $MD"
  echo "$(date -u +%FT%TZ) $ARM: copying the soup checkpoint node B -> node ${1^^}"
  onb "rsync -a -e '$XFER' $R6/$ARM/checkpoint/ root@$addr:$CK/"
  b=$(onb "$(sums "$R6/$ARM/checkpoint")") t=$("on$1" "$(sums "$CK")")
  [ -n "$b" ] && [ "$b" = "$t" ] || { echo "node ${1^^} checkpoint differs from node B's" >&2; return 3; }
  echo "checkpoint: $(wc -l <<< "$b") files, SHA-256 lists equal"
}
restage() {  # NODE MODEL_SHA256 LOADED
  "on$1" "cd $S && PYTHONPATH=$S python3 -m v2.eval.ix1.restage --package $FIX2 --out $PKG --checkpoint $CK --model-sha256 $2"
  "on$1" "python3 - $PKG/MODEL_MANIFEST.json $2 $3" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
assert m["identity"]["model_sha256"] == sys.argv[2], "identity"
assert m["parameters"]["loaded"] == int(sys.argv[3]), f"loaded {m['parameters']['loaded']}"
assert m["calibration"] is None
print(f"restaged: identity {sys.argv[2][:12]}, loaded {m['parameters']['loaded']:,}, T = 1")
EOF
  echo "manifest $("on$1" "sha256sum < $PKG/MODEL_MANIFEST.json | cut -c1-64")"
}
case "$STAGE" in
  stage)
    has_mirror d
    model=$(model_sha)
    count=$(loaded)
    ond "test ! -e $PKG" || { echo "$PKG exists: refusing to overwrite" >&2; exit 3; }
    copy_checkpoint d
    restage d "$model" "$count" ;;
  stage-c)
    has_mirror c
    model=$(model_sha)
    count=$(loaded)
    ond "test -f $PKG/MODEL_MANIFEST.json" || { echo "run stage (node D) first" >&2; exit 3; }
    onc "test ! -e $PKG" || { echo "node C $PKG exists: refusing to overwrite" >&2; exit 3; }
    if ! onc "test -e $FIX2"; then
      echo "$(date -u +%FT%TZ) relaying the fix2 template node D -> node C through node B"
      onb "$XFER root@${D#*@} 'tar -C ${FIX2%/*} -cf - ${FIX2##*/}' | \
        $XFER root@${C#*@} 'umask 077; mkdir -p ${FIX2%/*} && tar -C ${FIX2%/*} -xf -'"
    fi
    t=$(ond "$(sums "$FIX2")") c=$(onc "$(sums "$FIX2")")
    [ -n "$t" ] && [ "$t" = "$c" ] || { echo "node C's fix2 template differs from node D's" >&2; exit 3; }
    echo "fix2 template: $(wc -l <<< "$t") files, SHA-256 list equal to node D's"
    copy_checkpoint c
    restage c "$model" "$count"
    t=$(ond "$(sums "$PKG")") c=$(onc "$(sums "$PKG")")
    [ -n "$t" ] && [ "$t" = "$c" ] || { echo "node C's restaged package differs from node D's" >&2; exit 3; }
    echo "node C package: $(wc -l <<< "$c") files, SHA-256 list equal to node D's" ;;
  stage-b|stage-e)  # node B pulls from node D over its transfer key; node E gets node D's files relayed through node B
    node=${STAGE#stage-} model=$(model_sha)
    has_mirror "$node"
    ond "test -f $PKG/MODEL_MANIFEST.json" || { echo "run stage (node D) first" >&2; exit 3; }
    ond "python3 -c 'import json,sys; sys.exit(json.load(open(sys.argv[1]))[\"identity\"][\"model_sha256\"] != sys.argv[2])' \
      $PKG/MODEL_MANIFEST.json $model" || { echo "node D's package does not carry $ARM's identity" >&2; exit 3; }
    [ "$node" = b ] && sink="bash -c" || sink="$XFER root@${E#*@}"
    if [ "$node" = b ]; then  # node B trains: it has the base snapshot but none of IX1's image, kit or panel
      if ! onb "docker image inspect --format '{{.Id}}' $IMAGE 2> /dev/null | grep -q '^$IMAGE_ID'"; then
        echo "$(date -u +%FT%TZ) node B pulls image $IMAGE from node D"
        onb "$XFER root@${D#*@} 'docker save $IMAGE' | docker load -q"
      fi
      onb "docker image inspect --format '{{.Id}}' $IMAGE | grep -q '^$IMAGE_ID'" ||
        { echo "node B's $IMAGE is not the frozen build" >&2; exit 3; }
      echo "node B image $IMAGE: frozen build"
      onb "test -f $BASE/config.json" || { echo "node B has no base snapshot $BASE" >&2; exit 3; }
      q="cd $BASE && find -L . -type f | sort | xargs sha256sum | sha256sum"  # snapshot files are symlinks to blobs
      [ "$(ond "$q")" = "$(onb "$q")" ] ||
        { echo "node B's base snapshot differs from node D's" >&2; exit 3; }
      echo "node B base snapshot: SHA-256 list equal to node D's"
      dirs="$KIT $R/panel-8 $PKG $CACHE"
    else
      dirs="$PKG $CACHE"
    fi
    for dir in $dirs; do
      if ! "on$node" "test -e $dir"; then
        echo "$(date -u +%FT%TZ) relaying ${dir#/data/dev2/} node D -> node ${node^^}"
        onb "$XFER root@${D#*@} 'tar -C ${dir%/*} -cf - ${dir##*/}' | \
          $sink 'umask 077; mkdir -p ${dir%/*} && tar -C ${dir%/*} -xf -'"
      fi
      t=$(ond "$(sums "$dir")") c=$("on$node" "$(sums "$dir")")
      [ -n "$t" ] && [ "$t" = "$c" ] || { echo "node ${node^^}'s ${dir##*/} differs from node D's" >&2; exit 3; }
      echo "node ${node^^} ${dir##*/}: $(wc -l <<< "$c") files, SHA-256 list equal to node D's"
    done
    "on$node" "test -e $CACHE.sha256" ||
      onb "$XFER root@${D#*@} 'cat $CACHE.sha256' | $sink 'umask 077; cat > $CACHE.sha256'"
    [ "$(ond "sha256sum < $CACHE.sha256")" = "$("on$node" "sha256sum < $CACHE.sha256")" ] ||
      { echo "node ${node^^}'s cache digest file differs from node D's" >&2; exit 3; }
    echo "node ${node^^} cache digest file equal to node D's" ;;
  control)
    has_mirror d
    C0=$R/runs/DEV2.0-27B-budget-control
    ond "test ! -e $C0/control.json" || { echo "the restage control already ran ($C0)" >&2; exit 3; }
    ond "cd $S && bash v2/eval/ix1/launch.sh ref --src $M --model DEV2.0-27B-budget --gpu 4 --run $C0 \
      --rows $R/panel-8/compat-86.gold-free.jsonl.gz --cache $R/parity/DEV2.0-27B/cache-frozen > $R/logs/m6-control.log 2>&1 && \
      PYTHONPATH=$S python3 -m v2.eval.ix1.parity --kit $R/parity/DEV2.0-27B/kit/results.jsonl --ref $C0/ref.jsonl \
        --out $C0/control.json > /dev/null"
    ond "python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({k: d[k] for k in (\"pass\", \"requests\", \"statuses\", \"max_abs_dp\")})); sys.exit(0 if d[\"pass\"] else 1)' $C0/control.json" ;;
  audit)
    has_mirror d
    A6=$R/runs/m6-audit X6=/data/dev2/private/27b/m6-data
    ond "test ! -e $R/logs/m6-audit.exit" || { echo "the M6 audit already ran ($A6)" >&2; exit 3; }
    ond "mkdir -p $R/logs; setsid nohup bash -c 'cd $S && PYTHONHASHSEED=0 PYTHONPATH=$S python3 -m v2.eval.ix1.contamination \
      --panel $R/panel-8 --train a20ib12pn=$X6/mixtures-m6pn-1/a20ib12pn.train.jsonl \
      --train a20ib1x=$X6/mixtures-m6-1/a20ib1x.train.jsonl --workers 24 --out $A6; echo \$? > $R/logs/m6-audit.exit' \
      > $R/logs/m6-audit.log 2>&1 < /dev/null &"
    echo "$(date -u +%FT%TZ) M6 contamination audit started on node D (CPU)"
    until ond "test -f $R/logs/m6-audit.exit"; do sleep 60; done
    ond "echo exit \$(cat $R/logs/m6-audit.exit); tail -n 3 $R/logs/m6-audit.log"
    (umask 077 && mkdir -p "${LOCAL%/*}/audit")
    ond "cat $A6/audit.json" > "${LOCAL%/*}/audit/audit.json"
    chmod 600 "${LOCAL%/*}/audit/audit.json" ;;
  audit-arm)  # the arm's training file (mixtures-m6-1, or -m6pn-1 for M6-IB2PN, on node B; the seed-2 mixture's file
    # must be byte-identical), checked against the mixture's SHA-256 list, copied to node D over node B's transfer key
    # and audited there (CPU) -> ix1/runs/m6-audit-ARM
    has_mirror d
    case $ARM in  # DATA-DIR:MIXTURES:MIX (the seeds' builds are DATA-DIR/MIXTURES-1 and MIXTURES-2, byte-identical)
      M6-IB) mixes="m6-data:mixtures-m6:a20ib1" ;;
      M6-IBX) mixes="m6-data:mixtures-m6:a20ib1x" ;;
      M6-IB2) mixes="m6-data:mixtures-m6:a20ib12" ;;
      M6-IB2PN) mixes="m6-data:mixtures-m6pn:a20ib12pn" ;;
      M6-IBxIB2-*) mixes="m6-data:mixtures-m6:a20ib1 m6-data:mixtures-m6:a20ib12" ;;
      M7-IB124ML) mixes="m7-data:mixtures-m7:a20ib124ml" ;;
      M7-IB14ML) mixes="m7-data:mixtures-m7:a20ib14ml" ;;
      M8-IB14) mixes="m7-data:mixtures-m7:a20ib14" ;;
      M8-IB124) mixes="m7-data:mixtures-m7:a20ib124" ;;
      X7-IBxIB2xIB14ML) mixes="m6-data:mixtures-m6:a20ib1 m6-data:mixtures-m6:a20ib12 m7-data:mixtures-m7:a20ib14ml" ;;
      X7-4ARM) mixes="m6-data:mixtures-m6:a20ib1 m6-data:mixtures-m6:a20ib12 m7-data:mixtures-m7:a20ib14ml
        m7-data:mixtures-m7:a20ib124ml" ;;
      X8-IBxIB2-8) mixes="m6-data:mixtures-m6:a20ib1 m6-data:mixtures-m6:a20ib12" ;;
      X8-ML) mixes="m6-data:mixtures-m6:a20ib1 m6-data:mixtures-m6:a20ib12 m7-data:mixtures-m7:a20ib14ml
        m7-data:mixtures-m7:a20ib124ml m7-data:mixtures-m7:a20ib14 m7-data:mixtures-m7:a20ib124" ;;
      X9-LRH2 | X9-LRH2xM50 | X9-LRH | X9-LRHxM50 | X9-IBxIB2-10)
        mixes="m6-data:mixtures-m6:a20ib1 m6-data:mixtures-m6:a20ib12" ;;
      X9-ML0) mixes="m6-data:mixtures-m6:a20ib1 m6-data:mixtures-m6:a20ib12 m9-data:mixtures-m9:a20ib1ml
        m9-data:mixtures-m9:a20ib12ml" ;;
    esac
    A7=$R/runs/m6-audit-$ARM
    ond "test ! -e $R/logs/m6-audit-$ARM.exit" || { echo "the $ARM audit already ran ($A7)" >&2; exit 3; }
    args=""
    for spec in $mixes; do
      IFS=: read -r data m mix <<< "$spec"
      X6=/data/dev2/private/27b/$data
      T7=$X6/audit-$ARM
      want=$(onb "cd $X6 && awk '\$2 == \"$mix.train.jsonl\" { print \$1 }' $m-1.sha256")
      [[ "$want" =~ ^[0-9a-f]{64}$ ]] || { echo "$m-1.sha256 lists no $mix.train.jsonl" >&2; exit 3; }
      [ "$(onb "sha256sum < $X6/$m-2/$mix.train.jsonl | cut -c1-64")" = "$want" ] ||
        { echo "build 2's $mix.train.jsonl differs from build 1's: audit both" >&2; exit 3; }
      ond "test -f $T7/$mix.train.jsonl" ||
        onb "$XFER root@${D#*@} 'umask 077; mkdir -p $T7' && cat $X6/$m-1/$mix.train.jsonl | \
          $XFER root@${D#*@} 'cat > $T7/$mix.train.jsonl'"
      [ "$(ond "sha256sum < $T7/$mix.train.jsonl | cut -c1-64")" = "$want" ] ||
        { echo "node D's copy of $mix.train.jsonl is not the listed file" >&2; exit 3; }
      echo "$mix.train.jsonl ${want:0:12} (every seed's file) on node D"
      args+=" --train $mix=$T7/$mix.train.jsonl"
    done
    ond "mkdir -p $R/logs; setsid nohup bash -c 'cd $S && PYTHONHASHSEED=0 PYTHONPATH=$S python3 -m v2.eval.ix1.contamination \
      --panel $R/panel-8$args --workers 24 --out $A7; echo \$? > $R/logs/m6-audit-$ARM.exit' \
      > $R/logs/m6-audit-$ARM.log 2>&1 < /dev/null &"
    echo "$(date -u +%FT%TZ) $ARM contamination audit started on node D (CPU)"
    until ond "test -f $R/logs/m6-audit-$ARM.exit"; do sleep 60; done
    ond "echo exit \$(cat $R/logs/m6-audit-$ARM.exit); tail -n 3 $R/logs/m6-audit-$ARM.log"
    ond "python3 -c 'import json,sys; a=json.load(open(sys.argv[1])); p=a[\"planted_control\"]; print(\"planted control\", p[\"found\"], \"/\", p[\"planted\"]); sys.exit(0 if p[\"found\"] == p[\"planted\"] and not p.get(\"missed\") else 1)' $A7/audit.json" ;;
  parity)  # M6_PARITY_GPU: dN (node D, default d4; a bare N = node D), eN or bN (node E / B after stage-e / stage-b;
    # the parity record is then copied to node D, whose record run / score read); a 27B hold on the GPU steps aside
    pg=${M6_PARITY_GPU:-d4}
    [[ "$pg" =~ ^[0-7]$ ]] && pg=d$pg
    [[ "$pg" =~ ^(d[0-7]|e[0-367]|b[015])$ ]] ||
      { echo "M6_PARITY_GPU: node D GPU0-7 (dN), node E GPU0-3 / 6-7 (eN) or node B GPU0 / 1 / 5 (bN), not '$pg'" >&2
        exit 2; }
    pn=${pg:0:1} pgpu=${pg:1}
    has_mirror "$pn"
    "on$pn" "test -f $PKG/MODEL_MANIFEST.json" || { echo "node ${pn^^} has no restaged package (stage / stage-e)" >&2; exit 3; }
    ond "test ! -e $R/parity/$ARM/parity.json" || { echo "parity of $ARM already ran" >&2; exit 3; }
    "on$pn" "test ! -e $R/logs/m6-parity-$ARM.exit" || { echo "parity of $ARM already ran on node ${pn^^}" >&2; exit 3; }
    "on$pn" "f=/data/dev2/leases/gpu$pgpu.lock/owner; [ -s \$f ] || exit 0; grep -qx 'track=eval-ix1' \$f && exit 0; \
      { grep -qx 'track=27b' \$f && grep -qx 'status=reserved-idle' \$f; } || grep -q '^status=released' \$f || \
      { echo 'node ${pn^^} gpu$pgpu: neither an idle 27B lease nor released' >&2; exit 3; }; \
      mv \$f /data/dev2/leases/gpu$pgpu.lock/owner.m6-set-aside-\$(date -u +%Y%m%dT%H%M%SZ)" || exit 3
    "on$pn" "mkdir -p $R/logs; setsid nohup bash -c 'cd $S && bash v2/eval/ix1/launch.sh parity --src $M --model $ARM \
      --gpu $pgpu --run $R/parity/$ARM --rows $R/panel-8/compat-86.gold-free.jsonl.gz; \
      echo \$? > $R/logs/m6-parity-$ARM.exit' > $R/logs/m6-parity-$ARM.log 2>&1 < /dev/null &"
    echo "$(date -u +%FT%TZ) $ARM parity gate started on node ${pn^^} GPU$pgpu (two 27B passes, ~10 min)"
    until "on$pn" "test -f $R/logs/m6-parity-$ARM.exit"; do sleep 60; done
    "on$pn" "cat $R/logs/m6-parity-$ARM.exit; python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({k: d[k] for k in (\"pass\", \"requests\", \"statuses\", \"max_abs_dp\")})); sys.exit(0 if d[\"pass\"] else 1)' $R/parity/$ARM/parity.json"
    if [ "$pn" != d ]; then
      [ "$pn" = e ] && from="$XFER root@${E#*@}" || from="bash -c"
      onb "$from 'cat $R/parity/$ARM/parity.json' | \
        $XFER root@${D#*@} 'umask 077; mkdir -p $R/parity/$ARM && cat > $R/parity/$ARM/parity.json'"
      [ "$("on$pn" "sha256sum < $R/parity/$ARM/parity.json")" = "$(ond "sha256sum < $R/parity/$ARM/parity.json")" ] ||
        { echo "node D's copy of the parity record differs from node ${pn^^}'s" >&2; exit 3; }
      echo "parity record copied node ${pn^^} -> node D (SHA-256 equal)"
    fi ;;
  hold | unhold)  # hold: each idle GPU of M6_INDEX_GPUS (rocm-smi use <= 5%, VRAM <= 2 GiB) whose owner is absent, a
    # released one or an eval-ix1 run with every shard ended and none of its containers running gets a 27B
    # reserved-idle owner naming ARM (COORDINATION 2026-10-02 15:50: node D GPU0-7, node E GPU0-3 / 6-7 and node B
    # GPU0 / 1 / 5 are 27B's); the earlier owner is kept as owner.prev-27b-hold-<UTC>. run and parity set a hold aside.
    # unhold: a 27B hold of ARM on those GPUs -> status released
    read -r -a entries <<< "${M6_INDEX_GPUS:-}"
    [ ${#entries[@]} -ge 1 ] || { echo "M6_INDEX_GPUS lists no GPU" >&2; exit 2; }
    for e in "${entries[@]}"; do
      [[ "$e" =~ ^(d[0-7]|b[015]|e[0-367])$ ]] ||
        { echo "M6_INDEX_GPUS: node D GPU0-7, node B GPU0 / 1 / 5 or node E GPU0-3 / 6-7, not '$e'" >&2; exit 2; }
    done
    for e in "${entries[@]}"; do
      node=${e:0:1} g=${e:1}
      if [ "$STAGE" = unhold ]; then
        "on$node" "f=/data/dev2/leases/gpu$g.lock/owner; grep -qx 'purpose=27B Index hold for $ARM' \$f 2> /dev/null || \
          { echo 'node ${node^^} gpu$g: no 27B hold of $ARM'; exit 0; }; \
          printf 'track=27b\nstatus=released (27B Index hold of $ARM ended)\nlast_job_end_utc=%s\n' \
          \"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\" > \$f; echo 'node ${node^^} gpu$g: released'" < /dev/null
        continue
      fi
      "on$node" "python3 - $node $g $ARM" <<'EOF' || true
import glob, json, os, subprocess, sys, time
node, g, arm = sys.argv[1].upper(), sys.argv[2], sys.argv[3]
lock = f"/data/dev2/leases/gpu{g}.lock"
owner = f"{lock}/owner"
card = json.loads(subprocess.run(["rocm-smi", "--showuse", "--showmeminfo", "vram", "--json"],
                                 capture_output=True, text=True, check=True).stdout)[f"card{g}"]
if float(card["GPU use (%)"]) > 5 or int(card["VRAM Total Used Memory (B)"]) > 2 * 2**30:
    sys.exit(print(f"node {node} gpu{g}: busy"))
text = open(owner).read() if os.path.exists(owner) else ""
lines = set(text.split("\n"))
released = any(line.startswith("status=released") for line in lines)
if f"purpose=27B Index hold for {arm}" in lines:
    sys.exit(print(f"node {node} gpu{g}: already held for {arm}"))
if text.strip() and "track=eval-ix1" in lines and not released:
    run = next((line.split("=", 1)[1] for line in lines if line.startswith("run_dir=")), "")
    shards = glob.glob(f"{run}/shard-*")  # panel-8: a run is over once its 8 shards have ended
    open_shards = [w for w in shards if not os.path.exists(f"{w}/end_epoch")]
    names = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True).stdout
    tag = "ix1-" + os.path.basename(run).lower().replace(".", "_") + "-"
    if not run or len(shards) < 8 or open_shards or any(name.startswith(tag) for name in names.split()):
        sys.exit(print(f"node {node} gpu{g}: its eval-ix1 run {os.path.basename(run) or '?'} is in progress"))
elif text.strip() and "track=eval-ix1" not in lines and not released:
    track = next((line for line in lines if line.startswith("track=")), "track=?")
    sys.exit(print(f"node {node} gpu{g}: leased ({track})"))
stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
os.makedirs(lock, exist_ok=True)
if text.strip():
    os.replace(owner, f"{owner}.prev-27b-hold-{stamp}")
with open(owner, "w") as out:
    out.write(f"track=27b\nstatus=reserved-idle\npurpose=27B Index hold for {arm}\n"
              f"start_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
print(f"node {node} gpu{g}: held for {arm}")
EOF
    done ;;
  run)
    plan=$(placement) || exit 2
    has_mirror d
    ond "python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))[\"pass\"] else 1)' $R/parity/$ARM/parity.json" ||
      { echo "parity gate missing or failed" >&2; exit 3; }
    for node in b c e; do
      grep -q "^$node " <<< "$plan" || continue
      has_mirror $node
      t=$(ond "$(sums "$PKG")") c=$("on$node" "test -f $PKG/MODEL_MANIFEST.json && $(sums "$PKG")" || true)
      [ -n "$t" ] && [ "$t" = "$c" ] ||
        { echo "node ${node^^} has no package equal to node D's: run stage-$node first" >&2; exit 3; }
      [ "$(ond "$(sums "$CACHE")"; ond "sha256sum < $CACHE.sha256")" = \
        "$("on$node" "$(sums "$CACHE")"; "on$node" "sha256sum < $CACHE.sha256")" ] ||
        { echo "node ${node^^}'s frozen cache or its digest file differs from node D's" >&2; exit 3; }
      if ! "on$node" "test -f $R/parity/$ARM/parity.json"; then
        ond "cat $R/parity/$ARM/parity.json" |
          "on$node" "umask 077; mkdir -p $R/parity/$ARM && cat > $R/parity/$ARM/parity.json"
      fi
      [ "$(ond "sha256sum < $R/parity/$ARM/parity.json")" = "$("on$node" "sha256sum < $R/parity/$ARM/parity.json")" ] ||
        { echo "node ${node^^}'s parity record differs from node D's" >&2; exit 3; }
    done
    while read -r node _ gpus; do  # 27B's idle leases (track=27b, reserved-idle; e.g. a hold) and released owners step aside
      for g in $gpus; do
        "on$node" "f=/data/dev2/leases/gpu$g.lock/owner; [ -s \$f ] || exit 0; grep -qx 'track=eval-ix1' \$f && exit 0; \
          { grep -qx 'track=27b' \$f && grep -qx 'status=reserved-idle' \$f; } || grep -q '^status=released' \$f || \
          { echo 'node ${node^^} gpu$g: neither an idle 27B lease nor released' >&2; exit 3; }; \
          mv \$f /data/dev2/leases/gpu$g.lock/owner.m6-set-aside-\$(date -u +%Y%m%dT%H%M%SZ) && \
          echo 'node ${node^^} gpu$g: owner set aside'" < /dev/null || exit 3
      done
    done <<< "$plan"
    stagger=${M6_INDEX_STAGGER:-300} after=${M6_INDEX_AFTER:-}
    [[ "$stagger" =~ ^[0-9]{1,4}$ ]] || { echo "M6_INDEX_STAGGER: seconds, not '$stagger'" >&2; exit 2; }
    [[ "$after" =~ ^[0-7]?$ ]] || { echo "M6_INDEX_AFTER: one shard index, not '$after'" >&2; exit 2; }
    while read -r node shards gpus; do  # ssh reads stdin: without < /dev/null it eats the plan's other lines
      "on$node" "mkdir -p $R/logs; M6_INDEX_STAGGER=$stagger M6_INDEX_AFTER=$after setsid nohup bash \
$S/v2/27b/m6/m6-index-run.sh $SHA $ARM $node $shards $gpus >> $R/logs/m6-index-$ARM-$node.log 2>&1 < /dev/null & \
echo node $node m6-index-run \$!: shards $shards on GPU $gpus" < /dev/null
    done <<< "$plan"
    sleep 20
    while read -r node _; do "on$node" "tail -n 3 $R/logs/m6-index-$ARM-$node.log" < /dev/null; done <<< "$plan" ;;
  status)
    for node in d b c e; do
      "on$node" "test -d $R/runs/$ARM" || continue
      "on$node" "python3 - $R/runs/$ARM $node" <<'EOF'
import glob, os, sys, time
run, node, total = sys.argv[1], sys.argv[2], 0.0
for w in sorted(glob.glob(f"{run}/shard-*")):
    if not os.path.exists(f"{w}/start_epoch"):
        print(f"node {node} {os.path.basename(w)}: waiting")
        continue
    rows = sum(1 for _ in open(f"{w}/results.jsonl")) if os.path.exists(f"{w}/results.jsonl") else 0
    start = float(open(f"{w}/start_epoch").read())
    end = float(open(f"{w}/end_epoch").read()) if os.path.exists(f"{w}/end_epoch") else None
    code = open(f"{w}/exit_code").read().strip() if os.path.exists(f"{w}/exit_code") else "-"
    total += ((end or time.time()) - start) / 3600
    print(f"node {node} {os.path.basename(w)}: {rows} records, {'ended exit ' + code if end else 'running'}")
print(f"node {node}: GPU-h so far (current intervals) {total:.2f}")
EOF
      "on$node" "tail -n 4 $R/logs/m6-index-$ARM-$node.log 2> /dev/null || echo 'node $node: no run log'"
    done ;;
  collect)
    found=0
    for node in b c e; do
      ks=$("on$node" "cd $R/runs/$ARM 2> /dev/null && for k in 0 1 2 3 4 5 6 7; do test -d shard-\$k && echo \$k; done" || true)
      [ -n "$ks" ] || continue
      found=1
      case $node in b) from="bash -c" ;; c) from="$XFER root@${C#*@}" ;; e) from="$XFER root@${E#*@}" ;; esac
      for k in $ks; do
        if ! "on$node" "test -f $R/runs/$ARM/shard-$k/end_epoch && test \"\$(cat $R/runs/$ARM/shard-$k/exit_code)\" = 0"; then
          echo "node ${node^^} shard $k has not ended with exit code 0" >&2; exit 3
        fi
        ond "test ! -e $R/runs/$ARM/shard-$k" || { echo "node D already has shard-$k of $ARM" >&2; exit 3; }
      done
      ond "umask 077; mkdir -p $R/runs/$ARM"
      for k in $ks; do  # relayed through node B's transfer key: the workstation's path to node D is slow
        onb "$from 'tar -C $R/runs/$ARM --exclude=shard-$k/triton --exclude=shard-$k/home -cf - shard-$k' | \
          $XFER root@${D#*@} 'tar -C $R/runs/$ARM -xf -'"
        c=$("on$node" "cd $R/runs/$ARM/shard-$k && find . \\( -path ./triton -o -path ./home \\) -prune -o -type f -print | sort | xargs sha256sum")
        t=$(ond "cd $R/runs/$ARM/shard-$k && find . -type f | sort | xargs sha256sum")
        [ -n "$c" ] && [ "$c" = "$t" ] || { echo "node D's copy of shard $k differs from node ${node^^}'s" >&2; exit 3; }
        echo "shard $k: $(wc -l <<< "$t") files copied node ${node^^} -> node D, SHA-256 lists equal"
      done
      onb "$from 'cd $R/runs/$ARM && tar -cf - launcher-run-only-*.json' | \
        $XFER root@${D#*@} 'tar -C $R/runs/$ARM --keep-old-files -xf -'"
      echo "node ${node^^} launcher records copied"
    done
    [ $found = 1 ] || { echo "no node besides node D ran a shard of $ARM" >&2; exit 3; } ;;
  score)
    has_mirror d
    base=${M6_INDEX_BASE:-} files=""
    if [ -n "$base" ]; then
      [[ "$base" =~ $ARM_RE && "$base" != "$ARM" ]] || { echo "M6_INDEX_BASE: another arm, not '$base'" >&2; exit 2; }
      ond "test -f $R/runs/$base/merged/results.jsonl && test -f $R/runs/$base/merged/receipt.json" ||
        { echo "no scored Index run of $base on node D" >&2; exit 3; }
      bl=$(tr '[:upper:]' '[:lower:]' <<< "$base")
      files="family-delta-vs-$bl.json paired-boot-vs-$bl.json"
    fi
    ond "for k in 0 1 2 3 4 5 6 7; do test \"\$(cat $R/runs/$ARM/shard-\$k/exit_code 2>/dev/null)\" = 0 || exit 1; done" ||
      { echo "not every shard ended with exit code 0 on node D (collect node C's shards first)" >&2; exit 3; }
    boot="cd $R/.. && PYTHONPATH=$S:\$PWD/kit-19ad28ec venv/bin/python -m v2.eval.ix1.paired_boot --suite-dir suite-0.2 \
      --new $R/runs/$ARM/merged/results.jsonl --external $R/../external/index021-frontier-gap-2026-10-01.json \
      --replicates 2000 --seed 20261002 --workers 24"
    vs_base=true
    [ -z "$base" ] || vs_base="PYTHONPATH=$S python3 -m v2.eval.ix1.family_delta --base $base=$R/runs/$base/merged/compare.json \
        --new $ARM=$R/runs/$ARM/merged/compare.json --out $R/runs/$ARM/family-delta-vs-$bl.json > /dev/null && \
      ($boot --base $R/runs/$base/merged/results.jsonl --out $R/runs/$ARM/paired-boot-vs-$bl.json \
        > $R/logs/m6-boot-$ARM-vs-$base.log 2>&1)"
    ond "cd $S && bash v2/eval/ix1/score.sh --src $M --model $ARM --size 27B --panel $R/panel-8 > $R/logs/m6-score-$ARM.log 2>&1 && \
      PYTHONPATH=$S python3 -m v2.eval.ix1.family_delta --base A20r=$R/runs/DEV2.0-27B/merged-budget/compare.json \
        --new $ARM=$R/runs/$ARM/merged/compare.json --out $R/runs/$ARM/family-delta-vs-a20r.json > /dev/null && \
      PYTHONPATH=$S python3 -m v2.eval.ix1.family_delta --base M5-L128=$R/runs/M5-L128/merged/compare.json \
        --new $ARM=$R/runs/$ARM/merged/compare.json --out $R/runs/$ARM/family-delta-vs-m5-l128.json > /dev/null && \
      { ($boot --base $R/runs/DEV2.0-27B/merged-budget-r/results.jsonl --out $R/runs/$ARM/paired-boot-vs-a20r.json \
        > $R/logs/m6-boot-$ARM.log 2>&1) & a=\$!; (cd $S && $vs_base); b=\$?; wait \$a && test \$b = 0; }"
    (umask 077 && mkdir -p "$LOCAL")
    for f in merged/compare.json merged/receipt.json merged/port.json family-delta-vs-a20r.json \
      family-delta-vs-m5-l128.json paired-boot-vs-a20r.json $files; do
      ond "cat $R/runs/$ARM/$f" > "$LOCAL/$(basename "$f")"
    done
    [ -z "$base" ] || ond "cat $R/runs/$base/merged/receipt.json" > "$LOCAL/receipt-$bl.json"
    ond "cat $R/runs/DEV2.0-27B/merged-budget/compare.json" > "$LOCAL/compare-a20r.json"
    ond "cat $R/runs/DEV2.0-27B/merged-budget-r/receipt.json" > "$LOCAL/receipt-a20r.json"
    ond "cat $R/parity/$ARM/parity.json" > "$LOCAL/parity.json"
    ond "cat $R/runs/DEV2.0-27B-budget-control/control.json 2> /dev/null" > "$LOCAL/control-a20r.json" || true
    chmod 600 "$LOCAL"/*.json
    python3 - "$LOCAL/receipt.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
print(json.dumps({k: r[k] for k in ("rows", "statuses", "gpu_hours", "results_sha256", "panel_run_ids_sha256")}))
EOF
    echo "private outputs in $LOCAL (never copy a value into a commit, record, gist or card)" ;;
  release)  # a GPU whose M6 owner was set aside (owner.m6-set-aside-<UTC>, node D GPU0-3) gets that owner back
    for node in d b c e; do
      case $node in d) gs="0 1 2 3 4 5 6 7" ;; b) gs="0 1 5" ;; c) gs="1 2 3 4 5 6 7" ;; e) gs="0 1 2 3 6 7" ;; esac
      "on$node" "docker ps --format '{{.Names}}' | grep -q '^$TAG'" &&
        { echo "an $ARM Index container is still running on node ${node^^}" >&2; exit 3; }
      "on$node" "for g in $gs; do f=/data/dev2/leases/gpu\$g.lock/owner; grep -q 'IX1 .* $ARM\$' \$f 2>/dev/null || continue; \
        m=\$(ls -t /data/dev2/leases/gpu\$g.lock/owner.m6-set-aside-* 2> /dev/null | head -n 1); \
        if [ -n \"\$m\" ]; then cp -p \"\$m\" \$f; echo node $node gpu\$g back to its 27b M6 owner; continue; fi; \
        printf 'track=eval-ix1\nstatus=released (27B M6 Index run of $ARM done)\nlast_job_end_utc=%s\n' \
        \"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\" > \$f; echo node $node gpu\$g released; done"
    done ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
