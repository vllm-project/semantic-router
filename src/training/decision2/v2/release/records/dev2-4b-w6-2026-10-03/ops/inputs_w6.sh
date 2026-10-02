#!/usr/bin/env bash
# Decision-2.0-Nox-4B wave-6 release (4B owner, decoder M17b): release inputs onto node A (workstation side), as the
# 4b-SDMLxALL release's inputs_xall.sh. Node A pulls from nodes C and F directly (transfer key); per-file SHA-256
# lists are compared at both ends.
#   bf16   node F models/ix1/af/ckpt/AF-4b-CAND-bf16 (+ its bf16-copy receipt; the BF16 copy the Index run scored)
#          -> node A /data/dev2/runs/release/inputs/dev2-4b-<key>/bf16/{checkpoint,bf16-copy.json}
#   index  the run node (F, or A for AFxALL3): the full-panel paired bootstrap vs DEV2.0-4B-SDMLxALL-bf16 (the current
#          revision's weights; ixchain.sh's paired-boot-full-vs-ref.json), both IX1 run receipts, the candidate's kit
#          index.json; node C: the row-level contamination audit of wave 6's member TRAIN files (m17b-ops.sh audit6);
#          node F: the restaged package's MODEL_MANIFEST.json -> node A /data/dev2/private/release/4bif/CAND/
#          (mode 700; values stay private)
# The autotune cache is the current revision's frozen phase A cache on node A (release/triton/runtime-a-4b).
# Usage: inputs_w6.sh CAND bf16|index
set -euo pipefail
CAND=${1:?CAND} STAGE=${2:?STAGE}
case $CAND in AFxALL | AFxALL2) RNODE=f ;; AFxALL3) RNODE=a ;; XALLx | XALLU2) RNODE=c ;; *) echo "CAND: AFxALL | AFxALL2 | AFxALL3 | XALLx | XALLU2" >&2; exit 2 ;; esac
KEY=${CAND,,}
NODES=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
addr() { awk -F= -v k="node-$1" '$1 == k { print substr($0, length(k) + 2); exit }' "$NODES"; }
on() { local n=$1; shift; ssh -n -o BatchMode=yes -o ConnectTimeout=30 "$(addr "$n")" "$@"; }
KEYARGS="-i /root/.ssh/d2_temp_cd -o BatchMode=yes -o ConnectTimeout=30"
IX=/data/dev2/private/eval/index021/ix1
NAME=AF-4b-$CAND-bf16 BASE=DEV2.0-4B-SDMLxALL-bf16
MD=/data/dev2/models/ix1/af
BF=$MD/ckpt/$NAME BFR=$MD/receipts/$NAME-bf16-copy.json
PKGM=$MD/$NAME-r13d42143/MODEL_MANIFEST.json
RUNDIR=$IX/runs/$NAME
BASEDIR=$IX/af/refs/$BASE
[ "$RNODE" = c ] && BASEDIR=$IX/runs/$BASE
AUDIT=$IX/audit/m17b-w6/out/audit.json
IN=/data/dev2/runs/release/inputs/dev2-4b-$KEY
PRIV=/data/dev2/private/release/4bif/$CAND
sums() { printf '%s' "cd '$1' && find . -type f | LC_ALL=C sort | xargs -d '\n' -r -P 8 -n 8 sha256sum | sort -k2"; }
same() { # same <node> <dir> <node> <dir>: per-file SHA-256 lists equal
  local a b
  a=$(on "$1" "$(sums "$2")") b=$(on "$3" "$(sums "$4")")
  [ -n "$a" ] && [ "$a" = "$b" ] || { echo "$2 on node $1 differs from $4 on node $3" >&2; exit 3; }
  echo "$(wc -l <<< "$a") files equal: node $1 $2 = node $3 $4"
}
fetch() { # fetch <node> <src> <node-A dst>
  if [ "$1" = a ]; then
    on a "umask 077; cp -a $2 $3"
  else
    on a "umask 077; rsync -a -e 'ssh $KEYARGS' $(addr "$1"):$2 $3"
  fi
}
case "$STAGE" in
  bf16)
    on a "test ! -e $IN/bf16" || { echo "$IN/bf16 exists" >&2; exit 3; }
    on a "mkdir -p $IN/bf16"
    fetch f "$BF/" "$IN/bf16/checkpoint/"
    fetch f "$BFR" "$IN/bf16/bf16-copy.json"
    on a "chmod -R go+rX $IN/bf16"
    same f "$BF" a "$IN/bf16/checkpoint"
    [ "$(on f "sha256sum < $BFR")" = "$(on a "sha256sum < $IN/bf16/bf16-copy.json")" ] ||
      { echo "bf16 receipt differs" >&2; exit 3; }
    on a "python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(\"bf16 copy\", r[\"source_model_sha256\"][:12], \"->\", r[\"model_sha256\"])' $IN/bf16/bf16-copy.json" ;;
  index)
    { on "$RNODE" "test -f $RUNDIR/paired-boot-full-vs-ref.json" && on c "test -f $AUDIT"; } || { echo "no bootstrap or audit for $NAME" >&2; exit 3; }
    [ "$(on "$RNODE" "python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[\"reference_run\"])' $RUNDIR/m10ix-ref.json")" = "$BASEDIR" ] \
      || { echo "$NAME's bootstrap is not against $BASEDIR" >&2; exit 3; }
    on a "umask 077; mkdir -p $PRIV && chmod 700 /data/dev2/private/release/4bif $PRIV"
    for pair in "$RNODE:$RUNDIR/paired-boot-full-vs-ref.json:boot-full-vs-xall.json" \
      "$RNODE:$RUNDIR/merged/receipt.json:receipt.json" "$RNODE:$BASEDIR/merged/receipt.json:base-receipt.json" \
      "$RNODE:$RUNDIR/merged/kit/index.json:kit-index.json" "c:$AUDIT:audit.json" "f:$PKGM:package-manifest.json"; do
      node=${pair%%:*} rest=${pair#*:}; src=${rest%%:*} dst=${rest#*:}
      on a "test ! -e $PRIV/$dst" || { echo "$PRIV/$dst exists" >&2; exit 3; }
      fetch "$node" "$src" "$PRIV/$dst"
      [ "$(on "$node" "sha256sum < $src")" = "$(on a "sha256sum < $PRIV/$dst")" ] || { echo "$dst differs" >&2; exit 3; }
      echo "$dst $(on a "sha256sum < $PRIV/$dst | cut -c1-16")"
    done ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
