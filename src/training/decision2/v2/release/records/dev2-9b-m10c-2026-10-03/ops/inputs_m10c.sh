#!/usr/bin/env bash
# Decision-2.0-Lux-9B Index-first successor of KIB4-a40 (9B M10 amendment 7): release inputs of one candidate onto
# node A (workstation side, ssh control only). Copies go node to node: node A pulls from nodes C-F with the temporary
# key, and node B relays through node C (nodes A and B hold the key authorized on C-F). Per-file SHA-256 lists (or file
# SHA-256) are compared at both ends.
#   bf16   the BF16 copy the Index run scored and its bf16-copy.json -> node A
#          /data/dev2/runs/release/inputs/dev2-9b-m10c-CAND/bf16/{checkpoint,bf16-copy.json}
#          (an M10 point: models/ix1/9b-m10/bf16/CAND on IXNODE; an arm-factory point AF-<name>: node A
#          models/ix1/af/ckpt/AF-<name>-bf16 and receipts/AF-<name>-bf16-bf16-copy.json)
#   index  from IXNODE: the gate bootstrap vs M10-KIB4-a40-bf16 (m10/ix.sh gate), the candidate's IX1 run receipt and
#          kit index.json, the restaged package's MODEL_MANIFEST.json; from node C: M10-KIB4-a40-bf16's run receipt and
#          the row-level contamination audit (M10C_AUDIT, default ix1/audit/m10) -> node A
#          /data/dev2/private/release/m10c/CAND/ (mode 700; values stay private)
# Usage: inputs_m10c.sh CAND bf16 IXNODE | CAND index IXNODE
set -euo pipefail
CAND=${1:?CAND} STAGE=${2:?STAGE} IXNODE=${3:?IXNODE}
NODES=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
addr() { awk -F= -v k="node-$1" '$1 == k { print substr($0, length(k) + 2); exit }' "$NODES"; }
on() { local n=$1; shift; ssh -n -o BatchMode=yes -o ConnectTimeout=30 "$(addr "$n")" "$@"; }
KEY="-i /root/.ssh/d2_temp_cd -o BatchMode=yes -o ConnectTimeout=30"
IX=/data/dev2/private/eval/index021/ix1
AUDIT=${M10C_AUDIT:-m10}
IN=/data/dev2/runs/release/inputs/dev2-9b-m10c-$CAND
PRIV=/data/dev2/private/release/m10c/$CAND
RELAY=/data/dev2/tmp/m10c-relay
if [[ "$CAND" == AF-* ]]; then
  AFD=/data/dev2/models/ix1/af NAME=$CAND-bf16
  CK=$AFD/ckpt/$NAME RECEIPT=$AFD/receipts/$NAME-bf16-copy.json PKG=$AFD/$NAME-re51f9881
  [ "$IXNODE" = a ] || { echo "arm-factory 9B points are on node A" >&2; exit 2; }
else
  MD=/data/dev2/models/ix1/9b-m10 NAME=M10-$CAND-bf16
  CK=$MD/bf16/$CAND RECEIPT=$MD/bf16/$CAND.receipt/bf16-copy.json PKG=$MD/$CAND-bf16-re51f9881
fi
sums() { printf '%s' "cd '$1' && find . -type f | LC_ALL=C sort | xargs -d '\n' -r -P 8 -n 8 sha256sum | sort -k2"; }
fetch() { # fetch <node> <src> <node-A dst>; a directory source ends in /
  case "$1" in
    a) on a "umask 077; cp -a $2 $3" ;;
    b) local r
       r=$RELAY/$CAND-$(basename "$3")
       on c "test ! -e $r && mkdir -p $RELAY"
       on b "rsync -a -e 'ssh $KEY' $2 $(addr c):$r"
       on a "umask 077; rsync -a -e 'ssh $KEY' $(addr c):$r$([[ "$2" == */ ]] && echo /) $3"
       on c "rm -rf $r" ;;
    *) on a "umask 077; rsync -a -e 'ssh $KEY' $(addr "$1"):$2 $3" ;;
  esac
}
case "$STAGE" in
  bf16)
    on a "test ! -e $IN/bf16" || { echo "$IN/bf16 exists" >&2; exit 3; }
    on a "mkdir -p $IN/bf16"
    fetch "$IXNODE" "$CK/" "$IN/bf16/checkpoint/"
    fetch "$IXNODE" "$RECEIPT" "$IN/bf16/bf16-copy.json"
    on a "chmod -R go+rX $IN/bf16"
    a=$(on "$IXNODE" "$(sums "$CK")") b=$(on a "$(sums "$IN/bf16/checkpoint")")
    [ -n "$a" ] && [ "$a" = "$b" ] || { echo "the BF16 checkpoint differs on node A" >&2; exit 3; }
    [ "$(on "$IXNODE" "sha256sum < $RECEIPT")" = "$(on a "sha256sum < $IN/bf16/bf16-copy.json")" ] ||
      { echo "bf16 receipt differs" >&2; exit 3; }
    echo "$(wc -l <<< "$a") files equal"
    on a "python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(\"bf16 copy\", r[\"source_model_sha256\"][:12], \"->\", r[\"model_sha256\"])' $IN/bf16/bf16-copy.json" ;;
  index)
    G=$IX/m10/gate/$NAME
    if ! on "$IXNODE" "test -f $G/DONE && test -f $G/paired-boot-full-vs-kib4a40.json" || \
      ! on c "test -f $IX/audit/$AUDIT/out/audit.json"; then
      echo "no gate bootstrap for $NAME on node $IXNODE, or no audit $AUDIT" >&2; exit 3
    fi
    on a "umask 077; mkdir -p $PRIV && chmod 700 /data/dev2/private/release/m10c $PRIV"
    for pair in "$IXNODE:$G/paired-boot-full-vs-kib4a40.json:boot-full-vs-kib4a40.json" \
      "$IXNODE:$G/ref.json:gate-ref.json" \
      "$IXNODE:$IX/runs/$NAME/merged/receipt.json:receipt.json" \
      "$IXNODE:$IX/runs/$NAME/merged/kit/index.json:kit-index.json" \
      "$IXNODE:$PKG/MODEL_MANIFEST.json:package-manifest.json" \
      "c:$IX/runs/M10-KIB4-a40-bf16/merged/receipt.json:base-receipt.json" "c:$IX/audit/$AUDIT/out/audit.json:audit.json"; do
      node=${pair%%:*} rest=${pair#*:}; src=${rest%%:*} dst=${rest#*:}
      on a "test ! -e $PRIV/$dst" || { echo "$PRIV/$dst exists" >&2; exit 3; }
      fetch "$node" "$src" "$PRIV/$dst"
      [ "$(on "$node" "sha256sum < $src")" = "$(on a "sha256sum < $PRIV/$dst")" ] || { echo "$dst differs" >&2; exit 3; }
      echo "$dst $(on a "sha256sum < $PRIV/$dst | cut -c1-16")"
    done ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
