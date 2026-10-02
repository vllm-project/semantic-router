#!/usr/bin/env bash
# Decision-2.0-Lux-9B Index-first successor from 9B M10: release inputs of one candidate onto node A (workstation side,
# ssh control only; copies go node to node, node A pulling from node C with the temporary key). Per-file SHA-256 lists
# (or file SHA-256) are compared at both ends.
#   bf16   node C models/ix1/9b-m10/bf16/CAND and its bf16-copy.json (the BF16 copy the Index run scored) -> node A
#          /data/dev2/runs/release/inputs/dev2-9b-m10-CAND/bf16/{checkpoint,bf16-copy.json}
#   index  from the Index node IXNODE (c or a): the full-panel paired bootstrap vs K-a13IB-bf16, the candidate's IX1 run
#          receipt and kit index.json, the restaged package's MODEL_MANIFEST.json; from node C: K-a13IB-bf16's run
#          receipt and the M10 row-level contamination audit -> node A /data/dev2/private/release/m10/CAND/ (mode 700;
#          values stay private)
# Usage: inputs_m10.sh CAND bf16 | CAND index IXNODE
set -euo pipefail
CAND=${1:?CAND} STAGE=${2:?STAGE} IXNODE=${3:-c}
NODES=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
addr() { awk -F= -v k="node-$1" '$1 == k { print substr($0, length(k) + 2); exit }' "$NODES"; }
on() { local n=$1; shift; ssh -n -o BatchMode=yes -o ConnectTimeout=30 "$(addr "$n")" "$@"; }
KEY="-i /root/.ssh/d2_temp_cd -o BatchMode=yes -o ConnectTimeout=30"
IX=/data/dev2/private/eval/index021/ix1
MD=/data/dev2/models/ix1/9b-m10
NAME=M10-$CAND-bf16
IN=/data/dev2/runs/release/inputs/dev2-9b-m10-$CAND
PRIV=/data/dev2/private/release/m10/$CAND
sums() { printf '%s' "cd '$1' && find . -type f | LC_ALL=C sort | xargs -d '\n' -r -P 8 -n 8 sha256sum | sort -k2"; }
fetch() { # fetch <node> <src> <node-A dst>
  if [ "$1" = a ]; then on a "umask 077; cp -a $2 $3"; else on a "umask 077; rsync -a -e 'ssh $KEY' $(addr "$1"):$2 $3"; fi
}
case "$STAGE" in
  bf16)
    on a "test ! -e $IN/bf16" || { echo "$IN/bf16 exists" >&2; exit 3; }
    on a "mkdir -p $IN/bf16"
    fetch c "$MD/bf16/$CAND/" "$IN/bf16/checkpoint/"
    fetch c "$MD/bf16/$CAND.receipt/bf16-copy.json" "$IN/bf16/bf16-copy.json"
    on a "chmod -R go+rX $IN/bf16"
    a=$(on c "$(sums "$MD/bf16/$CAND")") b=$(on a "$(sums "$IN/bf16/checkpoint")")
    [ -n "$a" ] && [ "$a" = "$b" ] || { echo "the BF16 checkpoint differs on node A" >&2; exit 3; }
    [ "$(on c "sha256sum < $MD/bf16/$CAND.receipt/bf16-copy.json")" = "$(on a "sha256sum < $IN/bf16/bf16-copy.json")" ] ||
      { echo "bf16 receipt differs" >&2; exit 3; }
    echo "$(wc -l <<< "$a") files equal"
    on a "python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(\"bf16 copy\", r[\"source_model_sha256\"][:12], \"->\", r[\"model_sha256\"])' $IN/bf16/bf16-copy.json" ;;
  index)
    [[ "$IXNODE" == a || "$IXNODE" == c ]] || { echo "IXNODE a or c" >&2; exit 2; }
    if ! on "$IXNODE" "test -f $IX/runs/$NAME/paired-boot-full-vs-ref.json" || ! on c "test -f $IX/audit/m10/out/audit.json"; then
      echo "no bootstrap or audit for $NAME" >&2; exit 3
    fi
    on a "umask 077; mkdir -p $PRIV && chmod 700 /data/dev2/private/release/m10 $PRIV"
    for pair in "$IXNODE:$IX/runs/$NAME/paired-boot-full-vs-ref.json:boot-full-vs-ka13ib.json" \
      "$IXNODE:$IX/runs/$NAME/merged/receipt.json:receipt.json" \
      "$IXNODE:$IX/runs/$NAME/merged/kit/index.json:kit-index.json" \
      "$IXNODE:$MD/$CAND-bf16-re51f9881/MODEL_MANIFEST.json:package-manifest.json" \
      "c:$IX/runs/K-a13IB-bf16/merged/receipt.json:base-receipt.json" "c:$IX/audit/m10/out/audit.json:audit.json"; do
      node=${pair%%:*} rest=${pair#*:}; src=${rest%%:*} dst=${rest#*:}
      on a "test ! -e $PRIV/$dst" || { echo "$PRIV/$dst exists" >&2; exit 3; }
      fetch "$node" "$src" "$PRIV/$dst"
      [ "$(on "$node" "sha256sum < $src")" = "$(on a "sha256sum < $PRIV/$dst")" ] || { echo "$dst differs" >&2; exit 3; }
      echo "$dst $(on a "sha256sum < $PRIV/$dst | cut -c1-16")"
    done ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
