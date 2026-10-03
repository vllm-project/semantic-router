#!/usr/bin/env bash
# Decision-2.0-Nox-4B M17 4b-SDMLxALL release (user decision 2026-10-02 23:02 UTC+8; decoder M17): release inputs onto
# node A (workstation side), as the SDML release's inputs_sdml.sh. Node A reaches node C / D but not node F, so
# node-F directories go through node B and a node-C transit directory; per-file SHA-256 lists are compared at both ends.
#   bf16      node F models/ix1/dec-m17/DEV2.0-4B-SDMLxALL-bf16-ckpt (+ its bf16-copy.json; the BF16 copy the Index
#             run scored) -> node A /data/dev2/runs/release/inputs/dev2-4b-xall/bf16/{checkpoint,bf16-copy.json}
#   index     node C: the full-panel paired bootstrap vs IS-4b-LHA10SDML-bf16 (the current revision's weights), both
#             IX1 run receipts, the candidate's kit index.json and the row-level contamination audit of the members'
#             TRAIN files (m17-index.sh audit5); node F: the restaged package's MODEL_MANIFEST.json
#             -> node A /data/dev2/private/release/4bif/XALL/ (mode 700; values stay private)
# The autotune cache is the current revision's frozen phase A cache on node A (release/triton/runtime-a-4b).
# Usage: inputs_xall.sh bf16|index
set -euo pipefail
STAGE=${1:?STAGE}
NODES=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
addr() { awk -F= -v k="node-$1" '$1 == k { print substr($0, length(k) + 2); exit }' "$NODES"; }
on() { local n=$1; shift; ssh -o BatchMode=yes -o ConnectTimeout=30 "$(addr "$n")" "$@"; }
KEY="-i /root/.ssh/d2_temp_cd -o BatchMode=yes -o ConnectTimeout=30"
FR=/data/dev2/runs/dec/formal
IX=/data/dev2/private/eval/index021/ix1
NAME=DEV2.0-4B-SDMLxALL-bf16 BASE=IS-4b-LHA10SDML-bf16
MD=/data/dev2/models/ix1/dec-m17
BF=$MD/$NAME-ckpt BFR=$MD/$NAME-bf16-copy.json MNODE=f
PKGM=$MD/$NAME-r13d42143/MODEL_MANIFEST.json
BOOT=$IX/runs/$NAME/m17-boot-full-vs-sdml.json
IN=/data/dev2/runs/release/inputs/dev2-4b-xall
PRIV=/data/dev2/private/release/4bif/XALL
TRANSIT=/data/dev2/runs/release/4bif-relay
sums() { printf '%s' "cd '$1' && find . -type f | LC_ALL=C sort | xargs -d '\n' -r -P 8 -n 8 sha256sum | sort -k2"; }
same() { # same <node> <dir> <node> <dir>: per-file SHA-256 lists equal
  local a b
  a=$(on "$1" "$(sums "$2")") b=$(on "$3" "$(sums "$4")")
  [ -n "$a" ] && [ "$a" = "$b" ] || { echo "$2 on node $1 differs from $4 on node $3" >&2; exit 3; }
  echo "$(wc -l <<< "$a") files equal: node $1 $2 = node $3 $4"
}
fetch() { # fetch <node> <src> <node-A dst>: node C / D directly, node F through node B and a node-C transit directory
  if [ "$1" = f ]; then
    local stamp x=x; stamp=$(date -u +%Y%m%dT%H%M%SZ)-$$
    [ "${2%/}" != "$2" ] && x=x/
    on c "mkdir -p $TRANSIT/$stamp"
    on b "mkdir -p $TRANSIT/$stamp && rsync -a -e 'ssh $KEY' $(addr f):$2 $TRANSIT/$stamp/$x && \
      rsync -a -e 'ssh $KEY' $TRANSIT/$stamp/ $(addr c):$TRANSIT/$stamp/ && rm -rf $TRANSIT/$stamp"
    on a "umask 077; rsync -a -e 'ssh $KEY' $(addr c):$TRANSIT/$stamp/$x $3"
    on c "rm -rf $TRANSIT/$stamp"
  else
    on a "umask 077; rsync -a -e 'ssh $KEY' $(addr "$1"):$2 $3"
  fi
}
case "$STAGE" in
  bf16)
    on a "test ! -e $IN/bf16" || { echo "$IN/bf16 exists" >&2; exit 3; }
    on a "mkdir -p $IN/bf16"
    fetch "$MNODE" "$BF/" "$IN/bf16/checkpoint/"
    fetch "$MNODE" "$BFR" "$IN/bf16/bf16-copy.json"
    on a "chmod -R go+rX $IN/bf16"
    same "$MNODE" "$BF" a "$IN/bf16/checkpoint"
    [ "$(on "$MNODE" "sha256sum < $BFR")" = "$(on a "sha256sum < $IN/bf16/bf16-copy.json")" ] ||
      { echo "bf16 receipt differs" >&2; exit 3; }
    on a "python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(\"bf16 copy\", r[\"source_model_sha256\"][:12], \"->\", r[\"model_sha256\"])' $IN/bf16/bf16-copy.json" ;;
  index)
    { on c "test -f $BOOT" && on c "test -f $IX/audit/m17xall/out/audit.json"; } || { echo "no bootstrap or audit for $NAME" >&2; exit 3; }
    on a "umask 077; mkdir -p $PRIV && chmod 700 /data/dev2/private/release/4bif $PRIV"
    for pair in "c:$BOOT:boot-full-vs-sdml.json" "c:$IX/runs/$NAME/merged/receipt.json:receipt.json" \
      "c:$IX/runs/$BASE/merged/receipt.json:base-receipt.json" \
      "c:$IX/runs/$NAME/merged/kit/index.json:kit-index.json" "c:$IX/audit/m17xall/out/audit.json:audit.json" \
      "f:$PKGM:package-manifest.json"; do
      node=${pair%%:*} rest=${pair#*:}; src=${rest%%:*} dst=${rest#*:}
      on a "test ! -e $PRIV/$dst" || { echo "$PRIV/$dst exists" >&2; exit 3; }
      fetch "$node" "$src" "$PRIV/$dst"
      [ "$(on "$node" "sha256sum < $src")" = "$(on a "sha256sum < $PRIV/$dst")" ] || { echo "$dst differs" >&2; exit 3; }
      echo "$dst $(on a "sha256sum < $PRIV/$dst | cut -c1-16")"
    done ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
