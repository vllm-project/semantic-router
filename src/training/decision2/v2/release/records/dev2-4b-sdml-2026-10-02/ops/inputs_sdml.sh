#!/usr/bin/env bash
# Decision-2.0-Nox-4B M15 4b-LHA10SDML release (user decision 2026-10-02 15:45 UTC+8; decoder M17): release inputs onto
# node A (workstation side), as the Index-first release's inputs4b.sh. Node A reaches node C / D but not node F, so
# node-F directories go through node B and a node-C transit directory; per-file SHA-256 lists are compared at both ends.
#   bf16      node F models/ix1/index-sweep/bf16/IS-4b-LHA10SDML (+ .receipt/bf16-copy.json; the BF16 copy the Index
#             run scored) -> node A /data/dev2/runs/release/inputs/dev2-4b-sdml/bf16/{checkpoint,bf16-copy.json}
#   index     node D: the full-panel cross bootstrap vs M17 4b-LHS17SD (the current revision's weights), the Index
#             run's receipt and kit index.json and the restaged package's MODEL_MANIFEST.json; node C: the current
#             revision's IX1 run receipt and the row-level contamination audit of the M15 TRAIN (m17-index.sh
#             audit3) -> node A /data/dev2/private/release/4bif/SDML/ (mode 700; values stay private)
#   caches    the formal run's and its mlx-diag run's persisted autotune caches (node F) -> node A $IN/{formal,mlx}-cache,
#             each checked against the run's cache-after manifest
# Usage: inputs_sdml.sh bf16|index|caches
set -euo pipefail
STAGE=${1:?STAGE}
NODES=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
addr() { awk -F= -v k="node-$1" '$1 == k { print substr($0, length(k) + 2); exit }' "$NODES"; }
on() { local n=$1; shift; ssh -o BatchMode=yes -o ConnectTimeout=30 "$(addr "$n")" "$@"; }
KEY="-i /root/.ssh/d2_temp_cd -o BatchMode=yes -o ConnectTimeout=30"
FR=/data/dev2/runs/dec/formal
IX=/data/dev2/private/eval/index021/ix1
NAME=IS-4b-LHA10SDML-bf16 RUN=$FR/m17/m17-4b-LHA10SDML RNODE=f
BF=/data/dev2/models/ix1/index-sweep/bf16/IS-4b-LHA10SDML MNODE=f
PKGM=/data/dev2/models/ix1/index-sweep/4b-LHA10SDML-bf16-r13d42143/MODEL_MANIFEST.json
BOOT=$IX/index-sweep/cross/IS-4b-LHA10SDML-bf16-vs-LHS17SD-bf16-full.json
IN=/data/dev2/runs/release/inputs/dev2-4b-sdml
PRIV=/data/dev2/private/release/4bif/SDML
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
    fetch "$MNODE" "$BF.receipt/bf16-copy.json" "$IN/bf16/bf16-copy.json"
    on a "chmod -R go+rX $IN/bf16"
    same "$MNODE" "$BF" a "$IN/bf16/checkpoint"
    [ "$(on "$MNODE" "sha256sum < $BF.receipt/bf16-copy.json")" = "$(on a "sha256sum < $IN/bf16/bf16-copy.json")" ] ||
      { echo "bf16 receipt differs" >&2; exit 3; }
    on a "python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(\"bf16 copy\", r[\"source_model_sha256\"][:12], \"->\", r[\"model_sha256\"])' $IN/bf16/bf16-copy.json" ;;
  index)
    { on d "test -f $BOOT" && on c "test -f $IX/audit/m17sdml/out/audit.json"; } || { echo "no bootstrap or audit for $NAME" >&2; exit 3; }
    on a "umask 077; mkdir -p $PRIV && chmod 700 /data/dev2/private/release/4bif $PRIV"
    for pair in "d:$BOOT:boot-full-vs-s17.json" "d:$IX/runs/$NAME/merged/receipt.json:receipt.json" \
      "c:$IX/runs/DEV2.0-4B-LHS17SD-bf16/merged/receipt.json:base-receipt.json" \
      "d:$IX/runs/$NAME/merged/kit/index.json:kit-index.json" "c:$IX/audit/m17sdml/out/audit.json:audit.json" \
      "d:$PKGM:package-manifest.json"; do
      node=${pair%%:*} rest=${pair#*:}; src=${rest%%:*} dst=${rest#*:}
      on a "test ! -e $PRIV/$dst" || { echo "$PRIV/$dst exists" >&2; exit 3; }
      fetch "$node" "$src" "$PRIV/$dst"
      [ "$(on "$node" "sha256sum < $src")" = "$(on a "sha256sum < $PRIV/$dst")" ] || { echo "$dst differs" >&2; exit 3; }
      echo "$dst $(on a "sha256sum < $PRIV/$dst | cut -c1-16")"
    done ;;
  caches)
    for kind in formal mlx; do
      r=$RUN; [ "$kind" = mlx ] && r=$RUN-mlx
      on a "test ! -e $IN/$kind-cache" || { echo "$IN/$kind-cache exists" >&2; exit 3; }
      want=$(on a "python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[\"cache\"][\"after_manifest_sha256\"])' $r/M5-RECEIPT.json")
      stamp=$(date -u +%Y%m%dT%H%M%SZ)-$$
      on c "mkdir -p $TRANSIT/$stamp"
      if [ "$RNODE" = f ]; then
        on b "mkdir -p $TRANSIT/$stamp && rsync -a -e 'ssh $KEY' $(addr f):$r-cache/ $TRANSIT/$stamp/cache/ && \
          rsync -a -e 'ssh $KEY' $TRANSIT/$stamp/cache/ $(addr c):$TRANSIT/$stamp/cache/ && rm -rf $TRANSIT/$stamp"
      else
        on b "rsync -a -e 'ssh $KEY' $r-cache/ $(addr c):$TRANSIT/$stamp/cache/"
      fi
      on a "mkdir -p $IN && rsync -a -e 'ssh $KEY' $(addr c):$TRANSIT/$stamp/cache/ $IN/$kind-cache/"
      on c "rm -rf $TRANSIT/$stamp"
      got=$(on a "cd $IN/$kind-cache && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | cut -c1-64")
      [ "$got" = "$want" ] || { echo "$kind cache manifest $got is not the run's cache-after $want" >&2; exit 3; }
      on a "cd $IN/$kind-cache && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum > $IN/$kind-cache.sha256"
      echo "$kind cache on node A: manifest $got (= $(basename "$r") cache-after)"
    done ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
