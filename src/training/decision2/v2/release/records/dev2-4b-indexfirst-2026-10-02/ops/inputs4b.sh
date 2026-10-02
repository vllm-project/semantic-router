#!/usr/bin/env bash
# Decision-2.0-Nox-4B Index-first release (COORDINATION 2026-10-02 09:55; worker 5e7b8132): release inputs of the chosen
# candidate onto node A (workstation side). Node A reaches node C but not node B, so node-B directories go through a
# node-C transit directory; per-file SHA-256 lists are compared at both ends and the transit copy is removed.
#   bf16      node C /data/dev2/models/ix1/dec-4bif/NAME-ckpt and NAME-bf16-copy.json (the BF16 copy the Index run
#             scored) -> node A /data/dev2/runs/release/inputs/dev2-4b-4bif-CAND/bf16/{checkpoint,bf16-copy.json}
#   index     node C: the full-panel paired bootstrap, both IX1 run receipts, the candidate's kit index.json, the
#             restaged package's MODEL_MANIFEST.json (identity and loaded count of exactly these weights, for
#             card_index) and the contamination audit -> node A /data/dev2/private/release/4bif/CAND/ (mode 700;
#             values stay private)
#   caches    the formal run's and its mlx-diag run's persisted autotune caches (node B, or node F for M13) -> node A
#             $IN/{formal,mlx}-cache, each checked against the run's cache-after manifest
# Usage: inputs4b.sh CAND bf16|index|caches        CAND: a75 | a50 | UP | SDB | S10 | S17
set -euo pipefail
CAND=${1:?CAND} STAGE=${2:?STAGE}
NODES=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
addr() { awk -F= -v k="node-$1" '$1 == k { print substr($0, length(k) + 2); exit }' "$NODES"; }
on() { local n=$1; shift; ssh -o BatchMode=yes -o ConnectTimeout=30 "$(addr "$n")" "$@"; }
KEY="-i /root/.ssh/d2_temp_cd -o BatchMode=yes -o ConnectTimeout=30"
FR=/data/dev2/runs/dec/formal
IX=/data/dev2/private/eval/index021/ix1
MD=/data/dev2/models/ix1/dec-4bif MNODE=c AUDIT=audit/4bif/out/audit.json BNODE=c BOOT=
case "$CAND" in
  a75) NAME=DEV2.0-4B-LHA10SD-a75-bf16 RUN=$FR/m16/m16-4b-LHA10SD-a75 RNODE=b ;;
  a50) NAME=DEV2.0-4B-LHA10SD-a50-bf16 RUN=$FR/m16/m16-4b-LHA10SD-a50 RNODE=b ;;
  UP) NAME=DEV2.0-4B-LHA10UP-bf16 RUN=$FR/4bif/4bif-4b-LHA10UP RNODE=b ;;
  SDB) NAME=DEV2.0-4B-LHA10SD-bf16 RUN=$FR/m13/m13-4b-LHA10SD RNODE=f ;;
  # decoder M17 (hand-over dec-m17-handover-2026-10-02.md): BF16 copy and restaged package on node F, the audit of
  # both M17 TRAIN files on node C; S17's bootstrap ran on node D on SHA-256-checked copies of the two results files
  S10) NAME=DEV2.0-4B-LHS10SD-bf16 RUN=$FR/m17/m17-4b-LHS10SD RNODE=f MD=/data/dev2/models/ix1/dec-m17 MNODE=f
    AUDIT=audit/m17/out/audit.json ;;
  S17) NAME=DEV2.0-4B-LHS17SD-bf16 RUN=$FR/m17/m17-4b-LHS17SD RNODE=f MD=/data/dev2/models/ix1/dec-m17 MNODE=f
    AUDIT=audit/m17/out/audit.json BNODE=d BOOT=$IX/runs/4bif-bootcopy/DEV2.0-4B-LHS17SD-bf16/4bif-boot-full-vs-lh.json ;;
  *) echo "bad CAND $CAND" >&2; exit 2 ;;
esac
BOOT=${BOOT:-$IX/runs/$NAME/4bif-boot-full-vs-lh.json}
IN=/data/dev2/runs/release/inputs/dev2-4b-4bif-$CAND
PRIV=/data/dev2/private/release/4bif/$CAND
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
    fetch "$MNODE" "$MD/$NAME-ckpt/" "$IN/bf16/checkpoint/"
    fetch "$MNODE" "$MD/$NAME-bf16-copy.json" "$IN/bf16/bf16-copy.json"
    on a "chmod -R go+rX $IN/bf16"
    same "$MNODE" "$MD/$NAME-ckpt" a "$IN/bf16/checkpoint"
    [ "$(on "$MNODE" "sha256sum < $MD/$NAME-bf16-copy.json")" = "$(on a "sha256sum < $IN/bf16/bf16-copy.json")" ] ||
      { echo "bf16 receipt differs" >&2; exit 3; }
    on a "python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(\"bf16 copy\", r[\"source_model_sha256\"][:12], \"->\", r[\"model_sha256\"])' $IN/bf16/bf16-copy.json" ;;
  index)
    { on "$BNODE" "test -f $BOOT" && on c "test -f $IX/$AUDIT"; } || { echo "no bootstrap or audit for $NAME" >&2; exit 3; }
    on a "umask 077; mkdir -p $PRIV && chmod 700 /data/dev2/private/release/4bif $PRIV"
    for pair in "$BNODE:$BOOT:boot-full-vs-lh.json" "c:$IX/runs/$NAME/merged/receipt.json:receipt.json" \
      "c:$IX/runs/DEV2.0-4B-LH/merged/receipt.json:base-receipt.json" "c:$IX/runs/$NAME/merged/kit/index.json:kit-index.json" \
      "c:$IX/$AUDIT:audit.json" "$MNODE:$MD/$NAME-r13d42143/MODEL_MANIFEST.json:package-manifest.json"; do
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
