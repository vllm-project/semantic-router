#!/usr/bin/env bash
# Arm factory soups, release-form staging and private Index runs (prereg "Candidates"), workstation side: ssh control
# only, every copy is node to node (M10 ix.sh's routes: A / B hold the key authorized on C-F; C-F pairs and A <-> B
# relay through a third node). Each candidate is measured once, on its BF16 release copy, with the IX1 harness through
# M10's node-side chain (v2/9b/lux9b/m10/ixchain.sh: parity gate, shards, scoring, family delta and the two paired
# bootstraps, 2,000 replicates, seed 20261002) against the tier's current-release run: 4B IS-4b-LHA10SDML-bf16 (node
# C, panel-8), 9B K-a13IB-bf16 (node A copy, panel-7). Index values stay in the node-private runs and
# ~/code/decision2-program/private/arm-factory/; this script prints none.
#
# usage: af-ix.sh MIRROR_SHA STAGE ARGS...
#   soup NODE NAME GPU|- MEMBER...  af-soup.sh detached on NODE (log logs/soup-NAME.log)
#   stage NODE NAME                 af-stage.sh on NODE (BF16 copy + restage, checks)
#   soupcopy NAME FROM TO           a built soup (soup/NAME with its markers) FROM -> TO (SHA-256 lists equal)
#   pkgcopy NAME FROM TO            a staged package AF-NAME-bf16 FROM -> TO (lists equal, manifest identity repeated)
#   lease NODE "GPUS"               arm-factory leases released by their chains (or absent) -> track=eval-ix1 idle
#   chain NODE PANEL "GPUS" SHARDS NAME...  ixchain.sh detached on NODE, M10_SHARDS=SHARDS greedy over GPUS
#   status NODE NAME...             parity, shards, merged, bootstraps
#   fetch NODE NAME...              small private summaries -> private/arm-factory/AF-NAME-bf16/ (mode 700)
set -euo pipefail
SHA=${1:?MIRROR_SHA} STAGE=${2:?STAGE}
shift 2
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
NODES=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
host() { grep "^node-$1=" "$NODES" | cut -d= -f2-; }
on() {  # NODE CMD: stdin closed; retries connection failures (exit 255) only
  local n=$1 h rc i
  shift
  h=$(host "$n")
  [ -n "$h" ] || { echo "unknown node $n" >&2; return 2; }
  for i in 1 2 3 4; do
    ssh -n -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 "$h" "$@" && return 0
    rc=$?
    [ "$rc" = 255 ] || return "$rc"
    sleep $((i * 10))
  done
  return 255
}
onin() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$(host "$1")" "${@:2}"; }
M=/data/dev2/src/$SHA-src_training_decision2
S=$M/src/training/decision2
R=/data/dev2/private/eval/index021/ix1
MD=/data/dev2/models/ix1/af
KEY="ssh -i /root/.ssh/d2_temp_cd -o BatchMode=yes"
LOCAL=${AF_LOCAL:-$HOME/code/decision2-program/private/arm-factory}
size_of() { case $1 in a) echo 9b ;; *) echo 4b ;; esac; }
tag_of() { case $1 in 4b-*) echo r13d42143 ;; *) echo re51f9881 ;; esac; }
sums() { echo "cd '$1' && find . -type f | LC_ALL=C sort | xargs -r -P 8 -n 4 sha256sum | LC_ALL=C sort -k2"; }
copy() {  # SRC_NODE SRC_DIR DST_NODE DST_DIR: whole directory, SHA-256 lists equal on both sides
  local sn=$1 sd=$2 dn=$3 dd=$4 a b relay
  on "$dn" "test ! -e '$dd'" || { echo "$dd exists on node $dn" >&2; return 3; }
  on "$dn" "mkdir -p '$(dirname "$dd")'"
  case "$sn$dn" in
    a[c-f] | b[c-f]) on "$sn" "rsync -a -e '$KEY' '$sd/' '$(host "$dn"):$dd.part/'" ;;
    [c-f]a | [c-f]b) on "$dn" "rsync -a -e '$KEY' '$(host "$sn"):$sd/' '$dd.part/'" ;;
    [c-f][c-f])
      relay=/data/dev2/tmp/af-relay/$sn$dn-$(basename "$dd")
      on a "test ! -e '$relay' && mkdir -p '$(dirname "$relay")' && rsync -a -e '$KEY' '$(host "$sn"):$sd/' '$relay/' && \
        rsync -a -e '$KEY' '$relay/' '$(host "$dn"):$dd.part/' && rm -rf '$relay'" ;;
    ab | ba)
      relay=/data/dev2/tmp/af-relay/$sn$dn-$(basename "$dd")
      on "$sn" "rsync -a -e '$KEY' '$sd/' '$(host c):$relay/'"
      on "$dn" "rsync -a -e '$KEY' '$(host c):$relay/' '$dd.part/'"
      on c "rm -rf '$relay'" ;;
    *) echo "no transfer route $sn -> $dn" >&2; return 2 ;;
  esac
  on "$dn" "mv -T '$dd.part' '$dd'"
  a=$(on "$sn" "$(sums "$sd")") b=$(on "$dn" "$(sums "$dd")")
  [ -n "$a" ] && [ "$a" = "$b" ] || { echo "copy on node $dn differs from node $sn" >&2; return 3; }
  echo "copied $(wc -l <<< "$a") files node $sn -> node $dn, SHA-256 lists equal ($(sha256sum <<< "$a" | cut -c1-16))"
}

case "$STAGE" in
  soup)
    N=${1:?NODE} NAME=${2:?NAME} G=${3:?GPU}
    shift 3
    on "$N" "test -f $S/v2/af/ops/af-soup.sh" || { echo "mirror $SHA is not on node $N" >&2; exit 2; }
    on "$N" "mkdir -p /data/dev2/runs/af/logs && AF_NODE=$N AF_SKIP_UNFINISHED=${AF_SKIP_UNFINISHED:-0} setsid nohup \
      bash $S/v2/af/ops/af-soup.sh $(basename "$M") $NAME $G $* > /data/dev2/runs/af/logs/soup-$NAME.log 2>&1 < /dev/null &"
    echo "soup $NAME launched on node $N" ;;
  stage)
    N=${1:?NODE} NAME=${2:?NAME}
    on "$N" "AF_NODE=$N bash $S/v2/af/ops/af-stage.sh $(basename "$M") $NAME" ;;
  soupcopy)
    NAME=${1:?NAME} FROM=${2:?FROM} N=${3:?TO}
    d=/data/dev2/runs/af/$(size_of "$FROM")/soup/$NAME
    on "$FROM" "test -f $d/DONE" || { echo "no built soup $NAME on node $FROM" >&2; exit 3; }
    copy "$FROM" "$d" "$N" "$d" ;;
  pkgcopy)
    NAME=${1:?NAME} FROM=${2:?FROM} N=${3:?TO}
    p=$MD/AF-$NAME-bf16-$(tag_of "$NAME")
    copy "$FROM" "$p" "$N" "$p"
    [ "$(on "$FROM" "sha256sum < $p/MODEL_MANIFEST.json")" = "$(on "$N" "sha256sum < $p/MODEL_MANIFEST.json")" ] \
      || { echo "manifest differs" >&2; exit 3; } ;;
  lease)
    N=${1:?NODE} G=${2:?GPUS}
    on "$N" "for g in $G; do d=/data/dev2/leases/gpu\$g.lock; mkdir -p \$d; \
      if [ -s \$d/owner ] && ! grep -qx 'track=eval-ix1' \$d/owner; then \
        grep -q '^track=arm-factory' \$d/owner && grep -qE '^status=(released|idle)' \$d/owner \
          || { echo gpu\$g is busy or held by another track; exit 1; }; \
        mv \$d/owner \$d/owner.prev-af-\$(date -u +%Y%m%dT%H%M%SZ); fi; \
      [ -s \$d/owner ] || printf 'track=eval-ix1\nstatus=idle\npurpose=IX1 arm factory private Index runs (COORDINATION 2026-10-02 22:00)\nstart_utc=%s\n' \
        \"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\" > \$d/owner; echo gpu\$g: \$(head -2 \$d/owner | tr '\n' ' '); done" ;;
  chain)
    N=${1:?NODE} PANEL=${2:?PANEL} G=${3:?GPUS} SH=${4:?SHARDS}
    shift 4
    specs=""
    for NAME in "$@"; do
      case $NAME in
        4b-*) [ "$N" = c ] || { echo "4B Index runs are scored on node C" >&2; exit 2; }
              specs+=" AF-$NAME-bf16=4B=$R/runs/IS-4b-LHA10SDML-bf16" ;;
        *) if [ "$N" = c ]; then ref=$R/runs/K-a13IB-bf16; else ref=$R/m10/refs/K-a13IB-bf16; fi
           specs+=" AF-$NAME-bf16=9B=$ref" ;;
      esac
      on "$N" "test -f $MD/AF-$NAME-bf16-$(tag_of "$NAME")/MODEL_MANIFEST.json" || { echo "AF-$NAME-bf16 is not staged on node $N" >&2; exit 3; }
    done
    on "$N" "test -f $S/v2/9b/lux9b/m10/ixchain.sh" || { echo "mirror $SHA is not on node $N" >&2; exit 2; }
    on "$N" "mkdir -p $R/logs; M10_SHARDS=$SH setsid nohup bash $S/v2/9b/lux9b/m10/ixchain.sh $M $PANEL '$G'$specs \
      > $R/logs/af-chain-$PANEL-\$(date -u +%H%M%S).out 2>&1 < /dev/null & echo chain started on node $N" ;;
  status)
    N=${1:?NODE}
    shift
    for NAME in "$@"; do
      onin "$N" "python3 - $R AF-$NAME-bf16" << 'EOF'
import glob, json, os, sys, time
R, m = sys.argv[1:]
p = f"{R}/parity/{m}/parity.json"
par = json.load(open(p))["pass"] if os.path.exists(p) else None
shards, gpuh = [], 0.0
for w in sorted(glob.glob(f"{R}/runs/{m}/shard-*")):
    n = sum(1 for _ in open(f"{w}/results.jsonl")) if os.path.exists(f"{w}/results.jsonl") else 0
    start = float(open(f"{w}/start_epoch").read()) if os.path.exists(f"{w}/start_epoch") else None
    end = float(open(f"{w}/end_epoch").read()) if os.path.exists(f"{w}/end_epoch") else None
    code = open(f"{w}/exit_code").read().strip() if os.path.exists(f"{w}/exit_code") else "-"
    if start:
        gpuh += ((end or time.time()) - start) / 3600
    shards.append(f"{os.path.basename(w)[6:]}:{n}{'/x' + code if end else ''}")
merged = f"{R}/runs/{m}/merged/receipt.json"
rec = json.load(open(merged)) if os.path.exists(merged) else {}
boots = [v for v in ("full", "transfer") if os.path.exists(f"{R}/runs/{m}/paired-boot-{v}-vs-ref.json")]
print(f"{m}: parity {par}; shards {' '.join(shards) or '-'}; run GPU-h {gpuh:.2f}; merged {rec.get('statuses', '-')}; boots {boots}")
EOF
    done ;;
  fetch)
    N=${1:?NODE}
    shift
    for NAME in "$@"; do
      m=AF-$NAME-bf16 d=$LOCAL/AF-$NAME-bf16
      (umask 077 && mkdir -p "$d")
      for f in merged/compare.json merged/receipt.json merged/kit/index.json family-delta-vs-ref.json \
        paired-boot-full-vs-ref.json paired-boot-transfer-vs-ref.json; do
        out=$(basename "$f")
        [ "$f" = merged/kit/index.json ] && out=kit-index.json
        on "$N" "cat $R/runs/$m/$f" > "$d/$out" 2> /dev/null || rm -f "$d/$out"
      done
      on "$N" "cat $R/parity/$m/parity.json" > "$d/parity.json" 2> /dev/null || rm -f "$d/parity.json"
      chmod 700 "$d" && chmod 600 "$d"/*.json 2> /dev/null || true
      echo "$m -> $d ($(ls "$d" | wc -l) files)"
    done ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
