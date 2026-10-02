#!/usr/bin/env bash
# 9B M10 private Index runs (Index-first rule; prereg lux9b-m10-prereg-2026-10-02.md), workstation side (ssh control
# only; every copy is node to node). The Index sweep's driver (index-sweep/ix.sh at 4a133383f) cut down to M10's
# candidates: each candidate is measured once, on its BF16 release copy restaged into the 9B IX1 package
# (DEV2.0-9B-e51f9881, the runtime of the K-a13IB-bf16 reference run; T = 1, calibration none), then paired bootstraps
# vs K-a13IB-bf16, the current Lux-9B release's run. Index values stay in node-private run directories and
# ~/code/decision2-program/private/9b-m10/; this script prints none.
#
# usage: ix.sh MIRROR_SHA STAGE ARGS...
#   env    NODE            the bootstrap environment (index021 venv, suite-0.2, kit-19ad28ec, external) from node A
#   pkg    NODE            copy the 9B IX1 package from node C to NODE (SHA-256 lists equal)
#   ref    NODE            copy K-a13IB-bf16's run (merged results, compare, receipt) from node C to NODE
#   panel  NODE PANEL [FROM]  copy an IX1 panel directory from node FROM (default A) to NODE
#   ship   NAME NODE       node B soup/NAME (an FP32 point of post.sh) -> NODE models/ix1/9b-m10/ckpt/NAME, SHA-256
#                          lists equal; its model SHA-256 (the soup build's) is written next to it
#   bf16   NODE NAME       v2.release.bf16_copy of the shipped point in the scored image (CPU, no network), then
#                          restaged as M10-NAME-bf16 (identity / loaded count / T = 1 checks)
#   lease  NODE "GPUS"     owner files of other tracks' released leases -> track=eval-ix1 idle (old file kept)
#   chain  NODE PANEL "GPUS" NAME...   m10/ixchain.sh detached on NODE (parity gates, runs, scoring, bootstraps)
#   status NODE NAME...    per model: parity verdict, shards, merged rows, bootstraps present
#   fetch  NODE NAME...    small private summaries -> ~/code/decision2-program/private/9b-m10/NAME/ (mode 700)
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
MD=/data/dev2/models/ix1
B9=/data/dev2/runs/9b/m10
BASEPKG=DEV2.0-9B-e51f9881 LOADED=7940895744 REF=K-a13IB-bf16
KEY="ssh -i /root/.ssh/d2_temp_cd -o BatchMode=yes"
LOCAL=${M10_LOCAL:-$HOME/code/decision2-program/private/9b-m10}
sums() { echo "cd '$1' && find . -type f | LC_ALL=C sort | xargs -r -P 8 -n 4 sha256sum | LC_ALL=C sort -k2"; }
copy() {  # SRC_NODE SRC_DIR DST_NODE DST_DIR: whole directory, SHA-256 lists equal on both sides
  local sn=$1 sd=$2 dn=$3 dd=$4 a b
  on "$dn" "test ! -e '$dd'" || { echo "$dd exists on node $dn" >&2; return 3; }
  on "$dn" "mkdir -p '$(dirname "$dd")'"
  case "$sn$dn" in  # node A / B hold the key authorized on C-F; C-F pairs relay through node A
    a[c-f] | b[c-f]) on "$sn" "rsync -a -e '$KEY' '$sd/' '$(host "$dn"):$dd.part/'" ;;
    [c-f]a) on a "rsync -a -e '$KEY' '$(host "$sn"):$sd/' '$dd.part/'" ;;
    [c-f][c-f])
      local relay
      relay=/data/dev2/tmp/m10-relay/$sn$dn-$(basename "$dd")
      on a "test ! -e '$relay' && mkdir -p '$(dirname "$relay")' && rsync -a -e '$KEY' '$(host "$sn"):$sd/' '$relay/' && \
        rsync -a -e '$KEY' '$relay/' '$(host "$dn"):$dd.part/' && rm -rf '$relay'" ;;
    *) echo "no transfer route $sn -> $dn" >&2; return 2 ;;
  esac
  on "$dn" "mv -T '$dd.part' '$dd'"
  a=$(on "$sn" "$(sums "$sd")") b=$(on "$dn" "$(sums "$dd")")
  [ -n "$a" ] && [ "$a" = "$b" ] || { echo "copy on node $dn differs from node $sn" >&2; return 3; }
  echo "copied $(wc -l <<< "$a") files node $sn -> node $dn, SHA-256 lists equal ($(sha256sum <<< "$a" | cut -c1-16))"
}
check_manifest() {  # NODE PKG MODEL
  onin "$1" "python3 - $2/MODEL_MANIFEST.json $3 $LOADED" << 'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
assert m["identity"]["model_sha256"] == sys.argv[2], "identity"
assert m["parameters"]["loaded"] == int(sys.argv[3]), f"loaded {m['parameters']['loaded']}"
assert m["calibration"] is None
print(f"restaged: identity {sys.argv[2][:12]}, loaded {m['parameters']['loaded']:,}, T = 1")
EOF
}

case "$STAGE" in
  env)
    N=${1:?NODE}
    for x in venv suite-0.2 kit-19ad28ec external; do
      on "$N" "test -e $R/../$x" && { echo "node $N has $x"; continue; }
      copy a "$R/../$x" "$N" "$R/../$x"
    done ;;
  pkg)
    N=${1:?NODE}
    on "$N" "test -d $MD/$BASEPKG" && { echo "node $N has $BASEPKG"; exit 0; }
    copy c "$MD/$BASEPKG" "$N" "$MD/$BASEPKG" ;;
  ref)
    N=${1:?NODE}
    [ "$N" != c ] || { echo "node C reads its own runs" >&2; exit 2; }
    on c "test -f $R/runs/$REF/merged/results.jsonl" || { echo "no $REF run on node C" >&2; exit 3; }
    on "$N" "test -f $R/m10/refs/$REF/merged/results.jsonl" && { echo "node $N has $REF"; exit 0; }
    on c "umask 077; rm -rf $R/m10/export/$REF && mkdir -p $R/m10/export/$REF/merged && \
      cp -p $R/runs/$REF/merged/{results.jsonl,compare.json,receipt.json} $R/m10/export/$REF/merged/"
    on "$N" "umask 077; mkdir -p $R/m10/refs"
    copy c "$R/m10/export/$REF" "$N" "$R/m10/refs/$REF"
    on c "rm -rf $R/m10/export/$REF" ;;
  panel)
    N=${1:?NODE} PNAME=${2:?PANEL} FROM=${3:-a}
    on "$N" "test -f $R/$PNAME/panel.json" && { echo "node $N has $PNAME"; exit 0; }
    copy "$FROM" "$R/$PNAME" "$N" "$R/$PNAME" ;;
  ship)
    NAME=${1:?NAME} N=${2:?NODE}
    dir=$(on b "cat $B9/soup/$NAME/DONE")
    model=$(on b "grep -o '\"model_sha256\": *\"[0-9a-f]\{64\}\"' $B9/soup/$NAME/build.stdout.log | tail -1 | grep -o '[0-9a-f]\{64\}'")
    [[ "$model" =~ ^[0-9a-f]{64}$ ]] || { echo "no model SHA-256 in $NAME's build log" >&2; exit 3; }
    copy b "$dir" "$N" "$MD/9b-m10/ckpt/$NAME"
    on "$N" "echo $model > $MD/9b-m10/ckpt/$NAME.model_sha256"
    echo "$NAME shipped to node $N, model ${model:0:12}" ;;
  bf16)
    N=${1:?NODE} NAME=${2:?NAME}
    ck=$MD/9b-m10/ckpt/$NAME out=$MD/9b-m10/bf16/$NAME bpkg=$MD/9b-m10/$NAME-bf16-re51f9881
    on "$N" "test -f $S/v2/release/bf16_copy.py" || { echo "mirror $SHA is not on node $N" >&2; exit 2; }
    on "$N" "test -f '$ck/decision_config.json' && test ! -e '$out' && test ! -e '$bpkg' && test -d $MD/$BASEPKG" \
      || { echo "no shipped point at $ck, no $BASEPKG, or $out / $bpkg exists" >&2; exit 3; }
    want=$(on "$N" "cat $ck.model_sha256")
    on "$N" "test \"\$(docker image inspect -f '{{.Id}}' decision20-train-fast:host2)\" = sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54" \
      || { echo "node $N lacks the scored image" >&2; exit 3; }
    on "$N" "umask 022; mkdir -p $out.receipt && docker run --rm --network none -e HIP_VISIBLE_DEVICES= -e CUDA_VISIBLE_DEVICES= \
      -e ROCR_VISIBLE_DEVICES= -e PYTHONPATH=$S -v $S:$S:ro -v '$ck':'$ck':ro -v $MD/9b-m10/bf16:$MD/9b-m10/bf16 -w $S \
      --entrypoint python3 decision20-train-fast:host2 -B -m v2.release.bf16_copy --source '$ck' --output $out \
      --receipt $out.receipt/bf16-copy.json > /dev/null"
    read -r source_sha model < <(on "$N" "python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r[\"source_model_sha256\"], r[\"model_sha256\"])' $out.receipt/bf16-copy.json")
    [ "$source_sha" = "$want" ] || { echo "the copy's source identity $source_sha is not $NAME's" >&2; exit 3; }
    echo "M10-$NAME-bf16: identity ${source_sha:0:12} -> ${model:0:12}, receipt $(on "$N" "sha256sum < $out.receipt/bf16-copy.json | cut -c1-64")"
    on "$N" "umask 022; cd $S && PYTHONPATH=$S python3 -m v2.eval.ix1.restage --package $MD/$BASEPKG --out $bpkg \
      --checkpoint $out --model-sha256 $model"
    check_manifest "$N" "$bpkg" "$model"
    echo "M10-$NAME-bf16 manifest $(on "$N" "sha256sum < $bpkg/MODEL_MANIFEST.json | cut -c1-64")" ;;
  lease)
    N=${1:?NODE} G=${2:?GPUS}
    on "$N" "stamp=\$(date -u +%Y%m%dT%H%M%SZ); for g in $G; do d=/data/dev2/leases/gpu\$g.lock; mkdir -p \$d; \
      if [ -s \$d/owner ] && ! grep -qx 'track=eval-ix1' \$d/owner; then grep -q '^status=released' \$d/owner || \
      { echo gpu\$g is held by another track; exit 1; }; mv \$d/owner \$d/owner.prev-\$stamp; fi; \
      [ -s \$d/owner ] || printf 'track=eval-ix1\nstatus=idle\npurpose=9B M10 private Index runs (7e1c9ce8)\nstart_utc=%s\n' \
      \"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\" > \$d/owner; echo gpu\$g: \$(head -1 \$d/owner); done" ;;
  chain)
    N=${1:?NODE} PANEL=${2:?PANEL} G=${3:?GPUS}
    shift 3
    if [ "$N" = c ]; then refdir=$R/runs/$REF; else refdir=$R/m10/refs/$REF; fi
    specs=""
    for NAME in "$@"; do specs+=" $NAME=9B=$refdir"; done
    on "$N" "test -f $S/v2/9b/lux9b/m10/ixchain.sh" || { echo "mirror $SHA is not on node $N" >&2; exit 2; }
    on "$N" "mkdir -p $R/logs; setsid nohup bash $S/v2/9b/lux9b/m10/ixchain.sh $M $PANEL '$G'$specs \
      > $R/logs/m10ix-chain-$PANEL.out 2>&1 < /dev/null & echo chain started on node $N" ;;
  status)
    N=${1:?NODE}
    shift
    for NAME in "$@"; do
      onin "$N" "python3 - $R $NAME" << 'EOF'
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
    done
    on "$N" "tail -n 6 $R/logs/m10ix-chain-*.log 2>/dev/null" ;;
  fetch)
    N=${1:?NODE}
    shift
    for NAME in "$@"; do
      d=$LOCAL/$NAME
      (umask 077 && mkdir -p "$d")
      for f in merged/compare.json merged/receipt.json family-delta-vs-ref.json paired-boot-full-vs-ref.json \
        paired-boot-transfer-vs-ref.json; do
        on "$N" "cat $R/runs/$NAME/$f" > "$d/$(basename "$f")" 2> /dev/null || rm -f "$d/$(basename "$f")"
      done
      on "$N" "cat $R/parity/$NAME/parity.json" > "$d/parity.json" 2> /dev/null || rm -f "$d/parity.json"
      chmod 600 "$d"/*.json 2> /dev/null || true
      echo "$NAME -> $d ($(ls "$d" | wc -l) files)"
    done ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
