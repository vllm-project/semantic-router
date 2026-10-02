#!/usr/bin/env bash
# Index sweep (COORDINATION 2026-10-02 10:00; Index-first release rule of 09:55), workstation side: private Jev
# Decision Index 0.2.1 runs of the frozen 0.8B / 2B / 9B breadth candidates with the IX1 harness, each restaged into
# its tier's IX1-scored package (the runtime and package the current release's IX1 run used; T = 1, calibration none),
# then paired bootstraps vs that tier's current-release run. Index values stay in node-private run directories and
# ~/code/decision2-program/private/index-sweep/; this script prints none.
#
# usage: ix.sh MIRROR_SHA STAGE ARGS...
#   env    NODE            the bootstrap environment (index021 venv, suite-0.2, kit-19ad28ec, external) from node A
#   pkgcopy SRC NODE NAME  copy NAME's staged package from node SRC to NODE (SHA-256 lists equal)
#   pkg    NODE TIER       copy the tier's IX1 package from node C to NODE (SHA-256 lists equal)
#   ref    NODE TIER       copy the tier's reference run (merged results, compare, receipt) from node C to
#                          NODE ix1/index-sweep/refs/<REF>/ (node D through a node A relay directory)
#   stage  NODE NAME       the candidate's frozen soup -> NODE (SHA-256 lists equal; used in place when already
#                          there), v2.eval.ix1.restage into the tier package, then identity / loaded count / T = 1 checks;
#                          ISWEEP_SOUP_FROM=node:dir reads a verified relay copy instead of the table's soup
#   panel  NODE PANEL [FROM]  copy an IX1 panel directory from node FROM (default A) to NODE (SHA-256 lists equal)
#   relay  NAME NODE       copy the table's soup to NODE models/ix1/index-sweep/relay/NAME (for a two-hop route)
#   bf16   NODE NAME       the release weights: v2.release.bf16_copy of NAME's FP32 soup on NODE in the scored image
#                          (CPU, no network) -> models/ix1/index-sweep/bf16/NAME (+ .receipt/bf16-copy.json), then
#                          restaged as NAME-bf16 with the copy's model SHA-256 (identity / loaded count / T = 1 checks)
#   lease  NODE "GPUS"     owner files of other tracks' released leases -> track=eval-ix1 idle (old file kept)
#   chain  NODE PANEL "GPUS" NAME...   chain.sh detached on NODE (parity gates, runs, scoring, bootstraps)
#   status NODE NAME...    per model: parity verdict, shards, merged rows, bootstraps present; chain log tail
#   fetch  NODE NAME...    private outputs -> ~/code/decision2-program/private/index-sweep/NAME/ (mode 700)
# Node C keeps the reference runs; elsewhere they are read from ix1/index-sweep/refs/.
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
onin() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$(host "$1")" "${@:2}"; }  # NODE CMD with this stdin
M=/data/dev2/src/$SHA-src_training_decision2
S=$M/src/training/decision2
R=/data/dev2/private/eval/index021/ix1
MD=/data/dev2/models/ix1
KEY="ssh -i /root/.ssh/d2_temp_cd -o BatchMode=yes"
LOCAL=${ISWEEP_LOCAL:-$HOME/code/decision2-program/private/index-sweep}

declare -A TIER=(
  [IS-08b-RASD]=0.8B [IS-08b-RAUP]=0.8B [IS-08b-RASDML]=0.8B
  [IS-2b-RA]=2B [IS-2b-RASD]=2B [IS-2b-RAUP]=2B [IS-2b-RA-a75]=2B
  [IS-L9IB]=9B [IS-K-a12IB]=9B
  [IS-K-a13IBX]=9B [IS-L9IBX]=9B [IS-4b-LHA10SDML]=4B
)
declare -A SOUP=(  # node:directory of the frozen FP32 soup (or a copy whose content manifest equals the soup's)
  [IS-08b-RASD]=a:/data/dev2/runs/dec/m16/inputs/arms/08b-RASD
  [IS-08b-RAUP]=a:/data/dev2/runs/dec/m14/soup/08b-RAUP/build/08b-RAUP-soup
  [IS-08b-RASDML]=e:/data/dev2/runs/dec/m15/soup/08b-RASDML/build/08b-RASDML-soup
  [IS-2b-RA]=b:/data/dev2/runs/dec/m16/inputs/arms/2b-RA
  [IS-2b-RASD]=b:/data/dev2/runs/dec/m16/inputs/arms/2b-RASD
  [IS-2b-RAUP]=b:/data/dev2/runs/dec/m14/soup/2b-RAUP/build/2b-RAUP-soup
  [IS-2b-RA-a75]=b:/data/dev2/runs/dec/m16/points/2b-RA-a75/build/2b-RA-a75
  [IS-L9IB]=a:/data/dev2/runs/9b/m9/soup/L9IB/build/L9IB-soup
  [IS-K-a12IB]=a:/data/dev2/runs/9b/m9/soup/K-a12IB/build/K-a12IB
  [IS-K-a13IBX]=a:/data/dev2/runs/9b/m9/soup/K-a13IBX/build/K-a13IBX
  [IS-L9IBX]=a:/data/dev2/runs/9b/m9/soup/L9IBX/build/L9IBX-soup
  [IS-4b-LHA10SDML]=f:/data/dev2/runs/dec/m15/soup/4b-LHA10SDML/build/4b-LHA10SDML-soup
)
declare -A MODEL=(  # model_sha256 from each soup's build log (the identity its development readouts carry)
  [IS-08b-RASD]=02c170864d0c00895236e2c0c5ee592f38d8360878c53a56da875197ead0e8de
  [IS-08b-RAUP]=0bf0401de81e82e1a3d4c55738986e9599714b4b9403e4d03ddf3b9de3f3161e
  [IS-08b-RASDML]=db5cccad9d17b9388f76dd35bc5c06b09b3b2be8d7ddf2b8aef568058b126342
  [IS-2b-RA]=3d4fde06c4b6a8043df9674c6b3a0c883e52df08cfd526be867ae3143ea5dfef
  [IS-2b-RASD]=341c2bd2873731a24a2df49e8c0bd7da59a1d23aa9429e0757235174890ca8a3
  [IS-2b-RAUP]=37c85fffb548e94c7a292ec4bc59915b3791d3c1dc9836948911fe753bc8e84b
  [IS-2b-RA-a75]=bbba9fad97c8fd0434f97fb641e68a14023f12cfc9f64addd555b027bfe80cdd
  [IS-L9IB]=d7c48f9a869a795cb76d2c64d633cc58af91474c94db4577ec82466af189625b
  [IS-K-a12IB]=68fed4cb24ecefb9499f4b532a4a34df621ab89138bc03f2086965f5dc0c35d0
  [IS-K-a13IBX]=d559b85c2254c5fd77c7e207b02f157b0174cbecf877b1a53d09ee36ff00abc4
  [IS-L9IBX]=695c0ce2cdece30482ee2fad287caf3a101d78c062351bbb3b8d20ee0dbc5b5f
  [IS-4b-LHA10SDML]=1b51567523426b896ae50afeefca4f133c2e3c220c349ebfb9b9c630600dd19d
)
declare -A BASEPKG=([0.8B]=DEV2.0-0.8B-bede7938 [2B]=DEV2.0-2B-a53cf66a [4B]=DEV2.0-4B-13d42143 [9B]=DEV2.0-9B-e51f9881)
declare -A LOADED=([0.8B]=753446208 [2B]=1883930944 [4B]=4208383488 [9B]=7940895744)
declare -A REF=([0.8B]=DEV2.0-0.8B [2B]=DEV2.0-2B [4B]=DEV2.0-4B-LH [9B]=K-a13IB-bf16)
sums() { echo "cd '$1' && find . -type f | LC_ALL=C sort | xargs -r -P 8 -n 4 sha256sum | LC_ALL=C sort -k2"; }
pkgdir() {  # NODE NAME -> package directory of the mirror's launch.sh entry
  on "$1" "grep -o '^  \[$2\]=\"[^\"]*\"' $S/v2/eval/ix1/launch.sh" | sed -E 's/.* ([^ ]+)"$/\1/'
}
copy() {  # SRC_NODE SRC_DIR DST_NODE DST_DIR: whole directory, SHA-256 lists equal on both sides
  local sn=$1 sd=$2 dn=$3 dd=$4 a b
  on "$dn" "test ! -e '$dd'" || { echo "$dd exists on node $dn" >&2; return 3; }
  on "$dn" "mkdir -p '$(dirname "$dd")'"
  case "$sn$dn" in  # node A / B hold the key authorized on C-F; C-F pairs relay through node A
    a[c-f] | b[c-f]) on "$sn" "rsync -a -e '$KEY' '$sd/' '$(host "$dn"):$dd.part/'" ;;
    [c-f]a) on a "rsync -a -e '$KEY' '$(host "$sn"):$sd/' '$dd.part/'" ;;
    [c-f][c-f])
      local relay
      relay=/data/dev2/tmp/isweep-relay/$sn$dn-$(basename "$dd")
      on a "test ! -e '$relay' && mkdir -p '$(dirname "$relay")' && rsync -a -e '$KEY' '$(host "$sn"):$sd/' '$relay/' && \
        rsync -a -e '$KEY' '$relay/' '$(host "$dn"):$dd.part/' && rm -rf '$relay'" ;;
    *) echo "no transfer route $sn -> $dn" >&2; return 2 ;;
  esac
  on "$dn" "mv -T '$dd.part' '$dd'"
  a=$(on "$sn" "$(sums "$sd")") b=$(on "$dn" "$(sums "$dd")")
  [ -n "$a" ] && [ "$a" = "$b" ] || { echo "copy on node $dn differs from node $sn" >&2; return 3; }
  echo "copied $(wc -l <<< "$a") files node $sn -> node $dn, SHA-256 lists equal ($(sha256sum <<< "$a" | cut -c1-16))"
}

case "$STAGE" in
  env)
    N=${1:?NODE}
    for x in venv suite-0.2 kit-19ad28ec external; do
      on "$N" "test -e $R/../$x" && { echo "node $N has $x"; continue; }
      copy a "$R/../$x" "$N" "$R/../$x"
    done ;;
  pkgcopy)
    SN=${1:?SRC_NODE} N=${2:?NODE} NAME=${3:?NAME}
    pkg=$(pkgdir "$SN" "$NAME")
    if [[ "$pkg" != $MD/index-sweep/* ]] || ! on "$SN" "test -f $pkg/MODEL_MANIFEST.json"; then
      echo "no staged $NAME package on node $SN" >&2
      exit 3
    fi
    copy "$SN" "$pkg" "$N" "$pkg" ;;
  panel)
    N=${1:?NODE} PNAME=${2:?PANEL} FROM=${3:-a}
    on "$N" "test -f $R/$PNAME/panel.json" && { echo "node $N has $PNAME"; exit 0; }
    copy "$FROM" "$R/$PNAME" "$N" "$R/$PNAME" ;;
  relay)
    NAME=${1:?NAME} N=${2:?NODE}
    src=${SOUP[$NAME]:?unknown NAME}
    copy "${src%%:*}" "${src#*:}" "$N" "$MD/index-sweep/relay/$NAME" ;;
  pkg)
    N=${1:?NODE} T=${2:?TIER}
    on "$N" "test -d $MD/${BASEPKG[$T]}" && { echo "node $N has ${BASEPKG[$T]}"; exit 0; }
    copy c "$MD/${BASEPKG[$T]}" "$N" "$MD/${BASEPKG[$T]}" ;;
  ref)
    N=${1:?NODE} T=${2:?TIER} ref=${REF[${2:-}]}
    [ "$N" != c ] || { echo "node C reads its own runs" >&2; exit 2; }
    on c "test -f $R/runs/$ref/merged/results.jsonl" || { echo "no $ref run on node C" >&2; exit 3; }
    on "$N" "test -f $R/index-sweep/refs/$ref/merged/results.jsonl" && { echo "node $N has $ref"; exit 0; }
    on c "umask 077; rm -rf $R/index-sweep/export/$ref && mkdir -p $R/index-sweep/export/$ref/merged && \
      cp -p $R/runs/$ref/merged/{results.jsonl,compare.json,receipt.json} $R/index-sweep/export/$ref/merged/"
    on "$N" "umask 077; mkdir -p $R/index-sweep/refs"
    copy c "$R/index-sweep/export/$ref" "$N" "$R/index-sweep/refs/$ref"
    on c "rm -rf $R/index-sweep/export/$ref" ;;
  stage)
    N=${1:?NODE} NAME=${2:?NAME}
    T=${TIER[$NAME]:?unknown NAME} src=${ISWEEP_SOUP_FROM:-${SOUP[$NAME]}} model=${MODEL[$NAME]}
    sn=${src%%:*} sd=${src#*:}
    on "$N" "test -f $S/v2/eval/ix1/restage.py" || { echo "mirror $SHA is not on node $N" >&2; exit 2; }
    pkg=$(pkgdir "$N" "$NAME")
    [[ "$pkg" == $MD/index-sweep/* ]] || { echo "mirror $SHA has no launch.sh entry for $NAME" >&2; exit 2; }
    on "$N" "test ! -e $pkg" || { echo "$pkg exists on node $N: refusing to overwrite" >&2; exit 3; }
    on "$N" "test -d $MD/${BASEPKG[$T]}" || { echo "node $N lacks ${BASEPKG[$T]}; run pkg first" >&2; exit 3; }
    on "$sn" "test -f '$sd/decision_config.json'" || { echo "no soup at node $sn $sd" >&2; exit 3; }
    echo "$NAME soup content manifest $(on "$sn" "$(sums "$sd")" | sha256sum | cut -c1-16) (node $sn)"
    if [ "$sn" = "$N" ]; then
      ck=$sd
      echo "soup on node $N, used in place"
    else
      ck=$MD/index-sweep/ckpt/$(basename "$pkg")
      copy "$sn" "$sd" "$N" "$ck"
    fi
    on "$N" "umask 022; cd $S && PYTHONPATH=$S python3 -m v2.eval.ix1.restage --package $MD/${BASEPKG[$T]} --out $pkg \
      --checkpoint $ck --model-sha256 $model"
    onin "$N" "python3 - $pkg/MODEL_MANIFEST.json $model ${LOADED[$T]}" << 'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
assert m["identity"]["model_sha256"] == sys.argv[2], "identity"
assert m["parameters"]["loaded"] == int(sys.argv[3]), f"loaded {m['parameters']['loaded']}"
assert m["calibration"] is None
print(f"restaged: identity {sys.argv[2][:12]}, loaded {m['parameters']['loaded']:,}, T = 1")
EOF
    echo "$NAME manifest $(on "$N" "sha256sum < $pkg/MODEL_MANIFEST.json | cut -c1-64")" ;;
  bf16)
    N=${1:?NODE} NAME=${2:?NAME}
    T=${TIER[$NAME]:?unknown NAME} src=${SOUP[$NAME]}
    on "$N" "test -f $S/v2/release/bf16_copy.py" || { echo "mirror $SHA is not on node $N" >&2; exit 2; }
    pkg=$(pkgdir "$N" "$NAME") bpkg=$(pkgdir "$N" "$NAME-bf16")
    [[ "$bpkg" == $MD/index-sweep/* ]] || { echo "mirror $SHA has no launch.sh entry for $NAME-bf16" >&2; exit 2; }
    if [ "${src%%:*}" = "$N" ]; then ck=${src#*:}; else ck=$MD/index-sweep/ckpt/$(basename "$pkg"); fi
    out=$MD/index-sweep/bf16/$NAME
    on "$N" "test -f '$ck/decision_config.json' && test ! -e '$out' && test ! -e '$bpkg'" \
      || { echo "no FP32 soup at $ck, or $out / $bpkg exists" >&2; exit 3; }
    on "$N" "test \"\$(docker image inspect -f '{{.Id}}' decision20-train-fast:host2)\" = sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54" \
      || { echo "node $N lacks the scored image" >&2; exit 3; }
    on "$N" "umask 022; mkdir -p $out.receipt && docker run --rm --network none -e HIP_VISIBLE_DEVICES= -e CUDA_VISIBLE_DEVICES= \
      -e ROCR_VISIBLE_DEVICES= -e PYTHONPATH=$S -v $S:$S:ro -v '$ck':'$ck':ro -v $MD/index-sweep/bf16:$MD/index-sweep/bf16 -w $S \
      --entrypoint python3 decision20-train-fast:host2 -B -m v2.release.bf16_copy --source '$ck' --output $out \
      --receipt $out.receipt/bf16-copy.json > /dev/null"
    read -r source_sha model < <(on "$N" "python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r[\"source_model_sha256\"], r[\"model_sha256\"])' $out.receipt/bf16-copy.json")
    [ "$source_sha" = "${MODEL[$NAME]}" ] || { echo "the copy's source identity $source_sha is not $NAME's" >&2; exit 3; }
    echo "$NAME-bf16: identity ${source_sha:0:12} -> ${model:0:12}, receipt $(on "$N" "sha256sum < $out.receipt/bf16-copy.json | cut -c1-64")"
    on "$N" "umask 022; cd $S && PYTHONPATH=$S python3 -m v2.eval.ix1.restage --package $MD/${BASEPKG[$T]} --out $bpkg \
      --checkpoint $out --model-sha256 $model"
    onin "$N" "python3 - $bpkg/MODEL_MANIFEST.json $model ${LOADED[$T]}" << 'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
assert m["identity"]["model_sha256"] == sys.argv[2], "identity"
assert m["parameters"]["loaded"] == int(sys.argv[3]), f"loaded {m['parameters']['loaded']}"
assert m["calibration"] is None
print(f"restaged: identity {sys.argv[2][:12]}, loaded {m['parameters']['loaded']:,}, T = 1")
EOF
    echo "$NAME-bf16 manifest $(on "$N" "sha256sum < $bpkg/MODEL_MANIFEST.json | cut -c1-64")" ;;
  lease)
    N=${1:?NODE} G=${2:?GPUS}
    on "$N" "stamp=\$(date -u +%Y%m%dT%H%M%SZ); for g in $G; do d=/data/dev2/leases/gpu\$g.lock; mkdir -p \$d; \
      if [ -s \$d/owner ] && ! grep -qx 'track=eval-ix1' \$d/owner; then grep -q '^status=released' \$d/owner || \
      { echo gpu\$g is held by another track; exit 1; }; mv \$d/owner \$d/owner.prev-\$stamp; fi; \
      [ -s \$d/owner ] || printf 'track=eval-ix1\nstatus=idle\npurpose=Index sweep (private Index runs of frozen candidates)\nstart_utc=%s\n' \
      \"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\" > \$d/owner; echo gpu\$g: \$(head -1 \$d/owner); done" ;;
  chain)
    N=${1:?NODE} PANEL=${2:?PANEL} G=${3:?GPUS}
    shift 3
    specs=""
    for NAME in "$@"; do
      T=${TIER[${NAME%-bf16}]:?unknown $NAME}
      if [ "$N" = c ]; then refdir=$R/runs/${REF[$T]}; else refdir=$R/index-sweep/refs/${REF[$T]}; fi
      specs+=" $NAME=$T=$refdir"
    done
    on "$N" "test -f $S/v2/dec/ops/index-sweep/chain.sh" || { echo "mirror $SHA is not on node $N" >&2; exit 2; }
    on "$N" "mkdir -p $R/logs; setsid nohup bash $S/v2/dec/ops/index-sweep/chain.sh $M $PANEL '$G'$specs \
      > $R/logs/isweep-chain-$PANEL.out 2>&1 < /dev/null & echo chain started on node $N" ;;
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
    on "$N" "tail -n 6 $R/logs/isweep-chain-*.log 2>/dev/null" ;;
  fetch)
    N=${1:?NODE}
    shift
    for NAME in "$@"; do
      d=$LOCAL/$NAME
      (umask 077 && mkdir -p "$d")
      for f in merged/compare.json merged/receipt.json merged/port.json merged/latency.json family-delta-vs-ref.json \
        paired-boot-full-vs-ref.json paired-boot-transfer-vs-ref.json isweep-ref.json; do
        on "$N" "cat $R/runs/$NAME/$f" > "$d/$(basename "$f")" 2> /dev/null || rm -f "$d/$(basename "$f")"
      done
      on "$N" "cat $R/parity/$NAME/parity.json" > "$d/parity.json" 2> /dev/null || rm -f "$d/parity.json"
      chmod 600 "$d"/*.json 2> /dev/null || true
      echo "$NAME -> $d ($(ls "$d" | wc -l) files)"
    done ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
