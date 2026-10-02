#!/usr/bin/env bash
# Decoder M18 part A (prereg dec-m18-prereg-2026-10-02.md): the no-training candidates, from build to the IX1 package.
# Workstation side (ssh control only); every computation runs on the nodes, in containers; node-to-node copies go
# through node A's or node B's temporary key, never through the workstation. Prints hashes and counts only.
#
# Usage: m18-cand.sh MIRROR_SHA NAME STAGE [NODE] ["GPUS"]
#   build    on the point's node (2B: node B, or node E for points of M18 soups; 0.8B: node A): ops/m16/m16_interp.py build in image dbe5f32b (CPU,
#            --network none) -> /data/dev2/runs/dec/m18/points/<point>/build/<point>; M16's existing points are adopted
#            (their build directory and model SHA-256 are recorded, nothing is rebuilt)
#   ship     2B only: the FP32 point (node B's interpolations, node F / A arm soups) -> node E, and the 2B IX1 base
#            package node A -> node E if absent; for points built on node E, their inputs from their home nodes
#            (node B's key reaches C-F only; 2B points are copied, restaged and run on node E)
#   bf16     WORK node (0.8B: node A, 2B: node E): test_bf16_copy, then v2.release.bf16_copy in image host2 (CPU, --network none)
#   restage  WORK node: v2.eval.ix1.restage onto the tier's IX1 package with the copy's model SHA-256 (identity, loaded
#            count and calibration none checked)
#   push     NODE: the restaged package (and the tier's IX1 base package if absent) node A -> NODE (lists equal)
#   pool     NODE "GPUS": m18_ixpool.py detached for NAME (NAME may be a comma list) over the node's panel
#   status   NODE: shard ends / exit codes of NAME
#   relay    NODE: runs/NAME (merged inputs: shards and launcher records, without triton / home copies) -> node C
#   score    node C: score.sh over the panel the run used
#   boot     node C (CPU, detached): family_delta and the full-panel paired bootstrap vs BASE_RUN (default: the tier's
#            DEV2.0 IX1 run; set BASE_RUN to the current release's run) -> runs/NAME/m18-boot-vs-<BASE_RUN>.json
#   fetch    node C -> the local private folder: small summaries only (compare headline, bootstrap headline)
set -euo pipefail
SHA=${1:?MIRROR_SHA} NAMES=${2:?NAME} STAGE=${3:?STAGE} NODE=${4:-} GPUS=${5:-}
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
NODES=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
addr() { awk -F= -v k="node-$1" '$1 == k { print substr($0, length(k) + 2); exit }' "$NODES"; }
on() { local n=$1; shift; ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 "$(addr "$n")" "$@"; }
KEY="-i /root/.ssh/d2_temp_cd -o BatchMode=yes -o ConnectTimeout=30"
M=/data/dev2/src/$SHA-src_training_decision2
S=$M/src/training/decision2
R=/data/dev2/private/eval/index021/ix1
MD=/data/dev2/models/ix1/dec-m18
P=/data/dev2/runs/dec/m18/points
DEC_IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
IMAGE=decision20-train-fast:host2
IMAGE_ID=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
NAME=${NAMES%%,*}
# name -> tier, the point's node path (FP32), and either "adopt" or "release-path arm-path alpha" (node-local paths)
case "$NAME" in
  M18-2b-*) TIER=2b REF=DEV2.0-2B REV=a53cf66a0d9d492a84b6617b61e7ce35fcd03af0 LOADED=1883930944 SRCNODE=b WORK=e ;;
  M18-08b-*) TIER=08b REF=DEV2.0-0.8B REV=bede7938a8c209c09f27400b79eed57948d6b75e LOADED=753446208 SRCNODE=a WORK=a ;;
  *) echo "bad NAME $NAME" >&2; exit 2 ;;
esac
POINT=${NAME#M18-}; POINT=${POINT%-bf16}
D=/data/dev2/runs/dec
case "$POINT" in
  2b-RAUP-a75 | 2b-RAUP-a50 | 08b-RAUP-a75 | 08b-RAUP-a50) BUILD=adopt FP32=$D/m16/points/$POINT/build/$POINT ;;
  2b-UPRA) BUILD="$D/m16/inputs/arms/2b-RA $D/m14/soup/2b-RAUP/build/2b-RAUP-soup 0.5" ;;
  2b-UPRAa75) BUILD="$D/m16/points/2b-RA-a75/build/2b-RA-a75 $D/m14/soup/2b-RAUP/build/2b-RAUP-soup 0.5" ;;
  08b-UPRA) BUILD="$D/m16/inputs/arms/08b-RA $D/m14/soup/08b-RAUP/build/08b-RAUP-soup 0.5" ;;
  08b-UPRAa75) BUILD="$D/m16/points/08b-RA-a75/build/08b-RA-a75 $D/m14/soup/08b-RAUP/build/08b-RAUP-soup 0.5" ;;
  2b-RS17UP | 2b-RAUPM) BUILD=soup SRCNODE=f FP32=$D/m18/soup/$POINT/build/$POINT-soup ;;
  2b-RAM) BUILD=soup SRCNODE=a FP32=$D/m18/soup/$POINT/build/$POINT-soup ;;
  2b-SWRA) BUILD="$D/m16/inputs/arms/2b-RA $D/m18/soup/2b-RS17UP/build/2b-RS17UP-soup 0.5" SRCNODE=e ;;
  2b-UPRAM) BUILD="$D/m18/soup/2b-RAM/build/2b-RAM-soup $D/m18/soup/2b-RAUPM/build/2b-RAUPM-soup 0.5" SRCNODE=e ;;
  08b-RAM-a75) BUILD="$D/m14/inputs/refs/DEV2.0-0.8B/$REV $D/m18/soup/08b-RAM/build/08b-RAM-soup 0.75" ;;
  2b-U5) BUILD="multi $D/m14/soup/2b-RAUP/build/2b-RAUP-soup $D/m16/inputs/arms/2b-RA $D/m18/soup/2b-RS17UP/build/2b-RS17UP-soup $D/m18/soup/2b-RAUPM/build/2b-RAUPM-soup $D/m18/soup/2b-RAM/build/2b-RAM-soup" SRCNODE=e ;;
  2b-U4) BUILD="multi $D/m14/soup/2b-RAUP/build/2b-RAUP-soup $D/m16/inputs/arms/2b-RA $D/m18/soup/2b-RAUPM/build/2b-RAUPM-soup $D/m18/soup/2b-RAM/build/2b-RAM-soup" SRCNODE=e ;;
  2b-MLRAM) BUILD="$D/m18/soup/2b-RAM/build/2b-RAM-soup $D/m15/soup/2b-RASDML/build/2b-RASDML-soup 0.5" SRCNODE=e ;;
  2b-U3ML) BUILD="multi $D/m15/soup/2b-RASDML/build/2b-RASDML-soup $D/m18/soup/2b-RAM/build/2b-RAM-soup $D/m16/inputs/arms/2b-RASD" SRCNODE=e ;;
  08b-RRM) BUILD="multi $D/m16/inputs/arms/08b-RA $D/m18/soup/08b-RAM/build/08b-RAM-soup" ;;
  08b-RRM-a75) BUILD="$D/m14/inputs/refs/DEV2.0-0.8B/$REV $P/08b-RRM/build/08b-RRM 0.75" ;;
  *) echo "no recipe for $POINT" >&2; exit 2 ;;
esac
[[ "$BUILD" == adopt || "$BUILD" == soup ]] || FP32=$P/$POINT/build/$POINT
home() {  # <node path>: the node an input of an M18 point lives on
  case $1 in */m16/inputs/arms/2b-* | */m14/soup/2b-RAUP/*) echo b ;; */m18/soup/2b-RAM/*) echo a ;; */m18/soup/* | */m15/soup/*) echo f ;; *) echo "$SRCNODE" ;; esac
}
PKG=$MD/$NAME-r${REV:0:8} CK=$MD/$NAME-ckpt BASEPKG=/data/dev2/models/ix1/$REF-${REV:0:8}
sums() {
  local skip=""
  [ "${2:-}" = lean ] && skip="-not -path '*/triton/*' -not -path '*/home/*'"
  printf '%s' "cd '$1' && find . -type f $skip | LC_ALL=C sort | xargs -d '\n' -P 8 -n 4 sha256sum | sort -k2"
}
hop() {  # <from> <to> <parent> <name> [lean]: tar stream between nodes through node A's key (never the workstation)
  local from=$1 to=$2 parent=$3 item=$4 lean=${5:-} a b ex="" src dst
  [ "$lean" = lean ] && ex="--exclude=*/triton --exclude=*/home"
  on "$to" "test ! -e '$parent/$item'" || { echo "$parent/$item exists on node $to" >&2; exit 3; }
  src="tar -C '$parent' $ex -cf - '$item'"
  dst="umask 022; mkdir -p '$parent/.m18-part-$item' && tar -C '$parent/.m18-part-$item' -xf - && mv -T '$parent/.m18-part-$item/$item' '$parent/$item' && rmdir '$parent/.m18-part-$item'"
  # node A / B hold the key authorized on C-F (not on each other): a copy starts on A / B or relays through A
  if [ "$from" = a ] || [ "$from" = b ]; then on "$from" "$src | ssh $KEY $(addr "$to") \"$dst\""
  elif [ "$to" = a ] || [ "$to" = b ]; then on "$to" "ssh $KEY $(addr "$from") \"$src\" | bash -c \"$dst\""
  else on a "ssh $KEY $(addr "$from") \"$src\" | ssh $KEY $(addr "$to") \"$dst\""; fi
  a=$(on "$from" "$(sums "$parent/$item" "$lean")") b=$(on "$to" "$(sums "$parent/$item" "$lean")")
  [ -n "$a" ] && [ "$a" = "$b" ] || { echo "$item: node $to copy differs from node $from" >&2; exit 3; }
  echo "$item node $from -> node $to: $(wc -l <<< "$a") files, SHA-256 lists equal"
}
case "$STAGE" in
  build)
    on "$SRCNODE" "test -f $S/v2/dec/ops/m16/m16_interp.py" || { echo "mirror $SHA is not on node $SRCNODE" >&2; exit 2; }
    [[ "$BUILD" == soup ]] && { on "$SRCNODE" "cat ${FP32%/build/*}/DONE"; exit 0; }
    if [[ "$BUILD" == adopt ]]; then
      on "$SRCNODE" "tail -1 $D/m16/points/$POINT/build.stdout.log"
      [ "$SRCNODE" = b ] && on b "test ! -e $P/$POINT && mkdir -p $P/$POINT/build && cp -al $FP32 $P/$POINT/build/$POINT && echo adopted-from-m16 > $P/$POINT/DONE"
      exit 0
    fi
    on "$SRCNODE" "test ! -e $P/$POINT" || { echo "$POINT already built or building" >&2; exit 3; }
    if [[ "$BUILD" == multi* ]]; then  # uniform FP32 soup of the listed artifacts (v2.dec.soup)
      members=""
      for x in ${BUILD#multi }; do members+=" --member $x"; done
      on "$SRCNODE" "umask 022; mkdir -p $P/$POINT/build && docker run --name m18-build-$POINT --rm --network none --shm-size 16g \
        -e PYTHONPATH=/code:/opt/decision-fla -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
        --mount type=bind,src=$S,dst=/code,readonly --mount type=bind,src=$D,dst=$D,readonly \
        --mount type=bind,src=$P/$POINT/build,dst=/out -w /code $DEC_IMAGE \
        python3 -m v2.dec.soup $members --output /out/$POINT \
        > $P/$POINT/build.stdout.log 2> $P/$POINT/build.stderr.log && echo $SHA > $P/$POINT/DONE && tail -1 $P/$POINT/build.stdout.log"
      exit 0
    fi
    read -r rel arm alpha <<< "$BUILD"
    on "$SRCNODE" "umask 022; mkdir -p $P/$POINT/build && docker run --name m18-build-$POINT --rm --network none --shm-size 16g \
      -e PYTHONPATH=/code:/opt/decision-fla -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
      --mount type=bind,src=$S,dst=/code,readonly --mount type=bind,src=$D,dst=$D,readonly \
      --mount type=bind,src=$P/$POINT/build,dst=/out -w /code $DEC_IMAGE \
      python3 v2/dec/ops/m16/m16_interp.py build --release $rel --arm $arm --alpha $alpha --output /out/$POINT \
      > $P/$POINT/build.stdout.log 2> $P/$POINT/build.stderr.log && echo $SHA > $P/$POINT/DONE && tail -1 $P/$POINT/build.stdout.log" ;;
  ship)
    [ "$TIER" = 2b ] || { echo "ship is for 2B points" >&2; exit 2; }
    if [ "$SRCNODE" = e ]; then  # the build's inputs, each from its home node if node E lacks it
      for x in $BUILD; do
        [[ "$x" == /* ]] || continue
        on e "test -f $x/decision_config.json" && continue
        on e "mkdir -p $(dirname "$x")"
        hop "$(home "$x")" e "$(dirname "$x")" "$(basename "$x")"
      done
      exit 0
    fi
    if [ "$BUILD" = soup ]; then
      on e "mkdir -p $(dirname "$FP32")"
      hop "$SRCNODE" e "$(dirname "$FP32")" "$(basename "$FP32")"
      exit 0
    fi
    on e "mkdir -p $P/$POINT/build /data/dev2/models/ix1"
    on e "test -f $BASEPKG/MODEL_MANIFEST.json" || hop a e "$(dirname "$BASEPKG")" "$(basename "$BASEPKG")"
    hop b e "$P/$POINT/build" "$POINT" ;;
  bf16)
    src=$FP32; [ "$SRCNODE" = b ] && src=$P/$POINT/build/$POINT
    on "$WORK" "test -f $src/decision_config.json" || { echo "no FP32 point $src on node $WORK" >&2; exit 3; }
    on "$WORK" "test ! -e $CK" || { echo "$CK exists" >&2; exit 3; }
    on "$WORK" "test \"\$(docker image inspect -f '{{.Id}}' $IMAGE)\" = $IMAGE_ID" || { echo "image $IMAGE is not $IMAGE_ID" >&2; exit 3; }
    cpu="docker run --rm --network none -e HIP_VISIBLE_DEVICES= -e CUDA_VISIBLE_DEVICES= -e ROCR_VISIBLE_DEVICES= \
      -e PYTHONPATH=$S -v $S:$S:ro -v $src:$src:ro -v $MD:$MD -w $S --entrypoint python3 $IMAGE -B"
    on "$WORK" "umask 022; mkdir -p $MD && $cpu -m unittest v2.release.tests.test_bf16_copy > $MD/$NAME-test_bf16_copy.log 2>&1 && \
      $cpu -m v2.release.bf16_copy --source $src --output $CK --receipt $MD/$NAME-bf16-copy.json > $MD/$NAME-bf16.log 2>&1" \
      || { echo "bf16 copy FAILED (see $MD/$NAME-*.log on node $WORK)" >&2; exit 3; }
    on "$WORK" "python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(\"bf16\", r[\"source_model_sha256\"][:12], \"->\", r[\"model_sha256\"][:12])' $MD/$NAME-bf16-copy.json" ;;
  restage)
    model=$(on "$WORK" "python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[\"model_sha256\"])' $MD/$NAME-bf16-copy.json")
    [[ "$model" =~ ^[0-9a-f]{64}$ ]] || { echo "bad bf16 receipt" >&2; exit 3; }
    on "$WORK" "grep -q '^  \[$NAME\]=\"$REF $REV $PKG\"' $S/v2/eval/ix1/launch.sh" || { echo "no DIAGNOSTIC entry $NAME -> $PKG" >&2; exit 2; }
    on "$WORK" "test ! -e $PKG" || { echo "$PKG exists" >&2; exit 3; }
    on "$WORK" "cd $S && PYTHONPATH=$S python3 -B -m v2.eval.ix1.restage --package $BASEPKG --out $PKG --checkpoint $CK --model-sha256 $model" > /dev/null
    on "$WORK" "python3 - $PKG/MODEL_MANIFEST.json $model $LOADED" << 'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
assert m["identity"]["model_sha256"] == sys.argv[2], "identity"
assert m["parameters"]["loaded"] == int(sys.argv[3]), f"loaded {m['parameters']['loaded']}"
assert m["calibration"] is None
print(f"restaged: identity {sys.argv[2][:12]}, loaded {m['parameters']['loaded']:,}, T = 1")
EOF
    ;;
  push)
    [ -n "$NODE" ] && [ "$NODE" != "$WORK" ] || { echo "push NODE (not $WORK)" >&2; exit 2; }
    on "$NODE" "mkdir -p $MD"
    on "$NODE" "test -f $BASEPKG/MODEL_MANIFEST.json" || hop a "$NODE" "$(dirname "$BASEPKG")" "$(basename "$BASEPKG")"
    hop "$WORK" "$NODE" "$MD" "$(basename "$PKG")" ;;
  pool)
    [ -n "$NODE" ] && [ -n "$GPUS" ] || { echo "pool NODE GPUS" >&2; exit 2; }
    panel=$(on "$NODE" "ls -d $R/panel-* | sort -t- -k2 -n | tail -1 | xargs basename")
    names=${NAMES//,/ }
    timeout 60 ssh -n -o BatchMode=yes "$(addr "$NODE")" "mkdir -p $R/logs && setsid nohup python3 $S/v2/dec/ops/m18/m18_ixpool.py --mirror $M --panel $panel --gpus '$GPUS' $names \
      >> $R/logs/m18-pool-$(date -u +%H%M%S).log 2>&1 < /dev/null & disown; echo pool started on node $NODE GPUs $GPUS panel $panel" ;;
  status)
    for n in ${NAMES//,/ }; do
      on "${NODE:-a}" "cd $R; printf '%s parity=%s ' $n \"\$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[\"pass\"])' parity/$n/parity.json 2>/dev/null || echo -)\"; \
        for d in runs/$n/shard-*; do [ -d \$d ] || continue; printf '%s:%s ' \${d##*-} \"\$(cat \$d/exit_code 2>/dev/null || echo run)\"; done; echo"
    done ;;
  relay)
    [ -n "$NODE" ] || { echo "relay NODE" >&2; exit 2; }
    on c "mkdir -p $R/runs"
    hop "$NODE" c "$R/runs" "$NAME" lean
    on c "printf 'run copied from node $NODE (panel %s), %s\n' \"\$(cat $R/runs/$NAME/m18-panel)\" \$(date -u +%FT%TZ) > $R/runs/$NAME/COPIED-FROM-NODE-$NODE.txt" ;;
  score)
    panel=$(on c "cat $R/runs/$NAME/m18-panel")
    size=2B; [ "$TIER" = 08b ] && size=0.8B
    on c "cd $S && bash v2/eval/ix1/score.sh --src $M --model $NAME --size $size --panel $R/$panel > $R/logs/m18-score-$NAME.log 2>&1; e=\$?; tail -2 $R/logs/m18-score-$NAME.log; exit \$e" ;;
  boot)
    base=${BASE_RUN:-$REF}
    on c "test -f $R/runs/$NAME/merged/results.jsonl && test -f $R/runs/$base/merged/results.jsonl" || { echo "missing merged results" >&2; exit 3; }
    out=$R/runs/$NAME/m18-boot-vs-$base.json
    on c "test ! -e $out" || { echo "$out exists" >&2; exit 3; }
    on c "umask 077; cd $R/.. && PYTHONPATH=$S python3 -m v2.eval.ix1.family_delta --base $base=$R/runs/$base/merged/compare.json \
      --new $NAME=$R/runs/$NAME/merged/compare.json --out $R/runs/$NAME/m18-family-vs-$base.json > /dev/null; \
      setsid nohup bash -c 'CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= PYTHONPATH=$S:\$PWD/kit-19ad28ec nice -n 10 \
      venv/bin/python -m v2.eval.ix1.paired_boot --suite-dir suite-0.2 --base $R/runs/$base/merged/results.jsonl \
      --new $R/runs/$NAME/merged/results.jsonl --external $R/../external/index021-frontier-gap-2026-10-01.json \
      --replicates 2000 --seed 20261002 --workers 24 --out $out; echo \$? > $out.exit' > $out.log 2>&1 < /dev/null &"
    echo "$(date -u +%FT%TZ) $NAME bootstrap vs $base started on node C" ;;
  fetch)
    LOCAL=$HOME/code/decision2-program/private/dec-m18
    (umask 077 && mkdir -p "$LOCAL")
    on c "cd $R/runs/$NAME && python3 - $NAME" << 'EOF' >> "$LOCAL/summary.jsonl"
import glob, json, sys
name = sys.argv[1]
out = {"name": name}
try:
    c = json.load(open("merged/compare.json"))
    out["compare_keys"] = sorted(c)[:12]
    for k in ("headline", "index", "balanced_skill", "score"):
        if k in c:
            out[k] = c[k]
except Exception as e:
    out["compare_error"] = str(e)
for f in sorted(glob.glob("m18-boot-vs-*.json")):
    b = json.load(open(f))
    out[f[:-5]] = b.get("headline")
print(json.dumps(out))
EOF
    tail -1 "$LOCAL/summary.jsonl" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["name"], "fetched (private)")' ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
