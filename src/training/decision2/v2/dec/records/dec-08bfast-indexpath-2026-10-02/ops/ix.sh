#!/usr/bin/env bash
# 0.8B / 2B Index-path releases: the private Index run on exactly the release weights (the v2.release.bf16_copy of a
# winner's FP32 checkpoint), as the 9B release did (lux9b/m9/ix.sh, K-a13IB-bf16), for the card's Index values and
# the release evidence. Run on the workstation; values stay private. The IX node is node C (default), node D
# (IX_NODE=d) or node A (IX_NODE=a); every GPU job uses the one GPU IX_GPU (node C GPU0 never), shards one after
# another, because the other node C / D GPUs hold the 27B M6 and the Index-sweep runs.
#   bf16     node C: the FP32 checkpoint that IX1 staged (dec-indexpath/<point>-ckpt) checked file by file against
#            the formal run's PACKAGE.sha256 (node A), then v2.release.bf16_copy in the scored image (CPU, no
#            network) -> dec-indexpath/<point>-bf16-ckpt and bf16-copy.json
#   push     IX_NODE=a|d: that copy and its receipt node C -> the IX node through node A (SHA-256 lists equal)
#   base     IX_NODE=a|d: the tier's IX1 package node C -> the IX node if it is missing there (SHA-256 lists equal)
#   claim    IX_GPU's owner file on the IX node, when its owner released the GPU ("status=released") and it is
#            idle, becomes this run's eval-ix1 entry (the old entry is kept as owner.before-ixp-<UTC>)
#   stage    v2.eval.ix1.restage of the copy into the tier's IX1 package (DEV2.0-0.8B bede7938 / DEV2.0-2B a53cf66a)
#            on the IX node with the copy's model SHA-256; checks the loaded count and the identity
#   parity   launch.sh parity on IX_GPU (86 requests); parity.json must pass
#   run      launch.sh run: every shard of IX_PANEL (panel-7 on node C, panel-8 on node D, panel-3 on node A: the
#            same rows) on IX_GPU, one after another, with the parity step's frozen cache; then score.sh (dual scoring)
#   pull     IX_NODE=a|d: runs/<name>/merged and the launcher receipts the IX node -> node C through node A
#   boot     node C: paired bootstraps vs the tier's DEV2.0 IX1 run (full panel, and transfer-only without HoVer,
#            When2Call, iSarcasmEval, GSM8K, BPoMP), 2,000 replicates, seed 20261002; family delta; row comparison
#            with the FP32 run of the same point (identical answers expected)
#   release  the IX node's IX_GPU owner file, if this name's runs wrote it -> status released
# Usage: [IX_NODE=a|c|d] IX_GPU=N ix.sh MIRROR_SHA NAME STAGE
#   NAME: M16-08b-RA-a75-bf16 | M16-08b-RASD-a75-bf16 | M16-2b-RASD-a25-bf16
set -euo pipefail
SHA=${1:?MIRROR_SHA} NAME=${2:?NAME} STAGE=${3:?STAGE}
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
IX_NODE=${IX_NODE:-c} IX_GPU=${IX_GPU:-}
case "$IX_NODE" in
  a) PANEL=${IX_PANEL:-panel-3} ;; c) PANEL=${IX_PANEL:-panel-7} ;; d) PANEL=${IX_PANEL:-panel-8} ;;
  *) echo "IX_NODE a, c or d" >&2; exit 2 ;;
esac
case "$STAGE" in claim | parity | run | release)
  [[ "$IX_GPU" =~ ^[1-7]$ ]] || { echo "IX_GPU 1-7 is required for $STAGE" >&2; exit 2; } ;;
esac
NODES=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
A=$(grep '^node-a=' "$NODES" | cut -d= -f2-) C=$(grep '^node-c=' "$NODES" | cut -d= -f2-)
X=$(grep "^node-$IX_NODE=" "$NODES" | cut -d= -f2-)
ona() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$A" "$@"; }
onc() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$C" "$@"; }
onx() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$X" "$@"; }
KEY="-i /root/.ssh/d2_temp_cd -o BatchMode=yes -o ConnectTimeout=20"
case "$NAME" in
  M16-08b-RA-a75-bf16) POINT=08b-RA-a75 SIZE=0.8B REF=DEV2.0-0.8B REV8=bede7938 LOADED=753446208 ;;
  M16-08b-RASD-a75-bf16) POINT=08b-RASD-a75 SIZE=0.8B REF=DEV2.0-0.8B REV8=bede7938 LOADED=753446208 ;;
  M16-2b-RASD-a25-bf16) POINT=2b-RASD-a25 SIZE=2B REF=DEV2.0-2B REV8=a53cf66a LOADED=1883930944 ;;
  *) echo "bad NAME $NAME" >&2; exit 2 ;;
esac
FP32=M16-$POINT
M=/data/dev2/src/$SHA-src_training_decision2
S=$M/src/training/decision2
R=/data/dev2/private/eval/index021/ix1
MD=/data/dev2/models/ix1/dec-indexpath
CK=$MD/$POINT-ckpt BCK=$MD/$POINT-bf16-ckpt PKG=$MD/$POINT-bf16-r$REV8
BASEPKG=/data/dev2/models/ix1/$REF-$REV8
FORMAL=/data/dev2/runs/dec/formal/m16/m16-$POINT
IMAGE=decision20-train-fast:host2
IMAGE_ID=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
EXCLUDE="HoVer When2Call iSarcasmEval GSM8K BPoMP"
sums() { echo "cd $1 && find . -type f | LC_ALL=C sort | xargs -r sha256sum"; }
# copy_from_c DIR: node C's DIR (absent on the IX node) -> the same path on the IX node, through node A.
copy_from_c() {
  local d=$1 a b
  onx "test ! -e $d" || { echo "$d exists on node $IX_NODE" >&2; exit 3; }
  onx "umask 077; mkdir -p $(dirname "$d")"
  if [ "$IX_NODE" = a ]; then
    ona "ssh $KEY $C 'tar -C $(dirname "$d") -cf - $(basename "$d")' | tar -C $(dirname "$d") -xf -"
  else
    ona "ssh $KEY $C 'tar -C $(dirname "$d") -cf - $(basename "$d")' | ssh $KEY $X 'tar -C $(dirname "$d") -xf -'"
  fi
  a=$(onc "$(sums "$d")") b=$(onx "$(sums "$d")")
  [ -n "$a" ] && [ "$a" = "$b" ] || { echo "node $IX_NODE copy of $d differs" >&2; exit 3; }
  echo "$d: $(wc -l <<< "$a") files, SHA-256 lists equal"
}
onx "test -f $S/v2/eval/ix1/launch.sh" || { echo "mirror $SHA is not on node $IX_NODE" >&2; exit 2; }
onx "grep -q '^  \[$NAME\]=\"$REF [0-9a-f]* $PKG\"' $S/v2/eval/ix1/launch.sh" ||
  { echo "mirror $SHA has no DIAGNOSTIC entry $NAME -> $PKG" >&2; exit 2; }
case "$STAGE" in
  bf16)
    onc "test ! -e $BCK" || { echo "$BCK exists: refusing to overwrite" >&2; exit 3; }
    want=$(ona "sed -n 's#^\([0-9a-f]\{64\}\)  ./m6/m16-$POINT/checkpoint/\(.*\)\$#\1  \2#p' $FORMAL/PACKAGE.sha256 | LC_ALL=C sort -k2")
    [ -n "$want" ] || { echo "no checkpoint entries in $FORMAL/PACKAGE.sha256" >&2; exit 3; }
    got=$(onc "cd $CK && find . -type f | sed 's#^\./##' | LC_ALL=C sort | xargs sha256sum")
    [ "$want" = "$got" ] || { echo "node C $CK differs from the formal checkpoint" >&2; exit 3; }
    echo "FP32 checkpoint = the formal run's: $(wc -l <<< "$got") files"
    onc "test \"\$(docker image inspect -f '{{.Id}}' $IMAGE)\" = $IMAGE_ID"
    onc "umask 077; mkdir -p $BCK.receipt && docker run --rm --network none -e HIP_VISIBLE_DEVICES= -e CUDA_VISIBLE_DEVICES= \
      -e ROCR_VISIBLE_DEVICES= -e PYTHONPATH=$S -v $S:$S:ro -v $CK:$CK:ro -v $MD:$MD -w $S --entrypoint python3 $IMAGE \
      -B -m v2.release.bf16_copy --source $CK --output $BCK --receipt $BCK.receipt/bf16-copy.json"
    onc "python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(json.dumps({k: r.get(k) for k in (\"source_model_sha256\", \"model_sha256\")}))' $BCK.receipt/bf16-copy.json"
    echo "receipt $(onc "sha256sum < $BCK.receipt/bf16-copy.json | cut -c1-64")" ;;
  push)
    [ "$IX_NODE" != c ] || { echo "push is for IX_NODE=a|d" >&2; exit 2; }
    copy_from_c "$BCK"
    copy_from_c "$BCK.receipt" ;;
  base)
    [ "$IX_NODE" != c ] || { echo "base is for IX_NODE=a|d" >&2; exit 2; }
    if onx "test -f $BASEPKG/MODEL_MANIFEST.json"; then
      a=$(onc "$(sums "$BASEPKG")") b=$(onx "$(sums "$BASEPKG")")
      [ "$a" = "$b" ] || { echo "node $IX_NODE $BASEPKG differs from node C's" >&2; exit 3; }
      echo "$BASEPKG present on node $IX_NODE and equal to node C's"
    else
      copy_from_c "$BASEPKG"
    fi ;;
  claim)
    onx "f=/data/dev2/leases/gpu$IX_GPU.lock/owner; test -f \$f && grep -q '^status=released' \$f || { echo 'gpu$IX_GPU is not released' >&2; exit 3; }; \
      rocm-smi --showuse --showmeminfo vram --json | python3 -c 'import json,sys; c=json.load(sys.stdin)[\"card$IX_GPU\"]; sys.exit(0 if float(c[\"GPU use (%)\"]) <= 5 and int(c[\"VRAM Total Used Memory (B)\"]) < 2 * 2**30 else 1)' \
      || { echo 'gpu$IX_GPU is busy' >&2; exit 3; }; \
      cp -p \$f /data/dev2/leases/gpu$IX_GPU.lock/owner.before-ixp-\$(date -u +%Y%m%dT%H%M%SZ); \
      printf 'track=eval-ix1\nstatus=claimed for the 0.8B / 2B Index-path release run $NAME (the previous owner released it for any track)\nstart_utc=%s\n' \
      \"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\" > \$f; echo gpu$IX_GPU claimed on node $IX_NODE" ;;
  stage)
    model=$(onx "python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[\"model_sha256\"])' $BCK.receipt/bf16-copy.json")
    [[ "$model" =~ ^[0-9a-f]{64}$ ]] || { echo "bad model SHA-256 in the bf16-copy receipt" >&2; exit 3; }
    onx "test ! -e $PKG" || { echo "$PKG exists: refusing to overwrite" >&2; exit 3; }
    onx "cd $S && PYTHONPATH=$S python3 -m v2.eval.ix1.restage --package $BASEPKG --out $PKG --checkpoint $BCK --model-sha256 $model"
    onx "python3 - $PKG/MODEL_MANIFEST.json $model $LOADED" << 'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
assert m["identity"]["model_sha256"] == sys.argv[2], "identity"
assert m["parameters"]["loaded"] == int(sys.argv[3]), f"loaded {m['parameters']['loaded']}"
assert m["calibration"] is None
print(f"restaged: identity {sys.argv[2][:12]}, loaded {m['parameters']['loaded']:,}, T = 1")
EOF
    echo "manifest $(onx "sha256sum < $PKG/MODEL_MANIFEST.json | cut -c1-64")" ;;
  parity)
    onx "test -f $PKG/MODEL_MANIFEST.json" || { echo "run stage first" >&2; exit 3; }
    onx "test ! -e $R/logs/ixp-parity-$NAME.exit" || { echo "parity of $NAME already ran" >&2; exit 3; }
    onx "mkdir -p $R/logs; setsid nohup bash -c 'cd $S && bash v2/eval/ix1/launch.sh parity --src $M --model $NAME --gpu $IX_GPU \
      --run $R/parity/$NAME --rows $R/$PANEL/compat-86.gold-free.jsonl.gz; echo \$? > $R/logs/ixp-parity-$NAME.exit' \
      > $R/logs/ixp-parity-$NAME.log 2>&1 < /dev/null &"
    echo "$(date -u +%FT%TZ) $NAME parity gate started on node $IX_NODE GPU$IX_GPU"
    until onx "test -f $R/logs/ixp-parity-$NAME.exit"; do sleep 30; done
    onx "cat $R/logs/ixp-parity-$NAME.exit; python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({k: d[k] for k in (\"pass\", \"requests\", \"statuses\", \"max_abs_dp\")})); sys.exit(0 if d[\"pass\"] else 1)' $R/parity/$NAME/parity.json" ;;
  run)
    onx "python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))[\"pass\"] else 1)' $R/parity/$NAME/parity.json" ||
      { echo "parity gate has not passed" >&2; exit 3; }
    n=$(onx "ls $R/$PANEL/shard-0-of-*.jsonl.gz | sed 's/.*-of-\([0-9]*\).jsonl.gz/\1/'")
    [[ "$n" =~ ^[0-9]+$ ]] || { echo "no shards in $PANEL" >&2; exit 3; }
    gpus=$(printf "$IX_GPU %.0s" $(seq "$n"))
    for k in $(seq 0 $((n - 1))); do
      if ! onx "test -f $R/runs/$NAME/shard-$k/end_epoch"; then
        onx "test ! -e $R/runs/$NAME/shard-$k" || { echo "shard $k started earlier and has not ended" >&2; exit 3; }
        onx "cd $S && bash v2/eval/ix1/launch.sh run --src $M --model $NAME --gpus '$gpus' --run $R/runs/$NAME \
          --rows-dir $R/$PANEL --cache $R/parity/$NAME/cache-frozen --only $k"
        echo "$(date -u +%FT%TZ) $NAME shard $k of $n started on node $IX_NODE GPU$IX_GPU"
        until onx "test -f $R/runs/$NAME/shard-$k/end_epoch"; do sleep 45; done
      fi
      e=$(onx "cat $R/runs/$NAME/shard-$k/exit_code")
      [ "$e" = 0 ] || { echo "shard $k exit $e" >&2; exit 4; }
    done
    echo "$(date -u +%FT%TZ) $NAME: shards done"
    onx "cd $S && bash v2/eval/ix1/score.sh --src $M --model $NAME --size $SIZE --panel $R/$PANEL > $R/logs/ixp-score-$NAME.log 2>&1; \
      e=\$?; tail -2 $R/logs/ixp-score-$NAME.log; exit \$e" ;;
  pull)
    [ "$IX_NODE" != c ] || { echo "pull is for IX_NODE=a|d" >&2; exit 2; }
    onc "test ! -e $R/runs/$NAME" || { echo "node C already has runs/$NAME" >&2; exit 3; }
    onc "umask 077; mkdir -p $R/runs/$NAME"
    if [ "$IX_NODE" = a ]; then
      ona "cd $R/runs/$NAME && tar -cf - merged launcher-*.json | ssh $KEY $C 'umask 077; tar -C $R/runs/$NAME -xf -'"
    else
      ona "ssh $KEY $X 'cd $R/runs/$NAME && tar -cf - merged launcher-*.json' | ssh $KEY $C 'umask 077; tar -C $R/runs/$NAME -xf -'"
    fi
    a=$(onx "cd $R/runs/$NAME && find merged launcher-*.json -type f | LC_ALL=C sort | xargs sha256sum")
    b=$(onc "cd $R/runs/$NAME && find merged launcher-*.json -type f | LC_ALL=C sort | xargs sha256sum")
    [ -n "$a" ] && [ "$a" = "$b" ] || { echo "node C copy of runs/$NAME differs" >&2; exit 3; }
    onc "printf 'merged/ and launcher receipts copied from node $IX_NODE (run there on $PANEL) through node A, %s\n' \$(date -u +%FT%TZ) > $R/runs/$NAME/COPIED-FROM-NODE-$IX_NODE.txt"
    echo "runs/$NAME: $(wc -l <<< "$a") files, SHA-256 lists equal" ;;
  boot)
    onc "test -f $R/runs/$NAME/merged/results.jsonl" || { echo "no merged results on node C" >&2; exit 3; }
    onc "umask 077; cd $R/.. && PYTHONPATH=$S python3 -m v2.eval.ix1.family_delta --base $REF=$R/runs/$REF/merged/compare.json \
      --new $NAME=$R/runs/$NAME/merged/compare.json --out $R/runs/$NAME/family-delta-vs-ref.json > /dev/null"
    for variant in full transfer; do
      x="" out=paired-boot-full-vs-ref.json
      [ "$variant" = transfer ] && x="--exclude $EXCLUDE" out=paired-boot-vs-ref.json
      onc "test ! -e $R/runs/$NAME/$out" || { echo "$out exists" >&2; exit 3; }
      onc "umask 077; cd $R/.. && setsid nohup bash -c 'CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
        PYTHONPATH=$S:\$PWD/kit-19ad28ec nice -n 10 venv/bin/python -m v2.eval.ix1.paired_boot --suite-dir suite-0.2 \
        --base $R/runs/$REF/merged/results.jsonl --new $R/runs/$NAME/merged/results.jsonl \
        --external $R/../external/index021-frontier-gap-2026-10-01.json --replicates 2000 --seed 20261002 --workers 32 \
        $x --out $R/runs/$NAME/$out; echo \$? > $R/logs/ixp-boot-$NAME-$variant.exit' > $R/logs/ixp-boot-$NAME-$variant.log 2>&1 < /dev/null &"
    done
    echo "$(date -u +%FT%TZ) $NAME: bootstraps started"
    onc "python3 - $R/runs/$FP32/merged/results.jsonl $R/runs/$NAME/merged/results.jsonl" << 'EOF'
import json, sys
def rows(path):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            answers = (r.get("response") or {}).get("answers")
            out[r["run_id"]] = (r["status"], json.dumps(answers, sort_keys=True))
    return out
a, b = rows(sys.argv[1]), rows(sys.argv[2])
same = sum(1 for k in a if b.get(k) == a[k])
choices = sum(
    1 for k in a if k in b and a[k][0] == b[k][0]
    and {q: v.get("choice") for q, v in (json.loads(a[k][1]) or {}).items()}
    == {q: v.get("choice") for q, v in (json.loads(b[k][1]) or {}).items()}
)
print(json.dumps({"fp32_rows": len(a), "bf16_rows": len(b), "shared_ids": len(set(a) & set(b)),
                  "identical_status_and_answers": same, "identical_choices": choices}))
EOF
    until onc "test -f $R/logs/ixp-boot-$NAME-full.exit && test -f $R/logs/ixp-boot-$NAME-transfer.exit"; do sleep 60; done
    onc "cat $R/logs/ixp-boot-$NAME-full.exit $R/logs/ixp-boot-$NAME-transfer.exit; sha256sum $R/runs/$NAME/paired-boot-full-vs-ref.json $R/runs/$NAME/paired-boot-vs-ref.json | cut -c1-64" ;;
  release)
    onx "docker ps --format '{{.Names}}' | grep -q '^ix1-$(tr 'A-Z.' 'a-z_' <<< "$NAME")-'" &&
      { echo "a $NAME Index container is still running" >&2; exit 3; }
    onx "f=/data/dev2/leases/gpu$IX_GPU.lock/owner; if grep -q 'IX1 .* $NAME\$' \$f 2>/dev/null; then \
      printf 'track=eval-ix1\nstatus=released (0.8B / 2B Index-path release run of $NAME done)\nlast_job_end_utc=%s\n' \
      \"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\" > \$f; echo gpu$IX_GPU released; fi" ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
