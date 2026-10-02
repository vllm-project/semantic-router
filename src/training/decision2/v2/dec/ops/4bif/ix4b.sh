#!/usr/bin/env bash
# 4B Index-first release (COORDINATION 2026-10-02 09:55; workstation side): private Decision Index runs of the frozen
# 4B breadth candidates with IX1's harness (v2/eval/ix1: image host2, kit 87d4650b, the 86-request parity gate, dual
# scoring), each restaged into the released LH package DEV2.0-4B 13d42143 (Decision-2.0-Nox-4B's runtime and package;
# T = 1, calibration none) and compared with IX1's run of that package (node C runs/DEV2.0-4B-LH, panel run IDs
# 6455d7be...). A candidate is measured on the weights a release ships: the v2.release.bf16_copy of its frozen FP32
# soup or interpolation point (the M14 / M15 / M16 directories on node B / F, read only), built on the Index node in
# a CPU-only container of the scored image host2. M13 4b-LHA10SD keeps its existing FP32 run (dec M15 part A, node D
# runs/DEV2.0-4B-LHA10SD), imported to node C. Index values stay in the node private directories and in
# ~/code/decision2-program/private/4b-indexfirst/; this script prints none.
#
# Usage: ix4b.sh MIRROR_SHA CAND STAGE [NODE] [GPU | "GPUS"]
#   CAND     UP (M14 4b-LHA10UP soup) | a75 | a50 (M16 4b-LHA10SD-a75 / -a50) | SDML (M15 4b-LHA10SDML soup)
#            | SDB (the M13 4b-LHA10SD soup as BF16) | SD (M13's FP32 run: import and boot only)
#   stage    NODE (c | d): the FP32 source -> NODE /data/dev2/models/ix1/dec-4bif/src/CAND-fp32 over node B's
#            transfer key (SDML: node F -> node B first; per-file SHA-256 lists equal at every hop); on NODE, image
#            host2 without network or GPU: test_bf16_copy, then v2.release.bf16_copy -> NAME-ckpt (receipt
#            NAME-bf16-copy.json; its source fingerprint must be the frozen model SHA-256); then v2.eval.ix1.restage
#            onto DEV2.0-4B-13d42143 with the copy's model SHA-256, checking the identity, the loaded count
#            4,208,383,488 and calibration none
#   parity   NODE GPU: launch.sh parity over the 86 compatibility requests; parity/NAME/parity.json must pass
#   run      NODE "GPUS": node C with 7 GPUs: launch.sh run over panel-7; otherwise lanes.sh over panel-8 (the 8 shards
#            dealt round-robin to the GPUs, each GPU's shards one after another); detached on the node; the panel
#            used is recorded in runs/NAME/4bif-panel for score
#   pool     NODE "GPUS", CAND a comma list: pool.py on NODE, detached: in order, each candidate's parity gate and
#            its panel-8 shards, each dispatched to a GPU of the list that is idle and not held by another job's lease
#   void     NODE: a parity attempt the harness refused before its reference pass (busy or foreign GPU) -> void/
#   status   NODE: per shard records, end and exit code; GPU-h of the current intervals
#   score    NODE: score.sh (merge, port + kit, compare) once every shard ended 0
#   pull     node D -> node C through node B (transfer keys; the workstation link is too slow): merged/ results,
#            compare, port, receipt and kit index (SHA-256 equal on both sides)
#   import   SD only: node D's runs/DEV2.0-4B-LHA10SD/merged -> node C (results SHA-256 = its receipt's)
#   boot     node C, CPU, detached: family_delta vs DEV2.0-4B-LH; the paired bootstrap (v2.eval.ix1.paired_boot,
#            2,000 replicates, seed 20261002) full panel -> runs/NAME/4bif-boot-full-vs-lh.json and transfer-only
#            (--exclude HoVer When2Call iSarcasmEval GSM8K BPoMP) -> runs/NAME/4bif-boot-transfer-vs-lh.json
#   fetch    node C -> the local private folder (mode 700): parity, receipt, compare, port, kit index, family delta
#            and both bootstraps
#   release  NODE: the owner files launch.sh wrote for NAME -> released (no NAME container running)
set -euo pipefail
SHA=${1:?MIRROR_SHA} CAND=${2:?CAND} STAGE=${3:?STAGE} NODE=${4:-} GPU=${5:-}
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
NODES=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
addr() { awk -F= -v k="node-$1" '$1 == k { print substr($0, length(k) + 2); exit }' "$NODES"; }
on() { local n=$1; shift; ssh -o BatchMode=yes -o ConnectTimeout=30 "$(addr "$n")" "$@"; }
if [ "$STAGE" = pool ]; then
  # CAND is a comma list here: every candidate's parity gate and panel-8 shards over the free GPUs of NODE
  [[ "$NODE" == c || "$NODE" == d ]] || { echo "NODE must be c or d" >&2; exit 2; }
  names=()
  for c in ${CAND//,/ }; do names+=("$(bash "$0" "$SHA" "$c" name)"); done
  M=/data/dev2/src/$SHA-src_training_decision2 R=/data/dev2/private/eval/index021/ix1
  on "$NODE" "test -f $M/src/training/decision2/v2/dec/ops/4bif/pool.py" || { echo "mirror $SHA is not on node $NODE" >&2; exit 2; }
  on "$NODE" "mkdir -p $R/logs && setsid nohup python3 $M/src/training/decision2/v2/dec/ops/4bif/pool.py --mirror $M \
    --panel panel-8 --gpus '$GPU' ${names[*]} >> $R/logs/4bif-pool-$NODE.log 2>&1 < /dev/null &"
  echo "$(date -u +%FT%TZ) pool on node $NODE GPUs ${GPU// /,}: ${names[*]} (log logs/4bif-pool-$NODE.log)"
  exit 0
fi
case "$CAND" in
  UP) NAME=DEV2.0-4B-LHA10UP-bf16 SRC_NODE=b SRC=/data/dev2/runs/dec/m14/soup/4b-LHA10UP/build/4b-LHA10UP-soup
    FP32=9e80e7654965f539b78fe91c6428253ccf565bf29a9550d1f4c94e911b77007e ;;
  a75) NAME=DEV2.0-4B-LHA10SD-a75-bf16 SRC_NODE=b SRC=/data/dev2/runs/dec/m16/points/4b-LHA10SD-a75/build/4b-LHA10SD-a75
    FP32=96d09b160fd0f3b8673620ef37c1d226cab5afa9d1634f156c4151206a2ccc73 ;;
  a50) NAME=DEV2.0-4B-LHA10SD-a50-bf16 SRC_NODE=b SRC=/data/dev2/runs/dec/m16/points/4b-LHA10SD-a50/build/4b-LHA10SD-a50
    FP32=84f462f2515526601d2726b9f78cda75135d5c66ba0cee14fc4b25d30450a464 ;;
  SDML) NAME=DEV2.0-4B-LHA10SDML-bf16 SRC_NODE=f SRC=/data/dev2/runs/dec/m15/soup/4b-LHA10SDML/build/4b-LHA10SDML-soup
    FP32=1b51567523426b896ae50afeefca4f133c2e3c220c349ebfb9b9c630600dd19d ;;
  SDB) NAME=DEV2.0-4B-LHA10SD-bf16 SRC_NODE=b SRC=/data/dev2/runs/dec/m16/inputs/arms/4b-LHA10SD
    FP32=255021e0a3f2af49d41ffbd3425d541e6d7749b8cd95a8e8a7b764a81e8a037d ;;
  SD) NAME=DEV2.0-4B-LHA10SD SRC_NODE="" SRC="" FP32=255021e0a3f2af49d41ffbd3425d541e6d7749b8cd95a8e8a7b764a81e8a037d ;;
  *) echo "bad CAND $CAND" >&2; exit 2 ;;
esac
M=/data/dev2/src/$SHA-src_training_decision2
S=$M/src/training/decision2
R=/data/dev2/private/eval/index021/ix1
MD=/data/dev2/models/ix1/dec-4bif
BASEPKG=/data/dev2/models/ix1/DEV2.0-4B-13d42143
PKG=$MD/$NAME-r13d42143 CK=$MD/$NAME-ckpt FSRC=$MD/src/$CAND-fp32
LOADED=4208383488
REF=DEV2.0-4B-LH
KEY="-i /root/.ssh/d2_temp_cd -o BatchMode=yes -o ConnectTimeout=30"
EXCLUDE="HoVer When2Call iSarcasmEval GSM8K BPoMP"
LOCAL=${IX4B_LOCAL:-$HOME/code/decision2-program/private/4b-indexfirst}/$NAME
IMAGE=decision20-train-fast:host2
IMAGE_ID=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
sums() { printf '%s' "cd '$1' && find . -type f | LC_ALL=C sort | xargs -d '\n' -P 8 -n 4 sha256sum | sort -k2"; }
panel_of() { [ "$1" = c ] && echo panel-7 || echo panel-8; }
need_node() { [[ "$NODE" == c || "$NODE" == d ]] || { echo "NODE must be c or d" >&2; exit 2; }; }
mirror_on() { on "$1" "test -f $S/v2/eval/ix1/launch.sh" || { echo "mirror $SHA is not on node $1" >&2; exit 2; }; }
case "$STAGE" in
  name) echo "$NAME" ;;
  stage)
    need_node; mirror_on "$NODE"
    [ -n "$SRC_NODE" ] || { echo "$CAND has no source checkpoint" >&2; exit 2; }
    on "$NODE" "grep -q '^  \[$NAME\]=\"DEV2.0-4B 13d4214361d0d4fdb0d5002f9a8eae79e8c6a73f $PKG\"' $S/v2/eval/ix1/launch.sh" ||
      { echo "mirror $SHA has no DIAGNOSTIC entry $NAME -> $PKG" >&2; exit 2; }
    on "$NODE" "test ! -e $PKG && test ! -e $CK && test ! -e $FSRC" || { echo "$NAME is already staged on node $NODE" >&2; exit 3; }
    on "$NODE" "test \"\$(docker image inspect -f '{{.Id}}' $IMAGE)\" = $IMAGE_ID" || { echo "image $IMAGE is not $IMAGE_ID" >&2; exit 3; }
    src=$SRC
    if [ "$SRC_NODE" = f ]; then
      src=/data/dev2/runs/dec/4bif/inputs/$CAND
      if ! on b "test -d $src"; then
        on b "mkdir -p $(dirname "$src") && rsync -a -e 'ssh $KEY' $(addr f):$SRC/ $src.part/ && mv $src.part $src"
      fi
      a=$(on f "$(sums "$SRC")") b=$(on b "$(sums "$src")")
      [ -n "$a" ] && [ "$a" = "$b" ] || { echo "node B copy of the node-F soup differs" >&2; exit 3; }
      echo "node-F soup on node B: $(wc -l <<< "$a") files, SHA-256 lists equal"
    fi
    on "$NODE" "umask 077; mkdir -p $MD/src"
    echo "$(date -u +%FT%TZ) $NAME: copying the FP32 source node B -> node $NODE"
    on b "rsync -a -e 'ssh $KEY' $src/ $(addr "$NODE"):$FSRC/"
    a=$(on b "$(sums "$src")") c=$(on "$NODE" "$(sums "$FSRC")")
    [ -n "$a" ] && [ "$a" = "$c" ] || { echo "node $NODE copy differs from node B's" >&2; exit 3; }
    echo "FP32 source: $(wc -l <<< "$a") files, SHA-256 lists equal"
    cpu="docker run --rm --network none -e HIP_VISIBLE_DEVICES= -e CUDA_VISIBLE_DEVICES= -e ROCR_VISIBLE_DEVICES= \
      -e PYTHONPATH=$S -v $S:$S:ro -v $FSRC:$FSRC:ro -v $MD:$MD -w $S --entrypoint python3 $IMAGE -B"
    on "$NODE" "$cpu -m unittest v2.release.tests.test_bf16_copy > $MD/$NAME-test_bf16_copy.log 2>&1 && \
      $cpu -m v2.release.bf16_copy --source $FSRC --output $CK --receipt $MD/$NAME-bf16-copy.json > $MD/$NAME-bf16.log 2>&1" ||
      { echo "bf16 copy FAILED (see $MD/$NAME-*.log on node $NODE)" >&2; exit 3; }
    model=$(on "$NODE" "python3 - $MD/$NAME-bf16-copy.json $FP32" << 'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
assert r["source_model_sha256"] == sys.argv[2], f"source fingerprint {r['source_model_sha256']} != {sys.argv[2]}"
print(r["model_sha256"])
EOF
    )
    [[ "$model" =~ ^[0-9a-f]{64}$ ]] || { echo "bad bf16 receipt" >&2; exit 3; }
    echo "bf16 copy: source ${FP32:0:12} -> $model (receipt $(on "$NODE" "sha256sum < $MD/$NAME-bf16-copy.json | cut -c1-64"))"
    on "$NODE" "cd $S && PYTHONPATH=$S python3 -m v2.eval.ix1.restage --package $BASEPKG --out $PKG --checkpoint $CK --model-sha256 $model"
    on "$NODE" "python3 - $PKG/MODEL_MANIFEST.json $model $LOADED" << 'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
assert m["identity"]["model_sha256"] == sys.argv[2], "identity"
assert m["parameters"]["loaded"] == int(sys.argv[3]), f"loaded {m['parameters']['loaded']}"
assert m["calibration"] is None
print(f"restaged: identity {sys.argv[2][:12]}, loaded {m['parameters']['loaded']:,}, T = 1")
EOF
    echo "manifest $(on "$NODE" "sha256sum < $PKG/MODEL_MANIFEST.json | cut -c1-64")" ;;
  parity)
    need_node; mirror_on "$NODE"
    [[ "$GPU" =~ ^[0-7]$ ]] || { echo "parity needs a GPU" >&2; exit 2; }
    [[ "$NODE$GPU" != c0 ]] || { echo "node C GPU0 is foreign" >&2; exit 2; }
    on "$NODE" "test -f $PKG/MODEL_MANIFEST.json" || { echo "run stage first" >&2; exit 3; }
    on "$NODE" "test ! -e $R/logs/4bif-parity-$NAME.exit" || { echo "parity of $NAME already ran" >&2; exit 3; }
    P=$(panel_of "$NODE")
    on "$NODE" "mkdir -p $R/logs; setsid nohup bash -c 'cd $S && bash v2/eval/ix1/launch.sh parity --src $M --model $NAME \
      --gpu $GPU --run $R/parity/$NAME --rows $R/$P/compat-86.gold-free.jsonl.gz; echo \$? > $R/logs/4bif-parity-$NAME.exit' \
      > $R/logs/4bif-parity-$NAME.log 2>&1 < /dev/null &"
    echo "$(date -u +%FT%TZ) $NAME parity gate started on node $NODE GPU$GPU"
    until on "$NODE" "test -f $R/logs/4bif-parity-$NAME.exit"; do sleep 45; done
    on "$NODE" "cat $R/logs/4bif-parity-$NAME.exit; python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); \
      print(json.dumps({k: d[k] for k in (\"pass\", \"requests\", \"statuses\", \"max_abs_dp\")})); \
      sys.exit(0 if d[\"pass\"] and d[\"requests\"] == 86 else 1)' $R/parity/$NAME/parity.json" ;;
  void)
    need_node
    on "$NODE" "test -f $R/logs/4bif-parity-$NAME.exit && test ! -e $R/parity/$NAME/parity.json && test ! -e $R/parity/$NAME/ref/start_epoch" ||
      { echo "nothing to void: no refused parity attempt of $NAME (a refused attempt has no reference pass)" >&2; exit 3; }
    on "$NODE" "docker ps --format '{{.Names}}' | grep -q '^ix1-parity-$(tr 'A-Z.' 'a-z_' <<< "$NAME")-'" &&
      { echo "a $NAME parity container is running" >&2; exit 3; }
    on "$NODE" "v=$R/void/4bif-$NAME-\$(date -u +%Y%m%dT%H%M%SZ) && mkdir -p \$v && mv $R/parity/$NAME \$v/parity && \
      mv $R/logs/4bif-parity-$NAME.exit $R/logs/4bif-parity-$NAME.log \$v/ && echo voided into \$v" ;;
  run)
    need_node; mirror_on "$NODE"
    on "$NODE" "python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))[\"pass\"] else 1)' $R/parity/$NAME/parity.json" ||
      { echo "parity gate missing or failed" >&2; exit 3; }
    read -r -a g <<< "$GPU"
    [[ " $GPU " != *" 0 "* || "$NODE" != c ]] || { echo "node C GPU0 is foreign" >&2; exit 2; }
    on "$NODE" "test ! -e $R/runs/$NAME/4bif-panel" || { echo "$NAME already launched on node $NODE" >&2; exit 3; }
    if [ "$NODE" = c ] && [ "${#g[@]}" = 7 ]; then
      on c "mkdir -p $R/runs/$NAME && echo panel-7 > $R/runs/$NAME/4bif-panel && setsid nohup bash -c 'cd $S && \
        bash v2/eval/ix1/launch.sh run --src $M --model $NAME --gpus \"$GPU\" --run $R/runs/$NAME --rows-dir $R/panel-7 \
        --cache $R/parity/$NAME/cache-frozen' > $R/logs/4bif-run-$NAME.log 2>&1 < /dev/null &"
    else
      [ "${#g[@]}" -ge 1 ] && [ "${#g[@]}" -le 8 ] || { echo "1 to 8 GPUs" >&2; exit 2; }
      plan=""
      for i in "${!g[@]}"; do
        ks=""; for ((k = i; k < 8; k += ${#g[@]})); do ks="${ks:+$ks,}$k"; done
        plan="${plan:+$plan }${g[$i]}:$ks"
      done
      on "$NODE" "mkdir -p $R/runs/$NAME && echo panel-8 > $R/runs/$NAME/4bif-panel && \
        setsid nohup bash $S/v2/dec/ops/4bif/lanes.sh $M $NAME panel-8 '$plan' > $R/logs/4bif-run-$NAME.log 2>&1 < /dev/null &"
    fi
    echo "$(date -u +%FT%TZ) $NAME full run launched on node $NODE GPU ${GPU// /,}" ;;
  status)
    need_node
    on "$NODE" "python3 - $R/runs/$NAME" << 'EOF'
import glob, os, sys, time
run, total = sys.argv[1], 0.0
for w in sorted(glob.glob(f"{run}/shard-*"), key=lambda p: int(p.rsplit("-", 1)[1])):
    k = w.rsplit("-", 1)[1]
    rows = sum(1 for _ in open(f"{w}/results.jsonl")) if os.path.exists(f"{w}/results.jsonl") else 0
    start = float(open(f"{w}/start_epoch").read()) if os.path.exists(f"{w}/start_epoch") else None
    end = float(open(f"{w}/end_epoch").read()) if os.path.exists(f"{w}/end_epoch") else None
    code = open(f"{w}/exit_code").read().strip() if os.path.exists(f"{w}/exit_code") else "-"
    if start:
        total += ((end or time.time()) - start) / 3600
    print(f"shard {k}: {rows} records, {'ended exit ' + code if end else ('running' if start else 'waiting')}")
print(f"GPU-h so far (current intervals) {total:.2f}")
EOF
    ;;
  score)
    need_node; mirror_on "$NODE"
    P=$(on "$NODE" "cat $R/runs/$NAME/4bif-panel") n=${P#panel-}
    on "$NODE" "for k in \$(seq 0 $((n - 1))); do test \"\$(cat $R/runs/$NAME/shard-\$k/exit_code 2>/dev/null)\" = 0 || exit 1; done" ||
      { echo "not every shard ended with exit code 0" >&2; exit 3; }
    on "$NODE" "test ! -e $R/runs/$NAME/merged" || { echo "$NAME is already scored" >&2; exit 3; }
    on "$NODE" "cd $S && bash v2/eval/ix1/score.sh --src $M --model $NAME --size 4B --panel $R/$P > $R/logs/4bif-score-$NAME.log 2>&1"
    on "$NODE" "python3 - $R/runs/$NAME/merged" << 'EOF'
import json, sys
r = json.load(open(f"{sys.argv[1]}/receipt.json"))
c = json.load(open(f"{sys.argv[1]}/compare.json"))
gate = {k: v for k, v in c.items() if "pass" in k or k == "scorer_gate"}
print(json.dumps({"rows": r["rows"], "statuses": r["statuses"], "gpu_hours": r["gpu_hours"],
                  "results_sha256": r["results_sha256"], "panel_run_ids_sha256": r["panel_run_ids_sha256"],
                  "scorers": gate}))
EOF
    ;;
  pull|import)
    if [ "$STAGE" = import ]; then [ "$CAND" = SD ] || { echo "import is for SD" >&2; exit 2; }; fi
    if [ "$STAGE" = pull ]; then [ "$CAND" != SD ] || { echo "use import for SD" >&2; exit 2; }; fi
    on c "test ! -e $R/runs/$NAME/merged" || { echo "node C already has runs/$NAME/merged" >&2; exit 3; }
    on d "test -f $R/runs/$NAME/merged/receipt.json" || { echo "node D has no scored $NAME" >&2; exit 3; }
    on c "umask 077; mkdir -p $R/runs/$NAME/merged.part/kit"
    for f in results.jsonl compare.json port.json receipt.json latency.json kit/index.json; do
      on b "ssh $KEY $(addr d) 'cat $R/runs/$NAME/merged/$f' | ssh $KEY $(addr c) 'umask 077; cat > $R/runs/$NAME/merged.part/$f'"
      a=$(on d "sha256sum < $R/runs/$NAME/merged/$f | cut -c1-64") c=$(on c "sha256sum < $R/runs/$NAME/merged.part/$f | cut -c1-64")
      [ "$a" = "$c" ] || { echo "$f differs after the copy" >&2; exit 3; }
    done
    on c "python3 -c 'import hashlib,json,sys; r=json.load(open(sys.argv[1]+\"/receipt.json\")); \
      h=hashlib.sha256(open(sys.argv[1]+\"/results.jsonl\",\"rb\").read()).hexdigest(); \
      sys.exit(0 if h == r[\"results_sha256\"] else 1)' $R/runs/$NAME/merged.part" || { echo "results SHA-256 is not the receipt's" >&2; exit 3; }
    on c "mv $R/runs/$NAME/merged.part $R/runs/$NAME/merged && printf '{\"schema\": \"4bif-copy/1\", \"from\": \"node D runs/$NAME/merged\", \"utc\": \"%s\"}\n' \
      \"\$(date -u +%FT%TZ)\" > $R/runs/$NAME/COPIED-FROM-NODE-D.json"
    echo "$NAME merged outputs on node C (6 files, SHA-256 equal; results = receipt)" ;;
  boot)
    mirror_on c
    on c "test -f $R/runs/$NAME/merged/results.jsonl && test -f $R/runs/$REF/merged/results.jsonl" || { echo "missing results" >&2; exit 3; }
    on c "test ! -e $R/runs/$NAME/4bif-boot-full-vs-lh.json && test ! -e $R/logs/4bif-boot-$NAME.started" || { echo "boot of $NAME already started" >&2; exit 3; }
    on c "cd $S && PYTHONPATH=$S python3 -m v2.eval.ix1.family_delta --base $REF=$R/runs/$REF/merged/compare.json \
      --new $NAME=$R/runs/$NAME/merged/compare.json --out $R/runs/$NAME/family-delta-vs-lh.json > /dev/null"
    boot="cd $R/.. && CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= PYTHONPATH=$S:$R/../kit-19ad28ec nice -n 10 \
      venv/bin/python -m v2.eval.ix1.paired_boot --suite-dir suite-0.2 --base $R/runs/$REF/merged/results.jsonl \
      --new $R/runs/$NAME/merged/results.jsonl --external $R/../external/index021-frontier-gap-2026-10-01.json \
      --replicates 2000 --seed 20261002 --workers 20"
    on c "umask 077; date -u +%FT%TZ > $R/logs/4bif-boot-$NAME.started; \
      setsid nohup bash -c '$boot --out $R/runs/$NAME/4bif-boot-full-vs-lh.json; echo \$? > $R/logs/4bif-boot-full-$NAME.exit' \
        > $R/logs/4bif-boot-full-$NAME.log 2>&1 < /dev/null & \
      setsid nohup bash -c '$boot --exclude $EXCLUDE --out $R/runs/$NAME/4bif-boot-transfer-vs-lh.json; echo \$? > $R/logs/4bif-boot-transfer-$NAME.exit' \
        > $R/logs/4bif-boot-transfer-$NAME.log 2>&1 < /dev/null &"
    echo "$(date -u +%FT%TZ) $NAME bootstraps started on node C (CPU)" ;;
  fetch)
    on c "test \"\$(cat $R/logs/4bif-boot-full-$NAME.exit)\" = 0 && test \"\$(cat $R/logs/4bif-boot-transfer-$NAME.exit)\" = 0" ||
      { echo "bootstraps of $NAME not finished with exit 0" >&2; exit 3; }
    (umask 077 && mkdir -p "$LOCAL")
    for f in merged/compare.json merged/receipt.json merged/port.json merged/kit/index.json family-delta-vs-lh.json \
      4bif-boot-full-vs-lh.json 4bif-boot-transfer-vs-lh.json; do
      out=$(basename "$f"); [ "$f" = merged/kit/index.json ] && out=kit-index.json
      on c "cat $R/runs/$NAME/$f" > "$LOCAL/$out"
    done
    [ "$CAND" = SD ] && pnode=d || pnode=$(on c "test -f $R/parity/$NAME/parity.json" && echo c || echo d)
    on "$pnode" "cat $R/parity/$NAME/parity.json" > "$LOCAL/parity.json"
    chmod 600 "$LOCAL"/*.json
    echo "private outputs in $LOCAL (never copy a value into a commit, record, gist or card)" ;;
  release)
    need_node
    on "$NODE" "docker ps --format '{{.Names}}' | grep -q '^ix1-$(tr 'A-Z.' 'a-z_' <<< "$NAME")-'" &&
      { echo "a $NAME Index container is still running" >&2; exit 3; }
    on "$NODE" "for g in 0 1 2 3 4 5 6 7; do f=/data/dev2/leases/gpu\$g.lock/owner; grep -q 'IX1 .* $NAME\$' \$f 2>/dev/null || continue; \
      printf 'track=eval-ix1\nstatus=released (4B Index-first run of $NAME done)\nlast_job_end_utc=%s\n' \
      \"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\" > \$f; echo gpu\$g released; done" ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
