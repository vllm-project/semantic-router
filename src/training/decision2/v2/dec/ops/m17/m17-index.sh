#!/usr/bin/env bash
# Decoder M17 Index runs (amendment 1; COORDINATION 2026-10-02 09:55 Index-first rule), workstation side. The method is
# the 4B Index-first worker's (dec/ops/4bif/ix4b.sh): a candidate is measured on the weights a release ships, the
# v2.release.bf16_copy of its frozen FP32 soup (source fingerprint = the soup's model SHA-256), restaged onto the
# released LH IX1 package DEV2.0-4B 13d42143 (identity, loaded count 4,208,383,488, calibration none), through IX1's
# harness (image host2, kit 87d4650b, the 86-request parity gate, panel-8, dual scoring) and compared with IX1's run
# of the released package on node C (runs/DEV2.0-4B-LH). Inference on M17's GPUs (node E GPU0-3, node F GPU2/3/6/7);
# scoring and bootstraps on node C (CPU). Index values stay in node private directories and in
# ~/code/decision2-program/private/m17/; this script prints none.
#
# Usage: m17-index.sh MIRROR_SHA ARM STAGE [NODE] ["GPUS"]
#   ARM      LHS10SD | LHS17SD (the M17 arm soups on node F); stage 2: LHS17UP | LHS23SD | LHS17IB4 | LHS17IB4X or an
#            interpolation point (soup/4b-ARM with DONE and MODEL_SHA256, as an arm soup). With M17_REF=S17 the
#            reference run is the current release's (runs/DEV2.0-4B-LHS17SD-bf16) and boot / fetch write *-vs-s17.json;
#            with M17_REF=SDML (amendment 2) it is the released 4b-LHA10SDML's (runs/IS-4b-LHA10SDML-bf16, merged/ copied
#            from node D by refcopy-sdml) and they write *-vs-sdml.json
#   stage    node F: test_bf16_copy, v2.release.bf16_copy (CPU container of host2, no network) -> NAME-ckpt, then
#            v2.eval.ix1.restage onto DEV2.0-4B-13d42143 with the copy's model SHA-256 -> NAME-r13d42143 (checked)
#   ship     NODE: the restaged package node F -> NODE through node A (per-file SHA-256 lists equal)
#   assets   NODE: kit-87d4650b and panel-8 node C -> NODE through node A, once (lists equal; kit HEAD checked)
#   lease    NODE "GPUS": M17's owner files -> the harness form (track=eval-ix1, status=released, dec-m17 named)
#   pool     NODE "GPUS": m17_ixpool.py, detached, for ARM's NAME (ARM may be a comma list)
#   status   NODE: per-shard records, ends and exit codes
#   stop-pool NODE: stop NODE's m17_ixpool.py (its shard containers run on and record their own ends)
#   lanes    NODE "GPU:k,k ..." ["GPU:k ..."]: m17-lanes.sh, detached: ARM's remaining shards per GPU, each lane after
#            the shard already running on its GPU (second list)
#   parity-copy NODE: ARM's passed parity gate (parity.json, records and frozen cache) node E -> NODE through node A,
#            so the candidate keeps one parity gate when its shards run on NODE
#   relay    NODE: runs/NAME (shards, launcher records; without the shards' triton / home copies) NODE -> node C
#            through node A (lists equal)
#   score    node C: score.sh --size 4B over panel-8 (merge, port + kit, compare)
#   boot     node C, CPU, detached: family_delta vs DEV2.0-4B-LH and the paired bootstrap of the full panel
#            (v2.eval.ix1.paired_boot, 2,000 replicates, seed 20261002) -> runs/NAME/m17-boot-full-vs-lh.json (-vs-s17 with M17_REF=S17)
#   fetch    node C -> the local private folder (mode 700): parity, receipt, compare, port, kit index, family delta,
#            bootstrap, and the transfer-only weighted delta (family delta without HoVer, When2Call, iSarcasmEval,
#            GSM8K and BPoMP)
#   release  NODE "GPUS": owner files back to track=dec-m17 (no M17 Index container running)
#   audit2-stage / audit2-run / audit2-status (ARM ignored): the same audit of the stage-2 TRAIN files (node F's
#            locked copies: 4b-LHS23SD, 4b-LHS17IB4, 4b-LHS17IB4X; 4b-LHS17UP trains on the audited 4b-LHS17SD file)
#            -> node C ix1/audit/m17s2
#   audit3-stage / audit3-run / audit3-status (ARM ignored): the same audit of the M15 4b-LHA10SDML TRAIN (node F
#            m15/data/4b, locked fef6b036 in READY-m15.json) for its release -> node C ix1/audit/m17sdml
#   audit-stage / audit-run / audit-status (ARM ignored): the row-level Index contamination audit (integrity check;
#            v2.eval.ix1.contamination, the IX1 method with 200 planted controls, as the 4B worker's audit4b.sh) of both
#            M17 TRAIN files on node C, CPU only; the files come from node E's data lock copies (hash-checked);
#            outputs private (node C ix1/audit/m17/out); counts only are printed
set -euo pipefail
SHA=${1:?MIRROR_SHA} ARM=${2:?ARM} STAGE=${3:?STAGE} NODE=${4:-} GPUS=${5:-}
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
NODES=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
addr() { awk -F= -v k="node-$1" '$1 == k { print substr($0, length(k) + 2); exit }' "$NODES"; }
on() { local n=$1; shift; ssh -o BatchMode=yes -o ConnectTimeout=30 "$(addr "$n")" "$@"; }
name_of() { echo "DEV2.0-4B-$1-bf16"; }
M=/data/dev2/src/$SHA-src_training_decision2
S=$M/src/training/decision2
R=/data/dev2/private/eval/index021/ix1
MD=/data/dev2/models/ix1/dec-m17
BASEPKG=/data/dev2/models/ix1/DEV2.0-4B-13d42143
LOADED=4208383488
REF=DEV2.0-4B-LH VS=lh
[ "${M17_REF:-LH}" = S17 ] && REF=DEV2.0-4B-LHS17SD-bf16 VS=s17
[ "${M17_REF:-LH}" = SDML ] && REF=IS-4b-LHA10SDML-bf16 VS=sdml
SDML_RESULTS=a459ce7c5be0383c355c49a702a4377f22b48e07cff7d2c653c85c50c0a34ea6
KEY="-i /root/.ssh/d2_temp_cd -o BatchMode=yes -o ConnectTimeout=30"
IMAGE=decision20-train-fast:host2
IMAGE_ID=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
EXCLUDE=(HoVer When2Call iSarcasmEval GSM8K BPoMP)
sums() {  # <dir> [lean]: per-file SHA-256 list (lean: without the shards' triton / home directories)
  local skip=""
  [ "${2:-}" = lean ] && skip="-not -path '*/triton/*' -not -path '*/home/*'"
  printf '%s' "cd '$1' && find . -type f $skip | LC_ALL=C sort | xargs -d '\n' -P 8 -n 4 sha256sum | sort -k2"
}
hop() {  # <from> <to> <parent> <name> [lean]: tar stream through node A, then equal per-file SHA-256 lists
  local from=$1 to=$2 parent=$3 item=$4 lean=${5:-} a b ex=""
  [ "$lean" = lean ] && ex="--exclude=*/triton --exclude=*/home"
  on "$to" "test ! -e '$parent/$item'" || { echo "$parent/$item exists on node $to" >&2; exit 3; }
  on a "ssh $KEY $(addr "$from") \"tar -C '$parent' $ex -cf - '$item'\" | ssh $KEY $(addr "$to") \
    \"umask 077; mkdir -p '$parent/.m17-part' && tar -C '$parent/.m17-part' -xf - && mv -T '$parent/.m17-part/$item' '$parent/$item' && rmdir '$parent/.m17-part'\""
  a=$(on "$from" "$(sums "$parent/$item" "$lean")") b=$(on "$to" "$(sums "$parent/$item" "$lean")")
  [ -n "$a" ] && [ "$a" = "$b" ] || { echo "$item: node $to copy differs from node $from" >&2; exit 3; }
  echo "$item node $from -> node $to: $(wc -l <<< "$a") files, SHA-256 lists equal"
}
A=$R/audit/m17
A2=$R/audit/m17s2
A3=$R/audit/m17sdml
SDML_TRAIN=fef6b036f33de6756dab083fd21ab462ec2975cfa63120f2452d3c9145d33dd4
case "$STAGE" in
  refcopy-sdml)
    hop d c "$R/runs/IS-4b-LHA10SDML-bf16" merged
    [ "$(on c "sha256sum < $R/runs/IS-4b-LHA10SDML-bf16/merged/results.jsonl | cut -c1-64")" = $SDML_RESULTS ] \
      || { echo "the copied IS-4b-LHA10SDML-bf16 results are not $SDML_RESULTS" >&2; exit 3; } ;;
  audit2-stage)
    on c "test ! -e $A2/train/FILES.txt" || { echo "audit2 already staged" >&2; exit 3; }
    on c "umask 077; mkdir -p $A2/train"
    for x in 4b-s2/4b-LHS23SD=c629913c3570505d919f4ed48af49e4a24e50286d2991e5bec396f6fb7bec6a8 \
      4b-s3/4b-LHS17IB4 4b-s3/4b-LHS17IB4X; do
      p=${x%%=*} a=$(basename "${x%%=*}") want=${x#*=}
      [ "$want" = "$x" ] && want=$(on f "python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[\"arms\"][sys.argv[2]])' /data/dev2/runs/dec/m17/data/READY-m17s3.json $a")
      on a "ssh $KEY $(addr f) 'cat /data/dev2/runs/dec/m17/data/$p/train.jsonl' | ssh $KEY $(addr c) 'umask 077; cat > $A2/train/$a.train.jsonl'"
      [ "$(on c "sha256sum < $A2/train/$a.train.jsonl | cut -c1-64")" = "$want" ] || { echo "$a TRAIN copy is not $want" >&2; exit 3; }
      on c "printf '%s %s rows %s\n' $a $want \"\$(wc -l < $A2/train/$a.train.jsonl)\" >> $A2/train/FILES.txt"
    done
    on c "cat $A2/train/FILES.txt" ;;
  audit2-run)
    on c "test -f $S/v2/eval/ix1/contamination.py && test -f $A2/train/FILES.txt && test ! -e $A2/out" || { echo "not staged, or already run" >&2; exit 3; }
    on c "umask 077; setsid nohup bash -c 'cd $S && CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= PYTHONHASHSEED=0 PYTHONPATH=$S nice -n 10 \
      python3 -m v2.eval.ix1.contamination --panel $R/panel-7 --train 4b-LHS23SD=$A2/train/4b-LHS23SD.train.jsonl \
      --train 4b-LHS17IB4=$A2/train/4b-LHS17IB4.train.jsonl --train 4b-LHS17IB4X=$A2/train/4b-LHS17IB4X.train.jsonl \
      --workers 24 --out $A2/out; echo \$? > $A2/exit' > $A2/audit.log 2>&1 < /dev/null &"
    echo "$(date -u +%FT%TZ) audit2 started on node C (CPU)" ;;
  audit3-stage)
    on c "test ! -e $A3/train/FILES.txt" || { echo "audit3 already staged" >&2; exit 3; }
    [ "$(on f "python3 -c 'import json; print(json.load(open(\"/data/dev2/runs/dec/m15/data/READY-m15.json\"))[\"arms\"][\"4b-LHA10SDML\"])'")" = $SDML_TRAIN ] \
      || { echo "READY-m15.json does not lock 4b-LHA10SDML at $SDML_TRAIN" >&2; exit 3; }
    on c "umask 077; mkdir -p $A3/train"
    on a "ssh $KEY $(addr f) 'cat /data/dev2/runs/dec/m15/data/4b/4b-LHA10SDML/train.jsonl' | ssh $KEY $(addr c) 'umask 077; cat > $A3/train/4b-LHA10SDML.train.jsonl'"
    [ "$(on c "sha256sum < $A3/train/4b-LHA10SDML.train.jsonl | cut -c1-64")" = $SDML_TRAIN ] || { echo "4b-LHA10SDML TRAIN copy is not $SDML_TRAIN" >&2; exit 3; }
    on c "printf '4b-LHA10SDML %s rows %s\n' $SDML_TRAIN \"\$(wc -l < $A3/train/4b-LHA10SDML.train.jsonl)\" >> $A3/train/FILES.txt; cat $A3/train/FILES.txt" ;;
  audit3-run)
    on c "test -f $S/v2/eval/ix1/contamination.py && test -f $A3/train/FILES.txt && test ! -e $A3/out" || { echo "not staged, or already run" >&2; exit 3; }
    on c "umask 077; setsid nohup bash -c 'cd $S && CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= PYTHONHASHSEED=0 PYTHONPATH=$S nice -n 10 \
      python3 -m v2.eval.ix1.contamination --panel $R/panel-7 --train 4b-LHA10SDML=$A3/train/4b-LHA10SDML.train.jsonl \
      --workers 24 --out $A3/out; echo \$? > $A3/exit' > $A3/audit.log 2>&1 < /dev/null &"
    echo "$(date -u +%FT%TZ) audit3 started on node C (CPU)" ;;
  audit-stage)
    on c "test ! -e $A/train/FILES.txt" || { echo "audit already staged" >&2; exit 3; }
    on c "umask 077; mkdir -p $A/train"
    for x in LHS10SD=72fa2d844fbf94be890858b9b66af0e26e12a62011929bf1eb0de1ebe93ef025 \
      LHS17SD=14bce13ce926b354e214581e7cf4718d03f80c975b36dbce6731fbc6517e25a0; do
      a=${x%%=*} want=${x#*=}
      on a "ssh $KEY $(addr e) 'cat /data/dev2/runs/dec/m17/data/4b/4b-$a/train.jsonl' | ssh $KEY $(addr c) 'umask 077; cat > $A/train/4b-$a.train.jsonl'"
      [ "$(on c "sha256sum < $A/train/4b-$a.train.jsonl | cut -c1-64")" = "$want" ] || { echo "4b-$a TRAIN copy is not $want" >&2; exit 3; }
      on c "printf '4b-%s %s rows %s\n' $a $want \"\$(wc -l < $A/train/4b-$a.train.jsonl)\" >> $A/train/FILES.txt"
    done
    on c "cat $A/train/FILES.txt" ;;
  audit-run)
    on c "test -f $S/v2/eval/ix1/contamination.py && test -f $A/train/FILES.txt && test ! -e $A/out" || { echo "not staged, or already run" >&2; exit 3; }
    on c "umask 077; setsid nohup bash -c 'cd $S && CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= PYTHONHASHSEED=0 PYTHONPATH=$S nice -n 10 \
      python3 -m v2.eval.ix1.contamination --panel $R/panel-7 --train 4b-LHS10SD=$A/train/4b-LHS10SD.train.jsonl \
      --train 4b-LHS17SD=$A/train/4b-LHS17SD.train.jsonl --workers 24 --out $A/out; echo \$? > $A/exit' > $A/audit.log 2>&1 < /dev/null &"
    echo "$(date -u +%FT%TZ) audit started on node C (CPU)" ;;
  audit-status | audit2-status | audit3-status)
    [ "$STAGE" = audit2-status ] && A=$A2
    [ "$STAGE" = audit3-status ] && A=$A3
    on c "cat $A/exit 2>/dev/null || echo running; test -f $A/out/audit.json && python3 - $A/out/audit.json" << 'EOF'
import json, sys
a = json.load(open(sys.argv[1]))
print(json.dumps({"index_rows": a["index_rows"], "planted": a["planted_control"]["found"],
                  "missed": len(a["planted_control"]["missed"]),
                  "sets": {k: {"lines": v["training_lines"], "duplicate_rows": v["duplicate_rows"],
                               "item_rows": v["item_rows"]} for k, v in a["training_sets"].items()}}))
EOF
    exit 0 ;;
esac
case "$STAGE" in audit-stage | audit-run | audit2-stage | audit2-run | audit3-stage | audit3-run | refcopy-sdml) exit 0 ;; esac
[[ "$ARM" =~ ^(LHS|SDML)[0-9A-Za-z-]+(,(LHS|SDML)[0-9A-Za-z-]+)*$ ]] || { echo "bad ARM $ARM" >&2; exit 2; }
NAME=$(name_of "${ARM%%,*}")
PKG=$MD/$NAME-r13d42143 CK=$MD/$NAME-ckpt
SOUP=/data/dev2/runs/dec/m17/soup/4b-${ARM%%,*}
case "$STAGE" in
  stage)
    on f "test -f $S/v2/eval/ix1/launch.sh && grep -q '^  \[$NAME\]=\"DEV2.0-4B 13d4214361d0d4fdb0d5002f9a8eae79e8c6a73f $PKG\"' $S/v2/eval/ix1/launch.sh" \
      || { echo "mirror $SHA on node F lacks the DIAGNOSTIC entry $NAME -> $PKG" >&2; exit 2; }
    on f "test -f $SOUP/DONE && test -f $SOUP/MODEL_SHA256" || { echo "no soup of 4b-$ARM" >&2; exit 3; }
    on f "test ! -e $PKG && test ! -e $CK" || { echo "$NAME is already staged" >&2; exit 3; }
    on f "test \"\$(docker image inspect -f '{{.Id}}' $IMAGE)\" = $IMAGE_ID" || { echo "image $IMAGE is not $IMAGE_ID" >&2; exit 3; }
    src=$(on f "cat $SOUP/DONE") fp32=$(on f "cat $SOUP/MODEL_SHA256")
    cpu="docker run --rm --network none -e HIP_VISIBLE_DEVICES= -e CUDA_VISIBLE_DEVICES= -e ROCR_VISIBLE_DEVICES= \
      -e PYTHONPATH=$S -v $S:$S:ro -v $src:$src:ro -v $MD:$MD -w $S --entrypoint python3 $IMAGE -B"
    on f "umask 022; mkdir -p $MD && $cpu -m unittest v2.release.tests.test_bf16_copy > $MD/$NAME-test_bf16_copy.log 2>&1 && \
      $cpu -m v2.release.bf16_copy --source $src --output $CK --receipt $MD/$NAME-bf16-copy.json > $MD/$NAME-bf16.log 2>&1" \
      || { echo "bf16 copy FAILED (see $MD/$NAME-*.log on node F)" >&2; exit 3; }
    model=$(on f "python3 - $MD/$NAME-bf16-copy.json $fp32" << 'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
assert r["source_model_sha256"] == sys.argv[2], f"source fingerprint {r['source_model_sha256']} != {sys.argv[2]}"
print(r["model_sha256"])
EOF
    )
    [[ "$model" =~ ^[0-9a-f]{64}$ ]] || { echo "bad bf16 receipt" >&2; exit 3; }
    echo "bf16 copy: source ${fp32:0:12} -> ${model:0:12}"
    on f "cd $S && PYTHONPATH=$S python3 -B -m v2.eval.ix1.restage --package $BASEPKG --out $PKG --checkpoint $CK --model-sha256 $model" > /dev/null
    on f "python3 - $PKG/MODEL_MANIFEST.json $model $LOADED" << 'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
assert m["identity"]["model_sha256"] == sys.argv[2], "identity"
assert m["parameters"]["loaded"] == int(sys.argv[3]), f"loaded {m['parameters']['loaded']}"
assert m["calibration"] is None
print(f"restaged: identity {sys.argv[2][:12]}, loaded {m['parameters']['loaded']:,}, T = 1")
EOF
    echo "manifest $(on f "sha256sum < $PKG/MODEL_MANIFEST.json | cut -c1-64")" ;;
  ship)
    [ -n "$NODE" ] && [ "$NODE" != f ] || { echo "ship needs a target node other than F" >&2; exit 2; }
    hop f "$NODE" "$MD" "$NAME-r13d42143" ;;
  assets)
    [ -n "$NODE" ] || { echo "assets needs a node" >&2; exit 2; }
    on "$NODE" "umask 077; mkdir -p $R/logs $R/runs $R/parity"
    if on "$NODE" "test ! -e $R/../kit-87d4650b"; then hop c "$NODE" "$(dirname "$R")" kit-87d4650b; fi
    if on "$NODE" "test ! -e $R/panel-8"; then hop c "$NODE" "$R" panel-8; fi
    on "$NODE" "test \"\$(git -C $R/../kit-87d4650b rev-parse HEAD)\" = 87d4650b42b377c0291a89c1f1a879f9b31082bf" \
      || { echo "kit on node $NODE is not at 87d4650b" >&2; exit 3; }
    echo "node $NODE: kit 87d4650b and panel-8 in place" ;;
  lease | release)
    [ -n "$NODE" ] && [ -n "$GPUS" ] || { echo "$STAGE needs NODE and GPUS" >&2; exit 2; }
    for g in $GPUS; do
      case "$NODE:$g" in e:0 | e:1 | e:2 | e:3 | e:6 | e:7 | f:[2-7] | c:[1-7] | d:[4-7]) ;;
        *) echo "node $NODE GPU$g is not M17's, the eval fast lane's or the shared Index pool's" >&2; exit 2 ;;
      esac
    done
    if [ "$STAGE" = lease ]; then
      on "$NODE" "for g in $GPUS; do f=/data/dev2/leases/gpu\$g.lock/owner; mkdir -p \$(dirname \$f); \
        if grep -qs '^status=busy' \$f; then echo \"gpu\$g is busy (\$(head -3 \$f | tr '\n' ' '))\"; exit 3; fi; \
        printf 'track=eval-ix1\nstatus=released (eval fast lane, decoder M17 stage 2)\npurpose=IX1 dec-m17 eval-fast\nstart_utc=%s\n' \
        \"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\" > \$f; echo gpu\$g handed to the harness form; done"
    else
      on "$NODE" "docker ps --format '{{.Names}}' | grep -q '^ix1-.*dev2_0-4b-lhs' && { echo 'an M17 Index container is running'; exit 3; }; \
        for g in $GPUS; do printf 'track=dec-m17\nstatus=released\npurpose=decoder M17 (Index runs finished)\nlast_job_end_utc=%s\n' \
        \"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\" > /data/dev2/leases/gpu\$g.lock/owner; echo gpu\$g released; done"
    fi ;;
  pool)
    [ -n "$NODE" ] && [ -n "$GPUS" ] || { echo "pool needs NODE and GPUS" >&2; exit 2; }
    names=()
    for a in ${ARM//,/ }; do
      n=$(name_of "$a")
      on "$NODE" "test -f $MD/$n-r13d42143/MODEL_MANIFEST.json" || { echo "$n is not on node $NODE" >&2; exit 3; }
      names+=("$n")
    done
    on "$NODE" "test -f $S/v2/dec/ops/m17/m17_ixpool.py && test -f $R/panel-8/panel.json" || { echo "mirror or panel-8 missing on node $NODE" >&2; exit 2; }
    on "$NODE" "mkdir -p $R/logs && setsid nohup python3 $S/v2/dec/ops/m17/m17_ixpool.py --mirror $M --panel panel-8 \
      --gpus '$GPUS' ${names[*]} >> $R/logs/m17-pool-$NODE.log 2>&1 < /dev/null &"
    echo "$(date -u +%FT%TZ) pool on node $NODE GPUs ${GPUS// /,}: ${names[*]} (log logs/m17-pool-$NODE.log)" ;;
  status)
    on "$NODE" "tail -3 $R/logs/m17-pool-$NODE.log; python3 - $R/runs/$NAME" << 'EOF'
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
  stop-pool)
    on "$NODE" "pkill -f 'm17_ixpool.py --mirror' && echo stopped || echo 'no pool running'" ;;
  lanes)
    [ -n "$NODE" ] && [ -n "$GPUS" ] || { echo "lanes needs NODE and a plan" >&2; exit 2; }
    on "$NODE" "test -f $S/v2/dec/ops/m17/m17-lanes.sh && python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))[\"pass\"] else 1)' $R/parity/$NAME/parity.json" \
      || { echo "mirror lacks m17-lanes.sh, or $NAME has no passed parity gate on node $NODE" >&2; exit 3; }
    on "$NODE" "mkdir -p $R/runs/$NAME && echo panel-8 > $R/runs/$NAME/m17-panel && setsid nohup bash $S/v2/dec/ops/m17/m17-lanes.sh $M $NAME panel-8 '$GPUS' '${6:-}' \
      >> $R/logs/m17-lanes-$NAME-$NODE.log 2>&1 < /dev/null &"
    echo "$(date -u +%FT%TZ) $NAME lanes on node $NODE: $GPUS (after: ${6:-none})" ;;
  parity-copy)
    [ -n "$NODE" ] && [ "$NODE" != e ] || { echo "parity-copy needs a target other than node E" >&2; exit 2; }
    on e "python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))[\"pass\"] else 1)' $R/parity/$NAME/parity.json" \
      || { echo "$NAME has no passed parity gate on node E" >&2; exit 3; }
    on "$NODE" "umask 077; mkdir -p $R/parity $R/logs $R/runs"
    hop e "$NODE" "$R/parity" "$NAME" ;;
  relay)
    [ -n "$NODE" ] && [ "$NODE" != c ] || { echo "relay needs the run's node" >&2; exit 2; }
    on "$NODE" "for k in 0 1 2 3 4 5 6 7; do test \"\$(cat $R/runs/$NAME/shard-\$k/exit_code 2>/dev/null)\" = 0 || exit 1; done" \
      || { echo "not every shard of $NAME ended with exit code 0" >&2; exit 3; }
    on "$NODE" "! find $R/runs/$NAME -iname '*gold*' | grep -q ." || { echo "gold-named files in the run" >&2; exit 3; }
    hop "$NODE" c "$R/runs" "$NAME" lean ;;
  score)
    on c "test -f $S/v2/eval/ix1/score.sh" || { echo "mirror $SHA is not on node C" >&2; exit 2; }
    on c "test ! -e $R/runs/$NAME/merged" || { echo "$NAME is already scored" >&2; exit 3; }
    on c "cd $S && bash v2/eval/ix1/score.sh --src $M --model $NAME --size 4B --panel $R/panel-8 > $R/logs/m17-score-$NAME.log 2>&1"
    on c "python3 - $R/runs/$NAME/merged" << 'EOF'
import json, sys
r = json.load(open(f"{sys.argv[1]}/receipt.json"))
c = json.load(open(f"{sys.argv[1]}/compare.json"))
gate = {k: v for k, v in c.items() if "pass" in k or k == "scorer_gate"}
print(json.dumps({"rows": r["rows"], "statuses": r["statuses"], "gpu_hours": r["gpu_hours"],
                  "results_sha256": r["results_sha256"], "panel_run_ids_sha256": r["panel_run_ids_sha256"],
                  "scorers": gate}))
EOF
    ;;
  boot)
    on c "test -f $R/runs/$NAME/merged/results.jsonl && test -f $R/runs/$REF/merged/results.jsonl" || { echo "missing results" >&2; exit 3; }
    on c "test ! -e $R/runs/$NAME/m17-boot-full-vs-$VS.json && test ! -e $R/logs/m17-boot-$NAME.started" || { echo "boot of $NAME already started" >&2; exit 3; }
    on c "cd $S && PYTHONPATH=$S python3 -m v2.eval.ix1.family_delta --base $REF=$R/runs/$REF/merged/compare.json \
      --new $NAME=$R/runs/$NAME/merged/compare.json --out $R/runs/$NAME/family-delta-vs-$VS.json > /dev/null"
    boot="cd $R/.. && CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= PYTHONPATH=$S:$R/../kit-19ad28ec nice -n 10 \
      venv/bin/python -m v2.eval.ix1.paired_boot --suite-dir suite-0.2 --base $R/runs/$REF/merged/results.jsonl \
      --new $R/runs/$NAME/merged/results.jsonl --external $R/../external/index021-frontier-gap-2026-10-01.json \
      --replicates 2000 --seed 20261002 --workers 16"
    on c "umask 077; date -u +%FT%TZ > $R/logs/m17-boot-$NAME.started; \
      setsid nohup bash -c '$boot --out $R/runs/$NAME/m17-boot-full-vs-$VS.json; echo \$? > $R/logs/m17-boot-full-$NAME.exit' \
        > $R/logs/m17-boot-full-$NAME.log 2>&1 < /dev/null &"
    echo "$(date -u +%FT%TZ) $NAME bootstrap started on node C (CPU)" ;;
  fetch)
    on c "test \"\$(cat $R/logs/m17-boot-full-$NAME.exit 2>/dev/null)\" = 0" || { echo "bootstrap of $NAME not finished with exit 0" >&2; exit 3; }
    LOCAL=${M17_LOCAL:-$HOME/code/decision2-program/private/m17}/$NAME
    (umask 077 && mkdir -p "$LOCAL")
    for f in merged/compare.json merged/receipt.json merged/port.json merged/kit/index.json family-delta-vs-$VS.json \
      m17-boot-full-vs-$VS.json; do
      out=$(basename "$f"); [ "$f" = merged/kit/index.json ] && out=kit-index.json
      on c "cat $R/runs/$NAME/$f" > "$LOCAL/$out"
    done
    pnode=${NODE:-e}
    on "$pnode" "cat $R/parity/$NAME/parity.json" > "$LOCAL/parity.json"
    python3 - "$LOCAL" "$VS" "${EXCLUDE[@]}" << 'EOF'
import json, sys
from pathlib import Path
local, vs, exclude = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
d = json.loads((local / f"family-delta-vs-{vs}.json").read_text())
rows = d["benchmarks"]
excluded = [n for n in exclude if n in rows]
out = {"excluded": excluded, "weighted_delta_sum": d["weighted_delta_sum"],
       "transfer_only_weighted_delta": round(d["weighted_delta_sum"] - sum(rows[n]["weighted_delta"] for n in excluded), 3)}
(local / "transfer-only.json").write_text(json.dumps(out, indent=1) + "\n")
EOF
    chmod 700 "$LOCAL" && chmod 600 "$LOCAL"/*.json
    echo "private outputs in $LOCAL (never copy a value into a commit, record, gist or card)" ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
