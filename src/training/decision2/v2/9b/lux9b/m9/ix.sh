#!/usr/bin/env bash
# 9B M9 private Index diagnostic of the frozen stage-3 soup K-a13IB (follow-up A; workstation side). IX1's harness
# (v2/eval/ix1: image host2, kit 87d4650b, panel-7, the 86-request parity gate, dual scoring) as IX1 ran DEV2.0-9B:
# the FP32 soup restaged into DEV2.0-9B e51f9881 (the package IX1 scored; T = 1, calibration none), DEV2.0-9B's frozen
# autotune cache for the full run, the same 7-way panel on node C GPU1-7 (node C GPU0 never). Control K-a13-fp32:
# DEV2.0-9B's own FP32 soup (m4/K-a13-build/soup) restaged the same way and read by the package's entry point over
# the 86 compatibility requests against IX1's DEV2.0-9B kit parity results (shows the restage path answers as the
# released package). Index values stay in node C /data/dev2/private/eval/index021/ix1/runs/NAME/ and the local
# private folder; this script prints none.
# K-a13IB-bf16: the release weights (v2.release.bf16_copy of K-a13IB, release inputs dev2-9b-ka13ib/bf16) restaged
# the same way, for the release card's Index values on exactly the released weights (coordinator 2026-10-02 02:05).
# Usage: ix.sh MIRROR_SHA NAME STAGE      NAME: K-a13IB | K-a13IB-bf16 | K-a13-fp32
#   stage    node A soup -> node C /data/dev2/models/ix1/9b/NAME-ckpt over node A's transfer key (SHA-256 lists
#            equal), then v2.eval.ix1.restage -> /data/dev2/models/ix1/9b/NAME-e51f9881 with the soup's model SHA-256
#            (its M9 readout manifest; K-a13IB-bf16: its bf16-copy receipt); checks the loaded count (7,940,895,744) and the identity
#   control  K-a13-fp32 only: launch.sh ref on node C GPU1 over compat-86 with DEV2.0-9B's frozen cache, then
#            v2.eval.ix1.parity against parity/DEV2.0-9B/kit/results.jsonl -> runs/K-a13-fp32/control.json
#   parity   launch.sh parity on node C GPU1 (foreground on the node, ~0.1 GPU-h); parity/NAME/parity.json must pass
#   run      launch.sh run: 7 shards on node C GPU1-7 with DEV2.0-9B's frozen cache (detached containers)
#   status   per shard: records written, ended, exit code; GPU-h so far
#   score    after all 7 shards ended 0: score.sh (merge, port + kit, compare incl. the frontier peer), family_delta vs
#            DEV2.0-9B's IX1 run, the paired bootstrap (v2.eval.ix1.paired_boot); private outputs copied to
#            ~/code/decision2-program/private/9b-ka13ib/ (mode 700; K-a13IB-bf16 into its bf16/ subfolder)
#   transfer K-a13IB: the paired bootstrap without the benchmarks of its training families and formats (HoVer,
#            When2Call, iSarcasmEval, GSM8K, BPoMP; coordinator 2026-10-02 02:05, internal records only)
#   release  node C GPU1-7 owner files that launch.sh wrote for NAME -> status released (no NAME container running)
set -euo pipefail
SHA=${1:?MIRROR_SHA} NAME=${2:?NAME} STAGE=${3:?STAGE}
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
NODES=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
A=$(grep '^node-a=' "$NODES" | cut -d= -f2-) C=$(grep '^node-c=' "$NODES" | cut -d= -f2-)
ona() { ssh -o BatchMode=yes "$A" "$@"; }
onc() { ssh -o BatchMode=yes "$C" "$@"; }
case "$NAME" in
  K-a13IB) SOUP=/data/dev2/runs/9b/m9/soup/K-a13IB/build/K-a13IB MANIFEST=/data/dev2/runs/9b/m9/lines/K-a13IB ;;
  K-a13-fp32) SOUP=/data/dev2/runs/9b/m4/K-a13-build/soup MANIFEST=/data/dev2/runs/9b/m9/lines/C0 ;;
  K-a13IB-bf16) SOUP=/data/dev2/runs/release/inputs/dev2-9b-ka13ib/bf16/checkpoint MANIFEST= ;;
  *) echo "bad NAME $NAME" >&2; exit 2 ;;
esac
M=/data/dev2/src/$SHA-src_training_decision2
S=$M/src/training/decision2
R=/data/dev2/private/eval/index021/ix1
MD=/data/dev2/models/ix1/9b
BASEPKG=/data/dev2/models/ix1/DEV2.0-9B-e51f9881
PKG=$MD/$NAME-e51f9881 CK=$MD/$NAME-ckpt
CACHE=$R/parity/DEV2.0-9B/cache-frozen
GPUS="1 2 3 4 5 6 7"
LOADED=7940895744
LOCAL=${M9_IX_LOCAL:-$HOME/code/decision2-program/private/9b-ka13ib}
[ "$NAME" = K-a13IB-bf16 ] && LOCAL=$LOCAL/bf16
TRANSFER_EXCLUDE="HoVer When2Call iSarcasmEval GSM8K BPoMP"
onc "test -f $S/v2/eval/ix1/launch.sh" || { echo "mirror $SHA is not on node C" >&2; exit 2; }
onc "grep -q '^  \[$NAME\]=\"DEV2.0-9B [0-9a-f]* $PKG\"' $S/v2/eval/ix1/launch.sh" ||
  { echo "mirror $SHA has no DIAGNOSTIC entry $NAME -> $PKG" >&2; exit 2; }
sums() { echo "cd $1 && find . -type f | sort | xargs -P 8 -n 4 sha256sum | sort -k2"; }
case "$STAGE" in
  stage)
    if [ "$NAME" = K-a13IB-bf16 ]; then
      model=$(ona "python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[\"model_sha256\"])' \
        $SOUP/../bf16-copy.json")
    else
      model=$(ona "python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[\"model_sha256\"])' \
        $MANIFEST/dev/dev.predictions.jsonl.manifest.json")
    fi
    [[ "$model" =~ ^[0-9a-f]{64}$ ]] || { echo "bad model SHA-256 in the readout manifest" >&2; exit 3; }
    onc "test ! -e $PKG" || { echo "$PKG exists: refusing to overwrite" >&2; exit 3; }
    onc "umask 077; mkdir -p $MD"
    echo "$(date -u +%FT%TZ) $NAME: copying the soup node A -> node C"
    ona "rsync -a -e 'ssh -i /root/.ssh/d2_temp_cd -o BatchMode=yes' $SOUP/ $C:$CK/"
    a=$(ona "$(sums "$SOUP")") c=$(onc "$(sums "$CK")")
    [ -n "$a" ] && [ "$a" = "$c" ] || { echo "node C copy differs from node A's" >&2; exit 3; }
    echo "soup: $(wc -l <<< "$a") files, SHA-256 lists equal"
    onc "cd $S && PYTHONPATH=$S python3 -m v2.eval.ix1.restage --package $BASEPKG --out $PKG --checkpoint $CK --model-sha256 $model"
    onc "python3 - $PKG/MODEL_MANIFEST.json $model $LOADED" << 'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
assert m["identity"]["model_sha256"] == sys.argv[2], "identity"
assert m["parameters"]["loaded"] == int(sys.argv[3]), f"loaded {m['parameters']['loaded']}"
assert m["calibration"] is None
print(f"restaged: identity {sys.argv[2][:12]}, loaded {m['parameters']['loaded']:,}, T = 1")
EOF
    echo "manifest $(onc "sha256sum < $PKG/MODEL_MANIFEST.json | cut -c1-64")" ;;
  control)
    [ "$NAME" = K-a13-fp32 ] || { echo "control is for K-a13-fp32" >&2; exit 2; }
    onc "test ! -e $R/runs/$NAME/ref.jsonl" || { echo "control already ran" >&2; exit 3; }
    onc "cd $S && bash v2/eval/ix1/launch.sh ref --src $M --model $NAME --gpu 1 --run $R/runs/$NAME \
      --rows $R/panel-8/compat-86.gold-free.jsonl.gz --cache $CACHE > $R/logs/m9-control-$NAME.log 2>&1"
    onc "cd $S && PYTHONPATH=$S python3 -m v2.eval.ix1.parity --kit $R/parity/DEV2.0-9B/kit/results.jsonl \
      --ref $R/runs/$NAME/ref.jsonl --out $R/runs/$NAME/control.json" ;;
  parity)
    onc "test -f $PKG/MODEL_MANIFEST.json" || { echo "run stage first" >&2; exit 3; }
    onc "test ! -e $R/logs/m9-parity-$NAME.exit" || { echo "parity of $NAME already ran" >&2; exit 3; }
    onc "mkdir -p $R/logs; setsid nohup bash -c 'cd $S && bash v2/eval/ix1/launch.sh parity --src $M --model $NAME --gpu 1 \
      --run $R/parity/$NAME --rows $R/panel-8/compat-86.gold-free.jsonl.gz; echo \$? > $R/logs/m9-parity-$NAME.exit' \
      > $R/logs/m9-parity-$NAME.log 2>&1 < /dev/null &"
    echo "$(date -u +%FT%TZ) $NAME parity gate started on node C GPU1"
    until onc "test -f $R/logs/m9-parity-$NAME.exit"; do sleep 60; done
    onc "cat $R/logs/m9-parity-$NAME.exit; python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({k: d[k] for k in (\"pass\", \"requests\", \"statuses\", \"max_abs_dp\")})); sys.exit(0 if d[\"pass\"] else 1)' $R/parity/$NAME/parity.json" ;;
  run)
    onc "python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))[\"pass\"] else 1)' $R/parity/$NAME/parity.json" ||
      { echo "parity gate missing or failed" >&2; exit 3; }
    onc "cd $S && bash v2/eval/ix1/launch.sh run --src $M --model $NAME --gpus '$GPUS' --run $R/runs/$NAME \
      --rows-dir $R/panel-7 --cache $CACHE" ;;
  status)
    onc "python3 - $R/runs/$NAME" << 'EOF'
import os, sys, time
run, total = sys.argv[1], 0.0
for k in range(7):
    w = f"{run}/shard-{k}"
    if not os.path.exists(f"{w}/launched") and not os.path.exists(f"{w}/end_epoch"):
        print(f"shard {k}: not started")
        continue
    rows = sum(1 for _ in open(f"{w}/results.jsonl")) if os.path.exists(f"{w}/results.jsonl") else 0
    start = float(open(f"{w}/start_epoch").read()) if os.path.exists(f"{w}/start_epoch") else None
    end = float(open(f"{w}/end_epoch").read()) if os.path.exists(f"{w}/end_epoch") else None
    code = open(f"{w}/exit_code").read().strip() if os.path.exists(f"{w}/exit_code") else "-"
    if start:
        total += ((end or time.time()) - start) / 3600
    print(f"shard {k}: {rows} records, {'ended exit ' + code if end else 'running'}")
print(f"GPU-h so far (current intervals) {total:.2f}")
EOF
    ;;
  score)
    [ "$NAME" = K-a13IB ] || [ "$NAME" = K-a13IB-bf16 ] || { echo "score is for K-a13IB and K-a13IB-bf16" >&2; exit 2; }
    onc "for k in 0 1 2 3 4 5 6; do test \"\$(cat $R/runs/$NAME/shard-\$k/exit_code 2>/dev/null)\" = 0 || exit 1; done" ||
      { echo "not every shard ended with exit code 0" >&2; exit 3; }
    onc "cd $S && bash v2/eval/ix1/score.sh --src $M --model $NAME --size 9B --panel $R/panel-7 > $R/logs/m9-score-$NAME.log 2>&1 && \
      PYTHONPATH=$S python3 -m v2.eval.ix1.family_delta --base DEV2.0-9B=$R/runs/DEV2.0-9B/merged/compare.json \
        --new $NAME=$R/runs/$NAME/merged/compare.json --out $R/runs/$NAME/family-delta-vs-dev20-9b.json > /dev/null && \
      cd $R/.. && PYTHONPATH=$S:\$PWD/kit-19ad28ec venv/bin/python -m v2.eval.ix1.paired_boot --suite-dir suite-0.2 \
        --base $R/runs/DEV2.0-9B/merged/results.jsonl --new $R/runs/$NAME/merged/results.jsonl \
        --external $R/../external/index021-frontier-gap-2026-10-01.json --replicates 2000 --seed 20261002 --workers 24 \
        --out $R/runs/$NAME/paired-boot-vs-dev20-9b.json > $R/logs/m9-boot-$NAME.log 2>&1"
    (umask 077 && mkdir -p "$LOCAL")
    for f in merged/compare.json merged/receipt.json merged/port.json family-delta-vs-dev20-9b.json \
      paired-boot-vs-dev20-9b.json; do
      onc "cat $R/runs/$NAME/$f" > "$LOCAL/$(basename "$f")"
    done
    onc "cat $R/runs/DEV2.0-9B/merged/compare.json" > "$LOCAL/compare-dev20-9b.json"
    onc "cat $R/parity/$NAME/parity.json" > "$LOCAL/parity.json"
    onc "cat $R/runs/K-a13-fp32/control.json 2> /dev/null" > "$LOCAL/control-k-a13-fp32.json" || true
    chmod 600 "$LOCAL"/*.json
    python3 - "$LOCAL/receipt.json" << 'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
print(json.dumps({k: r[k] for k in ("rows", "statuses", "gpu_hours", "results_sha256", "panel_run_ids_sha256") if k in r}))
EOF
    echo "private outputs in $LOCAL (never copy a value into a commit, record, gist or card)" ;;
  transfer)
    [ "$NAME" = K-a13IB ] || { echo "transfer is for K-a13IB" >&2; exit 2; }
    onc "test -f $R/runs/$NAME/paired-boot-vs-dev20-9b.json" || { echo "score $NAME first" >&2; exit 3; }
    # shellcheck disable=SC2086
    onc "cd $R/.. && PYTHONPATH=$S:\$PWD/kit-19ad28ec venv/bin/python -m v2.eval.ix1.paired_boot --suite-dir suite-0.2 \
        --base $R/runs/DEV2.0-9B/merged/results.jsonl --new $R/runs/$NAME/merged/results.jsonl \
        --external $R/../external/index021-frontier-gap-2026-10-01.json --replicates 2000 --seed 20261002 --workers 24 \
        --exclude $TRANSFER_EXCLUDE --out $R/runs/$NAME/paired-boot-transfer-vs-dev20-9b.json > $R/logs/m9-boot-transfer-$NAME.log 2>&1"
    (umask 077 && mkdir -p "$LOCAL")
    onc "cat $R/runs/$NAME/paired-boot-transfer-vs-dev20-9b.json" > "$LOCAL/paired-boot-transfer-vs-dev20-9b.json"
    chmod 600 "$LOCAL/paired-boot-transfer-vs-dev20-9b.json"
    echo "private output $LOCAL/paired-boot-transfer-vs-dev20-9b.json (never copy a value into a commit, record, gist or card)" ;;
  release)
    onc "docker ps --format '{{.Names}}' | grep -q '^ix1-$(tr 'A-Z.' 'a-z_' <<< "$NAME")-'" &&
      { echo "a $NAME Index container is still running" >&2; exit 3; }
    onc "for g in $GPUS; do f=/data/dev2/leases/gpu\$g.lock/owner; grep -q 'IX1 .* $NAME\$' \$f 2>/dev/null || continue; \
      printf 'track=eval-ix1\nstatus=released (9B M9 Index diagnostic of $NAME done)\nlast_job_end_utc=%s\n' \
      \"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\" > \$f; echo gpu\$g released; done" ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
