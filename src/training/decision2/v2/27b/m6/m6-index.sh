#!/usr/bin/env bash
# ~27B M6 private Index run of one frozen formal finalist (workstation side; prereg amendment 4, the Index path of
# COORDINATION 2026-10-02 02:05: one Index run per frozen finalist, item 1' needs a significantly positive paired
# Index delta vs A20r; runs use the eval allowance on node D GPU4-7). IX1's harness (v2/eval/ix1: image host2, kit
# 87d4650b, panel-8, the 86-request parity gate, dual scoring) set up as IX1 ran the M5-L128 diagnostic: the frozen
# soup checkpoint restaged into DEV2.0-27B 4e89288d with the forward-budget runtime (fix2 package, e876fbe; T = 1),
# A20r's frozen autotune cache, the same panel. The mirror carries the ARM's DIAGNOSTIC entry in
# v2/eval/ix1/launch.sh and is on node D (mirror_to_node.sh ... node-d). Index values stay in node D
# /data/dev2/private/eval/index021/ix1/ and the local private folder; this script prints none.
# Usage: m6-index.sh MIRROR_SHA ARM STAGE
#   stage    node B m6/ARM/checkpoint -> node D /data/dev2/models/ix1/m6/ARM-ckpt over node B's transfer key (SHA-256
#            lists equal), then v2.eval.ix1.restage -> /data/dev2/models/ix1/m6/ARM-re876fbe with the model SHA-256 of
#            m6/ARM/package/PACKAGE.json; checks the loaded count (27,497,508,864) and the identity
#   control  once for M6 (any ARM): A20r's own package through the same runtime (DEV2.0-27B-budget) read by its entry
#            point over the 86 compatibility requests on node D GPU4, vs IX1's A20r kit results -> must pass
#   parity   launch.sh parity on node D GPU4 (detached; waits for it, ~0.1 GPU-h); parity/ARM/parity.json must pass
#   run      m6-index-run.sh detached on node D (log ix1/logs/m6-index-ARM.log): 8 shards on the GPUs of
#            M6_INDEX_GPUS (default "4 5 6 7"; a GPU another track's Index run holds is left out)
#   status   per shard: records written, ended, exit code; GPU-h so far
#   score    after all 8 shards ended 0: score.sh (merge, port + kit scoring, compare incl. the frontier peer),
#            family_delta vs A20r's IX1 run (merged-budget) and vs M5-L128's, and the paired bootstrap vs A20r
#            (paired_boot: 2,000 replicates, seed 20261002; item 1' (b) = its 95% lower bound > 0); private outputs
#            copied to ~/code/decision2-program/private/m6/ARM/ (mode 700)
#   audit    once for M6 (any ARM; CPU): v2.eval.ix1.contamination of a20ib12pn (it contains every arm's rows) and
#            a20ib1x against the panel -> ix1/runs/m6-audit (the card's "no overlap with Index test items")
#   release  node D GPU4-7 owner files that launch.sh wrote for ARM -> status released (no ARM container running)
set -euo pipefail
SHA=${1:?MIRROR_SHA} ARM=${2:?ARM} STAGE=${3:?STAGE}
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
[[ "$ARM" =~ ^M6-(IB|IBX|IB2|IB2PN)$ ]] || { echo "bad ARM $ARM" >&2; exit 2; }
NODES=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
B=$(grep '^node-b=' "$NODES" | cut -d= -f2-) D=$(grep '^node-d=' "$NODES" | cut -d= -f2-)
onb() { ssh -o BatchMode=yes "$B" "$@"; }
ond() { ssh -o BatchMode=yes "$D" "$@"; }
M=/data/dev2/src/$SHA-src_training_decision2
S=$M/src/training/decision2
R=/data/dev2/private/eval/index021/ix1
R6=/data/dev2/runs/27b/m6
MD=/data/dev2/models/ix1/m6
FIX2=/data/dev2/models/ix1/fix2/DEV2.0-27B-4e89288d-re876fbe
PKG=$MD/$ARM-re876fbe CK=$MD/$ARM-ckpt
LOADED=27497508864
LOCAL=${M6_INDEX_LOCAL:-$HOME/code/decision2-program/private/m6}/$ARM
ond "test -f $S/v2/eval/ix1/launch.sh" || { echo "mirror $SHA is not on node D" >&2; exit 2; }
ond "grep -q '^  \[$ARM\]=\"DEV2.0-27B [0-9a-f]* $PKG\"' $S/v2/eval/ix1/launch.sh" ||
  { echo "mirror $SHA has no DIAGNOSTIC entry $ARM -> $PKG" >&2; exit 2; }
sums() { echo "cd $1 && find . -type f | sort | xargs -P 8 -n 4 sha256sum | sort -k2"; }
case "$STAGE" in
  stage)
    onb "test -f $R6/$ARM/package/PACKAGE.json && test -f $R6/$ARM/checkpoint/soup_manifest.json" ||
      { echo "no frozen package / soup for $ARM on node B" >&2; exit 3; }
    model=$(onb "python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[\"model_sha256\"])' $R6/$ARM/package/PACKAGE.json")
    [[ "$model" =~ ^[0-9a-f]{64}$ ]] || { echo "bad model SHA-256 in PACKAGE.json" >&2; exit 3; }
    ond "test ! -e $PKG" || { echo "$PKG exists: refusing to overwrite" >&2; exit 3; }
    ond "umask 077; mkdir -p $MD"
    echo "$(date -u +%FT%TZ) $ARM: copying the soup checkpoint node B -> node D"
    onb "rsync -a -e 'ssh -i /root/.ssh/d2_temp_cd -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes' \
      $R6/$ARM/checkpoint/ root@${D#*@}:$CK/"
    b=$(onb "$(sums "$R6/$ARM/checkpoint")") d=$(ond "$(sums "$CK")")
    [ -n "$b" ] && [ "$b" = "$d" ] || { echo "node D checkpoint differs from node B's" >&2; exit 3; }
    echo "checkpoint: $(wc -l <<< "$b") files, SHA-256 lists equal"
    ond "cd $S && PYTHONPATH=$S python3 -m v2.eval.ix1.restage --package $FIX2 --out $PKG --checkpoint $CK --model-sha256 $model"
    ond "python3 - $PKG/MODEL_MANIFEST.json $model $LOADED" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
assert m["identity"]["model_sha256"] == sys.argv[2], "identity"
assert m["parameters"]["loaded"] == int(sys.argv[3]), f"loaded {m['parameters']['loaded']}"
assert m["calibration"] is None
print(f"restaged: identity {sys.argv[2][:12]}, loaded {m['parameters']['loaded']:,}, T = 1")
EOF
    echo "manifest $(ond "sha256sum < $PKG/MODEL_MANIFEST.json | cut -c1-64")" ;;
  control)
    C=$R/runs/DEV2.0-27B-budget-control
    ond "test ! -e $C/control.json" || { echo "the restage control already ran ($C)" >&2; exit 3; }
    ond "cd $S && bash v2/eval/ix1/launch.sh ref --src $M --model DEV2.0-27B-budget --gpu 4 --run $C \
      --rows $R/panel-8/compat-86.gold-free.jsonl.gz --cache $R/parity/DEV2.0-27B/cache-frozen > $R/logs/m6-control.log 2>&1 && \
      PYTHONPATH=$S python3 -m v2.eval.ix1.parity --kit $R/parity/DEV2.0-27B/kit/results.jsonl --ref $C/ref.jsonl \
        --out $C/control.json > /dev/null"
    ond "python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({k: d[k] for k in (\"pass\", \"requests\", \"statuses\", \"max_abs_dp\")})); sys.exit(0 if d[\"pass\"] else 1)' $C/control.json" ;;
  audit)
    A6=$R/runs/m6-audit X6=/data/dev2/private/27b/m6-data
    ond "test ! -e $R/logs/m6-audit.exit" || { echo "the M6 audit already ran ($A6)" >&2; exit 3; }
    ond "mkdir -p $R/logs; setsid nohup bash -c 'cd $S && PYTHONHASHSEED=0 PYTHONPATH=$S python3 -m v2.eval.ix1.contamination \
      --panel $R/panel-8 --train a20ib12pn=$X6/mixtures-m6pn-1/a20ib12pn.train.jsonl \
      --train a20ib1x=$X6/mixtures-m6-1/a20ib1x.train.jsonl --workers 24 --out $A6; echo \$? > $R/logs/m6-audit.exit' \
      > $R/logs/m6-audit.log 2>&1 < /dev/null &"
    echo "$(date -u +%FT%TZ) M6 contamination audit started on node D (CPU)"
    until ond "test -f $R/logs/m6-audit.exit"; do sleep 60; done
    ond "echo exit \$(cat $R/logs/m6-audit.exit); tail -n 3 $R/logs/m6-audit.log"
    (umask 077 && mkdir -p "${LOCAL%/*}/audit")
    ond "cat $A6/audit.json" > "${LOCAL%/*}/audit/audit.json" ;;
  parity)
    ond "test -f $PKG/MODEL_MANIFEST.json" || { echo "run stage first" >&2; exit 3; }
    ond "test ! -e $R/logs/m6-parity-$ARM.exit" || { echo "parity of $ARM already ran" >&2; exit 3; }
    ond "mkdir -p $R/logs; setsid nohup bash -c 'cd $S && bash v2/eval/ix1/launch.sh parity --src $M --model $ARM --gpu 4 \
      --run $R/parity/$ARM --rows $R/panel-8/compat-86.gold-free.jsonl.gz; echo \$? > $R/logs/m6-parity-$ARM.exit' \
      > $R/logs/m6-parity-$ARM.log 2>&1 < /dev/null &"
    echo "$(date -u +%FT%TZ) $ARM parity gate started on node D GPU4 (two 27B passes, ~10 min)"
    until ond "test -f $R/logs/m6-parity-$ARM.exit"; do sleep 60; done
    ond "cat $R/logs/m6-parity-$ARM.exit; python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({k: d[k] for k in (\"pass\", \"requests\", \"statuses\", \"max_abs_dp\")})); sys.exit(0 if d[\"pass\"] else 1)' $R/parity/$ARM/parity.json" ;;
  run)
    ond "python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))[\"pass\"] else 1)' $R/parity/$ARM/parity.json" ||
      { echo "parity gate missing or failed" >&2; exit 3; }
    GPUS=${M6_INDEX_GPUS:-4 5 6 7}
    [[ "$GPUS" =~ ^[4-7]( [4-7]){0,3}$ ]] || { echo "M6_INDEX_GPUS: node D GPU4-7 only, not '$GPUS'" >&2; exit 2; }
    ond "mkdir -p $R/logs; setsid nohup bash $S/v2/27b/m6/m6-index-run.sh $SHA $ARM $GPUS > $R/logs/m6-index-$ARM.log 2>&1 < /dev/null & echo m6-index-run \$!"
    sleep 20
    ond "head -n 3 $R/logs/m6-index-$ARM.log" ;;
  status)
    ond "python3 - $R/runs/$ARM" <<'EOF'
import json, os, sys, time
run, total = sys.argv[1], 0.0
for k in range(8):
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
    ond "tail -n 4 $R/logs/m6-index-$ARM.log 2> /dev/null || echo 'no run log yet'" ;;
  score)
    ond "for k in 0 1 2 3 4 5 6 7; do test \"\$(cat $R/runs/$ARM/shard-\$k/exit_code 2>/dev/null)\" = 0 || exit 1; done" ||
      { echo "not every shard ended with exit code 0" >&2; exit 3; }
    ond "cd $S && bash v2/eval/ix1/score.sh --src $M --model $ARM --size 27B --panel $R/panel-8 > $R/logs/m6-score-$ARM.log 2>&1 && \
      PYTHONPATH=$S python3 -m v2.eval.ix1.family_delta --base A20r=$R/runs/DEV2.0-27B/merged-budget/compare.json \
        --new $ARM=$R/runs/$ARM/merged/compare.json --out $R/runs/$ARM/family-delta-vs-a20r.json > /dev/null && \
      PYTHONPATH=$S python3 -m v2.eval.ix1.family_delta --base M5-L128=$R/runs/M5-L128/merged/compare.json \
        --new $ARM=$R/runs/$ARM/merged/compare.json --out $R/runs/$ARM/family-delta-vs-m5-l128.json > /dev/null && \
      cd $R/.. && PYTHONPATH=$S:\$PWD/kit-19ad28ec venv/bin/python -m v2.eval.ix1.paired_boot --suite-dir suite-0.2 \
        --base $R/runs/DEV2.0-27B/merged-budget/results.jsonl --new $R/runs/$ARM/merged/results.jsonl \
        --external $R/../external/index021-frontier-gap-2026-10-01.json --replicates 2000 --seed 20261002 --workers 24 \
        --out $R/runs/$ARM/paired-boot-vs-a20r.json > $R/logs/m6-boot-$ARM.log 2>&1"
    (umask 077 && mkdir -p "$LOCAL")
    for f in merged/compare.json merged/receipt.json merged/port.json family-delta-vs-a20r.json \
      family-delta-vs-m5-l128.json paired-boot-vs-a20r.json; do
      ond "cat $R/runs/$ARM/$f" > "$LOCAL/$(basename "$f")"
    done
    ond "cat $R/runs/DEV2.0-27B/merged-budget/compare.json" > "$LOCAL/compare-a20r.json"
    ond "cat $R/parity/$ARM/parity.json" > "$LOCAL/parity.json"
    ond "cat $R/runs/DEV2.0-27B-budget-control/control.json 2> /dev/null" > "$LOCAL/control-a20r.json" || true
    chmod 600 "$LOCAL"/*.json
    python3 - "$LOCAL/receipt.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
print(json.dumps({k: r[k] for k in ("rows", "statuses", "gpu_hours", "results_sha256", "panel_run_ids_sha256")}))
EOF
    echo "private outputs in $LOCAL (never copy a value into a commit, record, gist or card)" ;;
  release)
    ond "docker ps --format '{{.Names}}' | grep -q '^ix1-$(tr 'A-Z.' 'a-z_' <<< "$ARM")-'" &&
      { echo "an $ARM Index container is still running" >&2; exit 3; }
    ond "for g in 4 5 6 7; do f=/data/dev2/leases/gpu\$g.lock/owner; grep -q 'IX1 .* $ARM\$' \$f 2>/dev/null || continue; \
      printf 'track=eval-ix1\nstatus=released (27B M6 Index run of $ARM done)\nlast_job_end_utc=%s\n' \
      \"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\" > \$f; echo gpu\$g released; done" ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
