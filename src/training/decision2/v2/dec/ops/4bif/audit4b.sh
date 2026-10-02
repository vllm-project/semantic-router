#!/usr/bin/env bash
# 4B Index-first release (workstation side): the row-level Index contamination audit (v2.eval.ix1.contamination, the
# IX1 method: NFKC / casefold / \w+, exact leaves and sentences, exhaustive 13-grams, template filter, 200 planted
# controls) of the candidates' training file on node C, CPU only. Every 4B candidate (M13 4b-LHA10SD, M14 4b-LHA10UP,
# the M16 interpolations of the released LH with 4b-LHA10SD) trained on M12's locked 4b-LHA10 TRAIN (d41cdd1a...,
# 72,847 rows: the released LH mixture as its first 58,739 rows, then 14,108 IB rows); M15 4b-LHA10SDML adds only
# copies of released rows. The released LH block (the first 58,739 rows) is audited in the same run for comparison.
# Outputs (private): node C /data/dev2/private/eval/index021/ix1/audit/4bif/out/{audit,items,duplicates}.json; this
# script prints counts only.
# Usage: audit4b.sh MIRROR_SHA stage | run | status
set -euo pipefail
SHA=${1:?MIRROR_SHA} STAGE=${2:?STAGE}
NODES=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
addr() { awk -F= -v k="node-$1" '$1 == k { print substr($0, length(k) + 2); exit }' "$NODES"; }
on() { local n=$1; shift; ssh -o BatchMode=yes -o ConnectTimeout=30 "$(addr "$n")" "$@"; }
KEY="-i /root/.ssh/d2_temp_cd -o BatchMode=yes -o ConnectTimeout=30"
TRAIN_B=/data/dev2/runs/dec/m14/inputs/m12/4b/4b-LHA10/train.jsonl
TRAIN_SHA=d41cdd1aa1e63b47394ed8fa61a87bf461cb4d2cea08cc7ff12f2e58b36dc9d5
LH_ROWS=58739
S=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2
R=/data/dev2/private/eval/index021/ix1
A=$R/audit/4bif
case "$STAGE" in
  stage)
    on c "test ! -e $A/train/4b-LHA10.train.jsonl" || { echo "already staged" >&2; exit 3; }
    on c "umask 077; mkdir -p $A/train"
    on b "rsync -a -e 'ssh $KEY' $TRAIN_B $(addr c):$A/train/4b-LHA10.train.jsonl"
    on c "cd $A/train && test \"\$(sha256sum < 4b-LHA10.train.jsonl | cut -c1-64)\" = $TRAIN_SHA && \
      head -n $LH_ROWS 4b-LHA10.train.jsonl > 4b-LH-base.train.jsonl && chmod 600 *.jsonl && \
      printf '4b-LHA10 %s rows %s\n4b-LH-base %s rows %s\n' $TRAIN_SHA \"\$(wc -l < 4b-LHA10.train.jsonl)\" \
        \"\$(sha256sum < 4b-LH-base.train.jsonl | cut -c1-64)\" \"\$(wc -l < 4b-LH-base.train.jsonl)\" | tee FILES.txt" ;;
  run)
    on c "test -f $S/v2/eval/ix1/contamination.py && test -f $A/train/FILES.txt && test ! -e $A/out" || { echo "not staged, or already run" >&2; exit 3; }
    on c "umask 077; setsid nohup bash -c 'cd $S && CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= PYTHONHASHSEED=0 PYTHONPATH=$S nice -n 10 \
      python3 -m v2.eval.ix1.contamination --panel $R/panel-7 --train 4b-LHA10=$A/train/4b-LHA10.train.jsonl \
      --train 4b-LH-base=$A/train/4b-LH-base.train.jsonl --workers 24 --out $A/out; echo \$? > $A/exit' > $A/audit.log 2>&1 < /dev/null &"
    echo "$(date -u +%FT%TZ) audit started on node C (CPU)" ;;
  status)
    on c "cat $A/exit 2>/dev/null || echo running; python3 - $A/out/audit.json" << 'EOF'
import json, sys
a = json.load(open(sys.argv[1]))
print(json.dumps({"index_rows": a["index_rows"], "planted": a["planted_control"]["found"],
                  "missed": len(a["planted_control"]["missed"]),
                  "sets": {k: {"lines": v["training_lines"], "duplicate_rows": v["duplicate_rows"],
                               "item_rows": v["item_rows"]} for k, v in a["training_sets"].items()}}))
EOF
    ;;
  *) echo "usage: audit4b.sh MIRROR_SHA stage | run | status" >&2; exit 2 ;;
esac
