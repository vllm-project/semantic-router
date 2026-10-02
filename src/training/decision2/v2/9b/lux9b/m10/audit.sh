#!/usr/bin/env bash
# 9B M10 amendment 7 row-level Index contamination audit (workstation side; node C, CPU only): v2.eval.ix1.contamination
# (the IX1 method: NFKC / casefold / \w+, exact leaves and sentences, exhaustive 13-grams, template filter, 200 planted
# controls) of every TRAIN file a continuation candidate can contain, in one run: KIB4 (also the TRAIN of the arm
# factory's KIB4 s4 / s5, KIB4W2 and KIB4L2 seeds), KX, KSW (the M10 audit's copies), KIB = K-a13IB's TRAIN (M9 lock
# 2cd09292..., for points that contain K-a13IB's arm soup) and the arm factory's KIB4R (its own audit's copy). Each file
# is read in place after its SHA-256 is checked against its lock.
# Outputs (private): node C ix1/audit/m10c/{FILES.txt,out/{audit,items,duplicates}.json}; this script prints counts only.
# Usage: audit.sh MIRROR_SHA run | status
set -euo pipefail
SHA=${1:?MIRROR_SHA} STAGE=${2:?STAGE}
NODES=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
addr() { awk -F= -v k="node-$1" '$1 == k { print substr($0, length(k) + 2); exit }' "$NODES"; }
on() { local n=$1; shift; ssh -o BatchMode=yes -o ConnectTimeout=30 "$(addr "$n")" "$@"; }
S=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2
R=/data/dev2/private/eval/index021/ix1
A=$R/audit/m10c
SETS="KIB4=$R/audit/m10/train/KIB4.train.jsonl=2e72bcfd191832dc
KX=$R/audit/m10/train/KX.train.jsonl=f1d9ecf8a0977d78
KSW=$R/audit/m10/train/KSW.train.jsonl=e5cc44bb45fc0a04
KIB=/data/dev2/runs/9b/m9/data/kib/train.jsonl=2cd09292580450a7
KIB4R=$R/audit/af9b-kib4r/train/KIB4R.train.jsonl=e95e32ce696ba823"
case "$STAGE" in
  run)
    on c "test -f $S/v2/eval/ix1/contamination.py && test ! -e $A" < /dev/null || { echo "no mirror, or already run" >&2; exit 3; }
    args=""
    on c "umask 077; mkdir -p $A" < /dev/null
    while IFS='=' read -r name path want; do
      got=$(on c "sha256sum < $path | cut -c1-16" < /dev/null)
      [ "$got" = "$want" ] || { echo "$name: $path is $got, not $want" >&2; on c "rm -rf $A" < /dev/null; exit 3; }
      on c "printf '%s %s rows %s %s\n' $name \$(sha256sum < $path | cut -c1-64) \$(wc -l < $path) $path >> $A/FILES.txt" < /dev/null
      args+=" --train $name=$path"
    done <<< "$SETS"
    on c "umask 077; setsid nohup bash -c 'cd $S && CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= PYTHONHASHSEED=0 PYTHONPATH=$S nice -n 10 \
      python3 -m v2.eval.ix1.contamination --panel $R/panel-7$args --workers 24 --out $A/out; echo \$? > $A/exit' \
      > $A/audit.log 2>&1 < /dev/null &" < /dev/null
    echo "$(date -u +%FT%TZ) audit started on node C (CPU)" ;;
  status)
    on c "cat $A/exit 2> /dev/null || echo running; python3 - $A/out/audit.json" << 'EOF'
import json, sys
a = json.load(open(sys.argv[1]))
print(json.dumps({"index_rows": a["index_rows"], "planted": a["planted_control"]["found"],
                  "of": a["planted_control"]["planted"], "missed": len(a["planted_control"]["missed"]),
                  "sets": {k: {"lines": v["training_lines"], "duplicate_rows": v["duplicate_rows"],
                               "item_rows": v["item_rows"]} for k, v in a["training_sets"].items()}}))
EOF
    ;;
  *) echo "usage: audit.sh MIRROR_SHA run | status" >&2; exit 2 ;;
esac
