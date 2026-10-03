#!/usr/bin/env bash
# Decoder M17b (4B owner, M17 continuation) ops, workstation side: ssh control only. Index values stay in the node
# private directories and ~/code/decision2-program/private/m17b/; this script prints none.
#
# Usage: m17b-ops.sh MIRROR_SHA STAGE
#   audit6-run     node C, CPU, detached: one row-level Index contamination audit (v2.eval.ix1.contamination, the IX1
#                  method with 200 planted controls, panel-7) of the ten distinct TRAIN files of wave 6's members
#                  (prereg dec-m17b-wave6-prereg-2026-10-03.md), from the hash-checked copies of M17's audits 1-4 and
#                  the arm factory's wave-2 audit -> ix1/audit/m17b-w6 (release IF3)
#   audit6-status  exit code and counts: Index rows, planted found / missed, per set lines, duplicate and item rows
set -euo pipefail
SHA=${1:?MIRROR_SHA} STAGE=${2:?STAGE}
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "MIRROR_SHA must be a full commit SHA" >&2; exit 2; }
NODES=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
addr() { awk -F= -v k="node-$1" '$1 == k { print substr($0, length(k) + 2); exit }' "$NODES"; }
on() { local n=$1; shift; ssh -o BatchMode=yes -o ConnectTimeout=30 "$(addr "$n")" "$@"; }
M=/data/dev2/src/$SHA-src_training_decision2
S=$M/src/training/decision2
R=/data/dev2/private/eval/index021/ix1
A6=$R/audit/m17b-w6
case "$STAGE" in
  audit6-run)
    on c "test -f $S/v2/eval/ix1/contamination.py && test ! -e $A6/out" || { echo "no mirror, or already run" >&2; exit 3; }
    trains=""
    for x in "m17 4b-LHS10SD" "m17 4b-LHS17SD" "m17s2 4b-LHS23SD" "m17s2 4b-LHS17IB4" "m17s2 4b-LHS17IB4X" \
      "m17sdml 4b-LHA10SDML" "m17s4 4b-SDMLIB4" "m17s4 4b-LHS17ML" "af4b-w2 4b-LHS17IB4ML" "af4b-w2 4b-LHS23IB4"; do
      read -r d a <<< "$x"
      d=$R/audit/$d/train
      want=$(on c "awk -v a=$a '\$1 == a { print \$2 }' $d/FILES.txt")
      [[ "$want" =~ ^[0-9a-f]{64}$ ]] && [ "$(on c "sha256sum < $d/$a.train.jsonl | cut -c1-64")" = "$want" ] \
        || { echo "$a: no hash-checked TRAIN copy in $d" >&2; exit 3; }
      echo "$a ${want:0:12} ok"
      trains="$trains --train $a=$d/$a.train.jsonl"
    done
    on c "umask 077; mkdir -p $A6 && setsid nohup bash -c 'cd $S && CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= PYTHONHASHSEED=0 PYTHONPATH=$S nice -n 10 \
      python3 -m v2.eval.ix1.contamination --panel $R/panel-7 $trains --workers 32 --out $A6/out; echo \$? > $A6/exit' > $A6/audit.log 2>&1 < /dev/null &"
    echo "$(date -u +%FT%TZ) audit6 started on node C (CPU)" ;;
  audit6-status)
    on c "cat $A6/exit 2>/dev/null || echo running; test -f $A6/out/audit.json && python3 - $A6/out/audit.json" << 'EOF'
import json, sys
a = json.load(open(sys.argv[1]))
print(json.dumps({"index_rows": a["index_rows"], "planted": a["planted_control"]["found"],
                  "missed": len(a["planted_control"]["missed"]),
                  "sets": {k: {"lines": v["training_lines"], "duplicate_rows": v["duplicate_rows"],
                               "item_rows": v["item_rows"]} for k, v in a["training_sets"].items()}}))
EOF
    ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
