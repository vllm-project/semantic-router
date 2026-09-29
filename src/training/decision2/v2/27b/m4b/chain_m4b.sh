#!/usr/bin/env bash
# M4b training chain on node B GPU0-2 (host side): the arm-seeds in the prereg's launch order, each through
# run_ff_arm.sh (onestep, reload, full). A failed arm-seed stops only itself (recorded in its receipts and
# CHAIN.jsonl); the chain moves on. Before each arm-seed the milestone's GPU-hours so far (every receipt under
# /data/dev2/runs/27b/m4b) plus that arm-seed's cap plus the evaluation reserve must stay within the 36 GPU-hour
# ceiling, else the chain stops there (prereg "Budget").
# Usage: chain_m4b.sh MIRROR_SHA [ARM-SEED ...]   (default: A1-s1 A2-s1 A1-s2 A2-s2 A3-s1)
set -uo pipefail
echo "$(date -u +%FT%TZ) m4b chain $* start"
SHA=$1
shift
ITEMS=("$@")
[ "${#ITEMS[@]}" -gt 0 ] || ITEMS=(A1-s1 A2-s1 A1-s2 A2-s2 A3-s1)
CODE=/data/dev2/src/$SHA/src/training/decision2
[ -d "$CODE" ] || CODE=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2
[ -f "$CODE/v2/27b/m4b/run_ff_arm.sh" ] || { echo "no m4b code in mirror $SHA" >&2; exit 2; }
ROOT=/data/dev2/runs/27b/m4b
TEACHERS=${TEACHERS:-/data/dev2/private/27b/m4b-data/teachers-1}
CEILING=${CEILING:-36}
RESERVE=${RESERVE:-6.0}
ARM_CAP=${ARM_CAP:-8.3}
LOG=$ROOT/CHAIN.jsonl

used() {  # GPU-hours of every M4b receipt so far, archived attempts included
  python3 -c "
import glob, json, sys
paths = glob.glob(sys.argv[1] + '/**/*.json', recursive=True)
total = 0.0
for p in paths:
    try:
        r = json.load(open(p))
    except Exception:
        continue
    if isinstance(r, dict) and 'gpu_hours' in r and 'exit_code' in r and 'container_id' in r:
        total += float(r['gpu_hours'])
print(f'{total:.4f}')" "$ROOT"
}
record() {  # ITEM STATUS NOTE
  printf '{"utc": "%s", "item": "%s", "status": "%s", "gpu_hours_total": %s, "note": "%s"}\n' \
    "$(date -u +%FT%TZ)" "$1" "$2" "$(used)" "$3" >> "$LOG"
}

for item in "${ITEMS[@]}"; do
  arm=${item%-*} seed=${item#*-}
  case $arm in
    A1) teacher="" kl=0 ;;
    A2) teacher=$TEACHERS/teacher-lux.jsonl kl=1.0 ;;
    A3) teacher=$TEACHERS/teacher-aj.jsonl kl=1.0 ;;
    *) echo "unknown arm $arm" >&2; exit 2 ;;
  esac
  total=$(used)
  if ! python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) + float(sys.argv[2]) + float(sys.argv[3]) <= float(sys.argv[4]) + 1e-9 else 1)" \
    "$total" "$ARM_CAP" "$RESERVE" "$CEILING"; then
    echo "$(date -u +%FT%TZ) budget: $total used + $ARM_CAP cap + $RESERVE reserve > $CEILING; chain stops before $item"
    record "$item" skipped-budget "$total used"
    break
  fi
  echo "$(date -u +%FT%TZ) chain: $item (teacher ${teacher:-none}, KL $kl), $total GPU-h used"
  record "$item" started ""
  if STAGES=onestep,reload,full TEACHER_FILE=$teacher TEACHER_KL=$kl ARM_CAP=$ARM_CAP \
    bash "$CODE/v2/27b/m4b/run_ff_arm.sh" "$arm" "$seed" "$SHA"; then
    record "$item" complete ""
  else
    record "$item" failed "see $ROOT/$item/receipts"
  fi
done
echo "$(date -u +%FT%TZ) m4b chain done"
