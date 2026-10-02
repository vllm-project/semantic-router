#!/usr/bin/env bash
# Decoder M17 stage-2 data on node F (prereg dec-m17-stage2-prereg-2026-10-02.md, "Arms" / "Data"), CPU only (host
# python3, standard library), from an exact mirror. Idempotent; a failed build is not rerun.
#   1. 4b-LHS23SD: m17_data.py on stage 1's verified inputs (same flags, sentfin dropped, seed 20261002) at share .23
#      (the whole IB1-r3 + IB2 pool is .232 of T) into data/4b-s2;
#   2. 4b-LHS17UP: hard links of stage 1's locked 4b-LHS17SD TRAIN, ids and teacher (checked against READY-m17.json)
#      plus weights.jsonl (m17_weights.py: released rows 1.5, IB rows 1.0);
#   3. data/READY-m17s2.json: the SHA-256 of every stage-2 TRAIN, teacher and weights file, written once.
#
# usage: m17-prep2.sh <mirror-dir> <prereg-commit>
set -euo pipefail
SRC=$1 PREREG=$2
R=/data/dev2/runs/dec
M=$R/m17
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/dec/ops/m17
D=$M/data/4b-s2
mkdir -p "$M/data" "$M/logs"  # m17_data.py creates $D itself
log() { echo "$(date -u +%FT%TZ) prep2 $*" | tee -a "$M/OPERATIONS.log"; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
check() { [ "$(sha "$1")" = "$2" ] || { log "FAILED: $1 is not $2"; exit 1; }; }
BASE=$R/m10/data/m10-4b-base/train.jsonl BASE_SHA=c385406e8f78a2ae257cf2479b7088a523be18009e55e16caf59327c34260f09
A10=$R/m12/data/4b/4b-LHA10 A10_SHA=d41cdd1aa1e63b47394ed8fa61a87bf461cb4d2cea08cc7ff12f2e58b36dc9d5
A25=$R/m12/data/4b/4b-LHA A25_SHA=e9d30c8f89ce215ff4e1b5f57e5eafd194c1822d1108c63ba5c52e7d119fd810
TEACHER=$R/m15/inputs/teacher-4b.jsonl TEACHER_SHA=7639fab17c719bb3ed7a18bd16397130f109c8bd204d17c45f6743d6f0c17496
S17=$M/data/4b/4b-LHS17SD
S17_SHA=14bce13ce926b354e214581e7cf4718d03f80c975b36dbce6731fbc6517e25a0
S17_TEACHER_SHA=374f4fa68c8ae32f7b85fc8beeb988a1e4de43be544d4cfbf68193b2fe2fcea2

[ -f "$M/data/READY-m17s2.json" ] && { log "READY-m17s2.json exists; nothing to do"; exit 0; }
if [ ! -f "$D/report.json" ]; then
  [ -e "$D.FAILED" ] && { log "S23 build failed earlier; not rerun"; exit 1; }
  check "$BASE" $BASE_SHA
  check "$TEACHER" $TEACHER_SHA
  check "$A10/train.jsonl" $A10_SHA
  check "$A25/train.jsonl" $A25_SHA
  python3 -B "$OPS/m17_data.py" --base "$BASE" --base-sha $BASE_SHA --base-train "$A10/train.jsonl" \
    --base-train-sha $A10_SHA --base-ids "$A10/train.ids.jsonl" --pool-train "$A25/train.jsonl" --pool-sha $A25_SHA \
    --pool-ids "$A25/train.ids.jsonl" --teacher "$TEACHER" --teacher-sha $TEACHER_SHA --drop sentfin \
    --arm 4b-LHS23SD=0.23 --seed 20261002 --output "$D" > "$D.log" 2>&1 \
    || { touch "$D.FAILED"; log "FAILED: S23 build (see $D.log)"; exit 1; }
  log "S23 built: $(tail -1 "$D.log" | cut -c1-600)"
fi

check "$S17/train.jsonl" $S17_SHA
check "$S17/teacher-s.jsonl" $S17_TEACHER_SHA
U=$D/4b-LHS17UP
mkdir -p "$U"
for f in train.jsonl train.ids.jsonl teacher-s.jsonl; do [ -e "$U/$f" ] || ln "$S17/$f" "$U/$f"; done
python3 -B "$OPS/m17_weights.py" --train "$U/train.jsonl" --train-sha $S17_SHA --ids "$U/train.ids.jsonl" \
  --released-weight 1.5 --ib-weight 1.0 --name 4b-LHS17UP --output "$D" > "$D/4b-LHS17UP.weights.log" 2>&1 \
  || { log "FAILED: UP weights (see $D/4b-LHS17UP.weights.log)"; exit 1; }
log "UP weights built: $(cut -c1-600 "$D/4b-LHS17UP.weights.log")"

python3 - "$D" "$PREREG" "$M/data/READY-m17s2.json" << 'EOF'
import hashlib, json, sys
from pathlib import Path
d, prereg, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()
lock = {"schema": "dec-m17-ready/1", "record": "dec-m17-stage2-prereg-2026-10-02.md", "record_commit": prereg,
        "arms": {}, "teachers": {}, "weights": {}}
for arm in ("4b-LHS17UP", "4b-LHS23SD"):
    lock["arms"][arm] = sha(d / arm / "train.jsonl")
    lock["teachers"][arm] = sha(d / arm / "teacher-s.jsonl")
    if (d / arm / "weights.jsonl").exists():
        lock["weights"][arm] = sha(d / arm / "weights.jsonl")
with open(out, "x") as f:
    json.dump(lock, f, indent=1)
print(json.dumps(lock))
EOF
log "READY-m17s2.json written: $(tr -d '\n ' < "$M/data/READY-m17s2.json" | cut -c1-900)"
