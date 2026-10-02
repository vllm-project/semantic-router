#!/usr/bin/env bash
# 9B M10 data on node B (CPU jobs in the 9B image through m10/launch.sh --cpu), and the data lock data/READY-m10.json.
#   prep.sh kup   KUP weights (x60 rows 1.5, IB rows 1) over K-a13IB's TRAIN (node C's m9/data/kib, copied, hash-checked)
#   prep.sh kibm  KIBM TRAIN: m9_data.py --match-tokens with IB1-r3 minus `sentfin`, IB2, and IB3-r2 (`mqa`), the
#                 x60 cut seed of K-a13IB, K-a13's 60,183,732 native tokens
#   prep.sh kib4  KIB4 TRAIN (amendment 3): the same with IB4 phase 1 (`76cea510`) in place of IB3-r2
# Each mode adds its arm to the lock (files relative to /data/dev2/runs/9b/m10 with SHA-256, and the trainer's data
# flags as container paths); an arm already in the lock is never changed.
# usage: M10_NODE=b prep.sh kup|kibm|kib4 <mirror-dir>
set -euo pipefail
MODE=$1 SRC=$2
M=/data/dev2/runs/9b/m10
L=/data/dev2/src/$SRC/src/training/decision2/v2/9b/lux9b/m10/launch.sh
I=$M/inputs/m9
LOCK=$M/data/READY-m10.json
IB1=$I/inputs/ib/ib1-31b200a3/m6/ib1/ib1.train.jsonl
IB2=$I/inputs/ib/ib2-c5dbdd0a/m6/ib2/ib2.train.jsonl
IB3=$M/inputs/ib3-1c8452da/m6/ib3/ib3.train.jsonl
IB4=$M/inputs/ib4-p1-76cea510/m6/ib4/p1/ib4.train.jsonl
IDS=$I/inputs/x60-ids/mx-xl-full-r2.ids.jsonl
mkdir -p "$M/data"
log() { echo "$(date -u +%FT%TZ) prep $*" | tee -a "$M/OPERATIONS.log"; }
check() {  # <file> <sha256>
  [ "$(sha256sum < "$1" | cut -d' ' -f1)" = "$2" ] || { log "hash differs: $1"; exit 1; }
}
add_lock() {  # <ARM> <json args list> <rel file>...
  python3 - "$LOCK" "$M" "$@" << 'EOF'
import hashlib, json, os, sys
lock_path, m, arm, args, *files = sys.argv[1:]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()
lock = json.load(open(lock_path)) if os.path.exists(lock_path) else {"schema": "lux9b-m10-ready/1", "arms": {}}
if arm in lock["arms"]:
    sys.exit(f"{arm} is already in the lock")
lock["arms"][arm] = {"args": json.loads(args), "files": {rel: sha(f"{m}/{rel}") for rel in files}}
tmp = lock_path + ".tmp"
with open(tmp, "w") as f:
    json.dump(lock, f, indent=2)
os.replace(tmp, lock_path)
print(json.dumps(lock["arms"][arm]))
EOF
}
check "$I/data/x60/train.jsonl" a66131b1165513128e5a51af78587c4ddefcc836773e724c4514a592e33fc0e0
check "$I/data/x60/teacher.jsonl" cdcd99c1550d35531df0982ae85116fe3ea5035bc9a09c1ef7b5b3063a32f2c9
case $MODE in
  kup)
    check "$I/data/kib/train.jsonl" 2cd09292580450a74adcc9f1df006f49e171fbf1e1fa96467a205576ee8493dc
    check "$I/data/kib/teacher.jsonl" 7e3f8bf7e46793cb35bcd3ca0064ff25f8463b2f75b4bad439a67f468029d65d
    M10_NODE=b bash "$L" prep-kup "$SRC" "$M/data/KUP" --cpu -- v2/9b/lux9b/m10/m10_weights.py \
      --train-dir /runs/m10/inputs/m9/data/kib --x60-train /runs/m10/inputs/m9/data/x60/train.jsonl \
      --x60-weight 1.5 --output /out/weights.jsonl
    add_lock KUP '["--train", "/runs/m10/inputs/m9/data/kib/train.jsonl", "--teacher", "/runs/m10/inputs/m9/data/kib/teacher.jsonl", "--example-weights", "/runs/m10/data/KUP/weights.jsonl"]' \
      inputs/m9/data/kib/train.jsonl inputs/m9/data/kib/teacher.jsonl data/KUP/weights.jsonl
    log "KUP weights built and locked"
    ;;
  kibm)
    check "$IB1" 1e1b08f3d37f9051ffe2e4b99fd7be673f315d05cc74a27a8d350eae2bb706c5
    check "$IB2" ee137efa8bbf86e5c62574f3b8fbd6204063e095a8da514f32600ba3d51d1cfa
    check "$IDS" 7843afb7b2bbb315902b6748d6559532f384283d8efd836b935daac5f55dbd4a
    check "$IB3" 9d92d92a207109a585ceead7fa5e1bdb0b2c61859de8227a2a6058428f852dea
    M10_NODE=b bash "$L" prep-kibm "$SRC" "$M/data/kibm-build" --cpu -- v2/9b/lux9b/m9_data.py \
      --x60-dir /runs/m10/inputs/m9/data/x60 \
      --ib1 /runs/m10/inputs/m9/inputs/ib/ib1-31b200a3/m6/ib1/ib1.train.jsonl \
      --ib2 /runs/m10/inputs/m9/inputs/ib/ib2-c5dbdd0a/m6/ib2/ib2.train.jsonl \
      --ib3 /runs/m10/inputs/ib3-1c8452da/m6/ib3/ib3.train.jsonl --exclude-family sentfin \
      --match-tokens 60183732 --x60-ids /runs/m10/inputs/m9/inputs/x60-ids/mx-xl-full-r2.ids.jsonl \
      --keep-seed 20261001:m9-s3:keep --output /out/kibm
    add_lock KIBM '["--train", "/runs/m10/data/kibm-build/kibm/train.jsonl", "--teacher", "/runs/m10/data/kibm-build/kibm/teacher.jsonl"]' \
      data/kibm-build/kibm/train.jsonl data/kibm-build/kibm/teacher.jsonl
    log "KIBM TRAIN built and locked"
    ;;
  kib4)
    check "$IB1" 1e1b08f3d37f9051ffe2e4b99fd7be673f315d05cc74a27a8d350eae2bb706c5
    check "$IB2" ee137efa8bbf86e5c62574f3b8fbd6204063e095a8da514f32600ba3d51d1cfa
    check "$IDS" 7843afb7b2bbb315902b6748d6559532f384283d8efd836b935daac5f55dbd4a
    check "$IB4" 6045b456d4b3032db4b76d70d792e3e9e30aad13e16a8f86fb3e99bc325806fb
    check "${IB4%.jsonl}.tokens.jsonl" 2af4f8a5104e24d0924474104f0c0ded775ffb2090954ab782cf0ad92642b7e6
    M10_NODE=b bash "$L" prep-kib4 "$SRC" "$M/data/kib4-build" --cpu -- v2/9b/lux9b/m9_data.py \
      --x60-dir /runs/m10/inputs/m9/data/x60 \
      --ib1 /runs/m10/inputs/m9/inputs/ib/ib1-31b200a3/m6/ib1/ib1.train.jsonl \
      --ib2 /runs/m10/inputs/m9/inputs/ib/ib2-c5dbdd0a/m6/ib2/ib2.train.jsonl \
      --ib4 /runs/m10/inputs/ib4-p1-76cea510/m6/ib4/p1/ib4.train.jsonl --exclude-family sentfin \
      --match-tokens 60183732 --x60-ids /runs/m10/inputs/m9/inputs/x60-ids/mx-xl-full-r2.ids.jsonl \
      --keep-seed 20261001:m9-s3:keep --output /out/kib4
    add_lock KIB4 '["--train", "/runs/m10/data/kib4-build/kib4/train.jsonl", "--teacher", "/runs/m10/data/kib4-build/kib4/teacher.jsonl"]' \
      data/kib4-build/kib4/train.jsonl data/kib4-build/kib4/teacher.jsonl
    log "KIB4 TRAIN built and locked"
    ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
