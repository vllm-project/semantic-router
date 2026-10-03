#!/usr/bin/env bash
# Arm factory 9B batch-2 data on node B (amendment 4), CPU jobs in the 9B image through af-launch.sh --cpu:
#   kib4r   KIB4's construction (M10 prep.sh kib4: m9_data.py --match-tokens with IB1-r3 minus `sentfin`, IB2 and IB4
#           phase 1 at K-a13's 60,183,732 native tokens), with a new x60 cut seed `20261003:af-r1:keep` instead of
#           K-a13IB's `20261001:m9-s3:keep`: the same blocks, a different kept subset of the released x60 rows.
# Inputs are M10's node B copies, hash-checked exactly as prep.sh checks them. Output data/kib4r-build/kib4r.
#
#   kib4r2  the same with x60 cut seed `20261003:af-r2:keep` (amendment 5).
# usage: AF_NODE=b AF_SIZE=9b af-prep9b.sh <mirror-dir> kib4r|kib4r2
set -euo pipefail
SRC=$1 MODE=$2
I=/data/dev2/runs/9b/m10/inputs
L=/data/dev2/src/$SRC/src/training/decision2/v2/af/ops/af-launch.sh
M=/data/dev2/runs/af/9b
log() { echo "$(date -u +%FT%TZ) prep9b $*" | tee -a "$M/OPERATIONS.log"; }
check() { [ "$(sha256sum < "$1" | cut -d' ' -f1)" = "$2" ] || { log "hash differs: $1"; exit 1; }; }
case $MODE in
  kib4r) KEEP=20261003:af-r1:keep ;;
  kib4r2) KEEP=20261003:af-r2:keep ;;  # amendment 5
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
mkdir -p "$M/data"
[ ! -e "$M/data/$MODE-build" ] || { log "data/$MODE-build exists"; exit 1; }
check "$I/m9/data/x60/train.jsonl" a66131b1165513128e5a51af78587c4ddefcc836773e724c4514a592e33fc0e0
check "$I/m9/data/x60/teacher.jsonl" cdcd99c1550d35531df0982ae85116fe3ea5035bc9a09c1ef7b5b3063a32f2c9
check "$I/m9/inputs/ib/ib1-31b200a3/m6/ib1/ib1.train.jsonl" 1e1b08f3d37f9051ffe2e4b99fd7be673f315d05cc74a27a8d350eae2bb706c5
check "$I/m9/inputs/ib/ib2-c5dbdd0a/m6/ib2/ib2.train.jsonl" ee137efa8bbf86e5c62574f3b8fbd6204063e095a8da514f32600ba3d51d1cfa
check "$I/m9/inputs/x60-ids/mx-xl-full-r2.ids.jsonl" 7843afb7b2bbb315902b6748d6559532f384283d8efd836b935daac5f55dbd4a
check "$I/ib4-p1-76cea510/m6/ib4/p1/ib4.train.jsonl" 6045b456d4b3032db4b76d70d792e3e9e30aad13e16a8f86fb3e99bc325806fb
check "$I/ib4-p1-76cea510/m6/ib4/p1/ib4.train.tokens.jsonl" 2af4f8a5104e24d0924474104f0c0ded775ffb2090954ab782cf0ad92642b7e6
P=/r9b/m10/inputs
AF_NODE=b AF_SIZE=9b bash "$L" "prep-$MODE" "$SRC" "$M/data/$MODE-build" --cpu -- v2/9b/lux9b/m9_data.py \
  --x60-dir $P/m9/data/x60 \
  --ib1 $P/m9/inputs/ib/ib1-31b200a3/m6/ib1/ib1.train.jsonl \
  --ib2 $P/m9/inputs/ib/ib2-c5dbdd0a/m6/ib2/ib2.train.jsonl \
  --ib4 $P/ib4-p1-76cea510/m6/ib4/p1/ib4.train.jsonl --exclude-family sentfin \
  --match-tokens 60183732 --x60-ids $P/m9/inputs/x60-ids/mx-xl-full-r2.ids.jsonl \
  --keep-seed "$KEEP" --output "/out/$MODE"
log "${MODE^^} TRAIN built: train $(sha256sum < "$M/data/$MODE-build/$MODE/train.jsonl" | cut -c1-16), teacher $(sha256sum < "$M/data/$MODE-build/$MODE/teacher.jsonl" | cut -c1-16)"
