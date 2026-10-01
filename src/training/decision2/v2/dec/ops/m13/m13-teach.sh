#!/usr/bin/env bash
# shellcheck disable=SC2034  # the per-tier values (NAME_<tier>) are read by name through tv()
# Decoder M13 self-distillation targets (prereg dec-m13-prereg-2026-10-01.md, "Self-distillation"), on M13 GPUs of
# node E / F before any M13 training chain starts. The teacher is the tier's released model, read exactly as its
# readouts load it (same --source-path) and served uncalibrated (the release packages carry calibration null), so
# T = 1 for every type: v2.dec.teacher_label --teacher-kind dec --uncalibrated over every row of the tier's released
# TRAIN file (the typed rows; no IB row is labeled).
#   08b: DEV2.0-0.8B (node E)   2b: DEV2.0-2B (node F)   4b: the released LH = M10's LH soup (node F)
# Pre-warm: one small shard (rows at positions = 0 mod 1000) runs alone on the tier's read cache, then one shard per
# listed GPU in parallel. The shards are joined in shard order into m13/data/<tier>/teacher-sd.jsonl, which must hold
# every released TRAIN id exactly once. A failed job stops the tier (no rerun).
#
# usage: M13_NODE=e|f m13-teach.sh <mirror-dir> <tier> <gpu> [<gpu> ...]
set -euo pipefail
SRC=$1 TIER=$2
shift 2
GPUS=("$@")
NODE=${M13_NODE:?set M13_NODE=e or f}
M=/data/dev2/runs/dec/m13
OPS=/data/dev2/src/$SRC/src/training/decision2/v2/dec/ops/m13
T=$M/teach/$TIER
OUT=$M/data/$TIER/teacher-sd.jsonl
log() { echo "$(date -u +%FT%TZ) teach-$TIER $*" | tee -a "$M/OPERATIONS.log"; }
tv() { local n=$1_$2; echo "${!n}"; }
TEACHER_08b=/models/DEV2.0-0.8B/bede7938a8c209c09f27400b79eed57948d6b75e
SOURCE_08b=/models/Decision-1.0-Eos-0.8B/363c4a5e56afc115b1c78c837633956d0bbb63ab
TRAIN_08b=/runs/m11/inputs/m6-e8f-r2clean/train.jsonl
TEACHER_2b=/models/DEV2.0-2B/a53cf66a0d9d492a84b6617b61e7ce35fcd03af0
SOURCE_2b=/models/Decision-1.0-Sol-2B/ce0c018a28de16d6639b1cd203b761bf643b89e6
TRAIN_2b=/runs/m11/inputs/m4-v2m-ret-r2/train.jsonl
TEACHER_4b=/runs/m10/soup/LH/build/LH-soup
SOURCE_4b=/models/Qwen--Qwen3.5-4B-Base/1001bb4d826a52d1f399e183466143f4da7b741b
TRAIN_4b=/runs/m10/data/m10-4b-base/train.jsonl
case $NODE:$TIER in e:08b | f:2b | f:4b) ;; *) echo "no M13 teacher for $TIER on node $NODE" >&2; exit 2 ;; esac
[ -f "$OUT" ] && { log "teacher targets exist; nothing to do"; exit 0; }
[ ! -d "$T" ] || { log "an earlier labeling of $TIER exists without its output; not rerun"; exit 1; }
mkdir -p "$T"
host() { echo "/data/dev2/runs/dec/${1#/runs/}"; }
lease() {  # <gpu> <status> <purpose>
  mkdir -p "/data/dev2/leases/gpu$1.lock"
  printf 'track=dec-m13\nstatus=%s\npurpose=decoder M13 %s\nstart_utc=%s\nexpected_end_utc=%s\n' \
    "$2" "$3" "$(date -u +%FT%TZ)" "$(date -u -d '+60 min' +%FT%TZ)" > "/data/dev2/leases/gpu$1.lock/owner"
}
label() {  # <job> <gpu> <shard index> <shard count>
  M13_NODE=$NODE M13_CACHE=$TIER-read bash "$OPS/m13-launch.sh" "$1" "$SRC" "$T/$1" --gpu "$2" -- \
    -m v2.dec.teacher_label --teacher-kind dec --uncalibrated --teacher-path "$(tv TEACHER "$TIER")" \
    --teacher-source-path "$(tv SOURCE "$TIER")" --teacher-repo "dec-m13-sd-$TIER" \
    --teacher-revision "$(basename "$(tv TEACHER "$TIER")")" --train "$(tv TRAIN "$TIER")" \
    --output "/out/labels.jsonl" --shard-index "$3" --shard-count "$4"
}
for g in "${GPUS[@]}"; do lease "$g" busy "self-distillation targets $TIER"; done
log "pre-warm shard on GPU${GPUS[0]}"
label "teach-$TIER-warm" "${GPUS[0]}" 0 1000 || { log "pre-warm FAILED"; exit 1; }
n=${#GPUS[@]} pids=()
for i in "${!GPUS[@]}"; do
  label "teach-$TIER-s$i" "${GPUS[$i]}" "$i" "$n" &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
for g in "${GPUS[@]}"; do lease "$g" idle "self-distillation targets $TIER finished"; done
[ $fail = 0 ] || { log "a shard FAILED"; exit 1; }
python3 - "$(host "$(tv TRAIN "$TIER")")" "$OUT" "$n" "$T" "$TIER" << 'EOF'
import json, os, sys
train, out, n, root, tier = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
want = [json.loads(line)["id"] for line in open(train)]
seen = set()
with open(out + ".pending", "x") as sink:
    for i in range(n):
        for line in open(f"{root}/teach-{tier}-s{i}/labels.jsonl"):
            rid = json.loads(line)["id"]
            if rid in seen:
                raise SystemExit(f"repeated id {rid}")
            seen.add(rid)
            sink.write(line)
if seen != set(want) or len(want) != len(set(want)):
    raise SystemExit(f"coverage: {len(seen)} labeled vs {len(set(want))} released ids")
os.replace(out + ".pending", out)
print(len(seen))
EOF
log "teacher targets $TIER: $(wc -l < "$OUT") rows, sha256 $(sha256sum "$OUT" | cut -d' ' -f1)"
for i in $(seq 0 $((n - 1))); do
  log "shard $i agreement: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print({k: round(v["accuracy"], 4) for k, v in d["train_label_agreement"].items()}, round(d["seconds"]))' "$T/teach-$TIER-s$i/labels.jsonl.manifest.json")"
done
