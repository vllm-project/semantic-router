#!/usr/bin/env bash
# M4b full-parameter FSDP2 arm-seed on node B GPU0-2 (host side).
# Usage: run_ff_arm.sh ARM SEED MIRROR_SHA    (SEED is s1 or s2)
# Stages (STAGES, comma list): probe (P0 24-update probe, no checkpoint), onestep, reload, full.
# Every GPU stage runs through launch3.py and writes a receipt under $RUN/receipts; the arm stops at
# the first failed stage and a stage whose receipt already exists with exit 0 is skipped.
set -euo pipefail

ARM=$1 SEED=$2 SHA=$3
echo "$(date -u +%FT%TZ) start ARM=$ARM SEED=$SEED MIRROR=$SHA STAGES=${STAGES:-probe,onestep,reload,full} GPUS=${GPUS:-0,1,2}"
case "$SEED" in
  s1) SEED_VALUE=20260926 ;;
  s2) SEED_VALUE=20260928 ;;
  *) echo "unknown seed $SEED (s1|s2)" >&2; exit 2 ;;
esac
CODE=/data/dev2/src/$SHA/src/training/decision2
[ -d "$CODE" ] || CODE=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2
RUN=/data/dev2/runs/27b/m4b/$ARM-$SEED
STAGES=${STAGES:-probe,onestep,reload,full}
GPUS=${GPUS:-0,1,2}
TEACHER_FILE=${TEACHER_FILE:-}
TEACHER_KL=${TEACHER_KL:-0}
PRECISION=${PRECISION:-autocast}
MAX_BATCH_TOKENS=${MAX_BATCH_TOKENS:-32768}
PROBE_UPDATES=${PROBE_UPDATES:-24}
FULL_CAP=${FULL_CAP:-5.0}
ARM_CAP=${ARM_CAP:-6.0}
# Stage caps in GPU-hours (wall cap = GPU-hours / GPUs); amendment 2 may re-set them from the probe.
PROBE_CAP=${PROBE_CAP:-1.2}
ONESTEP_CAP=${ONESTEP_CAP:-0.9}
RELOAD_CAP=${RELOAD_CAP:-0.3}
TRAIN_FILE=${TRAIN_FILE:-/data/dev2/private/27b/m3-data/mixtures-m3-1/a7.train.jsonl}
SELECT_FILE=${SELECT_FILE:-/data/decision20-20260926/data/rights_clean_goemotions_v2/select.jsonl}
MODEL_DIR=${MODEL_DIR:-/data/decision20-20260926/models/Qwen3.8-27B}
REVISION=${REVISION:-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}
# Arm-local autotune cache, seeded once from a verified copy of M3-A-s1's training cache.
SEED_CACHE=${SEED_CACHE:-/data/dev2/runs/27b/M3-A-s1/triton-cache}
SEED_CACHE_SHA=${SEED_CACHE_SHA:-}
CACHE=$RUN/triton-cache
export TMPDIR=/data/dev2/tmp/m4b-$ARM-$SEED

[ -d "$CODE/v2/27b/m4b" ] || { echo "missing mirror $SHA" >&2; exit 2; }
for path in "$TRAIN_FILE" "$SELECT_FILE" "$MODEL_DIR/config.json" ${TEACHER_FILE:+"$TEACHER_FILE"}; do
  [ -e "$path" ] || { echo "missing $path" >&2; exit 2; }
done
case "$PRECISION" in autocast | mp) ;; *) echo "PRECISION must be autocast or mp" >&2; exit 2 ;; esac
IFS=, read -r -a GPU_LIST <<< "$GPUS"
NGPU=${#GPU_LIST[@]}
FIRST_GPU=${GPU_LIST[0]}
mkdir -p "$RUN/receipts" "$TMPDIR"
cd "$CODE"
export PYTHONPATH=$CODE PYTHONDONTWRITEBYTECODE=1

has() { case ",$STAGES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
log() { echo "$(date -u +%FT%TZ) $*"; }
receipt_ok() {  # NAME: its receipt exists with exit 0
  [ -f "$RUN/receipts/$1.json" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['exit_code'] == 0 else 1)" "$RUN/receipts/$1.json"
}
exit_of() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['exit_code'])" "$1"; }
arm_used() {  # GPU-hours of this arm-seed's receipts; the probe is charged to P0
  python3 -c "import glob,json,os,sys; print(sum(json.load(open(p))['gpu_hours'] for p in glob.glob(sys.argv[1] + '/*.json') if not os.path.basename(p).startswith('probe')))" "$RUN/receipts"
}
stage_cap() {  # NAME GPU_H NGPU -> wall-clock cap in hours; refuses when ARM_CAP would be exceeded
  local used
  used=$(arm_used)
  python3 -c "
import sys
name, gpu_h, n, used, cap = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
if not name.startswith('probe') and used + gpu_h > cap + 1e-9:
    raise SystemExit(f'{name}: {used:.3f} used + {gpu_h:.3f} would exceed the {cap} GPU-hour arm cap')
print(f'{gpu_h / n:.4f}')" "$1" "$2" "$3" "$used" "$ARM_CAP"
}
cache_digest() {  # STAGE PHASE: tree hash of the arm cache into triton-cache.stages.jsonl
  local digest
  digest=$(python3 -m v2.27b.triton_cache digest "$CACHE")
  printf '{"stage": "%s", "phase": "%s", "utc": "%s", "sha256": "%s"}\n' "$1" "$2" "$(date -u +%FT%TZ)" "$digest" >> "$CACHE.stages.jsonl"
}

if [ ! -f "$CACHE.copy.json" ]; then
  expect=$SEED_CACHE_SHA
  [ -n "$expect" ] || expect=$(python3 -m v2.27b.triton_cache digest "$SEED_CACHE")
  log "seeding $CACHE from $SEED_CACHE ($expect)"
  python3 -m v2.27b.triton_cache copy --frozen "$SEED_CACHE" --dest "$CACHE" --expect "$expect"
fi

python3 -m v2.27b.m4b.launch3 lease --gpus "$GPUS" --purpose "m4b $ARM-$SEED" --status reserved-idle
release() {
  python3 -m v2.27b.m4b.launch3 lease --gpus "$GPUS" --purpose "m4b $ARM-$SEED done" --status idle || true
  python3 -m v2.27b.triton_cache finish --dest "$CACHE" || true
}
trap release EXIT

COMMON=(
  --mount "$CODE:/code" --mount "$MODEL_DIR:$MODEL_DIR" --mount "$TRAIN_FILE:/data/train.jsonl"
  --mount "$SELECT_FILE:/data/select.jsonl" --mount "$CACHE:/triton-cache:rw" --mount "$TMPDIR:$TMPDIR:rw"
  --env PYTHONPATH=/code:/opt/decision-fla --env "TMPDIR=$TMPDIR"
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env "DEC_MIRROR_SHA=$SHA"
  --env TRITON_CACHE_AUTOTUNING=1 --env TRITON_CACHE_DIR=/triton-cache
)
TEACHER_MOUNT=() TEACHER_ARGS=()
if [ -n "$TEACHER_FILE" ]; then
  TEACHER_MOUNT=(--mount "$TEACHER_FILE:/data/teacher.jsonl")
  TEACHER_ARGS=(--teacher /data/teacher.jsonl --teacher-partial --teacher-kl-weight "$TEACHER_KL")
fi
TRAIN_ARGS=(
  python3 -m torch.distributed.run --standalone --nproc-per-node "$NGPU" -m v2.27b.m4b.train_ff
  --model-path "$MODEL_DIR" --revision "$REVISION" --train /data/train.jsonl --select /data/select.jsonl
  --arm "$ARM" --seed "$SEED_VALUE" --precision "$PRECISION" --max-batch-tokens "$MAX_BATCH_TOKENS"
  --max-batch-rows 64 --update-rows 64 --max-length 4096 --head-dim 256 --backbone-lr 1e-5
  --head-lr 1e-4 --weight-decay 0.01 --warmup-ratio 0.1 --brier-weight 0.5 --clip 1.0
  --gradient-checkpointing on "${TEACHER_ARGS[@]}"
)
launch() {  # NAME GPU_H GPUS PURPOSE OUT_DIR -- argv...
  local name=$1 gpu_h=$2 gpus=$3 purpose=$4 out=$5 n wall status=0
  shift 6
  IFS=, read -r -a n <<< "$gpus"
  # Callers run launch under || so errexit is off here: every step checks its own status.
  wall=$(stage_cap "$name" "$gpu_h" "${#n[@]}") || return 1
  mkdir -p "$out" || return 1
  cache_digest "$name" before || return 1
  log "stage $name on GPU $gpus, cap $gpu_h GPU-h ($wall h wall)"
  python3 -m v2.27b.m4b.launch3 --name "d2-27b-m4b-$ARM-$SEED-$name" --gpus "$gpus" --cap-hours "$wall" \
    --purpose "$ARM-$SEED $purpose" --receipt "$RUN/receipts/$name.json" "${COMMON[@]}" \
    "${TEACHER_MOUNT[@]}" --mount "$out:/out:rw" -- "$@" || status=$?
  cache_digest "$name" after
  return "$status"
}
stage() {  # NAME ...: launch unless an earlier run of NAME passed; any failure stops the arm-seed
  if receipt_ok "$1"; then
    log "stage $1 already passed; skipped"
    return
  fi
  launch "$@" || { log "arm $ARM-$SEED stopped: stage $1 failed"; exit 1; }
}

if has probe; then
  stage probe "$PROBE_CAP" "$GPUS" "P0 $PROBE_UPDATES-update probe" "$RUN/probe" -- \
    "${TRAIN_ARGS[@]}" --max-updates "$PROBE_UPDATES" --output /out/run
fi
if has onestep; then
  stage onestep "$ONESTEP_CAP" "$GPUS" "one-step preflight" "$RUN/onestep" -- \
    "${TRAIN_ARGS[@]}" --onestep --output /out/run
fi
if has reload; then
  stage reload "$RELOAD_CAP" "$FIRST_GPU" "reload through the inference path" "$RUN/onestep" -- \
    python3 -m v2.27b.m4b.reload_check --run-dir /out/run --select /data/select.jsonl --rows 32 \
    --max-length 4096 --output /out/reload.json
fi
if has full; then
  # One restart from scratch (new run directory) after exit 139, inside the arm-seed cap.
  run=run attempt=1
  while [ ! -f "$RUN/full/$run/COMPLETE.json" ]; do
    name=full
    [ "$attempt" -gt 1 ] && name=full-r$attempt
    used=$(arm_used)
    cap=$(python3 -c "import sys; print(f'{min(float(sys.argv[1]), float(sys.argv[2]) - float(sys.argv[3])):.4f}')" "$FULL_CAP" "$ARM_CAP" "$used")
    if python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) > 0 else 1)" "$cap"; then :; else
      log "arm $ARM-$SEED stopped: $used GPU-hours used of the $ARM_CAP arm cap"
      exit 1
    fi
    launch "$name" "$cap" "$GPUS" "full one-pass attempt $attempt" "$RUN/full" -- \
      "${TRAIN_ARGS[@]}" --output "/out/$run" || true
    [ -f "$RUN/full/$run/COMPLETE.json" ] && break
    code=$(exit_of "$RUN/receipts/$name.json")
    if [ "$code" != 139 ] || [ "$attempt" -ge 2 ]; then
      log "arm $ARM-$SEED stopped: exit $code on full attempt $attempt"
      exit 1
    fi
    attempt=$((attempt + 1))
    run=run-r$attempt
  done
  echo "$run" > "$RUN/full/RUN_DIR"
fi
log "arm $ARM-$SEED stages $STAGES complete"
