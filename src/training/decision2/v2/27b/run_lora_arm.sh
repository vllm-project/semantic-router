#!/usr/bin/env bash
# Matched 27B LoRA arm on the byte-identical BEST368 pipeline (node B host side).
# Usage: run_lora_arm.sh ARM GPU SOURCE_MOUNT MODEL_SUBPATH REVISION SEED MIRROR_SHA
# SOURCE_MOUNT is bind-mounted at /source; MODEL_SUBPATH is the model directory
# inside it ("." for a plain directory, "snapshots/<rev>" for an HF cache repo).
# Stages stop at the first failure; every GPU stage writes a launch receipt.
set -euo pipefail

ARM=$1 GPU=$2 SOURCE_MOUNT=$3 MODEL_SUBPATH=$4 REVISION=$5 SEED=$6 SHA=$7
CODE=/data/dev2/src/$SHA/src/training/decision2
RUN=/data/dev2/runs/27b/$ARM
DATA=/data/decision20-20260926/data
BENCH=/data/decision20-20260926/runs
IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
MODEL=/source/$MODEL_SUBPATH
STAGES=${STAGES:-admit,onestep,reload,full,readout}
# Milestone 2 knobs (defaults reproduce Milestone 1): TRAIN_FILE host path of the
# frozen mixture, SAVE_EVERY checkpoint spacing, ARM_CAP cumulative GPU-hour cap,
# FULL_CAP per full attempt, AHO comma list NAME=HOST_ROWS read at the BEST checkpoint.
TRAIN_FILE=${TRAIN_FILE:-$DATA/qwen38_27b_full4096_v1/train.jsonl}
SAVE_EVERY=${SAVE_EVERY:-92}
ARM_CAP=${ARM_CAP:-3.5}
FULL_CAP=${FULL_CAP:-3.0}
AHO=${AHO:-}
# Teacher replay (KL arms): REPLAY_FILE host path of teacher_probs rows, with the
# trainer's REPLAY_FRACTION and REPLAY_KL weight; unset means no replay.
REPLAY_FILE=${REPLAY_FILE:-}
REPLAY_FRACTION=${REPLAY_FRACTION:-0}
REPLAY_KL=${REPLAY_KL:-0}
# Training-stage import path; amendment 3 appends the image FLA overlay (/opt/decision-fla).
TRAIN_PYTHONPATH=${TRAIN_PYTHONPATH:-/pipeline:/code}
# Amendment 4: TRITON_AUTOTUNE_CACHE=1 shares one on-disk Triton autotune cache
# across this arm's training-stage containers so every process reuses the same
# kernel configurations.
TRAIN_EXTRA=()
if [ "${TRITON_AUTOTUNE_CACHE:-0}" = 1 ]; then
  mkdir -p "/data/dev2/runs/27b/$ARM/triton-cache"
  TRAIN_EXTRA=(--mount "/data/dev2/runs/27b/$ARM/triton-cache:/triton-cache:rw"
    --env TRITON_CACHE_AUTOTUNING=1 --env TRITON_CACHE_DIR=/triton-cache)
fi

[ -d "$CODE/v2/27b" ] || { echo "missing mirror $SHA" >&2; exit 2; }
mkdir -p "$RUN/receipts"
cd "$CODE"

has() { case ",$STAGES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

if [ ! -d "$RUN/pipeline" ]; then
  mkdir "$RUN/pipeline"
  tar -x -C "$RUN/pipeline" -f v2/27b/pinned/best368-pipeline-2026-09-26.tar
fi
python3 - "$RUN/pipeline" v2/27b/pinned/best368-pipeline-2026-09-26.manifest.json <<'EOF'
import hashlib, json, pathlib, sys
root, manifest = pathlib.Path(sys.argv[1]), json.load(open(sys.argv[2]))
actual = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*.py"))}
if actual != manifest["files_sha256"]:
    raise SystemExit("vendored pipeline differs from its manifest")
print("vendored pipeline verified", len(actual), "files")
EOF

AHO_MOUNTS=() AHO_ARGS=()
IFS=, read -r -a AHO_SPECS <<< "$AHO"
for spec in "${AHO_SPECS[@]}"; do
  [ -n "$spec" ] || continue
  AHO_MOUNTS+=(--mount "${spec#*=}:/data/aho-${spec%%=*}.jsonl")
  AHO_ARGS+=(--aho "${spec%%=*}=/data/aho-${spec%%=*}.jsonl")
done
REPLAY_MOUNTS=() REPLAY_ARGS=() REPLAY_ADMIT=() REPLAY_PARTITION=()
if [ -n "$REPLAY_FILE" ]; then
  REPLAY_MOUNTS=(--mount "$REPLAY_FILE:/data/replay.jsonl")
  REPLAY_ARGS=(--replay /data/replay.jsonl --replay-fraction "$REPLAY_FRACTION"
    --replay-kl-weight "$REPLAY_KL")
  REPLAY_ADMIT=(--mount "type=bind,src=$REPLAY_FILE,dst=/replay.jsonl,readonly")
  REPLAY_PARTITION=(--partition replay=/replay.jsonl)
fi
COMMON_MOUNTS=(
  --mount "$RUN/pipeline:/pipeline" --mount "$CODE:/code" --mount "$SOURCE_MOUNT:/source"
  --mount "$TRAIN_FILE:/data/train.jsonl" "${AHO_MOUNTS[@]}" "${REPLAY_MOUNTS[@]}"
  --mount "$DATA/rights_clean_goemotions_v2/select.jsonl:/data/select.jsonl"
  --mount "$DATA/rights_clean_goemotions_v2/cal.jsonl:/data/cal.jsonl"
  --mount "$BENCH/dev.prompts.jsonl:/data/dev.prompts.jsonl"
  --mount "$BENCH/css-transfer-v1/css-pilot.prompts.jsonl:/data/css-pilot.prompts.jsonl"
)
CONTRACT_ARGS=(
  --init-kind posttrained
  --base-revision "$REVISION" --train /data/train.jsonl --select /data/select.jsonl
  --cal /data/cal.jsonl --objective ce_brier --brier-weight 0.5 --train-mode lora
  --lora-rank 8 --lora-alpha 16 --lora-dropout 0.05 --lora-lr 2e-5 --epochs 1
  --microbatch 1 --accumulation 16 --eval-batch 1 --max-length 4096 --head-dim 256
  --backbone-lr 1e-6 --head-lr 1e-4 --weight-decay 0.01 --warmup-ratio 0.05
  --seed "$SEED" --gradient-checkpointing "${REPLAY_ARGS[@]}"
)
TRAIN_ARGS=(python3 -m training.model.train --model-path "$MODEL" "${CONTRACT_ARGS[@]}")
RESUME_ARGS=(python3 -m training.model.train --source-path "$MODEL" "${CONTRACT_ARGS[@]}")
exit_of() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['exit_code'])" "$1"; }
launch() {  # name cap purpose out_dir -- argv...  (PYPATH selects the import path)
  local name=$1 cap=$2 purpose=$3 out=$4
  shift 5
  mkdir -p "$out"
  python3 -m v2.27b.launch --name "d2-27b-$ARM-$name" --gpu "$GPU" --cap-hours "$cap" \
    --purpose "$ARM $purpose" --receipt "$RUN/receipts/$name.json" "${COMMON_MOUNTS[@]}" \
    --mount "$out:/out:rw" --env "PYTHONPATH=${PYPATH:-/pipeline:/code}" \
    "${EXTRA[@]}" -- "$@"
}
EXTRA=()
train_launch() {  # launch with the training-stage import path and extra mounts/env
  local status=0
  EXTRA=("${TRAIN_EXTRA[@]}")
  PYPATH=$TRAIN_PYTHONPATH launch "$@" || status=$?
  EXTRA=()
  return "$status"
}

if has admit; then
  mkdir -p "$RUN/admit"
  docker run --rm --network none --entrypoint python3 -e PYTHONPATH=/code -e PYTHONDONTWRITEBYTECODE=1 \
    --mount "type=bind,src=$CODE,dst=/code,readonly" --mount "type=bind,src=$SOURCE_MOUNT,dst=/source,readonly" \
    --mount "type=bind,src=$DATA,dst=/datasrc,readonly" --mount "type=bind,src=$RUN/admit,dst=/out" \
    --mount "type=bind,src=$TRAIN_FILE,dst=/train.jsonl,readonly" "${REPLAY_ADMIT[@]}" \
    -w /code "$IMAGE" -m v2.27b.admit_tokens --source "$MODEL" --family qwen --limit 4096 \
    --partition train=/train.jsonl "${REPLAY_PARTITION[@]}" \
    --partition select=/datasrc/rights_clean_goemotions_v2/select.jsonl \
    --partition cal=/datasrc/rights_clean_goemotions_v2/cal.jsonl --output /out/admission.json
  python3 -c "import json,sys; r=json.load(open('$RUN/admit/admission.json')); sys.exit(0 if r['all_admitted'] else 3)"
fi
if has tech24; then
  train_launch tech24 0.4 "24-update runtime technical probe" "$RUN/tech24" -- \
    "${TRAIN_ARGS[@]}" --max-steps 24 --save-every 24 --output /out/run
fi
if has onestep; then
  train_launch onestep 0.5 "one-step preflight" "$RUN/onestep" -- "${TRAIN_ARGS[@]}" \
    --max-steps 1 --save-every 1 --output /out/run
fi
if has reload; then
  train_launch reload 0.5 "reload parity" "$RUN/onestep" -- python3 -m v2.27b.preflight_reload \
    --run-dir /out/run --step 1 --source-path "$MODEL" --select /data/select.jsonl \
    --rows 32 --max-length 4096 --output /out/reload-parity.json
fi
if has full; then
  # Amendment 2: exit 139 (native runtime fault) is an interruption, not an outcome.
  # Before the first checkpoint: one fresh restart in a new directory; after it:
  # exact resume from the latest complete checkpoint. At most two recoveries.
  run=${RUN_NAME:-run}
  attempt=${ATTEMPT_START:-1}
  while [ ! -f "$RUN/full/$run/COMPLETE.json" ]; do
    used=$(python3 -c "import glob,json; print(sum(json.load(open(p))['gpu_hours'] for p in glob.glob('$RUN/receipts/*.json')))")
    if python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) < float(sys.argv[2]) else 1)" "$used" "$ARM_CAP"; then :; else
      echo "arm $ARM stopped: cumulative GPU-hours $used reached the $ARM_CAP cap" >&2
      exit 1
    fi
    name=full
    [ "$attempt" -gt 1 ] && name=full-r$attempt
    latest=$(find "$RUN/full/$run" -maxdepth 1 -type d -name 'checkpoint-*' ! -name '*.pending' 2>/dev/null | sort | tail -1 || true)
    if [ -n "$latest" ]; then
      train_launch "$name" "$FULL_CAP" "exact resume from $(basename "$latest")" "$RUN/full" -- "${RESUME_ARGS[@]}" \
        --save-every "$SAVE_EVERY" --resume "/out/$run/$(basename "$latest")" --output "/out/$run" || true
    else
      train_launch "$name" "$FULL_CAP" "full one-epoch arm" "$RUN/full" -- "${TRAIN_ARGS[@]}" \
        --save-every "$SAVE_EVERY" --output "/out/$run" || true
    fi
    [ -f "$RUN/full/$run/COMPLETE.json" ] && break
    code=$(exit_of "$RUN/receipts/$name.json")
    if [ "$code" != 139 ] || [ "$attempt" -ge 3 ]; then
      echo "arm $ARM stopped: exit $code on attempt $attempt" >&2
      exit 1
    fi
    if [ -z "$(find "$RUN/full/$run" -maxdepth 1 -type d -name 'checkpoint-*' ! -name '*.pending' 2>/dev/null || true)" ]; then
      run=run-r$((attempt + 1))
    fi
    attempt=$((attempt + 1))
  done
  echo "$run" > "$RUN/full/RUN_DIR"
fi
if has readout; then
  run=$(cat "$RUN/full/RUN_DIR")
  launch readout 1.0 "CAL fit and one DEV/CSS-pilot readout" "$RUN/full" -- \
    python3 -m v2.27b.readout --run-dir "/out/$run" --source-path "$MODEL" --cal /data/cal.jsonl \
    --panel dev=/data/dev.prompts.jsonl --panel css-pilot=/data/css-pilot.prompts.jsonl \
    "${AHO_ARGS[@]}" --model-id "decision2-27b-$ARM" --out-dir /out
fi
echo "arm $ARM stages $STAGES complete"
