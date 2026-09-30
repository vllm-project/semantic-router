#!/usr/bin/env bash
# 27B MoE milestone: one arm-seed on an official MoE base with M4-A20r's training contract (node host side).
# Usage: moe-arm.sh NODE ARM GPU BASE SEED MIRROR [STAGES]
#   NODE a|b; GPU node A 3-5 / node B 6-7; BASE gemma-4-26B-A4B-it | gemma-4-26B-A4B | Qwen3.5-35B-A3B |
#   Qwen3.5-35B-A3B-Base (under /data/dev2/models/moe); SEED 20260926 (s1) | 20260928 (s2);
#   MIRROR the mirror directory name under /data/dev2/src; STAGES from probe,admit,onestep,reload,full
#   (default admit,onestep,reload,full).
# Env: EXPERTS (grouped_mm | eager; required for training stages), FULL_CAP (GPU-h per full attempt,
#   default 13.0), ARM_CAP (cumulative per arm-seed, default 14.0).
# The contract is M4-A20r's: a20 (SHA-256 checked), rank 32 / alpha 64 / dropout 0.05 on every non-expert
# projection, LoRA 2e-5 / head 1e-4, ce_brier 0.5, one pass, micro-batch 1 x 16, SELECT700 selection, a
# fresh training autotune cache seeded from T0. The trainer is the MoE pipeline (BEST368 + MoE backbones).
set -euo pipefail
NODE=$1 ARM=$2 GPU=$3 BASE_NAME=$4 SEED=$5 MIRROR=$6 STAGES=${7:-admit,onestep,reload,full}
CODE=/data/dev2/src/$MIRROR/src/training/decision2
RUN=/data/dev2/runs/27b-moe/$ARM
MODELS=/data/dev2/models/moe
DATA=/data/decision20-20260926/data
MIX=/data/dev2/private/27b/m4-data/mixtures-m4-1/a20.train.jsonl
MIX_SHA=4aa0dc964505682b2840fa5167a7ec14983e5a0cea427480bed1996f1befc0d4
T0=/data/dev2/runs/27b/m4-train-cache-T0
T0_SHA=1933eb36d746a3bf3b716967ce97f977620eab0f7a390f751e79bfd4f8b2e21f
IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
PIPE_TAR=v2/27b/moe/pinned/moe-pipeline-2026-10-01.tar
PIPE_MANIFEST=v2/27b/moe/pinned/moe-pipeline-2026-10-01.manifest.json
SAVE_EVERY=446
FULL_CAP=${FULL_CAP:-13.0}
ARM_CAP=${ARM_CAP:-14.0}
EXPERTS=${EXPERTS:-}
case "$NODE:$GPU" in a:3 | a:4 | a:5 | b:6 | b:7) ;; *) echo "node $NODE GPU$GPU is outside the MoE allocation" >&2; exit 2 ;; esac
case "$BASE_NAME" in
  gemma-4-26B-A4B-it) REV=4d7ae4984b7db7de8f8457170b3f1a419ee76d52 KIND=posttrained FAMILY=gemma LIMIT=4736 ;;
  gemma-4-26B-A4B) REV=24548b62aa021d562695c04aaf7758a1ea47990b KIND=base FAMILY=gemma LIMIT=4736 ;;
  Qwen3.5-35B-A3B) REV=59d61f3ce65a6d9863b86d2e96597125219dc754 KIND=posttrained FAMILY=qwen LIMIT=4096 ;;
  Qwen3.5-35B-A3B-Base) REV=0f0813072d2358973511097385626f21fcb6d422 KIND=base FAMILY=qwen LIMIT=4096 ;;
  *) echo "unknown base $BASE_NAME" >&2; exit 2 ;;
esac
case "$ARM:$SEED" in *-s1:20260926 | *-s2:20260928 | *-probe:20260926) ;; *) echo "$ARM does not take seed $SEED" >&2; exit 2 ;; esac
MODEL_DIR=$MODELS/$BASE_NAME
[ -d "$CODE/v2/27b/moe" ] && [ -d "$MODEL_DIR" ] && [ -d "$T0" ] || { echo "missing mirror, base or T0" >&2; exit 2; }
[ "$(sha256sum < "$MIX" | cut -c1-64)" = "$MIX_SHA" ] || { echo "a20 mixture changed" >&2; exit 2; }
case ",$STAGES," in *,onestep,* | *,full,* | *,reload,*)
  case "$EXPERTS" in grouped_mm | eager) ;; *) echo "EXPERTS must be grouped_mm or eager" >&2; exit 2 ;; esac ;;
esac
export DEV2_NODE=$NODE TMPDIR=/data/dev2/tmp
mkdir -p "$RUN/receipts"
cd "$CODE"
has() { case ",$STAGES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

if [ ! -d "$RUN/pipeline" ]; then
  mkdir "$RUN/pipeline"
  tar -x -C "$RUN/pipeline" -f "$PIPE_TAR"
fi
python3 - "$RUN/pipeline" "$PIPE_MANIFEST" <<'EOF'
import hashlib, json, pathlib, sys
root, manifest = pathlib.Path(sys.argv[1]), json.load(open(sys.argv[2]))
actual = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*.py"))}
if actual != manifest["files_sha256"]:
    raise SystemExit("MoE pipeline differs from its manifest")
print("MoE pipeline verified", len(actual), "files")
EOF

if [ ! -e "$RUN/triton-cache" ]; then
  PYTHONPATH=$CODE python3 -m v2.27b.triton_cache copy --frozen "$T0" --expect "$T0_SHA" --dest "$RUN/triton-cache"
fi
COMMON_MOUNTS=(
  --mount "$RUN/pipeline:/pipeline" --mount "$CODE:/code" --mount "$MODEL_DIR:/source"
  --mount "$MIX:/data/train.jsonl"
  --mount "$DATA/rights_clean_goemotions_v2/select.jsonl:/data/select.jsonl"
  --mount "$DATA/rights_clean_goemotions_v2/cal.jsonl:/data/cal.jsonl"
  --mount "$RUN/triton-cache:/triton-cache:rw"
)
ENVS=(--env PYTHONPATH=/pipeline:/code:/opt/decision-fla --env TRITON_CACHE_AUTOTUNING=1 --env TRITON_CACHE_DIR=/triton-cache)
CONTRACT_ARGS=(
  --init-kind "$KIND" --base-revision "$REV" --train /data/train.jsonl --select /data/select.jsonl
  --cal /data/cal.jsonl --objective ce_brier --brier-weight 0.5 --train-mode lora
  --lora-rank 32 --lora-alpha 64 --lora-dropout 0.05 --lora-lr 2e-5 --epochs 1
  --microbatch 1 --accumulation 16 --eval-batch 1 --max-length "$LIMIT" --head-dim 256
  --backbone-lr 1e-6 --head-lr 1e-4 --weight-decay 0.01 --warmup-ratio 0.05
  --seed "$SEED" --gradient-checkpointing --experts-implementation "${EXPERTS:-eager}"
)
TRAIN_ARGS=(python3 -m training.model.train --model-path /source "${CONTRACT_ARGS[@]}")
RESUME_ARGS=(python3 -m training.model.train --source-path /source "${CONTRACT_ARGS[@]}")
launch() {  # name cap purpose out_dir -- argv...
  local name=$1 cap=$2 purpose=$3 out=$4
  shift 5
  mkdir -p "$out"
  python3 -m v2.27b.moe.launch --name "d2-27b-moe-$ARM-$name" --gpu "$GPU" --cap-hours "$cap" \
    --purpose "$ARM $purpose" --receipt "$RUN/receipts/$name.json" "${COMMON_MOUNTS[@]}" \
    --mount "$out:/out:rw" "${ENVS[@]}" -- "$@"
}
used() { python3 -c "import glob,json; print(sum(json.load(open(p))['gpu_hours'] for p in glob.glob('$RUN/receipts/*.json')))"; }

if has probe; then
  launch probe 0.5 "experts-kernel probe (eager vs grouped_mm)" "$RUN/probe" -- python3 -m v2.27b.moe.experts_probe \
    --model-path /source --revision "$REV" --source-stage "$KIND" --select /data/select.jsonl \
    --train /data/train.jsonl --max-length "$LIMIT" --output /out/experts-probe.json
fi
if has admit; then
  mkdir -p "$RUN/admit"
  docker run --rm --network none --entrypoint python3 -e PYTHONPATH=/code -e PYTHONDONTWRITEBYTECODE=1 \
    --mount "type=bind,src=$CODE,dst=/code,readonly" --mount "type=bind,src=$MODEL_DIR,dst=/source,readonly" \
    --mount "type=bind,src=$DATA,dst=/datasrc,readonly" --mount "type=bind,src=$RUN/admit,dst=/out" \
    --mount "type=bind,src=$MIX,dst=/train.jsonl,readonly" \
    -w /code "$IMAGE" -m v2.27b.admit_tokens --source /source --family "$FAMILY" --limit "$LIMIT" \
    --partition train=/train.jsonl \
    --partition select=/datasrc/rights_clean_goemotions_v2/select.jsonl \
    --partition cal=/datasrc/rights_clean_goemotions_v2/cal.jsonl --output /out/admission.json
  python3 -c "import json,sys; r=json.load(open('$RUN/admit/admission.json')); sys.exit(0 if r['all_admitted'] else 3)"
fi
if has onestep; then
  launch onestep 0.6 "one-step preflight" "$RUN/onestep" -- "${TRAIN_ARGS[@]}" --max-steps 1 --save-every 1 --output /out/run
fi
if has reload; then
  launch reload 0.5 "reload parity" "$RUN/onestep" -- python3 -m v2.27b.moe.preflight_reload \
    --run-dir /out/run --step 1 --source-path /source --select /data/select.jsonl \
    --rows 32 --max-length "$LIMIT" --output /out/reload-parity.json
fi
if has full; then
  # One attempt per arm-seed; exit 139 (native fault) allows at most two exact recoveries (M3 rule).
  run=run attempt=1
  while [ ! -f "$RUN/full/$run/COMPLETE.json" ]; do
    spent=$(used)
    python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) < float(sys.argv[2]) else 1)" "$spent" "$ARM_CAP" \
      || { echo "arm $ARM stopped: cumulative GPU-hours $spent reached the $ARM_CAP cap" >&2; exit 1; }
    name=full
    [ "$attempt" -gt 1 ] && name=full-r$attempt
    cap=$(python3 -c "import sys; print(round(min(float(sys.argv[1]), float(sys.argv[2]) - float(sys.argv[3])), 4))" \
      "$FULL_CAP" "$ARM_CAP" "$spent")
    latest=$(find "$RUN/full/$run" -maxdepth 1 -type d -name 'checkpoint-*' ! -name '*.pending' 2>/dev/null | sort | tail -1 || true)
    if [ -n "$latest" ]; then
      launch "$name" "$cap" "exact resume from $(basename "$latest")" "$RUN/full" -- "${RESUME_ARGS[@]}" \
        --save-every "$SAVE_EVERY" --resume "/out/$run/$(basename "$latest")" --output "/out/$run" || true
    else
      launch "$name" "$cap" "full one-epoch arm" "$RUN/full" -- "${TRAIN_ARGS[@]}" \
        --save-every "$SAVE_EVERY" --output "/out/$run" || true
    fi
    [ -f "$RUN/full/$run/COMPLETE.json" ] && break
    [ -f "$RUN/STOP" ] && { echo "arm $ARM stopped by its preregistered rule ($(cat "$RUN/STOP"))" >&2; exit 0; }
    code=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['exit_code'])" "$RUN/receipts/$name.json")
    if [ "$code" != 139 ] || [ "$attempt" -ge 3 ]; then
      echo "arm $ARM stopped: exit $code on attempt $attempt" >&2
      exit 1
    fi
    attempt=$((attempt + 1))
  done
  echo "$run" > "$RUN/full/RUN_DIR"
  PYTHONPATH=$CODE python3 -m v2.27b.triton_cache finish --dest "$RUN/triton-cache" > /dev/null
fi
echo "arm $ARM stages $STAGES complete"
