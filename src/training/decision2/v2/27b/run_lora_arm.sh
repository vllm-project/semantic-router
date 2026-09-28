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

COMMON_MOUNTS=(
  --mount "$RUN/pipeline:/pipeline" --mount "$CODE:/code" --mount "$SOURCE_MOUNT:/source"
  --mount "$DATA/qwen38_27b_full4096_v1/train.jsonl:/data/train.jsonl"
  --mount "$DATA/rights_clean_goemotions_v2/select.jsonl:/data/select.jsonl"
  --mount "$DATA/rights_clean_goemotions_v2/cal.jsonl:/data/cal.jsonl"
  --mount "$BENCH/dev.prompts.jsonl:/data/dev.prompts.jsonl"
  --mount "$BENCH/css-transfer-v1/css-pilot.prompts.jsonl:/data/css-pilot.prompts.jsonl"
)
TRAIN_ARGS=(
  python3 -m training.model.train --model-path "$MODEL" --init-kind posttrained
  --base-revision "$REVISION" --train /data/train.jsonl --select /data/select.jsonl
  --cal /data/cal.jsonl --objective ce_brier --brier-weight 0.5 --train-mode lora
  --lora-rank 8 --lora-alpha 16 --lora-dropout 0.05 --lora-lr 2e-5 --epochs 1
  --microbatch 1 --accumulation 16 --eval-batch 1 --max-length 4096 --head-dim 256
  --backbone-lr 1e-6 --head-lr 1e-4 --weight-decay 0.01 --warmup-ratio 0.05
  --seed "$SEED" --gradient-checkpointing
)
launch() {  # name cap purpose out_dir -- argv...
  local name=$1 cap=$2 purpose=$3 out=$4
  shift 5
  mkdir -p "$out"
  python3 -m v2.27b.launch --name "d2-27b-$ARM-$name" --gpu "$GPU" --cap-hours "$cap" \
    --purpose "$ARM $purpose" --receipt "$RUN/receipts/$name.json" "${COMMON_MOUNTS[@]}" \
    --mount "$out:/out:rw" --env PYTHONPATH=/pipeline:/code -- "$@"
}

if has admit; then
  mkdir -p "$RUN/admit"
  docker run --rm --network none --entrypoint python3 -e PYTHONPATH=/code -e PYTHONDONTWRITEBYTECODE=1 \
    --mount "type=bind,src=$CODE,dst=/code,readonly" --mount "type=bind,src=$SOURCE_MOUNT,dst=/source,readonly" \
    --mount "type=bind,src=$DATA,dst=/datasrc,readonly" --mount "type=bind,src=$RUN/admit,dst=/out" \
    -w /code "$IMAGE" -m v2.27b.admit_tokens --source "$MODEL" --family qwen --limit 4096 \
    --partition train=/datasrc/qwen38_27b_full4096_v1/train.jsonl \
    --partition select=/datasrc/rights_clean_goemotions_v2/select.jsonl \
    --partition cal=/datasrc/rights_clean_goemotions_v2/cal.jsonl --output /out/admission.json
  python3 -c "import json,sys; r=json.load(open('$RUN/admit/admission.json')); sys.exit(0 if r['all_admitted'] else 3)"
fi
if has onestep; then
  launch onestep 0.5 "one-step preflight" "$RUN/onestep" -- "${TRAIN_ARGS[@]}" \
    --max-steps 1 --save-every 1 --output /out/run
fi
if has reload; then
  launch reload 0.5 "reload parity" "$RUN/onestep" -- python3 -m v2.27b.preflight_reload \
    --run-dir /out/run --step 1 --source-path "$MODEL" --select /data/select.jsonl \
    --rows 32 --max-length 4096 --output /out/reload-parity.json
fi
if has full; then
  launch full 3.0 "full 458-update arm" "$RUN/full" -- "${TRAIN_ARGS[@]}" \
    --save-every 92 --output /out/run
fi
if has readout; then
  launch readout 1.0 "CAL fit and one DEV/CSS-pilot readout" "$RUN/full" -- \
    python3 -m v2.27b.readout --run-dir /out/run --source-path "$MODEL" --cal /data/cal.jsonl \
    --panel dev=/data/dev.prompts.jsonl --panel css-pilot=/data/css-pilot.prompts.jsonl \
    --model-id "decision2-27b-$ARM" --out-dir /out
fi
echo "arm $ARM stages $STAGES complete"
