#!/usr/bin/env bash
# M4b P0 engineering preflight on node B GPU0-2 (host side): the tiny random Qwen3.5 3-rank FSDP2 parity check
# (fsdp_parity, rank 0 also runs the unsharded reference), then the 27B 24-update probe on A1-s1's settings
# (run_ff_arm.sh STAGES=probe). Receipts under /data/dev2/runs/27b/m4b/P0/receipts and the A1-s1 run dir.
# Usage: run_p0.sh MIRROR_SHA [STAGES]   (STAGES: parity,probe; default both)
set -euo pipefail
echo "$(date -u +%FT%TZ) m4b P0 $* start"
SHA=$1 STAGES=${2:-parity,probe}
CODE=/data/dev2/src/$SHA/src/training/decision2
[ -d "$CODE" ] || CODE=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2
[ -f "$CODE/v2/27b/m4b/fsdp_parity.py" ] || { echo "no m4b code in mirror $SHA" >&2; exit 2; }
GPUS=${GPUS:-0,1,2}
P0=/data/dev2/runs/27b/m4b/P0
export TMPDIR=/data/dev2/tmp/m4b-P0
mkdir -p "$P0/receipts" "$P0/parity" "$P0/triton-cache" "$TMPDIR"
cd "$CODE"
export PYTHONPATH=$CODE PYTHONDONTWRITEBYTECODE=1

has() { case ",$STAGES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

if has parity && [ ! -f "$P0/parity/parity.json" ]; then
  python3 -m v2.27b.m4b.launch3 lease --gpus "$GPUS" --purpose "m4b P0 parity" --status reserved-idle
  IFS=, read -r -a n <<< "$GPUS"
  python3 -m v2.27b.m4b.launch3 --name d2-27b-m4b-p0-parity --gpus "$GPUS" --cap-hours 0.25 \
    --purpose "P0 tiny-model FSDP2 parity" --receipt "$P0/receipts/parity.json" \
    --mount "$CODE:/code" --mount "$P0/parity:/out:rw" --mount "$TMPDIR:$TMPDIR:rw" \
    --mount "$P0/triton-cache:/triton-cache:rw" --env TRITON_CACHE_AUTOTUNING=1 \
    --env TRITON_CACHE_DIR=/triton-cache \
    --env PYTHONPATH=/code:/opt/decision-fla --env "TMPDIR=$TMPDIR" --env HF_HUB_OFFLINE=1 \
    -- python3 -m torch.distributed.run --nnodes 1 --master-addr 127.0.0.1 --master-port 29500 \
    --nproc-per-node "${#n[@]}" \
    -m v2.27b.m4b.fsdp_parity --output /out
  python3 -m v2.27b.m4b.launch3 lease --gpus "$GPUS" --purpose "m4b P0 parity done" --status idle
fi
if has parity; then
  python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print(json.dumps({k: r[k] for k in ('status', 'world_size', 'gates') if k in r})); sys.exit(0 if r.get('status') == 'PASS' else 1)" "$P0/parity/parity.json"
fi
if has probe; then
  STAGES=probe GPUS=$GPUS bash "$CODE/v2/27b/m4b/run_ff_arm.sh" A1 s1 "$SHA"
fi
echo "$(date -u +%FT%TZ) m4b P0 $STAGES complete"
