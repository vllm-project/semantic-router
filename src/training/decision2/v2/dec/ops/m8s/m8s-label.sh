#!/usr/bin/env bash
# Decoder M8-small A20r teacher targets on node B (prereg dec-m8s-prereg-2026-09-30.md, "Teacher targets"): one
# network-less container of the ~27B kernel image per job on an M8-small GPU, with a fresh cp -a copy of A20r's frozen
# autotune cache (tree-hash checked before, post-run hash after), the A20r soup checkpoint and its pinned base mounted
# read-only.
#   m8s-label.sh parity <gpu>             CAL698 re-collection vs A20r's stored CAL logits -> m8s/labels/parity.json
#   m8s-label.sh label <gpu> <k> <K>      shard k of K of the union of both tiers' top-up rows (needs parity PASS)
#                                         -> m8s/labels/shard-<k>-of-<K>.jsonl (+ .manifest.json)
set -uo pipefail
. "$(dirname "$0")/m8s-lib.sh"
MODE=$1 GPU=$2
A20R=/data/dev2/runs/27b/M4-A20r-soup
CKPT=$A20R/soup/checkpoint
PKG_CAL=$A20R/package/calibration.json
MODEL_SHA=2e07451107a2ca9735064cc01f7e43e8522b4bae48f2d77bdf29f7ea2fe6362c
BASE=/data/decision20-20260926/models/Qwen3.8-27B
FROZEN=/data/dev2/runs/27b/m3-f2/f1-scored-cache
FROZEN_SHA=03b172f1a6adeef6c6a6c491d04389b355c9d8579480008023f408c8659b502b
CAL=$SELCAL/cal.jsonl CAL_SHA=19cc1a8c4ebe6fd13031079f6b7d131046ce503f435f894e3ea672f91c2ed41f
STORED=$A20R/cal698/cal.logits.jsonl
L=$M/labels
mkdir -p "$L"
case $MODE in
  parity) NAME=parity OUT=$L/parity.json ARGS=(--cal "$CAL" --cal-sha256 "$CAL_SHA" --stored-logits "$STORED") ;;
  label)
    K=$3 N=$4 NAME=shard-$3-of-$4 OUT=$L/shard-$3-of-$4.jsonl
    grep -qs '"status": "PASS"' "$L/parity.json" || { log "label $NAME refused: no parity PASS"; exit 1; }
    ARGS=(--shard-index "$K" --shard-count "$N")
    for t in 2b 08b; do
      [ -f "$M/data/$t/topup/READY" ] && [ "$(sha "$M/data/$t/topup/train.jsonl")" = "$(awk '{print $1; exit}' "$M/data/$t/topup/READY")" ] \
        || { log "label $NAME refused: $t top-up not READY"; exit 1; }
      ARGS+=(--train "$M/data/$t/topup/train.jsonl")
    done
    ;;
  *) sed -n '2,9p' "$0"; exit 2 ;;
esac
[ -n "${RENDER[$GPU]:-}" ] || { echo "GPU$GPU is not an M8-small GPU" >&2; exit 2; }
[ -e "$OUT" ] && { log "label $NAME: $OUT exists; skipped"; exit 0; }
[ -e "$L/$NAME.launch.json" ] && { log "label $NAME failed earlier (receipt $L/$NAME.launch.json); not rerun"; exit 1; }
wait_gpu "$GPU" 160
lease "$GPU" busy "A20r targets $NAME"
cd "$S" && export PYTHONPATH=$S
python3 -m v2.27b.triton_cache copy --frozen "$FROZEN" --expect "$FROZEN_SHA" --dest "$L/$NAME.triton-cache" \
  > "$L/$NAME.cache-copy.log" 2>&1 || { log "label $NAME: frozen cache copy FAILED"; exit 1; }
argv=(docker run --name "dec-m8s-label-$NAME" --rm --network none --shm-size 16g
  --device /dev/kfd --device "${RENDER[$GPU]}" -e ROCR_VISIBLE_DEVICES=0 -e HIP_VISIBLE_DEVICES=0
  -e PYTHONPATH=/code:/opt/decision-fla -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1
  -e TRITON_CACHE_AUTOTUNING=1 -e TRITON_CACHE_DIR=/triton-cache -e HIP_FORCE_DEV_KERNARG=1
  --mount "type=bind,src=$S,dst=/code,readonly"
  --mount "type=bind,src=$A20R,dst=$A20R,readonly"
  --mount "type=bind,src=$BASE,dst=$BASE,readonly"
  --mount "type=bind,src=$SELCAL,dst=$SELCAL,readonly"
  --mount "type=bind,src=$M/data,dst=$M/data,readonly"
  --mount "type=bind,src=$L,dst=/out"
  --mount "type=bind,src=$L/$NAME.triton-cache,dst=/triton-cache"
  -w /code "$IMAGE" python3 /code/v2/dec/ops/m8s/m8s_label.py "$MODE" --checkpoint "$CKPT" --source-path "$BASE"
  --calibration "$PKG_CAL" --model-sha256 "$MODEL_SHA" --max-length 32768 "${ARGS[@]}" --output "/out/$(basename "$OUT")")
start=$(date -u +%FT%TZ)
log "label $NAME start on GPU$GPU"
"${argv[@]}" > "$L/$NAME.stdout.log" 2> "$L/$NAME.stderr.log"
rc=$?
end=$(date -u +%FT%TZ)
python3 -m v2.27b.triton_cache finish --dest "$L/$NAME.triton-cache" > "$L/$NAME.cache-finish.log" 2>&1
python3 - "$L/$NAME.launch.json" "$NAME" "$start" "$end" "$rc" "$GPU" "$SRC" "${argv[@]}" <<'EOF'
import json, sys
path, name, start, end, rc, gpu, src, *argv = sys.argv[1:]
json.dump({"job": f"dec-m8s-label-{name}", "source_mirror": src, "image_id": argv[argv.index("-w") + 2],
           "start_utc": start, "end_utc": end, "exit_status": int(rc), "gpu": f"node B GPU{gpu}",
           "docker_argv": argv}, open(path, "x"), indent=2)
EOF
gpu_seconds "$L/$NAME.launch.json" "labels" "A20r targets $NAME"
lease "$GPU" idle "A20r targets $NAME finished (exit $rc)"
log "label $NAME finished (exit $rc): $(tail -1 "$L/$NAME.stdout.log" 2>/dev/null)"
exit "$rc"
