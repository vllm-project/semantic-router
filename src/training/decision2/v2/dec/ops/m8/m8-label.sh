#!/usr/bin/env bash
# Decoder M8 A20r teacher labels on node A (prereg dec-m8-prereg-2026-09-30.md, "Teacher targets"). One shard per
# GPU: the input is the first 80 typed-final gold-free prompts followed by the shard's slice prompts
# (m8/label/shard-<k>.prompts.jsonl, relayed from node B), collected by A20r's unchanged scored collector
# (v2.27b.typed_collect_kernel, image dbe5f32b, HIP_FORCE_DEV_KERNARG=1, a fresh copy of the formal run's persisted
# autotune cache f474e2e9, 32,768 tokens, the package's T = 1 calibration.json), then the parity smoke
# (m8_teacher.py smoke) against A20r's stored formal typed-final predictions.
#
#   m8-label.sh launch|run <mirror-dir> <shard> <gpu>
#
# GPU0 / GPU1 are shared leases (entry owner.dec-m8-label); GPU5 is the decoder's own GPU (its owner file).
# Outputs under /data/dev2/runs/dec/m8/label/run-<shard>/: predictions.jsonl (+ manifest, runtime sidecar),
# smoke.json, GPU-SECONDS.json, DONE or FAILED. A failed shard is never relabeled (prereg stop rule).
set -u
MODE=$1 SRC=$2 K=$3 GPU=$4
M=/data/dev2/runs/dec/m8
L=$M/label
S=/data/dev2/src/$SRC/src/training/decision2
IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
PKG=/data/dev2/runs/release/dev2-27b-a20r-20260930/c1pkg/package/DEV2.0-27B
PKG_MANIFEST=0d7953cc5d85a6fde6bdf71913795d1d7849e32fbed7df10103c6da6669b254a
BASE=/data/decision20-20260926/models/Qwen3.8-27B
CAL=/data/dev2/runs/27b/M4-A20r-soup/package/calibration.json
CAL_SHA=518e19cd0f89a6e8f63e666a02c9ca4fc8743f873c765183d1e745f476cd3157
FROZEN=/data/dev2/runs/27b/M4-A20r-soup/formal/triton-cache
FROZEN_SHA=f474e2e9dbdf3a2900465bce74256326ba2756250861b45359742685982cb5f7
STORED=/data/dev2/runs/27b/M4-A20r-soup/formal/output/typed-final.predictions.jsonl
STORED_SHA=ea573b2d50be50efd3048ee427309232b1f9b998b98d9997644bd82a51442d0e
SMOKE_PROMPTS=/data/dev2/private/panels/goldfree/typed-final.prompts.jsonl
SMOKE_ITEMS=80
REVISION=5323310327e52d4eadd119cd10accac9b106c97d
R=$L/run-$K
case $GPU in
  0 | 1) LEASE=/data/dev2/leases/gpu$GPU.lock/owner.dec-m8-label ;;
  5) LEASE=/data/dev2/leases/gpu5.lock/owner ;;
  *) echo "M8 labels run on node A GPU0, GPU1 or GPU5" >&2; exit 2 ;;
esac
mkdir -p "$L/logs"
if [ "$MODE" = launch ]; then
  mkdir "$L/logs/launch-$K.lock" 2>/dev/null || { echo "M8 label shard $K already launched"; exit 0; }
  setsid nohup bash "$0" run "$SRC" "$K" "$GPU" > "$L/logs/shard-$K.log" 2>&1 < /dev/null &
  echo $! > "$L/logs/shard-$K.pid"
  echo "$(date -u +%FT%TZ) M8 label shard $K launched on GPU$GPU from $SRC (pid $(cat "$L/logs/shard-$K.pid"))" | tee -a "$M/OPERATIONS.log"
  exit 0
fi
[ "$MODE" = run ] || { echo "unknown mode $MODE" >&2; exit 2; }
log() { echo "$(date -u +%FT%TZ) label-$K $*" | tee -a "$M/OPERATIONS.log"; }
fail() { log "FAILED: $*"; echo "$* ($(date -u +%FT%TZ))" > "$R/FAILED"; release; exit 1; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
lease() {  # <status> <purpose>
  printf 'track=dec\nstatus=%s\npurpose=decoder M8 %s\nstart_utc=%s\nexpected_end_utc=%s\n' "$1" "$2" \
    "$(date -u +%FT%TZ)" "$(date -u -d '+60 min' +%FT%TZ)" > "$LEASE"
}
release() {
  if [ "$GPU" = 5 ]; then
    printf 'track=dec\nstatus=idle (decoder M8 label shard %s finished)\nend_utc=%s\n' "$K" "$(date -u +%FT%TZ)" > "$LEASE"
  else
    printf 'track=dec\nstatus=released (decoder M8 label shard %s)\nlast_job_end_utc=%s\n' "$K" "$(date -u +%FT%TZ)" > "$LEASE"
  fi
}
[ ! -e "$R" ] || { echo "$R exists; a shard is never relabeled" >&2; exit 1; }
mkdir -p "$R/out"
IN=$L/shard-$K.prompts.jsonl
[ -f "$IN" ] && [ "$(sha "$IN")" = "$(awk -v f="shard-$K.prompts.jsonl" '$2 == f {print $1}' "$L/SHARDS.sha256")" ] \
  || fail "shard input $IN missing or not the locked file (SHARDS.sha256)"
[ "$(sha "$CAL")" = "$CAL_SHA" ] || fail "calibration $CAL is not $CAL_SHA"
[ "$(sha "$STORED")" = "$STORED_SHA" ] || fail "stored predictions are not $STORED_SHA"
[ "$(sha "$PKG/MODEL_MANIFEST.json")" = "$PKG_MANIFEST" ] || fail "package manifest is not $PKG_MANIFEST"
python3 - "$SMOKE_PROMPTS" "$S" <<'EOF' || fail "typed-final gold-free prompts differ from the frozen panel"
import hashlib, sys
sys.path.insert(0, sys.argv[2])
from v2.eval import panels
want = panels.expected_files()["goldfree/typed-final.prompts.jsonl"]
got = hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest()
sys.exit(0 if got == want else 1)
EOF
{ head -n "$SMOKE_ITEMS" "$SMOKE_PROMPTS"; cat "$IN"; } > "$R/input.prompts.jsonl"
(cd "$S" && python3 -m v2.27b.triton_cache copy --frozen "$FROZEN" --expect "$FROZEN_SHA" --dest "$R/triton-cache") \
  > "$R/cache-copy.log" 2>&1 || fail "frozen cache copy (see $R/cache-copy.log)"
lease busy "A20r label shard $K (node A GPU$GPU)"
log "start: $(wc -l < "$R/input.prompts.jsonl") prompts ($SMOKE_ITEMS smoke + shard $K) on GPU$GPU"
t0=$(date -u +%s) start=$(date -u +%FT%TZ)
docker run --rm --name "dec-m8-label-$K" --network none \
  --device /dev/kfd --device /dev/dri --group-add video --ipc host --security-opt seccomp=unconfined \
  -e ROCR_VISIBLE_DEVICES="$GPU" -e HIP_FORCE_DEV_KERNARG=1 -e TRITON_CACHE_AUTOTUNING=1 \
  -e TRITON_CACHE_DIR=/cache -e PYTHONPATH=/code:/opt/decision-fla -e PYTHONDONTWRITEBYTECODE=1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --mount "type=bind,src=$S,dst=/code,readonly" \
  --mount "type=bind,src=$BASE,dst=$BASE,readonly" \
  --mount "type=bind,src=$PKG,dst=$PKG,readonly" \
  --mount "type=bind,src=$CAL,dst=/cal/calibration.json,readonly" \
  --mount "type=bind,src=$R/input.prompts.jsonl,dst=/in/input.prompts.jsonl,readonly" \
  --mount "type=bind,src=$R/out,dst=/out" \
  --mount "type=bind,src=$R/triton-cache,dst=/cache" \
  -w /code --entrypoint python3 "$IMAGE" -m v2.27b.typed_collect_kernel \
  --checkpoint "$PKG" --source-path "$BASE" --model-id llm-semantic-router/DEV2.0-27B --model-revision "$REVISION" \
  --max-length 32768 --calibration /cal/calibration.json --input /in/input.prompts.jsonl \
  --output /out/predictions.jsonl > "$R/collect.stdout.log" 2> "$R/collect.stderr.log"
rc=$?
t1=$(date -u +%s)
python3 - "$R/GPU-SECONDS.json" "$K" "$GPU" "$start" "$(date -u +%FT%TZ)" "$((t1 - t0))" "$rc" <<'EOF'
import json, sys
path, k, gpu, start, end, wall, rc = sys.argv[1:]
json.dump({"schema": "dec-m8-gpu-seconds/1", "job": f"label shard {k}", "node": "A", "gpu": int(gpu),
           "shared": gpu in ("0", "1"), "start_utc": start, "end_utc": end, "gpu_seconds": int(wall),
           "exit_status": int(rc)}, open(path, "x"), indent=1)
EOF
[ $rc = 0 ] || fail "collection exit $rc (see $R/collect.stderr.log)"
(cd "$S" && python3 -m v2.27b.triton_cache finish --dest "$R/triton-cache") > "$R/cache-finish.log" 2>&1 \
  || log "cache finish report failed (report only)"
(cd "$S" && python3 -B v2/dec/ops/m8/m8_teacher.py smoke --stored "$STORED" --shard-predictions "$R/out/predictions.jsonl" \
  --items "$SMOKE_ITEMS" --output "$R/smoke.json") > "$R/smoke.log" 2>&1 || fail "parity smoke (see $R/smoke.json)"
date -u +%FT%TZ > "$R/DONE"
release
log "done: $(($(wc -l < "$R/out/predictions.jsonl") - SMOKE_ITEMS)) slice predictions; smoke $(tail -1 "$R/smoke.log")"
