# shellcheck shell=bash disable=SC2034
# Shared host-side helpers of the ~27B kernel-path scripts (Milestone 3; sourced, not run).
# The caller sets S (mirror decision2 directory), SRC (mirror name) and GPU, and runs from $S
# with PYTHONPATH=$S.
BASE=${BASE:-/data/decision20-20260926/models/Qwen3.8-27B}
IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
KERNEL_SPEC=$S/v2/27b/adapters/typed-lora-kernel.json
CAL_FILE=${CAL_FILE:-/data/dev2/hf-cache/datasets--llm-semantic-router--decision-2.0-training-data/snapshots/ed87a03ab80ca5b9560780bba51a83a77ff47d14/m2/cal/CAL698/cal.jsonl}
CAL_SHA256=${CAL_SHA256:-19cc1a8c4ebe6fd13031079f6b7d131046ce503f435f894e3ea672f91c2ed41f}
PANEL_ROOT=/data/dev2/private/panels
LEASE=/data/dev2/leases/gpu$GPU.lock/owner
[ -f "$KERNEL_SPEC" ] || { echo "missing kernel adapter in mirror $SRC" >&2; exit 2; }

model_sha() {  # CALIBRATION -> its bound model_sha256
  python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['model_sha256'])" "$1"
}
cache_copy() {  # FROZEN EXPECTED_SHA DEST: fresh cp -a copy; fails unless its tree hash is EXPECTED_SHA
  python3 -m v2.27b.triton_cache copy --frozen "$1" --expect "$2" --dest "$3"
}
cache_finish() {  # DEST: post-run tree hash and added/changed/removed files into DEST.post.json
  python3 -m v2.27b.triton_cache finish --dest "$1"
}
kernel_env() {  # CACHE_DIR -> KENV (runner / launcher environment of the kernel path)
  KENV=(--env TRITON_CACHE_AUTOTUNING=1 --env "TRITON_CACHE_DIR=$1" --env HIP_FORCE_DEV_KERNARG=1)
}
take_lease() {  # the lease must be ours and idle; the runner rewrites it as KEY=VALUE lines
  python3 - "$LEASE" <<'EOF'
import importlib, pathlib, sys
lease = importlib.import_module("v2.27b.launch").read_lease(pathlib.Path(sys.argv[1]))
if lease.get("track") != "27b" or lease.get("status") == "running":
    raise SystemExit(f"lease not usable: track={lease.get('track')} status={lease.get('status')}")
EOF
  printf 'track=27b\npurpose=%s\n' "$1" > "$LEASE"
}
release_lease() {
  python3 - "$LEASE" "$1" <<'EOF'
import importlib, json, pathlib, sys
owner = pathlib.Path(sys.argv[1])
lease = importlib.import_module("v2.27b.launch").read_lease(owner)
lease.update({"track": "27b", "status": "reserved-idle", "container": None, "last_run": sys.argv[2]})
owner.write_text(json.dumps(lease, sort_keys=True) + "\n", encoding="utf-8")
EOF
}
runner() {  # RUN_DIR CKPT CACHE_DIR PURPOSE [runner options]... -- same_panel collect arguments
  local dir=$1 ckpt=$2 cache=$3 purpose=$4 status=0 opts=()
  shift 4
  while [ $# -gt 0 ] && [ "$1" != -- ]; do opts+=("$1"); shift; done
  [ $# -gt 0 ] && shift
  case "$cache/" in "$dir"/*) ;; *) opts+=(--mount-rw "$cache") ;; esac
  kernel_env "$cache"
  python3 -c "import importlib, sys; importlib.import_module('v2.27b.launch').wait_until_idle(int(sys.argv[1]))" "$GPU"
  take_lease "$purpose"
  "$S/v2/eval/run_same_panel.sh" --gpu "$GPU" --track 27b --src "$SRC" --run-dir "$dir" \
    --model-dir "$ckpt" --mount "$BASE" --image "$IMAGE" "${KENV[@]}" "${opts[@]}" \
    --purpose "$purpose" -- --adapter-spec "$KERNEL_SPEC" --model-path "$ckpt" "$@" || status=$?
  release_lease "$dir"
  return "$status"
}
launcher() {  # NAME CAP PURPOSE RECEIPT CACHE_DIR -- launch.py mounts/env... -- container argv
  local name=$1 cap=$2 purpose=$3 receipt=$4 cache=$5
  shift 5
  [ "$1" = -- ] && shift
  kernel_env "$cache"
  python3 -m v2.27b.launch --name "$name" --gpu "$GPU" --cap-hours "$cap" --purpose "$purpose" \
    --receipt "$receipt" --mount "$S:/code" --mount "$BASE:$BASE" \
    --env PYTHONPATH=/code:/opt/decision-fla "${KENV[@]}" "$@"
}
