# shellcheck shell=bash disable=SC2034
# Shared host-side helpers of the ~27B kernel-path scripts (Milestone 3; sourced, not run).
# The caller sets S (mirror decision2 directory), SRC (mirror name) and GPU, and runs from $S
# with PYTHONPATH=$S.
# DRY_RUN=1: no container touches a GPU and no lease changes. Every GPU launch (launch.py, the eval
# runner) is printed after its mount sources are checked and its container command line is run
# through the target module's argparse (argcheck, a CPU-only container of the pinned image).
# Frozen-cache copies and host CPU steps whose inputs exist still run.
S=$(cd "$S" && pwd -P)
BASE=${BASE:-/data/decision20-20260926/models/Qwen3.8-27B}
IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
KERNEL_SPEC=$S/v2/27b/adapters/typed-lora-kernel.json
CAL_FILE=${CAL_FILE:-/data/dev2/hf-cache/datasets--llm-semantic-router--decision-2.0-training-data/snapshots/ed87a03ab80ca5b9560780bba51a83a77ff47d14/m2/cal/CAL698/cal.jsonl}
CAL_SHA256=${CAL_SHA256:-19cc1a8c4ebe6fd13031079f6b7d131046ce503f435f894e3ea672f91c2ed41f}
PANEL_ROOT=/data/dev2/private/panels
LEASE=/data/dev2/leases/gpu$GPU.lock/owner
DRY_RUN=${DRY_RUN:-0}
[ -f "$KERNEL_SPEC" ] || { echo "missing kernel adapter in mirror $SRC" >&2; exit 2; }

model_sha() {  # CALIBRATION -> its bound model_sha256 (a placeholder in a dry run before the fit)
  if [ "$DRY_RUN" = 1 ] && [ ! -f "$1" ]; then
    echo "dry-run-model-sha256-of-$1"
    return
  fi
  python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['model_sha256'])" "$1"
}
dry() {  # COMMAND...: run it, or print it with DRY_RUN=1
  if [ "$DRY_RUN" = 1 ]; then
    printf 'dry-run:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}
need() {  # PATH...: every path exists
  local path
  for path in "$@"; do
    [ -e "$path" ] || { echo "missing $path" >&2; return 2; }
  done
}
need_mounts() {  # launcher / runner options up to --: every --mount, --mount-rw, --model-dir source exists
  while [ $# -gt 0 ] && [ "$1" != -- ]; do
    case "$1" in
      --mount | --mount-rw | --model-dir) need "${2%%:*}" || return; shift 2 ;;
      *) shift ;;
    esac
  done
}
argcheck() {  # MODULE ARGS...: the module's argparse accepts ARGS (CPU-only container, no GPU device)
  docker run --rm --network none -e HIP_VISIBLE_DEVICES= -e ROCR_VISIBLE_DEVICES= \
    -e "PYTHONPATH=$S" -e PYTHONDONTWRITEBYTECODE=1 --mount "type=bind,src=$S,dst=$S,readonly" \
    -w "$S" --entrypoint python3 "$IMAGE" -m v2.27b.kernel_readout argcheck -- "$@"
}
cache_copy() {  # FROZEN EXPECTED_SHA DEST: fresh cp -a copy; fails unless its tree hash is EXPECTED_SHA
  python3 -m v2.27b.triton_cache copy --frozen "$1" --expect "$2" --dest "$3"
}
cache_finish() {  # DEST: post-run tree hash, classified changes and the frozen check into DEST.post.json
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
  local cmd=("$S/v2/eval/run_same_panel.sh" --gpu "$GPU" --track 27b --src "$SRC" --run-dir "$dir"
    --model-dir "$ckpt" --mount "$BASE" --image "$IMAGE" "${KENV[@]}" "${opts[@]}"
    --purpose "$purpose" -- --adapter-spec "$KERNEL_SPEC" --model-path "$ckpt" "$@")
  if [ "$DRY_RUN" = 1 ]; then
    need_mounts "${cmd[@]:1}" && need "$cache" "/data/dev2/src/$SRC/.dev2-mirror.json" || return
    dry "${cmd[@]}"
    argcheck v2.eval.same_panel collect --run-dir "$dir" --panel-root "$PANEL_ROOT" \
      --adapter-spec "$KERNEL_SPEC" --model-path "$ckpt" "$@"
    return
  fi
  python3 -c "import importlib, sys; importlib.import_module('v2.27b.launch').wait_until_idle(int(sys.argv[1]))" "$GPU"
  take_lease "$purpose"
  "${cmd[@]}" || status=$?
  release_lease "$dir"
  return "$status"
}
launcher() {  # NAME CAP PURPOSE RECEIPT CACHE_DIR -- launch.py mounts/env... -- container argv
  local name=$1 cap=$2 purpose=$3 receipt=$4 cache=$5 i
  shift 5
  [ "$1" = -- ] && shift
  kernel_env "$cache"
  local cmd=(python3 -m v2.27b.launch --name "$name" --gpu "$GPU" --cap-hours "$cap" --purpose "$purpose"
    --receipt "$receipt" --mount "$S:/code" --mount "$BASE:$BASE"
    --env PYTHONPATH=/code:/opt/decision-fla "${KENV[@]}" "$@")
  if [ "$DRY_RUN" = 1 ]; then
    [ ! -e "$receipt" ] || { echo "receipt $receipt exists" >&2; return 1; }
    need_mounts "${cmd[@]:3}" && need "$cache" || return
    dry "${cmd[@]}"
    for ((i = 1; i <= $#; i++)); do
      if [ "${!i}" = -- ]; then
        argcheck "${@:i+3}"
        return
      fi
    done
    echo "launcher: no container command" >&2
    return 2
  fi
  "${cmd[@]}"
}
