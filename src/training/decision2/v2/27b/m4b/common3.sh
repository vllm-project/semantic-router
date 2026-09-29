# shellcheck shell=bash disable=SC2034
# Host-side helpers of the M4b single-GPU drivers (sourced after v2/27b/kernel_common.sh, not run).
# kernel_common.sh's helpers with node B GPU0-2 and track 27b-m4b: launcher() goes through
# v2/27b/m4b/launch3.py (--gpus GPU), runner() passes --track 27b-m4b to the eval runner. The lease is
# launch3's main owner file (KEY=VALUE, track=27b-m4b, taken once with `launch3 lease`); every
# co-tenant owner.<name> (the eval track's owner.eval) must be released or idle before a job starts.
# The eval runner rewrites the owner file for its job; release_lease() restores launch3's fields.
# Drivers resolve the mirror before sourcing (MIRROR_SHA -> /data/dev2/src/<sha>[-src_training_decision2]):
#   SRC=$MIRROR_SHA; [ -d /data/dev2/src/$SRC ] || SRC=$MIRROR_SHA-src_training_decision2
TRACK=27b-m4b
M4B=/data/dev2/runs/27b/m4b
case "$GPU" in 0 | 1 | 2) ;; *) echo "GPU$GPU is outside the M4b allocation (node B GPU0-2)" >&2; exit 2 ;; esac
LEASE_KEEP=

take_lease() {  # PURPOSE: the owner file must be ours and idle, co-tenants released or idle
  LEASE_KEEP=$(python3 - "$GPU" "$LEASE" <<'EOF'
import importlib, json, pathlib, sys
launch3 = importlib.import_module("v2.27b.m4b.launch3")
gpu, owner = int(sys.argv[1]), pathlib.Path(sys.argv[2])
lease = launch3.base.read_lease(owner)
if lease.get("track") != launch3.TRACK or lease.get("status") == "running":
    raise SystemExit(f"lease not usable: track={lease.get('track')} status={lease.get('status')}")
launch3.check_cotenants(gpu)
print(json.dumps(lease, sort_keys=True))
EOF
  )
  printf 'track=%s\npurpose=%s\n' "$TRACK" "$1" > "$LEASE"
}
release_lease() {  # RUN_DIR
  python3 - "$LEASE" "$1" "$LEASE_KEEP" <<'EOF'
import importlib, json, pathlib, sys
launch3 = importlib.import_module("v2.27b.m4b.launch3")
owner, run, keep = pathlib.Path(sys.argv[1]), sys.argv[2], json.loads(sys.argv[3] or "{}")
after = launch3.base.read_lease(owner)
keep.update({k: after[k] for k in ("last_job_end_utc", "last_job_exit") if k in after})
keep.update({"track": launch3.TRACK, "status": "reserved-idle", "container": None, "last_run": run})
launch3.write_owner(owner, keep)
EOF
}
runner() {  # RUN_DIR CKPT CACHE_DIR PURPOSE [runner options]... -- same_panel collect arguments
  local dir=$1 ckpt=$2 cache=$3 purpose=$4 status=0 opts=()
  shift 4
  while [ $# -gt 0 ] && [ "$1" != -- ]; do opts+=("$1"); shift; done
  [ $# -gt 0 ] && shift
  case "$cache/" in "$dir"/*) ;; *) opts+=(--mount-rw "$cache") ;; esac
  kernel_env "$cache"
  local cmd=("$S/v2/eval/run_same_panel.sh" --gpu "$GPU" --track "$TRACK" --src "$SRC" --run-dir "$dir"
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
launcher() {  # NAME CAP PURPOSE RECEIPT CACHE_DIR -- launch3 mounts/env... -- container argv
  local name=$1 cap=$2 purpose=$3 receipt=$4 cache=$5 i
  shift 5
  [ "$1" = -- ] && shift
  kernel_env "$cache"
  local cmd=(python3 -m v2.27b.m4b.launch3 --name "$name" --gpus "$GPU" --cap-hours "$cap" --purpose "$purpose"
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
verify_mirror() {  # MIRROR_SHA: the mirror record names that commit
  python3 - "/data/dev2/src/$SRC/.dev2-mirror.json" "$1" <<'EOF'
import json, sys
record = json.load(open(sys.argv[1]))
if record["commit"] != sys.argv[2] or record["path"] != "src/training/decision2":
    raise SystemExit(f"{sys.argv[1]} is commit {record['commit']} path {record['path']}")
print(f"mirror {record['commit']} tree {record['tree']} ({record['files']} files)")
EOF
}
verify_cache() {  # FROZEN EXPECTED_SHA: the frozen cache's tree hash (read only)
  local got
  got=$(python3 -m v2.27b.triton_cache digest "$1")
  [ "$got" = "$2" ] || { echo "$1 is tree $got, expected $2" >&2; return 1; }
  echo "frozen cache $1 tree $got"
}
verify_lease() {  # the GPU lease is 27b-m4b's and no co-tenant is busy (read only)
  python3 - "$GPU" "$LEASE" <<'EOF'
import importlib, pathlib, sys
launch3 = importlib.import_module("v2.27b.m4b.launch3")
gpu, owner = int(sys.argv[1]), pathlib.Path(sys.argv[2])
lease = launch3.base.read_lease(owner) if owner.is_file() else {}
if lease.get("track") != launch3.TRACK:
    raise SystemExit(f"GPU{gpu} owner is track {lease.get('track')}; take it with launch3 lease first")
launch3.check_cotenants(gpu)
print(f"GPU{gpu} lease: track {lease['track']} status {lease.get('status')}")
EOF
}
no_autotune_added() {  # CACHE_DIR: its post-run check added no autotune entry
  python3 - "$1.post.json" <<'EOF'
import json, sys
post = json.load(open(sys.argv[1]))
added = post["classified"]["autotune"]["added"]
if added:
    raise SystemExit(f"{sys.argv[1]}: the run added {len(added) if isinstance(added, list) else added} autotune entries")
print(json.dumps({"cache": sys.argv[1], "post_sha256": post["post_sha256"], "autotune_added": 0}))
EOF
}
