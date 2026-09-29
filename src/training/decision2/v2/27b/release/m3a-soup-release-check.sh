#!/usr/bin/env bash
# Release-path check of Milestone 3 finalist F1 (M3-A-soup) on node B GPU6: build, native examples in two
# processes, card example, full-panel parity against the scored T = 1 predictions. No --upload, no --collect.
# release.sh only accepts work dirs under /data/dev2/runs/release/; this check writes under
# /data/dev2/runs/27b/m3-release-check, reached as release/../27b/... (that parent is mounted read-only so the
# path also resolves inside the example containers).
set -euo pipefail
TS=$(date -u +%Y%m%dT%H%M%SZ)
SRC=35fa052d2b7c3ad0f9b2ee9bd1529e6b28076029-src_training_decision2; S=/data/dev2/src/$SRC/src/training/decision2
C=/data/dev2/runs/27b/m3-release-check
IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
BASE=/data/decision20-20260926/models/Qwen3.8-27B
G=/data/dev2/private/panels/goldfree; P=/data/dev2/runs/27b/M3-A-soup/formal/output
FROZEN=/data/dev2/runs/27b/M3-A-soup/formal/triton-cache
FROZEN_SHA=03b172f1a6adeef6c6a6c491d04389b355c9d8579480008023f408c8659b502b
TC=$C/triton/f1-copy-$TS
W=/data/dev2/runs/release/../27b/m3-release-check/release-$TS
GPU=6
LEASE=/data/dev2/leases/gpu$GPU.lock/owner
export TMPDIR=/data/dev2/tmp PYTHONPATH=$S
cd "$S"

python3 -c "import importlib, sys; importlib.import_module('v2.27b.launch').wait_until_idle(int(sys.argv[1]))" "$GPU"
python3 - "$LEASE" <<'EOF'
import importlib, pathlib, sys
lease = importlib.import_module("v2.27b.launch").read_lease(pathlib.Path(sys.argv[1]))
if lease.get("track") != "27b" or lease.get("status") == "running":
    raise SystemExit(f"lease not usable: track={lease.get('track')} status={lease.get('status')}")
EOF
printf 'track=27b\npurpose=%s\n' "M3-A-soup release-path check (no upload)" > "$LEASE"
# shellcheck disable=SC2329
release_lease() {
  python3 - "$LEASE" "$C/release-$TS" <<'EOF'
import importlib, json, pathlib, sys
owner = pathlib.Path(sys.argv[1])
lease = importlib.import_module("v2.27b.launch").read_lease(owner)
lease.update({"track": "27b", "status": "reserved-idle", "container": None, "last_run": sys.argv[2]})
owner.write_text(json.dumps(lease, sort_keys=True) + "\n", encoding="utf-8")
EOF
}
trap release_lease EXIT

mkdir -p "$C/triton"
python3 -m v2.27b.triton_cache copy --frozen "$FROZEN" --expect "$FROZEN_SHA" --dest "$TC"
set -x
"$S/v2/release/release.sh" --spec "$C/spec/dev2-27b-m3a-soup.candidate.json" --src "$SRC" --work "$W" \
  --image "$IMAGE" --gpu "$GPU" --track 27b --threads 4 \
  --site /opt/decision-fla --require-kernels --base-path "$BASE" \
  --env HIP_FORCE_DEV_KERNARG=1 --env TRITON_CACHE_AUTOTUNING=1 --env "TRITON_CACHE_DIR=$TC" --mount-rw "$TC" \
  --env "HF_HUB_CACHE=$C/hf-cache" --mount "$C/hf-cache" --mount /data/dev2/runs/release \
  --mount "$G" --mount "$P" \
  --parity "typed-final:$G/typed-final.prompts.jsonl:$P/typed-final.predictions.jsonl:1600" \
  --parity "css15:$G/css15.prompts.jsonl:$P/css15.predictions.jsonl:6547" \
  --parity "public231:$G/public231.prompts.jsonl:$P/public231.predictions.jsonl:231" || status=$?
set +x
python3 -m v2.27b.triton_cache finish --dest "$TC" || true
exit "${status:-0}"
