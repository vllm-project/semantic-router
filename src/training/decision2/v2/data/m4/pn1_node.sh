#!/usr/bin/env bash
# PN1 operations on node B (prereg v2/data/records/m4-pn1-prereg-2026-09-29.md). Runs only from an exact
# mirror (/data/dev2/src/<sha>[-src_training_decision2], see v2/common/mirror_to_node.sh).
#
# Usage:
#   pn1_node.sh download                     pin the Tatoeba export: host curl, bzip2 -t, source-manifest.json
#   pn1_node.sh cpu <pn1_build args...>      one CPU step in the offline image (no GPU device)
#   pn1_node.sh gpu <3|4> <job> <budget-s> -- <pn1_gpu args...>
#                                            one GPU job in the offline image on GPU3 or GPU4
#   pn1_node.sh dry <pn1_gpu args...>        the same job with --dry-run in a CPU container (no GPU)
#   pn1_node.sh warm <model-dir>             read the weight files once on the host (page cache; no GPU)
#   pn1_node.sh seed-cache <triton-cache>    start the persisted Triton cache from a copy (before any job)
#   pn1_node.sh lease <3|4> <status> [note]  rewrite this track's lease entry gpuN.lock/owner.data
#   pn1_node.sh gpuh                         GPU-hours of the recorded PN1 jobs
#
# Writes only under /data/dev2: the export and all text under /data/dev2/private/, logs, GPU-TIME receipts
# and the Triton cache under /data/dev2/runs/data/m4-pn1/. A GPU job needs the owner entry of gpuN.lock not
# running and 0% VRAM; it writes gpuN.lock/owner.data (track=data), passes only /dev/kfd and that GPU's
# render node with ROCR/HIP/CUDA_VISIBLE_DEVICES=0, checks the PCI bus inside, runs --network none with
# HF_HUB_OFFLINE=1, and hands the job its budget minus 60 s for container start. No job starts once the
# recorded jobs reach 1.05 GPU-h.
set -euo pipefail

usage() { sed -n '2,/^set -euo pipefail$/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//' >&2; exit 2; }

S=$(cd "$(dirname "$0")/../../.." && pwd -P)
M=$(cd "$S/../../.." && pwd -P)
[ -f "$M/.dev2-mirror.json" ] || { echo "not an exact mirror: $M" >&2; exit 2; }
COMMIT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "$M/.dev2-mirror.json")
TREE=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tree"])' "$M/.dev2-mirror.json")
IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
EXPORT=2026-09-26
BASE=https://downloads.tatoeba.org/exports
SRC=/data/dev2/private/sources/tatoeba-$EXPORT
PRIV=/data/dev2/private/data/m4-pn1
RUNS=/data/dev2/runs/data/m4-pn1
TC=$RUNS/triton-cache
HF=/data/dev2/hf-cache
GPUH_STOP=1.05
declare -A RENDER=([3]=/dev/dri/renderD153 [4]=/dev/dri/renderD161)
declare -A BUS=([3]=9b [4]=a3)
export TMPDIR=/data/dev2/tmp
mkdir -p "$TMPDIR"

die() { echo "pn1_node: $*" >&2; exit 1; }
now() { date -u +%FT%TZ; }

download() {
  [ ! -e "$SRC/source-manifest.json" ] || die "export already pinned at $SRC"
  mkdir -p "$SRC" && chmod 700 "$SRC"
  local files=(links.tar.bz2 sentences_CC0.tar.bz2) lang f
  for lang in ara cmn deu fra jpn kor rus spa eng; do
    files+=("per_language/$lang/${lang}_sentences_detailed.tsv.bz2")
  done
  for f in "${files[@]}"; do
    mkdir -p "$(dirname "$SRC/$f")"
    curl -sSfL --retry 3 --max-time 1800 -D "$SRC/$f.headers" -o "$SRC/$f" "$BASE/$f"
    bzip2 -tq "$SRC/$f" || die "$f fails bzip2 -t"
  done
  python3 - "$SRC" "$BASE" "$EXPORT" "${files[@]}" <<'EOF'
import email.utils, hashlib, json, os, sys, time
root, base, export, *files = sys.argv[1:]
entries, dates = {}, set()
for name in files:
    blocks = open(os.path.join(root, name + ".headers"), encoding="latin-1").read().strip().split("\r\n\r\n")
    headers = {}
    for line in blocks[-1].splitlines()[1:]:
        key, _, value = line.partition(":")
        headers[key.strip().lower()] = value.strip()
    path = os.path.join(root, name)
    size = os.path.getsize(path)
    if "content-length" in headers and int(headers["content-length"]) != size:
        sys.exit(f"{name}: size {size} != Content-Length {headers['content-length']}")
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    modified = headers.get("last-modified")
    if not modified:
        sys.exit(f"{name}: no Last-Modified header; the export cannot be pinned")
    dates.add(email.utils.parsedate_to_datetime(modified).date().isoformat())
    entries[name] = {"url": f"{base}/{name}", "last_modified": modified, "size": size, "sha256": digest.hexdigest()}
if dates != {export}:
    sys.exit(f"export dates {sorted(dates)} are not the single pinned date {export}")
manifest = {"schema": "dev2-m4-pn1-source/1", "export_date": export, "base_url": base,
            "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": entries}
with open(os.path.join(root, "source-manifest.json"), "x") as stream:
    json.dump(manifest, stream, indent=1, sort_keys=True)
    stream.write("\n")
print(json.dumps({name: entry["sha256"][:12] for name, entry in entries.items()}))
EOF
  rm -f "$SRC"/*.headers "$SRC"/per_language/*/*.headers
  chmod -R a-w "$SRC"
}

record() {  # KIND NAME START END STATUS [BUDGET GPU]
  local dir=$RUNS/$1-time
  mkdir -p "$dir"
  python3 - "$dir/$2.json" "$@" "$COMMIT" "$TREE" "$IMAGE" <<'EOF'
import datetime, json, sys
path, kind, name, start, end, status, *rest = sys.argv[1:]
commit, tree, image = rest[-3:]
extra = rest[:-3]
parse = lambda s: datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
wall = (parse(end) - parse(start)).total_seconds()
record = {"schema": "dev2-gpu-time/1" if kind == "gpu" else "dev2-cpu-time/1", "track": "data", "job": name,
          "start_utc": start, "end_utc": end, "wall_seconds": wall, "exit_code": int(status),
          "source_commit": commit, "source_tree": tree, "image_id": image}
if kind == "gpu":
    record.update(budget_seconds=float(extra[0]), gpu=int(extra[1]), gpus=1, gpu_hours=wall / 3600)
json.dump(record, open(path, "x"), indent=1, sort_keys=True)
EOF
}

gpuh() {
  python3 - "$RUNS/gpu-time" <<'EOF'
import glob, json, os, sys
total = sum(json.load(open(p))["gpu_hours"] for p in glob.glob(os.path.join(sys.argv[1], "*.json")))
print(f"{total:.6f}")
EOF
}

write_lease() {  # GPU STATUS NOTE [START EXPECTED_END JOB]
  local dir=/data/dev2/leases/gpu$1.lock tmp
  mkdir -p "$dir"
  tmp=$(mktemp "$dir/.owner.data.XXXX")
  {
    echo "track=data"
    echo "purpose=PN1 generation/judge (research & data, M4; GPU3-4 lent to data)"
    echo "status=$2"
    echo "note=$3"
    [ -n "${4:-}" ] && echo "start_utc=$4"
    [ -n "${5:-}" ] && echo "expected_end_utc=$5"
    [ -n "${6:-}" ] && echo "job=$6"
    echo "updated_utc=$(now)"
    echo "source_commit=$COMMIT"
  } > "$tmp"
  mv "$tmp" "$dir/owner.data"
}

cpu() {
  mkdir -p "$PRIV" "$RUNS"
  chmod 700 "$PRIV"
  local name="cpu-$1-$(date -u +%Y%m%dT%H%M%SZ)" start status
  start=$(now)
  set +e
  docker run --rm --name "dev2-data-pn1-$name" --network none \
    -e HIP_VISIBLE_DEVICES= -e ROCR_VISIBLE_DEVICES= -e CUDA_VISIBLE_DEVICES= \
    -e PYTHONPATH="$S" -e PYTHONDONTWRITEBYTECODE=1 \
    -v "$M:$M:ro" -v "$SRC:$SRC:ro" -v "$PRIV:$PRIV" -v "$RUNS:$RUNS:ro" \
    -w "$S" --entrypoint python3 "$IMAGE" -m v2.data.m4.pn1_build "$@"
  status=$?
  set -e
  record cpu "$name" "$start" "$(now)" "$status"
  return "$status"
}

gpu() {
  local gpu=$1 job=$2 budget=$3 start status used vram lease run
  shift 3
  [ "${1:-}" = -- ] && shift
  [ -n "${RENDER[$gpu]:-}" ] || die "GPU $gpu is not lent to research & data"
  [[ "$job" =~ ^[a-z0-9-]+$ ]] || die "bad job name"
  lease=/data/dev2/leases/gpu$gpu.lock
  if grep -q '^status=running' "$lease/owner" 2>/dev/null; then die "gpu$gpu owner entry is running"; fi
  vram=$(rocm-smi -d "$gpu" --showmemuse 2>/dev/null | awk -F': ' '/VRAM%/ {v = $NF} END {print v}')
  [ "$vram" = 0 ] || die "gpu$gpu is not idle (VRAM%=${vram:-unknown})"
  used=$(gpuh)
  python3 -c "import sys; sys.exit(float(sys.argv[1]) >= float(sys.argv[2]))" "$used" "$GPUH_STOP" \
    || die "recorded PN1 GPU time $used h reached the $GPUH_STOP h stop"
  run=$RUNS/jobs/$job
  [ ! -e "$run" ] || die "job $job already ran"
  mkdir -p "$run" "$TC" "$PRIV"
  start=$(now)
  write_lease "$gpu" running "job $job" "$start" "$(date -u -d "+$budget seconds" +%FT%TZ)" "$job"
  set +e
  docker run --rm --name "dev2-data-pn1-$job" --network none \
    --device /dev/kfd --device "${RENDER[$gpu]}" --group-add video --ipc host --shm-size 16g \
    --security-opt seccomp=unconfined \
    -e ROCR_VISIBLE_DEVICES=0 -e HIP_VISIBLE_DEVICES=0 -e CUDA_VISIBLE_DEVICES=0 \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_CACHE="$HF" \
    -e PYTHONPATH="$S:/opt/decision-fla" -e PYTHONDONTWRITEBYTECODE=1 \
    -e TRITON_CACHE_DIR="$TC" -e TRITON_CACHE_AUTOTUNING=1 -e HIP_FORCE_DEV_KERNARG=1 \
    -v "$M:$M:ro" -v "$HF:$HF:ro" -v "$PRIV:$PRIV" -v "$TC:$TC" \
    -w "$S" --entrypoint python3 "$IMAGE" -m v2.data.m4.pn1_gpu "$@" \
    --budget-seconds "$((budget - 60))" --expect-pci-bus "${BUS[$gpu]}" \
    > "$run/stdout.log" 2> "$run/stderr.log"
  status=$?
  set -e
  record gpu "$job" "$start" "$(now)" "$status" "$budget" "$gpu"
  write_lease "$gpu" idle "last job $job exit $status; more PN1 jobs may follow" "$start" "" "$job"
  return "$status"
}

dry() {
  docker run --rm --name "dev2-data-pn1-dry-$(date -u +%H%M%S)" --network none \
    -e HIP_VISIBLE_DEVICES= -e ROCR_VISIBLE_DEVICES= -e CUDA_VISIBLE_DEVICES= \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_CACHE="$HF" \
    -e PYTHONPATH="$S:/opt/decision-fla" -e PYTHONDONTWRITEBYTECODE=1 \
    -v "$M:$M:ro" -v "$HF:$HF:ro" -v "$PRIV:$PRIV:ro" \
    -w "$S" --entrypoint python3 "$IMAGE" -m v2.data.m4.pn1_gpu "$@" --budget-seconds 1 --dry-run
}

case "${1:-}" in
  download) download ;;
  cpu) shift; cpu "$@" ;;
  gpu) shift; gpu "$@" ;;
  dry) shift; dry "$@" ;;
  warm) cat "$2"/*.safetensors > /dev/null ;;
  seed-cache)
    [ ! -e "$TC" ] || die "$TC exists"
    mkdir -p "$RUNS"
    cp -a "$2" "$TC"
    chmod -R u+w "$TC"
    (cd "$TC" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum) > "$RUNS/triton-cache.seed.sha256"
    echo "seeded from $2: $(wc -l < "$RUNS/triton-cache.seed.sha256") files, manifest $(sha256sum "$RUNS/triton-cache.seed.sha256" | cut -d' ' -f1)" ;;
  lease) write_lease "$2" "$3" "${4:-}" ;;
  gpuh) gpuh ;;
  *) usage ;;
esac
