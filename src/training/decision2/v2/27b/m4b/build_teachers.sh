#!/usr/bin/env bash
# M4b teacher files, built twice and compared byte for byte (node B host; CPU only). Each build is
# v2.27b.m4b.build_teachers in its own network-less, GPU-less container of the pinned image, with the code
# mirror, the TRAIN directory, the teacher sources, the HF dataset cache and the base (tokenizer, for native
# token counts) mounted read-only and only the output parent writable.
# Usage: build_teachers.sh MIRROR_SHA
#   -> /data/dev2/private/27b/m4b-data/teachers-1, teachers-2 (each: teacher-lux.jsonl, teacher-aj.jsonl,
#      S.ids.txt, MANIFEST.json) and teachers-compare.json. Run fetch_teacher_sources.sh first.
set -euo pipefail
echo "m4b build_teachers $* start $(date -u +%FT%TZ)"

MIRROR=$1
S=/data/dev2/src/$MIRROR/src/training/decision2
[ -d "$S" ] || S=/data/dev2/src/$MIRROR-src_training_decision2/src/training/decision2
[ -f "$S/v2/27b/m4b/build_teachers.py" ] || { echo "no m4b code in mirror $MIRROR" >&2; exit 2; }
S=$(cd "$S" && pwd -P)
IMAGE=sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1
BASE=${BASE:-/data/decision20-20260926/models/Qwen3.8-27B}
D=/data/dev2/private/27b/m4b-data
SPEC=$S/v2/27b/m4b/teacher-sources-m4b.json
RO=(
  /data/dev2/private/27b/m3-data/mixtures-m3-1
  /data/dev2/private/27b/m2-data/hf-12912429/m3
  /data/dev2/hf-cache/datasets--llm-semantic-router--decision-2.0-training-data
  # The dataset repo's blobs are symlinks into the cache's shared store.
  /data/dev2/hf-cache/blobs
  "$BASE"
)
for n in 1 2; do
  [ ! -e "$D/teachers-$n" ] || { echo "$D/teachers-$n exists" >&2; exit 66; }
done
[ ! -e "$D/teachers-compare.json" ] || { echo "$D/teachers-compare.json exists" >&2; exit 66; }
mkdir -p "$D"
export TMPDIR=/data/dev2/tmp

mounts=(--mount "type=bind,src=$S,dst=$S,readonly" --mount "type=bind,src=$D,dst=$D")
for path in "${RO[@]}"; do
  [ -e "$path" ] || { echo "missing $path" >&2; exit 2; }
  mounts+=(--mount "type=bind,src=$path,dst=$path,readonly")
done
for n in 1 2; do
  docker run --rm --name "d2-27b-m4b-teachers-$n" --network none --cpus "${BUILD_CPUS:-8}" \
    -e HIP_VISIBLE_DEVICES= -e ROCR_VISIBLE_DEVICES= -e "PYTHONPATH=$S" -e PYTHONDONTWRITEBYTECODE=1 \
    -e TOKENIZERS_PARALLELISM=false -e HF_HUB_OFFLINE=1 "${mounts[@]}" -w "$S" --entrypoint python3 "$IMAGE" \
    -m v2.27b.m4b.build_teachers --sources "$SPEC" --source "$BASE" --output-dir "$D/teachers-$n" \
    2>&1 | tee "$D/teachers-$n.log"
done

python3 - "$D" <<'EOF'
import hashlib, json, pathlib, sys
d = pathlib.Path(sys.argv[1])
one, two = d / "teachers-1", d / "teachers-2"
names = sorted(p.name for p in one.iterdir())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
result = {
    "schema": "decision2-27b-m4b-teachers-compare/1",
    "files": {n: sha(one / n) for n in names},
    "same_names": names == sorted(p.name for p in two.iterdir()),
    "identical": {n: (one / n).read_bytes() == (two / n).read_bytes() for n in names},
}
result["byte_identical"] = result["same_names"] and all(result["identical"].values())
with (d / "teachers-compare.json").open("x") as stream:
    json.dump(result, stream, indent=1, sort_keys=True)
    stream.write("\n")
print(json.dumps({"byte_identical": result["byte_identical"], "files": result["files"]}))
raise SystemExit(0 if result["byte_identical"] else 1)
EOF
echo "m4b build_teachers complete"
