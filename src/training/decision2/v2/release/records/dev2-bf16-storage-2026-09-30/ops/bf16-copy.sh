#!/usr/bin/env bash
# BF16 storage revisions of the released FP32 packages (coordinator note 2026-09-30 06:10, 27B release path step 1),
# step 1: the v2.release.bf16_copy of each released FP32 checkpoint (Linear projection weights BF16 exactly as BF16
# autocast rounds them; every other tensor FP32 bit for bit). Every fingerprint file of the source checkpoint is first
# checked against the released MODEL_MANIFEST.json (identity.fingerprint_files). Scored image, CPU only, no network;
# the module's unit tests run first in the same image.
# Usage (node A, from the exact mirror holding this file): bash <mirror>/.../ops/bf16-copy.sh 0.8B 2B 4B
set -euo pipefail
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
R=$S/v2/release/records
IMAGE=decision20-train-fast:host2
test "$(docker image inspect -f '{{.Id}}' $IMAGE)" = sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
echo "mirror: $S"
cpu=(docker run --rm --network none -e HIP_VISIBLE_DEVICES= -e CUDA_VISIBLE_DEVICES= -e ROCR_VISIBLE_DEVICES=
  -e PYTHONPATH="$S" -v "$S:$S:ro" -w "$S" --entrypoint python3)
"${cpu[@]}" "$IMAGE" -B -m unittest -v v2.release.tests.test_bf16_copy
for tier in "$@"; do
  case "$tier" in
    0.8B) key=0p8b src=/data/dev2/runs/dec/m2/formal-candidates/staging-16c0929a/m2/E8F-soup/checkpoint ;;
    2B) key=2b src=/data/dev2/runs/dec/m3/formal-candidates/staging-545a6784/m3/S2T-soup/checkpoint ;;
    4B) key=4b src=/data/dev2/runs/dec/m4/formal-candidates/staging-e8656221/m4/N4XF-soup/checkpoint ;;
    *) echo "tier must be 0.8B, 2B or 4B" >&2; exit 2 ;;
  esac
  manifest=$R/dev2-c1-card-pass-2026-09-29/$key/package-text/MODEL_MANIFEST.json
  out=/data/dev2/runs/release/inputs/dev2-$key-bf16
  python3 - "$manifest" "$src" <<'PY'
import hashlib, json, sys
from pathlib import Path
manifest, src = json.load(open(sys.argv[1])), Path(sys.argv[2])
files = manifest["identity"]["fingerprint_files"]
for name, expected in sorted(files.items()):
    h = hashlib.sha256()
    with (src / name).open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    assert h.hexdigest() == expected, f"{name} differs from the released manifest"
print(f"source verified: {len(files)} fingerprint files equal the released manifest (identity {manifest['identity']['model_sha256'][:12]})")
PY
  mkdir "$out"
  "${cpu[@]}" -v "$src:$src:ro" -v "$out:$out" "$IMAGE" \
    -B -m v2.release.bf16_copy --source "$src" --output "$out/checkpoint" --receipt "$out/bf16-copy.json"
  echo "$tier receipt $(sha256sum < "$out/bf16-copy.json" | cut -c1-64) bytes $(du -sb "$out/checkpoint" | cut -f1)"
done
