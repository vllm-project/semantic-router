#!/usr/bin/env bash
# DEV2.0-8B storage test, step 1: BF16-storage copy of the scored K-a13 soup with v2.release.bf16_copy (Linear
# projection weights BF16 as autocast rounds them; every other tensor FP32 bit for bit). Scored image, CPU only,
# no network; the module's unit tests run first in the same image. Node A, from the exact mirror holding this file.
set -euo pipefail
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
R=/data/dev2/runs/9b
SOUP=$R/m4/K-a13-build/soup
O=/data/dev2/runs/release/inputs/dev2-8b-bf16
IMAGE=decision20-train-fast:host2
test "$(docker image inspect -f '{{.Id}}' $IMAGE)" = sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
echo "mirror: $S"
(cd $R/m4 && sha256sum -c --quiet K-a13-build/SHA256SUMS)
echo "source verified: $(wc -l < $R/m4/K-a13-build/SHA256SUMS) files"
mkdir $O
docker run --rm --network none -e HIP_VISIBLE_DEVICES= -e CUDA_VISIBLE_DEVICES= -e ROCR_VISIBLE_DEVICES= \
  -e PYTHONPATH="$S" -v "$S:$S:ro" -v "$SOUP:$SOUP:ro" -v "$O:$O" -w "$S" --entrypoint python3 $IMAGE \
  -B -m unittest -v v2.release.tests.test_bf16_copy
docker run --rm --network none -e HIP_VISIBLE_DEVICES= -e CUDA_VISIBLE_DEVICES= -e ROCR_VISIBLE_DEVICES= \
  -e PYTHONPATH="$S" -v "$S:$S:ro" -v "$SOUP:$SOUP:ro" -v "$O:$O" -w "$S" --entrypoint python3 $IMAGE \
  -B -m v2.release.bf16_copy --source "$SOUP" --output "$O/checkpoint" --receipt "$O/bf16-copy.json"
sha256sum $O/bf16-copy.json
du -sb $O/checkpoint
