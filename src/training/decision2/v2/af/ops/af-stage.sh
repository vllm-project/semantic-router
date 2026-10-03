#!/usr/bin/env bash
# Arm factory release-form staging (the owners' method: M17 m17-index.sh "stage", M10 ix.sh "bf16"), node side, CPU:
# the v2.release.bf16_copy of a built FP32 soup / point (its unit test first; image host2 f83b1d10, no network; the
# copy's source fingerprint must equal the soup's model SHA-256), then v2.eval.ix1.restage onto the tier's released
# IX1 package (4B: DEV2.0-4B-13d42143, loaded 4,208,383,488; 9B: DEV2.0-9B-e51f9881, loaded 7,940,895,744), with the
# identity / loaded-count / calibration-none checks. Package: /data/dev2/models/ix1/af/AF-<NAME>-bf16-r<base8>, the
# IX1 DIAGNOSTIC entry AF-<NAME>-bf16.
#
# usage: AF_NODE=a|b|c|f af-stage.sh <mirror-dir> <NAME>
set -euo pipefail
SRC=$1 NAME=$2
NODE=${AF_NODE:?set AF_NODE}
case $NODE in
  a) SIZE=9b BASEPKG=/data/dev2/models/ix1/DEV2.0-9B-e51f9881 LOADED=7940895744 TAG=re51f9881 ;;
  *) SIZE=4b BASEPKG=/data/dev2/models/ix1/DEV2.0-4B-13d42143 LOADED=4208383488 TAG=r13d42143 ;;
esac
S=/data/dev2/src/$SRC/src/training/decision2
M=/data/dev2/runs/af/$SIZE
MD=/data/dev2/models/ix1/af
IMAGE=decision20-train-fast:host2
IMAGE_ID=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
IX=AF-$NAME-bf16
PKG=$MD/$IX-$TAG CK=$MD/ckpt/$IX
log() { echo "$(date -u +%FT%TZ) stage-$NAME $*" | tee -a "$M/OPERATIONS.log"; }
grep -qE "^for _af in (.* )?$NAME .*# arm factory ${SIZE^^} " "$S/v2/eval/ix1/launch.sh" \
  || { echo "the mirror's IX1 launcher has no arm-factory entry for $NAME" >&2; exit 2; }
[ -f "$M/soup/$NAME/DONE" ] && [ -f "$M/soup/$NAME/MODEL_SHA256" ] || { echo "no built soup $NAME" >&2; exit 3; }
[ ! -e "$PKG" ] && [ ! -e "$CK" ] || { echo "$IX is already staged" >&2; exit 3; }
[ -d "$BASEPKG" ] || { echo "no base package $BASEPKG" >&2; exit 3; }
[ "$(docker image inspect -f '{{.Id}}' "$IMAGE")" = "$IMAGE_ID" ] || { echo "image $IMAGE is not $IMAGE_ID" >&2; exit 3; }
src=$(cat "$M/soup/$NAME/DONE") fp32=$(cat "$M/soup/$NAME/MODEL_SHA256")
mkdir -p "$MD/ckpt" "$MD/receipts"
cpu=(docker run --rm --network none -e HIP_VISIBLE_DEVICES= -e CUDA_VISIBLE_DEVICES= -e ROCR_VISIBLE_DEVICES=
  -e PYTHONPATH="$S" -v "$S:$S:ro" -v "$src:$src:ro" -v "$MD:$MD" -w "$S" --entrypoint python3 "$IMAGE" -B)
"${cpu[@]}" -m unittest v2.release.tests.test_bf16_copy > "$MD/receipts/$IX-test_bf16_copy.log" 2>&1
"${cpu[@]}" -m v2.release.bf16_copy --source "$src" --output "$CK" --receipt "$MD/receipts/$IX-bf16-copy.json" \
  > "$MD/receipts/$IX-bf16.log" 2>&1
model=$(python3 - "$MD/receipts/$IX-bf16-copy.json" "$fp32" << 'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
assert r["source_model_sha256"] == sys.argv[2], f"source fingerprint {r['source_model_sha256']} != {sys.argv[2]}"
print(r["model_sha256"])
EOF
)
[[ "$model" =~ ^[0-9a-f]{64}$ ]] || { echo "bad bf16 receipt" >&2; exit 3; }
(cd "$S" && PYTHONPATH=$S python3 -B -m v2.eval.ix1.restage --package "$BASEPKG" --out "$PKG" --checkpoint "$CK" \
  --model-sha256 "$model" > "$MD/receipts/$IX-restage.log")
python3 - "$PKG/MODEL_MANIFEST.json" "$model" "$LOADED" << 'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
assert m["identity"]["model_sha256"] == sys.argv[2], "identity"
assert m["parameters"]["loaded"] == int(sys.argv[3]), f"loaded {m['parameters']['loaded']}"
assert m["calibration"] is None
EOF
log "staged $IX: FP32 ${fp32:0:12} -> BF16 ${model:0:12}, loaded $LOADED, T = 1, manifest $(sha256sum < "$PKG/MODEL_MANIFEST.json" | cut -c1-16)"
