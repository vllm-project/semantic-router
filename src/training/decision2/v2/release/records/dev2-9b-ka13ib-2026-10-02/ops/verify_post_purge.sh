#!/usr/bin/env bash
# Post-purge verification of Decision-2.0-Lux-9B@259a4550 on node A (CPU only; no GPU, no lease).
set -euo pipefail
TS=$(date -u +%Y%m%dT%H%M%SZ)
W=/data/dev2/runs/release/dev2-9b-ka13ib-verify-$TS
S=/data/dev2/src/9e84f05f48760a5f4df236e37e4d4f861926b988-src_training_decision2/src/training/decision2
V=/data/dev2/src/787abdc54946ccdb05e52cc5c34ecb619dced237-src_training_decision2/src/training/decision2
PKG=/data/dev2/runs/release/dev2-9b-ka13ib-release-20261001T201936Z/package/Decision-2.0-Lux-9B
FROZEN=/data/dev2/runs/release/dev2-9b-ka13ib-prerelease-20261001T191732Z/package/Decision-2.0-Lux-9B
REPO=llm-semantic-router/Decision-2.0-Lux-9B
REV=259a45502bfca2f585a59ef99079533106e0b136
HFPY=/data/dev2/tools/hf-cli/bin/python
IMAGE=decision20-train-fast:host2
IMAGE_ID=sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54
export TMPDIR=/data/dev2/tmp HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p "$TMPDIR" "$W/receipts" "$W/download"
D=$W/download/Decision-2.0-Lux-9B
echo "work $W"
cd "$S"
date -u +%TZ; echo "download"
PYTHONPATH=$S "$HFPY" -m v2.release.hub download --repo "$REPO" --revision "$REV" --dest "$D" --output "$W/receipts/download.json"
date -u +%TZ; echo "tree"
PYTHONPATH=$S python3 -m v2.release.hub tree --package "$PKG" --download "$D" --output "$W/receipts/tree.json"
date -u +%TZ; echo "identity"
PYTHONPATH=$V python3 - "$D" "$FROZEN" > "$W/receipts/identity.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
from training.model.infer import checkpoint_fingerprint
d, frozen = Path(sys.argv[1]), Path(sys.argv[2])
fp = checkpoint_fingerprint(d)
m = json.loads((d / "MODEL_MANIFEST.json").read_text())
fm_bytes = (frozen / "MODEL_MANIFEST.json").read_bytes()
fm = json.loads(fm_bytes)
weights = sorted(p for p in fm["files_sha256"] if p.endswith(".safetensors"))
print(json.dumps({
    "schema": "dev2-verify-identity/1",
    "download_manifest_sha256": hashlib.sha256((d / "MODEL_MANIFEST.json").read_bytes()).hexdigest(),
    "model_sha256": fp["model_sha256"],
    "manifest_identity": m["identity"]["model_sha256"],
    "identity_equal": fp["model_sha256"] == m["identity"]["model_sha256"],
    "fingerprint_files": len(fp["files_sha256"]),
    "frozen_manifest_sha256": hashlib.sha256(fm_bytes).hexdigest(),
    "frozen_identity": fm["identity"]["model_sha256"],
    "weight_files": len(weights),
    "weights_equal_frozen": all(fp["files_sha256"].get(p) == fm["files_sha256"][p] for p in weights),
    "passed": fp["model_sha256"] == m["identity"]["model_sha256"] == fm["identity"]["model_sha256"]
    and all(fp["files_sha256"].get(p) == fm["files_sha256"][p] for p in weights),
}, indent=1))
PY
cat "$W/receipts/identity.json"
date -u +%TZ; echo "verify_bundle (image, no GPU devices)"
[[ "$(docker image inspect -f '{{.Id}}' "$IMAGE")" == "$IMAGE_ID" ]]
docker run --rm --network none -e HIP_VISIBLE_DEVICES= -e CUDA_VISIBLE_DEVICES= -v "$D:$D:ro" -w /tmp \
  --entrypoint python3 "$IMAGE" -I -B -c 'import json, sys; sys.path.insert(0, sys.argv[1]); from decision2 import verify_bundle; m = verify_bundle(sys.argv[1]); print(json.dumps({"verify_bundle": "ok", "files": len(m.get("files_sha256") or {}), "identity": (m.get("identity") or {}).get("model_sha256")}))' \
  "$D" | tee "$W/receipts/verify-bundle.json"
date -u +%TZ; echo "done"
python3 - "$W" <<'PY'
import json, sys
from pathlib import Path
w = Path(sys.argv[1]) / "receipts"
for name in ("download", "tree"):
    r = json.loads((w / f"{name}.json").read_text())
    print(name, {k: r.get(k) for k in ("files", "bytes", "seconds", "passed", "missing", "extra", "hash_mismatch", "differs_from_pre_upload_package", "manifest_sha256") if k in r})
PY
echo "W=$W"
