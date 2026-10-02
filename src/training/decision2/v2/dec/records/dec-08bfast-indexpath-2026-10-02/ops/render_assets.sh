#!/usr/bin/env bash
# Card assets of a 0.8B / 2B Index-first successor, rendered on the workstation (as the 9B release did): the tier's
# current-main card spec (0.8B round 3, 2B round 4) with the successor's candidate report and mlx-diag score (if run), every report read from a byte-identical
# local copy of its node A file (LOCAL/<node A path>), the private Index input built on node A, the Inter fonts and
# the repository logo; v2.release.card_assets of this checkout under Python 3.12.13, matplotlib 3.11.2 and
# Pillow 12.3.0 (uv). The receipt card-assets.json pins the report, Index, logo and font SHA-256s, so the release
# build re-checks it against node A's files. Output stays private.
# Usage: render_assets.sh 0p8b|2b LOCAL INDEX_JSON MODEL_SHA256 OUTPUT_DIR [FONTS_DIR]
set -euo pipefail
KEY=${1:?0p8b|2b} LOCAL=${2:?local root} INDEX=${3:?index json} MODEL=${4:?model sha256} OUT=${5:?output}
FONTS=${6:-/tmp/fonts/extras/ttf}
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
REPO_ROOT=$(cd "$S/../../.." && pwd)
LOGO=$REPO_ROOT/website/static/img/artworks/vllm-sr-logo.dark.png
[[ "$MODEL" =~ ^[0-9a-f]{64}$ ]] || { echo "MODEL_SHA256 must be 64 hex" >&2; exit 2; }
test ! -e "$OUT" || { echo "$OUT exists" >&2; exit 3; }
umask 077
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
CARD=card3
[ "$KEY" = 2b ] && CARD=card4
python3 - "$S/v2/release/specs/dev2-$KEY-$CARD.json" "$KEY" "$LOCAL" "$TMP/spec.json" <<'PY'
import json, sys
from pathlib import Path
base, key, local, out = sys.argv[1:5]
s = json.loads(Path(base).read_text())
run = f"/data/dev2/runs/release/dev2-{key}-ixf-t1"
for i, e in enumerate(s["card"]["reports"]):
    if i == 0:
        assert e["role"] == "candidate"
        e["report"] = f"{run}/REPORT.json"
        e.pop("mlx", None)
        if Path(local + f"{run}-mlx/mlx-diag.score.json").is_file():
            e["mlx"] = f"{run}-mlx/mlx-diag.score.json"
    for field in ("report", "mlx"):
        if e.get(field):
            path = Path(local + e[field])
            assert path.is_file(), path
            e[field] = str(path)
Path(out).write_text(json.dumps(s))
PY
mkdir -p "$OUT"
cd "$S"
uv run --python 3.12.13 --with matplotlib==3.11.2 --with pillow==12.3.0 --with numpy \
  env PYTHONPATH="$S" python -m v2.release.card_assets --spec "$TMP/spec.json" --index "$INDEX" --logo "$LOGO" \
  --fonts "$FONTS" --model-sha256 "$MODEL" --output "$OUT" > /dev/null
python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(json.dumps({"software": r["software"], "index": r["inputs"]["index_sha256"][:12], "files": len(r["files"])}))' "$OUT/card-assets.json"
echo "receipt $(sha256sum < "$OUT/card-assets.json" | cut -c1-64)"
