#!/usr/bin/env bash
# Decision-2.0-Vega-27B successor over M6-IB (27B worker, continuation #5): the card assets on node A with the default
# v2.release.card_assets (banner concept A, the four charts) in the card render environment of the earlier rounds
# (render_env.sh of dev2-27b-indexfirst-2026-10-02: CPython 3.12.13, matplotlib 3.11.2, Pillow 12.3.0, the Inter 4.1
# fonts and the vLLM-SR logo of their receipts). The render spec is the current revision's spec
# (specs/dev2-27b-27bif-M6-IB.json) with the ARM's report and mlx-diag score (make_27bx.paths); the inputs are read in
# place on node A and the outputs (assets/ and card-assets.json, which pins input and output digests) go to
# $PRIV/27b/, where the release build re-checks the receipt against the original files.
# Usage (node A): bash <mirror>/v2/release/records/dev2-27b-xarm-2026-10-02/ops/render27bx.sh ARM
set -euo pipefail
ARM=${1:?ARM}
S=$(cd "$(dirname "$0")/../../../../.." && pwd)
OPS=$S/v2/release/records/dev2-27b-xarm-2026-10-02/ops
ENV=${CARD_RENDER_ENV:-/data/dev2/tools/card-render}
[[ "$("$ENV/venv/bin/python" -c 'import sys, matplotlib, PIL; print(sys.version.split()[0], matplotlib.__version__, PIL.__version__)')" \
  == "3.12.13 3.11.2 12.3.0" ]] || { echo "render environment differs (run render_env.sh)" >&2; exit 2; }
[[ "$(sha256sum < "$ENV/logo.png" | cut -c1-16)" == e6b8428eb67f9318 ]] || { echo "logo differs" >&2; exit 2; }
read -r PRIV CKPT < <(cd "$S" && PYTHONPATH=$S python3 -c '
import importlib.util, sys
spec = importlib.util.spec_from_file_location("make_27bx", sys.argv[1]); m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m); p = m.paths(sys.argv[2]); print(p["private"], p["checkpoint"])' "$OPS/make_27bx.py" "$ARM")
[[ -f "$PRIV/decision-index-card.json" ]] || { echo "run index27bx.sh $ARM first" >&2; exit 3; }
[[ ! -e "$PRIV/27b" ]] || { echo "$PRIV/27b exists" >&2; exit 3; }
W=$(mktemp -d /data/dev2/tmp/render27bx-XXXXXX)
trap 'rm -rf "$W"' EXIT
(cd "$S" && PYTHONPATH=$S python3 - "$OPS/make_27bx.py" "$ARM" "$W/render-spec.json" <<'PY')
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("make_27bx", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
p = m.paths(sys.argv[2])
s = json.loads(m.BASE_SPEC.read_text())
reports = s["card"]["reports"]
assert reports[0]["role"] == "candidate" and reports[0]["label"] == m.NAME
reports[0]["report"] = f"{p['run']}/REPORT.json"
reports[0]["mlx"] = f"{p['mlx']}/mlx-diag.score.json"
json.dump(s, open(sys.argv[3], "w"), indent=2)
PY
identity=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["output"]["model_sha256"])' "$CKPT/soup_manifest.json")
[[ "$identity" =~ ^[0-9a-f]{64}$ ]] || { echo "no soup identity" >&2; exit 3; }
umask 077
mkdir -p "$PRIV/27b"
(cd "$S" && PYTHONPATH=$S "$ENV/venv/bin/python" -B -m v2.release.card_assets --spec "$W/render-spec.json" \
  --index "$PRIV/decision-index-card.json" --logo "$ENV/logo.png" --fonts "$ENV/fonts" --model-sha256 "$identity" \
  --output "$PRIV/27b") > "$PRIV/render.log"
echo "card-assets.json $(sha256sum < "$PRIV/27b/card-assets.json" | cut -c1-64) -> $PRIV/27b"
