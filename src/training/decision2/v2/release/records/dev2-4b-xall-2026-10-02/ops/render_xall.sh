#!/usr/bin/env bash
# Decision-2.0-Nox-4B M17 4b-SDMLxALL release (decoder M17; workstation side), as the SDML release's
# render_sdml.sh: the card assets with the default
# v2.release.card_assets (banner concept A, the four charts) in the card render environment of the earlier rounds
# (Python 3.12.13, matplotlib 3.11.2, Pillow 12.3.0, the Inter 4.1 fonts and the vLLM-SR logo of their receipts).
# The render spec is the current revision's card spec (specs/dev2-4b-ra.json) with the candidate's report and
# mlx-diag score (make_xall.paths). Every input the render reads (the card's reports and mlx-diag scores, the private
# Index input built by index_xall.sh) is copied from node A into a private local root with its SHA-256 checked; the
# outputs (assets/ and card-assets.json, which pins input and output digests) go back to node A $PRIV/4b/, where the
# release build re-checks the receipt against the original files.
# Usage (from this worktree): render_xall.sh
set -euo pipefail
CAND=XALL
NODES=${DEV2_NODES_FILE:-$HOME/.config/decision2/nodes.env}
addr() { awk -F= -v k="node-$1" '$1 == k { print substr($0, length(k) + 2); exit }' "$NODES"; }
on() { local n=$1; shift; ssh -o BatchMode=yes -o ConnectTimeout=30 "$(addr "$n")" "$@"; }
ENV=${CARD_RENDER_ENV:-/tmp/card-render}
WT=$(git rev-parse --show-toplevel)
SRC=$WT/src/training/decision2
OPS=$SRC/v2/release/records/dev2-4b-xall-2026-10-02/ops
LOCAL=${RENDER4B_LOCAL:-$HOME/code/decision2-program/private/4b-indexfirst/render}/$CAND
[[ "$("$ENV/venv/bin/python" -c 'import sys, matplotlib, PIL; print(sys.version.split()[0], matplotlib.__version__, PIL.__version__)')" \
  == "3.12.13 3.11.2 12.3.0" ]] || { echo "render environment differs" >&2; exit 2; }
[[ "$(sha256sum < "$ENV/logo.png" | cut -c1-16)" == e6b8428eb67f9318 ]] || { echo "logo differs" >&2; exit 2; }
[[ ! -e "$LOCAL" ]] || { echo "$LOCAL exists" >&2; exit 3; }
(umask 077 && mkdir -p "$LOCAL/root" "$LOCAL/out")
read -r PRIV IN < <(cd "$SRC" && PYTHONPATH=$SRC python3 -c '
import importlib.util, sys
spec = importlib.util.spec_from_file_location("make_xall", sys.argv[1]); m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m); p = m.paths(sys.argv[2]); print(p["private"], p["in"])' "$OPS/make_xall.py" "$CAND")
(cd "$SRC" && PYTHONPATH=$SRC python3 - "$OPS/make_xall.py" "$CAND" "$LOCAL" <<'PY')
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("make_xall", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
p = m.paths(sys.argv[2])
s = json.loads(m.BASE_SPEC.read_text())
reports = s["card"]["reports"]
assert reports[0]["role"] == "candidate" and reports[0]["label"] == m.NAME
reports[0]["report"] = f"{p['run']}/REPORT.json"
reports[0]["mlx"] = f"{p['mlx']}/mlx-diag.score.json"
files = [e[k] for e in reports for k in ("report", "mlx") if e.get(k)] + [f"{p['private']}/decision-index-card.json"]
open(f"{sys.argv[3]}/files.txt", "w").write("\n".join(files) + "\n")
root = sys.argv[3] + "/root"
for e in reports:
    for k in ("report", "mlx"):
        if e.get(k):
            e[k] = root + e[k]
json.dump(s, open(f"{sys.argv[3]}/render-spec.json", "w"), indent=2)
PY
while read -r f; do
  mkdir -p "$LOCAL/root$(dirname "$f")"
  on a "cat '$f'" < /dev/null > "$LOCAL/root$f"
  [[ "$(on a "sha256sum < '$f'" < /dev/null)" == "$(sha256sum < "$LOCAL/root$f")" ]] || { echo "$f differs after the copy" >&2; exit 3; }
done < "$LOCAL/files.txt"
identity=$(on a "python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[\"model_sha256\"])' $IN/bf16/bf16-copy.json")
[[ "$identity" =~ ^[0-9a-f]{64}$ ]] || { echo "no BF16 identity on node A" >&2; exit 3; }
index=$LOCAL/root$(sed -n '$p' "$LOCAL/files.txt")
(cd "$SRC" && PYTHONPATH=$SRC "$ENV/venv/bin/python" -B -m v2.release.card_assets --spec "$LOCAL/render-spec.json" \
  --index "$index" --logo "$ENV/logo.png" --fonts "$ENV/fonts" --model-sha256 "$identity" --output "$LOCAL/out") > "$LOCAL/render.log"
on a "test ! -e $PRIV/4b" || { echo "node A $PRIV/4b exists" >&2; exit 3; }
on a "umask 077; mkdir -p $PRIV/4b/assets"
for f in card-assets.json assets/banner.png assets/jevarena.png assets/jevarena-types.png assets/index-pareto.png assets/index-areas.png; do
  on a "umask 077; cat > $PRIV/4b/$f" < "$LOCAL/out/$f"
  [[ "$(on a "sha256sum < $PRIV/4b/$f")" == "$(sha256sum < "$LOCAL/out/$f")" ]] || { echo "$f differs on node A" >&2; exit 3; }
done
echo "card-assets.json $(sha256sum < "$LOCAL/out/card-assets.json" | cut -c1-64) -> node A $PRIV/4b (worktree $(git -C "$WT" rev-parse --short HEAD))"
